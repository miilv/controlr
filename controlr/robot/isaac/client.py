"""``IsaacRobot``: controlr's Robot backed by the Isaac Sim server.

Runs in controlr's normal python — no Isaac / PHANTOM imports. It
(optionally) launches ``scripts/isaac_server.sh``, waits for the socket, and
then speaks the numpy-only protocol in ``protocol.py``.

Mapping of controlr Actions to the server: every action becomes one motion
group — its ``q_path`` waypoints (IK along the straight TCP line) and its
``q_target``. Actions that went through ``SafetyEnvelope.filter`` carry them
(the envelope already did IK with the same UR3 kinematics), so the backend
executes exactly what the envelope approved. Unfiltered actions are resolved by
the configured envelope, so the sim never runs an unchecked target. The group
duration keeps the PEAK speed of the min-jerk profile within the configured TCP
and joint speed limits (``motion.group_duration``); the server passes through
the waypoints in one smooth motion and settles.

Frames: the Isaac stage's world frame IS the UR controller base frame
(PHANTOM builds it that way), so ``T_cam_base`` is the inverse of the
calibrated ``world_from_cv`` of PHANTOM's D435 model.
"""

from __future__ import annotations

import dataclasses
import os
import secrets
import subprocess
import tempfile
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from controlr.robot.base import Robot
from controlr.robot.isaac import motion, protocol
from controlr.robot.isaac.tasks import START_Q, get_task
from controlr.robot.kinematics import UR3Kinematics, rotation_angle
from controlr.robot.safety import SafetyEnvelope
from controlr.robot.spec import ur3_cb3_spec
from controlr.types import (Action, CameraInfo, EventLevel, ExecReport, GoalReport, Observation, RobotSpec,
                            RobotState, SafetyEvent)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Verified by rendering the D435 view while stepping the TCP along each base
# axis (docs/img/isaac_axes_*.png, see controlr/robot/isaac/README.md).
ISAAC_BASE_DOC = (
    "Robot base frame (UR controller 'base'; origin at the robot's mounting flange, which is at "
    "the RIGHT edge of the camera image, level with the near wall of the blue box). "
    "In the camera image: +x points to the RIGHT (the mat and the blue box are at negative x, "
    "left of the robot); +y points AWAY from the camera, i.e. up the image (the blue box is at "
    "larger y than the mat); +z points up from the table, toward the camera, which looks down "
    "at about 25 degrees from vertical, so raising the gripper makes it look larger and shifts "
    "it slightly up the image. The mat surface is at z = {table_mm:.0f} mm."
)

DEFAULTS: dict = {
    "host": protocol.DEFAULT_HOST,
    "port": protocol.DEFAULT_PORT,
    "launch": True,                     # start scripts/isaac_server.sh if nobody listens
    "server_script": str(REPO_ROOT / "scripts" / "isaac_server.sh"),
    "server_args": [],                  # extra args for server.py (e.g. ["--solver-iterations", "32"])
    "startup_timeout_s": 300.0,
    "camera": "scene",
    "tcp_speed_m_s": None,              # None -> SafetyConfig.max_tcp_speed_m_s (0.15)
    "joint_speed_rad_s": 1.0,           # PHANTOM hardware.yaml arm.limits.joint_speed_rad_s
    "rot_speed_rad_s": 1.0,
    "min_segment_s": 0.1,
    "force_stop_n": None,               # None -> SafetyConfig.contact_force_stop_n
    "contact_report_n": 2.0,            # peak contact force above which the model is told
    "tracking_warn_m": 0.005,
    "settle": {},                       # server settle overrides (joint_speed_rad_s, max_s, ...)
    "log_path": None,                   # server stdout/stderr when launched (default: temp file)
}


class IsaacRobot(Robot):
    def __init__(self, params: dict | None = None, *, safety=None) -> None:
        from controlr.config import SafetyConfig
        p = {**DEFAULTS, **(params or {})}
        self.p = p
        self.safety_cfg = safety or SafetyConfig()
        self.proc: subprocess.Popen | None = None
        self.conn = None
        self._connect()
        info = self.call("info")
        self.server_info = info
        cam = info["camera"]
        self.camera = CameraInfo(name=p["camera"], width=int(cam["width"]), height=int(cam["height"]),
                                 K=np.asarray(cam["K"], float), T_cam_base=np.asarray(cam["T_cam_base"], float))
        grip = info["gripper"]
        mat_top = float(info["scene_info"]["mat"]["top_z"])
        tcp_offset = (0.0, 0.0, float(grip["tcp_offset_m"]), 0.0, 0.0, 0.0)
        spec = ur3_cb3_spec(home_q=tuple(p.get("home_q", START_Q)), table_z=mat_top, tcp_offset=tcp_offset,
                            gripper_max_mm=round(float(grip["max_m"]) * 1000.0, 1),
                            name="UR3 CB3 + Robotiq 2F-85 with DM-Tac W2L pads (Isaac Sim reconstruction)")
        self.spec: RobotSpec = dataclasses.replace(spec, base_frame_doc=ISAAC_BASE_DOC.format(table_mm=mat_top * 1000))
        self.kin = UR3Kinematics(tcp_offset=tcp_offset)
        self._fallback = SafetyEnvelope(self.spec, self.safety_cfg, self.kin)
        self.task_instruction = ""
        self.episode: dict = {}
        self.last_reset_s: float | None = None
        self._reference: RobotState | None = None

    @classmethod
    def from_config(cls, cfg) -> "IsaacRobot":
        return cls(cfg.robot.params, safety=cfg.safety)

    # ------------------------------------------------------------ connection
    def _try_connect(self):
        try:
            return Client((self.p["host"], int(self.p["port"])), authkey=protocol.authkey_from_env())
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return None

    def _connect(self) -> None:
        self.conn = self._try_connect()
        if self.conn is not None:
            return
        if not self.p["launch"]:
            raise ConnectionError(f"no Isaac server at {self.p['host']}:{self.p['port']} and launch=false")
        env = dict(os.environ)
        if not env.get(protocol.AUTHKEY_ENV):
            # a fresh secret for a server we own; our own process uses it too
            env[protocol.AUTHKEY_ENV] = os.environ[protocol.AUTHKEY_ENV] = secrets.token_hex(16)
        if self.p["log_path"]:
            log = open(self.p["log_path"], "w")
        else:
            log = tempfile.NamedTemporaryFile("w", prefix="controlr_isaac_", suffix=".log", delete=False)
        log_path = self.log_path = log.name
        cmd = ["bash", self.p["server_script"], "--port", str(int(self.p["port"])), *map(str, self.p["server_args"])]
        with log:
            self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + float(self.p["startup_timeout_s"])
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Isaac server exited with {self.proc.returncode}; see {log_path}")
            self.conn = self._try_connect()
            if self.conn is not None:
                return
            time.sleep(1.0)
        self.proc.terminate()
        raise TimeoutError(f"Isaac server not ready after {self.p['startup_timeout_s']} s; see {log_path}")

    def call(self, op: str, **args):
        self.conn.send(protocol.request(op, **args))
        return protocol.unwrap(self.conn.recv())

    # --------------------------------------------------------------- helpers
    def _state(self, s: dict) -> RobotState:
        return RobotState(t=float(s["t"]), q=np.asarray(s["q"], float), tcp_pos=np.asarray(s["tcp_pos"], float),
                          tcp_rotvec=np.asarray(s["tcp_rotvec"], float), gripper_mm=float(s["gripper_m"]) * 1000.0,
                          gripper_closed=bool(s["gripper_closed"]), holding=bool(s["holding"]))

    def _obs(self, o: dict) -> Observation:
        st = self._state(o["state"])
        return Observation(t=st.t, images={self.camera.name: np.asarray(o["rgb"], np.uint8)},
                           cameras={self.camera.name: self.camera}, state=st)

    # ------------------------------------------------------------------- api
    def reset(self, task_cfg: dict, seed: int | None = None) -> Observation:
        if dataclasses.is_dataclass(task_cfg):
            task_cfg = dataclasses.asdict(task_cfg)
        task_cfg = dict(task_cfg or {})
        spec = get_task(task_cfg.get("name") or "waffle_pick_place")
        res = self.call("reset", task=spec.name, seed=seed, params=dict(task_cfg.get("params") or {}))
        self.episode = res["episode"]
        self.last_reset_s = float(res["reset_s"])
        self.task_instruction = task_cfg.get("instruction") or spec.instruction
        obs = self._obs(res["obs"])
        # the manual's "home" column = the pose this episode actually starts from
        q0 = np.asarray(self.episode.get("start_q", obs.state.q), float)
        self.spec = dataclasses.replace(self.spec, home_q=tuple(float(v) for v in q0))
        pos, rv = self.kin.fk(q0)
        self._reference = dataclasses.replace(obs.state, q=q0, tcp_pos=pos, tcp_rotvec=rv)
        self._fallback.reset(self._reference)
        return obs

    def reference_state(self) -> RobotState | None:
        """FK of the COMMANDED start joints: the measured reset pose carries a few
        mrad of settle sag/jitter, which would make the cached manual's tool lines
        and the rotation=none reference differ between episodes."""
        return self._reference

    def observe(self) -> Observation:
        return self._obs(self.call("observe"))

    def state(self) -> RobotState:
        return self._state(self.call("state"))

    def _resolve(self, actions: list[Action], state: RobotState) -> tuple[list[Action], list[SafetyEvent]]:
        """Use the envelope's q_target when present; otherwise run a default
        envelope so nothing unchecked reaches the sim."""
        if all(a.q_target is not None for a in actions):
            return list(actions), []
        return self._fallback.filter(actions, state)

    def _duration(self, q0: np.ndarray, waypoints: np.ndarray) -> float:
        """Group duration with the PEAK speeds of the min-jerk profile within the limits
        (15/8 x average; review control-safety #7)."""
        v = float(self.p["tcp_speed_m_s"] or self.safety_cfg.max_tcp_speed_m_s)
        pts = [np.asarray(q0, float)] + [np.asarray(w, float) for w in waypoints]
        Ts = [self.kin.fk_matrix(q) for q in pts]
        d = float(sum(np.linalg.norm(b[:3, 3] - a[:3, 3]) for a, b in zip(Ts, Ts[1:])))
        ang = float(sum(rotation_angle(b[:3, :3] @ a[:3, :3].T) for a, b in zip(Ts, Ts[1:])))
        return motion.group_duration(pts[0], np.asarray(pts[1:]), d, ang, tcp_speed=v,
                                     joint_speed=float(self.p["joint_speed_rad_s"]),
                                     rot_speed=float(self.p["rot_speed_rad_s"]),
                                     min_s=float(self.p["min_segment_s"]))

    def execute(self, actions: list[Action]) -> ExecReport:
        before = self.state()
        executed, events = self._resolve(actions, before)
        if not executed:
            return ExecReport(requested=list(actions), executed=[], events=events, state_before=before,
                              state_after=before, duration_s=0.0)
        qs, grips, durs, group = [], [], [], []
        q_prev = np.asarray(before.q, float)
        for gi, a in enumerate(executed):
            rows = [np.asarray(w, float) for w in (a.q_path or ())] + [np.asarray(a.q_target, float)]
            moving = not np.allclose(rows[-1], q_prev, atol=1e-6) or len(rows) > 1
            dur = self._duration(q_prev, np.asarray(rows)) if moving else 0.0
            for k, q in enumerate(rows):
                last = k == len(rows) - 1
                qs.append(q)
                grips.append(float(a.gripper) if (last and a.gripper is not None) else np.nan)
                durs.append(dur if last else 0.0)
                group.append(gi)
            q_prev = rows[-1]
        force_stop = self.p["force_stop_n"] if self.p["force_stop_n"] is not None else self.safety_cfg.contact_force_stop_n
        r = self.call("execute", q=np.asarray(qs), gripper=np.asarray(grips), durations=np.asarray(durs),
                      group=np.asarray(group), force_stop_n=float(force_stop),
                      object_force_stop_n=float(self.safety_cfg.object_force_stop_n),
                      settle=dict(self.p["settle"]))
        events.extend(self._events(r))
        after = self._state(r["state"])
        n_done = int(r["segments_done"])
        # on a stop in action k (n_done = k) action k was partly executed: keep it
        return ExecReport(requested=list(actions), executed=executed[:n_done + 1] if r["stopped"] else executed,
                          events=events, state_before=before, state_after=after,
                          duration_s=float(r["sim_s"]), stopped=bool(r["stopped"]))

    def _events(self, r: dict) -> list[SafetyEvent]:
        ev: list[SafetyEvent] = []
        if r["stopped"]:
            ev.append(SafetyEvent(EventLevel.STOP, "unstable" if r.get("unstable") else "collision",
                                  f"motion stopped: {r['stop_reason']}; the arm holds its current pose"))
        thr = float(self.p["contact_report_n"])
        obj_warn = 0.5 * float(self.safety_cfg.object_force_stop_n or 0.0)
        names = {"arm-table": "arm touched the table/mat", "gripper-table": "gripper touched the table/mat",
                 "arm-box": "arm touched the blue box", "gripper-box": "gripper touched the blue box",
                 "arm-object": "arm touched the packet", "gripper-object": "gripper/fingers touched the packet"}
        for key, f in sorted(r["contacts_peak_n"].items()):
            if f < thr or key not in names:
                continue
            # fingers on the packet are normal (grasping) unless pressed hard
            level = EventLevel.INFO if key == "gripper-object" and not (obj_warn and f > obj_warn) else EventLevel.WARN
            ev.append(SafetyEvent(level, "contact" if level is EventLevel.INFO else "collision",
                                  f"{names[key]} (peak {f:.0f} N)"))
        if float(r["tcp_err_m"]) > float(self.p["tracking_warn_m"]):
            ev.append(SafetyEvent(EventLevel.WARN, "tracking",
                                  f"TCP ended {r['tcp_err_m'] * 1000:.0f} mm from the commanded position (blocked?)"))
        if not r["settled"]:
            ev.append(SafetyEvent(EventLevel.WARN, "settle", "the arm had not fully settled when the image was taken"))
        if not r.get("gripper_mechanics_ok", True):
            ev.append(SafetyEvent(EventLevel.WARN, "gripper", "gripper linkage reported an abnormal state"))
        return ev

    def check_goal(self) -> GoalReport:
        g = self.call("check_goal")
        return GoalReport(success=bool(g["success"]), progress=g.get("progress"), message=str(g["message"]),
                          metrics=dict(g.get("metrics", {})))

    def close(self) -> None:
        try:
            if self.conn is not None:
                self.call("shutdown" if self.proc is not None else "close")
                self.conn.close()
        except (OSError, EOFError, protocol.RemoteError):
            pass
        self.conn = None
        if self.proc is not None:
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
            self.proc = None
