"""``So101Robot``: controlr's Robot for the real SO-101 follower arm with a wrist camera.

Runs in controlr's python; the arm and the camera belong to ``server.py`` on the machine they
are plugged into, reached over ``multiprocessing.connection`` (normally through an ssh tunnel:
the rig itself cannot reach the LLM router). See docs/SO101.md.

What happens to an approved action (``CartesianEnvelope``: ``Action.tcp_target``):
* the target orientation is turned to the one 5 joints can hold at that point
  (``kinematics.reachable_rotation``: the tool keeps its pitch and points away from the base);
* IK every 5 mm along the straight TCP line (``kinematics.line_path``); an unreachable
  stretch shortens the move and is reported like the envelope's clamps;
* the server streams the joint path min-jerk timed (TCP speed ``tcp_speed_m_s``, joint speed
  capped on the server), stops the arm if a joint is blocked, settles, and the next
  observation is a frame exposed after the arm stood still.

Everything the model is told comes from real sensors: joint encoders (FK), servo tracking
error / current (a blocked joint), the gripper's position after a close (something between
the jaws). There is no automatic success check: ``check_goal`` never succeeds, so configs set
``episode.trust_done`` and the operator records a verdict (``scripts/so101_verdict.py``).
"""

from __future__ import annotations

import io
import json
import math
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from controlr.robot.base import Robot
from controlr.robot.kinematics import matrix_to_rotvec, rotvec_to_matrix
from controlr.robot.so101 import kinematics as K
from controlr.robot.so101 import protocol as P
from controlr.robot.spec import so101_spec
from controlr.types import (Action, CameraInfo, EventLevel, ExecReport, GoalReport, Observation, RobotState,
                            SafetyEvent)

DEFAULTS: dict = {
    "host": P.DEFAULT_HOST,
    "port": P.DEFAULT_PORT,
    "connect_timeout_s": 10.0,
    "camera": "wrist",                  # camera name the model sees
    # JSON from scripts/so101_calibrate.py: {"K", "dist", "width", "height", "T_tcp_cam"}.
    # None: K from hfov_deg and the stock wrist mount from the URDF (uncalibrated).
    "camera_calib": None,
    "hfov_deg": 80.0,
    "jpeg_quality": 92,
    "tool_pitch_deg": 60.0,             # 90 = straight down (the only pitch where yaw works)
    "tool_roll_deg": 90.0,              # about the tool axis: 90 = the stock wrist camera's image upright
                                        # (image up = forward, jaws closing sideways); 0 = image turned 90 deg
    "home_tcp": [0.24, 0.0, 0.10],      # m, base frame, at tool_pitch_deg / tool_roll_deg
    "gripper_max_mm": 70.0,             # jaw opening at 100 % (approximate: linear map)
    "gripper_open_pct": 40.0,           # GRIP open = this much: both jaws stay in the wrist image up to ~40 %
    "tcp_speed_m_s": None,              # peak TCP speed of a move (None: safety.max_tcp_speed_m_s;
                                        # the server caps the joint speed too)
    "reset_duration_s": 4.0,
    # collision = a joint drawing more than current_stop_ma for stop_ticks (free moves 2026-10-14:
    # <= 280 mA); tracking error only as a backstop for a hard block: a loaded joint in a stretched
    # pose lags ~10 deg with nothing in the way (false STOPs at 8 deg, so101_ball_down run 1)
    "track_stop_deg": 20.0,
    "current_stop_ma": [400.0, 600.0, 600.0, 400.0, 300.0],   # per arm joint; None = off
    "stop_ticks": 3,
    "arrive_warn_m": 0.010,
    "view_note": True,                  # per-turn VIEW line: how the base axes look in the wrist image
    "view_long_edge": 448,              # px of the image the model gets (for the VIEW lengths)
    "operator_reset": False,            # ask on stdin to reset the scene before each episode
    "workspace_lo": None,               # override so101_spec's box
    "workspace_hi": None,
}


def _intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]])


class So101Robot(Robot):
    def __init__(self, params: dict | None = None, *, rotation: str = "none", tcp_speed_m_s: float = 0.05,
                 conn=None) -> None:
        self.p = {**DEFAULTS, **(params or {})}
        if self.p["tcp_speed_m_s"] is None:
            self.p["tcp_speed_m_s"] = tcp_speed_m_s
        self.pitch = math.radians(float(self.p["tool_pitch_deg"]))
        if rotation == "yaw" and self.p["tool_pitch_deg"] < 89.5:
            raise ValueError("so101: action.rotation=yaw needs robot.params.tool_pitch_deg=90 (a tilted tool on "
                             "5 joints cannot turn about the vertical); use rotation=none")
        if rotation == "full":
            raise ValueError("so101: action.rotation=full is not reachable with 5 joints")
        self.rotation = rotation
        self.conn = conn if conn is not None else self._connect()
        self.info = self._call("info")
        self.limits = np.asarray(self.info["limits_rad"], float)
        self.gmax_m = float(self.p["gripper_max_mm"]) / 1000.0          # width at 100 %
        self.open_pct = min(float(self.p["gripper_open_pct"]),
                            float((self.info.get("gripper_range") or [0, 100])[1]))
        ws = {}
        if self.p.get("workspace_lo"):
            ws["workspace_lo"] = tuple(self.p["workspace_lo"])
        if self.p.get("workspace_hi"):
            ws["workspace_hi"] = tuple(self.p["workspace_hi"])
        self.home_q = self._solve_home()
        self.spec = so101_spec(limits_rad=self.limits, tool_pitch_deg=float(self.p["tool_pitch_deg"]),
                               gripper_max_mm=float(self.p["gripper_max_mm"]) * self.open_pct / 100.0,
                               joint_speed=float(self.info.get("max_joint_speed", 1.0)),
                               home_q=tuple(float(v) for v in self.home_q), **ws)
        self._load_camera()
        self._state: RobotState | None = None
        self._closed_cmd = False
        self._holding: bool | None = None
        self._ref: RobotState | None = None

    @classmethod
    def from_config(cls, cfg) -> "So101Robot":
        params = {"view_long_edge": cfg.observation.size, **dict(cfg.robot.params or {})}
        return cls(params, rotation=cfg.action.rotation, tcp_speed_m_s=cfg.safety.max_tcp_speed_m_s)

    # ---------------------------------------------------------------- rpc
    def _connect(self):
        deadline = time.monotonic() + float(self.p["connect_timeout_s"])
        while True:
            try:
                return Client((self.p["host"], int(self.p["port"])), authkey=P.authkey_from_env())
            except (ConnectionRefusedError, OSError) as e:
                if time.monotonic() > deadline:
                    raise ConnectionError(
                        f"no SO-101 server at {self.p['host']}:{self.p['port']} ({e}); start it on the rig "
                        "(scripts/so101_server.sh) and open the tunnel (docs/SO101.md)") from e
                time.sleep(0.5)

    def _call(self, op: str, **args):
        self.conn.send(P.request(op, **args))
        return P.unwrap(self.conn.recv())

    # ------------------------------------------------------------ helpers
    def _load_camera(self) -> None:
        cam = self.info.get("camera") or {}
        w, h = int(cam.get("width", 1280)), int(cam.get("height", 720))
        self.calibrated = False
        self.T_tcp_cam = K.DEFAULT_T_TCP_CAM.copy()
        self.K = _intrinsics(w, h, float(self.p["hfov_deg"]))
        if self.p.get("camera_calib"):
            c = json.loads(Path(self.p["camera_calib"]).expanduser().read_text())
            sx, sy = w / float(c.get("width", w)), h / float(c.get("height", h))
            self.K = np.asarray(c["K"], float) * np.array([[sx], [sy], [1.0]])
            if c.get("T_tcp_cam") is not None:
                self.T_tcp_cam = np.asarray(c["T_tcp_cam"], float)
                self.calibrated = True
        self.cam_size = (w, h)

    def home_rotation(self) -> np.ndarray:
        x, y = float(self.p["home_tcp"][0]), float(self.p["home_tcp"][1])
        az = math.atan2(y - K.PAN_AXIS_XY[1], x - K.PAN_AXIS_XY[0])
        return K.tool_rotation(self.pitch, 0.0 if self.pitch >= math.radians(89.5) else az,
                               math.radians(float(self.p["tool_roll_deg"])))

    def _solve_home(self) -> np.ndarray:
        R = self.home_rotation()
        best = None
        roll = -float(self.p["tool_roll_deg"])          # the wrist_roll joint turns the other way
        for seed in ([0, -30, 60, 60, roll], [0, 30, 0, 30, roll], [0, -60, 90, 30, roll], [0, 0, 45, 45, roll]):
            q, pe, ae = K.ik(self.p["home_tcp"], R, np.radians(seed), self.limits, iters=300)
            if best is None or pe + 0.05 * ae < best[1] + 0.05 * best[2]:
                best = (q, pe, ae)
        q, pe, ae = best  # type: ignore[misc]
        if pe > 0.002 or ae > 0.035:
            raise ValueError(f"so101: home_tcp {self.p['home_tcp']} at pitch {self.p['tool_pitch_deg']} deg is "
                             f"not reachable ({pe * 1000:.0f} mm / {math.degrees(ae):.0f} deg off)")
        return q

    def _robot_state(self, s: dict) -> RobotState:
        q = np.asarray(s["q"], float)
        T = K.fk(q)
        return RobotState(t=float(s["t"]), q=q, tcp_pos=T[:3, 3].copy(), tcp_rotvec=matrix_to_rotvec(T[:3, :3]),
                          gripper_mm=float(s["gripper"]) / 100.0 * self.gmax_m * 1000.0,
                          gripper_closed=self._closed_cmd, holding=self._holding)

    def _camera_info(self, q) -> CameraInfo:
        return CameraInfo(name=self.p["camera"], width=self.cam_size[0], height=self.cam_size[1], K=self.K.copy(),
                          T_cam_base=K.camera_T_cam_base(q, self.T_tcp_cam))

    def _view_note(self, info: CameraInfo, state: RobotState) -> tuple[str, ...]:
        """Which way each base axis runs in THIS image, measured from the jaw tips: an object
        that appears in that direction from the tips lies toward that axis. With the camera
        looking down within 35 deg of vertical, z runs along the view (said so)."""
        if not self.p["view_note"]:
            return ()
        from controlr.observation.renderers import describe_axes   # numpy/PIL only
        view = np.linalg.inv(info.T_cam_base)[:3, 2]
        steep = float(-view[2]) > math.cos(math.radians(35.0))
        text, _, size = describe_axes(info, state.tcp_pos, int(self.p["view_long_edge"])).rpartition(" in the ")
        parts = text.split(", ")
        if steep:
            parts = [x for x in parts if not x.startswith("+z")]
            parts.append("+z points toward the camera (height: read z in STATE)")
        parts[-1] += f" in the {size}"
        # The px lengths above are at the jaw tips' height. Objects on the table are farther from
        # the camera, so the same pixel offset there is MORE millimetres: reading it with the tips'
        # scale made every estimate short (the ball runs, 2026-10-14). Give the table's scale too.
        table = self._table_scale(info, state)
        return (f"VIEW: from the jaw tips in the {info.name} image, " + ", ".join(parts)
                + (f"; on the table (z = 0) 100 mm is only {table} - use that scale for things lying on "
                   "the table" if table else ""),)

    def _table_scale(self, info: CameraInfo, state: RobotState) -> str:
        from controlr.observation.renderers import _project_segment

        o = np.array([state.tcp_pos[0], state.tcp_pos[1], self.spec.table_z or 0.0])
        scale = int(self.p["view_long_edge"]) / max(info.width, info.height)
        out = []
        for name, d in (("x", (1, 0, 0)), ("y", (0, 1, 0))):
            seg = _project_segment(info, o, o + 0.1 * np.asarray(d, float), info.width, info.height)
            if seg is None:
                return ""
            out.append(f"{math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]) * scale:.0f} px along {name}")
        return " and ".join(out)

    def _grip_pct(self, width_m: float) -> float:
        return float(np.clip(width_m / self.gmax_m * 100.0, 0.0, self.open_pct))

    # ----------------------------------------------------------- contract
    def reset(self, task_cfg: dict, seed: int | None = None) -> Observation:
        if self.p["operator_reset"]:
            input("[so101] set up the scene for the next episode, then press Enter ")
        res = self._call("move_joints", q=self.home_q, gripper=self.open_pct,
                         duration_s=float(self.p["reset_duration_s"]))
        self._closed_cmd, self._holding = False, None
        if res.get("stopped"):
            raise RuntimeError(f"so101: the arm was blocked while moving home ({res.get('stop_joint')}); "
                               "clear the way and retry")
        T = K.fk(self.home_q)
        self._ref = RobotState(t=0.0, q=self.home_q.copy(), tcp_pos=T[:3, 3].copy(),
                               tcp_rotvec=matrix_to_rotvec(T[:3, :3]),
                               gripper_mm=self.gmax_m * self.open_pct / 100.0 * 1000.0,
                               gripper_closed=False)
        return self.observe()

    def reference_state(self) -> RobotState | None:
        return self._ref

    def observe(self) -> Observation:
        from PIL import Image

        res = self._call("observe", jpeg_quality=int(self.p["jpeg_quality"]))
        state = self._robot_state(res["state"])
        self._state = state
        rgb = np.asarray(Image.open(io.BytesIO(res["jpeg"])).convert("RGB"))
        info = self._camera_info(state.q)
        return Observation(t=state.t, images={info.name: rgb}, cameras={info.name: info}, state=state,
                           notes=self._view_note(info, state))

    def state(self) -> RobotState:
        self._state = self._robot_state(self._call("state"))
        return self._state

    def _stop_args(self) -> dict:
        cur = self.p.get("current_stop_ma")
        return {"track_stop_rad": math.radians(float(self.p["track_stop_deg"])),
                "current_stop_ma": cur if cur else None, "stop_ticks": int(self.p["stop_ticks"])}

    def execute(self, actions: list[Action]) -> ExecReport:
        before = self.state()
        q = np.asarray(before.q, float)
        events: list[SafetyEvent] = []
        executed: list[Action] = []
        runs: list[dict] = []
        stopped = False
        wall = 0.0
        for a in actions:
            g = None if a.gripper is None else self._grip_pct(float(a.gripper))
            rows: list[np.ndarray] = [q]
            dist = 0.0
            target = None
            if a.tcp_target is not None:
                p = np.asarray(a.tcp_target[:3], float)
                R = K.reachable_rotation(p, rotvec_to_matrix(np.asarray(a.tcp_target[3:6], float)))
                path, frac, why = K.line_path(q, p, R, self.limits)
                if not path or frac < 0.1:
                    events.append(SafetyEvent(EventLevel.WARN, "ik_fail",
                                              f"{a.raw or 'move'}: target {why or 'not reachable'} -> move skipped",
                                              brief=f"move skipped: target {why or 'not reachable'}"))
                    if g is None:
                        continue
                    path = []
                elif frac < 1.0:
                    events.append(SafetyEvent(EventLevel.WARN, "clamp",
                                              f"{a.raw or 'move'}: moved {frac * 100:.0f} % of the way ({why})",
                                              brief=f"move shortened: {why}"))
                if path:
                    rows = path
                    dist = float(np.linalg.norm(K.fk(path[-1])[:3, 3] - K.fk(q)[:3, 3]))
                    target = K.fk(path[-1])[:3, 3]
            elif g is None:
                continue                                   # HOLD
            # the jaws keep their width during the motion and change after it (as in the sim)
            calls = []
            if target is not None:
                calls.append((rows, [math.nan] * len(rows), dist / max(1e-3, float(self.p["tcp_speed_m_s"]))))
            if g is not None:
                calls.append(([rows[-1]], [g], 0.0))
            res: dict = {}
            for rows_k, grip_k, dur in calls:
                res = self._call("execute", q=np.array(rows_k), gripper=np.array(grip_k), duration_s=dur,
                                 stop=self._stop_args())
                wall += float(res.get("wall_s", 0.0))
                runs.append({k: res.get(k) for k in ("stopped", "stop_reason", "stop_joint", "ticks", "wall_s",
                                                     "duration_s", "peak_current_ma", "peak_track_err_rad",
                                                     "gripper_blocked", "final_err_rad", "offset_rad")})
                if res.get("stopped"):
                    break
            executed.append(a)
            if g is not None and not res.get("stopped"):
                self._closed_cmd = g < self.open_pct / 2       # "open" is open_pct (40 %), not 100
                self._holding = bool(res.get("gripper_blocked")) if self._closed_cmd else False
            q = np.asarray(res["state"]["q"], float)
            if res.get("stopped"):
                stopped = True
                j = str(res.get("stop_joint") or "a joint").replace("_", " ")
                why = str(res.get("stop_reason") or "")
                events.append(SafetyEvent(EventLevel.STOP, "contact",
                                          f"the arm hit something ({j}: {why}) and stopped where it is",
                                          brief="the arm pushed against something and stopped"))
                break
            if target is not None:
                d = float(np.linalg.norm(K.fk(q)[:3, 3] - target))
                if d > float(self.p["arrive_warn_m"]):
                    events.append(SafetyEvent(EventLevel.WARN, "arrival",
                                              f"the gripper stopped {d * 1000:.0f} mm from its target",
                                              brief=f"stopped {d * 1000:.0f} mm short of the target"))
        after = self.state()
        return ExecReport(requested=list(actions), executed=executed, events=events, state_before=before,
                          state_after=after, duration_s=wall, stopped=stopped,
                          backend={"runs": runs, "camera_calibrated": self.calibrated})

    def check_goal(self) -> GoalReport:
        return GoalReport(success=False, progress=None, message="no automatic success check on the real rig")

    def hold(self) -> None:
        self._call("hold")

    def scene_record(self) -> dict | None:
        return {"home_q_deg": np.degrees(self.home_q).round(2).tolist(),
                "home_tcp_mm": (np.asarray(self.p["home_tcp"], float) * 1000).round(1).tolist(),
                "tool_pitch_deg": float(self.p["tool_pitch_deg"]), "camera_calibrated": self.calibrated,
                "server_fake": bool(self.info.get("fake"))}

    def close(self) -> None:
        try:
            self._call("close")
        except Exception:  # noqa: BLE001
            pass
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass
