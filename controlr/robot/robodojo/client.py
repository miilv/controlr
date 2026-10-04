"""``RoboDojoRobot``: one RoboDojo episode, seen through controlr's ``Robot`` contract.

RoboDojo's eval client owns the episode; its shim (``shim/deploy.py``) connected to
``controlr robodojo-serve`` and announced the episode (``hello``). This object sends the
shim requests over that connection (see ``protocol.py``) and translates frames:

* RoboDojo reports and plans FLANGE poses (link6, world, quaternion wxyz); controlr's TCP is
  the grasp point ``ARX_X5_GRASP_OFFSET_M`` along the flange +x, and its tool frame has z
  along that axis (``R_tcp = R_flange @ M``). The model never sees a flange.
* Grippers: RoboDojo 0..1 (1 = open) <-> controlr width in metres (``gripper_max_mm``).
* Arms: controlr ``L``/``R`` <-> RoboDojo ``left``/``right``.

numpy + stdlib only (no Isaac): tested against the real shim with a fake TASK_ENV.
"""

from __future__ import annotations

import time

import numpy as np

from controlr.robot.base import Robot
from controlr.robot.kinematics import matrix_to_rotvec, rotation_angle, rotvec_to_matrix
from controlr.robot.robodojo import protocol as P
from controlr.robot.spec import ARX_X5_GRASP_OFFSET_M, arx_x5_dual_spec
from controlr.types import (
    Action, CameraInfo, EventLevel, ExecReport, GoalReport, Observation, RobotState, SafetyEvent,
)

ARM_NAMES = {"L": "left", "R": "right"}
# controlr tool axes in flange axes: tool x = flange y, tool y = flange z, tool z = flange x
M_TCP = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
ARRIVE_POS_M = 0.010          # arrival error above this is reported (a real robot knows it too)
ARRIVE_ANG_RAD = np.radians(5.0)


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q, float).reshape(4) / np.linalg.norm(q))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R) -> np.ndarray:
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q) * (1.0 if q[0] >= 0 else -1.0)


def flange_to_tcp(flange) -> tuple[np.ndarray, np.ndarray]:
    """RoboDojo flange pose [x y z qw qx qy qz] -> controlr TCP (pos, rotvec)."""
    f = np.asarray(flange, float).reshape(7)
    Rf = quat_to_matrix(f[3:7])
    pos = f[:3] + ARX_X5_GRASP_OFFSET_M * Rf[:, 0]
    return pos, matrix_to_rotvec(Rf @ M_TCP)


def tcp_to_flange(pos, rotvec) -> np.ndarray:
    R = rotvec_to_matrix(np.asarray(rotvec, float))
    Rf = R @ M_TCP.T
    flange = np.asarray(pos, float).reshape(3) - ARX_X5_GRASP_OFFSET_M * Rf[:, 0]
    return np.concatenate([flange, matrix_to_quat(Rf)])


class RoboDojoRobot(Robot):
    """One episode over an accepted shim connection (``conn``) announced by ``episode``."""

    def __init__(self, conn, episode: dict, params: dict | None = None) -> None:
        params = dict(params or {})
        self.conn = conn
        self.episode = dict(episode)
        self.spec = arx_x5_dual_spec(gripper_max_mm=float(params.get("gripper_max_mm", 80.0)))
        self.gmax = self.spec.gripper_max_mm / 1000.0
        self.task_instruction = str(self.episode.get("instruction") or "")
        self.control_hz = float(self.episode.get("control_hz") or 25.0)
        self.show_steps = bool(params.get("show_steps", True))
        # joint travel per env step (rad) and gripper range per step: what one env step may move.
        # 0.05 / 0.25 = RoboDojo's own LLM adapter (tracked cleanly); larger = fewer steps per move
        self.arm_step_rad = float(params.get("arm_step_rad", 0.05))
        self.grip_step = float(params.get("grip_step", 0.25))
        self._state: RobotState | None = None
        self._ended = False
        self._success = False
        self._steps = (0, int(self.episode.get("step_lim") or 0))

    # ---------------------------------------------------------------- rpc
    def _call(self, op: str, **args):
        self.conn.send(P.request(op, **args))
        return P.unwrap(self.conn.recv())

    # ------------------------------------------------------------ helpers
    def _arm_state(self, t: float, a: dict) -> RobotState:
        pos, rv = flange_to_tcp(a["flange"])
        g = float(a.get("grip_cmd", a.get("grip", 1.0)))
        return RobotState(t=t, q=np.asarray(a["q"], float), tcp_pos=pos, tcp_rotvec=rv,
                          gripper_mm=float(np.clip(a.get("grip", g), 0.0, 1.0)) * self.gmax * 1000.0,
                          gripper_closed=g < 0.5)

    def _robot_state(self, arms: dict) -> RobotState:
        t = self._steps[0] / self.control_hz
        per = {name: self._arm_state(t, arms[rd]) for name, rd in ARM_NAMES.items() if rd in arms}
        first = per[self.spec.arms[0]]
        return RobotState(t=first.t, q=first.q, tcp_pos=first.tcp_pos, tcp_rotvec=first.tcp_rotvec,
                          gripper_mm=first.gripper_mm, gripper_closed=first.gripper_closed, arms=per)

    def _note(self) -> tuple[str, ...]:
        used, lim = self._steps
        if not self.show_steps or not lim:
            return ()
        left = max(0, lim - used)
        return (f"STEPS: {left} of {lim} left ({left / self.control_hz:.1f} s of arm motion)",)

    def _track(self, res: dict) -> None:
        self._ended = bool(res.get("ended", self._ended))
        self._success = bool(res.get("success", self._success))
        self._steps = (int(res.get("steps_used", self._steps[0])), int(res.get("step_lim", self._steps[1])))

    # ----------------------------------------------------------- contract
    def reset(self, task_cfg: dict, seed: int | None = None) -> Observation:
        # RoboDojo already reset the scene to the layout it chose; nothing to sample here
        return self.observe()

    def observe(self) -> Observation:
        res = self._call("observe")
        self._track(res)
        state = self._robot_state(res["arms"])
        self._state = state
        cams = {}
        for name, img in res["images"].items():
            h, w = np.asarray(img).shape[:2]
            K = np.asarray(res.get("K", {}).get(name, np.eye(3)), float)
            Twc = res.get("T_world_cam", {}).get(name)
            T_cam_base = np.linalg.inv(np.asarray(Twc, float)) if Twc is not None else np.eye(4)
            cams[name] = CameraInfo(name=name, width=int(w), height=int(h), K=K, T_cam_base=T_cam_base)
        return Observation(t=state.t, images={k: np.asarray(v, np.uint8) for k, v in res["images"].items()},
                           cameras=cams, state=state, notes=self._note())

    def state(self) -> RobotState:
        if self._state is None:
            self.observe()
        return self._state  # type: ignore[return-value]

    def _steps_payload(self, actions: list[Action]) -> list[dict]:
        steps: list[dict] = []
        last_step: object = object()
        for a in actions:
            rd = ARM_NAMES.get(a.arm or self.spec.arms[0])
            if rd is None:
                continue
            if a.step is None or a.step != last_step or rd in steps[-1]:
                steps.append({})
            last_step = a.step
            cmd: dict = {"flange": None, "grip": None}
            if a.tcp_target is not None:
                cmd["flange"] = tcp_to_flange(a.tcp_target[:3], a.tcp_target[3:6]).tolist()
            if a.gripper is not None:
                cmd["grip"] = float(np.clip(a.gripper / self.gmax, 0.0, 1.0))
            steps[-1][rd] = cmd
        return steps

    def execute(self, actions: list[Action]) -> ExecReport:
        before = self.state()
        t0 = time.perf_counter()
        res = self._call("execute", steps=self._steps_payload(actions), arm_step_rad=self.arm_step_rad,
                         grip_step=self.grip_step)
        self._track(res)
        after = self.observe().state
        events: list[SafetyEvent] = []
        for step in res.get("steps", []):
            for rd, info in step.get("arms", {}).items():
                arm = next((k for k, v in ARM_NAMES.items() if v == rd), rd)
                if info.get("status") != "Success":
                    events.append(SafetyEvent(EventLevel.WARN, "ik_fail",
                                              f"arm {arm}: target not reachable (planner: {info.get('status')}) "
                                              f"-> arm {arm} did not move",
                                              brief=f"arm {arm} move skipped: target not reachable"))
        # arrival check against the last commanded target of each arm (real-sensor fact)
        targets = {}
        for a in actions:
            if a.tcp_target is not None:
                targets[a.arm or self.spec.arms[0]] = a.tcp_target
        for arm, tgt in sorted(targets.items()):
            s = (after.arms or {}).get(arm, after)
            d = float(np.linalg.norm(np.asarray(s.tcp_pos) - np.asarray(tgt[:3])))
            ang = rotation_angle(rotvec_to_matrix(tgt[3:6]) @ rotvec_to_matrix(s.tcp_rotvec).T)
            if (d > ARRIVE_POS_M or ang > ARRIVE_ANG_RAD) and not any(
                    e.kind == "ik_fail" and f"arm {arm}" in e.message for e in events):
                events.append(SafetyEvent(EventLevel.WARN, "arrival",
                                          f"arm {arm} stopped {d * 1000:.0f} mm / {np.degrees(ang):.0f} deg "
                                          f"from its target",
                                          brief=f"arm {arm} stopped {d * 1000:.0f} mm from its target"))
        if res.get("ended"):
            events.append(SafetyEvent(EventLevel.INFO, "env_end",
                                      "the episode ended: " + ("task complete" if res.get("success")
                                                               else "no env steps left")))
        return ExecReport(requested=list(actions), executed=list(actions), events=events,
                          state_before=before, state_after=after,
                          duration_s=float(res.get("sim_s", 0.0)),
                          backend={"env_steps": res.get("env_steps"), "steps_used": res.get("steps_used"),
                                   "step_lim": res.get("step_lim"), "wall_s": res.get("wall_s"),
                                   "sim_s": res.get("sim_s"), "client_s": time.perf_counter() - t0,
                                   "plan": res.get("steps")})

    def check_goal(self) -> GoalReport:
        res = self._call("check_goal")
        self._track(res)
        ok = bool(res.get("success"))
        return GoalReport(success=ok, progress=None,
                          message="task complete" if ok else "task not complete",
                          metrics={"ended": bool(res.get("ended")), "steps_used": res.get("steps_used"),
                                   "step_lim": res.get("step_lim")})

    def episode_over(self) -> str | None:
        if not self._ended:
            return None
        return "task_complete" if self._success else "step_limit"

    def scene_record(self) -> dict | None:
        return {k: self.episode.get(k) for k in ("task", "layout_id", "seed", "step_lim", "instruction",
                                                 "cameras")}
