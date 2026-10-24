"""Scripted arm motion on a ``RoboDojoRunner`` (dev tooling: D0 moves, G2 trajectories).

Same motion rules as the controlr shim (``controlr/robot/robodojo/shim/deploy.py``) and RoboDojo's
own LLM adapter: an arm target is a flange (link6) pose planned by RoboDojo's cuRobo planner, the
joint path is resampled so no joint moves more than ``step_rad`` per env step, every env step
commands both arms and both grippers, grippers move after the arms arrive.

Trajectories are recorded as the list of per-step action dicts (``Trajectory.actions``), so a
run can be replayed bit-for-bit as RoboDojo actions on another instance (G2a/G2b) without the
planner in the loop.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from dojo.simlab.layout import quat_mul

ARMS = ("left", "right")
TCP_OFFSET = np.array([0.1501, 0.0, 0.0])        # grasp point along the flange +x (controlr docs/ROBODOJO.md)
STEP_RAD = 0.05
GRIP_STEP = 0.25


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tcp_of(flange) -> np.ndarray:
    f = np.asarray(flange, float)
    return f[:3] + quat_to_mat(f[3:]) @ TCP_OFFSET


def flange_for_tcp(tcp_xyz, quat) -> np.ndarray:
    """Flange pose whose TCP sits at ``tcp_xyz`` with orientation ``quat`` (wxyz)."""
    q = np.asarray(quat, float)
    return np.concatenate([np.asarray(tcp_xyz, float) - quat_to_mat(q) @ TCP_OFFSET, q])


def resample(path: np.ndarray, steps: int) -> np.ndarray:
    path = np.asarray(path, float)
    if len(path) < 2:
        return np.repeat(path[-1:], steps, axis=0)
    src = np.linspace(0.0, 1.0, len(path))
    dst = np.linspace(0.0, 1.0, steps + 1)[1:]
    return np.stack([np.interp(dst, src, path[:, j]) for j in range(path.shape[1])], axis=1)


def n_steps(path: np.ndarray, step_rad: float = STEP_RAD) -> int:
    path = np.asarray(path, float)
    if len(path) < 2:
        return 1
    travel = np.abs(np.diff(path, axis=0)).sum(axis=0)
    return max(1, int(np.ceil(float(travel.max()) / step_rad - 1e-9)))


@dataclass
class Trajectory:
    """Every env-step action sent, in order (replayable as-is)."""
    actions: list[dict] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        # exact: float32 -> float64 is lossless and JSON floats round-trip, so a replay sends the
        # very same bits (rounding to 9 decimals moved some commands by 1 ulp and the contact
        # phase amplified it to 0.4 mm)
        return {"actions": [{k: np.asarray(v, np.float32).astype(np.float64).tolist() for k, v in a.items()}
                            for a in self.actions], "segments": self.segments}

    @staticmethod
    def from_json(d: dict) -> "Trajectory":
        return Trajectory(actions=[{k: np.asarray(v, np.float32) for k, v in a.items()} for a in d["actions"]],
                          segments=list(d.get("segments", [])))


class Mover:
    """Hold targets for both arms; plan, step, record."""

    def __init__(self, runner, env_idx: int = 0, step_rad: float = STEP_RAD, grip_step: float = GRIP_STEP,
                 on_step: Callable[[int], None] | None = None):
        self.r = runner
        self.env_idx = env_idx
        self.step_rad = step_rad
        self.grip_step = grip_step
        self.on_step = on_step
        st = runner.obs_state(env_idx)
        self.q = {a: np.asarray(st[f"{a}_arm_joint_state"], float) for a in ARMS}
        self.g = {a: float(st[f"{a}_ee_joint_state"][0]) for a in ARMS}
        self.home_q = copy.deepcopy(self.q)
        self.home_flange = {a: np.asarray(st[f"{a}_ee_pose"], float) for a in ARMS}
        self.traj = Trajectory()
        self.ended = False

    # --------------------------------------------------------------- primitives
    def action(self) -> dict:
        d = {}
        for a in ARMS:
            d[f"{a}_arm_joint_state"] = np.asarray(self.q[a], np.float32)
            d[f"{a}_ee_joint_state"] = np.asarray([self.g[a]], np.float32)
        return d

    def _step(self) -> bool:
        act = self.action()
        self.traj.actions.append(copy.deepcopy(act))
        ends = self.r.step(act)
        if self.on_step is not None:
            self.on_step(len(self.traj.actions))
        self.ended = bool(ends[self.env_idx])
        return self.ended

    def measured_q(self, arm: str) -> np.ndarray:
        return np.asarray(self.r.obs_state(self.env_idx)[f"{arm}_arm_joint_state"], float)

    def plan(self, arm: str, flange) -> np.ndarray | None:
        rm = self.r.env.robot_manager
        robot = rm.get_robot_by_arm_name(f"{arm}_arm")
        planner = rm.planner.get(robot.robot_name)
        if planner is None:
            return None
        res = planner.plan_path(self.measured_q(arm), np.asarray(flange, float),
                                real_robot_pose=copy.deepcopy(robot.entity_origin_pose))
        if str(res.get("status")) != "Success" or res.get("position") is None:
            return None
        p = res["position"]
        if hasattr(p, "detach"):
            p = p.detach().cpu().numpy()
        p = np.asarray(p, float).reshape(-1, 6)
        return p if len(p) else None

    # --------------------------------------------------------------- moves
    def joint_paths(self, paths: dict[str, np.ndarray], name: str = "") -> int:
        n = max((n_steps(p, self.step_rad) for p in paths.values()), default=0)
        way = {a: resample(p, n) for a, p in paths.items()}
        done = 0
        for i in range(n):
            for a, w in way.items():
                self.q[a] = w[i]
            done += 1
            if self._step():
                break
        self.traj.segments.append({"name": name, "kind": "arm", "arms": sorted(paths), "steps": done})
        return done

    def joints(self, targets: dict[str, np.ndarray], name: str = "") -> int:
        """Straight joint-space move (no planner)."""
        return self.joint_paths({a: np.stack([self.q[a], np.asarray(t, float)]) for a, t in targets.items()}, name)

    def flange(self, targets: dict[str, np.ndarray], name: str = "") -> dict[str, bool]:
        paths, ok = {}, {}
        for a, f in targets.items():
            p = self.plan(a, f)
            ok[a] = p is not None
            if p is not None:
                paths[a] = p
        if paths:
            self.joint_paths(paths, name)
        return ok

    def tcp(self, targets: dict[str, np.ndarray], quat: dict[str, np.ndarray] | None = None, name: str = "") -> dict:
        """TCP positions (env frame) with the home flange orientation unless ``quat`` is given."""
        fl = {}
        for a, xyz in targets.items():
            q = (quat or {}).get(a, self.home_flange[a][3:])
            fl[a] = flange_for_tcp(xyz, q)
        return self.flange(fl, name)

    def grip(self, values: dict[str, float], name: str = "") -> int:
        n = max((int(np.ceil(abs(float(v) - self.g[a]) / self.grip_step - 1e-9)) for a, v in values.items()), default=0)
        for a, v in values.items():
            self.g[a] = float(np.clip(v, 0.0, 1.0))
        done = 0
        for _ in range(n):
            done += 1
            if self._step():
                break
        self.traj.segments.append({"name": name, "kind": "grip", "values": {a: float(v) for a, v in values.items()},
                                   "steps": done})
        return done

    def wait(self, steps: int, name: str = "wait") -> int:
        done = 0
        for _ in range(steps):
            done += 1
            if self._step():
                break
        self.traj.segments.append({"name": name, "kind": "wait", "steps": done})
        return done

    def home(self, name: str = "home") -> int:
        return self.joints({a: self.home_q[a] for a in ARMS}, name)


def replay(runner, traj: Trajectory, on_step: Callable[[int], None] | None = None, env_idx: int = 0) -> int:
    """Send a recorded action list again (no planning). Returns steps sent before the episode ended."""
    n = 0
    for act in traj.actions:
        ends = runner.step(act)
        n += 1
        if on_step is not None:
            on_step(n)
        if ends[env_idx]:
            break
    return n


def down_quat(home_quat, pitch_deg: float = 0.0) -> np.ndarray:
    """Home flange orientation tilted about world x (helper for looking further down)."""
    h = np.radians(pitch_deg) / 2
    q = quat_mul([np.cos(h), np.sin(h), 0.0, 0.0], home_quat)
    return q / np.linalg.norm(q)
