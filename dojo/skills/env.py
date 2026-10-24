"""Env adapters: what the executor drives.

* ``RunnerEnv`` wraps simlab's ``RoboDojoRunner`` (RoboDojo's own EvalEnv on our layouts; runs in
  RoboDojo's python). Actions are official joint action dicts; proprio is the official obs
  (joints, flange poses, commanded grippers). Ground truth (objects, reward, measured fingers) is
  DEV-ONLY and goes through ``gt_objects`` / ``reward`` only.
* ``KinematicEnv`` is a CPU stand-in for tests: joints track targets at <= 0.2 rad per step,
  a closed gripper picks up the object around its TCP, an opened one drops it straight down.
"""

from __future__ import annotations

import copy
import math
from typing import Protocol

import numpy as np

from dojo.skills.geometry import T_from, T_to_pose7, inv_T, quat_to_R
from dojo.skills.kinematics import ARMS, TABLE_Z, X5Kin

MAX_JOINT_STEP = 0.2          # 5 rad/s x 40 ms (X5 actuator velocity_limit_sim)


class Env(Protocol):
    step_lim: int

    def step(self, action: dict) -> bool: ...          # True when the episode ended
    def proprio(self) -> dict: ...                      # official obs state
    @property
    def steps_used(self) -> int: ...


class RunnerEnv:
    """``dojo.simlab.runner.RoboDojoRunner`` (env 0) behind the executor's interface."""

    def __init__(self, runner, env_idx: int = 0):
        self.r = runner
        self.i = env_idx

    @property
    def step_lim(self) -> int:
        return int(self.r.env.step_lim)

    @property
    def steps_used(self) -> int:
        return int(self.r.env.take_action_cnt[self.i])

    def step(self, action: dict) -> bool:
        act = {k: np.asarray(v, np.float32) for k, v in action.items()}
        return bool(self.r.step(act)[self.i])

    def proprio(self) -> dict:
        return self.r.obs_state(self.i)

    # ---- dev-only
    def gt_objects(self) -> list[dict]:
        return self.r.objects_gt(self.i)

    def reward(self) -> dict:
        return self.r.reward(self.i)

    def score(self) -> float:
        return self.r.score(self.i)

    def measured_gripper(self) -> dict:
        g = self.r.robot_gt(self.i)
        return {a: g[a].get("gripper_open_measured") for a in ARMS}

    @property
    def instruction(self) -> str:
        return str(self.r.env.obs_manager.instruction[self.i])


class KinematicEnv:
    """Kinematic stand-in (no physics): for unit tests and dry runs of programs."""

    def __init__(self, objects: list[dict], step_lim: int = 1000, kin: X5Kin | None = None,
                 instruction: str = "", attach_radius: float = 0.012):
        self.kin = kin or X5Kin()
        self.step_lim = step_lim
        self._steps = 0
        self.q = {a: np.zeros(6) for a in ARMS}
        self.g = {a: 1.0 for a in ARMS}
        self.objects = copy.deepcopy(objects)        # rows like RoboDojoRunner.objects_gt()
        for o in self.objects:
            o["pos"] = np.asarray(o["pos"], float)
            o["quat"] = np.asarray(o["quat"], float)
            o["half"] = np.asarray(o.get("half", np.full(3, 0.02)), float)
            o.setdefault("type", "Rigid")
            o.setdefault("clutter", False)
            o.setdefault("inst_name", o["label"])
        self.attached: dict[str, tuple[str, np.ndarray]] = {}
        self.instruction = instruction
        self.attach_radius = attach_radius
        self.ended = False
        self.log: list[dict] = []

    @property
    def steps_used(self) -> int:
        return self._steps

    def proprio(self) -> dict:
        st = {}
        for a in ARMS:
            st[f"{a}_arm_joint_state"] = self.q[a].copy()
            st[f"{a}_ee_pose"] = T_to_pose7(self.kin.flange(a, self.q[a]))
            st[f"{a}_ee_joint_state"] = np.array([self.g[a]])
        return st

    def gt_objects(self) -> list[dict]:
        rows = []
        for o in self.objects:
            r = dict(o)
            r["half"] = np.asarray(o["half"]).tolist()
            r["pos"] = np.asarray(o["pos"]).tolist()
            r["quat"] = np.asarray(o["quat"]).tolist()
            rows.append(r)
        return rows

    def _obj(self, label: str) -> dict:
        return next(o for o in self.objects if o["label"] == label)

    def step(self, action: dict) -> bool:
        if self.ended or self._steps >= self.step_lim:
            self.ended = True
            return True
        self._steps += 1
        for a in ARMS:
            tgt = np.asarray(action[f"{a}_arm_joint_state"], float)
            d = tgt - self.q[a]
            self.q[a] = self.q[a] + np.clip(d, -MAX_JOINT_STEP, MAX_JOINT_STEP)
            g_new = float(np.clip(np.asarray(action[f"{a}_ee_joint_state"]).reshape(-1)[0], 0, 1))
            T_tcp = self.kin.tcp(a, self.q[a])
            if g_new < 0.5 and a not in self.attached:
                lab = self._object_at(T_tcp[:3, 3])
                if lab is not None:
                    o = self._obj(lab)
                    self.attached[a] = (lab, inv_T(T_tcp) @ T_from(o["pos"], quat=o["quat"]))
            if g_new > 0.6 and a in self.attached:
                lab, _ = self.attached.pop(a)
                self._drop(lab)
            self.g[a] = g_new
            if a in self.attached:
                lab, T_rel = self.attached[a]
                T = T_tcp @ T_rel
                o = self._obj(lab)
                o["pos"] = T[:3, 3].copy()
                o["quat"] = T_to_pose7(T)[3:]
        self.log.append({"step": self._steps, "attached": {a: v[0] for a, v in self.attached.items()}})
        if self._steps >= self.step_lim:
            self.ended = True
        return self.ended

    def _object_at(self, p) -> str | None:
        best, bd = None, 1e9
        for o in self.objects:
            if o.get("type") == "Geometry":
                continue
            loc = quat_to_R(o["quat"]).T @ (p - o["pos"])
            if np.all(np.abs(loc) <= o["half"] + self.attach_radius):
                d = float(np.linalg.norm(loc))
                if d < bd:
                    best, bd = o["label"], d
        return best

    def _drop(self, label: str) -> None:
        o = self._obj(label)
        R = quat_to_R(o["quat"])
        hz = float(np.abs(R @ np.diag(o["half"])).sum(axis=1)[2])
        support = TABLE_Z
        for other in self.objects:
            if other["label"] == label or other.get("type") == "Geometry":
                continue
            Ro = quat_to_R(other["quat"])
            ho = np.abs(Ro @ np.diag(other["half"])).sum(axis=1)
            if (abs(other["pos"][0] - o["pos"][0]) < ho[0] and abs(other["pos"][1] - o["pos"][1]) < ho[1]
                    and other["pos"][2] < o["pos"][2]):
                support = max(support, other["pos"][2] + ho[2])
        o["pos"] = np.array([o["pos"][0], o["pos"][1], support + hz])


def kinematic_from_world(world, **kw) -> KinematicEnv:
    """KinematicEnv from a WorldState (object boxes from the asset bboxes)."""
    rows = []
    for o in world.objects.values():
        rows.append({"label": o.label, "inst_name": o.label, "type": o.obj_type, "category": o.category,
                     "category_idx": o.idx, "clutter": o.clutter, "pos": o.pos.copy(), "quat": o.quat.copy(),
                     "half": o.asset.extents / 2})
    return KinematicEnv(rows, step_lim=world.step_lim or 1000, instruction=world.instruction, **kw)


def angle_deg(a, b) -> float:
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(a, b))))))
