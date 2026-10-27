"""Env adapters: what the executor drives.

* ``RunnerEnv`` wraps simlab's ``RoboDojoRunner`` (RoboDojo's own EvalEnv on our layouts; runs in
  RoboDojo's python). Actions are official joint action dicts; proprio is the official obs
  (joints, flange poses, grippers: RoboDojo's reading, or the strict command with ``DOJO_GRIP_OBS=command``,
  ``dojo.simlab.gripobs``). Ground truth (objects, reward, measured fingers) is
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

    def __init__(self, runner, env_idx: int = 0, grip_obs: str | None = None):
        from dojo.simlab.gripobs import GripObs
        self.r = runner
        self.i = env_idx
        self.grip = GripObs(grip_obs)         # DOJO_GRIP_OBS (default robodojo: proprio unchanged)

    @property
    def step_lim(self) -> int:
        return int(self.r.env.step_lim)

    @property
    def steps_used(self) -> int:
        return int(self.r.env.take_action_cnt[self.i])

    def step(self, action: dict) -> bool:
        """One counted env step for THIS env only (other envs, e.g. twin copies, hold still)."""
        act = {k: np.asarray(v, np.float32) for k, v in action.items()}
        self.grip.stepped(act)
        self.r.env.take_action_batch([act], env_idx_list=[self.i])
        return bool(self.r.env.end_flag[self.i])

    def proprio(self) -> dict:
        return self.grip.apply(self.r.obs_state(self.i))

    # ---- dev-only
    def gt_objects(self) -> list[dict]:
        return self.r.objects_gt(self.i)

    def reward(self) -> dict:
        return self.r.reward(self.i)

    def garment_gt(self, label: str | None = None) -> dict | None:
        return self.r.garment_gt(self.i, label)

    def score(self) -> float:
        return self.r.score(self.i)

    def actors_gt(self) -> dict:
        """DEV-ONLY: joint positions (arm + gripper) of the scene's scripted arms (robots that are not ours: an opponent,
        a demonstrator). G3 answers "is a scripted actor still moving" from it; at eval a watch read on the frames."""
        g = self.r.robot_gt(self.i)
        return {k: v["joint_pos"] for k, v in g.items() if v.get("type") != "target"}

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
            o["center"] = np.asarray(o.get("center", np.zeros(3)), float)      # bbox centre, object frame
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
            r["center"] = np.asarray(o["center"]).tolist()
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
            # a grasp happens when the fingers close (open -> closed), not whenever closed fingers touch
            # (a closing step below half open; narrow grasps open to 0.37 and close in 4 steps of < 0.15)
            if g_new < 0.5 and self.g[a] > g_new + 1e-6 and a not in self.attached:
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
        self._push_over()
        self.log.append({"step": self._steps, "attached": {a: v[0] for a, v in self.attached.items()}})
        if self._steps >= self.step_lim:
            self.ended = True
        return self.ended

    def _push_over(self) -> None:
        """Minimal tipping model: a closed fingertip moving horizontally into the upper part of a standing
        free object (longest axis vertical) tips it 90 deg about its far bottom edge."""
        tips = getattr(self, "_tips", {})
        now = {}
        for a in ARMS:
            T = self.kin.tcp(a, self.q[a])
            now[a] = T[:3, 3] + T[:3, 0] * 0.0126
            if self.g[a] > 0.1 or a in self.attached or a not in tips:     # closed (pushing) fingers only
                continue
            v = now[a] - tips[a]
            v[2] = 0.0
            if np.linalg.norm(v) < 1e-4:
                continue
            u = v / np.linalg.norm(v)
            held = {lab for lab, _ in self.attached.values()}
            for o in self.objects:
                if o.get("type") != "Rigid" or o["label"] in held:
                    continue
                R = quat_to_R(o["quat"])
                k = int(np.argmax(o["half"]))
                if abs(R[2, k]) < 0.94:
                    continue
                c = o["pos"] + R @ o["center"]
                loc = R.T @ (now[a] - c)
                h = np.abs(R @ np.diag(o["half"])).sum(axis=1)
                if not (np.all(np.abs(loc) <= o["half"] + 0.002) and now[a][2] > c[2] + 0.2 * h[2]):
                    continue
                axis = np.cross([0.0, 0.0, 1.0], u)                  # tip about the far bottom edge
                ext_u = float(np.abs(R.T @ u) @ o["half"])
                pivot = c + u * ext_u - np.array([0.0, 0.0, h[2]])
                from dojo.skills.geometry import axis_angle_R
                Rr = axis_angle_R(axis, math.pi / 2)
                c2 = pivot + Rr @ (c - pivot)
                R2 = Rr @ R
                o["quat"] = T_to_pose7(T_from(np.zeros(3), R2))[3:]
                o["pos"] = c2 - R2 @ o["center"]
        self._tips = now

    def _object_at(self, p) -> str | None:
        best, bd = None, 1e9
        for o in self.objects:
            if o.get("type") == "Geometry":
                continue
            loc = quat_to_R(o["quat"]).T @ (p - o["pos"]) - o["center"]
            if np.all(np.abs(loc) <= o["half"] + self.attach_radius):
                d = float(np.linalg.norm(loc))
                if d < bd:
                    best, bd = o["label"], d
        return best

    def _drop(self, label: str) -> None:
        o = self._obj(label)
        R = quat_to_R(o["quat"])
        # hooks (dry runs: a Geometry row's ``hang_points`` = [[x, y, z], ...], world): an object let go with a hook
        # point inside its box (a mug's handle over a rack's arm) hangs there
        for g in self.objects:
            for hp in (g.get("hang_points") or []) if g.get("type") == "Geometry" else []:
                loc = R.T @ (np.asarray(hp, float) - o["pos"]) - o["center"]
                if np.all(np.abs(loc) <= o["half"] + 0.005):
                    return
        R = self._disc_falls_flat(o, R)
        off = R @ o["center"]                    # root -> bbox centre (world)
        o["pos"] = o["pos"] + off
        try:
            self._drop_centre(o, R)
        finally:
            o["pos"] = o["pos"] - off

    def _disc_falls_flat(self, o: dict, R):
        """A thin disc (a coin) let go on its edge falls flat towards its lean, about its lowest rim point."""
        h = np.asarray(o["half"], float)
        k = int(np.argmin(h))
        a, b = np.delete(h, k)
        n = R[:, k]
        if not (h[k] < 0.0025 and abs(a - b) <= 0.05 * max(a, b)) or abs(float(n[2])) > 0.9:
            return R
        from dojo.skills.geometry import axis_angle_R
        up = np.array([0.0, 0.0, 1.0]) - n * float(n[2])
        up /= np.linalg.norm(up)
        lean = np.array([up[0], up[1], 0.0])
        if np.linalg.norm(lean) < 1e-6:
            lean = np.array([n[0], n[1], 0.0])
        lean /= np.linalg.norm(lean)
        c = o["pos"] + R @ o["center"]
        bottom = c - up * a
        ax = np.cross(up, lean)
        ang = math.acos(max(-1.0, min(1.0, float(up @ lean))))
        R2 = axis_angle_R(ax / np.linalg.norm(ax), ang) @ R
        c2 = np.array([bottom[0] + lean[0] * a, bottom[1] + lean[1] * a, c[2]])
        o["quat"] = T_to_pose7(T_from(np.zeros(3), R2))[3:]
        o["pos"] = c2 - R2 @ o["center"]
        return R2

    def _drop_centre(self, o: dict, R) -> None:
        label = o["label"]
        hz = float(np.abs(R @ np.diag(o["half"])).sum(axis=1)[2])
        support = TABLE_Z
        h_self = np.abs(R @ np.diag(o["half"])).sum(axis=1)
        if not (-0.70 <= o["pos"][0] <= 0.70 and -0.60 <= o["pos"][1] <= 0.50):     # off the table
            support = 0.05
            for g in self.objects:
                if g.get("type") == "Geometry":
                    hg = np.abs(quat_to_R(g["quat"]) @ np.diag(g["half"])).sum(axis=1)
                    if abs(g["pos"][0] - o["pos"][0]) < hg[0] and abs(g["pos"][1] - o["pos"][1]) < hg[1]:
                        support = g["pos"][2] - hg[2] + 0.01         # a container: its bottom
            o["pos"] = np.array([o["pos"][0], o["pos"][1], support + hz])
            return
        # pegs (dry runs: a Geometry row's ``pegs`` = [[x, y, plate_z], ...], world): a piece let go with its root
        # within 4 mm of a peg's axis slides down onto the plate (or the pieces already on the peg)
        root = o["pos"] - R @ o["center"]
        for g in self.objects:
            for px, py, pz in (g.get("pegs") or []) if g.get("type") == "Geometry" else []:
                if math.hypot(root[0] - px, root[1] - py) < 0.004 and o["pos"][2] - hz > pz:
                    o["pos"] = o["pos"] + np.array([px - root[0], py - root[1], 0.0])
                    support = pz
                    for other in self.objects:
                        if other["label"] == label or other.get("type") == "Geometry":
                            continue
                        Ro = quat_to_R(other["quat"])
                        if math.hypot(other["pos"][0] - px, other["pos"][1] - py) < 0.004:
                            ho = np.abs(Ro @ np.diag(other["half"])).sum(axis=1)
                            oc = other["pos"] + Ro @ other["center"]
                            if oc[2] < o["pos"][2]:
                                support = max(support, oc[2] + ho[2])
                    o["pos"] = np.array([o["pos"][0], o["pos"][1], support + hz])
                    return
        for other in self.objects:
            if other["label"] == label or other.get("type") == "Geometry":
                continue
            Ro = quat_to_R(other["quat"])
            ho = np.abs(Ro @ np.diag(other["half"])).sum(axis=1)
            oc = other["pos"] + Ro @ other["center"]
            # xy footprints (axis-aligned bounds) overlap and the other one is below
            if (abs(oc[0] - o["pos"][0]) < ho[0] + h_self[0] - 1e-3
                    and abs(oc[1] - o["pos"][1]) < ho[1] + h_self[1] - 1e-3
                    and oc[2] + ho[2] <= o["pos"][2] - hz + 0.01):
                support = max(support, oc[2] + ho[2])
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


class KinematicVerifier:
    """Plan + verify stand-in for tests: candidates roll on deep copies of a KinematicEnv."""

    def __init__(self, env: KinematicEnv):
        self.env = env
        self.stats = {"calls": 0, "rollouts": 0}

    def rollout(self, candidates, k: int = 1, held_labels=(), src: int = 0):
        from dojo.skills.verify import Outcome
        out = []
        for ci, acts in enumerate(candidates):
            for s in range(k):
                e = copy.deepcopy(self.env)
                for a in acts:
                    e.step(a)
                objs = {o["label"]: (np.asarray(o["pos"], float), np.asarray(o["quat"], float)) for o in e.objects}
                out.append(Outcome(ci, s, objs, {a: e.q[a].copy() for a in ARMS}, False, False, 0.0, e.ended))
        self.stats["calls"] += 1
        self.stats["rollouts"] += len(candidates) * k
        return out
