"""ARX X5 kinematics (both arms) on the public RoboDojo URDF, numpy only.

Chain constants are copied from RoboDojo's ``Assets/Robots/x5/X5A.urdf`` (MIT, RoboDojo
Contributors); ``X5Kin.from_urdf`` re-reads them from the file so a test can check the copy.
Arm roots and the TCP come from ``env_cfg/robot/dual_x5.yml`` and ``robot_config.yml``
(``gripper_bias: 0.145``).

Frames:
* ``flange`` = URDF ``link6`` = what the official obs reports as ``{arm}_ee_pose``;
  flange +x is the approach axis (out of the gripper), +y the closing axis (``joint7`` moves +y,
  ``joint8`` -y), +z = x × y.
* ``tcp`` = flange ∘ (0.145, 0, 0): the grasp point between the pads. Fingertips are 12.6 mm
  further out (link7 mesh extends to x = 0.071 past the joint7 origin at 0.0866).
* Gripper: RoboDojo's normalised command g in [0, 1] -> finger joint q = -0.01 + 0.054 g (each
  finger, clipped by PhysX to [0, 0.044]); pad gap = 2 q. "Open >= 0.8" (the scorer's banking
  threshold) is a measured gap >= 66 mm.

IK: damped least squares on the 6-D TCP pose from several seeds; the solution closest to the
current joints wins (proximal joints weighted more). The sim's joint limits are measured, not
the URDF's ±10 rad (``JOINT_LIMITS``).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dojo.skills.geometry import T_from, angle_between_R, quat_to_R, rotvec, rpy_to_R

ARMS = ("left", "right")
# env_cfg/robot/dual_x5.yml: default_root_pos / default_root_rot (w, x, y, z), first robot = left
ARM_ROOTS = {"left": ((-0.3, -0.45, 0.765), (0.707, 0.0, 0.0, 0.707)),
             "right": ((0.3, -0.45, 0.765), (0.707, 0.0, 0.0, 0.707))}
TCP_OFFSET = 0.145            # robot_config.yml gripper_bias
TIP_BEYOND_TCP = 0.0126       # fingertip past the TCP along +x
GRIP_SCALE = (-0.01, 0.044)   # robot_config.yml gripper_scale (finger joint for g = 0 / 1)
FINGER_MAX = 0.044            # URDF joint7/8 upper limit
TABLE_Z = 0.765
MAX_FLANGE_FROM_SHOULDER = 0.64   # upper arm 0.264 + forearm 0.251 + wrist offsets ~0.12

# X5A.urdf, joint1..joint6: (origin xyz, origin rpy, axis)
X5_CHAIN = (
    ((0.0, 0.0, 0.0605), (0.0, 0.0, 0.0), (0, 0, 1)),
    ((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0, 1, 0)),
    ((-0.264, 0.0, 0.0), (3.1416, 0.0, 0.0), (0, 1, 0)),
    ((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0, 1, 0)),
    ((0.06775, 0.0005, -0.0865), (0.0, 0.0, 0.0), (0, 0, 1)),
    ((0.02895, 0.0, 0.0865), (-3.1416, 0.0, 0.0), (1, 0, 0)),
)
# Working joint range used by IK (rad). The asset (ARX.usd) and the URDF both allow ±10 rad on
# j1-j5 and ±179.9 deg on j6, and self-collision is enabled, so the real limit is self-contact:
# keep IK on the branch the arm uses from home (j2, j3 >= 0, |j4|, |j5| < 1.9).
JOINT_LIMITS = np.array([[-2.9, 2.9], [-0.05, 3.4], [-0.05, 3.3], [-1.9, 1.9], [-1.9, 1.9], [-3.14, 3.14]])
# The sim stops joint 5 at |q5| 1.571-1.585 rad, on both sides and at any wrist roll (self-contact; the asset declares
# +-10 rad): of 7394 G3 trace steps commanding |q5| > 1.59, 6822 stopped there, and the TCP stayed 20-100 mm off the
# plan (fill_pen drops let go beside the holder; journal 2026-10-28-dojo-h95-fill). "measured" keeps IK inside it.
J5_STOP = 1.571
JOINT_LIMITS_MEASURED = JOINT_LIMITS.copy()
JOINT_LIMITS_MEASURED[4] = (-1.56, 1.56)
JOINT_LIMIT_SETS = {"wide": JOINT_LIMITS, "measured": JOINT_LIMITS_MEASURED}   # RuntimeConfig.joint_limits
# proximal joints cost more when choosing among IK solutions
JOINT_WEIGHTS = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.6])


def grip_to_gap(g: float) -> float:
    """Pad gap (m) for a normalised gripper value (free fingers)."""
    q = GRIP_SCALE[0] + float(g) * (GRIP_SCALE[1] - GRIP_SCALE[0])
    return 2 * min(max(q, 0.0), FINGER_MAX)


def gap_to_grip(gap: float) -> float:
    q = gap / 2
    return (q - GRIP_SCALE[0]) / (GRIP_SCALE[1] - GRIP_SCALE[0])


@dataclass
class IKResult:
    ok: bool
    q: np.ndarray | None
    pos_err: float               # m
    rot_err: float               # rad
    note: str = ""


@dataclass
class X5Kin:
    chain: tuple = X5_CHAIN
    roots: dict = field(default_factory=lambda: dict(ARM_ROOTS))
    limits: np.ndarray = field(default_factory=lambda: JOINT_LIMITS.copy())
    tcp_offset: float = TCP_OFFSET

    def __post_init__(self):
        self._joint_T = [T_from(o, rpy_to_R(r)) for o, r, _ in self.chain]
        self._axes = [np.asarray(a, float) for _, _, a in self.chain]
        self._base = {}
        for arm, (p, q) in self.roots.items():
            self._base[arm] = T_from(p, quat=np.asarray(q, float) / np.linalg.norm(q))
        self._tcp = T_from((self.tcp_offset, 0, 0))

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_urdf(cls, path: str | Path, **kw) -> "X5Kin":
        root = ET.parse(path).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        chain = []
        for i in range(1, 7):
            j = joints[f"joint{i}"]
            o = j.find("origin")
            xyz = tuple(float(v) for v in o.get("xyz").split())
            rpy = tuple(float(v) for v in o.get("rpy").split())
            ax = tuple(float(v) for v in j.find("axis").get("xyz").split())
            chain.append((xyz, rpy, ax))
        return cls(chain=tuple(chain), **kw)

    # ------------------------------------------------------------------ FK
    def base(self, arm: str) -> np.ndarray:
        return self._base[arm]

    def frames(self, arm: str, q) -> list[np.ndarray]:
        """World pose of every joint frame after its rotation (index 5 = flange/link6)."""
        T = self._base[arm].copy()
        out = []
        for Tj, ax, qi in zip(self._joint_T, self._axes, np.asarray(q, float)):
            T = T @ Tj
            R = _axis_R(ax, qi)
            T = T.copy()
            T[:3, :3] = T[:3, :3] @ R
            out.append(T)
        return out

    def flange(self, arm: str, q) -> np.ndarray:
        return self.frames(arm, q)[-1]

    def tcp(self, arm: str, q) -> np.ndarray:
        return self.flange(arm, q) @ self._tcp

    def flange_from_tcp(self, T_tcp) -> np.ndarray:
        Tf = np.asarray(T_tcp, float).copy()
        Tf[:3, 3] = Tf[:3, 3] - Tf[:3, 0] * self.tcp_offset
        return Tf

    def tcp_from_flange(self, T_flange) -> np.ndarray:
        return np.asarray(T_flange, float) @ self._tcp

    def jacobian(self, arm: str, q, point_offset: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Geometric Jacobian (6x6: linear; angular) of the TCP. Returns (J, T_tcp)."""
        fr = self.frames(arm, q)
        Tt = fr[-1] @ (self._tcp if point_offset is None else T_from((point_offset, 0, 0)))
        p = Tt[:3, 3]
        A = np.stack([T[:3, :3] @ ax for T, ax in zip(fr, self._axes)])          # (6, 3) joint axes, world
        D = p - np.stack([T[:3, 3] for T in fr])                                  # (6, 3)
        J = np.empty((6, 6))
        J[0] = A[:, 1] * D[:, 2] - A[:, 2] * D[:, 1]
        J[1] = A[:, 2] * D[:, 0] - A[:, 0] * D[:, 2]
        J[2] = A[:, 0] * D[:, 1] - A[:, 1] * D[:, 0]
        J[3:] = A.T
        return J, Tt

    # ------------------------------------------------------------------ IK
    def solve(self, arm: str, T_target, q0, iters: int = 120, rot_w: float = 0.2, tol_p: float = 5e-4,
              tol_r: float = math.radians(0.3), damping: float = 0.02) -> IKResult:
        """One DLS run from ``q0`` towards the TCP pose ``T_target``."""
        q = np.clip(np.asarray(q0, float).copy(), self.limits[:, 0], self.limits[:, 1])
        Tt = np.asarray(T_target, float)
        W = np.diag([1, 1, 1, rot_w, rot_w, rot_w])
        ep = er = 1e9
        best_e, since = 1e9, 0
        for it in range(iters):
            J, T = self.jacobian(arm, q)
            e = np.r_[Tt[:3, 3] - T[:3, 3], rotvec(Tt[:3, :3] @ T[:3, :3].T)]
            ep, er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            if ep < tol_p and er < tol_r:
                return IKResult(True, q, ep, er)
            tot = ep + rot_w * er
            if tot < best_e * 0.98:
                best_e, since = tot, 0
            else:
                since += 1
                if since > 12 and tot > 0.01:          # stalled far from the target (limits / out of reach)
                    break
            Jw, ew = W @ J, W @ e
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + damping ** 2 * np.eye(6), ew)
            n = np.abs(dq).max()
            if n > 0.4:
                dq *= 0.4 / n
            q = np.clip(q + dq, self.limits[:, 0], self.limits[:, 1])
        return IKResult(ep < 2e-3 and er < math.radians(1.0), q, ep, er, note="max iters")

    def ik(self, arm: str, T_target, q_current, seeds: list | None = None, rng_seed: int = 0,
           n_random: int = 4, accept_local: float = 1e9, **kw) -> IKResult:
        """Best of several DLS runs: current joints, wrist ±pi, home, a few fixed random seeds.
        Picks the valid solution closest to ``q_current`` (weighted). Deterministic, so identical calls are
        memoised (relay legs and handover searches re-screen the same grasps: planning is 80-95 % of G3 wall)."""
        if seeds is None and not kw:
            key = (arm, np.round(np.asarray(T_target, float)[:3, :], 6).tobytes(),
                   np.round(np.asarray(q_current, float), 6).tobytes(), rng_seed, n_random, accept_local)
            cache = self.__dict__.setdefault("_ik_cache", {})
            hit = cache.get(key)
            if hit is not None:
                return IKResult(hit.ok, None if hit.q is None else hit.q.copy(), hit.pos_err, hit.rot_err, hit.note)
            r = self._ik(arm, T_target, q_current, None, rng_seed, n_random, accept_local)
            if len(cache) > 300000:
                cache.clear()
            cache[key] = IKResult(r.ok, None if r.q is None else r.q.copy(), r.pos_err, r.rot_err, r.note)
            return r
        return self._ik(arm, T_target, q_current, seeds, rng_seed, n_random, accept_local, **kw)

    def _ik(self, arm: str, T_target, q_current, seeds: list | None = None, rng_seed: int = 0,
            n_random: int = 4, accept_local: float = 1e9, **kw) -> IKResult:
        qc = np.asarray(q_current, float)
        Tt = np.asarray(T_target, float)
        shoulder = (self._base[arm] @ self._joint_T[0] @ self._joint_T[1])[:3, 3]
        if np.linalg.norm(Tt[:3, 3] - Tt[:3, 0] * self.tcp_offset - shoulder) > MAX_FLANGE_FROM_SHOULDER:
            return IKResult(False, None, 1e9, 1e9, "out of reach")
        cand = [qc]
        for s in (math.pi, -math.pi):
            q = qc.copy()
            q[5] = q[5] + s
            cand.append(q)
        cand.append(np.zeros(6))
        cand.append(np.array([0.0, 1.2, 1.0, -0.5, 0.0, 0.0]))
        # the base turned towards the target, arm reaching out (far / high targets converge from here)
        pb = np.linalg.inv(self._base[arm]) @ np.r_[Tt[:3, 3], 1.0]
        th = float(np.clip(math.atan2(pb[1], pb[0]), self.limits[0, 0], self.limits[0, 1]))
        cand.append(np.array([th, 1.2, 1.0, -0.5, 0.0, 0.0]))
        cand.append(np.array([th, 1.6, 1.4, -0.3, 0.0, 0.0]))
        if seeds:
            cand.extend(np.asarray(s, float) for s in seeds)
        rng = np.random.default_rng(rng_seed)
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        for _ in range(n_random):
            cand.append(np.clip(qc + rng.normal(0, 0.6, 6), lo, hi))
        best, best_cost, last = None, 1e18, None
        for i, s in enumerate(cand):
            r = self.solve(arm, T_target, s, **kw)
            last = r
            if not r.ok:
                continue
            cost = float(np.sum(JOINT_WEIGHTS * np.abs(r.q - qc)))
            if cost < best_cost:
                best, best_cost = r, cost
            if i == 0 and cost < accept_local:      # the local solution from the current joints
                break
        if best is None:
            return IKResult(False, None, last.pos_err if last else 1e9, last.rot_err if last else 1e9, "no seed converged")
        return best

    # ------------------------------------------------------------------ geometry for collision checks
    def link_points(self, arm: str, q, n_seg: int = 4, frames: list | None = None) -> np.ndarray:
        """Points sampled along the joint chain from joint2 to the flange, world frame (N,3).
        Coarse: for table clearance and box obstacles (the gripper is ``palm_points`` +
        ``finger_points``)."""
        fr = frames if frames is not None else self.frames(arm, q)
        pts = []
        prev = fr[1][:3, 3]
        for T in fr[2:]:
            p = T[:3, 3]
            for t in np.linspace(0, 1, n_seg + 1)[1:]:
                pts.append(prev + t * (p - prev))
            prev = p
        return np.asarray(pts)

    def palm_points(self, T_flange) -> np.ndarray:
        """Corners of the palm / finger-rail box (link6 mesh bounds x 0..0.084, y ±0.087, z ±0.03)."""
        loc = np.array([(x, y, z) for x in (0.0, 0.084) for y in (-0.08, 0.0, 0.08) for z in (-0.03, 0.03)])
        return (np.asarray(T_flange)[:3, :3] @ loc.T).T + np.asarray(T_flange)[:3, 3]

    def gripper_points(self, T_flange, gripper: float = 1.0) -> np.ndarray:
        return np.vstack([self.palm_points(T_flange), self.finger_points(T_flange, gripper)])

    def finger_points(self, T_flange, gripper: float = 1.0) -> np.ndarray:
        """Finger surface samples in world. From the link7/8 meshes: the finger tapers from a 37 mm
        block at its root (flange x 0.072-0.107) to 13 mm (x 0.107-0.127) to 3 mm at the tip
        (x 0.127-0.158); its inner face is at y = q + 0.4 mm (q = finger joint)."""
        qf = min(max(GRIP_SCALE[0] + gripper * (GRIP_SCALE[1] - GRIP_SCALE[0]), 0.0), FINGER_MAX)
        zones = ((0.075, 0.105, 0.0378, 0.028), (0.107, 0.126, 0.0134, 0.015), (0.128, 0.1576, 0.0034, 0.009))
        loc = []
        for s in (1, -1):
            for x0, x1, yo, zh in zones:
                for x in (x0, x1):
                    for y in (0.0004, yo):
                        for z in (-zh, zh):
                            loc.append((x, s * (qf + y), z))
        loc = np.asarray(loc)
        return (np.asarray(T_flange)[:3, :3] @ loc.T).T + np.asarray(T_flange)[:3, 3]


def _axis_R(ax, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    if ax[0] == 1 and ax[1] == 0 and ax[2] == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax[1] == 1 and ax[0] == 0 and ax[2] == 0:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if ax[2] == 1 and ax[0] == 0 and ax[1] == 0:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    from dojo.skills.geometry import axis_angle_R
    return axis_angle_R(ax, a)


def pose_err(Ta, Tb) -> tuple[float, float]:
    """(position m, rotation rad) between two 4x4 poses."""
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), angle_between_R(Ta[:3, :3], Tb[:3, :3])


def down_R(yaw: float) -> np.ndarray:
    """TCP orientation pointing straight down (flange +x = world -z) with the closing axis (+y)
    at heading ``yaw`` (rad) in the world xy plane."""
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def tilted_R(yaw: float, pitch_deg: float, heading: float) -> np.ndarray:
    """Approach axis tilted ``pitch_deg`` from straight down towards horizontal heading ``heading``
    (the direction the gripper points in xy), closing axis horizontal at ``yaw``... projected to
    stay orthogonal. pitch 90 = straight down (as PyRUA counts it)."""
    p = math.radians(pitch_deg)
    x = np.array([math.cos(p) * math.cos(heading), math.cos(p) * math.sin(heading), -math.sin(p)])
    y0 = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    y = y0 - x * float(x @ y0)
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(x, [0, 0, 1.0])
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def quat_R(q) -> np.ndarray:
    return quat_to_R(q)
