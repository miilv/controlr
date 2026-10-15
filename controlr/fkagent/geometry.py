"""Joint-space geometry for the FK agent: offsets, FK in mm/deg, a joint sensitivity table, and
the motion constraints every agent target goes through before the arm moves.

Two joint frames (as in ``controlr.robot.so101.client``):
* MEASURED = LeRobot DEGREES (zero = middle of the calibrated range): what the server and the
  teleop recordings report;
* MODEL = URDF zero: ``model = measured + JOINT_OFFSETS_DEG``. Everything the agent sees and
  proposes (``q_deg``) is in the MODEL frame; only ``ServerArm`` converts back.

Constraints (``Limits``) are checked on the TCP at the target AND densely along the planned
path (start pose -> above the target -> straight down to it), on the other link points, on joint
limits and on the step size between consecutive targets. A target that breaks them is pulled
back toward the previous target (``plan_target``) or rejected; the agent is told either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from controlr.robot.so101 import kinematics as K

ARM = K.ARM_JOINTS
# configs/so101.yaml robot.params.joint_offsets_deg (fit from 13 table touches, rms 1.6 mm)
JOINT_OFFSETS_DEG = np.array([0.0, -5.07, -10.06, -1.36, 0.0])
# A survey start pose (MODEL deg): TCP ~(195, 8, 145) mm, the wrist camera ~270 mm up looking
# down at about x = 230 mm (stock mount); footprint on the table roughly x 100..380, y +-240 mm.
# The demos' start pose (folded rest, TCP ~150/5/17 mm) sees only the floor next to the base.
SURVEY_Q_DEG = (0.0, -50.0, 20.0, 90.0, -90.0)


def to_model(q_meas_deg, offsets=JOINT_OFFSETS_DEG) -> np.ndarray:
    return np.asarray(q_meas_deg, float)[..., :5] + np.asarray(offsets, float)


def to_measured(q_model_deg, offsets=JOINT_OFFSETS_DEG) -> np.ndarray:
    return np.asarray(q_model_deg, float)[..., :5] - np.asarray(offsets, float)


def fk_T(q_deg) -> np.ndarray:
    """4x4 base->TCP (metres) for MODEL joint angles in degrees."""
    return K.fk(np.radians(np.asarray(q_deg, float)[:5]))


def tcp_mm(q_deg) -> np.ndarray:
    return fk_T(q_deg)[:3, 3] * 1000.0


def tool_pitch_deg(q_deg) -> float:
    """Tool axis angle below the horizontal (90 = pointing straight down)."""
    z = fk_T(q_deg)[:3, 2]
    return math.degrees(math.asin(max(-1.0, min(1.0, -float(z[2])))))


def pose_summary(q_deg) -> dict:
    p = tcp_mm(q_deg)
    return {"tcp_mm": [round(float(v), 1) for v in p], "tool_pitch_deg": round(tool_pitch_deg(q_deg), 1)}


def jacobian_mm_per_deg(q_deg, h: float = 0.5) -> np.ndarray:
    """(3, 5): TCP mm moved per +1 deg of each joint at ``q_deg`` (central differences)."""
    q = np.asarray(q_deg, float)[:5]
    J = np.zeros((3, 5))
    for i in range(5):
        a, b = q.copy(), q.copy()
        a[i] += h
        b[i] -= h
        J[:, i] = (tcp_mm(a) - tcp_mm(b)) / (2 * h)
    return J


def link_points_mm(q_deg) -> np.ndarray:
    """(6, 3): the 5 joint origins and the TCP, base frame, mm."""
    return K.joint_points(np.radians(np.asarray(q_deg, float)[:5])) * 1000.0


# ----------------------------------------------------------------------------- constraints

@dataclass
class Limits:
    box_lo_mm: tuple = (140.0, -150.0, 5.0)     # TCP workspace (table z = 0)
    box_hi_mm: tuple = (270.0, 150.0, 260.0)
    link_min_z_mm: float = 20.0                  # wrist/elbow origins stay this far above the table
    joint_lo_deg: tuple = (-107.0, -97.0, -94.0, -92.0, -154.0)   # MODEL frame; replaced by the
    joint_hi_deg: tuple = (107.0, 97.0, 94.0, 92.0, 160.0)        # server's limits when live
    max_step_mm: float = 80.0                    # TCP change between consecutive targets
    max_joint_step_deg: float = 35.0             # per joint, between consecutive targets
    approach_dz_mm: float = 40.0                 # come down onto the target from this far above
    path_step_deg: float = 1.0                   # path sampling for the checks
    path_tol_mm: float = 2.0                     # the box may be missed by this much on the path (IK noise)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_server_limits(cls, limits_rad_measured, offsets=JOINT_OFFSETS_DEG, **kw) -> "Limits":
        lim = np.degrees(np.asarray(limits_rad_measured, float)) + np.asarray(offsets, float)[:, None]
        return cls(joint_lo_deg=tuple(float(v) for v in lim[:, 0]),
                   joint_hi_deg=tuple(float(v) for v in lim[:, 1]), **kw)


def in_box(p_mm, lim: Limits, tol: float = 0.0) -> bool:
    lo, hi = np.asarray(lim.box_lo_mm) - tol, np.asarray(lim.box_hi_mm) + tol
    return bool(np.all(p_mm >= lo) and np.all(p_mm <= hi))


def box_violation(p_mm, lim: Limits) -> str:
    out = []
    for i, ax in enumerate("xyz"):
        if p_mm[i] < lim.box_lo_mm[i]:
            out.append(f"{ax} {p_mm[i]:.0f} < {lim.box_lo_mm[i]:.0f}")
        elif p_mm[i] > lim.box_hi_mm[i]:
            out.append(f"{ax} {p_mm[i]:.0f} > {lim.box_hi_mm[i]:.0f}")
    return ", ".join(out)


def densify(rows, step_deg: float) -> list[np.ndarray]:
    """Joint-linear interpolation through ``rows`` with at most ``step_deg`` per joint per sample."""
    rows = [np.asarray(r, float)[:5] for r in rows]
    out = [rows[0]]
    for a, b in zip(rows[:-1], rows[1:]):
        n = max(1, int(math.ceil(float(np.max(np.abs(b - a))) / step_deg)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return out


def check_path(rows_deg, lim: Limits) -> str:
    """'' if every densified sample keeps the TCP in the box, the links above ``link_min_z_mm``
    and the joints inside their limits; else the first problem."""
    lo, hi = np.asarray(lim.joint_lo_deg), np.asarray(lim.joint_hi_deg)
    for q in densify(rows_deg, lim.path_step_deg):
        if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
            i = int(np.argmax(np.maximum(lo - q, q - hi)))
            return f"{ARM[i]} {q[i]:.1f} deg outside its limits [{lo[i]:.0f}, {hi[i]:.0f}]"
        pts = link_points_mm(q)
        if not in_box(pts[-1], lim, lim.path_tol_mm):
            return f"the TCP leaves the workspace on the way ({box_violation(pts[-1], lim)} mm)"
        if float(np.min(pts[1:-1, 2])) < lim.link_min_z_mm:
            return "a wrist/elbow joint comes within %.0f mm of the table" % float(np.min(pts[1:-1, 2]))
    return ""


@dataclass
class Plan:
    ok: bool
    target_q_deg: np.ndarray | None          # what will be executed (after any clamping)
    requested_q_deg: np.ndarray | None
    approach: list = field(default_factory=list)   # start -> above -> target waypoints (deg)
    clamped: bool = False
    scale: float = 1.0
    notes: list = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict:
        r = lambda a: None if a is None else [round(float(v), 2) for v in a]  # noqa: E731
        return {"ok": self.ok, "target_q_deg": r(self.target_q_deg), "requested_q_deg": r(self.requested_q_deg),
                "clamped": self.clamped, "scale": round(self.scale, 3), "notes": list(self.notes),
                "reason": self.reason, "n_waypoints": len(self.approach)}


def approach_path(start_q, target_q, lim: Limits) -> list[np.ndarray]:
    """start -> (target TCP + approach_dz above, same tool orientation) -> straight down to the
    target. Falls back to a direct joint-linear move when the point above is not reachable."""
    start_q, target_q = np.asarray(start_q, float), np.asarray(target_q, float)
    if lim.approach_dz_mm <= 0:
        return [start_q, target_q]
    T = fk_T(target_q)
    above = T[:3, 3] + np.array([0.0, 0.0, lim.approach_dz_mm / 1000.0])
    lim_rad = np.radians(np.stack([lim.joint_lo_deg, lim.joint_hi_deg], axis=1))
    q_above, pe, ae = K.ik(above, T[:3, :3], np.radians(target_q), lim_rad, iters=200)
    if pe > 0.003 or ae > 0.05:
        return [start_q, target_q]
    down, frac, _ = K.line_path(q_above, T[:3, 3], T[:3, :3], lim_rad)
    if frac < 1.0 or not down:
        return [start_q, target_q]
    rows = [start_q, np.degrees(q_above)] + [np.degrees(q) for q in down[:-1]] + [target_q]
    return rows


def _feasible(start_q, q, ref_q, lim: Limits, *, first: bool) -> tuple[str, list]:
    p = tcp_mm(q)
    if not in_box(p, lim):
        return f"target TCP outside the workspace ({box_violation(p, lim)} mm)", []
    if not first:
        d = float(np.linalg.norm(p - tcp_mm(ref_q)))
        if d > lim.max_step_mm + 1e-6:
            return f"step of {d:.0f} mm from the previous target > {lim.max_step_mm:.0f} mm", []
        j = float(np.max(np.abs(q - ref_q)))
        if j > lim.max_joint_step_deg + 1e-6:
            return f"a joint changes {j:.0f} deg from the previous target > {lim.max_joint_step_deg:.0f}", []
    rows = approach_path(start_q, q, lim)
    why = check_path(rows, lim)
    return why, rows


def plan_target(start_q, requested_q, lim: Limits, prev_target_q=None) -> Plan:
    """Validate a joint target. Joint limits are clipped; a target that leaves the box, steps too
    far from the previous target, or whose path is unsafe is pulled back along the joint line
    from the previous target (or the start pose) to the farthest safe fraction; below 5 % of the
    way = rejected."""
    start_q = np.asarray(start_q, float)[:5]
    req = np.asarray(requested_q, float)[:5]
    notes: list[str] = []
    lo, hi = np.asarray(lim.joint_lo_deg), np.asarray(lim.joint_hi_deg)
    q = np.clip(req, lo, hi)
    if np.any(np.abs(q - req) > 1e-6):
        i = int(np.argmax(np.abs(q - req)))
        notes.append(f"{ARM[i]} clipped to its limit ({req[i]:.1f} -> {q[i]:.1f} deg)")
    first = prev_target_q is None
    ref = start_q if first else np.asarray(prev_target_q, float)[:5]
    why, rows = _feasible(start_q, q, ref, lim, first=first)
    if not why:
        return Plan(True, q, req, rows, bool(notes), 1.0, notes)
    # bisection on the fraction of the way from ref to q
    good, good_rows, bad = 0.0, None, 1.0
    for _ in range(14):
        s = 0.5 * (good + bad)
        w, r = _feasible(start_q, ref + s * (q - ref), ref, lim, first=first)
        if w:
            bad = s
        else:
            good, good_rows = s, r
    if good < 0.05 or good_rows is None:
        return Plan(False, None, req, [], False, 0.0, notes, reason=why)
    tq = ref + good * (q - ref)
    notes.append(f"{why}: moved only {good * 100:.0f} % of the way from the "
                 f"{'start pose' if first else 'previous target'}")
    return Plan(True, tq, req, good_rows, True, good, notes, reason=why)


def lift_path(q_deg, dz_mm: float, lim: Limits) -> list[np.ndarray]:
    """Straight up by ``dz_mm`` (capped by the box top) from ``q_deg``; [] if not reachable."""
    T = fk_T(q_deg)
    top = min(T[2, 3] * 1000.0 + dz_mm, lim.box_hi_mm[2]) / 1000.0
    p = T[:3, 3].copy()
    p[2] = top
    lim_rad = np.radians(np.stack([lim.joint_lo_deg, lim.joint_hi_deg], axis=1))
    path, frac, _ = K.line_path(np.radians(q_deg), p, T[:3, :3], lim_rad)
    if not path:
        return []
    return [np.asarray(q_deg, float)] + [np.degrees(x) for x in path]
