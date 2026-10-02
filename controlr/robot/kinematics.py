"""UR3 CB3 kinematics in the UR controller base frame (numpy only).

WHY a local copy instead of importing PHANTOM: controlr's harness process must
not depend on scipy / PHANTOM, and the safety envelope needs FK/IK on every
MOVE line. The DH constants and conventions are ported from PHANTOM
``phantom/sim/kinematics.py`` (nominal manufacturer UR3 values, not the
serial's factory calibration):

* the base is the UR controller ``base`` frame (z up) — the frame the teach
  pendant / RTDE ``getActualTCPPose`` report in;
* the DH end frame equals ROS ``tool0`` (not ROS ``flange``);
* the TCP is ``tool0 @ tcp_offset`` where ``tcp_offset`` is a UR-style pose
  (xyz, rotvec); default +0.18 m along tool z = Robotiq 2F-85 fingertip
  centre (PHANTOM ``configs/hardware.yaml`` arm.tcp_offset_m).

Poses are (position m, rotation vector rad). Euler angles appear only in the
LLM-facing helpers (``rpy_*``: extrinsic roll/pitch/yaw about base x/y/z, i.e.
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)), because "yaw" is what a model can reason
about in an image while a rotation vector is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3",
)
UR3_A = np.array([0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0])
UR3_D = np.array([0.1519, 0.0, 0.0, 0.11235, 0.08535, 0.0819])
UR3_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
DEFAULT_TCP_OFFSET = (0.0, 0.0, 0.18, 0.0, 0.0, 0.0)
# UR3 CB3 hardware joint range is +-360 deg on every joint (wrist 3 is infinite
# on CB3 but the controller still enforces +-360 deg by default).
UR3_JOINT_LIMITS = np.array([[-2 * np.pi, 2 * np.pi]] * 6)


# ---------------------------------------------------------------------------
# rotation helpers
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotvec_to_matrix(rv) -> np.ndarray:
    """Rodrigues formula."""
    rv = np.asarray(rv, dtype=float)
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3) + _skew(rv)
    k = _skew(rv / th)
    return np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * (k @ k)


def matrix_to_rotvec(R) -> np.ndarray:
    """Inverse Rodrigues, robust near 0 and pi (angle in [0, pi])."""
    R = np.asarray(R, dtype=float)
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(cos))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return 0.5 * w
    if th > np.pi - 1e-4:
        # sin(th) ~ 0: recover the axis from the symmetric part, sign from w.
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        axis = B[:, i] / np.sqrt(max(B[i, i], 1e-15))
        if np.dot(axis, w) < 0:
            axis = -axis
        return axis / np.linalg.norm(axis) * th
    return w * (th / (2.0 * np.sin(th)))


def rpy_to_matrix(rpy) -> np.ndarray:
    """Extrinsic x-y-z (roll about base x, then pitch about base y, then yaw
    about base z): R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(R) -> np.ndarray:
    """Inverse of ``rpy_to_matrix``; pitch in [-pi/2, pi/2]. At the gimbal
    lock (|pitch| = 90 deg) roll is set to 0 and folded into yaw."""
    R = np.asarray(R, dtype=float)
    sp = -R[2, 0]
    if abs(sp) > 1.0 - 1e-9:
        p = np.copysign(np.pi / 2, sp)
        y = np.arctan2(-R[0, 1], R[1, 1])
        return np.array([0.0, p, y])
    p = np.arcsin(np.clip(sp, -1.0, 1.0))
    r = np.arctan2(R[2, 1], R[2, 2])
    y = np.arctan2(R[1, 0], R[0, 0])
    return np.array([r, p, y])


def rotvec_to_rpy(rv) -> np.ndarray:
    return matrix_to_rpy(rotvec_to_matrix(rv))


def rpy_to_rotvec(rpy) -> np.ndarray:
    return matrix_to_rotvec(rpy_to_matrix(rpy))


def rotation_angle(R) -> float:
    """Geodesic angle of a rotation matrix (rad). atan2 form: accurate near 0,
    where arccos of the trace loses half the digits."""
    R = np.asarray(R, dtype=float)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return float(np.arctan2(0.5 * np.linalg.norm(w), 0.5 * (np.trace(R) - 1.0)))


def pose_to_matrix(pos, rotvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotvec_to_matrix(rotvec)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------

def dh_transform(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [c, -s * ca, s * sa, a * c],
        [s, c * ca, -c * sa, a * s],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


@dataclass
class IKOptions:
    """Tolerances are the *acceptance* criteria; the solver iterates a bit
    past them so accepted solutions are comfortably inside."""
    pos_tol: float = 1e-3                    # m
    rot_tol: float = float(np.deg2rad(0.5))  # rad
    max_iters: int = 150
    damping: float = 0.02
    step_limit: float = 0.25                 # rad, largest joint change per iteration
    rot_weight: float = 0.2                  # m per rad: scales orientation residual (conditioning)
    restarts: int = 8                        # extra seeds, used ONLY if the seed attempt fails
    restart_sigma: float = 0.6               # rad, perturbation of restart seeds
    max_joint_delta: float | None = None     # reject solutions farther than this from the seed (branch guard)


@dataclass
class UR3Kinematics:
    """FK / Jacobian / IK for the UR3 CB3 with a configurable TCP offset."""
    tcp_offset: tuple[float, ...] = DEFAULT_TCP_OFFSET
    joint_limits: np.ndarray = field(default_factory=lambda: UR3_JOINT_LIMITS.copy())
    options: IKOptions = field(default_factory=IKOptions)

    def __post_init__(self) -> None:
        off = np.asarray(self.tcp_offset, dtype=float)
        if off.shape != (6,):
            raise ValueError("tcp_offset must be (x, y, z, rx, ry, rz)")
        self._T_tool_tcp = pose_to_matrix(off[:3], off[3:])
        self.joint_limits = np.asarray(self.joint_limits, dtype=float).reshape(6, 2)

    # -- forward -------------------------------------------------------------
    def frames(self, q) -> np.ndarray:
        """Base, the six DH frames (last = tool0) and the TCP: shape (8,4,4).
        Used by the mock renderer to draw the arm."""
        q = np.asarray(q, dtype=float).reshape(6)
        out = [np.eye(4)]
        for qi, a, d, al in zip(q, UR3_A, UR3_D, UR3_ALPHA):
            out.append(out[-1] @ dh_transform(qi, a, d, al))
        out.append(out[-1] @ self._T_tool_tcp)
        return np.asarray(out)

    def fk_matrix(self, q) -> np.ndarray:
        return self.frames(q)[-1]

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        """TCP (pos m, rotvec rad) in the controller base frame."""
        T = self.fk_matrix(q)
        return T[:3, 3].copy(), matrix_to_rotvec(T[:3, :3])

    def jacobian(self, q) -> np.ndarray:
        """6x6 geometric Jacobian in the base frame: rows = (v, omega) of the TCP."""
        F = self.frames(q)
        p = F[-1][:3, 3]
        z = F[:6, :3, 2]
        o = F[:6, :3, 3]
        return np.vstack([np.cross(z, p - o).T, z.T])

    # -- inverse -------------------------------------------------------------
    def _solve(self, q0: np.ndarray, p_t: np.ndarray, R_t: np.ndarray | None,
               lim: np.ndarray, o: IKOptions) -> tuple[np.ndarray, float, float]:
        """One damped-least-squares descent from ``q0`` (projected onto the
        joint limits each step; stops early when converged or stalled).
        Returns (q, pos_err, rot_err)."""
        q = np.clip(q0.astype(float).copy(), lim[:, 0], lim[:, 1])
        w = np.r_[np.ones(3), np.full(3, o.rot_weight)]
        best_cost, stall = np.inf, 0
        for _ in range(o.max_iters):
            F = self.frames(q)
            T = F[-1]
            dp = p_t - T[:3, 3]
            pe = float(np.linalg.norm(dp))
            dr = matrix_to_rotvec(R_t @ T[:3, :3].T) if R_t is not None else np.zeros(3)
            re = float(np.linalg.norm(dr))
            if pe < 0.1 * o.pos_tol and re < 0.1 * o.rot_tol:
                return q, pe, re
            cost = pe + o.rot_weight * re
            if cost < best_cost - 1e-6:
                best_cost, stall = cost, 0
            else:
                stall += 1
                if stall >= 15:          # local minimum / limit-blocked: give up this seed
                    break
            z = F[:6, :3, 2]
            J = np.vstack([np.cross(z, T[:3, 3] - F[:6, :3, 3]).T, z.T])
            if R_t is not None:
                J = J * w[:, None]
                err = np.r_[dp, dr] * w
            else:
                J, err = J[:3], dp
            n = J.shape[0]
            dq = J.T @ np.linalg.solve(J @ J.T + o.damping ** 2 * np.eye(n), err)
            big = float(np.max(np.abs(dq)))
            if big > o.step_limit:
                dq *= o.step_limit / big
            q = np.clip(q + dq, lim[:, 0], lim[:, 1])
        T = self.fk_matrix(q)
        pe = float(np.linalg.norm(p_t - T[:3, 3]))
        re = rotation_angle(R_t @ T[:3, :3].T) if R_t is not None else 0.0
        return q, pe, re

    def ik(self, pos, rotvec=None, q_seed=None, joint_limits=None, *,
           orientation: str = "full", options: IKOptions | None = None,
           rng_seed: int = 0) -> np.ndarray | None:
        """Joint angles reaching TCP ``pos`` (+ orientation), nearest the seed.

        orientation: ``full`` (match ``rotvec``), ``lock`` (keep the seed's TCP
        orientation; ``rotvec`` ignored — what rotation=none modes want: the
        gripper keeps pointing where it points), ``free`` (position only).
        Restarts from perturbed seeds happen only when the seed attempt fails,
        so the common small-step case costs one descent and stays on the
        current branch. Returns None when no solution meets the tolerances
        (within ``joint_limits``, default the hardware limits).
        """
        o = options or self.options
        seed = np.asarray(q_seed if q_seed is not None else np.zeros(6), dtype=float).reshape(6)
        lim = self.joint_limits if joint_limits is None else np.asarray(joint_limits, float).reshape(6, 2)
        p_t = np.asarray(pos, dtype=float).reshape(3)
        if orientation == "full":
            if rotvec is None:
                raise ValueError("orientation='full' needs rotvec")
            R_t = rotvec_to_matrix(rotvec)
        elif orientation == "lock":
            R_t = self.fk_matrix(seed)[:3, :3]
        elif orientation == "free":
            R_t = None
        else:
            raise ValueError(f"unknown orientation mode {orientation!r}")

        def ok(q, pe, re) -> bool:
            if pe > o.pos_tol or re > o.rot_tol:
                return False
            return o.max_joint_delta is None or float(np.max(np.abs(q - seed))) <= o.max_joint_delta

        q, pe, re = self._solve(seed, p_t, R_t, lim, o)
        if ok(q, pe, re):
            return q
        rng = np.random.default_rng(rng_seed)
        for k in range(o.restarts):
            # widen the perturbation gradually so nearby branches are found first
            sigma = o.restart_sigma * (1.0 + k / 2.0)
            q, pe, re = self._solve(seed + rng.normal(0.0, sigma, 6), p_t, R_t, lim, o)
            if ok(q, pe, re):
                return q
        return None


# Module-level convenience (the ARCHITECTURE contract names ``fk``/``ik``).
_DEFAULT = UR3Kinematics()


def fk(q) -> tuple[np.ndarray, np.ndarray]:
    return _DEFAULT.fk(q)


def ik(pos, rotvec, q_seed, joint_limits=None, **kw) -> np.ndarray | None:
    return _DEFAULT.ik(pos, rotvec, q_seed, joint_limits, **kw)


def jacobian(q) -> np.ndarray:
    return _DEFAULT.jacobian(q)
