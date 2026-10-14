"""Joint zero offsets of the SO-101 from table touches (numpy only; CLI: scripts/so101_fit_offsets.py).

FK assumes the URDF zero of every joint is the middle of LeRobot's calibrated range (DEGREES
mode). The calibrated ranges are wider than the URDF's limits, so the mid-range zeros can be off
by several degrees. Here the operator teleoperates the arm and touches the CLOSED jaw tips to the
table (z = 0) at several spots, pausing ~1 s at each. Every pause is a constraint

    FK(q_measured + offset).z + dz = tip_z

and the offsets (shoulder_lift, elbow_flex, wrist_flex; optionally a base height dz) come out of a
regularised least-squares fit (prior offset 0). shoulder_pan turns about the vertical axis and
cannot change any height: it is not observable from touches. wrist_roll barely moves the TCP
(it lies ~8 mm off the roll axis): it is left at 0.

Recording format: scripts/so101_teleop_record.py (joints.jsonl + calib_so101_follower.json).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from controlr.robot.so101 import kinematics as K

ARM = K.ARM_JOINTS
FITTABLE = ("shoulder_lift", "elbow_flex", "wrist_flex")
TICKS_PER_REV = 4095            # LeRobot's DEGREES normalisation


# ------------------------------------------------------------------------- recordings

def norm_to_deg(v, range_min: float, range_max: float):
    """LeRobot RANGE_M100_100 value -> DEGREES (zero = middle of the calibrated range)."""
    raw = (np.asarray(v, float) + 100.0) / 200.0 * (range_max - range_min) + range_min
    return (raw - (range_min + range_max) / 2.0) * 360.0 / TICKS_PER_REV


def load_recording(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(t s, follower q deg (N, 5), gripper 0..100 (N,))`` of a teleop recording dir."""
    d = Path(path)
    cal = json.loads((d / "calib_so101_follower.json").read_text())
    t, q, g = [], [], []
    for line in (d / "joints.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:          # a torn last line after a crash
            continue
        f = rec.get("follower")
        if not f or "event" in rec:
            continue
        t.append(float(rec["t"]))
        q.append([float(norm_to_deg(f[j], cal[j]["range_min"], cal[j]["range_max"])) for j in ARM])
        g.append(float(f.get("gripper", math.nan)))
    return np.asarray(t, float), np.asarray(q, float).reshape(-1, 5), np.asarray(g, float)


@dataclass
class Dwell:
    t0: float
    t1: float
    q_deg: np.ndarray          # mean follower q over the dwell
    gripper: float
    source: str = ""
    surface_z_mm: float = 0.0    # height of the touched surface (table 0, or a block of known height)

    def as_dict(self) -> dict:
        return {"t0": round(self.t0, 3), "t1": round(self.t1, 3), "q_deg": np.round(self.q_deg, 3).tolist(),
                "gripper": round(self.gripper, 2), "source": self.source, "surface_z_mm": self.surface_z_mm}


def find_dwells(t, q_deg, gripper, *, still_deg: float = 0.5, min_s: float = 0.6, gripper_max: float = 8.0,
                min_lift_deg: float = -90.0, source: str = "") -> tuple[list[Dwell], dict]:
    """Touch dwells: windows of >= ``min_s`` where every joint stays within ``still_deg`` (max - min),
    with the jaws nearly closed (gripper <= ``gripper_max`` %) and not the folded rest pose
    (shoulder_lift > ``min_lift_deg``). Returns the dwells and how many still windows each rule
    rejected (to explain an empty result)."""
    t, q, g = np.asarray(t, float), np.asarray(q_deg, float), np.asarray(gripper, float)
    still: list[tuple[int, int]] = []
    i, n = 0, len(t)
    while i < n:
        lo, hi = q[i].copy(), q[i].copy()
        j = i
        while j + 1 < n:
            lo2, hi2 = np.minimum(lo, q[j + 1]), np.maximum(hi, q[j + 1])
            if np.any(hi2 - lo2 > still_deg):
                break
            lo, hi, j = lo2, hi2, j + 1
        if t[j] - t[i] >= min_s:
            still.append((i, j))
            i = j + 1
        else:
            i += 1
    dwells: list[Dwell] = []
    rejected = {"still_windows": len(still), "gripper_open": 0, "rest_pose": 0}
    for i, j in still:
        qm, gm = q[i:j + 1].mean(axis=0), float(np.nanmean(g[i:j + 1]))
        if not gm <= gripper_max:
            rejected["gripper_open"] += 1
            continue
        if qm[1] <= min_lift_deg:
            rejected["rest_pose"] += 1
            continue
        prev = dwells[-1] if dwells else None
        if prev is not None and t[i] - prev.t1 < 0.3 and np.all(np.abs(prev.q_deg - qm) < still_deg):
            w0, w1 = prev.t1 - prev.t0, t[j] - t[i]          # one dwell split by a slow drift
            prev.q_deg = (prev.q_deg * w0 + qm * w1) / max(w0 + w1, 1e-9)
            prev.t1 = float(t[j])
            continue
        dwells.append(Dwell(float(t[i]), float(t[j]), qm, gm, source))
    return dwells, rejected


# ------------------------------------------------------------------------------- model

def tcp_mm(q_deg, offsets_deg=None, dz_mm: float = 0.0) -> np.ndarray:
    """TCP position (mm, base frame) at measured angles ``q_deg`` with zero offsets added."""
    off = np.zeros(5) if offsets_deg is None else np.asarray(offsets_deg, float)
    p = K.fk(np.radians(np.asarray(q_deg, float)[:5] + off))[:3, 3] * 1000.0
    return p + np.array([0.0, 0.0, dz_mm])


@dataclass
class FitResult:
    joints: tuple[str, ...]
    fit_dz: bool
    offsets_deg: np.ndarray                 # (5,) per ARM joint (unfitted ones 0)
    dz_mm: float
    std: np.ndarray                         # per fitted parameter (deg, then mm for dz)
    residuals_mm: np.ndarray                # per dwell: model z - tip_z after the fit
    residuals_before_mm: np.ndarray
    cond: float                             # condition number of the data's information matrix
    prior_share: np.ndarray                 # posterior std / prior std per parameter (1 = data says nothing)
    weakest: str                            # the least determined parameter combination
    warnings: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return list(self.joints) + (["dz"] if self.fit_dz else [])

    @property
    def params(self) -> np.ndarray:
        idx = [ARM.index(j) for j in self.joints]
        return np.concatenate([self.offsets_deg[idx], [self.dz_mm] if self.fit_dz else []])


def fit_offsets(q_deg, *, tip_z_mm=0.0, joints=FITTABLE, fit_dz: bool = False,
                prior_sigma_deg: float = 10.0, prior_sigma_dz_mm: float = 20.0, noise_mm: float = 2.0,
                iters: int = 50) -> FitResult:
    """Regularised Gauss-Newton fit of joint zero offsets (deg) so that every touch's TCP is at
    ``tip_z_mm`` (a number, or one per touch). ``noise_mm``: assumed per-touch height scatter
    (operator, jaw compliance). ``dz`` is nearly degenerate with the joint offsets when every touch
    is at one height: fit it only with touches on two or more surface heights."""
    Q = np.asarray(q_deg, float).reshape(-1, 5)
    tip = np.broadcast_to(np.asarray(tip_z_mm, float), (len(Q),))
    joints = tuple(joints)
    if "shoulder_pan" in joints:
        raise ValueError("shoulder_pan turns about the vertical: no touch height depends on it")
    idx = [ARM.index(j) for j in joints]
    npar = len(idx) + int(fit_dz)
    sig = np.array([prior_sigma_deg] * len(idx) + ([prior_sigma_dz_mm] if fit_dz else []), float)

    def unpack(th):
        off = np.zeros(5)
        off[idx] = th[:len(idx)]
        return off, (float(th[-1]) if fit_dz else 0.0)

    def zres(th):
        off, dz = unpack(th)
        return np.array([tcp_mm(q, off, dz)[2] for q in Q]) - tip

    def jac(th, r0):
        J = np.zeros((len(Q), npar))
        for k in range(npar):
            d = th.copy()
            d[k] += 1e-4
            J[:, k] = (zres(d) - r0) / 1e-4
        return J

    th = np.zeros(npar)
    r_before = zres(th) if len(Q) else np.zeros(0)
    for _ in range(iters):
        r = zres(th)
        J = jac(th, r)
        A = J.T @ J / noise_mm ** 2 + np.diag(1.0 / sig ** 2)
        b = J.T @ r / noise_mm ** 2 + th / sig ** 2
        step = -np.linalg.solve(A, b)
        th = th + step
        if float(np.max(np.abs(step))) < 1e-6:
            break
    r = zres(th)
    J = jac(th, r) if len(Q) else np.zeros((0, npar))
    info = J.T @ J / noise_mm ** 2
    cov = np.linalg.inv(info + np.diag(1.0 / sig ** 2))
    dof = len(Q) - npar
    scale = max(1.0, float(r @ r) / noise_mm ** 2 / dof) if dof > 0 else 1.0
    std = np.sqrt(np.diag(cov) * scale)
    share = np.sqrt(np.diag(cov)) / sig
    # conditioning of the data alone, with parameters scaled by their prior sigma
    S = info * np.outer(sig, sig)
    ev, evec = np.linalg.eigh(S)
    cond = float(ev[-1] / ev[0]) if ev[0] > 1e-12 else math.inf
    names = list(joints) + (["dz"] if fit_dz else [])
    v = evec[:, 0] / np.max(np.abs(evec[:, 0]))
    weakest = " ".join(f"{c:+.2f}*{nm}" for c, nm in zip(v, names) if abs(c) > 0.1)
    warns = []
    if len(Q) < npar + 3:
        warns.append(f"only {len(Q)} touches for {npar} parameters: record at least {npar + 5}")
    for nm, s in zip(names, share):
        if s > 0.5:
            warns.append(f"{nm} is mostly the prior (posterior std {s * 100:.0f} % of the prior's): "
                         "the touches do not pin it down")
    if fit_dz and len(set(tip.tolist())) < 2:
        warns.append("dz with every touch at one height trades off against the joint offsets: add touches on a "
                     "block of known height (DIR@H) or drop --fit-dz")
    if cond > 1e3:
        warns.append(f"ill-conditioned (condition number {cond:.0f}); weakest direction {weakest}: vary the "
                     "reach (150-350 mm) and the tool tilt (steep and shallow) between touches")
    off, dz = unpack(th)
    return FitResult(joints, fit_dz, off, dz, std, r, r_before, cond, share, weakest, warns)
