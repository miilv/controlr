"""Grasp candidates per object, by method (COAP's catalogue: top / side / rim / antipodal).

A ``Grasp`` is a TCP pose in the world (flange +x = approach, +y = closing axis) plus the width
between the pads and the pre-grasp retreat. Candidates are generated from the asset geometry
(bbox in the object frame; the mesh from perception's catalogue when present) and the object's
current pose; ``skills.pick`` filters them by IK and collision and keeps the cheapest in env steps.

Methods:
* ``top``: approach straight down (or tilted up to 50 deg towards the arm for far objects), the
  closing axis along a horizontal object axis whose faces are antipodal (checked on the mesh:
  the triangular prism of build_tower only closes on its end faces) and narrower than the open
  gripper;
* ``side``: horizontal (or slightly downward) approach at mid-height of an upright object, closing
  horizontally (bottles standing up);
* ``rim``: one pad inside, one outside a bowl wall, approach down, closing radially;
* ``antipodal``: sampled on the mesh (pairs of opposed faces), any approach orthogonal to the
  closing axis that comes from above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dojo.skills.geometry import T_from
from dojo.skills.kinematics import TIP_BEYOND_TCP, X5Kin
from dojo.skills.world import ObjectState

MAX_GAP = 0.088               # pad gap fully open
CLEAR = 0.008                 # min pad clearance each side when approaching
PAD_DEPTH = 0.030             # how much of the object the pads cover from the top (top grasp)
TIP_TABLE_CLEAR = 0.004       # fingertip above the support surface


@dataclass
class Grasp:
    method: str
    T: np.ndarray                 # TCP pose (world)
    width: float                  # object width between the pads (m)
    pre_dist: float = 0.07        # pre-grasp retreat along -approach
    score: float = 0.0            # lower is better (preference before the step cost)
    note: str = ""
    lift_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    @property
    def approach(self) -> np.ndarray:
        return self.T[:3, 0]

    def pre(self, dist: float | None = None) -> np.ndarray:
        d = self.pre_dist if dist is None else dist
        P = self.T.copy()
        P[:3, 3] = P[:3, 3] - self.approach * d
        if self.approach[2] > -0.5:          # side-ish approach: also come from a bit higher
            P[2, 3] += 0.02
        return P

    def open_gap(self) -> float:
        return min(MAX_GAP, self.width + 2 * CLEAR + 0.006)


def _frame(approach, closing) -> np.ndarray:
    x = np.asarray(approach, float)
    x = x / np.linalg.norm(x)
    y = np.asarray(closing, float)
    y = y - x * float(x @ y)
    y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def antipodal_axes(obj: ObjectState, min_area_frac: float = 0.05, cos_tol: float = 0.966) -> list[int] | None:
    """Local axes (0/1/2) whose ± faces both exist on the mesh (normals within ~15 deg).
    None when no mesh is known (treat the object as its bbox)."""
    if obj.asset.mesh is None:
        return None
    V, F = obj.asset.mesh
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1) / 2
    nn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    tot = area.sum()
    out = []
    for k in range(3):
        pos = area[nn[:, k] > cos_tol].sum()
        neg = area[nn[:, k] < -cos_tol].sum()
        if pos > min_area_frac * tot / 6 and neg > min_area_frac * tot / 6:
            out.append(k)
    return out


def up_axis(obj: ObjectState) -> tuple[int, float]:
    """Local axis closest to world +z (index, sign)."""
    z = obj.R.T @ np.array([0.0, 0.0, 1.0])
    k = int(np.argmax(np.abs(z)))
    return k, float(np.sign(z[k]))


def top_grasps(obj: ObjectState, kin: X5Kin, arm: str, tilts=(0.0, 25.0, 40.0), along: tuple = (0.0,),
               max_width: float = MAX_GAP - 2 * CLEAR, depth: float = PAD_DEPTH, support_z: float | None = None) -> list[Grasp]:
    """Top-down (and tilted) grasps closing on a horizontal object axis."""
    out = []
    ku, _ = up_axis(obj)
    ext = obj.asset.extents
    axes_ok = antipodal_axes(obj)
    R = obj.R
    c = obj.center_w
    top = obj.top_z
    bottom = obj.bottom_z if support_z is None else support_z
    tip_z = max(bottom + TIP_TABLE_CLEAR, top - depth)
    tcp_z = tip_z + TIP_BEYOND_TCP
    base = kin.base(arm)[:3, 3]
    for k in range(3):
        if k == ku or ext[k] > max_width:
            continue
        if axes_ok is not None and k not in axes_ok:
            continue
        closing = R[:, k].copy()
        closing[2] = 0.0
        if np.linalg.norm(closing) < 0.5:
            continue
        closing /= np.linalg.norm(closing)
        perp = np.cross([0.0, 0.0, 1.0], closing)                # horizontal, along the other object axis
        k_other = 3 - k - ku
        for s_along in along:
            p = c + perp * s_along * ext[k_other] / 2
            p = np.array([p[0], p[1], tcp_z])
            # tilt towards the arm: approach gets a horizontal component pointing away from the base
            away = p[:2] - base[:2]
            sgn = 1.0 if float(np.dot(away, perp[:2])) >= 0 else -1.0
            for tilt in tilts:
                t = math.radians(tilt)
                appr = np.array([0, 0, -math.cos(t)]) + sgn * perp * math.sin(t)
                for flip in (1.0, -1.0):
                    Rg = _frame(appr, flip * closing)
                    pt = p.copy()
                    if tilt > 0:
                        # keep the fingertips' lowest point at tip_z: raise by the pad's tilt sweep
                        pt[2] += 0.010 * math.sin(t)
                    out.append(Grasp("top", T_from(pt, Rg), float(ext[k]), score=tilt / 40.0 + 0.05 * abs(s_along),
                                     note=f"axis {k} tilt {tilt:.0f}"))
    return out


def side_grasps(obj: ObjectState, kin: X5Kin, arm: str, heights=(0.5, 0.65), pitches=(0.0, 20.0, 35.0),
                n_heading: int = 7, max_width: float = MAX_GAP - 2 * CLEAR) -> list[Grasp]:
    """Horizontal-ish approach at a fraction of the object's height (upright objects)."""
    ku, _ = up_axis(obj)
    ext = obj.asset.extents
    widths = [ext[k] for k in range(3) if k != ku]
    width = float(min(widths))
    if width > max_width:
        return []
    out = []
    base = kin.base(arm)[:3, 3]
    c = obj.center_w
    lo, hi = obj.bottom_z, obj.top_z
    h0 = math.atan2(c[1] - base[1], c[0] - base[0])
    for frac in heights:
        z = lo + frac * (hi - lo)
        for dh in np.linspace(-math.radians(60), math.radians(60), n_heading):
            head = h0 + dh
            for pitch in pitches:
                pr = math.radians(pitch)
                appr = np.array([math.cos(pr) * math.cos(head), math.cos(pr) * math.sin(head), -math.sin(pr)])
                closing = np.array([-math.sin(head), math.cos(head), 0.0])
                # TCP a little inside the object's axis so the pads wrap the widest section
                p = np.array([c[0], c[1], z])
                for flip in (1.0, -1.0):
                    out.append(Grasp("side", T_from(p, _frame(appr, flip * closing)), width,
                                     score=abs(dh) / math.radians(60) * 0.5 + pitch / 70.0 + abs(frac - 0.5),
                                     note=f"h {frac:.2f} head {math.degrees(head):.0f} pitch {pitch:.0f}"))
    return out


def rim_grasps(obj: ObjectState, kin: X5Kin, arm: str, n: int = 16, wall: float = 0.006, depth: float = 0.018,
               tilts=(0.0, 25.0)) -> list[Grasp]:
    """Bowl rim: pads straddle the wall at the rim, closing radially, approach from above."""
    ext = obj.asset.extents
    r_out = float(min(ext[0], ext[1])) / 2
    r = r_out - wall / 2
    c = obj.center_w
    top = obj.top_z
    tcp_z = top - depth + TIP_BEYOND_TCP
    base = kin.base(arm)[:3, 3]
    h0 = math.atan2(base[1] - c[1], base[0] - c[0])         # rim side facing the arm first
    out = []
    for i in range(n):
        phi = h0 + 2 * math.pi * i / n
        radial = np.array([math.cos(phi), math.sin(phi), 0.0])
        p = c + radial * r
        p[2] = tcp_z
        tang = np.array([-math.sin(phi), math.cos(phi), 0.0])
        for tilt in tilts:
            t = math.radians(tilt)
            appr = np.array([0, 0, -math.cos(t)]) - radial * math.sin(t)     # tilt outward-in (from the rim side)
            for flip in (1.0, -1.0):
                Rg = _frame(appr, flip * radial)
                ang = abs(((phi - h0) + math.pi) % (2 * math.pi) - math.pi)
                out.append(Grasp("rim", T_from(p, Rg), wall, pre_dist=0.06,
                                 score=ang / math.pi + tilt / 50.0, note=f"phi {math.degrees(phi):.0f} tilt {tilt:.0f}"))
        del tang
    return out


def mesh_antipodal(obj: ObjectState, n_samples: int = 200, max_width: float = MAX_GAP - 2 * CLEAR,
                   rng_seed: int = 0) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Antipodal point pairs on the mesh (object frame): (midpoint, closing axis, width)."""
    if obj.asset.mesh is None:
        return []
    V, F = obj.asset.mesh
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1)
    nn = n / np.maximum(area[:, None], 1e-12)
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(F), size=min(n_samples, len(F) * 4), p=area / area.sum())
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])
    out = []
    for p, f in zip(pts, idx):
        d = -nn[f]
        hit = _ray_mesh(p + d * 1e-4, d, a, b, c)
        if hit is None:
            continue
        t, fh = hit
        if t > max_width or t < 0.004:
            continue
        if float(nn[fh] @ nn[f]) > -0.94:          # opposed normals within ~20 deg
            continue
        out.append((p + d * t / 2, d, float(t)))
    return out


def _ray_mesh(o, d, a, b, c) -> tuple[float, int] | None:
    """Nearest positive hit of a ray with triangles (Moller-Trumbore, vectorised)."""
    e1, e2 = b - a, c - a
    pv = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    tv = o - a
    u = np.einsum("ij,ij->i", tv, pv) * inv
    qv = np.cross(tv, e1)
    v = (qv @ d) * inv
    t = np.einsum("ij,ij->i", e2, qv) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
    if not hit.any():
        return None
    i = int(np.argmin(np.where(hit, t, np.inf)))
    return float(t[i]), i


def antipodal_grasps(obj: ObjectState, kin: X5Kin, arm: str, n_samples: int = 200, n_approach: int = 6) -> list[Grasp]:
    out = []
    R, P = obj.R, obj.pos
    for mid, axis, w in mesh_antipodal(obj, n_samples):
        mw = R @ mid + P
        aw = R @ axis
        if abs(aw[2]) > 0.5:                     # closing axis must be roughly horizontal
            continue
        # approaches orthogonal to the closing axis, from above
        ref = np.cross(aw, [0, 0, 1.0])
        ref /= np.linalg.norm(ref)
        for k in range(n_approach):
            th = -math.pi / 2 + math.pi * k / max(1, n_approach - 1)
            appr = math.cos(th) * np.array([0, 0, -1.0]) + math.sin(th) * ref
            appr -= aw * float(aw @ appr)
            if np.linalg.norm(appr) < 1e-6 or appr[2] > 0.2:
                continue
            Rg = _frame(appr, aw)
            out.append(Grasp("antipodal", T_from(mw, Rg), w, score=abs(th) / math.pi + 0.3, note=f"w {w*1000:.0f}mm"))
    return out


def candidates(obj: ObjectState, kin: X5Kin, arm: str, method: str = "auto", **kw) -> list[Grasp]:
    if method == "top":
        g = top_grasps(obj, kin, arm, **kw)
    elif method == "side":
        g = side_grasps(obj, kin, arm, **kw)
    elif method == "rim":
        g = rim_grasps(obj, kin, arm, **kw)
    elif method == "antipodal":
        g = antipodal_grasps(obj, kin, arm, **kw)
    else:
        cat = obj.category
        if cat == "bowl":
            g = rim_grasps(obj, kin, arm)
        elif cat == "bottle":
            ku, _ = up_axis(obj)
            long_k = int(np.argmax(obj.asset.extents))
            g = side_grasps(obj, kin, arm) + top_grasps(obj, kin, arm) if ku == long_k else top_grasps(obj, kin, arm)
        else:
            g = top_grasps(obj, kin, arm)
            if not g:
                g = antipodal_grasps(obj, kin, arm)
    return sorted(g, key=lambda x: x.score)
