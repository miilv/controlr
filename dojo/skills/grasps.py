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
from dojo.skills.kinematics import TABLE_Z, TIP_BEYOND_TCP, X5Kin
from dojo.skills.world import ObjectState

MAX_GAP = 0.088               # pad gap fully open
CLEAR = 0.008                 # pad clearance each side when approaching (less for wide objects)
MAX_WIDTH = MAX_GAP - 0.004   # widest graspable section (2 mm per side)
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
               max_width: float = MAX_WIDTH, depth: float = PAD_DEPTH, support_z: float | None = None) -> list[Grasp]:
    """Top-down (and tilted) grasps closing on a horizontal object axis."""
    out = []
    ku, _ = up_axis(obj)
    ext = obj.asset.extents
    axes_ok = antipodal_axes(obj)
    R = obj.R
    c = obj.center_w
    top = obj.top_z
    bottom = obj.bottom_z if support_z is None else support_z
    # TCP at the object's mid height (the widest section of a lying cylinder is its axis: pads above
    # it squeeze the cylinder out, twin 2026-10-24), but fingertips >= 4 mm above the support and the
    # palm (61 mm above the TCP) clear of the top
    tcp_z = max(bottom + TIP_TABLE_CLEAR + TIP_BEYOND_TCP, min(float(c[2]), top - 0.005), top - 0.055)
    del depth
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
                    out.append(Grasp("top", T_from(pt, Rg), float(ext[k]),
                                     score=tilt / 40.0 + 0.05 * abs(s_along) + (0.5 if ext[k] > 0.072 else 0.0)
                                     + 0.2 * float(ext[k]) / 0.07,
                                     note=f"axis {k} tilt {tilt:.0f}"))
    return out


def side_grasps(obj: ObjectState, kin: X5Kin, arm: str, heights=(0.5, 0.65), pitches=(0.0, 20.0, 35.0),
                n_heading: int = 9, max_width: float = MAX_WIDTH, spread_deg: float = 90.0) -> list[Grasp]:
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
        for dh in np.linspace(-math.radians(spread_deg), math.radians(spread_deg), n_heading):
            head = h0 + dh
            for pitch in pitches:
                pr = math.radians(pitch)
                appr = np.array([math.cos(pr) * math.cos(head), math.cos(pr) * math.sin(head), -math.sin(pr)])
                closing = np.array([-math.sin(head), math.cos(head), 0.0])
                # TCP a little inside the object's axis so the pads wrap the widest section
                p = np.array([c[0], c[1], z])
                for flip in (1.0, -1.0):
                    out.append(Grasp("side", T_from(p, _frame(appr, flip * closing)), width,
                                     score=abs(dh) / math.radians(spread_deg) * 0.5 + pitch / 70.0 + abs(frac - 0.5),
                                     note=f"h {frac:.2f} head {math.degrees(head):.0f} pitch {pitch:.0f}"))
    return out


def rim_grasps(obj: ObjectState, kin: X5Kin, arm: str, n: int = 16, wall: float = 0.006, depth: float = 0.024,
               tilts=(0.0, 15.0, 25.0)) -> list[Grasp]:
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


def wall_profile(obj: ObjectState, depth: float, band: float = 0.0015) -> tuple[float, float] | None:
    """(inner radius, outer radius) of a round container's wall ``depth`` below its rim (object frame,
    about the vertical through the bbox centre), from the mesh."""
    if obj.asset.mesh is None:
        return None
    V = obj.asset.mesh[0]
    c = obj.asset.center
    top = float(V[:, 2].max())
    m = np.abs(V[:, 2] - (top - depth)) < band
    if m.sum() < 8:
        return None
    r = np.hypot(V[m, 0] - c[0], V[m, 1] - c[1])
    return float(r.min()), float(r.max())


def wall_grasps(obj: ObjectState, kin: X5Kin, arm: str, n: int = 16, depths=(0.012, 0.02, 0.028),
                max_width: float = 0.03) -> list[Grasp]:
    """Pinch a round container's wall below the rim: one pad inside, one outside, the closing axis
    along the wall normal (radial, tilted by the wall slope) and the approach running down along the
    wall, so both pads lie flat on the wall faces and the container cannot pivot about the tangent
    (a lip pinch tips the bowl 20-40 deg as soon as it lifts: twin traces 2026-10-24)."""
    if obj.asset.mesh is None:
        return []
    out = []
    R, top_w = obj.R, obj.top_z
    c_w = obj.center_w
    base = kin.base(arm)[:3, 3]
    h0 = math.atan2(base[1] - c_w[1], base[0] - c_w[0])
    for d in depths:
        p0, p1, p2 = wall_profile(obj, d), wall_profile(obj, d - 0.005), wall_profile(obj, d + 0.005)
        if p0 is None or p1 is None or p2 is None:
            continue
        r_in, r_out = p0
        t = r_out - r_in
        if t > max_width or t <= 0:
            continue
        r_mid = (r_in + r_out) / 2
        dr_dz = ((p1[0] + p1[1]) / 2 - (p2[0] + p2[1]) / 2) / 0.010      # radius grows going up
        beta = math.atan(max(0.0, dr_dz))
        for i in range(n):
            phi = h0 + 2 * math.pi * i / n
            u = np.array([math.cos(phi), math.sin(phi), 0.0])
            pt = c_w + u * r_mid
            pt[2] = top_w - d
            appr = -(math.sin(beta) * u + math.cos(beta) * np.array([0.0, 0.0, 1.0]))
            normal = math.cos(beta) * u - math.sin(beta) * np.array([0.0, 0.0, 1.0])
            ang = abs(((phi - h0) + math.pi) % (2 * math.pi) - math.pi)
            for flip in (1.0, -1.0):
                out.append(Grasp("wall", T_from(pt, _frame(appr, flip * normal)), float(t), pre_dist=0.06,
                                 score=ang / math.pi + 0.3 * abs(d - 0.02) / 0.01,
                                 note=f"phi {math.degrees(phi):.0f} depth {d*1000:.0f}mm wall {t*1000:.1f}mm "
                                      f"slope {math.degrees(beta):.0f}"))
    del R
    return out


def mesh_antipodal(obj: ObjectState, n_samples: int = 200, max_width: float = MAX_WIDTH,
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


def box_grasps(obj: ObjectState, kin: X5Kin, arm: str, max_width: float = MAX_WIDTH, depth: float = 0.016,
               edge: float = 0.012, max_up: float = 0.35, ends: bool = True,
               tilts: tuple = (0.0, 25.0, -25.0)) -> list[Grasp]:
    """Grasps on every pair of opposed bbox faces narrower than the open gripper (antipodal on the mesh
    when one is known), approached along each of the other two body axes from either side: the pads sit
    ``depth`` inside the entry face (or at the centre of thinner sections), centred or shifted towards
    either end of the third axis, the approach also turned by ``tilts`` (deg) about the closing axis (the
    X5 rarely points its gripper exactly horizontal low over the table; 20-40 deg down it reaches).
    Approaches pointing up by more than ``max_up`` (from below) are dropped.

    A grasp fixes the object's orientation in the hand, so a re-orienting place (lying -> standing,
    tip down, upright) is planned with its pick (``Skills.plan_transfer``); this set covers the
    grasps those places need (e.g. the end of a lying tile, held so that it stands when the wrist
    turns 90 deg)."""
    ext = obj.asset.extents
    axes_ok = antipodal_axes(obj)
    R = obj.R
    T_c = T_from(obj.center_w, R)
    eye = np.eye(3)
    out = []
    for k in range(3):                                   # closing axis
        if ext[k] > max_width or (axes_ok is not None and k not in axes_ok):
            continue
        for j in range(3):                               # approach axis
            if j == k:
                continue
            m = 3 - j - k
            for s in (1.0, -1.0):                        # enter through the +s face, approach along -s
                d = min(float(ext[j]) / 2, depth)
                p_b = s * eye[j] * (float(ext[j]) / 2 - d)
                offs = [0.0]
                room = float(ext[m]) / 2 - edge
                if ends and room > 0.008:
                    offs += [room, -room]
                for tilt in tilts:
                    t = math.radians(tilt)
                    appr_b = -s * (math.cos(t) * eye[j] + math.sin(t) * eye[m])
                    if float((R @ appr_b)[2]) > max_up:
                        continue
                    for off in offs:
                        pb = p_b + eye[m] * off
                        for flip in (1.0, -1.0):
                            Tg = T_c @ T_from(pb, _frame(appr_b, flip * eye[k]))
                            # fingertips >= 5 mm above the object's support: raise the TCP (the pads still
                            # cover the object from there up; thin parts, < 4 mm left, are skipped)
                            floor = max(obj.bottom_z, TABLE_Z) + TIP_TABLE_CLEAR + 0.001
                            low = (Tg[2, 3] + min(0.0, float(Tg[2, 0])) * TIP_BEYOND_TCP
                                   - 0.009 * abs(float(Tg[2, 2])) - 0.004 * abs(float(Tg[2, 1])))
                            need = floor - low
                            if need > 0:
                                if obj.top_z - floor < 0.004:
                                    continue
                                Tg[2, 3] += need
                            down = -float(Tg[2, 0])
                            out.append(Grasp("box", Tg, float(ext[k]), pre_dist=0.06,
                                             score=0.6 * (1.0 - down) + 0.1 * abs(off) / max(room, 1e-3)
                                             + 0.2 * float(ext[k]) / 0.07 + abs(tilt) / 100.0,
                                             note=f"close {k} approach {'+' if s > 0 else '-'}{j} tilt {tilt:+.0f} "
                                                  f"off {off*1000:+.0f}mm"))
    return out


def slice_grasps(obj: ObjectState, kin: X5Kin, arm: str, max_width: float = MAX_WIDTH - 0.004, n_dir: int = 8,
                 band: float = 0.008, levels: int = 4, tilts=(0.0, 25.0, 40.0)) -> list[Grasp]:
    """Top-down grasps on horizontal slices of an irregular mesh (a figurine whose bbox is wider than the
    gripper but whose body is not): for each height and closing direction, the slice's width along that
    direction must fit, and the open fingers' path down to it must be free of the object's own points
    (no wider part above it in the finger corridor)."""
    if obj.asset.mesh is None:
        return []
    V = obj.asset.mesh[0] @ obj.R.T + obj.pos                 # world points
    zmin, zmax = float(V[:, 2].min()), float(V[:, 2].max())
    floor = max(zmin, TABLE_Z) + TIP_TABLE_CLEAR + 0.001
    out = []
    for h in np.linspace(zmax - 0.012, max(floor + TIP_BEYOND_TCP, zmin + 0.25 * (zmax - zmin)), levels):
        slab = V[np.abs(V[:, 2] - h) < band]
        if len(slab) < 6:
            continue
        cxy = (slab[:, :2].min(0) + slab[:, :2].max(0)) / 2
        for k in range(n_dir):
            th = math.pi * k / n_dir
            u = np.array([math.cos(th), math.sin(th)])
            v = np.array([-u[1], u[0]])
            pu, pv = (slab[:, :2] - cxy) @ u, (slab[:, :2] - cxy) @ v
            w = float(pu.max() - pu.min())
            if w > max_width or w < 0.006:
                continue
            mid_u = float(pu.max() + pu.min()) / 2
            mid_v = float(pv.max() + pv.min()) / 2
            c = cxy + u * mid_u + v * mid_v
            tcp_z = max(h, floor + TIP_BEYOND_TCP)
            # the open fingers' corridors (outside the slice along u, +-12 mm along v) down to the pads
            gap = min(MAX_GAP, w + 2 * CLEAR + 0.006)
            P = V[V[:, 2] > tcp_z - TIP_BEYOND_TCP]
            qu, qv = (P[:, :2] - c) @ u, (P[:, :2] - c) @ v
            in_fing = (np.abs(qu) > gap / 2 - 0.001) & (np.abs(qu) < gap / 2 + 0.014) & (np.abs(qv) < 0.012)
            if in_fing.any():
                continue
            closing = np.array([u[0], u[1], 0.0])
            perp = np.array([-u[1], u[0], 0.0])
            base = kin.base(arm)[:3, 3]
            sgn = 1.0 if float(perp[:2] @ (c - base[:2])) >= 0 else -1.0
            for tilt in tilts:                         # tilted away from the arm: reaches further / higher
                t = math.radians(tilt)
                appr = np.array([0.0, 0.0, -math.cos(t)]) + sgn * perp * math.sin(t)
                Rg = _frame(appr, closing)
                out.append(Grasp("slice", T_from([c[0], c[1], tcp_z + 0.010 * math.sin(t)], Rg), w,
                                 score=0.3 * (zmax - h) / max(zmax - zmin, 1e-3) + 0.3 * w / 0.08 + tilt / 100.0,
                                 note=f"slice z {h - zmin:.3f} dir {math.degrees(th):.0f} w {w*1000:.0f}mm "
                                      f"tilt {tilt:.0f}"))
    return out


def candidates(obj: ObjectState, kin: X5Kin, arm: str, method: str = "auto", **kw) -> list[Grasp]:
    if method == "top":
        g = top_grasps(obj, kin, arm, **kw)
    elif method == "side":
        g = side_grasps(obj, kin, arm, **kw)
    elif method == "rim":
        g = rim_grasps(obj, kin, arm, **kw)
    elif method == "wall":
        g = wall_grasps(obj, kin, arm, **kw) or rim_grasps(obj, kin, arm)
    elif method == "antipodal":
        g = antipodal_grasps(obj, kin, arm, **kw)
    elif method == "box":
        g = box_grasps(obj, kin, arm, **kw)
    else:
        cat = obj.category
        if cat == "bowl":
            g = wall_grasps(obj, kin, arm) or rim_grasps(obj, kin, arm)
        elif cat == "bottle":
            ku, _ = up_axis(obj)
            long_k = int(np.argmax(obj.asset.extents))
            if ku == long_k:      # standing: the bbox width is not the neck's, so side grasps only
                g = side_grasps(obj, kin, arm)
            else:
                g = top_grasps(obj, kin, arm)
        else:
            g = top_grasps(obj, kin, arm)
            if not g:
                g = slice_grasps(obj, kin, arm) or antipodal_grasps(obj, kin, arm)
    return sorted(g, key=lambda x: x.score)
