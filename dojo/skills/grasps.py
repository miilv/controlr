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
from dojo.skills.kinematics import TABLE_Z, TIP_BEYOND_TCP, X5Kin, gap_to_grip
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
    clear_w: float = 0.0          # open at least this wide (an irregular object's collider can exceed its mesh slice)
    narrow: bool = False          # open only 4 mm per side (a thin slice between neighbours)

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
        if self.narrow:
            return min(MAX_GAP, max(self.width, self.clear_w) + 0.008)
        return min(MAX_GAP, max(self.width, self.clear_w) + 2 * CLEAR + 0.006)


def narrowed(g: "Grasp") -> "Grasp":
    """The same grasp opened only 4 mm per side."""
    import dataclasses
    return dataclasses.replace(g, narrow=True)


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


def ring_center(obj: ObjectState, depth: float = 0.004, band: float = 0.003) -> np.ndarray:
    """Object-frame xy of a round container's axis: a circle fit to the mesh ring ``depth`` below its top, points
    off the circle (a mug's handle) dropped. The bbox centre when there is no mesh or ring."""
    c = np.asarray(obj.asset.center[:2], float)
    if obj.asset.mesh is None:
        return c
    V = obj.asset.mesh[0]
    top = float(V[:, 2].max())
    P = V[np.abs(V[:, 2] - (top - depth)) < band][:, :2]
    if len(P) < 12:
        return c
    keep = np.ones(len(P), bool)
    for _ in range(3):
        Q = P[keep]
        A = np.c_[2 * Q, np.ones(len(Q))]
        sol, *_ = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)
        ctr = sol[:2]
        r = np.linalg.norm(P - ctr, axis=1)
        med = float(np.median(r[keep]))
        new = np.abs(r - med) < max(0.004, 0.08 * med)
        if new.sum() < 12 or (new == keep).all():
            break
        keep = new
    return ctr


def wall_profile(obj: ObjectState, depth: float, band: float = 0.0015) -> tuple[float, float] | None:
    """(inner radius, outer radius) of a round container's wall ``depth`` below its rim (object frame,
    about its axis: ``ring_center``), from the mesh; points beyond the wall (a mug's handle) are dropped."""
    if obj.asset.mesh is None:
        return None
    V = obj.asset.mesh[0]
    c = ring_center(obj)
    top = float(V[:, 2].max())
    m = np.abs(V[:, 2] - (top - depth)) < band
    if m.sum() < 8:
        return None
    r = np.hypot(V[m, 0] - c[0], V[m, 1] - c[1])
    r = r[r < float(np.percentile(r, 50)) * 1.15 + 0.002]
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
    cr = ring_center(obj)
    c_w = R @ np.array([cr[0], cr[1], float(obj.asset.center[2])]) + obj.pos
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


def mesh_width_at(obj: ObjectState, T_grasp: np.ndarray, reach: float = 0.05, depths=(0.0, 0.006, 0.011),
                  max_normal_deg: float = 32.0) -> float | None:
    """The object's real width between the pads of a grasp (rays along the closing axis from both sides
    towards the TCP, at the TCP and a little deeper along the approach). None when either side misses
    (the grasp would close on a gap: between a pliers' handles) or when a contact face is not square to
    the closing axis (a rounded handle squeezed above its widest line pops out: twin, store_tools)."""
    if obj.asset.mesh is None:
        return float("nan")
    V, F = obj.asset.mesh
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    Rt, pt = obj.R.T, obj.pos
    y = T_grasp[:3, 1]
    y_l = Rt @ y
    cos_max = math.cos(math.radians(max_normal_deg))
    widths = []
    for dd in depths:
        p = T_grasp[:3, 3] + T_grasp[:3, 0] * dd
        hits = []
        for sgn in (1.0, -1.0):
            o_w = p + sgn * y * reach
            o_l, d_l = Rt @ (o_w - pt), Rt @ (-sgn * y)
            h = _ray_mesh(o_l, d_l, a, b, c)
            if h is None or h[0] > reach + 0.001:
                hits.append(None)
                continue
            fi = h[1]
            n = np.cross(b[fi] - a[fi], c[fi] - a[fi])
            n /= max(np.linalg.norm(n), 1e-12)
            hits.append(h[0] if abs(float(n @ y_l)) >= cos_max else None)
        if None in hits:
            continue
        widths.append(2 * reach - hits[0] - hits[1])
    if not widths:
        return None
    w = max(widths)
    return w if w > 0.002 else None


def mesh_com(asset) -> np.ndarray:
    """Centre of mass in the object frame: the volume centroid of a closed mesh (PhysX takes the mass
    properties from the colliders at uniform density), else the area-weighted surface centroid, else the
    bbox centre. A key: volume centroid at the bow / shaft junction (z -11 mm), surface centroid 5 mm lower."""
    mesh = getattr(asset, "mesh", None)
    if mesh is None:
        return np.asarray(asset.center, float)
    V, F = mesh
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    tot = float(vol.sum())
    ext = np.maximum(V.max(0) - V.min(0), 1e-6)
    if tot > 0.02 * float(np.prod(ext)):
        com = ((a + b + c) / 4.0 * vol[:, None]).sum(0) / tot
        if np.all(com >= V.min(0) - 1e-4) and np.all(com <= V.max(0) + 1e-4):
            return com
    ar = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return ((a + b + c) / 3 * ar[:, None]).sum(0) / max(float(ar.sum()), 1e-12)


def com_off(obj: ObjectState) -> float:
    """How far (m) the centre of mass sits from the bbox centre (0 without a mesh)."""
    if obj.asset.mesh is None:
        return 0.0
    return float(np.linalg.norm(mesh_com(obj.asset) - obj.asset.center))


def grasp_lever(obj: ObjectState, T_grasp: np.ndarray, centre=None) -> float:
    """Distance (m) from the object's centre of mass to the closing axis through the pads' contact centre
    (``centre``, world; default the TCP): the arm of the gravity torque that turns the object about the
    pads (a key held on its bow 9 mm off it swung 50 deg in the pads during a carry, twin)."""
    com_w = obj.R @ mesh_com(obj.asset) + obj.pos
    p = T_grasp[:3, 3] if centre is None else np.asarray(centre, float)
    d = com_w - p
    y = T_grasp[:3, 1]
    return float(np.linalg.norm(d - y * float(d @ y)))


def pad_contact(obj: ObjectState, T_grasp: np.ndarray, half_w: float = 0.008, x_range=(-0.008, 0.012),
                n: int = 5, tol: float = 0.0012, reach: float = 0.05, max_normal_deg: float = 32.0):
    """Where the pads touch the object once they close at ``T_grasp``: rays along the closing axis from both
    sides on an n x n grid over the pad (along the approach, across the finger); a side's contacts are its
    hits within ``tol`` of its outermost hit, on faces square to the closing axis. Returns (patch, centre):
    the smaller side's contact bbox as a fraction of the pad grid (flat faces, a key's bow from its sides,
    score high; a cylinder squeezed across its diameter, the bow from its rim, or an edge, low: the pads
    hold a line and the object turns about the closing axis) and the contacts' centre (world), or (0, None)."""
    if obj.asset.mesh is None:
        return 1.0, T_grasp[:3, 3].copy()
    V, F = obj.asset.mesh
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    Rt, pt = obj.R.T, obj.pos
    x, y, z = T_grasp[:3, 0], T_grasp[:3, 1], T_grasp[:3, 2]
    cos_max = math.cos(math.radians(max_normal_deg))
    dx, dz = (x_range[1] - x_range[0]) / (n - 1), 2 * half_w / (n - 1)
    side = {1.0: [], -1.0: []}
    for xi in np.linspace(x_range[0], x_range[1], n):
        for zi in np.linspace(-half_w, half_w, n):
            p = T_grasp[:3, 3] + x * xi + z * zi
            for sgn in (1.0, -1.0):
                h = _ray_mesh(Rt @ (p + sgn * y * reach - pt), Rt @ (-sgn * y), a, b, c)
                if h is None or h[0] > 2 * reach:
                    continue
                nrm = np.cross(b[h[1]] - a[h[1]], c[h[1]] - a[h[1]])
                nrm /= max(float(np.linalg.norm(nrm)), 1e-12)
                if abs(float(nrm @ (Rt @ y))) < cos_max:
                    continue
                side[sgn].append((reach - h[0], xi, zi))
    full = (x_range[1] - x_range[0] + dx) * (2 * half_w + dz)
    areas, cpts = [], []
    for sgn in (1.0, -1.0):
        h = side[sgn]
        if not h:
            return 0.0, None
        top = max(t for t, _, _ in h)
        pts = np.array([(xi, zi) for t, xi, zi in h if t >= top - tol])
        ext = pts.max(0) - pts.min(0)
        areas.append((ext[0] + dx) * (ext[1] + dz))
        cpts.append(pts)
    m = np.vstack(cpts).mean(0)
    pad_contact.depth = float(min(np.ptp(c[:, 0]) for c in cpts)) + dx      # contact extent along the approach
    return float(min(areas) / full), T_grasp[:3, 3] + x * m[0] + z * m[1]


def lift_hold(obj: ObjectState, T_grasp: np.ndarray) -> float:
    """Lower is better, for a pick that must then turn the object about the closing axis (an in-air handover's
    giver): the gravity lever (per cm) plus a penalty for contact under 8 mm deep along the approach (twin: a
    key's 4.8 mm shaft slid out at the lift or in the carry, 6/6; its 9.4 mm bow lifted, 13/13)."""
    patch, centre = pad_contact(obj, T_grasp)
    if centre is None:
        return 9.0
    depth = getattr(pad_contact, "depth", 0.0)
    # contact at the pad's edge holds nothing (twin: a key gripped where its shaft meets the bow, the bow
    # touching the pads' edges only, stayed on the table 6/6)
    edge = abs(float((np.asarray(centre) - T_grasp[:3, 3]) @ T_grasp[:3, 2]))
    return (grasp_lever(obj, T_grasp, centre) / 0.01 + max(0.0, 0.008 - depth) / 0.002 * 0.5
            + max(0.0, edge - 0.003) / 0.002 * 0.5)


def contact_patch(obj: ObjectState, T_grasp: np.ndarray, **kw) -> float:
    return pad_contact(obj, T_grasp, **kw)[0]


def hold_quality(obj: ObjectState, T_grasp: np.ndarray) -> float:
    """Lower is better: the gravity lever about the pads (per cm) plus the missing contact patch. Ranks
    grasps that must hold the object while the hand turns (in-air handovers, re-orienting carries)."""
    patch, centre = pad_contact(obj, T_grasp)
    if centre is None:
        return 9.0
    return grasp_lever(obj, T_grasp, centre) / 0.01 + 2.0 * (1.0 - min(1.0, patch / 0.5))


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
    from dojo.skills.motion import held_box
    irregular = held_box(obj, np.eye(4)).points is not None       # mesh-check every grasp of these
    com_w = None
    if irregular:      # grasp near the centre of mass (twin: pliers held 6 cm off it pivoted 30-40 mm in the pads)
        Vm, Fm = obj.asset.mesh
        a_, b_, c_ = Vm[Fm[:, 0]], Vm[Fm[:, 1]], Vm[Fm[:, 2]]
        ar = np.linalg.norm(np.cross(b_ - a_, c_ - a_), axis=1)
        com_w = R @ (((a_ + b_ + c_) / 3 * ar[:, None]).sum(0) / max(float(ar.sum()), 1e-12)) + obj.pos
    uneven = irregular or com_off(obj) > 0.008       # a wrench: box-like slices, its centre of mass 16 mm off
    out = []
    for k in range(3):                                   # closing axis
        if ext[k] > max_width or (axes_ok is not None and k not in axes_ok):
            continue
        for j in range(3):                               # approach axis
            if j == k:
                continue
            m = 3 - j - k
            for s, d in [(sg, dd) for sg in (1.0, -1.0) for dd in
                         ((min(float(ext[j]) / 2, depth),) + ((depth / 2,) if float(ext[j]) / 2 > depth else ()))]:
                # enter through the +s face, approach along -s; pads d inside it (also half as deep: a key's
                # bow held near its top keeps the fingertips over the lock)
                p_b = s * eye[j] * (float(ext[j]) / 2 - d)
                offs = [0.0]
                room = float(ext[m]) / 2 - edge
                if ends and room > 0.008:
                    offs += [room, -room]
                    if irregular and room > 0.016:      # quarter points too (a key's shaft below its bow,
                        offs += [room / 2, -room / 2]   # clear of the other hand's fingers on the bow)
                n_base = len(offs)
                if uneven and room > 0.03:
                    # stations every 6 mm within 4 cm of the centre of mass along the long axis (a hammer's or a
                    # wrench's handle next to its centre of mass, not its head; pliers at their joint)
                    c_m = float((mesh_com(obj.asset) - obj.asset.center)[m])
                    offs += [o_ for o_ in (c_m + 0.006 * i for i in range(-7, 8))
                             if abs(o_) <= room and min(abs(o_ - b_) for b_ in offs[:n_base]) > 0.002]
                for tilt in tilts:
                    t = math.radians(tilt)
                    appr_b = -s * (math.cos(t) * eye[j] + math.sin(t) * eye[m])
                    if float((R @ appr_b)[2]) > max_up:
                        continue
                    from_top = float((R @ appr_b)[2]) < -0.5 and abs(tilt) < 30
                    for oi, off in enumerate(offs):
                        if oi >= n_base and not from_top:
                            continue
                        pb = p_b + eye[m] * off
                        for flip in (1.0, -1.0):
                            Tg = T_c @ T_from(pb, _frame(appr_b, flip * eye[k]))
                            # fingertips >= 5 mm above the object's support: raise the TCP (the pads still
                            # cover the object from there up; thin parts, < 4 mm left, are skipped)
                            # 2.5 mm: a thin lying object (pliers, 24 mm) is then squeezed about its middle,
                            # not above it (twin: squeezed 5 mm high, the pliers popped out)
                            floor = max(obj.bottom_z, TABLE_Z) + 0.0025
                            # the fingers' lowest point from the same model the collision check uses (the
                            # analytic estimate sat 1 mm low on a slightly tilted hammer: every box grasp of a
                            # thin lying tool failed the 2 mm table margin, only slices were left)
                            g_open = gap_to_grip(min(MAX_GAP, float(ext[k]) + 2 * CLEAR + 0.006))
                            low = float(kin.finger_points(kin.flange_from_tcp(Tg), g_open)[:, 2].min())
                            need = floor - low
                            if need > 0:
                                if obj.top_z - floor < 0.004:
                                    continue
                                Tg[2, 3] += need
                            down = -float(Tg[2, 0])
                            w_real = mesh_width_at(obj, Tg) if irregular else float("nan")
                            if w_real is None or (w_real == w_real and w_real > max_width):
                                continue
                            out.append(Grasp("box", Tg, float(ext[k]) if w_real != w_real else float(w_real), pre_dist=0.06,
                                             clear_w=float(ext[k]) if irregular else 0.0,
                                             score=0.6 * (1.0 - down) + 0.1 * abs(off) / max(room, 1e-3)
                                             + 0.2 * float(ext[k]) / 0.07 + abs(tilt) / 100.0
                                             + (3.0 * float(np.linalg.norm(Tg[:2, 3] - com_w[:2])) / float(ext.max())
                                                if com_w is not None else 0.0),
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
