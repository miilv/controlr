"""Peg-in-hole geometry from the asset meshes: where and how an object goes INTO another one.

``fit_insertion(peg, hole)`` answers, from the two meshes only (no task knowledge):

* which end of the peg goes in (its long axis, the end with the smaller cross-section first);
* the receptacle's opening: a depth map from above (first mesh hit of a vertical ray per grid cell,
  receptacle frame) -> the cells deeper than the top surface;
* the peg's footprint: its surface samples within the hole's depth of the leading end, projected
  along the insertion axis;
* the yaw and xy offset that put the footprint inside the opening with the largest clearance (a
  brute-force search over yaw, then over translations on the grid's distance map);
* the floor under the footprint (how deep it goes).

Used by ``library.insert_into`` (insert_key's keyhole, a charger's prongs, pens in a holder): the
result is an object pose in the world, ready for ``Skills.plan_transfer`` (the grasp that reaches it)
and a straight descent along the insertion axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dojo.skills.geometry import T_from, rot_z


@dataclass
class HoleFit:
    T_obj: np.ndarray            # peg pose fully inserted (world), leading end on the floor
    yaw: float                   # rad, about the receptacle's vertical, of the peg's footprint frame
    clearance: float             # m, smallest distance from the footprint to the opening's wall
    floor_z: float               # world z of the floor the peg's end rests on
    top_z: float                 # world z of the receptacle's top surface
    depth: float                 # m, top - floor
    R_peg0: np.ndarray           # peg orientation (body -> world) with the insertion axis down, yaw 0
    note: str = ""


def _ray_down_grid(V, F, xs, ys, z0) -> np.ndarray:
    """First hit z of vertical rays (from z0 down) on a grid, -inf where nothing is hit."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    out = np.full((len(ys), len(xs)), -np.inf)
    # per triangle: barycentric test in xy for every grid point inside its xy bbox
    X, Y = np.meshgrid(xs, ys)
    for ta, tb, tc in zip(a, b, c):
        x0, x1 = min(ta[0], tb[0], tc[0]), max(ta[0], tb[0], tc[0])
        y0, y1 = min(ta[1], tb[1], tc[1]), max(ta[1], tb[1], tc[1])
        ix = np.where((xs >= x0) & (xs <= x1))[0]
        iy = np.where((ys >= y0) & (ys <= y1))[0]
        if not len(ix) or not len(iy):
            continue
        px, py = X[np.ix_(iy, ix)], Y[np.ix_(iy, ix)]
        d = (tb[1] - tc[1]) * (ta[0] - tc[0]) + (tc[0] - tb[0]) * (ta[1] - tc[1])
        if abs(d) < 1e-14:
            continue                                   # vertical triangle: seen edge-on from above
        l1 = ((tb[1] - tc[1]) * (px - tc[0]) + (tc[0] - tb[0]) * (py - tc[1])) / d
        l2 = ((tc[1] - ta[1]) * (px - tc[0]) + (ta[0] - tc[0]) * (py - tc[1])) / d
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        z = l1 * ta[2] + l2 * tb[2] + l3 * tc[2]
        z = np.where(inside & (z <= z0), z, -np.inf)
        sub = out[np.ix_(iy, ix)]
        out[np.ix_(iy, ix)] = np.maximum(sub, z)
    return out


def _surface_samples(V, F, step: float = 0.0006) -> np.ndarray:
    """Points on every triangle, about ``step`` apart (plus the vertices)."""
    pts = [V]
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    for ta, tb, tc in zip(a, b, c):
        L = max(np.linalg.norm(tb - ta), np.linalg.norm(tc - ta), np.linalg.norm(tc - tb))
        n = int(min(40, max(1, math.ceil(L / step))))
        u, v = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
        m = (u + v) <= 1.0
        u, v = u[m], v[m]
        pts.append(ta + np.outer(u, tb - ta) + np.outer(v, tc - ta))
    return np.vstack(pts)


def insertion_axis(V) -> tuple[int, float]:
    """(body axis, sign) pointing from the peg's wide end to its narrow (leading) end: the long axis,
    the end whose last 15 % of length has the smaller xy spread."""
    ext = V.max(0) - V.min(0)
    k = int(np.argmax(ext))
    lo, hi = V[:, k].min(), V[:, k].max()
    band = 0.15 * (hi - lo)
    others = [i for i in range(3) if i != k]

    def spread(m):
        P = V[m][:, others]
        return float(np.prod(P.max(0) - P.min(0) + 1e-4)) if len(P) else 1e9
    s_lo = spread(V[:, k] <= lo + band)
    s_hi = spread(V[:, k] >= hi - band)
    return k, (-1.0 if s_lo <= s_hi else 1.0)


def R_axis_down(k: int, sign: float) -> np.ndarray:
    """A rotation (body -> world) taking body axis ``sign * e_k`` to world -z."""
    a = np.zeros(3)
    a[k] = sign
    t = -a                                               # this body vector must point up
    up = np.array([0.0, 0.0, 1.0])
    # build R with R @ t = up: pick any orthonormal completion
    ref = np.eye(3)[(k + 1) % 3]
    x_b = ref - t * float(ref @ t)
    x_b /= np.linalg.norm(x_b)
    y_b = np.cross(t, x_b)
    B = np.stack([x_b, y_b, t], axis=1)                  # body frame columns
    W = np.eye(3)                                        # -> world x, y, up
    return W @ B.T


def _dist_inside(mask: np.ndarray, res: float) -> np.ndarray:
    """Distance (m) from every True cell to the nearest False cell (brute force on the boundary)."""
    solid = ~mask
    # boundary solid cells: solid cells next to a hole cell
    pad = np.pad(mask, 1, constant_values=False)
    near = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    bnd = np.argwhere(solid & near)
    if not len(bnd):
        bnd = np.argwhere(solid)
    holes = np.argwhere(mask)
    out = np.zeros(mask.shape)
    if not len(holes) or not len(bnd):
        return out
    for i in range(0, len(holes), 4096):
        h = holes[i:i + 4096]
        d = np.sqrt(((h[:, None, :] - bnd[None, :, :]) ** 2).sum(-1)).min(1)
        out[h[:, 0], h[:, 1]] = d * res
    return out


def fit_insertion(peg, hole, res: float = 0.0004, yaw_step_deg: float = 3.0, margin_top: float = 0.003,
                  max_open: float = 0.06, others=(), others_pad: float = 0.002, pack: bool = False,
                  min_clear: float = 0.002, others_band: float = 0.03) -> HoleFit | None:
    """Best insertion pose of ``peg`` (ObjectState with a mesh) into ``hole`` (ObjectState with a mesh,
    standing upright). ``others``: objects already in the opening (their mesh points inside it, padded
    by ``others_pad`` and within ``others_band`` of the top, are taken out of the free area: leaning objects
    share the floor, the rim is what fills up). ``pack``: of the poses with at least ``min_clear``, the one
    farthest from the opening's centre (against the wall, leaving room for the next object) instead of
    the one with the most clearance. None when the meshes are missing or nothing fits."""
    if peg.asset.mesh is None or hole.asset.mesh is None:
        return None
    Vh, Fh = hole.asset.mesh
    Vp, Fp = peg.asset.mesh
    top = float(Vh[:, 2].max())
    # depth map of the receptacle around its centre (its own frame)
    cx, cy = (Vh[:, 0].min() + Vh[:, 0].max()) / 2, (Vh[:, 1].min() + Vh[:, 1].max()) / 2
    half = min(max_open, float(max(Vh[:, 0].max() - Vh[:, 0].min(), Vh[:, 1].max() - Vh[:, 1].min())) / 2)
    xs = np.arange(cx - half, cx + half + 1e-9, res)
    ys = np.arange(cy - half, cy + half + 1e-9, res)
    D = _ray_down_grid(Vh, Fh, xs, ys, top + 0.01)
    opening = np.isfinite(D) & (D < top - margin_top)        # inside the receptacle's outline, below its top
    if opening.sum() < 4:
        return None
    floor = float(np.median(D[opening]))
    T_hw = np.eye(4)
    T_hw[:3, :3], T_hw[:3, 3] = hole.R.T, -hole.R.T @ hole.pos        # world -> receptacle frame
    for oth in others:
        if oth.asset.mesh is None:
            continue
        W = oth.asset.mesh[0] @ oth.R.T + oth.pos
        L = W @ T_hw[:3, :3].T + T_hw[:3, 3]
        L = L[(L[:, 2] > top - others_band) & (L[:, 2] < top + others_band)]
        if not len(L):
            continue
        pad = int(math.ceil(others_pad / res))
        ij = np.round((L[:, :2] - [xs[0], ys[0]]) / res).astype(int)
        for dx in range(-pad, pad + 1):
            for dy in range(-pad, pad + 1):
                ii, jj = ij[:, 1] + dy, ij[:, 0] + dx
                ok = (ii >= 0) & (ii < len(ys)) & (jj >= 0) & (jj < len(xs))
                opening[ii[ok], jj[ok]] = False
    if opening.sum() < 4:
        return None
    depth = top - floor
    # the peg with its leading end down, the slab that enters
    k, sgn = insertion_axis(Vp)
    R0 = R_axis_down(k, sgn)
    P = _surface_samples(Vp, Fp) @ R0.T
    tip = float(P[:, 2].min())
    slab = P[P[:, 2] <= tip + max(0.004, depth - 0.001)]
    dist = _dist_inside(opening, res)
    hole_idx = np.argwhere(opening)
    ctr = hole_idx.mean(0)

    def score(yaw_deg, cand):
        Rz = rot_z(math.radians(yaw_deg))
        Q = slab @ Rz.T
        cells = np.unique(np.round(Q[:, :2] / res).astype(int), axis=0)       # (dx, dy) cells about the root
        tx, ty = cand[:, 1], cand[:, 0]
        out = None
        for i in range(0, len(cand), 256):
            ix = tx[i:i + 256, None] + cells[None, :, 0]
            iy = ty[i:i + 256, None] + cells[None, :, 1]
            ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
            v = np.where(ok, dist[np.clip(iy, 0, len(ys) - 1), np.clip(ix, 0, len(xs) - 1)], 0.0)
            m = v.min(1)
            if pack:
                far = np.hypot(ty[i:i + 256] - ctr[0], tx[i:i + 256] - ctr[1]) * res
                key = np.where(m >= min_clear, far + 1.0, m)          # feasible ones first, then farthest out
            else:
                key = m
            j = int(np.argmax(key))
            if out is None or key[j] > out[5]:
                out = (float(m[j]), yaw_deg, int(tx[i + j]), int(ty[i + j]), cells, float(key[j]))
        return out

    # coarse yaw x coarse translations, then a fine pass around the best
    best = None
    for yaw_deg in np.arange(0.0, 360.0, 2 * yaw_step_deg):
        r = score(yaw_deg, hole_idx[::4])
        if r and (best is None or r[5] > best[5]):
            best = r
    if best is not None:
        y0 = best[1]
        for yaw_deg in np.arange(y0 - 2 * yaw_step_deg, y0 + 2 * yaw_step_deg + 1e-9, yaw_step_deg / 3):
            r = score(yaw_deg % 360.0, hole_idx)
            if r and r[5] > best[5]:
                best = r
    if best is None or best[0] <= 0.0:
        return None
    clear, yaw_deg, ix, iy, cells, _ = best
    root_local = np.array([xs[ix], ys[iy]])
    # floor under the footprint: the highest floor cell it covers (the leading end lands there)
    fx = np.clip(ix + cells[:, 0], 0, len(xs) - 1)
    fy = np.clip(iy + cells[:, 1], 0, len(ys) - 1)
    fl = D[fy, fx]
    fl = fl[np.isfinite(fl)]
    floor_c = float(fl.max()) if len(fl) else floor
    # world pose: receptacle frame -> world; peg root at root_local, its leading end on the floor
    Rh = hole.R
    Ry = rot_z(math.radians(yaw_deg))
    R_peg_local = Ry @ R0
    p_local = np.array([root_local[0], root_local[1], floor_c - tip])
    R_w = Rh @ R_peg_local
    p_w = hole.pos + Rh @ p_local
    return HoleFit(T_from(p_w, R_w), math.radians(yaw_deg), clear, float(hole.pos[2] + Rh[2] @ [0, 0, floor_c]),
                   float(hole.pos[2] + top), float(depth), R0,
                   note=f"axis {'+' if sgn > 0 else '-'}{k} yaw {yaw_deg:.0f} clearance {clear*1000:.2f} mm "
                        f"depth {depth*1000:.0f} mm root {np.round(root_local*1000, 1).tolist()} mm")


def refine_insertion(peg, hole, T_obj, half: float = 0.003, res: float = 0.0001, yaws_deg=(-3, -2, -1, 0, 1, 2, 3),
                     lead: float = 0.003, margin_top: float = 0.003, press_floor: bool = True):
    """Fine local fit of an insertion pose that is already close (an asset's insert frame on a receptacle's
    outlet frame): the peg's leading end (its surface within ``lead`` of the lowest point, at pose
    ``T_obj``) against the receptacle's holes (depth map at ``res``, cells more than ``margin_top`` below the
    local top), over xy offsets within ``half`` and small yaws about the vertical through the leading end's
    centre. Best = the most footprint points over holes, then the largest wall clearance, then the smallest
    shift (a charger's 0.8 mm blades into 1.0 mm slots: the outlet frames sit 0.5-1 mm off the slot pairs;
    twin, the plugs that went in sat 1 mm off the frame). ``press_floor``: the pose is also lowered until the
    leading end meets the floor under the footprint (the frames put the blade tips at the outlet surface; the
    checker wants them >= 15 mm below the strip's top). Returns (T_obj, info dict) or None."""
    if peg.asset.mesh is None or hole.asset.mesh is None:
        return None
    T_obj = np.asarray(T_obj, float)
    Vh, Fh = hole.asset.mesh
    Vp, Fp = peg.asset.mesh
    T_hw = np.eye(4)
    T_hw[:3, :3], T_hw[:3, 3] = hole.R.T, -hole.R.T @ hole.pos            # world -> receptacle frame
    T_loc = T_hw @ T_obj                                                   # peg pose in the receptacle frame
    P = _surface_samples(Vp, Fp, step=0.0003) @ T_loc[:3, :3].T + T_loc[:3, 3]
    tip = float(P[:, 2].min())
    foot = P[P[:, 2] <= tip + lead]
    c = foot[:, :2].mean(0)
    ext = float(np.abs(foot[:, :2] - c).max()) + half + 0.002
    xs = np.arange(c[0] - ext, c[0] + ext + 1e-9, res)
    ys = np.arange(c[1] - ext, c[1] + ext + 1e-9, res)
    near = np.all(np.abs(Vh[Fh].mean(1)[:, :2] - c) < ext + 0.01, axis=1)
    D = _ray_down_grid(Vh, Fh[near], xs, ys, float(Vh[:, 2].max()) + 0.01)
    fin = np.isfinite(D)
    if not fin.any():
        return None
    top = float(D[fin].max())
    hole_m = (~fin) | (D < top - margin_top)
    dist = _dist_inside(hole_m, res)
    best = None
    shifts = np.arange(-half, half + 1e-9, res)
    for yd in yaws_deg:
        Rz = rot_z(math.radians(yd))
        Q = (foot[:, :2] - c) @ Rz[:2, :2].T + c
        for dx in shifts:
            ix = np.round((Q[:, 0] + dx - xs[0]) / res).astype(int)
            okx = (ix >= 0) & (ix < len(xs))
            for dy in shifts:
                iy = np.round((Q[:, 1] + dy - ys[0]) / res).astype(int)
                ok = okx & (iy >= 0) & (iy < len(ys))
                if not ok.all():
                    continue
                inside = hole_m[iy, ix]
                frac = float(inside.mean())
                clr = float(dist[iy, ix].min()) if inside.all() else 0.0
                key = (round(frac, 3), clr, -math.hypot(dx, dy) - 1e-4 * abs(yd))
                if best is None or key > best[0]:
                    best = (key, yd, dx, dy, iy, ix)
    if best is None:
        return None
    key, yd, dx, dy, iy, ix = best
    Rz = np.eye(4)
    Rz[:3, :3] = rot_z(math.radians(yd))
    Tc = np.eye(4)
    Tc[:2, 3] = c
    Tsh = np.eye(4)
    Tsh[:2, 3] = (dx, dy)
    T_new_loc = Tsh @ Tc @ Rz @ np.linalg.inv(Tc) @ T_loc                   # turn about the footprint centre, shift
    fl = D[iy, ix]
    fl = fl[np.isfinite(fl)]
    floor = float(fl.max()) if len(fl) else None
    dz = 0.0
    if press_floor and floor is not None and key[0] >= 0.999:
        dz = floor - tip                                                   # leading end down onto the floor
        T_new_loc[2, 3] += dz
    T_new = np.linalg.inv(T_hw) @ T_new_loc
    return T_new, {"inside": key[0], "clearance": key[1], "shift_mm": (round(dx * 1000, 2), round(dy * 1000, 2)),
                   "yaw_deg": yd, "lowered_mm": round(-dz * 1000, 1), "floor": floor, "top": top}
