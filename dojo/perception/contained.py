"""Objects inside a static container base (fill_egg_holder: four eggs in a wicker egg basket).

B1 fits every object as resting on the table against the table background. Inside a container both fail:
- the objects rest on the container's floor (eggs: 17 mm above the table), and the container's near wall hides their
  lower part, so a table-resting fit slides toward the camera (46-108 mm off in Tools' rendered runs);
- the container itself is foreground, so foreground says nothing about where an object lies inside it.

Here, once the base is chosen, a child is a smooth, light, low-saturation region inside the base's silhouette (the
evidence the base's own texture is interrupted: eggs on wicker), fitted with the child resting on the base's top
surface (a height grid of the base mesh) and only its pixels in front of the base counted (depth test against the
base's render). Children of the group already placed outside the base's footprint (an egg on the table under the
rim) are kept.
"""
from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np

from .cameras import TABLE_Z
from .render import gl, load_asset

CELL = 0.002            # m: height-grid cell of the base mesh
SAMPLES = 120_000       # surface samples for the grid
MIN_BLOB_PX = 15        # head-view evidence blobs smaller than this are noise (a light wicker strand)
MIN_SCORE = 0.3         # a child fit needs this visible-silhouette IoU with the evidence
EV_L, EV_B, EV_OPEN = 175.0, 160.0, 9   # evidence: Lab L > EV_L, b* < EV_B (8-bit Lab), opened by an EV_OPEN px disk
VIEWS: tuple = tuple(v for v in os.environ.get("DOJO_CONTAINED_VIEWS", "").split(",") if v)  # cameras the fit scores in (empty: all)


@lru_cache(maxsize=8)
def height_grid(key: str):
    """Top surface of the base mesh in its own frame: max z of area-weighted surface samples per CELL.
    Returns (grid (ny, nx), x0, y0) with NaN where the mesh has no surface."""
    a = load_asset(key)
    tri = a.V[a.F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(a.F), SAMPLES, p=area / area.sum())
    u = rng.random((SAMPLES, 2))
    flip = u.sum(1) > 1
    u[flip] = 1 - u[flip]
    P = tri[idx, 0] + u[:, :1] * (tri[idx, 1] - tri[idx, 0]) + u[:, 1:] * (tri[idx, 2] - tri[idx, 0])
    x0, y0 = P[:, 0].min(), P[:, 1].min()
    ix = ((P[:, 0] - x0) / CELL).astype(int)
    iy = ((P[:, 1] - y0) / CELL).astype(int)
    g = np.full((iy.max() + 1, ix.max() + 1), -np.inf)
    np.maximum.at(g, (iy, ix), P[:, 2])
    g[~np.isfinite(g)] = np.nan
    return g, float(x0), float(y0)


@lru_cache(maxsize=32)
def is_container(key: str, depth: float = 0.015) -> bool:
    """A base whose top surface at its middle lies ``depth`` or more below its highest point (a basket's floor under its
    rim; not a mat, a stand or a shelf's flat top)."""
    g, x0, y0 = height_grid(key)
    if not np.isfinite(g).any():
        return False
    ny, nx = g.shape
    c = g[int(0.4 * ny):int(0.6 * ny) + 1, int(0.4 * nx):int(0.6 * nx) + 1]
    if not np.isfinite(c).any():
        return False
    return bool(np.nanmax(c) <= np.nanmax(g) - depth)


def surface_z(base_key: str, T: np.ndarray, x: float, y: float) -> float | None:
    """World z of the base's top surface at world (x, y) (its pose ``T``), the max over the 3 x 3 cells around it."""
    g, x0, y0 = height_grid(base_key)
    p = np.linalg.inv(T) @ np.array([x, y, 0.0, 1.0])
    # the grid is in the base frame; a static base is upright (yaw only), so local z = world z - T[2, 3]
    ix, iy = int((p[0] - x0) / CELL), int((p[1] - y0) / CELL)
    if not (1 <= ix < g.shape[1] - 1 and 1 <= iy < g.shape[0] - 1):
        return None
    w = g[iy - 1:iy + 2, ix - 1:ix + 2]
    if np.isnan(w).all():
        return None
    return float(np.nanmax(w) + T[2, 3])


def evidence(img: np.ndarray) -> np.ndarray:
    """Light, low-saturation regions wider than a wicker strand (a child against a textured base): uint8 mask.
    11 egg layouts (head view, GT silhouettes): egg recall 0.66, 3 % of the basket's own pixels."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    ev = ((lab[..., 0] > EV_L) & (lab[..., 2] < EV_B)).astype(np.uint8)
    return cv2.morphologyEx(ev, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EV_OPEN, EV_OPEN)))


class _View:
    def __init__(self, v, base_key: str, Tb: np.ndarray):
        self.v = v
        a = load_asset(base_key)
        d = gl().render(v.cam, [(base_key, a.Vg, a.Fg, Tb)], depth=True)
        self.base_depth = np.where(d > 0, d, np.inf).astype(np.float32)
        self.inside = np.isfinite(self.base_depth)
        self.ev = evidence(v.img) & self.inside.astype(np.uint8) & (v.robot == 0).astype(np.uint8)
        self.known = (v.robot == 0)


def _score(sm, T, views: list[_View], pad: int = 6) -> tuple[float, int]:
    """Visible-silhouette IoU with the evidence, averaged over the views that see the child (>= 30 visible px)."""
    g = gl()
    tot, n = 0.0, 0
    for w in views:
        d = g.render(w.v.cam, [(sm.key, sm.asset.Vg, sm.asset.Fg, T)], depth=True)
        s = d > 0
        if s.sum() < 30:
            continue
        vis = s & (d < w.base_depth) & w.known
        if vis.sum() < 30:
            continue
        ys, xs = np.nonzero(s)
        y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad + 1, max(xs.min() - pad, 0), xs.max() + pad + 1
        vb, eb, kb = vis[y0:y1, x0:x1], w.ev[y0:y1, x0:x1].astype(bool), w.known[y0:y1, x0:x1]
        # evidence the base hides (behind its near wall) is not this child's: only the child's visible pixels and
        # the evidence inside the base around it count
        inter = float((vb & eb).sum())
        union = float(((vb | eb) & kb).sum())
        tot += inter / union if union else 0.0
        n += 1
    return (tot / n if n else 0.0), n


def fit_children(views: dict, base_key: str, Tb: np.ndarray, child_sm, n_max: int, exclude=()) -> list[dict]:
    """Up to ``n_max`` children inside the base: one per head-view evidence blob (largest first), each fitted in
    (x, y, yaw) resting on the base's top surface. ``exclude``: world xy of children already placed (skip blobs
    within 25 mm). Returns [{"x", "y", "yaw", "z_off", "score", "n_views"}] best first."""
    from scipy.optimize import minimize

    head = views["cam_head"]
    vs = [_View(v, base_key, Tb) for name, v in views.items() if "@" not in name and (name in VIEWS or not VIEWS)]
    wh = next(w for w in vs if w.v is head)
    h_c = 0.5 * child_sm.height
    out = []
    for _round in range(4 * n_max):
        # the largest evidence blob left (several children touching make one blob: each fit removes its own pixels)
        n, lab, st, cents = cv2.connectedComponentsWithStats(wh.ev, connectivity=8)
        if n <= 1 or len(out) >= n_max:
            break
        j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[j, cv2.CC_STAT_AREA] < MIN_BLOB_PX:
            break
        blob = lab == j
        z_s = surface_z(base_key, Tb, float(Tb[0, 3]), float(Tb[1, 3])) or TABLE_Z
        X = head.cam.ray_plane(cents[j][None, :], z_s + h_c)[0]
        for _ in range(3):  # the ray meets the child's centre height over the surface under it
            zz = surface_z(base_key, Tb, float(X[0]), float(X[1]))
            if zz is None:
                break
            X = head.cam.ray_plane(cents[j][None, :], zz + h_c)[0]
        if not np.isfinite(X).all() or any(np.hypot(X[0] - e[0], X[1] - e[1]) < 0.025 for e in exclude):
            wh.ev[blob] = 0
            continue

        def pose(p):
            zz = surface_z(base_key, Tb, float(p[0]), float(p[1]))
            z_off = (zz if zz is not None else TABLE_Z) - TABLE_Z
            return child_sm.pose(float(p[0]), float(p[1]), float(p[2]), z_off), z_off

        best = None
        for yaw in np.linspace(0, np.pi, 8, endpoint=False):
            s, k = _score(child_sm, pose([X[0], X[1], yaw])[0], vs)
            if best is None or s > best[0]:
                best = (s, np.array([X[0], X[1], yaw]))
        r = minimize(lambda p: -_score(child_sm, pose(p)[0], vs)[0], best[1], method="Nelder-Mead",
                     options={"initial_simplex": np.array([best[1], best[1] + [0.006, 0, 0], best[1] + [0, 0.006, 0],
                                                           best[1] + [0, 0, 0.3]]), "maxiter": 60, "xatol": 5e-4,
                              "fatol": 1e-3})
        p = r.x if -r.fun >= best[0] else best[1]
        T = pose(p)[0]
        s, k = _score(child_sm, T, vs)
        if s < MIN_SCORE or any(np.hypot(p[0] - o["x"], p[1] - o["y"]) < 0.025 for o in out):
            wh.ev[blob] = 0
            continue
        out.append({"x": float(p[0]), "y": float(p[1]), "yaw": float(p[2]), "z_off": float(pose(p)[1]), "score": round(s, 3),
                    "n_views": k})
        for w in vs:  # this child explains its evidence in every view
            sil = gl().render(w.v.cam, [(child_sm.key, child_sm.asset.Vg, child_sm.asset.Fg, T)]) > 0
            w.ev[cv2.dilate(sil.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
    return sorted(out, key=lambda o: -o["score"])
