"""Yaw from a feature cut into an object's top face, for objects whose outline cannot show it.

insert_key's lock (``Geometry/key_slot``) is a copper cylinder, 100 mm across: its silhouette is the same at every yaw
(shape sym 36), so B1's yaw is arbitrary and the agent's or the prior's yaw wins by default (5.1 keyR 100003: 29.6 deg,
then 0.03 deg; truth 10.7 deg). The key enters through a keyhole in the top: a 9 mm round bulb and an 11 x 6 mm channel,
20 mm long and 30 mm deep. The keyhole's long axis gives the yaw mod 180 deg; the bulb end gives the sign.

The template comes from the mesh, so nothing is keyed on a task or an asset name:
- the top face = triangles at the mesh's highest level, facing up;
- the opening = what the top face's own outline encloses but the face does not cover;
- a feature exists when the shape's silhouette is (near) rotationally symmetric and the opening is small, off-round
  and elongated (a mug's open top, a ring's round hole or a plain disc has none).

The read back-projects the pixels inside the top disc that are darker than their neighbourhood onto the top plane (a ray through the opening crosses the
plane inside it), then matches the template opening over yaw and a small shift on a 0.5 mm grid. Its confidence is
the fit's chamfer and how much worse the best yaw at least ``RIVAL_SEP_DEG`` away fits (the 180 deg flip included): a wrist
view straight above (15-20 cm) shows the keyhole at 40-80 px and resolves both; the head view (about 6 x 3 px) gives a
coarse yaw at best.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .cameras import TABLE_Z

GRID = 0.0005                # m: rasterisation cell of the top face and of the matching
MIN_HOLE_M2 = 4e-6           # an opening smaller than 4 mm^2 is a mesh artefact
MAX_HOLE_FRAC = 0.25         # an opening this large a share of the top is a container's mouth, not a feature
MIN_ELONGATION = 1.4         # principal-axis ratio: a round hole (ring) carries no yaw
MIN_SYM = 8                  # only outlines symmetric at least this much (the outline gives the yaw otherwise)
ROI_PAD_M = 0.009            # the read looks this far beyond the opening's reach (the outline fit's xy error)
SIGN_MAX_PX_M = 0.001        # the bulb side needs the opening's ends resolved: at most 1 mm per pixel on the top
SIGN_MIN_ASYM = 0.5          # ... and the ends' width ratio at least this share of the template's (log)
LOCAL_SIGMA_M = 0.004        # the neighbourhood a pixel of the opening is darker than (Gaussian sigma on the top, m)
LOCAL_DARK = 20.0            # ... by this much lightness (LAB L, 0-255)
RIVAL_SEP_DEG = 45.0         # the margin compares against the best yaw at least this far away (the flip too)
MAX_COST_CAP = 0.35          # an accepted read's chamfer, as a share of its maximum (both terms at the cap)
MIN_RIVAL_RATIO = 1.5        # ... and the best yaw >= RIVAL_SEP_DEG away costs this many times more
MIN_DARK_PX = 12


@dataclass
class TopFeature:
    key: str
    tag: str
    z_top: float              # top face height above the shape's lowest point (rest pose), m
    radius: float             # radius of the top's filled outline about its centre, m
    centre: np.ndarray        # (2,) centre of the top outline, rest frame at pose (0, 0, 0), world xy
    hole: np.ndarray          # (N, 2) opening cells, same frame
    axis_deg: float           # the opening's principal axis, same frame
    elongation: float
    reach: float = 0.0        # the opening's farthest point from the centre, m

    def hole_world(self, T: np.ndarray, T0: np.ndarray) -> np.ndarray:
        """The opening's cells (N, 2) for a pose ``T``, given the rest pose ``T0`` the template was taken at."""
        M = T @ np.linalg.inv(T0)
        P = np.c_[self.hole, np.zeros(len(self.hole)), np.ones(len(self.hole))]
        return (P @ M.T)[:, :2]


def _raster_tris(P2: np.ndarray, lo: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    q = np.round((P2 - lo) / GRID * 16).astype(np.int32)
    for tri in q:
        cv2.fillConvexPoly(m, tri, 1, lineType=cv2.LINE_8, shift=4)
    return m


def _fill_holes(m: np.ndarray) -> np.ndarray:
    pad = np.pad(m, 1)
    inv = (1 - pad).astype(np.uint8)
    ff = inv.copy()
    cv2.floodFill(ff, None, (0, 0), 2)
    holes = (ff == 1).astype(np.uint8)
    return (pad | holes)[1:-1, 1:-1]


def top_feature_from_mesh(V: np.ndarray, F: np.ndarray, T0: np.ndarray, key: str = "", tag: str = "") -> TopFeature | None:
    """The opening in the top face of a mesh posed by ``T0`` (rest pose at yaw 0), or None."""
    Vw = V @ T0[:3, :3].T + T0[:3, 3]
    z_lo, z_top = float(Vw[:, 2].min()), float(Vw[:, 2].max())
    tri = Vw[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nn = np.linalg.norm(n, axis=1)
    up = (nn > 0) & (n[:, 2] / np.maximum(nn, 1e-12) > 0.95) & (tri[:, :, 2].min(1) > z_top - 0.0005)
    if not up.any():
        return None
    P2 = tri[up][:, :, :2]
    lo = P2.reshape(-1, 2).min(0) - 4 * GRID
    hi = P2.reshape(-1, 2).max(0) + 4 * GRID
    shape = (int(np.ceil((hi[1] - lo[1]) / GRID)) + 1, int(np.ceil((hi[0] - lo[0]) / GRID)) + 1)
    face = _raster_tris(P2, lo, shape)
    face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))   # seams between triangles
    filled = _fill_holes(face)
    hole = filled & (1 - face)
    hole = cv2.morphologyEx(hole, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_h, n_f = int(hole.sum()), int(filled.sum())
    if n_h * GRID * GRID < MIN_HOLE_M2 or n_h > MAX_HOLE_FRAC * n_f:
        return None
    # the largest opening only
    k, lab, st, _ = cv2.connectedComponentsWithStats(hole, connectivity=8)
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    vv, uu = np.nonzero(lab == j)
    H = np.c_[lo[0] + (uu + 0.5) * GRID, lo[1] + (vv + 0.5) * GRID]
    fv, fu = np.nonzero(filled)
    Fp = np.c_[lo[0] + (fu + 0.5) * GRID, lo[1] + (fv + 0.5) * GRID]
    c = Fp.mean(0)
    C = np.cov((H - H.mean(0)).T)
    w, U = np.linalg.eigh(C)
    elong = float(np.sqrt(max(w[1], 1e-12) / max(w[0], 1e-12)))
    if elong < MIN_ELONGATION:
        return None
    ax = U[:, 1]
    return TopFeature(key, tag, z_top - z_lo, float(np.percentile(np.linalg.norm(Fp - c, axis=1), 99)), c, H,
                      float(np.degrees(np.arctan2(ax[1], ax[0]))), elong, float(np.linalg.norm(H - c, axis=1).max()))


@lru_cache(maxsize=64)
def _cached(key: str, tag: str) -> TopFeature | None:
    from .inverse_layout import shape_models

    sm = next((s for s in shape_models(key) if s.tag == tag), None)
    if sm is None or min(sm.sym, 36) < MIN_SYM:
        return None
    return top_feature_from_mesh(sm.asset.V, sm.asset.F, sm.pose(0.0, 0.0, 0.0), key, tag)


def top_feature(sm) -> TopFeature | None:
    """The cached top feature of a shape model (``inverse_layout.ShapeModel``), or None."""
    try:
        return _cached(sm.key, sm.tag)
    except Exception:  # noqa: BLE001 (an optional read: no feature)
        return None


def dark_points(img: np.ndarray, cam, feat: TopFeature, sm, x: float, y: float, z_off: float = 0.0,
                robot: np.ndarray | None = None, margin: float = 0.003) -> tuple[np.ndarray, int, int]:
    """World xy of the dark pixels inside the top disc (shrunk by ``margin``) back-projected onto the top plane,
    the number of disc pixels seen, and the number of dark ones."""
    T0 = sm.pose(0.0, 0.0, 0.0)
    T = sm.pose(x, y, 0.0, z_off)
    c = (T @ np.linalg.inv(T0) @ np.r_[feat.centre, 0.0, 1.0])[:2]
    z_plane = TABLE_Z + z_off + feat.z_top           # the rest pose puts the lowest point on the table
    px_m = float(np.linalg.norm(cam.center - np.r_[c, z_plane])) / float(cam.K[0, 0])
    # around the opening only (its reach + the prior's xy error), not the whole top: the rim and the shading near it
    # are no part of the read (5.1 wrist hover: a 5 mm off outline fit put 1000+ rim pixels in the whole-disc ROI)
    r = max(min(feat.radius - margin, feat.reach + max(ROI_PAD_M, 4 * px_m)), 0.004)
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ring = np.c_[c[0] + r * np.cos(th), c[1] + r * np.sin(th), np.full(len(th), z_plane)]
    uv, zc = cam.project(ring)
    if (zc <= 0.01).any():
        return np.zeros((0, 2)), 0, 0
    roi = np.zeros((cam.h, cam.w), np.uint8)
    cv2.fillPoly(roi, [np.round(uv - 0.5).astype(np.int32)], 1)
    if robot is not None:
        roi &= (robot == 0).astype(np.uint8)
    n_roi = int(roi.sum())
    if n_roi < 20:
        return np.zeros((0, 2)), n_roi, 0
    # darker than its own neighbourhood on the top: the lock's top is shaded (5.1 wrist hover: L 60-200 across the disc,
    # the lower half under the gripper's shadow), while the opening is a sharp local drop (L ~45 against ~170 around it,
    # its floor lit from above). Normalised blur, so the disc's edge does not darken the neighbourhood
    L = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    sig = float(np.clip(LOCAL_SIGMA_M / px_m, 2.0, 25.0))
    w = cv2.GaussianBlur(roi.astype(np.float32), (0, 0), sig)
    local = cv2.GaussianBlur(L * roi, (0, 0), sig) / np.maximum(w, 1e-3)
    dark = (roi > 0) & (L - local < -LOCAL_DARK)
    vv, uu = np.nonzero(dark)
    if len(vv) == 0:
        return np.zeros((0, 2)), n_roi, 0
    P = cam.ray_plane(np.c_[uu + 0.5, vv + 0.5], z_plane)
    P = P[np.isfinite(P).all(1)][:, :2]
    return P, n_roi, int(len(vv))


def _end_widths(P: np.ndarray, axis: float, frac: float = 0.3) -> tuple[float, float] | None:
    """Widths (5-95 % across the axis) of the first and last ``frac`` of a point set's extent along ``axis`` (rad)."""
    a = np.array([np.cos(axis), np.sin(axis)])
    b = np.array([-a[1], a[0]])
    u, v = P @ a, P @ b
    lo, hi = np.percentile(u, [2, 98])
    if hi - lo <= 1e-6:
        return None
    out = []
    for m in (u <= lo + frac * (hi - lo), u >= hi - frac * (hi - lo)):
        if m.sum() < 5:
            return None
        out.append(float(np.percentile(v[m], 95) - np.percentile(v[m], 5)))
    return out[0], out[1]


def _end_asymmetry(H: np.ndarray, D: np.ndarray, axis: float) -> tuple[float, float, bool]:
    """log(width of the high end / low end) for the template at the fit and for the observed points, and whether the
    observed wide end is the template's narrow one (the fit is flipped)."""
    wt, wd = _end_widths(H, axis), _end_widths(D, axis)
    if wt is None or wd is None or min(wt + wd) <= 0:
        return 0.0, 0.0, False
    at, ad = float(np.log(wt[1] / wt[0])), float(np.log(wd[1] / wd[0]))
    return at, ad, bool(at * ad < 0)


def read_yaw(img: np.ndarray, cam, sm, x: float, y: float, z_off: float = 0.0, robot: np.ndarray | None = None,
             shift: float | None = None, yaw_step_deg: float = 3.0) -> dict | None:
    """The yaw of shape ``sm`` at (x, y) from its top feature in one view, or None when the shape has no feature:
    {"yaw" (rad, the pose parameter of ``sm.pose``), "x", "y", "ok" (the axis is read), "sign_ok" (and the bulb side),
    "chamfer_mm", "tail_mm", "ratio_axis", "ratio_flip", "n_dark", ...}.

    The cost is a truncated symmetric chamfer between the dark points on the top plane and the template opening
    (continuous in yaw and shift, unlike a grid overlap that quantises a 20 mm opening to ~3 deg). Coarse: every
    ``yaw_step_deg`` over the full turn, the shift that matches the centroids (clipped to ``shift``); fine: Nelder-Mead
    from the best three. Confidence: the misfit's tail (90th percentile) against the best fit across the axis
    (``ratio_axis``) and near the flip (``ratio_flip``), each at least ``MIN_RIVAL_RATIO`` times worse.

    The opening is anchored to the object's centre (the outline fit's xy, 1-2 mm), and that anchor is what tells the
    bulb end from the channel end: the opening flipped and slid 6 mm along its axis fits the dark pixels almost as well.
    So the shift stays small: 3 mm from a wrist above, up to 6 mm from the head (``shift`` overrides)."""
    feat = top_feature(sm)
    if feat is None:
        return None
    D, n_roi, n_dark = dark_points(img, cam, feat, sm, x, y, z_off, robot)
    if n_dark < MIN_DARK_PX or len(D) < MIN_DARK_PX:
        return {"ok": False, "why": "opening not seen", "n_dark": n_dark, "n_roi": n_roi}
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    T0 = sm.pose(0.0, 0.0, 0.0)
    centre = (sm.pose(x, y, 0.0, z_off) @ np.linalg.inv(T0) @ np.r_[feat.centre, 0.0, 1.0])[:2]
    rng = float(np.linalg.norm(cam.center - np.r_[centre, TABLE_Z + z_off + feat.z_top]))
    px_m = rng / float(cam.K[0, 0])                  # one pixel on the plane: ~3 mm from the head, ~0.5 mm 15 cm up
    cap = max(0.002, 3 * px_m)
    if shift is None:
        shift = ROI_PAD_M
    tree_d = cKDTree(D)
    cD = D.mean(0)

    def cost(yaw, dx, dy):
        H = feat.hole_world(sm.pose(x + dx, y + dy, yaw, z_off), T0)
        a = np.minimum(tree_d.query(H)[0], cap).mean()
        b = np.minimum(cKDTree(H).query(D)[0], cap).mean()
        return float(a + b)

    yaws = np.radians(np.arange(0.0, 360.0, yaw_step_deg))
    coarse = []
    for yw in yaws:
        t = cD - feat.hole_world(sm.pose(x, y, yw, z_off), T0).mean(0)
        t = np.clip(t, -shift, shift)
        coarse.append((cost(yw, *t), yw, float(t[0]), float(t[1])))
    c_arr = np.array([c[0] for c in coarse])
    n = len(coarse)
    minima = [k for k in range(n) if c_arr[k] <= c_arr[(k - 1) % n] and c_arr[k] <= c_arr[(k + 1) % n]]
    minima = sorted(minima, key=lambda k: c_arr[k])

    def fine(k):
        _, yw0, dx0, dy0 = coarse[k]
        r = minimize(lambda p: cost(*p), [yw0, dx0, dy0], method="Nelder-Mead",
                     options={"initial_simplex": [[yw0, dx0, dy0], [yw0 + np.radians(yaw_step_deg), dx0, dy0],
                                                  [yw0, dx0 + 0.001, dy0], [yw0, dx0, dy0 + 0.001]],
                              "xatol": 1e-4, "fatol": 1e-7, "maxiter": 400})
        p = r.x if max(abs(r.x[1]), abs(r.x[2])) <= shift + 0.002 else np.array([yw0, dx0, dy0])
        return (cost(*p), float(p[0]), float(p[1]), float(p[2]))

    fits = [fine(k) for k in minima[:3]]
    c, yw, dx, dy = min(fits)
    # rivals: the best fits at least RIVAL_SEP_DEG away, refined like the winner. Across the axis (45-135 deg) they test
    # the axis; near the flip (> 135 deg) the bulb side
    def sep(f):
        return abs((np.degrees(f[1] - yw) + 180.0) % 360.0 - 180.0)

    far = [k for k in minima if abs((np.degrees(yaws[k] - yw) + 180.0) % 360.0 - 180.0) >= RIVAL_SEP_DEG][:3]
    flip_k = int(np.argmin([abs(abs((np.degrees(yk - yw) + 180.0) % 360.0 - 180.0) - 180.0) for yk in yaws]))
    rivals = [f for f in fits if sep(f) >= RIVAL_SEP_DEG] + [fine(k) for k in dict.fromkeys(far + [flip_k])]

    # confidence: the tail of the misfit (90th percentile, both ways), which the bulb and the channel ends drive; the
    # mean chamfer of a flipped opening shifted along its axis is only ~20 % worse
    def tail(f):
        H = feat.hole_world(sm.pose(x + f[2], y + f[3], f[1], z_off), T0)
        return float(max(np.percentile(tree_d.query(H)[0], 90), np.percentile(cKDTree(H).query(D)[0], 90)))

    q_best = tail((c, yw, dx, dy))
    r_axis = min((tail(f) for f in rivals if RIVAL_SEP_DEG <= sep(f) < 180.0 - RIVAL_SEP_DEG), default=float("inf"))
    r_flip = min((tail(f) for f in rivals if sep(f) >= 180.0 - RIVAL_SEP_DEG), default=float("inf"))
    ratio_axis, ratio_flip = r_axis / max(q_best, 1e-9), r_flip / max(q_best, 1e-9)
    ok = bool(c <= MAX_COST_CAP * 2 * cap and ratio_axis >= MIN_RIVAL_RATIO)
    # the bulb side: with the shift free (the outline fit's xy is a few mm off), a flipped opening slid along its axis
    # fits the points almost as well, so the side comes from the shape: the opening's two ends differ in width (the
    # keyhole's 8.8 mm bulb, its 6 mm channel). Compare the observed ends with the template's at the fitted axis
    asym_t, asym_d, flipped = _end_asymmetry(feat.hole_world(sm.pose(x + dx, y + dy, yw, z_off), T0), D, yw + np.radians(feat.axis_deg))
    sign_ok = bool(ok and px_m <= SIGN_MAX_PX_M and abs(asym_t) > 0.05 and abs(asym_d) >= SIGN_MIN_ASYM * abs(asym_t))
    if sign_ok and flipped:
        k_flip = int(np.argmin([abs(abs((np.degrees(yk - yw) + 180.0) % 360.0 - 180.0) - 180.0) for yk in yaws]))
        c, yw, dx, dy = fine(k_flip)
    return {"ok": ok, "sign_ok": sign_ok, "yaw": float((yw + np.pi) % (2 * np.pi) - np.pi), "x": float(x + dx),
            "y": float(y + dy), "chamfer_mm": round(1000 * c, 2), "tail_mm": round(1000 * q_best, 2),
            "ratio_axis": round(float(min(ratio_axis, 99.0)), 2), "ratio_flip": round(float(min(ratio_flip, 99.0)), 2),
            "end_asym": round(float(asym_d), 3), "end_asym_template": round(float(asym_t), 3), "px_mm": round(1000 * px_m, 2),
            "n_dark": n_dark, "n_roi": n_roi, "camera": getattr(cam, "name", None)}
