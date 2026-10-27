"""Height above the table from calibrated views (plane sweep), using only the official RGB and the public camera
model (``cameras.py``: head fixed, wrists from FK).

Every camera pose is known, so any two views that see the same table region form a stereo rig. For a candidate
height h, the plane z = TABLE_Z + h induces a homography between a reference view (the head) and each other view;
warping the other view through it and scoring the photo-consistency (ZNCC over a window) per reference pixel gives a
cost curve over h. Its minimum is the pixel's height. Two uses:

- ``height``: height above the table per reference pixel (NaN where no other view sees the pixel or the match is
  ambiguous);
- ``proud``: the pixel is NOT explained by the table plane (its h = 0 warp disagrees) but IS explained by a plane
  >= ``h_fg`` above it. A cast shadow or a decal lies on the table in every view, so it is never proud; an object
  that stands up is, even with no colour contrast to its background (a black camera on a dark belt).

Horizontal planes make the warp exact for horizontal surfaces (object tops), so a fronto-parallel window works there.
Occlusion: each pixel keeps the best-scoring other view, so a point hidden in one view can still match in another.

    sw = plane_sweep(ref_view, [view_l, view_r])         # View(img RGB, cam, valid)
    sw.height, sw.proud, sw.cost_table, sw.cost_best
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .cameras import TABLE_Z, Camera

# where task configs put objects (scan.OBJECT_REGION) plus a margin: outside it the head sees the far table / walls at
# grazing angles to the wrists, where window matching is noise
REGION = ((-0.58, 0.58), (-0.40, 0.22))
DEFAULT_HEIGHTS = tuple(np.round(np.r_[np.arange(-0.01, 0.06, 0.002), np.arange(0.06, 0.30, 0.004)], 4))


@dataclass
class View:
    img: np.ndarray                 # RGB uint8 (H, W, 3)
    cam: Camera
    valid: np.ndarray | None = None  # uint8 1 = usable pixel (not robot, not off-table), None = all


@dataclass
class Sweep:
    heights: np.ndarray             # (n,) candidate heights above TABLE_Z (m)
    height: np.ndarray              # (H, W) float32, NaN = unknown
    cost_best: np.ndarray           # (H, W) best cost over all heights (0 = perfect match, 1 = anti-correlated)
    cost_table: np.ndarray          # (H, W) cost at h = 0 (inf = no other view)
    cost_raised: np.ndarray         # (H, W) best cost over h >= h_fg
    n_views: np.ndarray             # (H, W) uint8: other views that see the pixel at h = 0
    proud: np.ndarray               # (H, W) uint8: not on the table plane, explained above h_fg
    on_table: np.ndarray            # (H, W) uint8: explained by the table plane (a shadow / decal / bare table)
    known: np.ndarray               # (H, W) uint8: at least one other view, textured enough to decide
    valid: np.ndarray | None = None  # (H, W) uint8: the reference's usable pixels (region, not robot)
    scale: float = 1.0
    timings: dict = field(default_factory=dict)


def scaled_camera(cam: Camera, s: float) -> Camera:
    if s == 1.0:
        return cam
    K = cam.K.copy()
    K[0, 0] *= s
    K[1, 1] *= s
    K[0, 2] *= s
    K[1, 2] *= s
    return Camera(cam.name, K, cam.T_wc, int(round(cam.w * s)), int(round(cam.h * s)), dict(cam.meta))


def _plane_to_image(cam: Camera, z: float) -> np.ndarray:
    """3x3 homography: plane coordinates (x, y, 1) of the world plane at height z -> pixels (continuous, the
    ``Camera.project`` convention: pixel i covers [i, i+1))."""
    R, C = cam.T_wc[:3, :3], cam.T_wc[:3, 3]
    Rt = R.T
    return cam.K @ np.c_[Rt[:, 0], Rt[:, 1], Rt @ (np.array([0.0, 0.0, z]) - C)]


_SHIFT = np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]])   # integer pixel index -> continuous (centre)
_UNSHIFT = np.array([[1.0, 0, -0.5], [0, 1.0, -0.5], [0, 0, 1.0]])


def plane_homography(cam_ref: Camera, cam_src: Camera, z: float) -> np.ndarray:
    """ref pixel index (OpenCV, integer = pixel centre) -> src pixel index, through the world plane at height z."""
    H = _plane_to_image(cam_src, z) @ np.linalg.inv(_plane_to_image(cam_ref, z))
    return _UNSHIFT @ H @ _SHIFT


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def plane_sweep(ref: View, srcs: list[View], heights=DEFAULT_HEIGHTS, win: int = 7, scale: float = 0.5,
                tex_min: float = 0.004, h_fg: float = 0.012, margin: float = 0.08, table_ok: float = 0.25,
                unique: float = 0.03, agg: int = 0, fill: bool = False, min_blob: int = 0) -> Sweep:
    """Plane sweep in ``ref``'s image over horizontal planes at TABLE_Z + ``heights``.

    Cost per (pixel, h, view) = (1 - ZNCC) / 2 over a ``win`` x ``win`` window of grey levels (gain / offset
    invariant: the head and wrist cameras expose differently); a pixel keeps the best view per height. Windows whose
    grey-level std is below ``tex_min`` in the reference are "unknown" (no texture, nothing to decide).
    ``proud`` = cost_raised < cost_table - ``margin`` and cost_table > ``table_ok`` (the table plane does not explain
    the pixel, a raised plane does); ``on_table`` = cost_table <= ``table_ok``. ``height`` is the winner (sub-step
    parabola), NaN where unknown or the best is not ``unique`` (the best cost more than ``unique`` below the best
    cost at least 3 steps away)."""
    t0 = time.time()
    heights = np.asarray(heights, dtype=np.float64)
    rcam = scaled_camera(ref.cam, scale)
    h, w = rcam.h, rcam.w

    def prep(v: View):
        img = v.img if scale == 1.0 else cv2.resize(v.img, (int(round(v.img.shape[1] * scale)), int(round(v.img.shape[0] * scale))),
                                                     interpolation=cv2.INTER_AREA)
        val = np.ones(img.shape[:2], np.float32) if v.valid is None else (
            cv2.resize(v.valid.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        return _gray(img), val, scaled_camera(v.cam, scale)

    gr, vr, _ = prep(ref)
    box = lambda a: cv2.boxFilter(a, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT)  # noqa: E731
    mu_r = box(gr)
    var_r = np.maximum(box(gr * gr) - mu_r * mu_r, 0.0)
    textured = var_r > tex_min ** 2
    S = [prep(s) for s in srcs]
    t1 = time.time()
    vol = np.full((len(heights), h, w), np.inf, np.float32)
    nview0 = np.zeros((h, w), np.uint8)
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    i0 = int(np.argmin(np.abs(heights)))
    for k, hk in enumerate(heights):
        z = TABLE_Z + float(hk)
        best = vol[k]
        for gs, vs, cs in S:
            H = plane_homography(rcam, cs, z)
            den = H[2, 0] * uu + H[2, 1] * vv + H[2, 2]
            ws = cv2.warpPerspective(gs, H, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
            wv = cv2.warpPerspective(vs, H, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)
            wv = (wv > 0.5) & (den > 0)
            okw = box(wv.astype(np.float32)) > 0.999
            mu_s = box(ws)
            var_s = np.maximum(box(ws * ws) - mu_s * mu_s, 0.0)
            cov = box(gr * ws) - mu_r * mu_s
            zn = cov / np.sqrt(var_r * var_s + 1e-10)
            c = np.where(okw & (var_s > (0.5 * tex_min) ** 2), 0.5 * (1.0 - zn), np.inf).astype(np.float32)
            if agg > 1:  # a second, larger window over the per-pixel cost (unseen = cost 1)
                c = np.where(okw, cv2.boxFilter(np.where(np.isfinite(c), c, 0.5).astype(np.float32), -1, (agg, agg)), np.inf).astype(np.float32)
            np.minimum(best, c, out=best)
            if k == i0:
                nview0 += okw.astype(np.uint8)
    t2 = time.time()
    known = textured & (vr > 0.5) & (nview0 > 0)
    kbest = np.argmin(vol, axis=0)
    cbest = np.take_along_axis(vol, kbest[None], 0)[0]
    # sub-step parabola
    km, kp = np.clip(kbest - 1, 0, len(heights) - 1), np.clip(kbest + 1, 0, len(heights) - 1)
    cm = np.take_along_axis(vol, km[None], 0)[0]
    cp = np.take_along_axis(vol, kp[None], 0)[0]
    den = cm - 2 * cbest + cp
    off = np.where(np.isfinite(den) & (den > 1e-6) & (kbest > 0) & (kbest < len(heights) - 1), 0.5 * (cm - cp) / np.where(den > 1e-6, den, 1), 0.0)
    off = np.clip(off, -0.5, 0.5)
    hb = np.interp(kbest + off, np.arange(len(heights)), heights).astype(np.float32)
    # uniqueness: the best cost at least 3 steps away from the winner
    idx = np.arange(len(heights))[:, None, None]
    far = np.abs(idx - kbest[None]) >= 3
    c2 = np.where(far, vol, np.inf).min(axis=0)
    good = known & np.isfinite(cbest) & (c2 - cbest > unique)
    height = np.where(good, hb, np.nan).astype(np.float32)
    raised = heights >= h_fg
    craised = vol[raised].min(axis=0) if raised.any() else np.full((h, w), np.inf, np.float32)
    ctab = vol[i0]
    proud = known & (craised < ctab - margin) & (ctab > table_ok)
    on_table = known & (ctab <= table_ok)
    if fill:  # a uniform top (a white block, a black camera) is undecidable inside a proud rim: fill the rim's holes
        proud = fill_holes(proud.astype(np.uint8), on_table.astype(np.uint8)) > 0
    if min_blob > 0:
        n, lab, st, _ = cv2.connectedComponentsWithStats(proud.astype(np.uint8))
        keep = np.r_[False, st[1:, cv2.CC_STAT_AREA] >= min_blob]
        proud = keep[lab]
    t3 = time.time()
    return Sweep(heights=heights, height=height, cost_best=cbest, cost_table=ctab, cost_raised=craised, n_views=nview0,
                 proud=proud.astype(np.uint8), on_table=on_table.astype(np.uint8), known=known.astype(np.uint8),
                 valid=(vr > 0.5).astype(np.uint8), scale=scale,
                 timings={"prep_s": round(t1 - t0, 3), "sweep_s": round(t2 - t1, 3), "post_s": round(t3 - t2, 3),
                          "total_s": round(t3 - t0, 3), "n_planes": int(len(heights)), "n_views": len(srcs), "px": h * w})


def fill_holes(mask: np.ndarray, blocked: np.ndarray | None = None) -> np.ndarray:
    """Fill the holes of a uint8 mask (regions not connected to the border), except pixels in ``blocked``."""
    inv = (mask == 0).astype(np.uint8)
    border = np.zeros_like(inv, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    _, lab = cv2.connectedComponents(inv, connectivity=4)
    outside = np.unique(lab[border & (inv == 1)])
    holes = (inv == 1) & ~np.isin(lab, outside)
    if blocked is not None:
        holes &= blocked == 0
    return (mask.astype(bool) | holes).astype(np.uint8)


def table_region_mask(cam: Camera, region=REGION, z: float = TABLE_Z) -> np.ndarray:
    """uint8 1 where the pixel's ray hits the plane z inside ``region`` ((x0, x1), (y0, y1))."""
    vv, uu = np.mgrid[0:cam.h, 0:cam.w]
    X = cam.ray_plane(np.c_[uu.ravel() + 0.5, vv.ravel() + 0.5], z)
    ok = np.isfinite(X).all(1)
    X = np.nan_to_num(X)
    ok &= (X[:, 0] >= region[0][0]) & (X[:, 0] <= region[0][1]) & (X[:, 1] >= region[1][0]) & (X[:, 1] <= region[1][1])
    return ok.reshape(cam.h, cam.w).astype(np.uint8)


def home_views(images: dict, state: dict, robot_dilate: int = 4, region=REGION) -> tuple[View, list[View]]:
    """Reference = the head camera, others = both wrist cameras, each with its FK robot mask removed."""
    from .cameras import cameras_from_state
    from .render import robot_mask

    cams = cameras_from_state(state)
    views = {}
    for name, img in images.items():
        if name not in cams:
            continue
        rob = robot_mask(cams[name], state, dilate_px=robot_dilate)
        views[name] = View(img, cams[name], (1 - rob).astype(np.uint8))
    ref = views.pop("cam_head")
    if region is not None:
        ref.valid = ref.valid & table_region_mask(ref.cam, region)
    return ref, list(views.values())


def upsample(a: np.ndarray, shape, nearest: bool = True) -> np.ndarray:
    """A sweep map back to the full image size."""
    return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
