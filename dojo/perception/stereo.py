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


def _cost_volume(gr, var_r, rcam: Camera, S: list, heights, win: int, tex_min: float, agg: int, raised_from: float):
    """(volume: min over views per plane, n views seeing each pixel at the table plane, per view: (best cost over
    planes >= ``raised_from``, its plane index))."""
    h, w = gr.shape
    box = lambda a: cv2.boxFilter(a, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT)  # noqa: E731
    mu_r = box(gr)
    vol = np.full((len(heights), h, w), np.inf, np.float32)
    nview0 = np.zeros((h, w), np.uint8)
    per_view = [(np.full((h, w), np.inf, np.float32), np.zeros((h, w), np.int16)) for _ in S]
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    i0 = int(np.argmin(np.abs(heights)))
    for k, hk in enumerate(heights):
        z = TABLE_Z + float(hk)
        best = vol[k]
        for j, (gs, vs, cs) in enumerate(S):
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
            if hk >= raised_from:
                cb, kb = per_view[j]
                better = c < cb
                cb[better] = c[better]
                kb[better] = k
            if k == i0:
                nview0 += okw.astype(np.uint8)
    return vol, nview0, per_view


def _wta(vol, heights):
    kbest = np.argmin(vol, axis=0)
    cbest = np.take_along_axis(vol, kbest[None], 0)[0]
    km, kp = np.clip(kbest - 1, 0, len(heights) - 1), np.clip(kbest + 1, 0, len(heights) - 1)
    cm = np.take_along_axis(vol, km[None], 0)[0]
    cp = np.take_along_axis(vol, kp[None], 0)[0]
    with np.errstate(invalid="ignore"):
        den = cm - 2 * cbest + cp
        ok = np.isfinite(den) & (den > 1e-6) & (kbest > 0) & (kbest < len(heights) - 1)
        off = np.where(ok, 0.5 * (cm - cp) / np.where(ok, den, 1), 0.0)
    off = np.clip(np.nan_to_num(off), -0.5, 0.5)
    hb = np.interp(kbest + off, np.arange(len(heights)), heights).astype(np.float32)
    return kbest, cbest, hb


def _project_heights(rcam: Camera, cam: Camera, hmap: np.ndarray) -> np.ndarray:
    """ref pixels (OpenCV index) lifted to TABLE_Z + hmap along their rays -> pixel coordinates (index) in ``cam``."""
    h, w = hmap.shape
    vv, uu = np.mgrid[0:h, 0:w]
    d = rcam.rays(np.c_[uu.ravel() + 0.5, vv.ravel() + 0.5])
    o = rcam.center
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (TABLE_Z + hmap.ravel() - o[2]) / d[:, 2]
    X = o[None] + t[:, None] * d
    uv, z = cam.project(np.nan_to_num(X))
    uv = uv - 0.5
    uv[(z <= 0) | ~np.isfinite(t) | (t <= 0)] = -1e6
    return uv.reshape(h, w, 2)


def plane_sweep(ref: View, srcs: list[View], heights=DEFAULT_HEIGHTS, win: int = 7, scale: float = 0.5,
                tex_min: float = 0.004, h_fg: float = 0.012, margin: float = 0.08, table_ok: float = 0.25,
                unique: float = 0.03, agg: int = 0, fill: bool = False, min_blob: int = 0, lr: bool = False,
                lr_tol: float = 0.006, lr_rel: float = 0.15, lr_nb: int = 3, lr_hyst: int = 0,
                proud_unique: float = 0.0) -> Sweep:
    """Plane sweep in ``ref``'s image over horizontal planes at TABLE_Z + ``heights``.

    Cost per (pixel, h, view) = (1 - ZNCC) / 2 over a ``win`` x ``win`` window of grey levels (gain / offset
    invariant: the head and wrist cameras expose differently); a pixel keeps the best view per height. Windows whose
    grey-level std is below ``tex_min`` in the reference are "unknown" (no texture, nothing to decide).
    ``proud`` = cost_raised < cost_table - ``margin`` and cost_table > ``table_ok`` (the table plane does not explain
    the pixel, a raised plane does); ``on_table`` = cost_table <= ``table_ok``. ``height`` is the winner (sub-step
    parabola), NaN where unknown or the best is not ``unique`` (the best cost more than ``unique`` below the best
    cost at least 3 steps away).

    ``lr``: left-right consistency. Next to an object, a table point the head sees can be hidden by the object in a
    wrist view; the table plane then fails there and a raised plane matches the object's texture by chance (an
    "occlusion halo"). With ``lr`` each view also sweeps with itself as the reference and the head as the source; a
    proud pixel needs a view whose own best height at the matched pixel agrees with the raised height
    (within ``lr_tol`` + ``lr_rel`` * h, over the view's ``lr_nb`` x ``lr_nb`` neighbourhood). ``lr_hyst`` > 0:
    hysteresis instead, every raw proud component with >= ``lr_hyst`` consistent pixels is kept whole (a halo
    attached to its object stays, a free-standing halo or a shadow goes).

    ``proud_unique`` > 0: a proud pixel's best raised height must be unique (the best raised cost at least 3 steps
    away is ``proud_unique`` worse). A texture running along the epipolar lines (the belt's stripes, wood grain at
    some angles) matches at many heights equally well; an object's surface matches at one."""
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

    def stats(g):
        mu = cv2.boxFilter(g, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT)
        return np.maximum(cv2.boxFilter(g * g, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT) - mu * mu, 0.0)

    gr, vr, _ = prep(ref)
    var_r = stats(gr)
    textured = var_r > tex_min ** 2
    S = [prep(s) for s in srcs]
    t1 = time.time()
    vol, nview0, per_view = _cost_volume(gr, var_r, rcam, S, heights, win, tex_min, agg, h_fg)
    t2 = time.time()
    known = textured & (vr > 0.5) & (nview0 > 0)
    kbest, cbest, hb = _wta(vol, heights)
    idx = np.arange(len(heights))[:, None, None]
    far = np.abs(idx - kbest[None]) >= 3
    c2 = np.where(far, vol, np.inf).min(axis=0)
    with np.errstate(invalid="ignore"):
        good = known & np.isfinite(cbest) & (c2 - cbest > unique)
    height = np.where(good, hb, np.nan).astype(np.float32)
    raised = heights >= h_fg
    craised = vol[raised].min(axis=0) if raised.any() else np.full((h, w), np.inf, np.float32)
    ctab = vol[int(np.argmin(np.abs(heights)))]
    uniq_r = np.ones((h, w), bool)
    if proud_unique > 0 and raised.any():
        vr_ = vol[raised]
        kr = np.argmin(vr_, axis=0)
        ir = np.arange(vr_.shape[0])[:, None, None]
        c2r = np.where(np.abs(ir - kr[None]) >= 3, vr_, np.inf).min(axis=0)
        with np.errstate(invalid="ignore"):
            uniq_r = (c2r - craised) > proud_unique
    with np.errstate(invalid="ignore"):
        on_table = known & (ctab <= table_ok)
        if not lr:
            proud = known & (craised < ctab - margin) & (ctab > table_ok) & uniq_r
        else:
            proud = np.zeros((h, w), bool)
            raw = known & (craised < ctab - margin) & (ctab > table_ok) & uniq_r
            head_src = [(gr, vr, rcam)]
            for (gs, vs, cs), (cb, kb) in zip(S, per_view):
                cand = known & (cb < ctab - margin) & (ctab > table_ok) & uniq_r
                if not cand.any():
                    continue
                vol_s, _, _ = _cost_volume(gs, stats(gs), cs, head_src, heights, win, tex_min, agg, 1e9)
                ks, cs_best, hs = _wta(vol_s, heights)
                hs = np.where(np.isfinite(cs_best) & (vs > 0.5), hs, np.nan).astype(np.float32)
                kern = np.ones((lr_nb, lr_nb), np.uint8)
                hs_lo = cv2.erode(np.where(np.isfinite(hs), hs, np.float32(1e3)), kern)
                hs_hi = cv2.dilate(np.where(np.isfinite(hs), hs, np.float32(-1e3)), kern)
                hstar = heights[kb].astype(np.float32)
                uv = _project_heights(rcam, cs, hstar)
                qx = np.clip(np.round(uv[..., 0]).astype(np.int64), -1, cs.w)
                qy = np.clip(np.round(uv[..., 1]).astype(np.int64), -1, cs.h)
                inside = (qx >= 0) & (qx < cs.w) & (qy >= 0) & (qy < cs.h)
                lo = np.full((h, w), 1e3, np.float32)
                hi = np.full((h, w), -1e3, np.float32)
                lo[inside] = hs_lo[qy[inside], qx[inside]]
                hi[inside] = hs_hi[qy[inside], qx[inside]]
                tol = lr_tol + lr_rel * hstar
                ok = (hstar >= lo - tol) & (hstar <= hi + tol)
                proud |= cand & ok
            if lr_hyst > 0:
                n, lab, _, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8))
                cnt = np.bincount(lab[proud], minlength=n)
                keep = cnt >= lr_hyst
                keep[0] = False
                proud = keep[lab]
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


# --------------------------------------------------------------------------------------------
# Checking a layout record against the height map
# --------------------------------------------------------------------------------------------

def _depth_to_height(cam: Camera, depth: np.ndarray) -> np.ndarray:
    """Camera-z depth (0 = empty) -> height above TABLE_Z of the surface point (NaN where empty)."""
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    xc = (uu + 0.5 - cam.K[0, 2]) / cam.K[0, 0] * depth
    yc = (vv + 0.5 - cam.K[1, 2]) / cam.K[1, 1] * depth
    R, C = cam.T_wc[:3, :3], cam.T_wc[:3, 3]
    z = R[2, 0] * xc + R[2, 1] * yc + R[2, 2] * depth + C[2]
    return np.where(depth > 0, z - TABLE_Z, np.nan).astype(np.float32)


def object_heights(cam: Camera, items: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]):
    """items [(key, V_local, F, T_world)] -> (height above TABLE_Z of the front-most item's surface per pixel, NaN off
    the items; index of that item per pixel, -1 none). GL depth per item when available, else a CPU ray cast."""
    from .render import gl, silhouette, transform

    h, w = cam.h, cam.w
    best = np.full((h, w), np.inf, np.float32)
    idx = np.full((h, w), -1, np.int32)
    hgt = np.full((h, w), np.nan, np.float32)
    g = gl()
    if g is not None:
        for i, it in enumerate(items):
            d = g.render(cam, [it], depth=True)
            m = (d > 0) & (d < best)
            best[m] = d[m]
            idx[m] = i
        hgt = _depth_to_height(cam, np.where(np.isfinite(best), best, 0.0).astype(np.float32))
        return hgt, idx
    import trimesh

    vv, uu = np.mgrid[0:h, 0:w]
    for i, (_, V, F, T) in enumerate(items):
        Vw = transform(V, T)
        m = silhouette(cam, Vw, F) > 0
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        dirs = cam.rays(np.c_[xs + 0.5, ys + 0.5])
        mesh = trimesh.Trimesh(Vw, F, process=False)
        loc, ir, _ = mesh.ray.intersects_location(np.repeat(cam.center[None], len(xs), 0), dirs, multiple_hits=False)
        if len(ir) == 0:
            continue
        dist = np.linalg.norm(loc - cam.center, axis=1).astype(np.float32)
        py, px = ys[ir], xs[ir]
        closer = dist < best[py, px]
        py, px, dist, loc = py[closer], px[closer], dist[closer], loc[closer]
        best[py, px] = dist
        idx[py, px] = i
        hgt[py, px] = loc[:, 2] - TABLE_Z
    return hgt, idx


def _record_entries(record: dict):
    for typ in ("Rigid", "Geometry", "Articulation"):
        for cat, lst in (record.get(typ) or {}).items():
            for r in lst:
                yield typ, cat, r


def check_record(record: dict, images: dict, state: dict, cams: dict | None = None, robot_masks: dict | None = None,
                 sweep_kw: dict | None = None, dh_mm: float = 4.0, min_px: int = 8, blob_px: int = 30,
                 region=REGION) -> dict:
    """The height CHECK of a layout record: the head + wrist plane sweep of the record's own frame against each
    object's predicted surface (its mesh at its pose, front-most pixels only).

    Per object, ``_perception.stereo`` = {``dh_mm``: median of measured minus predicted height over its pixels with a
    measured height, ``h_px``, ``known`` / ``on_table`` / ``proud`` shares of its pixels, ``mismatch``: |dh| >
    ``dh_mm`` with >= ``min_px`` measured pixels, or (no measured height) its decided pixels lie on the table}. A
    mismatch says the object is not where the record puts it (a phantom on the bare table or on another object, a wrong
    resting pose, a stand 1-2 cm off). Proud blobs no object explains go to ``_perception.stereo_unexplained`` (world xy
    at their measured height, area): objects standing up where the record has none. Returns a summary."""
    from .cameras import cameras_from_state
    from .render import robot_mask

    t0 = time.time()
    cams = cams or cameras_from_state(state)
    views = {}
    for name, img in images.items():
        if name not in cams:
            continue
        rob = (robot_masks or {}).get(name)
        if rob is None:
            rob = robot_mask(cams[name], state, dilate_px=4)
        views[name] = View(img, cams[name], (1 - (rob > 0)).astype(np.uint8))
    if "cam_head" not in views or len(views) < 2:
        return {"skipped": "needs the head and a wrist view"}
    ref = views.pop("cam_head")
    if region is not None:
        ref.valid = ref.valid & table_region_mask(ref.cam, region)
    kw = {"agg": 5, "fill": True, "min_blob": 30, "lr": True, "scale": 1.0}
    kw.update(sweep_kw or {})
    sw = plane_sweep(ref, list(views.values()), **kw)
    cam = scaled_camera(ref.cam, sw.scale)
    summ = check_heights(record, cam, sw, dh_mm=dh_mm, min_px=min_px, blob_px=blob_px)
    summ.update({"views": [ref.cam.name] + list(views), "sweep_s": sw.timings["total_s"], "total_s": round(time.time() - t0, 3)})
    record["_perception"]["stereo_check"] = summ
    return summ


def check_heights(record: dict, cam: Camera, sw, dh_mm: float = 4.0, min_px: int = 8, blob_px: int = 30,
                  key: str = "stereo") -> dict:
    """The comparison half of ``check_record`` for any measured height map in ``cam`` (the plane sweep's, or a learned
    model's: ``mv3d``). ``sw`` needs ``height`` (m above TABLE_Z, NaN unknown), ``known``, ``on_table`` and ``proud``
    (uint8), all sized like ``cam``. Writes ``_perception[key]`` per object and ``_perception[key + "_unexplained"]``."""
    from .cameras import pose_to_T
    from .render import load_asset

    ents, items = [], []
    for typ, cat, r in _record_entries(record):
        akey = f"{typ}/{cat}/{int(r['category_idx']):05d}"
        try:
            a = load_asset(akey)
        except Exception:
            continue
        ents.append(r)
        items.append((akey, a.Vs, a.Fs, pose_to_T(r["default_pos"], r["default_ori"])))
    ph, pid = object_heights(cam, items)
    n_mis = 0
    for i, r in enumerate(ents):
        m = pid == i
        n = int(m.sum())
        out = {"px": n}
        if n:
            kn = m & (sw.known > 0)
            both = m & np.isfinite(sw.height) & np.isfinite(ph)
            out.update({"known": round(float(kn.sum() / n), 3),
                        "on_table": round(float((sw.on_table[kn] > 0).mean()), 3) if kn.any() else None,
                        "proud": round(float((sw.proud[kn] > 0).mean()), 3) if kn.any() else None,
                        "h_px": int(both.sum())})
            if both.sum() >= min_px:
                out["dh_mm"] = round(float(np.median(sw.height[both] - ph[both])) * 1000, 1)
                out["mismatch"] = bool(abs(out["dh_mm"]) > dh_mm)
            elif kn.sum() >= max(min_px, 0.3 * n):
                on = float((sw.on_table[kn] > 0).mean())
                pr = float((sw.proud[kn] > 0).mean())
                out["mismatch"] = bool(on >= 0.5 and pr < 0.2)
        n_mis += bool(out.get("mismatch"))
        r.setdefault("_perception", {})[key] = out
    expl = cv2.dilate((pid >= 0).astype(np.uint8), np.ones((11, 11), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats((sw.proud > 0).astype(np.uint8))
    unexpl = []
    for j in range(1, n):
        a = int(st[j, cv2.CC_STAT_AREA])
        cm = lab == j
        if a < blob_px or (expl[cm] > 0).mean() > 0.5:
            continue
        ys, xs = np.nonzero(cm)
        hs = sw.height[cm]
        hm = float(np.nanmedian(hs)) if np.isfinite(hs).any() else 0.02
        X = cam.ray_plane(np.array([[xs.mean() + 0.5, ys.mean() + 0.5]]), TABLE_Z + hm)[0]
        if np.isfinite(X).all():
            unexpl.append({"xy": [round(float(X[0]), 4), round(float(X[1]), 4)], "area_px": a,
                           "height_mm": round(hm * 1000, 1), "source": "stereo"})
    unexpl.sort(key=lambda u: -u["area_px"])
    p = record.setdefault("_perception", {})
    p[f"{key}_unexplained"] = unexpl[:8]
    return {"objects": len(ents), "mismatch": n_mis, "unexplained": len(unexpl)}
