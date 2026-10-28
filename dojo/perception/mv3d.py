"""Learned multi-view 3D from our calibrated cameras (VGGT / MASt3R), with OUR camera model in the loop.

The learned models predict depth (and their own cameras) up to a similarity; we know every camera exactly (head fixed,
wrists from FK, ``cameras.py``). So the useful output is each view's predicted depth, put into our world through our
own intrinsics and poses, with only the global scale (and, for diagnostics, the full similarity) taken from the model:

- ``align_sim3``: the model's cameras -> ours (rotation from the orientations, scale + translation from the centres);
  its residual says how well the model recovered our rig;
- ``depth_scale_table``: an alternative per-view scale from the table plane (the mode of known-plane depth / predicted
  depth over the table region): the table is most of every view, and its depth is known exactly;
- ``unproject``: a view's depth -> world points through our camera (OpenCV, pixel i covers [i, i+1));
- ``height_in``: the height above TABLE_Z each pixel of a camera sees, from the points of all views (z-buffered splat);
- ``fuse_topdown``: a top-down max-height grid over the object region from all views' points.

Model wrappers (``VGGTRunner``, ``Mast3rRunner``) import their repos lazily: torch and the model code live only in
the venv that runs them (``DOJO_MV_EXT`` = the directory holding the ``vggt`` / ``mast3r`` clones).

    run = VGGTRunner(weights_dir)
    out = run.infer([img_head, img_lw, img_rw])           # depth (S,h,w), conf, T_cw (S,4,4), K (S,3,3), timings
    s, R, t, res = align_sim3(out["T_cw"], [c.T_wc for c in cams])
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .cameras import TABLE_Z, Camera
from .stereo import REGION, scaled_camera, table_region_mask

# 640x480 -> 504x378 keeps the aspect exactly (scale 0.7875) and is a multiple of the ViT patch (14)
MODEL_W, MODEL_H = 504, 378


def model_camera(cam: Camera, w: int = MODEL_W) -> Camera:
    return scaled_camera(cam, w / cam.w)


def _ext_path(name: str) -> None:
    root = os.environ.get("DOJO_MV_EXT", os.path.expanduser("~/controlr-mv/ext"))
    for p in (os.path.join(root, name), os.path.join(root, name, "dust3r")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _vram_reset():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _vram_peak_gb() -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0


@dataclass
class VGGTRunner:
    """facebook/VGGT-1B. ``weights``: a directory with config.json + model.safetensors (bf16 is fine)."""
    weights: str
    device: str = "cuda"
    size: tuple[int, int] = (MODEL_W, MODEL_H)
    bf16_trunk: bool = False  # the aggregator (~0.9 B params) held in bf16; the heads stay fp32 on fp32 tokens

    def __post_init__(self):
        import torch

        _ext_path("vggt")
        from vggt.models.vggt import VGGT

        t0 = time.time()
        m = VGGT.from_pretrained(self.weights)
        m.track_head = None  # tracks are not used: saves memory and time
        if self.bf16_trunk:
            m.aggregator.to(torch.bfloat16)

            def tokens_fp32(mod, args, kwargs):
                return ([t.float() if t is not None else None for t in args[0]],) + tuple(args[1:]), kwargs
            for h in (m.camera_head, m.depth_head, m.point_head):
                if h is not None:
                    h.register_forward_pre_hook(tokens_fp32, with_kwargs=True)
        self.model = m.to(self.device).eval()
        self.dtype = torch.bfloat16
        self.load_s = round(time.time() - t0, 1)

    def infer(self, images: list[np.ndarray]) -> dict:
        import torch

        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        w, h = self.size
        x = np.stack([cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) for im in images]).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        _vram_reset()
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype):
            p = self.model(x)
        torch.cuda.synchronize()
        dt = time.time() - t0
        extri, intri = pose_encoding_to_extri_intri(p["pose_enc"], (h, w))
        T_cw = np.tile(np.eye(4), (len(images), 1, 1))
        T_cw[:, :3, :4] = extri[0].float().cpu().numpy()
        return {"depth": p["depth"][0, ..., 0].float().cpu().numpy(), "conf": p["depth_conf"][0].float().cpu().numpy(),
                "points": p["world_points"][0].float().cpu().numpy(), "points_conf": p["world_points_conf"][0].float().cpu().numpy(),
                "T_cw": T_cw, "K": intri[0].float().cpu().numpy(),
                "timings": {"infer_s": round(dt, 3), "vram_peak_gb": _vram_peak_gb(), "n_views": len(images)}}


@dataclass
class Mast3rRunner:
    """naver/MASt3R (ViT-L, 512, metric). Pairwise: dense pointmaps (view 1's depth in its own frame, metric) and
    24-d descriptors; ``matches`` = fast reciprocal nearest neighbours of the descriptors per pair, which we
    triangulate with OUR cameras (``triangulate``)."""
    weights: str
    device: str = "cuda"
    width: int = 512

    def __post_init__(self):
        _ext_path("mast3r")
        from mast3r.model import AsymmetricMASt3R

        t0 = time.time()
        self.model = AsymmetricMASt3R.from_pretrained(self.weights).to(self.device).eval()
        self.load_s = round(time.time() - t0, 1)

    def _view(self, img: np.ndarray, i: int) -> dict:
        import torch

        h = int(round(img.shape[0] * self.width / img.shape[1]))
        x = cv2.resize(img, (self.width, h), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(x).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)[None]
        return {"img": t, "true_shape": np.int32([[h, self.width]]), "idx": i, "instance": str(i)}

    def infer(self, images: list[np.ndarray], pairs: list[tuple[int, int]], subsample: int = 4, batch: int = 4) -> dict:
        """Per ordered pair (i, j): view i's metric depth + conf, and reciprocal matches (xy_i, xy_j) with the
        descriptor confidences. Pixel coordinates are integer indices at the model resolution."""
        import torch

        from dust3r.inference import inference
        from mast3r.fast_nn import fast_reciprocal_NNs

        views = [self._view(im, i) for i, im in enumerate(images)]
        _vram_reset()
        t0 = time.time()
        with torch.no_grad():
            out = inference([(views[i], views[j]) for i, j in pairs], self.model, self.device, batch_size=batch, verbose=False)
        t1 = time.time()
        p1, p2 = out["pred1"], out["pred2"]
        res = {"pairs": pairs, "depth": {}, "conf": {}, "matches": {}}
        for k, (i, j) in enumerate(pairs):
            res["depth"][(i, j)] = p1["pts3d"][k, ..., 2].float().numpy()
            res["conf"][(i, j)] = p1["conf"][k].float().numpy()
            d1, d2 = p1["desc"][k].to(self.device), p2["desc"][k].to(self.device)
            xy1, xy2 = fast_reciprocal_NNs(d1, d2, subsample_or_initxy1=subsample, device=self.device, dist="dot", block_size=2**13)
            c1 = p1["desc_conf"][k].float().numpy()[xy1[:, 1], xy1[:, 0]]
            c2 = p2["desc_conf"][k].float().numpy()[xy2[:, 1], xy2[:, 0]]
            res["matches"][(i, j)] = (xy1.astype(np.int16), xy2.astype(np.int16), np.minimum(c1, c2).astype(np.float16))
        torch.cuda.synchronize()
        res["timings"] = {"infer_s": round(t1 - t0, 3), "match_s": round(time.time() - t1, 3), "vram_peak_gb": _vram_peak_gb(),
                          "n_views": len(images), "n_pairs": len(pairs)}
        return res


@dataclass
class MapAnythingRunner:
    """facebook/map-anything with OUR calibration as input: every view gets its intrinsics and its metric camera pose
    (world = ours), so the model only has to predict depth consistent with a known rig."""
    weights: str
    device: str = "cuda"
    size: tuple[int, int] = (MODEL_W, MODEL_H)
    use_poses: bool = True

    def __post_init__(self):
        _ext_path("map-anything")
        from mapanything.models import MapAnything

        t0 = time.time()
        self.model = MapAnything.from_pretrained(self.weights).to(self.device).eval()
        self.load_s = round(time.time() - t0, 1)

    def infer(self, images: list[np.ndarray], cams: list[Camera]) -> dict:
        import torch

        w, h = self.size
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        views = []
        for im, c in zip(images, cams):
            mc = model_camera(c, w)
            x = (cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 - mean) / std
            K = mc.K.copy()
            K[0, 2] -= 0.5  # MapAnything puts pixel centres at integer coordinates
            K[1, 2] -= 0.5
            v = {"img": torch.from_numpy(x).permute(2, 0, 1)[None], "data_norm_type": ["dinov2"],
                 "intrinsics": torch.from_numpy(K).float()[None]}
            if self.use_poses:
                v["camera_poses"] = torch.from_numpy(mc.T_wc).float()[None]
                v["is_metric_scale"] = torch.tensor([True])
            views.append(v)
        _vram_reset()
        t0 = time.time()
        with torch.no_grad():
            preds = self.model.infer(views, memory_efficient_inference=False, use_amp=True, amp_dtype="bf16",
                                     apply_mask=False, mask_edges=False)
        torch.cuda.synchronize()
        dt = time.time() - t0
        depth = np.stack([p["depth_z"][0, ..., 0].float().cpu().numpy() for p in preds])
        conf = np.stack([p["conf"][0].float().cpu().numpy() for p in preds])
        T_wc = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in preds])
        return {"depth": depth, "conf": conf, "T_cw": np.stack([np.linalg.inv(T) for T in T_wc]),
                "K": np.stack([p["intrinsics"][0].float().cpu().numpy() for p in preds]),
                "timings": {"infer_s": round(dt, 3), "vram_peak_gb": _vram_peak_gb(), "n_views": len(images)}}


def triangulate(cam1: Camera, cam2: Camera, uv1: np.ndarray, uv2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint triangulation of continuous pixel coordinates (N, 2) in two known cameras. Returns (world points,
    max reprojection error in px over both views, ray angle in degrees)."""
    d1, d2 = cam1.rays(uv1), cam2.rays(uv2)
    o1, o2 = cam1.center, cam2.center
    b = o2 - o1
    a11 = (d1 * d1).sum(1)
    a12 = (d1 * d2).sum(1)
    a22 = (d2 * d2).sum(1)
    r1, r2 = d1 @ b, d2 @ b
    den = a11 * a22 - a12 ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (r1 * a22 - r2 * a12) / den
        t2 = (r1 * a12 - r2 * a11) / den
    X = 0.5 * ((o1 + t1[:, None] * d1) + (o2 + t2[:, None] * d2))
    q1, z1 = cam1.project(X)
    q2, z2 = cam2.project(X)
    err = np.maximum(np.linalg.norm(q1 - uv1, axis=1), np.linalg.norm(q2 - uv2, axis=1))
    err = np.where((z1 > 0) & (z2 > 0) & (t1 > 0) & (t2 > 0), err, np.inf)
    ang = np.degrees(np.arccos(np.clip(a12 / np.sqrt(a11 * a22), -1, 1)))
    return X, err, ang


# --------------------------------------------------------------------------------------------
# Geometry with our cameras
# --------------------------------------------------------------------------------------------

def _rot_mean(Rs: list[np.ndarray]) -> np.ndarray:
    """Chordal L2 mean of rotations."""
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def align_sim3(pred_T_cw: np.ndarray, known_T_wc: list[np.ndarray]) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """Similarity (s, R, t) mapping the model's world into ours: X_ours = s R X_model + t.

    R from the cameras' orientations (each view i gives R_wc_ours_i R_wc_model_i^T), then s and t by least squares on
    the centres with R fixed. Returns the residuals too: centre error (mm) and orientation error (deg) per view, and the
    plain centre-only Umeyama fit's residual for comparison."""
    pred_T_wc = [np.linalg.inv(T) for T in pred_T_cw]
    Rs = [Tk[:3, :3] @ Tp[:3, :3].T for Tk, Tp in zip(known_T_wc, pred_T_wc)]
    R = _rot_mean(Rs)
    Cp = np.array([T[:3, 3] for T in pred_T_wc])
    Ck = np.array([T[:3, 3] for T in known_T_wc])
    A = Cp @ R.T  # R C_p
    Am, Km = A.mean(0), Ck.mean(0)
    den = float(((A - Am) ** 2).sum())
    s = float(((A - Am) * (Ck - Km)).sum() / den) if den > 1e-12 else 1.0
    t = Km - s * Am
    C_fit = s * A + t
    pos_mm = np.linalg.norm(C_fit - Ck, axis=1) * 1000
    rot_deg = []
    for Tk, Tp in zip(known_T_wc, pred_T_wc):
        dR = (R @ Tp[:3, :3]).T @ Tk[:3, :3]
        rot_deg.append(float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))))
    res = {"pos_mm": [round(float(v), 1) for v in pos_mm], "rot_deg": [round(v, 2) for v in rot_deg],
           "pos_rms_mm": round(float(np.sqrt((pos_mm ** 2).mean())), 1), "rot_max_deg": round(max(rot_deg), 2)}
    if len(Ck) >= 3:
        su, Ru, tu = umeyama(Cp, Ck)
        ru = np.linalg.norm((su * Cp @ Ru.T + tu) - Ck, axis=1) * 1000
        res["umeyama_centres_rms_mm"] = round(float(np.sqrt((ru ** 2).mean())), 1)
        res["umeyama_scale"] = round(float(su), 5)
    return s, R, t, res


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    C = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var) if var > 1e-12 else 1.0
    return s, R, mu_d - s * R @ mu_s


def pixel_rays(cam: Camera) -> np.ndarray:
    """(h, w, 3) camera-frame rays with z = 1 for every pixel centre (pixel i covers [i, i+1))."""
    vv, uu = np.mgrid[0:cam.h, 0:cam.w].astype(np.float64)
    x = (uu + 0.5 - cam.K[0, 2]) / cam.K[0, 0]
    y = (vv + 0.5 - cam.K[1, 2]) / cam.K[1, 1]
    return np.stack([x, y, np.ones_like(x)], -1)


def unproject(depth: np.ndarray, cam: Camera) -> np.ndarray:
    """Camera-z depth (h, w) -> world points (h, w, 3) through ``cam`` (sized like depth)."""
    Xc = pixel_rays(cam) * depth[..., None]
    return Xc @ cam.T_wc[:3, :3].T + cam.T_wc[:3, 3]


def table_depth(cam: Camera, z: float = TABLE_Z) -> np.ndarray:
    """Camera-z depth of the plane z per pixel (NaN where the ray misses it)."""
    d = pixel_rays(cam) @ cam.T_wc[:3, :3].T  # world directions with camera-z = 1
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z - cam.T_wc[2, 3]) / d[..., 2]
    return np.where(t > 0, t, np.nan)


def depth_scale_table(depth: np.ndarray, cam: Camera, valid: np.ndarray | None = None, region=REGION,
                      bins: int = 400) -> tuple[float, float]:
    """Per-view scale s so that s * depth puts the table where it is: the mode of (known table depth / predicted depth)
    over the region's pixels (objects occupy a minority, so the table dominates the histogram). Returns (s, share of
    pixels within 1 % of the mode)."""
    td = table_depth(cam)
    m = table_region_mask(cam, region) > 0
    if valid is not None:
        m &= valid > 0
    m &= np.isfinite(td) & (depth > 1e-6)
    if m.sum() < 100:
        return float("nan"), 0.0
    r = np.log(td[m] / depth[m])
    lo, hi = np.percentile(r, [2, 98])
    hist, edges = np.histogram(r, bins=bins, range=(lo - 1e-6, hi + 1e-6))
    k = int(np.argmax(hist))
    c = 0.5 * (edges[k] + edges[k + 1])
    near = np.abs(r - c) < 0.01
    s = float(np.exp(np.median(r[near]))) if near.any() else float(np.exp(c))
    return s, round(float(near.mean()), 3)


def depth_correct_table(depth: np.ndarray, cam: Camera, valid: np.ndarray | None = None, region=REGION,
                        order: int = 2, iters: int = 4) -> tuple[np.ndarray, float]:
    """A smooth per-pixel scale field so that the table lands on the known plane: log(known table depth / predicted
    depth) over the region's table pixels, fitted by a polynomial of ``order`` in (u, v) with outlier rejection
    (objects stand closer than the plane, so they are outliers on one side). Corrects a learned model's low-frequency
    depth warp, not just its global scale. Returns (field (h, w): multiply the depth by it, share of inliers)."""
    h, w = depth.shape
    s0, _ = depth_scale_table(depth, cam, valid, region)
    if not np.isfinite(s0):
        return np.full((h, w), np.nan), 0.0
    td = table_depth(cam)
    m = table_region_mask(cam, region) > 0
    if valid is not None:
        m &= valid > 0
    m &= np.isfinite(td) & (depth > 1e-6)
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float64)
    un, vn = (uu + 0.5) / w - 0.5, (vv + 0.5) / h - 0.5
    terms = [np.ones_like(un)]
    for k in range(1, order + 1):
        for j in range(k + 1):
            terms.append(un ** (k - j) * vn ** j)
    A = np.stack(terms, -1)
    r = np.full((h, w), np.nan)
    r[m] = np.log(td[m] / depth[m])
    sel = m & (np.abs(r - np.log(s0)) < 0.03)
    c = None
    for it, thr in enumerate(np.linspace(0.02, 0.006, iters)):
        if sel.sum() < 50:
            break
        c, *_ = np.linalg.lstsq(A[sel], r[sel], rcond=None)
        res = r - A @ c
        sel = m & (np.abs(res) < thr)
    if c is None:
        return np.full((h, w), s0), 0.0
    return np.exp(A @ c), round(float(sel.sum() / max(m.sum(), 1)), 3)


def height_in(cam: Camera, points: list[np.ndarray], masks: list[np.ndarray] | None = None) -> np.ndarray:
    """Height above TABLE_Z of the nearest point each pixel of ``cam`` sees, from world points of any views
    (z-buffered nearest-pixel splat; NaN = no point). For the reference view's own points this is just its height."""
    return splat(cam, points, masks)[0]


def splat(cam: Camera, points: list[np.ndarray], masks: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(height above TABLE_Z, camera-z depth) of the nearest point per pixel of ``cam`` (NaN = no point)."""
    h, w = cam.h, cam.w
    best = np.full(h * w, np.inf)
    hgt = np.full(h * w, np.nan)
    for i, P in enumerate(points):
        X = P.reshape(-1, 3)
        ok = np.isfinite(X).all(1)
        if masks is not None and masks[i] is not None:
            ok &= masks[i].reshape(-1) > 0
        X = X[ok]
        uv, z = cam.project(X)
        u = np.floor(uv[:, 0]).astype(np.int64)
        v = np.floor(uv[:, 1]).astype(np.int64)
        ins = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = v[ins] * w + u[ins]
        zz = z[ins]
        hz = X[ins, 2] - TABLE_Z
        order = np.argsort(-zz)  # far first, near last: the last write wins
        idx, zz, hz = idx[order], zz[order], hz[order]
        closer = zz < best[idx]
        best[idx[closer]] = zz[closer]
        hgt[idx[closer]] = hz[closer]
    return hgt.reshape(h, w), np.where(np.isfinite(best), best, np.nan).reshape(h, w)


@dataclass
class TopDown:
    height: np.ndarray   # (ny, nx) max height above TABLE_Z per cell (NaN = no point)
    count: np.ndarray    # (ny, nx) points per cell
    x0: float
    y0: float
    cell: float

    def cell_of(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.floor((xy[:, 0] - self.x0) / self.cell).astype(np.int64)
        iy = np.floor((xy[:, 1] - self.y0) / self.cell).astype(np.int64)
        return ix, iy


def fuse_topdown(points: list[np.ndarray], masks: list[np.ndarray] | None = None, cell: float = 0.004, region=REGION,
                 h_max: float = 0.45, h_min: float = -0.03, q: float | None = None) -> TopDown:
    """All views' world points binned over the region's xy: per cell the max height (or the ``q`` quantile)."""
    (x0, x1), (y0, y1) = region
    nx, ny = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
    allp = []
    for i, P in enumerate(points):
        X = P.reshape(-1, 3)
        ok = np.isfinite(X).all(1)
        if masks is not None and masks[i] is not None:
            ok &= masks[i].reshape(-1) > 0
        allp.append(X[ok])
    X = np.concatenate(allp) if allp else np.zeros((0, 3))
    hz = X[:, 2] - TABLE_Z
    ix = np.floor((X[:, 0] - x0) / cell).astype(np.int64)
    iy = np.floor((X[:, 1] - y0) / cell).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (hz < h_max) & (hz > h_min)
    ix, iy, hz = ix[ok], iy[ok], hz[ok]
    flat = iy * nx + ix
    cnt = np.bincount(flat, minlength=nx * ny)
    if q is None:
        hm = np.full(nx * ny, -np.inf)
        np.maximum.at(hm, flat, hz)
    else:
        hm = np.full(nx * ny, np.nan)
        order = np.lexsort((hz, flat))
        f2, h2 = flat[order], hz[order]
        starts = np.r_[0, np.flatnonzero(np.diff(f2)) + 1]
        ends = np.r_[starts[1:], len(f2)]
        for a, b in zip(starts, ends):
            hm[f2[a]] = h2[a + int(q * (b - a - 1))]
    hm = np.where(cnt > 0, hm, np.nan)
    return TopDown(hm.reshape(ny, nx), cnt.reshape(ny, nx), x0, y0, cell)


# --------------------------------------------------------------------------------------------
# Consumer: raised-but-unexplained regions -> candidates fitted by B1's render-and-compare
# --------------------------------------------------------------------------------------------

def height_sweep_like(height: np.ndarray, valid: np.ndarray, h_fg: float = 0.012, h_table: float = 0.005):
    """A measured height map (m above TABLE_Z, NaN unknown) in the fields ``stereo.check_heights`` reads."""
    from types import SimpleNamespace

    known = (valid > 0) & np.isfinite(height)
    with np.errstate(invalid="ignore"):
        return SimpleNamespace(height=np.where(known, height, np.nan).astype(np.float32), known=known.astype(np.uint8),
                               proud=(known & (height > h_fg)).astype(np.uint8),
                               on_table=(known & (np.abs(height) < h_table)).astype(np.uint8))


def _rest_height(sm) -> float:
    V = sm.asset.V @ sm.R0.T
    return float(V[:, 2].max() - V[:, 2].min())


def _entries(record: dict):
    for typ in ("Rigid", "Geometry", "Articulation"):
        for cat, lst in (record.get(typ) or {}).items():
            for r in lst:
                yield typ, cat, r


def _others_masks(record: dict, views: dict, skip=None) -> dict:
    """Per view: the union silhouette of the record's objects (``skip``: an entry to leave out)."""
    from .cameras import pose_to_T
    from .refine import _shape_for

    out = {n: np.zeros((v.cam.h, v.cam.w), np.uint8) for n, v in views.items()}
    for typ, cat, r in _entries(record):
        if r is skip:
            continue
        try:
            T = pose_to_T(r["default_pos"], r["default_ori"])
            sm = _shape_for(f"{typ}/{cat}/{int(r['category_idx']):05d}", T[:3, :3], None)
        except Exception:
            continue
        for n, v in views.items():
            sm.sil(v.cam, T, out=out[n])
    return out


def _fit_at(keys: list[str], x0: float, y0: float, views: dict, others: dict, n_yaw: int = 8, z_off: float = 0.0):
    """B1's render-and-compare (``inverse_layout.refine``) from a seed point over candidate assets x resting poses x
    yaw starts. Returns (loss, key, shape, x, y, yaw) of the best or None."""
    from .inverse_layout import refine as nm_refine, shape_models

    best = None
    for key in keys:
        try:
            sms = shape_models(key)
        except Exception:
            continue
        for sm in sms:
            per = sm.yaw_period
            yaws = [k * per / n_yaw for k in range(n_yaw)] if per > 0.2 else [0.0]
            for yaw in yaws:
                c_off = sm.pose(0, 0, yaw)[:3, :3] @ ((sm.asset.V.min(0) + sm.asset.V.max(0)) / 2)
                try:
                    x, y, yy, loss, _ = nm_refine(sm, x0 - c_off[0], y0 - c_off[1], yaw, views, others, xy_step=0.006,
                                                  yaw_step=0.15, maxiter=150, fix_yaw=per < 0.2, z_off=z_off)
                except Exception:
                    continue
                if best is None or loss < best[0]:
                    best = (float(loss), key, sm, x, y, yy)
    return best


def propose_unexplained(record: dict, views: dict, height: np.ndarray, task: str, valid: np.ndarray | None = None,
                        min_area_px: int = 150, max_blobs: int = 6, k_cands: int = 3, max_loss: float = 0.45,
                        cap_mm: float = 40.0, clear_mm: float = 30.0, suspect_dh_mm: float = 15.0,
                        move_margin: float = 0.05, n_yaw: int = 8, fg_add: bool = True) -> list[dict]:
    """Raised regions of a measured head height map (``height``, full resolution, m above TABLE_Z) that no object of
    ``record`` explains become candidates: the record's missing labels (in the task's label groups, none of the group
    placed) and its height-check suspects (|dh| > ``suspect_dh_mm``: placed where the surface disagrees) are fitted at
    each blob by B1's render-and-compare (``views``: B1's segmented views; ``fg_add`` adds the blob's raised pixels to
    the head foreground, so a low-contrast object still has a silhouette). Accepted fits (loss <= ``max_loss``, within
    ``cap_mm`` of the blob, no other object within ``clear_mm``; a suspect only if it beats its own pose's loss by
    ``move_margin``) are written into ``record`` (``_perception.mv3d_proposed``). Returns the per-blob log."""
    import cv2

    from .cameras import R_to_quat_wxyz, pose_to_T
    from .inverse_layout import label_groups, make_roi_views, roi_loss, shape_models
    from .refine import _shape_for
    from .stereo import check_heights

    head = views["cam_head"]
    cam = head.cam
    valid = np.ones(height.shape, bool) if valid is None else valid > 0
    check_heights(record, cam, height_sweep_like(height, valid), key="mv3d", blob_px=min_area_px)
    blobs = list((record.get("_perception") or {}).get("mv3d_unexplained", []))[:max_blobs]
    if not blobs:
        return []
    present = {r.get("label") for _, _, r in _entries(record)}
    missing = []
    for g in label_groups(task):
        for lab in g["labels"]:
            if lab not in present:
                missing.append({"label": lab, "keys": list(g["candidates"]), "type": g["type"], "entry": None})
    suspects = []
    for typ, cat, r in _entries(record):
        st = (r.get("_perception") or {}).get("mv3d") or {}
        if st.get("dh_mm") is not None and abs(st["dh_mm"]) > suspect_dh_mm:
            suspects.append({"label": r.get("label"), "keys": [f"{typ}/{cat}/{int(r['category_idx']):05d}"], "type": typ,
                             "entry": r})
    cands = missing + suspects
    if not cands:
        return []
    raised = valid & np.isfinite(height) & (np.nan_to_num(height) > 0.012)
    expl = cv2.dilate(_others_masks(record, {"cam_head": head})["cam_head"], np.ones((11, 11), np.uint8)) > 0
    if fg_add:
        head.fg = (head.fg.astype(bool) | (raised & ~expl)).astype(np.uint8)
        head.known = (head.known.astype(bool) | (raised & ~expl)).astype(np.uint8)
    log, used = [], set()
    for b in blobs:
        bh = max(float(b.get("height_mm") or 20.0) / 1000, 0.008)
        x0, y0 = b["xy"]

        def rank(c):
            hs = []
            for k in c["keys"][:12]:
                try:
                    hs += [_rest_height(sm) for sm in shape_models(k)]
                except Exception:
                    pass
            return min((abs(np.log(bh / max(h, 1e-3))) for h in hs), default=9.0)
        pool = sorted([c for c in cands if id(c) not in used], key=rank)[:k_cands]
        entry_log = {"xy": b["xy"], "area_px": b["area_px"], "height_mm": b.get("height_mm"), "tried": []}
        best = None
        for c in pool:
            others = _others_masks(record, views, skip=c["entry"])
            fit = _fit_at(c["keys"][:4], x0, y0, views, others, n_yaw=n_yaw)
            if fit is None:
                continue
            loss, key, sm, x, y, yy = fit
            T = sm.pose(x, y, yy)
            ctr = T[:3, :3] @ ((sm.asset.V.min(0) + sm.asset.V.max(0)) / 2) + T[:3, 3]
            shift = float(np.hypot(ctr[0] - x0, ctr[1] - y0)) * 1000
            near = [r for _, _, r in _entries(record) if r is not c["entry"]
                    and np.hypot(r["default_pos"][0] - ctr[0], r["default_pos"][1] - ctr[1]) * 1000 < clear_mm]
            ok = loss <= max_loss and shift <= cap_mm and not near
            l0 = None
            if c["entry"] is not None:  # a suspect moves only if its new pose explains the views better
                r = c["entry"]
                T0 = pose_to_T(r["default_pos"], r["default_ori"])
                sm0 = _shape_for(c["keys"][0], T0[:3, :3], None)
                rv = make_roi_views(sm0, T0, views, others)
                l0 = float(roi_loss(sm0, T0, rv)[0]) if rv else 1.0
                ok = ok and loss < l0 - move_margin
            entry_log["tried"].append({"label": c["label"], "key": key, "loss": round(loss, 3), "shift_mm": round(shift, 1),
                                       "near": len(near), "ok": bool(ok), "suspect": c["entry"] is not None,
                                       "loss_before": None if l0 is None else round(l0, 3),
                                       "pos": [round(float(v), 5) for v in T[:3, 3]],
                                       "ori": [round(float(v), 6) for v in R_to_quat_wxyz(T[:3, :3])]})
            if ok and (best is None or loss < best[0]):
                best = (loss, c, key, T)
        if best is not None:
            loss, c, key, T = best
            used.add(id(c))
            typ, cat, idx = key.split("/")
            pos = [round(float(v), 5) for v in T[:3, 3]]
            ori = [round(float(v), 6) for v in R_to_quat_wxyz(T[:3, :3])]
            info = {"source": "mv3d_propose", "loss": round(loss, 3), "blob_xy": b["xy"]}
            if c["entry"] is not None:
                c["entry"].update(default_pos=pos, default_ori=ori)
                c["entry"].setdefault("_perception", {})["mv3d_proposed"] = {**info, "moved": True}
            else:
                record.setdefault(typ, {}).setdefault(cat, []).append(
                    {"label": c["label"], "category": cat, "category_idx": int(idx), "default_pos": pos, "default_ori": ori,
                     "_perception": {"mv3d_proposed": info}})
                record["_not_found"] = [n for n in record.get("_not_found", []) if n.get("label") != c["label"]]
            entry_log["accepted"] = {"label": c["label"], "key": key, "loss": round(loss, 3), "moved": c["entry"] is not None}
        log.append(entry_log)
    record.setdefault("_perception", {})["mv3d_propose"] = log
    return log


# --------------------------------------------------------------------------------------------
# Runtime: one GPU server (the model), CPU clients (the estimator workers)
# --------------------------------------------------------------------------------------------

ENV_ADDR = "DOJO_MV3D_ADDR"


def corrected_depths(depth: np.ndarray, cams: list[Camera], valids: list[np.ndarray], mode: str = "poly") -> list[np.ndarray]:
    """Per view: the model's depth (model resolution) times its table fit (``poly``: ``depth_correct_table``; ``table``:
    one scale), NaN where unusable. ``cams`` at the depth's resolution."""
    out = []
    for d, c, v in zip(depth, cams, valids):
        if mode == "poly":
            f, _ = depth_correct_table(d, c, v)
        else:
            f, _ = depth_scale_table(d, c, v)
        D = d * f
        out.append(np.where((v > 0) & np.isfinite(D) & (D > 0), D, np.nan).astype(np.float32))
    return out


def head_height(depths: list[np.ndarray], cams: list[Camera], head: Camera) -> np.ndarray:
    """The fused height map (m above TABLE_Z) in the full-resolution head camera ``head``: every view's corrected depth
    through our cameras, z-buffered into the head at the model resolution, nearest-upsampled."""
    import cv2

    pts = [unproject(np.nan_to_num(d), c) for d, c in zip(depths, cams)]
    msk = [np.isfinite(d) for d in depths]
    hm = height_in(model_camera(head, depths[0].shape[1]), pts, msk)
    return cv2.resize(hm.astype(np.float32), (head.w, head.h), interpolation=cv2.INTER_NEAREST)


class Mv3dClient:
    """Talks to ``serve`` over a unix socket (one per process, ``client()``). ``depth(images, cams, valids)`` -> the
    corrected per-view depths at the model resolution + timings; the geometry after it stays in the caller."""

    def __init__(self, addr: str):
        from multiprocessing.connection import Client

        self.addr = addr
        self.conn = Client(addr, family="AF_UNIX")

    def depth(self, images: list[np.ndarray], cams: list[Camera], valids: list[np.ndarray], mode: str = "poly") -> dict:
        req = {"op": "depth", "mode": mode, "images": [np.ascontiguousarray(i, np.uint8) for i in images],
               "cams": [(c.K.tolist(), c.T_wc.tolist(), c.w, c.h) for c in cams],
               "valids": [np.ascontiguousarray(v, np.uint8) for v in valids]}
        self.conn.send(req)
        r = self.conn.recv()
        if "error" in r:
            raise RuntimeError(f"mv3d server: {r['error']}")
        return {"depths": [d.astype(np.float32) for d in r["depths"]], "timings": r.get("timings", {})}

    def info(self) -> dict:
        self.conn.send({"op": "info"})
        return self.conn.recv()


_CLIENT: dict = {}


def client() -> Mv3dClient | None:
    """The process's client, or None (``DOJO_MV3D_ADDR`` unset, or no server: the caller keeps the plane sweep)."""
    addr = os.environ.get(ENV_ADDR, "")
    if not addr:
        return None
    c = _CLIENT.get(addr)
    if c is None:
        try:
            c = Mv3dClient(addr)
        except Exception as e:  # pragma: no cover - depends on a live server
            import warnings

            warnings.warn(f"mv3d server at {addr} unreachable ({e}); learned heights off")
            c = False
        _CLIENT[addr] = c
    return c or None


def learned_head_height(images: dict, cams: dict, robot_masks: dict | None = None, mode: str = "poly",
                        cl: Mv3dClient | None = None) -> tuple[np.ndarray, dict] | None:
    """The fused learned height map in the full-resolution head camera for a frame's views (head first), or None without
    a server. ``robot_masks``: per camera, the robot's silhouette (excluded from the table fit and the points)."""
    from .stereo import REGION, table_region_mask

    cl = cl or client()
    if cl is None or "cam_head" not in images:
        return None
    names = ["cam_head"] + [n for n in images if n != "cam_head" and n in cams]
    t0 = time.time()
    mcams = [model_camera(cams[n]) for n in names]
    valids = []
    for n, mc in zip(names, mcams):
        v = np.ones((mc.h, mc.w), np.uint8)
        rob = (robot_masks or {}).get(n)
        if rob is not None:
            v &= (cv2.resize((rob > 0).astype(np.uint8), (mc.w, mc.h), interpolation=cv2.INTER_NEAREST) == 0).astype(np.uint8)
        valids.append(v)
    r = cl.depth([images[n] for n in names], mcams, valids, mode)
    hm = head_height(r["depths"], mcams, cams["cam_head"])
    hm[table_region_mask(cams["cam_head"], REGION) == 0] = np.nan
    return hm, {**r["timings"], "views": names, "client_s": round(time.time() - t0, 3)}


class _ServerModel:
    def __init__(self, weights: str, bf16_trunk: bool = True):
        self.run = VGGTRunner(weights, bf16_trunk=bf16_trunk)

    def handle(self, req: dict) -> dict:
        cams = [Camera(f"v{i}", np.array(K), np.array(T), int(w), int(h)) for i, (K, T, w, h) in enumerate(req["cams"])]
        t0 = time.time()
        p = self.run.infer(req["images"])
        D = corrected_depths(p["depth"], cams, req["valids"], req.get("mode", "poly"))
        return {"depths": [d.astype(np.float16) for d in D],
                "timings": {**p["timings"], "fit_s": round(time.time() - t0 - p["timings"]["infer_s"], 3)}}

    def info(self) -> dict:
        import torch

        return {"model": "vggt", "vram_gb": round(torch.cuda.memory_allocated() / 2**30, 2)}


def serve(sock: str, weights: str | None = None, max_hours: float = 6.0, M=None, bf16_trunk: bool = True) -> None:
    """Answer depth requests on a unix socket until killed (or ``max_hours``); one thread per client, the GPU used by
    one request at a time. ``M``: a model with ``handle(req)`` / ``info()`` (tests pass a fake)."""
    import threading
    from multiprocessing.connection import Listener

    M = M or _ServerModel(weights, bf16_trunk=bf16_trunk)
    lock = threading.Lock()
    if os.path.exists(sock):
        os.unlink(sock)
    lst = Listener(sock, family="AF_UNIX")
    os.chmod(sock, 0o600)
    t_end = time.time() + max_hours * 3600

    def run(conn):
        try:
            while True:
                req = conn.recv()
                if req.get("op") == "info":
                    conn.send(M.info())
                    continue
                try:
                    with lock:
                        conn.send(M.handle(req))
                except Exception as e:  # report, keep serving
                    import traceback

                    conn.send({"error": f"{type(e).__name__}: {e} | {traceback.format_exc()[-400:]}"})
        except (EOFError, ConnectionResetError, BrokenPipeError):
            pass

    while time.time() < t_end:
        conn = lst.accept()
        threading.Thread(target=run, args=(conn,), daemon=True).start()


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--sock", required=True)
    s.add_argument("--weights", default=os.path.expanduser("~/dojo-data/models/mv/vggt-1b"))
    s.add_argument("--max-hours", type=float, default=6.0)
    s.add_argument("--fp32", action="store_true", help="keep the aggregator in fp32 (default: bf16 trunk)")
    a = ap.parse_args(argv)
    serve(os.path.expanduser(a.sock), a.weights, a.max_hours, bf16_trunk=not a.fp32)


if __name__ == "__main__":
    main()
