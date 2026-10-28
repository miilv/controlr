"""Multi-view feature-metric scoring of object pose hypotheses (AlignPose-style, discrete stage; research R3, E3).

For one object and a set of candidate poses (B1's pose, its yaw turns, other sources' candidates), every calibrated
view that sees it gives a square crop around the union of the candidates' projections. Each candidate is rendered
(catalogue per-face-colour mesh, lit from the camera, mapped through the renderer colour calibration) and composited
over the observed crop, robot pixels kept in front. DINOv2 patch features of the observed crop vs of each composite,
over the patches the candidates cover (their union, so every candidate is compared on the same pixels): the
feature-metric distance is the mean (1 - cosine). Views are weighted by their patch count. The candidate with the lowest
distance wins when it beats the incumbent by a margin.

Appearance resolves what silhouettes cannot (a flipped laptop, a T's yaw, which of two blobs), and the wrist views add
the resolution the head lacks (2.4 mm / px at the table). Torch / transformers are imported lazily.

    fm = FeatureMetric("dinov2_b")
    d = fm.score(key, [T0, T1, ...], views)   # views: [{"img", "cam", "robot"}] -> distances (n,) and per-view info
"""
from __future__ import annotations

import numpy as np

from .cameras import Camera


class PatchFeatures:
    """L2-normalised DINOv2 patch tokens (g x g x C) of uint8 RGB crops resized to ``size`` (multiple of 14)."""

    def __init__(self, name: str = "dinov2_b", size: int = 224, layer: int | None = None, device: str | None = None):
        import torch
        from transformers import AutoModel

        from .embed_id import MODELS, models_root

        sub, fam = MODELS[name]
        assert fam == "dinov2", name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(models_root() / sub, torch_dtype=self.dtype).eval().to(self.device)
        self.size, self.layer = size, layer
        self.g = size // 14
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)

    def __call__(self, imgs: list[np.ndarray], batch: int = 32) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        out = []
        with torch.inference_mode():
            for i in range(0, len(imgs), batch):
                x = torch.from_numpy(np.stack([np.asarray(a, np.uint8) for a in imgs[i:i + batch]])).to(self.device)
                x = x.permute(0, 3, 1, 2).float() / 255.0
                if x.shape[-1] != self.size:
                    x = F.interpolate(x, size=(self.size, self.size), mode="bicubic", antialias=True, align_corners=False)
                x = ((x - self.mean) / self.std).to(self.dtype)
                o = self.model(pixel_values=x, output_hidden_states=self.layer is not None)
                h = o.hidden_states[self.layer] if self.layer is not None else o.last_hidden_state
                p = h[:, 1:1 + self.g * self.g].float()
                out.append(F.normalize(p, dim=-1).reshape(len(p), self.g, self.g, -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.g, self.g, 1), np.float32)


def _box_of(cams_uv: np.ndarray, pad: float = 0.35, min_side: float = 48.0):
    lo, hi = cams_uv.min(0), cams_uv.max(0)
    side = max(float((hi - lo).max()) * (1 + 2 * pad), min_side)
    c = (lo + hi) / 2
    return (c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2)


class FeatureMetric:
    def __init__(self, model: str = "dinov2_b", size: int = 224, layer: int | None = None, color_M=None,
                 min_patches: int = 4, crop_px: int = 448):
        from .embed_id import Gallery

        self.pf = PatchFeatures(model, size=size, layer=layer)
        self.gal = Gallery(size=crop_px, lit=True, color_M=color_M)
        self.crop_px = crop_px
        self.min_patches = min_patches

    def set_color(self, color_M):
        self.gal.color_M = None if color_M is None else np.asarray(color_M, float)

    def score_each(self, key: str, poses: list[np.ndarray], views: list[dict]) -> tuple[np.ndarray, list[dict]]:
        """``score`` for candidates far apart (other blobs, swaps): each pose in its own crop, on its own patches."""
        out, info = [], []
        for T in poses:
            d, inf = self.score(key, [T], views)
            out.append(d[0])
            info.append(inf)
        return np.array(out), info

    def score(self, key: str, poses: list[np.ndarray], views: list[dict]) -> tuple[np.ndarray, list[dict]]:
        """Feature-metric distance of each pose (lower = better) over the views that see the object, and per-view
        info. ``views``: [{"name", "img", "cam", "robot" (uint8 mask or None)}]."""
        import cv2

        from .embed_id import composite, crop
        from .render import load_asset

        a = load_asset(key)
        V = a.Vs
        n = len(poses)
        num = np.zeros(n)
        den = 0.0
        info = []
        for v in views:
            cam: Camera = v["cam"]
            uvs = []
            for T in poses:
                uv, z = cam.project(V @ T[:3, :3].T + T[:3, 3])
                if (z <= 0).any():
                    uvs = None
                    break
                uvs.append(uv)
            if not uvs:
                continue
            allv = np.concatenate(uvs)
            if allv[:, 0].max() < 0 or allv[:, 1].max() < 0 or allv[:, 0].min() > cam.w or allv[:, 1].min() > cam.h:
                continue
            box = _box_of(allv)
            real = crop(v["img"], box, self.crop_px, interp=cv2.INTER_LINEAR)
            rob = None
            if v.get("robot") is not None:
                rob = crop((v["robot"] > 0).astype(np.uint8) * 255, box, self.crop_px, interp=cv2.INTER_NEAREST) > 127
            comps, alphas = [], []
            for T in poses:
                rgb, al = self.gal.render_box(cam, key, T, box, lit=True)
                al = al.astype(bool)
                if rob is not None:
                    al &= ~rob
                comps.append(composite(rgb, al, real))
                alphas.append(al)
            g = self.pf.g
            union = np.zeros((self.crop_px, self.crop_px), bool)
            for al in alphas:
                union |= al
            pm = cv2.resize(union.astype(np.float32), (g, g), interpolation=cv2.INTER_AREA) > 0.3
            if pm.sum() < self.min_patches:
                continue
            F = self.pf([real] + comps)
            fr, fc = F[0], F[1:]
            d = 1.0 - (fc[:, pm] * fr[pm][None]).sum(-1)  # (n, P)
            w = float(pm.sum())
            num += w * d.mean(1)
            den += w
            info.append({"view": v.get("name"), "patches": int(pm.sum()), "d": [round(float(x), 4) for x in d.mean(1)]})
        if den == 0:
            return np.full(n, np.nan), info
        return num / den, info


def yaw_turns(T: np.ndarray, n: int = 4, about_centre: np.ndarray | None = None) -> list[np.ndarray]:
    """The pose and its turns by 2 pi k / n about world z through ``about_centre`` (default: the pose's origin)."""
    from scipy.spatial.transform import Rotation

    c = T[:3, 3] if about_centre is None else np.asarray(about_centre, float)
    out = []
    for k in range(n):
        Rz = Rotation.from_euler("z", 2 * np.pi * k / n).as_matrix()
        T2 = np.eye(4)
        T2[:3, :3] = Rz @ T[:3, :3]
        T2[:3, 3] = Rz @ (T[:3, 3] - c) + c
        out.append(T2)
    return out


def _pose_xyyaw(T: np.ndarray, centre: np.ndarray, dx: float, dy: float, dyaw: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    Rz = Rotation.from_euler("z", dyaw).as_matrix()
    T2 = np.eye(4)
    T2[:3, :3] = Rz @ T[:3, :3]
    T2[:3, 3] = Rz @ (T[:3, 3] - centre) + centre + np.array([dx, dy, 0.0])
    return T2


def refine(fm: FeatureMetric, key: str, T0: np.ndarray, views: list[dict], iters: int = 40, step_xy: float = 0.006,
           step_yaw: float = 0.12, pad: float = 0.8) -> tuple[np.ndarray, float, float]:
    """AlignPose-style continuous stage: Nelder-Mead over (dx, dy, dyaw) of the feature-metric distance, each view's
    crop fixed at the start pose (padded, so the object can move inside it) and its observed features computed once.
    Returns (pose, distance at the start, distance at the end)."""
    import cv2
    from scipy.optimize import minimize

    from .embed_id import composite, crop
    from .render import load_asset

    a = load_asset(key)
    V = a.Vs
    centre = T0[:3, :3] @ ((a.V.min(0) + a.V.max(0)) / 2) + T0[:3, 3]
    g = fm.pf.g
    fixed = []
    for v in views:
        cam = v["cam"]
        uv, z = cam.project(V @ T0[:3, :3].T + T0[:3, 3])
        if (z <= 0).any() or uv[:, 0].max() < 0 or uv[:, 1].max() < 0 or uv[:, 0].min() > cam.w or uv[:, 1].min() > cam.h:
            continue
        box = _box_of(uv, pad=pad)
        real = crop(v["img"], box, fm.crop_px, interp=cv2.INTER_LINEAR)
        rob = None
        if v.get("robot") is not None:
            rob = crop((v["robot"] > 0).astype(np.uint8) * 255, box, fm.crop_px, interp=cv2.INTER_NEAREST) > 127
        fixed.append((cam, box, real, rob, fm.pf([real])[0]))
    if not fixed:
        return T0, float("nan"), float("nan")

    def dist(p):
        T = _pose_xyyaw(T0, centre, p[0], p[1], p[2])
        num, den = 0.0, 0.0
        comps, masks = [], []
        for cam, box, real, rob, fr in fixed:
            rgb, al = fm.gal.render_box(cam, key, T, box, lit=True)
            al = al.astype(bool)
            if rob is not None:
                al &= ~rob
            comps.append(composite(rgb, al, real))
            masks.append(cv2.resize(al.astype(np.float32), (g, g), interpolation=cv2.INTER_AREA) > 0.3)
        F = fm.pf(comps)
        for (cam, box, real, rob, fr), fc, pm in zip(fixed, F, masks):
            if pm.sum() < fm.min_patches:
                continue
            d = 1.0 - (fc[pm] * fr[pm]).sum(-1)
            num += float(pm.sum()) * float(d.mean())
            den += float(pm.sum())
        return num / den if den > 0 else 1.0

    d0 = dist(np.zeros(3))
    simplex = np.array([[0, 0, 0], [step_xy, 0, 0], [0, step_xy, 0], [0, 0, step_yaw]], float)
    r = minimize(dist, np.zeros(3), method="Nelder-Mead", options={"initial_simplex": simplex, "maxfev": iters,
                                                                    "xatol": 5e-4, "fatol": 1e-3})
    return _pose_xyyaw(T0, centre, *r.x), d0, float(r.fun)
