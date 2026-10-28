"""Identity by learned image embeddings: an observed object crop against catalogue renders of each candidate asset
(or against a reference image), cosine similarity of DINOv2 / SigLIP2 / CLIP-L image features.

    emb = Embedder("dinov2_b")                       # weights from DOJO_EMBED_MODELS (one dir per model)
    gal = Gallery(cam)                               # candidate renders, crop-aligned or tight
    views = gal.views(key, box, centroid_uv)         # [{"rgb", "alpha", "tight_rgb", "tight_alpha", ...}]
    s = emb(masked(obs_crop, mask)) @ emb([masked(v["rgb"], v["alpha"]) for v in views]).T

Renders come from the catalogue's per-face-colour meshes (``render.ColorMeshRenderer``), either unlit or lit by a
light at the camera, optionally mapped through the catalogue -> renderer colour map (``color_calib``). Crops:
``box`` is a square image region (x0, y0, x1, y1) around the observed blob; an aligned render uses the same region
(scale and perspective kept), a tight render the region around its own projection (scale normalised).

Torch and transformers are imported lazily: the module loads without them (tests, the default pipeline).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .cameras import Camera

CROP_PX = 224
BG_FILL = 124  # neutral grey behind masked crops (both the observed object and the renders)

# name -> (dir under the models root, family)
MODELS = {
    "dinov2_s": ("dinov2-small", "dinov2"),
    "dinov2_b": ("dinov2-base", "dinov2"),
    "dinov2_l": ("dinov2-large", "dinov2"),
    "siglip2_b": ("siglip2-base-patch16-224-vision", "siglip"),
    "clip_l": ("clip-vit-large-patch14-vision", "clip"),
}
_NORM = {"dinov2": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
         "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
         "siglip": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))}


def models_root() -> Path:
    return Path(os.environ.get("DOJO_EMBED_MODELS", os.path.expanduser("~/dojo-data/pbake-identity/models")))


class Embedder:
    """L2-normalised image features of uint8 RGB crops (any size; resized to the model's input)."""

    def __init__(self, name: str, root: Path | None = None, device: str | None = None):
        import torch

        sub, fam = MODELS[name]
        p = (root or models_root()) / sub
        self.name, self.family = name, fam
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float16 if self.device == "cuda" else torch.float32
        if fam == "dinov2":
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(p, torch_dtype=dt)
        elif fam == "clip":
            from transformers import CLIPVisionModelWithProjection

            self.model = CLIPVisionModelWithProjection.from_pretrained(p, torch_dtype=dt)
        else:
            from transformers import SiglipVisionModel

            self.model = SiglipVisionModel.from_pretrained(p, torch_dtype=dt)
        self.model.eval().to(self.device)
        self.dtype = dt
        self.size = 224
        m, s = _NORM[fam]
        self.mean = torch.tensor(m, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(s, device=self.device).view(1, 3, 1, 1)

    def __call__(self, imgs, batch: int = 64) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        out = []
        with torch.inference_mode():
            for i in range(0, len(imgs), batch):
                x = torch.from_numpy(np.stack([np.asarray(a, np.uint8) for a in imgs[i:i + batch]])).to(self.device)
                x = x.permute(0, 3, 1, 2).float() / 255.0
                if x.shape[-1] != self.size or x.shape[-2] != self.size:
                    x = F.interpolate(x, size=(self.size, self.size), mode="bicubic", antialias=True, align_corners=False)
                x = ((x - self.mean) / self.std).to(self.dtype)
                o = self.model(pixel_values=x)
                if self.family == "clip":
                    f = o.image_embeds
                else:  # DINOv2: the layer-normed CLS token; SigLIP: the attention-pooled head output
                    f = o.pooler_output
                out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)

    def tokens(self, imgs, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """DINOv2 only: (L2-normalised CLS (N, D), layer-normed patch tokens (N, g, g, D)) of uint8 RGB crops
        (``wrist_check.MuseScorer``: CLS + GeM-pooled patches)."""
        import torch
        import torch.nn.functional as F

        if self.family != "dinov2":
            raise ValueError(f"tokens() needs a DINOv2 model, not {self.name}")
        cls, pat = [], []
        with torch.inference_mode():
            for i in range(0, len(imgs), batch):
                x = torch.from_numpy(np.stack([np.asarray(a, np.uint8) for a in imgs[i:i + batch]])).to(self.device)
                x = x.permute(0, 3, 1, 2).float() / 255.0
                if x.shape[-1] != self.size or x.shape[-2] != self.size:
                    x = F.interpolate(x, size=(self.size, self.size), mode="bicubic", antialias=True, align_corners=False)
                o = self.model(pixel_values=((x - self.mean) / self.std).to(self.dtype))
                h = o.last_hidden_state.float()             # layer-normed: [CLS, patches...]
                g = int(round((h.shape[1] - 1) ** 0.5))
                cls.append(torch.nn.functional.normalize(o.pooler_output.float(), dim=-1).cpu().numpy())
                pat.append(h[:, 1:1 + g * g].reshape(h.shape[0], g, g, -1).cpu().numpy())
        return np.concatenate(cls), np.concatenate(pat)


def square_box(bbox, pad: float = 0.25, min_side: float = 32.0, w: int = 640, h: int = 480):
    """A square region (x0, y0, x1, y1) around an (x, y, w, h) bbox, padded; it may extend past the image."""
    x, y, bw, bh = bbox
    side = max(bw, bh) * (1 + 2 * pad)
    side = max(side, min_side)
    cx, cy = x + bw / 2.0, y + bh / 2.0
    return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


def crop(img: np.ndarray, box, size: int = CROP_PX, interp=None) -> np.ndarray:
    """The square region ``box`` of ``img`` resampled to size x size (outside the image: edge pixels)."""
    import cv2

    x0, y0, x1, y1 = box
    s = size / (x1 - x0)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float64)
    return cv2.warpAffine(img, M, (size, size), flags=interp if interp is not None else cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)


def masked(rgb: np.ndarray, alpha: np.ndarray, fill: int = BG_FILL) -> np.ndarray:
    a = (np.asarray(alpha, np.float32) > 0.5)[..., None]
    return np.where(a, rgb, np.uint8(fill)).astype(np.uint8)


def composite(rgb: np.ndarray, alpha: np.ndarray, bg: np.ndarray) -> np.ndarray:
    a = (np.asarray(alpha, np.float32) > 0.5)[..., None]
    return np.where(a, rgb, bg).astype(np.uint8)


class _LitColorRenderer:
    """``render.ColorMeshRenderer``'s per-face-colour meshes with a directional light at the camera (shading cues)."""

    def __init__(self, base, ambient: float = 0.45, light: float = 2.2):
        pr = base.pr
        self.base = base
        self.scene = pr.Scene(bg_color=[0, 0, 0, 0], ambient_light=[ambient] * 3)
        self.cam = pr.IntrinsicsCamera(300, 300, 320, 240, znear=0.005, zfar=10.0)
        self.cam_node = self.scene.add(self.cam, pose=np.eye(4))
        self.light_node = self.scene.add(pr.DirectionalLight(color=np.ones(3), intensity=light), pose=np.eye(4))
        self.nodes: dict = {}

    def render(self, cam: Camera, key: str, T: np.ndarray):
        b = self.base
        if (cam.w, cam.h) != (b.w, b.h):
            b.r.viewport_width, b.r.viewport_height = cam.w, cam.h
            b.w, b.h = cam.w, cam.h
        self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
        self.scene.set_pose(self.cam_node, cam.T_wc_usd())
        self.scene.set_pose(self.light_node, cam.T_wc_usd())
        n = self.nodes.get(key)
        if n is None:
            src = b._node(key)  # builds (and caches) the coloured mesh in the base scene
            n = self.nodes[key] = self.scene.add(src.mesh, pose=np.eye(4))
        n.mesh.is_visible = True  # the mesh is shared with the base scene: hide it again after the render
        self.scene.set_pose(n, T)
        try:
            col, d = b.r.render(self.scene)
        finally:
            n.mesh.is_visible = False
        return col, d


class Gallery:
    """Renders of candidate assets for one camera: every place orientation x ``n_yaw`` yaws, the asset's bbox centre
    on the ray through the observed blob's centroid at its own half height (``z0`` = the support's top)."""

    def __init__(self, size: int = CROP_PX, n_yaw: int = 8, z0: float | None = None, lit: bool = True, color_M=None):
        from .cameras import TABLE_Z
        from .render import ColorMeshRenderer

        self.size, self.n_yaw, self.z0 = size, n_yaw, TABLE_Z if z0 is None else z0
        self.r = ColorMeshRenderer(size, size)
        self.lit = _LitColorRenderer(self.r) if lit else None
        self.color_M = None if color_M is None else np.asarray(color_M, float)

    def _map(self, col: np.ndarray) -> np.ndarray:
        if self.color_M is None:
            return col[..., :3].astype(np.uint8)
        c = col[..., :3].reshape(-1, 3) / 255.0
        c = np.clip(np.c_[c, np.ones(len(c))] @ self.color_M, 0, 1)
        return (c.reshape(col.shape[:2] + (3,)) * 255).round().astype(np.uint8)

    def poses(self, key: str, cam: Camera, centroid_uv):
        from .cameras import TABLE_Z
        from .inverse_layout import shape_models

        out = []
        for sm in shape_models(key):
            X = cam.ray_plane(np.asarray(centroid_uv, float)[None, :], self.z0 + sm.height / 2)[0]
            if not np.isfinite(X).all():
                continue
            period = sm.yaw_period if sm.sym < 36 else 2 * np.pi
            n = self.n_yaw if sm.sym < 36 else max(2, self.n_yaw // 2)
            for yaw in np.arange(n) * period / n:
                T = sm.pose(0.0, 0.0, float(yaw), self.z0 - TABLE_Z)
                c = sm.bbox_world(T).mean(axis=0)
                T[0, 3] += X[0] - c[0]
                T[1, 3] += X[1] - c[1]
                out.append((sm, float(yaw), T))
        return out

    def render_box(self, cam: Camera, key: str, T: np.ndarray, box, lit: bool):
        from .render import crop_camera

        cc = crop_camera(cam, *box, self.size)
        col, d = (self.lit if (lit and self.lit is not None) else self.r).render(cc, key, T)
        return self._map(col), (d > 0).astype(np.uint8)

    def views(self, key: str, cam: Camera, box, centroid_uv, lit: bool = True, tight: bool = True,
              pad: float = 0.25) -> list[dict]:
        """[{"tag", "yaw", "rgb", "alpha" (aligned to ``box``), "tight_rgb", "tight_alpha"}]."""
        out = []
        for sm, yaw, T in self.poses(key, cam, centroid_uv):
            rgb, a = self.render_box(cam, key, T, box, lit)
            v = {"tag": sm.tag, "yaw": round(yaw, 3), "rgb": rgb, "alpha": a}
            if tight:
                uv, z = cam.project(sm.bbox_world(T))
                if (z > 0).all():
                    lo, hi = uv.min(0), uv.max(0)
                    tb = square_box((lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]), pad=pad)
                    v["tight_rgb"], v["tight_alpha"] = self.render_box(cam, key, T, tb, lit)
            out.append(v)
        return out


class EmbedIdentity:
    """One model + one gallery: similarity of an observed head blob to each candidate asset.

    ``crop="composite"``: the raw crop against renders composited over the background crop (the segmentation mask
    loses an object's dark parts on a dark belt; the raw crop keeps them); ``"masked"``: both on grey."""

    def __init__(self, model: str, color_M=None, lit: bool = False, n_yaw: int = 8, z0: float | None = None,
                 crop_mode: str = "composite"):
        self.emb = Embedder(model)
        self.gal = Gallery(n_yaw=n_yaw, z0=z0, lit=lit, color_M=color_M)
        self.lit, self.crop_mode = lit, crop_mode

    def _query(self, img, mask, box, bg):
        """(query crop, background crop or None)."""
        import cv2

        obs = crop(img, box)
        if self.crop_mode == "composite" and bg is not None:
            return obs, crop(bg, box)
        m = crop((np.asarray(mask) > 0).astype(np.uint8) * 255, box, interp=cv2.INTER_NEAREST) > 127
        return masked(obs, m), None

    def _sims(self, q, bgc, views: list[tuple[str, np.ndarray, np.ndarray]], keys) -> dict[str, float]:
        if not views:
            return {}
        ims = [composite(rgb, a, bgc) if bgc is not None else masked(rgb, a) for _, rgb, a in views]
        f = self.emb([q] + ims)
        return key_scores(f[1:] @ f[0], [k for k, _, _ in views], list(keys))

    def scores(self, img: np.ndarray, cam: Camera, mask: np.ndarray, bbox, centroid, keys, bg=None) -> dict[str, float]:
        """Per key: the best similarity over its gallery (place orientations x yaws, centred on the blob's ray)."""
        box = square_box(bbox)
        q, bgc = self._query(img, mask, box, bg)
        views = [(k, v["rgb"], v["alpha"]) for k in keys
                 for v in self.gal.views(k, cam, box, centroid, lit=self.lit, tight=False)]
        return self._sims(q, bgc, views, keys)

    def scores_at_poses(self, img: np.ndarray, cam: Camera, mask: np.ndarray, bbox, items, bg=None) -> dict[str, float]:
        """Per key: the similarity of the blob to that key rendered at the given pose(s) (``items`` = [(key, T)])."""
        box = square_box(bbox)
        q, bgc = self._query(img, mask, box, bg)
        views = [(k, *self.gal.render_box(cam, k, T, box, self.lit)) for k, T in items]
        return self._sims(q, bgc, views, sorted({k for k, _ in items}))


_IDENTITY: dict = {}


def identity(model: str, color_M=None, lit: bool = False, crop_mode: str = "composite", n_yaw: int = 8) -> EmbedIdentity:
    """Process-wide ``EmbedIdentity`` per setting (the model loads once)."""
    k = (model, lit, crop_mode, n_yaw, None if color_M is None else tuple(np.ravel(color_M).round(6)))
    if k not in _IDENTITY:
        _IDENTITY[k] = EmbedIdentity(model, color_M=color_M, lit=lit, n_yaw=n_yaw, crop_mode=crop_mode)
    return _IDENTITY[k]


def fuse(sims: dict[str, float], losses: dict[str, float], weight: float) -> list[tuple[str, float]]:
    """Keys by embedding similarity minus ``weight`` x our loss (1.0 for a key we did not fit), best first."""
    return sorted(((k, s - weight * losses.get(k, 1.0)) for k, s in sims.items()), key=lambda t: -t[1])


def key_scores(sim: np.ndarray, view_keys: list[str], keys: list[str], agg: str = "max") -> dict[str, float]:
    """Per candidate key: the best (or top-3 mean) similarity over its views."""
    out = {}
    vk = np.asarray(view_keys)
    for k in keys:
        s = np.sort(sim[vk == k])[::-1]
        if len(s) == 0:
            continue
        out[k] = float(s[0] if agg == "max" else s[:3].mean())
    return out
