"""Rendering for render-and-compare and the robot self-mask.

Two paths:
- ``silhouette()``: CPU, OpenCV ``fillPoly`` of projected triangles. Exact silhouettes (no z-buffer
  needed for one object), ~1 ms for a 2k-face mesh at 640x480. Used inside pose search.
- ``SceneRenderer``: pyrender offscreen (EGL) depth + instance-id maps for many meshes at once
  (robot links, all objects), and optional textured RGB. Used for occlusion reasoning, the robot
  mask and visual checks. Set ``PYOPENGL_PLATFORM=egl`` (done here if unset).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .cameras import Camera

GL_MAX_FACES = int(os.environ.get("DOJO_GL_MAX_FACES", "100000000"))  # decimation measurably hurt identity
TEX_MAX_PX = int(os.environ.get("DOJO_TEX_MAX_PX", "512"))
TEX_CACHE = 6

CATALOG_DIRS = (
    os.environ.get("DOJO_CATALOG", ""),
    "/root/dojo-data/perception/catalog",
    os.path.expanduser("~/dojo-perception-data/catalog"),
)


def catalog_root() -> Path:
    for d in CATALOG_DIRS:
        if d and Path(d, "index.json").exists():
            return Path(d)
    raise FileNotFoundError("perception catalog not found; set DOJO_CATALOG")


def cluster_decimate(V: np.ndarray, F: np.ndarray, cell: float = 0.002):
    """Vertex-clustering decimation: snap vertices to a grid of ``cell`` metres, drop degenerate faces.
    Silhouettes move by at most cell/2 * sqrt(3)."""
    q = np.floor(V / cell).astype(np.int64)
    _, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    nV = np.zeros((counts.size, 3))
    np.add.at(nV, inv, V)
    nV /= counts[:, None]
    nF = inv[F]
    ok = (nF[:, 0] != nF[:, 1]) & (nF[:, 1] != nF[:, 2]) & (nF[:, 0] != nF[:, 2])
    nF = np.unique(np.sort(nF[ok], axis=1), axis=0)
    return nV, nF


class Asset:
    """Catalogue entry: geometry, decimated silhouette mesh, info."""

    def __init__(self, key: str, root: Path | None = None, cell: float | None = None):
        root = root or catalog_root()
        self.key = key
        self.dir = root / key
        g = np.load(self.dir / "geom.npz")
        self.V = g["vertices"].astype(float)
        self.F = g["faces"].astype(np.int64)
        self.face_colors = g["colors"] if "colors" in g else None
        self.info = json.loads((self.dir / "info.json").read_text())
        ext = float(np.max(self.info["extents"]))
        cell = cell if cell is not None else float(np.clip(ext / 60.0, 0.0005, 0.004))
        if len(self.F) > 3000:
            self.Vs, self.Fs = cluster_decimate(self.V, self.F, cell)
        else:
            self.Vs, self.Fs = self.V, self.F
        # GL mesh: at most GL_MAX_FACES faces (vertex clustering, cell grown from ext/300 until it fits);
        # silhouettes move by at most cell/2*sqrt(3) (about 1 mm for a 25 cm bottle)
        self.Vg, self.Fg = self.V, self.F
        c = ext / 300.0
        while len(self.Fg) > GL_MAX_FACES:
            self.Vg, self.Fg = cluster_decimate(self.V, self.F, c)
            c *= 1.4
        self.gl_cell = 0.0 if self.Fg is self.F else c / 1.4

    @property
    def extents(self):
        return np.asarray(self.info["extents"], dtype=float)


@lru_cache(maxsize=512)
def load_asset(key: str) -> Asset:
    return Asset(key)


def transform(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    return V @ T[:3, :3].T + T[:3, 3]


def silhouette(cam: Camera, V_world: np.ndarray, F: np.ndarray, scale: float = 1.0, out=None,
               cull_back: bool = False) -> np.ndarray:
    """Binary silhouette (uint8 0/1) of a mesh already in world coordinates. ``scale`` < 1 renders at
    reduced resolution (same field of view)."""
    w, h = int(round(cam.w * scale)), int(round(cam.h * scale))
    m = np.zeros((h, w), np.uint8) if out is None else out
    uv, z = cam.project(V_world)
    tri_ok = (z[F] > 0.01).all(axis=1)
    if not tri_ok.any():
        return m
    # continuous image coords (pixel i spans [i, i+1)) -> OpenCV pixel-centre coords; 4 fractional bits
    P = ((uv[F[tri_ok]] * scale - 0.5) * 16).round().astype(np.int32)
    # cull triangles outside the image (keeps it fast and int-safe). fillPoly with many polygons uses
    # an even-odd rule (overlaps cancel), so each triangle is filled on its own with fillConvexPoly.
    lo, hi = P.min(axis=1), P.max(axis=1)
    keep = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < w * 16) & (lo[:, 1] < h * 16) & (np.abs(P).max(axis=(1, 2)) < 2**26)
    if cull_back:  # only for closed, outward-wound meshes: front faces are clockwise in image (v down)
        d1 = P[:, 1] - P[:, 0]
        d2 = P[:, 2] - P[:, 0]
        cross = d1[:, 0].astype(np.int64) * d2[:, 1] - d1[:, 1].astype(np.int64) * d2[:, 0]
        keep &= cross > 0
    for tri in P[keep]:
        cv2.fillConvexPoly(m, tri, 1, lineType=cv2.LINE_8, shift=4)
    return m


# --------------------------------------------------------------------------------------------
# pyrender scene renderer
# --------------------------------------------------------------------------------------------

def _pyrender():
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender  # noqa: WPS433

    if not getattr(pyrender.renderer.Renderer, "_dojo_patched", False):
        # pyrender sorts every mesh node (hidden ones too) by depth on each render; our scenes keep many
        # hidden nodes, and depth/seg passes need no ordering. Visible nodes only (keeps blending order
        # irrelevant: our meshes are opaque).
        def _visible_mesh_nodes(self, scene):
            return [n for n in scene.mesh_nodes if n.mesh.is_visible]

        pyrender.renderer.Renderer._sorted_mesh_nodes = _visible_mesh_nodes
        pyrender.renderer.Renderer._dojo_patched = True
    return pyrender


class SceneRenderer:
    """Offscreen renderer: add meshes with ids, render depth + id map (+ RGB) for a Camera."""

    def __init__(self, w: int = 640, h: int = 480):
        pr = _pyrender()
        self.pr = pr
        self.w, self.h = w, h
        self.renderer = pr.OffscreenRenderer(w, h)
        self._mesh_cache: dict = {}

    def close(self):
        self.renderer.delete()

    def _pr_mesh(self, key, V, F, color=None, textured_glb: Path | None = None):
        if key in self._mesh_cache:
            return self._mesh_cache[key]
        import trimesh

        if textured_glb is not None and Path(textured_glb).exists():
            sc = trimesh.load(textured_glb, force="scene")
            meshes = [self.pr.Mesh.from_trimesh(g, smooth=False) for g in sc.dump()]
        else:
            tm = trimesh.Trimesh(V, F, process=False)
            if color is not None:
                tm.visual.face_colors = np.r_[np.asarray(color) * 255, 255].astype(np.uint8)
            meshes = [self.pr.Mesh.from_trimesh(tm, smooth=False)]
        self._mesh_cache[key] = meshes
        return meshes

    def render(self, cam: Camera, items: list[dict], rgb: bool = False, ambient=0.6):
        """items: [{"id": int>0, "key": cache key, "V": (N,3) local, "F": (M,3), "T": 4x4 world pose,
        optional "glb": path for textured RGB, "color": (3,)}]. Returns depth (H,W) metres (0 = empty),
        id map (H,W) int32 (0 = background), and RGB (H,W,3) uint8 or None."""
        pr = self.pr
        scene = pr.Scene(bg_color=[0, 0, 0, 0], ambient_light=[ambient] * 3)
        node_ids = {}
        for it in items:
            for m in self._pr_mesh(it["key"], it.get("V"), it.get("F"), it.get("color"), it.get("glb") if rgb else None):
                n = scene.add(m, pose=it["T"])
                node_ids[n] = it["id"]
        K = cam.K
        pc = pr.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=10.0)
        scene.add(pc, pose=cam.T_wc_usd())
        if self.w != cam.w or self.h != cam.h:
            self.renderer.viewport_width, self.renderer.viewport_height = cam.w, cam.h
            self.w, self.h = cam.w, cam.h
        seg_map = {n: ((i & 255), (i >> 8) & 255, (i >> 16) & 255) for n, i in node_ids.items()}
        seg, depth = self.renderer.render(scene, flags=pr.RenderFlags.SEG, seg_node_map=seg_map)
        ids = seg[..., 0].astype(np.int32) | (seg[..., 1].astype(np.int32) << 8) | (seg[..., 2].astype(np.int32) << 16)
        color = None
        if rgb:
            light = pr.DirectionalLight(color=np.ones(3), intensity=2.5)
            scene.add(light, pose=cam.T_wc_usd())
            color, _ = self.renderer.render(scene)
        return depth, ids, color


# --------------------------------------------------------------------------------------------
# Robot
# --------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def robot_link_meshes(cell: float = 0.003):
    """{link: (V, F)} decimated visual meshes of the X5 in each link frame (URDF visual origin applied)."""
    import trimesh

    from .robot_model import X5Model

    model = X5Model.default()
    out = {}
    for link, vis in model.visuals.items():
        Vs, Fs, off = [], [], 0
        for fname, T in vis:
            p = model.mesh_path(fname)
            if not p.exists():
                continue
            m = trimesh.load(p, force="mesh")
            V = np.asarray(m.vertices, dtype=float)
            if p.suffix.lower() == ".glb" and np.abs(V).max() > 5:  # glb in mm
                V = V / 1000.0
            V = V @ T[:3, :3].T + T[:3, 3]
            F = np.asarray(m.faces, dtype=np.int64)
            Vs.append(V)
            Fs.append(F + off)
            off += len(V)
        if Vs:
            V, F = np.concatenate(Vs), np.concatenate(Fs)
            if len(F) > 3000:
                V, F = cluster_decimate(V, F, cell)
            out[link] = (V, F)
    return out


def robot_items(state: dict, id_base: int = 1000) -> list[dict]:
    """Render items for both arms from an observation state (official fields)."""
    from .robot_model import X5Model

    model = X5Model.default()
    meshes = robot_link_meshes()
    items = []
    for ai, arm in enumerate(("left", "right")):
        q = state.get(f"{arm}_arm_joint_state")
        if q is None:
            continue
        g = state.get(f"{arm}_ee_joint_state")
        g = None if g is None else float(np.ravel(g)[0])
        poses = model.fk_world(arm, q, g)
        for li, (link, (V, F)) in enumerate(meshes.items()):
            if link in poses:
                items.append({"id": id_base + 100 * ai + li, "key": f"robot:{arm}:{link}", "V": V, "F": F, "T": poses[link],
                              "link": link, "arm": arm})
    return items


def robot_mask(cam: Camera, state: dict, dilate_px: int = 0) -> np.ndarray:
    """Silhouette of both arms in a camera, from FK (GL if available, else CPU)."""
    g = gl()
    if g is not None:
        m = g.render(cam, [(it["key"], it["V"], it["F"], it["T"]) for it in robot_items(state)])
    else:
        m = np.zeros((cam.h, cam.w), np.uint8)
        for it in robot_items(state):
            silhouette(cam, transform(it["V"], it["T"]), it["F"], out=m)
    if dilate_px > 0:
        m = cv2.dilate(m, np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8))
    return m


# --------------------------------------------------------------------------------------------
# GL silhouettes (exact, full-resolution meshes, ~1 ms per render incl. readback)
# --------------------------------------------------------------------------------------------

class GLSilhouettes:
    """A persistent offscreen scene: one node per mesh key, toggled visible per call. Process-local."""

    def __init__(self, w: int = 640, h: int = 480):
        pr = _pyrender()
        self.pr = pr
        self.r = pr.OffscreenRenderer(w, h)
        self.w, self.h = w, h
        self.scene = pr.Scene(bg_color=[0, 0, 0, 0])
        self.cam = pr.IntrinsicsCamera(300, 300, 320, 240, znear=0.005, zfar=10.0)
        self.cam_node = self.scene.add(self.cam, pose=np.eye(4))
        self.nodes: dict[str, object] = {}

    def _node(self, key: str, V: np.ndarray, F: np.ndarray):
        n = self.nodes.get(key)
        if n is None:
            import trimesh

            # smooth=True keeps vertices shared (flat shading would unweld every face: ~3x the memory);
            # normals are irrelevant for depth / silhouettes
            m = self.pr.Mesh.from_trimesh(trimesh.Trimesh(V, F, process=False), smooth=True)
            m.is_visible = False
            n = self.scene.add(m, pose=np.eye(4))
            self.nodes[key] = n
        return n

    def render(self, cam: Camera, items: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], depth: bool = False):
        """items: [(key, V_local, F, T_world)]. Returns the union mask (uint8) or the depth map."""
        if (cam.w, cam.h) != (self.w, self.h):
            self.r.viewport_width, self.r.viewport_height = cam.w, cam.h
            self.w, self.h = cam.w, cam.h
        self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
        self.scene.set_pose(self.cam_node, cam.T_wc_usd())
        vis = []
        for key, V, F, T in items:
            n = self._node(key, V, F)
            n.mesh.is_visible = True
            self.scene.set_pose(n, T)
            vis.append(n)
        d = self.r.render(self.scene, flags=self.pr.RenderFlags.DEPTH_ONLY)
        for n in vis:
            n.mesh.is_visible = False
        return d if depth else (d > 0).astype(np.uint8)


_GL = None
_GL_ROI: dict = {}


def gl() -> GLSilhouettes | None:
    """Process-wide full-frame GL silhouette renderer, or None when disabled (DOJO_GL=0) or unavailable."""
    global _GL
    if _GL is None:
        if os.environ.get("DOJO_GL", "1") == "0":
            _GL = False
        else:
            try:
                _GL = GLSilhouettes()
            except Exception:  # no EGL / no GPU: fall back to the CPU paths
                _GL = False
    return _GL or None


def gl_roi(size: int) -> GLSilhouettes | None:
    """A second GL context with a fixed size x size viewport for region-of-interest renders (a cropped and
    scaled camera): a fixed viewport avoids framebuffer reallocation, and the readback is tiny."""
    if gl() is None:
        return None
    r = _GL_ROI.get(size)
    if r is None:
        r = _GL_ROI[size] = GLSilhouettes(size, size)
    return r


def crop_camera(cam: Camera, x0: float, y0: float, x1: float, y1: float, size: int) -> Camera:
    """The camera whose size x size image is the region [x0, x1) x [y0, y1) of ``cam``'s image, resampled."""
    sx, sy = size / (x1 - x0), size / (y1 - y0)
    K = cam.K.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] - x0) * sx
    K[1, 2] = (K[1, 2] - y0) * sy
    return Camera(cam.name, K, cam.T_wc, size, size, meta={"crop": (x0, y0, x1, y1)})


class TexturedRenderer:
    """Persistent textured scene (catalogue GLBs) with a camera-mounted light; one object per call."""

    def __init__(self, w: int = 640, h: int = 480):
        pr = _pyrender()
        self.pr = pr
        self.r = pr.OffscreenRenderer(w, h)
        self.w, self.h = w, h
        self.scene = pr.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.55] * 3)
        self.cam = pr.IntrinsicsCamera(300, 300, 320, 240, znear=0.005, zfar=10.0)
        self.cam_node = self.scene.add(self.cam, pose=np.eye(4))
        self.light_node = self.scene.add(pr.DirectionalLight(color=np.ones(3), intensity=2.0), pose=np.eye(4))
        self.nodes: dict[str, list] = {}

    def _nodes(self, key: str):
        ns = self.nodes.pop(key, None)
        if ns is None:
            import trimesh
            from PIL import Image

            sc = trimesh.load(catalog_root() / key / "visual.glb", force="scene")
            ns = []
            for g in sc.dump():
                mat = getattr(g.visual, "material", None)
                for attr in ("baseColorTexture", "image"):
                    img = getattr(mat, attr, None) if mat is not None else None
                    if img is not None and max(img.size) > TEX_MAX_PX:  # bound RAM: textures to <= 512 px
                        f = TEX_MAX_PX / max(img.size)
                        setattr(mat, attr, img.resize((max(1, int(img.size[0] * f)), max(1, int(img.size[1] * f))),
                                                      Image.BILINEAR))
                m = self.pr.Mesh.from_trimesh(g, smooth=False)
                m.is_visible = False
                ns.append(self.scene.add(m, pose=np.eye(4)))
            while len(self.nodes) >= TEX_CACHE:  # LRU: drop the oldest asset (GL buffers freed next render)
                old = next(iter(self.nodes))
                for n in self.nodes.pop(old):
                    self.scene.remove_node(n)
        self.nodes[key] = ns  # (re)insert as most recent
        return ns

    def render(self, cam: Camera, key: str, T: np.ndarray):
        if (cam.w, cam.h) != (self.w, self.h):
            self.r.viewport_width, self.r.viewport_height = cam.w, cam.h
            self.w, self.h = cam.w, cam.h
        self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
        self.scene.set_pose(self.cam_node, cam.T_wc_usd())
        self.scene.set_pose(self.light_node, cam.T_wc_usd())
        ns = self._nodes(key)
        for n in ns:
            n.mesh.is_visible = True
            self.scene.set_pose(n, T)
        col, d = self.r.render(self.scene)
        for n in ns:
            n.mesh.is_visible = False
        return col, d


_TEX = None


def tex() -> TexturedRenderer | None:
    global _TEX
    if _TEX is None:
        try:
            _TEX = TexturedRenderer() if os.environ.get("DOJO_GL", "1") != "0" else False
        except Exception:
            _TEX = False
    return _TEX or None
