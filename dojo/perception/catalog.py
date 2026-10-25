"""Asset catalogue: RoboDojo object USD/USDZ -> textured meshes + geometry + resting poses.

Run on a machine with the RoboDojo Assets (compute2), in the `perception` extra's venv:

    python -m dojo.perception.catalog build --assets /root/controlr-robodojo/hf_git/Assets \
        --out /root/dojo-data/perception/catalog [--only Rigid/block Rigid/bowl ...] [--workers 16]

Per asset, under ``<out>/<Type>/<category>/<index:05d>/``:

- ``visual.glb``  textured visual meshes (base-colour texture or factor), metres, in the asset's
  own frame (the frame the layout pose refers to: the default prim, its own xform excluded);
- ``geom.npz``    ``vertices`` (N,3) float32, ``faces`` (M,3) int32: all visual meshes welded,
  same frame; ``colors`` (M,3) uint8 mean base colour per face (for cheap appearance cues);
- ``info.json``   bbox, extents, the asset's ``metadata.json`` ``active.place`` frames (the layout
  generator puts an object in one of these, yaws it about world z and lets physics settle),
  trimesh stable poses, rotational symmetry guess, mean colour, source path.

The pose convention is RoboDojo's: a layout pose ``[x, y, z, qw, qx, qy, qz]`` places the asset's
default prim; mesh points are expressed in that prim's frame, times the stage's metersPerUnit.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

TYPES = ("Rigid", "Geometry", "Articulation", "Clutter", "Dynamic", "Garment", "Fluid")


# --------------------------------------------------------------------------------------------
# USD reading
# --------------------------------------------------------------------------------------------

def _asset_file(d: Path) -> Path | None:
    for name in ("object.usdz", "object.usd"):
        if (d / name).exists():
            return d / name
    return None


def _read_texture(resolved: str, usd_path: Path) -> bytes | None:
    """Bytes of a texture referenced from a USD/USDZ (``a.usdz[sub/path.png]`` or a plain path)."""
    if not resolved:
        return None
    if "[" in resolved and resolved.endswith("]"):
        pkg, inner = resolved[:-1].split("[", 1)
        try:
            with zipfile.ZipFile(pkg) as z:
                return z.read(inner)
        except (KeyError, OSError, zipfile.BadZipFile):
            return None
    p = Path(resolved)
    if not p.is_absolute():
        p = usd_path.parent / p
    return p.read_bytes() if p.exists() else None


def _shader_inputs(material):
    """Base colour (factor, texture asset path, uv transform) of a gltf/pbr.mdl or UsdPreviewSurface material."""
    from pxr import UsdShade

    factor = np.array([0.7, 0.7, 0.7])
    tex = None
    uvx = {"scale": (1.0, 1.0), "offset": (0.0, 0.0), "rotation": 0.0}
    if material is None:
        return factor, tex, uvx
    from pxr import Usd

    for prim in Usd.PrimRange(material.GetPrim()):
        if prim.GetTypeName() != "Shader":
            continue
        sh = UsdShade.Shader(prim)
        names = {i.GetBaseName(): i for i in sh.GetInputs()}
        if "base_color_factor" in names and names["base_color_factor"].Get() is not None:
            factor = np.array(names["base_color_factor"].Get(), dtype=float)[:3]
        if "diffuseColor" in names and names["diffuseColor"].Get() is not None:
            factor = np.array(names["diffuseColor"].Get(), dtype=float)[:3]
        if "diffuse_color_constant" in names and names["diffuse_color_constant"].Get() is not None:
            factor = np.array(names["diffuse_color_constant"].Get(), dtype=float)[:3]
        lname = prim.GetName().lower().replace("_", "")
        is_base = "basecolor" in lname or "diffuse" in lname or "albedo" in lname
        for key in ("texture", "file", "diffuse_texture"):
            if key in names and names[key].Get() is not None and (is_base or tex is None):
                ap = names[key].Get()
                if is_base or key == "diffuse_texture" or tex is None:
                    tex = ap.resolvedPath or ap.path
                    # gltf/pbr.mdl texture transforms are float2; UsdUVTexture's scale/bias are float4
                    # value multipliers (not uv) and are ignored here.
                    if "scale" in names and names["scale"].Get() is not None and len(names["scale"].Get()) == 2:
                        uvx["scale"] = tuple(names["scale"].Get())
                    if "offset" in names and names["offset"].Get() is not None and len(names["offset"].Get()) == 2:
                        uvx["offset"] = tuple(names["offset"].Get())
                    if "rotation" in names and names["rotation"].Get() is not None:
                        uvx["rotation"] = float(names["rotation"].Get())
                if is_base:
                    break
    if tex is not None:
        # UsdPreviewSurface: a connected texture replaces diffuseColor (its authored value is unused);
        # gltf/pbr.mdl: base_color_factor multiplies the texture.
        ids = {str(UsdShade.Shader(p).GetIdAttr().Get()) for p in Usd.PrimRange(material.GetPrim())
               if p.GetTypeName() == "Shader"}
        if "UsdPreviewSurface" in ids:
            factor = np.ones(3)
    return factor, tex, uvx


def _triangulate(counts: np.ndarray, indices: np.ndarray):
    """Fan-triangulate polygons. Returns (tri corner indices into `indices` (T,3), source face id (T,))."""
    tris, src = [], []
    start = 0
    for f, c in enumerate(counts):
        for k in range(1, c - 1):
            tris.append((start, start + k, start + k + 1))
            src.append(f)
        start += c
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3), np.asarray(src, dtype=np.int64)


def _is_hidden(prim) -> bool:
    from pxr import UsdGeom

    p = prim
    while p and p.IsValid() and str(p.GetPath()) != "/":
        if "collision" in p.GetName().lower():
            return True
        img = UsdGeom.Imageable(p)
        if img:
            vis = img.GetVisibilityAttr().Get()
            if vis == UsdGeom.Tokens.invisible:
                return True
            purpose = img.GetPurposeAttr().Get()
            if purpose in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                return True
        p = p.GetParent()
    return False


def read_usd_meshes(usd_path: Path):
    """Visual meshes of an asset, in the default prim's frame, metres.

    Returns (parts, meta) where each part is a dict with ``corners`` (T,3,3) float, ``uv`` (T,3,2) or
    None, ``factor`` (3,), ``texture`` bytes or None, ``uvx``; meta has metersPerUnit, upAxis and the
    default prim's own local transform (normally identity; overwritten by the layout pose at runtime).
    """
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(usd_path))
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    up = UsdGeom.GetStageUpAxis(stage)
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_w = np.array(cache.GetLocalToWorldTransform(root), dtype=float).T if root.IsA(UsdGeom.Xformable) else np.eye(4)
    root_inv = np.linalg.inv(root_w)
    parts = []
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh" or _is_hidden(prim):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if pts is None or counts is None or idx is None or len(pts) == 0:
            continue
        pts = np.asarray(pts, dtype=float)
        counts = np.asarray(counts, dtype=np.int64)
        idx = np.asarray(idx, dtype=np.int64)
        w = np.array(cache.GetLocalToWorldTransform(prim), dtype=float).T  # column-vector convention
        m = root_inv @ w
        pts_h = np.c_[pts, np.ones(len(pts))] @ m.T
        pts = pts_h[:, :3] * mpu
        tri_c, tri_src = _triangulate(counts, idx)
        if len(tri_c) == 0:
            continue
        corners = pts[idx[tri_c]]  # (T,3,3)
        # uv
        uv = None
        pv = UsdGeom.PrimvarsAPI(prim)
        for name in ("st", "st0", "UVMap", "uv"):
            if pv.HasPrimvar(name):
                st_pv = pv.GetPrimvar(name)
                vals = st_pv.ComputeFlattened()
                if vals is None:
                    continue
                vals = np.asarray(vals, dtype=float)
                interp = st_pv.GetInterpolation()
                if interp == "faceVarying" and len(vals) == len(idx):
                    uv = vals[tri_c]
                elif interp in ("vertex", "varying") and len(vals) == len(pts):
                    uv = vals[idx[tri_c]]
                elif interp == "uniform" and len(vals) == len(counts):
                    uv = np.repeat(vals[tri_src][:, None, :], 3, axis=1)
                break
        # materials: GeomSubsets split faces
        subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
        groups = []
        assigned = np.zeros(len(counts), dtype=bool)
        for ss in subsets:
            fidx = np.asarray(ss.GetIndicesAttr().Get() or [], dtype=np.int64)
            mat = UsdShade.MaterialBindingAPI(ss.GetPrim()).ComputeBoundMaterial()[0]
            if len(fidx):
                assigned[fidx] = True
                groups.append((fidx, mat))
        if not assigned.all():
            mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            groups.append((np.nonzero(~assigned)[0], mat))
        for fidx, mat in groups:
            sel = np.isin(tri_src, fidx)
            if not sel.any():
                continue
            factor, tex_path, uvx = _shader_inputs(mat)
            tex = _read_texture(tex_path, usd_path) if tex_path else None
            if tex is None:
                dc = pv.GetPrimvar("displayColor") if pv.HasPrimvar("displayColor") else None
                if mat is None and dc is not None and dc.Get() is not None and len(dc.Get()):
                    factor = np.asarray(dc.Get()[0], dtype=float)
            parts.append(
                {
                    "name": str(prim.GetPath()),
                    "corners": corners[sel],
                    "uv": None if uv is None else uv[sel],
                    "factor": factor,
                    "texture": tex,
                    "uvx": uvx,
                }
            )
    meta = {"meters_per_unit": mpu, "up_axis": str(up), "default_prim": str(root.GetPath()),
            "default_prim_xform": root_w.tolist()}
    return parts, meta


# --------------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------------

def _apply_uvx(uv: np.ndarray, uvx: dict) -> np.ndarray:
    s = np.asarray(uvx.get("scale", (1, 1)), dtype=float)
    o = np.asarray(uvx.get("offset", (0, 0)), dtype=float)
    r = np.deg2rad(float(uvx.get("rotation", 0.0)))
    if r:
        c, sn = np.cos(r), np.sin(r)
        uv = uv @ np.array([[c, sn], [-sn, c]])
    return uv * s + o


def _mean_texture_colour(tex: bytes | None, uv: np.ndarray | None, factor: np.ndarray):
    """Per-triangle colour: texture sampled at the triangle's uv centroid (times factor), else factor."""
    from PIL import Image

    n = 0 if uv is None else len(uv)
    if tex is None or uv is None:
        return None
    try:
        img = np.asarray(Image.open(io.BytesIO(tex)).convert("RGB"), dtype=float) / 255.0
    except Exception:
        return None
    h, w = img.shape[:2]
    c = uv.mean(axis=1)
    u = np.mod(c[:, 0], 1.0)
    v = np.mod(c[:, 1], 1.0)
    px = np.clip((u * (w - 1)).astype(int), 0, w - 1)
    py = np.clip(((1 - v) * (h - 1)).astype(int), 0, h - 1)
    del n
    return img[py, px] * factor[None, :]


def symmetry_guess(vertices: np.ndarray, n_test=(2, 3, 4, 6, 8, 12, 36)) -> dict:
    """Rotational symmetry about the asset's local z axis (the usual up axis), by geometry only.

    Rotate the point cloud by 360/n about z through the xy centroid and measure the symmetric chamfer
    distance relative to the extent. Returns the largest n with chamfer < 2 % of the extent (36 means
    "continuous"). Texture can break a geometric symmetry: the caller decides whether to use it.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(vertices, dtype=float)
    if len(v) > 4000:
        v = v[np.random.default_rng(0).choice(len(v), 4000, replace=False)]
    c = np.r_[0.5 * (v[:, :2].min(0) + v[:, :2].max(0)), 0.0]
    p = v - c
    ext = float(np.linalg.norm(v.max(0) - v.min(0))) or 1.0
    tree = cKDTree(p)
    best = 1
    scores = {}
    for n in n_test:
        a = 2 * np.pi / n
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        q = p @ R.T
        d1 = tree.query(q)[0].mean()
        d2 = cKDTree(q).query(p)[0].mean()
        s = 0.5 * (d1 + d2) / ext
        scores[n] = round(float(s), 5)
        if s < 0.02:
            best = max(best, n)
    return {"z_order": best, "chamfer_rel": scores, "xy_center": c[:2].tolist()}


def convert_asset(asset_dir: str, out_dir: str, with_stable: bool = True, glb: bool = True) -> dict:
    import trimesh
    from PIL import Image

    asset_dir = Path(asset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    usd = _asset_file(asset_dir)
    if usd is None:
        return {"asset": str(asset_dir), "error": "no usd"}
    parts, meta = read_usd_meshes(usd)
    if not parts:
        return {"asset": str(asset_dir), "error": "no visual meshes"}

    scene = trimesh.Scene()
    all_corners, all_cols = [], []
    for i, p in enumerate(parts):
        corners = p["corners"]
        T = len(corners)
        verts = corners.reshape(-1, 3)
        faces = np.arange(3 * T).reshape(T, 3)
        factor = np.clip(np.asarray(p["factor"], dtype=float), 0, 1)
        cols = _mean_texture_colour(p["texture"], None if p["uv"] is None else _apply_uvx(p["uv"], p["uvx"]), factor)
        if cols is None:
            cols = np.repeat(factor[None, :], T, axis=0)
        all_corners.append(corners)
        all_cols.append(cols)
        if p["texture"] is not None and p["uv"] is not None:
            try:
                img = Image.open(io.BytesIO(p["texture"])).convert("RGB")
                uv = _apply_uvx(p["uv"], p["uvx"]).reshape(-1, 2)
                mat = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=img, baseColorFactor=np.r_[factor, 1.0], metallicFactor=0.0, roughnessFactor=0.8
                )
                vis = trimesh.visual.TextureVisuals(uv=uv, material=mat)
                m = trimesh.Trimesh(vertices=verts, faces=faces, visual=vis, process=False)
            except Exception:
                m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                m.visual.face_colors = np.c_[(cols * 255).astype(np.uint8), np.full(T, 255, np.uint8)]
        else:
            m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            m.visual.face_colors = np.c_[(cols * 255).astype(np.uint8), np.full(T, 255, np.uint8)]
        scene.add_geometry(m, node_name=f"part{i}", geom_name=f"part{i}")
    if glb:  # geometry-only catalogues (glb=False) are enough for the estimator (face-colour texture term)
        scene.export(out_dir / "visual.glb")

    corners = np.concatenate(all_corners)
    cols = np.concatenate(all_cols)
    welded = trimesh.Trimesh(vertices=corners.reshape(-1, 3), faces=np.arange(3 * len(corners)).reshape(-1, 3), process=False)
    welded.merge_vertices(digits_vertex=6)
    np.savez_compressed(
        out_dir / "geom.npz",
        vertices=welded.vertices.astype(np.float32),
        faces=welded.faces.astype(np.int32),
        colors=(np.clip(cols, 0, 1) * 255).astype(np.uint8),
    )
    bmin, bmax = welded.vertices.min(0), welded.vertices.max(0)
    area_w = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    mean_col = (cols * area_w[:, None]).sum(0) / max(area_w.sum(), 1e-12)

    md_path = asset_dir / "metadata.json"
    md = json.loads(md_path.read_text()) if md_path.exists() else {}
    place = md.get("active", {}).get("place", {}) if isinstance(md.get("active"), dict) else {}
    place_frames = {}
    for tag, data in (place or {}).items():
        if isinstance(data, dict) and data.get("projection_circle", {}).get("center") is not None:
            place_frames[tag] = {
                "center": data["projection_circle"]["center"],
                "orientation": data.get("orientation"),
                "radius": data["projection_circle"].get("radius"),
            }
    stable = []
    if with_stable:
        try:
            hull = welded.convex_hull
            com = hull.center_mass if hull.is_volume else welded.vertices.mean(0)
            tfs, probs = trimesh.poses.compute_stable_poses(hull, center_mass=com, n_samples=1, threshold=0.0)
            for tf, pr in list(zip(tfs, probs))[:12]:
                if pr < 0.01:
                    continue
                stable.append({"T": np.round(tf, 6).tolist(), "prob": round(float(pr), 4)})
        except Exception as e:  # degenerate meshes (planes, cards)
            stable = [{"error": f"{type(e).__name__}: {e}"[:200]}]
    info = {
        "source": str(usd),
        "n_faces": int(len(welded.faces)),
        "n_vertices": int(len(welded.vertices)),
        "bbox_min": np.round(bmin, 6).tolist(),
        "bbox_max": np.round(bmax, 6).tolist(),
        "extents": np.round(bmax - bmin, 6).tolist(),
        "metadata_extents": md.get("geometry", {}).get("aligned_bbox", {}).get("extents"),
        "mass": md.get("physics", {}).get("mass"),
        "mean_color": np.round(mean_col, 4).tolist(),
        "n_textured_parts": int(sum(p["texture"] is not None for p in parts)),
        "n_parts": len(parts),
        "place": place_frames,
        "stable_poses": stable,
        "symmetry": symmetry_guess(welded.vertices),
        **meta,
    }
    info.update(annotations(asset_dir, usd, md))
    desc = asset_dir / "description.json"
    if desc.exists():
        try:
            d = json.loads(desc.read_text())
            info["names"] = d.get("caption", {}).get("name", [])[:3]
            info["colors_text"] = d.get("caption", {}).get("color", [])[:3]
        except Exception:
            pass
    (out_dir / "info.json").write_text(json.dumps(info, indent=1))
    return {"asset": str(asset_dir), "ok": True, "faces": info["n_faces"], "extents": info["extents"],
            "metadata_extents": info["metadata_extents"]}


def write_lowres(asset_out: Path, max_px: int = 512) -> bool:
    """``visual_lo.glb``: the visual GLB with base-colour textures downscaled to <= max_px (fast to load, small
    in RAM; used by the textured render-and-compare term, where 512 px made no measurable difference)."""
    import trimesh
    from PIL import Image

    src = asset_out / "visual.glb"
    dst = asset_out / "visual_lo.glb"
    if not src.exists():
        return False
    sc = trimesh.load(src, force="scene")
    for g in sc.geometry.values():
        mat = getattr(g.visual, "material", None)
        for attr in ("baseColorTexture", "image"):
            img = getattr(mat, attr, None) if mat is not None else None
            if img is not None and max(img.size) > max_px:
                f = max_px / max(img.size)
                setattr(mat, attr, img.resize((max(1, int(img.size[0] * f)), max(1, int(img.size[1] * f))), Image.BILINEAR))
    sc.export(dst)
    return True


def list_assets(objects_root: Path, only: list[str] | None = None):
    """(type, category, index, dir) for every asset dir under Object/RoboDojo/<Type>/<cat>/<idx>."""
    out = []
    for t in TYPES:
        tdir = objects_root / t
        if not tdir.is_dir():
            continue
        for cat in sorted(os.listdir(tdir)):
            cdir = tdir / cat
            if not cdir.is_dir():
                continue
            if only and f"{t}/{cat}" not in only and t not in only:
                continue
            for idx in sorted(os.listdir(cdir)):
                if idx.isdigit() and _asset_file(cdir / idx):
                    out.append((t, cat, int(idx), cdir / idx))
    return out


def build(assets: Path, out: Path, only=None, workers=8, force=False, glb: bool = True, keys: list | None = None):
    objects_root = assets / "Object" / "RoboDojo"
    todo = []
    want = set(keys) if keys else None
    for t, cat, idx, d in list_assets(objects_root, only):
        if want is not None and f"{t}/{cat}/{idx:05d}" not in want:
            continue
        od = out / t / cat / f"{idx:05d}"
        if not force and (od / "info.json").exists():
            continue
        todo.append((str(d), str(od)))
    print(f"{len(todo)} assets to convert", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(convert_asset, a, o, True, glb): a for a, o in todo}
        for i, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
            except Exception as e:
                r = {"asset": futs[f], "error": f"{type(e).__name__}: {e}"[:300]}
            results.append(r)
            if "error" in r or i % 25 == 0:
                print(i, json.dumps(r)[:300], flush=True)
    write_index(out)
    return results


def write_index(out: Path):
    index = {}
    for info in sorted(out.glob("*/*/*/info.json")):
        t, cat, idx = info.parts[-4:-1]
        d = json.loads(info.read_text())
        index[f"{t}/{cat}/{idx}"] = {
            "extents": d["extents"],
            "mean_color": d["mean_color"],
            "n_faces": d["n_faces"],
            "place": sorted(d.get("place", {})),
            "symmetry_z": d.get("symmetry", {}).get("z_order"),
            "names": d.get("names", []),
        }
    (out / "index.json").write_text(json.dumps(index, indent=0))
    print(f"index: {len(index)} assets", flush=True)


# Per-category attributes a checker reads that the geometry does not carry. number_card: the decal of
# Geometry/number_card/0000<i> shows the digit i (checked on the decal textures of all nine cards).
CATEGORY_ATTRIBUTES = {"number_card": lambda idx: {"digit": int(idx)}}


def _gf_to_np(M) -> np.ndarray:
    """Gf row-vector 4x4 -> column-vector numpy (p' = T @ p)."""
    return np.array([[M[r][c] for c in range(4)] for r in range(4)], dtype=float).T


def articulation_info(usd_path: Path) -> dict | None:
    """Joints of an articulated asset in the asset (default prim) frame, metres: type, bodies, anchor, axis,
    limits, drive target, and the moving body's bbox (its top centre is where a press lands)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    st = Usd.Stage.Open(str(usd_path))
    mpu = float(UsdGeom.GetStageMetersPerUnit(st) or 1.0)
    root = st.GetDefaultPrim()
    xc = UsdGeom.XformCache()
    W_root_inv = np.linalg.inv(_gf_to_np(xc.GetLocalToWorldTransform(root)))
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def in_root(prim) -> np.ndarray:
        T = W_root_inv @ _gf_to_np(xc.GetLocalToWorldTransform(prim))
        T[:3, 3] *= mpu
        return T

    joints = []
    for prim in st.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
        b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
        d = {"name": prim.GetName(), "type": prim.GetTypeName().replace("Physics", ""), "body0": b0[0] if b0 else None,
             "body1": b1[0] if b1 else None}
        if d["body0"]:
            T0 = in_root(st.GetPrimAtPath(d["body0"]))
            p0 = np.array(j.GetLocalPos0Attr().Get() or (0, 0, 0), dtype=float)
            q0 = j.GetLocalRot0Attr().Get()
            R0 = quat_wxyz_to_R_np([q0.GetReal(), *q0.GetImaginary()]) if q0 is not None else np.eye(3)
            S = np.linalg.norm(T0[:3, :3], axis=0)  # body scale applies to localPos
            d["anchor"] = np.round(T0[:3, :3] @ p0 * mpu + T0[:3, 3], 6).tolist()
            R_anchor = (T0[:3, :3] / S) @ R0
        else:
            R_anchor = np.eye(3)
        ax_attr = prim.GetAttribute("physics:axis")
        if ax_attr and ax_attr.Get():
            e = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[str(ax_attr.Get())]
            d["axis"] = np.round(R_anchor @ np.array(e, float), 6).tolist()
        lo, hi = prim.GetAttribute("physics:lowerLimit"), prim.GetAttribute("physics:upperLimit")
        unit = mpu if "Prismatic" in d["type"] else 1.0  # prismatic limits in stage units, revolute in degrees
        if lo and lo.Get() is not None and hi and hi.Get() is not None:
            d["limits"] = [round(float(lo.Get()) * unit, 6), round(float(hi.Get()) * unit, 6)]
        for kind in ("linear", "angular"):
            tp = prim.GetAttribute(f"drive:{kind}:physics:targetPosition")
            if tp and tp.Get() is not None:
                d["drive_target"] = round(float(tp.Get()) * (mpu if kind == "linear" else 1.0), 6)
                d["drive_stiffness"] = float(prim.GetAttribute(f"drive:{kind}:physics:stiffness").Get() or 0)
        if d["body1"]:
            r = bc.ComputeUntransformedBound(st.GetPrimAtPath(d["body1"])).ComputeAlignedRange()
            T1 = in_root(st.GetPrimAtPath(d["body1"]))
            lo3, hi3 = np.array(r.GetMin(), float), np.array(r.GetMax(), float)
            C = np.array([[x, y, z] for x in (lo3[0], hi3[0]) for y in (lo3[1], hi3[1]) for z in (lo3[2], hi3[2])])
            W = (C * mpu) @ T1[:3, :3].T + T1[:3, 3]  # body-local stage units -> asset frame, metres
            d["body1_bbox"] = [np.round(W.min(0), 6).tolist(), np.round(W.max(0), 6).tolist()]
            top = W.max(0)
            d["press_point"] = np.round([(W[:, 0].min() + W[:, 0].max()) / 2, (W[:, 1].min() + W[:, 1].max()) / 2, top[2]], 6).tolist()
        joints.append(d)
    return {"joints": joints} if joints else None


def quat_wxyz_to_R_np(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def annotations(asset_dir: Path, usd: Path, md: dict) -> dict:
    """Checker-relevant attributes: articulation joints (press / open axes), the metadata's functional frames,
    per-category attributes (number_card digit)."""
    out = {}
    func = (md.get("passive") or {}).get("functional")
    if func:
        out["functional"] = func
    sup = (md.get("passive") or {}).get("support")
    if sup:  # support frames (e.g. the tic-tac-toe board's cell/0..8: centre pose in the asset frame + radius)
        out["support"] = sup
    if asset_dir.parent.parent.name == "Articulation":
        try:
            a = articulation_info(usd)
            if a:
                out["articulation"] = a
        except Exception as e:  # a broken joint must not cost the asset its geometry
            out["articulation_error"] = f"{type(e).__name__}: {e}"[:200]
    f = CATEGORY_ATTRIBUTES.get(asset_dir.parent.name)
    if f:
        out["attributes"] = f(asset_dir.name)
    desc = asset_dir / "description.json"
    if desc.exists():  # two formats: {"caption": {name, color, ...}} or {"description": "yellow ice axe"}
        try:
            d = json.loads(desc.read_text())
            if d.get("description"):
                out["description"] = str(d["description"])
            cap = d.get("caption") or {}
            if cap.get("name"):
                out["names_all"] = [str(n) for n in cap["name"]]
            if cap.get("color"):
                out["colors_all"] = [str(n) for n in cap["color"]]
        except Exception:
            pass
    return out


def precompute(out: Path, keys: list[str]) -> int:
    """Runtime caches next to each asset: ``sil.npz`` (the decimated silhouette mesh render.Asset would build)
    and ``shape_cache.json`` (rotational symmetry per place frame). Cold loads drop from ~0.4 s to ~0.05 s per
    asset (general_pickup: 85 candidates)."""
    os.environ.setdefault("DOJO_CATALOG", str(out))
    from .inverse_layout import _place_R0, world_z_symmetry
    from .render import Asset

    n = 0
    for key in keys:
        d = out / key
        if not (d / "geom.npz").exists():
            continue
        a = Asset(key, root=out, use_cache=False)
        if a.Fs is not a.F:
            np.savez_compressed(d / "sil.npz", vertices=a.Vs, faces=a.Fs,  # exact: results stay bit-identical
                                cell=np.float32(a.sil_cell))
        frames = a.info.get("place") or {"default": {"center": [0, 0, 0, 1, 0, 0, 0]}}
        sym = {tag: int(world_z_symmetry(a.V @ _place_R0(fr).T, a.F)) for tag, fr in frames.items()}
        (d / "shape_cache.json").write_text(json.dumps({"sym": sym, "n_faces": int(len(a.F))}))
        n += 1
    return n


def annotate(assets: Path, out: Path, keys: list[str]) -> int:
    """Add ``annotations`` to already-converted assets' info.json (no re-conversion)."""
    n = 0
    for key in keys:
        src, info_p = assets / "Object" / "RoboDojo" / key, out / key / "info.json"
        usd = _asset_file(src)
        if usd is None or not info_p.exists():
            print("skip", key, flush=True)
            continue
        md_path = src / "metadata.json"
        md = json.loads(md_path.read_text()) if md_path.exists() else {}
        info = json.loads(info_p.read_text())
        info.update(annotations(src, usd, md))
        info_p.write_text(json.dumps(info, indent=1))
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--assets", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--only", nargs="*")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--force", action="store_true")
    b.add_argument("--no-glb", action="store_true", help="geometry + info only (no visual.glb)")
    b.add_argument("--keys-json", type=Path, default=None, help="JSON list of Type/category/idx keys to build")
    i = sub.add_parser("index")
    i.add_argument("--out", type=Path, required=True)
    lo = sub.add_parser("lowres")
    lo.add_argument("--out", type=Path, required=True)
    an = sub.add_parser("annotate", help="add joints / functional frames / category attributes to info.json")
    an.add_argument("--assets", type=Path, required=True)
    an.add_argument("--out", type=Path, required=True)
    an.add_argument("--keys-json", type=Path, required=True)
    pc = sub.add_parser("precompute", help="runtime caches: sil.npz + shape_cache.json per asset")
    pc.add_argument("--out", type=Path, required=True)
    pc.add_argument("--keys-json", type=Path, default=None, help="default: every asset in the catalogue")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        keys = json.loads(a.keys_json.read_text()) if a.keys_json else None
        build(a.assets, a.out, a.only, a.workers, a.force, glb=not a.no_glb, keys=keys)
    elif a.cmd == "precompute":
        keys = (json.loads(a.keys_json.read_text()) if a.keys_json else
                ["/".join(p.parts[-4:-1]) for p in sorted(a.out.glob("*/*/*/geom.npz"))])
        print(f"precomputed {precompute(a.out, keys)} assets")
    elif a.cmd == "annotate":
        print(f"annotated {annotate(a.assets, a.out, json.loads(a.keys_json.read_text()))} assets")
    elif a.cmd == "lowres":
        n = 0
        for info in sorted(a.out.glob("*/*/*/info.json")):
            n += write_lowres(info.parent)
        print(f"visual_lo.glb written for {n} assets")
    else:
        write_index(a.out)


if __name__ == "__main__":
    sys.exit(main())
