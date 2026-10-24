"""Render D0 states with Isaac Sim 6.0 as a PROXY for RoboDojo's Isaac Sim 5.1 renderer (compute3).

Why: Isaac Sim 5.1's RTX segfaults at startup on compute3 (RTX 5090, driver 595), so the D0 states
are produced by the physics-only 5.1 twin (``make_d0 --physics-only``), which writes per layout
``stage.usd`` (RoboDojo's own stage: room, MDL table/ground, dome light, robots, objects; asset
references absolute), ``cameras.json`` (RoboDojo's camera definitions) and ``frame_<k>.json`` (GT
object poses and every robot link pose). This script opens that exact stage in 6.0, writes the
recorded poses onto the prims (no physics), builds the three cameras as RoboDojo's CameraManager
does (Xform at the mount link / env root, local pose from ``pos`` + intrinsic-XYZ Euler degrees,
pinhole camera at identity, template focal length + apertures), renders 640x480 RGB with
IsaacLab's "quality" settings + DLAA (RoboDojo's DEFAULT_RENDER_CONFIG), and fills in the PNGs and
``meta.renderer = "isaac-6.0 (proxy for 5.1)"``.

Runs inside Isaac Sim 6.0's python (``scripts/simlab/isaac6.sh``); stdlib + numpy + pxr only.

    scripts/simlab/isaac6.sh dojo/simlab/render_d0_proxy.py --root ~/dojo-data/d0 --tasks stack_blocks push_T
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

RENDERER_TAG = "isaac-6.0 (proxy for 5.1)"
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
# IsaacLab apps/rendering_modes/quality.kit (RoboDojo: rendering_mode "quality") + RoboDojo's
# DEFAULT_RENDER_CONFIG (translucency, reflections, GI on; DLSS mode 2; DLAA via replicator).
QUALITY = {
    "/rtx/translucency/enabled": True, "/rtx/reflections/enabled": True, "/rtx/reflections/denoiser/enabled": True,
    "/rtx/directLighting/sampledLighting/enabled": True, "/rtx/sceneDb/ambientLightIntensity": 1.0,
    "/rtx/shadows/enabled": True, "/rtx/indirectDiffuse/enabled": True, "/rtx/indirectDiffuse/denoiser/enabled": True,
    "/rtx/ambientOcclusion/enabled": True, "/rtx/ambientOcclusion/denoiserMode": 0,
    "/rtx/raytracing/subpixel/mode": 1, "/rtx/raytracing/cached/enabled": True, "/rtx-transient/dlssg/enabled": False,
    "/rtx-transient/dldenoiser/enabled": True, "/rtx/post/dlss/execMode": 2, "/omni/replicator/RTSubframes": 3,
}


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal PNG writer (no PIL dependency in Isaac's python)."""
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
        return
    except Exception:  # noqa: BLE001
        pass
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = t + d
        return len(d).to_bytes(4, "big") + c + zlib.crc32(c).to_bytes(4, "big")
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0]))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    path.write_bytes(png)


def euler_xyz_intrinsic_deg_to_quat(e) -> tuple[float, float, float, float]:
    """RoboDojo ``utils.rotations.euler_angles_to_quat(degrees=True)`` (default extrinsic=False, i.e.
    scipy "XYZ" intrinsic): R = Rx(a) Ry(b) Rz(c); wxyz."""
    a, b, c = (math.radians(v) / 2 for v in e)
    qx = (math.cos(a), math.sin(a), 0.0, 0.0)
    qy = (math.cos(b), 0.0, math.sin(b), 0.0)
    qz = (math.cos(c), 0.0, 0.0, math.sin(c))

    def mul(p, q):
        w1, x1, y1, z1 = p
        w2, x2, y2, z2 = q
        return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    return mul(qx, mul(qy, qz))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="D0 root (contains <task>/<id>/)")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--subframes", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-stage", action="store_true", help="keep stage.usd after rendering (38 MB each)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if "Eval_Layout" in root.resolve().parts:
        raise SystemExit("refusing eval data")

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 640, "height": 480, "renderer": "RaytracedLighting",
                         "anti_aliasing": 3, "multi_gpu": False, "sync_loads": True})
    import carb
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom

    st = carb.settings.get_settings()
    for k, v in QUALITY.items():
        st.set(k, v)
    try:
        rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
    except Exception as e:  # noqa: BLE001
        print(f"[render] DLAA not set: {e}", flush=True)
    ctx = omni.usd.get_context()

    def set_world_pose(stage, cache, path: str, pos, quat) -> bool:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return False
        xf = UsdGeom.Xformable(prim)
        cur = cache.GetLocalToWorldTransform(prim)
        scale = [Gf.Vec3d(cur.GetRow3(i)).GetLength() for i in range(3)]     # world scale (rows = axes)
        q = np.asarray(quat, float)
        q = q / np.linalg.norm(q)                  # IsaacLab poses are float32, not exactly unit
        rot = Gf.Matrix3d(Gf.Rotation(Gf.Quatd(*map(float, q))))
        world = Gf.Matrix4d(1.0).SetScale(Gf.Vec3d(*scale)) * Gf.Matrix4d(rot, Gf.Vec3d(*map(float, pos)))
        parent = cache.GetLocalToWorldTransform(prim.GetParent())
        xf.MakeMatrixXform().Set(world * parent.GetInverse())
        return True

    todo = []
    for task in a.tasks:
        for d in sorted((root / task).iterdir()) if (root / task).is_dir() else []:
            if not d.is_dir() or not d.name.isdigit() or (a.ids and int(d.name) not in a.ids):
                continue
            if not (d / "stage.usd").exists() and not a.force:
                continue
            todo.append(d)
    print(f"[render] {len(todo)} layouts", flush=True)
    for d in todo:
        t0 = time.perf_counter()
        ctx.open_stage(str(d / "stage.usd"))
        for _ in range(200):
            app.update()
            if not ctx.is_new_stage() and ctx.get_stage_loading_status()[2] == 0:
                break
        stage = ctx.get_stage()
        cams = json.loads((d / "cameras.json").read_text())
        rps, anns = {}, {}
        for name in CAMS:
            c = cams[name]
            parent = c["mount"] or "/World/envs/env_0"
            xpath = f"{parent}/SimlabCam_{name}"
            xform = UsdGeom.Xform.Define(stage, xpath)
            q = euler_xyz_intrinsic_deg_to_quat(c["ori_euler_deg"])
            xform.MakeMatrixXform().Set(Gf.Matrix4d(Gf.Rotation(Gf.Quatd(*q)), Gf.Vec3d(*map(float, c["pos"]))))
            cam = UsdGeom.Camera.Define(stage, f"{xpath}/{name}")
            cam.CreateFocalLengthAttr().Set(float(c["focal_length"]))
            cam.CreateHorizontalApertureAttr().Set(float(c["horizontal_aperture"]))
            cam.CreateVerticalApertureAttr().Set(float(c["vertical_aperture"]))
            cam.CreateClippingRangeAttr().Set(Gf.Vec2f(*map(float, c["clipping_range"])))
            rps[name] = rep.create.render_product(f"{xpath}/{name}", tuple(c["resolution"]))
            anns[name] = rep.AnnotatorRegistry.get_annotator("rgb")
            anns[name].attach([rps[name]])
        for _ in range(20):          # shader / texture warm-up
            app.update()
        n_frames = 0
        for fj in sorted(d.glob("frame_*.json")):
            rec = json.loads(fj.read_text())
            if rec.get("meta", {}).get("renderer") and not a.force:
                continue
            gt = rec["gt"]
            org = np.asarray(gt.get("env_origin", [0, 0, 0]), float)
            cache = UsdGeom.XformCache()
            missing = []
            for o in gt["objects"]:
                if o.get("pos") is None or not o.get("prim_path"):
                    continue
                if not set_world_pose(stage, cache, o["prim_path"], np.asarray(o["pos"]) + org, o["quat"]):
                    missing.append(o["prim_path"])
            for arm, r in gt["robot"].items():
                lp = r.get("link_prims", {})
                for ln in sorted(r["links"], key=lambda n: lp.get(n, "").count("/")):
                    if ln not in lp:
                        continue
                    pose = np.asarray(r["links"][ln], float)
                    cache.Clear()
                    if not set_world_pose(stage, cache, lp[ln], pose[:3] + org, pose[3:]):
                        missing.append(lp[ln])
            rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0, pause_timeline=False)
            shapes = {}
            for name in CAMS:
                rgb = np.asarray(anns[name].get_data())[..., :3].astype(np.uint8)
                write_png(d / rec["obs"]["images"][name], np.ascontiguousarray(rgb))
                shapes[name] = list(rgb.shape)
            rec["obs"]["image_shape"] = shapes
            rec.setdefault("meta", {})
            rec["meta"].update({"renderer": RENDERER_TAG, "render_subframes": a.subframes,
                                "render_settings": "IsaacLab quality.kit + DLAA (RoboDojo DEFAULT_RENDER_CONFIG)",
                                "pose_prims_missing": missing})
            fj.write_text(json.dumps(rec))
            n_frames += 1
        for name in CAMS:
            anns[name].detach([rps[name]])
            rps[name].destroy()
        if not a.keep_stage:
            (d / "stage.usd").unlink(missing_ok=True)
        print(f"[render] {d.parent.name}/{d.name}: {n_frames} frames in {time.perf_counter() - t0:.1f}s", flush=True)
    app.close()


if __name__ == "__main__":
    sys.exit(main())
