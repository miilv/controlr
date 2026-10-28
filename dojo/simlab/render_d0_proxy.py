"""Render D0 states with Isaac Sim 6.0 as a PROXY for RoboDojo's Isaac Sim 5.1 renderer (compute3).

Why: Isaac Sim 5.1's RTX segfaults at startup on compute3 (RTX 5090, driver 595), so the D0 states
are produced by the physics-only 5.1 twin (``make_d0 --physics-only``), which writes per layout
``stage.usd`` (RoboDojo's own stage), ``cameras.json`` (RoboDojo's camera definitions) and
``frame_<k>.json`` (GT object poses and every robot link pose). ``proxy_render.ProxyRenderer`` opens that
stage in 6.0, writes the recorded poses, builds the cameras and renders; this script fills in the PNGs,
``gt.cameras`` and ``meta.renderer = "isaac-6.0 (proxy for 5.1)"``.

    scripts/simlab/isaac6.sh dojo/simlab/render_d0_proxy.py --root ~/dojo-data/d0 --tasks stack_blocks push_T
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # repo root (Isaac's python)
from dojo.simlab.proxy_render import (APP_CONFIG, CAMS, RENDERER_TAG,  # noqa: E402
                                      euler_xyz_intrinsic_deg_to_quat)

__all__ = ["euler_xyz_intrinsic_deg_to_quat", "write_png", "main"]


def write_png(path: Path, rgb: np.ndarray) -> None:
    """PNG via PIL if present, else a minimal zlib writer (Isaac's python may lack PIL)."""
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


def write_jpeg(path: Path, rgb: np.ndarray, quality: int) -> None:
    from PIL import Image
    Image.fromarray(rgb).save(path, quality=int(quality))


def write_instances(pr, d: Path, rec: dict) -> dict:
    """seg_render's code images (inst_<k>_<cam>.png, uint16) from the proxy's instance-id annotator."""
    from PIL import Image

    from dojo.simlab.seg_render import OTHER_CODE, ROBOT_CODE, map_instance_ids, relabel
    k = rec["frame"]
    inst = {"source": "isaac-6.0:instance_id_segmentation_fast", "codes": {"robot": ROBOT_CODE, "other": OTHER_CODE},
            "px": {}}
    for name in CAMS:
        data, id2l = pr.instance_ids(name)
        if data is None:
            inst["source"] = None
            continue
        if data.ndim == 3:
            data = data[..., 0]
        code = relabel(data, map_instance_ids(id2l, rec["gt"]["objects"], rec["gt"]["robot"]))
        Image.fromarray(code).save(d / f"inst_{k}_{name}.png")
        u, c = np.unique(code, return_counts=True)
        inst["px"][name] = {int(a): int(b) for a, b in zip(u, c)}
    return inst


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="D0 root (contains <task>/<id>/)")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--subframes", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-stage", action="store_true", help="keep stage.usd after rendering (38 MB each)")
    ap.add_argument("--any-dirs", action="store_true", help="every layout dir, not only <id> (seg_render's <id>_v<k>)")
    ap.add_argument("--instance", action="store_true",
                    help="also write inst_<k>_<cam>.png (uint16 seg_render codes) from the instance-id annotator")
    ap.add_argument("--jpeg", type=int, default=0, help="write RGB as JPEG of this quality instead of PNG (0 = PNG)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if "Eval_Layout" in root.resolve().parts:
        raise SystemExit("refusing eval data")
    from isaacsim import SimulationApp
    app = SimulationApp(APP_CONFIG)
    from dojo.simlab.proxy_render import ProxyRenderer
    pr = ProxyRenderer(app)
    todo = []
    for task in a.tasks:
        for d in sorted((root / task).iterdir()) if (root / task).is_dir() else []:
            if not d.is_dir() or not (d.name.isdigit() or a.any_dirs) or \
                    (a.ids and int(d.name.split("_")[0]) not in a.ids):
                continue
            if (d / "stage.usd").exists():
                todo.append(d)
    print(f"[render] {len(todo)} layouts", flush=True)
    for d in todo:
        t0 = time.perf_counter()
        pr.open(str(d / "stage.usd"), json.loads((d / "cameras.json").read_text()), instance=a.instance)
        n_frames = 0
        for fj in sorted(d.glob("frame_*.json")):
            rec = json.loads(fj.read_text())
            if rec.get("meta", {}).get("renderer") and not a.force:
                continue
            images, cam_gt, missing = pr.render(rec["gt"], subframes=a.subframes)
            for name in CAMS:
                if a.jpeg:
                    rec["obs"]["images"][name] = str(Path(rec["obs"]["images"][name]).with_suffix(".jpg"))
                    write_jpeg(d / rec["obs"]["images"][name], images[name], a.jpeg)
                else:
                    write_png(d / rec["obs"]["images"][name], images[name])
            if a.instance:
                rec["inst"] = write_instances(pr, d, rec)
            rec["gt"]["cameras"] = cam_gt
            rec["obs"]["image_shape"] = {n: list(im.shape) for n, im in images.items()}
            rec.setdefault("meta", {}).update({"renderer": RENDERER_TAG, "render_subframes": a.subframes,
                                               "render_settings": "IsaacLab quality.kit + DLAA (RoboDojo DEFAULT_RENDER_CONFIG)",
                                               "pose_prims_missing": missing})
            fj.write_text(json.dumps(rec))
            n_frames += 1
        if not a.keep_stage:
            (d / "stage.usd").unlink(missing_ok=True)
        print(f"[render] {d.parent.name}/{d.name}: {n_frames} frames in {time.perf_counter() - t0:.1f}s", flush=True)
    pr.close_stage()
    app.close()


if __name__ == "__main__":
    sys.exit(main())
