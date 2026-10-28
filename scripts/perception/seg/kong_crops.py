"""make_kong glyph crops for a tile classifier (stream S). DEV ONLY (GT masks + asset ids).

Every visible mahjong tile in the segmenter's renders (Isaac instance masks; ``seg_render`` dirs of 100030-49 variants:
train, the ``--val-ids`` draws: val) and in the held-out D0 frames (GT-pose masks from ``eval.py gt``; d0_ext
make_kong, ids 100000-09 / 100020-24: test) -> a square crop around its mask (+25 % margin, the background kept),
resized to ``--size`` px, under ``<out>/<split>/<category_idx>/``. ``meta.jsonl``: split, source, view, arm pose,
visible px, the tile's tilt from face-up (deg), the crop path.

    python scripts/perception/seg/kong_crops.py --render ~/dojo-data/segtrain/render,~/dojo-data/segtrain/render_held,...test \\
        --gt ~/dojo-data/segtrain/eval_gt \\
        --out ~/dojo-data/segtrain/kong_crops [--val-ids 100049] [--size 128] [--min-px 40]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
D0_EXT = Path.home() / "dojo-data/d0_ext"


def tilt_deg(quat) -> float:
    """Angle between the tile's local +z and world +z (0 = lying as authored)."""
    from dojo.perception.cameras import quat_wxyz_to_R

    return float(np.degrees(np.arccos(np.clip(quat_wxyz_to_R(quat)[2, 2], -1.0, 1.0))))


def crop(img: np.ndarray, mask: np.ndarray, size: int, margin: float = 0.25) -> np.ndarray:
    import cv2

    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min(), 8) * (0.5 + margin)
    x0, y0, x1, y1 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    H, W = img.shape[:2]
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return cv2.resize(img[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA if x1 - x0 > size else cv2.INTER_LINEAR)


def main(argv=None):
    import cv2
    from PIL import Image

    from dojo.perception.seg_twin import instances

    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-ids", default="100049")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--min-px", type=int, default=40)
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    val_ids = {s.strip() for s in a.val_ids.split(",")}
    meta = open(out / "meta.jsonl", "w") if out.mkdir(parents=True, exist_ok=True) is None else None
    n = 0

    def face_dot(o, cams, view):
        """cos of the angle between the tile's face normal (local +z) and the direction to the camera."""
        from dojo.perception.cameras import quat_wxyz_to_R

        c = np.asarray(cams[view]["T_world_cam_cv"], float)[:3, 3]
        v = c - np.asarray(o["pos"], float)
        return float(quat_wxyz_to_R(o["quat"])[:, 2] @ (v / np.linalg.norm(v)))

    def emit(split, src, img, codes, objs, view, pose, stem, cams=None, held_inst=None):
        nonlocal n
        for i, m in instances(codes, a.min_px).items():
            if i >= len(objs) or objs[i].get("category") != "mahjong" or objs[i].get("pos") is None:
                continue
            c = crop(img, m, a.size)
            cls = int(objs[i]["category_idx"])
            p = out / split / f"{cls:02d}" / f"{stem}_{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
            meta.write(json.dumps({"split": split, "src": src, "view": view, "pose": pose, "px": int(m.sum()),
                                   "tilt": round(tilt_deg(objs[i]["quat"]), 1), "cls": cls,
                                   "face_dot": None if cams is None else round(face_dot(objs[i], cams, view), 3),
                                   "is_held": bool(held_inst) and objs[i].get("inst_name") == held_inst,
                                   "path": str(p.relative_to(out))}) + "\n")
            n += 1

    dirs = [d for root in a.render.split(",") for d in sorted(Path(root).expanduser().glob("make_kong/*"))]
    for d in dirs:
        if not (d / "done.json").exists():
            continue
        if "test" in d.parent.parent.name:     # seg_render --test: held-out ids, stored records
            split = "test"
        else:
            split = "val" if d.name.split("_")[0] in val_ids else "train"
        for fj in sorted(d.glob("frame_*.json")):
            rec = json.loads(fj.read_text())
            k = rec["frame"]
            mv_ = rec.get("move") or {}
            pose = mv_.get("kind") or "home"
            if pose == "held":
                pose = f"held_{mv_.get('held_stage')}"
            for cam in CAMS:
                ip = d / f"inst_{k}_{cam}.png"
                if not ip.exists():
                    continue
                img = np.asarray(Image.open(d / f"frame_{k}_{cam}.png").convert("RGB"))
                codes = np.asarray(Image.open(ip)).astype(np.int64)
                emit(split, "render", img, codes, rec["gt"]["objects"], cam, pose, f"{d.parent.parent.name}_{d.name}_f{k}_{cam}",
                     rec["gt"]["cameras"], mv_.get("target_inst") if pose.startswith("held") else None)
    for mp in sorted(Path(a.gt).expanduser().glob("d0_ext/make_kong/*/gt_*.json")):
        m = json.loads(mp.read_text())
        ep, k = int(m["episode"]), m["frame"]
        if not (100000 <= ep <= 100009 or 100020 <= ep <= 100024):
            continue
        d = D0_EXT / "make_kong" / m["episode"]
        rec = json.loads((d / f"frame_{k}.json").read_text())
        pose = "home" if k == 0 else "look"
        for cam in m["cams"]:
            img = np.asarray(Image.open(d / f"frame_{k}_{cam}.png").convert("RGB"))
            codes = np.asarray(Image.open(mp.parent / f"gt_{k}_{cam}.png")).astype(np.int64)
            emit("test", "d0_ext", img, codes, rec["gt"]["objects"], cam, pose, f"{m['episode']}_f{k}_{cam}", rec["gt"]["cameras"])
    meta.close()
    print(f"[kong] {n} crops -> {out}", flush=True)


if __name__ == "__main__":
    main()
