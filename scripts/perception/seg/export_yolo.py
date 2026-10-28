"""Twin-trained segmenter: seg_render's labelled frames -> an Ultralytics YOLO-seg dataset. DEV ONLY (reads GT).

    python scripts/perception/seg/export_yolo.py --render ~/dojo-data/segtrain/render --out ~/dojo-data/segtrain/yolo_v1 \\
        [--val-ids 100049] [--cams cam_head,cam_left_wrist,cam_right_wrist] [--workers 8]

Per frame and camera: the instance codes (``inst_<j>_<cam>.png`` from Isaac's annotator; without one, ``gt_code_map``
from the frame's GT poses and GT cameras), every object with >= MIN_PX visible pixels -> one polygon (fragments joined)
with class = its catalogue asset (``Type/category/idx``; task objects, clutter and fixtures alike; the arms are not a
class). Images are symlinked. ``classes.json``: names (class id order) + instance counts; ``data.yaml`` for training
(``--rfs-t``: its train split is ``train_rfs.txt``, rare classes' images repeated).
Val split: the layouts of ``--val-ids`` (training draws too, never an evaluation id).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def repeat_factors(img_classes: list[set], t: float) -> list[float]:
    """LVIS repeat-factor sampling: per image, max over its classes of max(1, sqrt(t / f(c))), f(c) = the share of
    images that contain class c (an image without objects: 1)."""
    import math

    n = max(len(img_classes), 1)
    freq = Counter(c for cs in img_classes for c in cs)
    return [max([max(1.0, math.sqrt(t / (freq[c] / n))) for c in cs] or [1.0]) for cs in img_classes]


def frame_jobs(render: Path) -> list[tuple]:
    out = []
    for done in sorted(render.glob("*/*/done.json")):
        d = done.parent
        task = d.parent.name
        for fj in sorted(d.glob("frame_*.json")):
            k = fj.stem.split("_")[1]
            if k.isdigit():
                out.append((task, d.name, str(d), int(k)))
    return out


def codes_for(d: Path, k: int, cam: str, rec: dict) -> tuple[np.ndarray, str]:
    from PIL import Image

    p = d / f"inst_{k}_{cam}.png"
    if p.exists():
        return np.asarray(Image.open(p)).astype(np.uint16), "isaac"
    from dojo.perception.seg_twin import gt_camera, gt_code_map

    st = {kk: np.asarray(v, float) for kk, v in rec["obs"]["state"].items()}
    codes, _ = gt_code_map(gt_camera(rec["gt"], cam), rec["gt"]["objects"], st)
    return codes, "render"


def label_frame(job, cams=CAMS, min_px: int = 20) -> list[dict]:
    """[{"img": path, "cam", "source", "objs": [(key, polygon list, px, clutter)]}] for one frame."""
    from dojo.perception.seg_twin import asset_key, instances, mask_polygon

    task, lay, d, k = job
    d = Path(d)
    rec = json.loads((d / f"frame_{k}.json").read_text())
    objs = rec["gt"]["objects"]
    out = []
    for cam in cams:
        img = d / (rec["obs"].get("images") or {}).get(cam, f"frame_{k}_{cam}.png")   # .jpg from the 6.0 proxy --jpeg
        if not img.exists():
            continue
        codes, src = codes_for(d, k, cam, rec)
        rows = []
        for i, m in instances(codes, min_px).items():
            if i >= len(objs):
                continue
            poly = mask_polygon(m)
            if poly is None:
                continue
            rows.append((asset_key(objs[i]), poly.round(2).tolist(), int(m.sum()), bool(objs[i].get("clutter"))))
        out.append({"img": str(img), "task": task, "layout": lay, "frame": k, "cam": cam, "source": src,
                    "h": int(codes.shape[0]), "w": int(codes.shape[1]), "objs": rows})
    return out


def main(argv=None):
    from dojo.perception.seg_twin import yolo_line

    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-ids", default="100049")
    ap.add_argument("--cams", default=",".join(CAMS))
    ap.add_argument("--min-px", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rfs-t", type=float, default=0.0,
                    help="repeat-factor sampling threshold (LVIS; e.g. 0.02): a train list with rare-class images repeated")
    a = ap.parse_args(argv)
    render, out = Path(a.render).expanduser(), Path(a.out).expanduser()
    val_ids = {s.strip() for s in a.val_ids.split(",") if s.strip()}
    cams = tuple(a.cams.split(","))
    jobs = frame_jobs(render)
    print(f"[export] {len(jobs)} frames", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        frames = [f for fr in ex.map(label_frame, jobs, [cams] * len(jobs), [a.min_px] * len(jobs), chunksize=4) for f in fr]
    counts = Counter(k for f in frames for k, *_ in f["objs"])
    names = sorted(counts)
    cid = {k: i for i, k in enumerate(names)}
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    src = Counter()
    n_split = Counter()
    for f in frames:
        split = "val" if f["layout"].split("_")[0] in val_ids else "train"
        stem = f"{f['task']}__{f['layout']}__f{f['frame']}__{f['cam']}"
        dst = out / "images" / split / f"{stem}{Path(f['img']).suffix}"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(f["img"], dst)
        lines = [yolo_line(cid[k], np.asarray(p), f["w"], f["h"]) for k, p, _, _ in f["objs"]]
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        src[f["source"]] += 1
        n_split[split] += 1
    (out / "classes.json").write_text(json.dumps({"names": names, "counts": {k: counts[k] for k in names},
                                                  "frames": len(frames), "splits": dict(n_split), "sources": dict(src),
                                                  "val_ids": sorted(val_ids)}, indent=1))
    (out / "frames.jsonl").write_text("".join(json.dumps({kk: v for kk, v in f.items() if kk != "objs"} |
                                                         {"n": len(f["objs"]), "keys": [o[0] for o in f["objs"]]}) + "\n"
                                              for f in frames))
    train = "images/train"
    if a.rfs_t > 0:
        # LVIS repeat-factor sampling: an image is listed r = max over its classes of max(1, sqrt(t / f(c))) times
        # (f = share of training images with class c, rounded stochastically): rare look-alikes are seen more often
        import random

        tr = [f for f in frames if f["layout"].split("_")[0] not in val_ids]
        rnd = random.Random(0)
        lines = []
        for f, r in zip(tr, repeat_factors([{k for k, *_ in f["objs"]} for f in tr], a.rfs_t)):
            n = int(r) + (1 if rnd.random() < r - int(r) else 0)
            stem = f"{f['task']}__{f['layout']}__f{f['frame']}__{f['cam']}{Path(f['img']).suffix}"
            lines += [str(out / "images" / "train" / stem)] * n
        (out / "train_rfs.txt").write_text("\n".join(lines) + "\n")
        train = "train_rfs.txt"
        print(f"[export] repeat-factor sampling t={a.rfs_t}: {len(tr)} train images listed {len(lines)} times", flush=True)
    yaml = [f"path: {out}", f"train: {train}", "val: images/val", "names:"] + [f"  {i}: '{k}'" for i, k in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")
    print(f"[export] {len(frames)} images ({dict(n_split)}), {len(names)} classes, "
          f"{sum(counts.values())} instances, label source {dict(src)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
