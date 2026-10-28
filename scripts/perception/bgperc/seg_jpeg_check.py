"""Stream C, X2: does JPEG q95 (the sequences' storage) change the frozen segmenter's detections against the lossless
frame? Every sequence keeps frame 0 as PNG and JPEG: run the segmenter on both, match detections greedily by mask IoU,
and report the matched IoU, class agreement, and detections gained / lost (JPEG vs PNG).

    CUDA_VISIBLE_DEVICES= SEG_WEIGHTS=.../v2b/best.pt SEG_CLASSES=.../v2b/classes.json \\
        python scripts/perception/bgperc/seg_jpeg_check.py SEQ_ROOT [--valid VALID.json] [--out OUT.json]
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np


def load(path: str) -> np.ndarray:
    import cv2
    return np.ascontiguousarray(cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--valid", default=None, help="only the episodes this validity file marks valid")
    ap.add_argument("--out", default=None)
    ap.add_argument("--iou", type=float, default=0.5, help="a JPEG detection matches a PNG one at this mask IoU")
    a = ap.parse_args()
    from dojo.perception.seg_twin import _SegModel, iou
    w, c = os.environ["SEG_WEIGHTS"], os.environ["SEG_CLASSES"]
    classes = json.load(open(c))
    classes = classes if isinstance(classes, list) else classes.get("classes") or classes.get("names")
    model = _SegModel(w, classes)
    wh = hashlib.sha1(open(w, "rb").read()).hexdigest()[:12]
    names = sorted(os.listdir(a.root))
    if a.valid:
        ok = {r["name"] for r in json.load(open(a.valid)) if r.get("valid")}
        names = [n for n in names if n in ok]
    tot = {"frames": 0, "png_dets": 0, "jpg_dets": 0, "matched": 0, "class_same": 0, "lost": 0, "gained": 0}
    ious = []
    per = {}
    for n in names:
        for png in sorted(glob.glob(os.path.join(a.root, n, "frames", "0000_*.png"))):
            jpg = png[:-4] + ".jpg"
            if not os.path.exists(jpg):
                continue
            dp, dj = model.detect(load(png)), model.detect(load(jpg))
            used, m = set(), 0
            for mp, kp, _sp, _bp in dp:
                best = max(((iou(mp, mj), j) for j, (mj, _kj, _sj, _bj) in enumerate(dj) if j not in used),
                           default=(0.0, None))
                if best[1] is not None and best[0] >= a.iou:
                    used.add(best[1])
                    m += 1
                    ious.append(best[0])
                    tot["class_same"] += int(dj[best[1]][1] == kp)
            tot["frames"] += 1
            tot["png_dets"] += len(dp)
            tot["jpg_dets"] += len(dj)
            tot["matched"] += m
            tot["lost"] += len(dp) - m
            tot["gained"] += len(dj) - len(used)
            per[os.path.relpath(png, a.root)] = {"png": len(dp), "jpg": len(dj), "matched": m}
    differ = tot["lost"] + tot["gained"] + (tot["matched"] - tot["class_same"])
    res = {"weights_sha1": wh, "imgsz": model.imgsz, "conf": model.conf, "iou_thr": a.iou, **tot,
           "matched_iou_mean": round(float(np.mean(ious)), 4) if ious else None,
           "matched_iou_p5": round(float(np.percentile(ious, 5)), 4) if ious else None,
           "differ_frac": round(differ / max(1, tot["png_dets"]), 4), "per_frame": per}
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
