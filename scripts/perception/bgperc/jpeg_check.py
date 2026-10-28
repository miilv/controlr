"""Stream C: does JPEG q95 (4:4:4) change B1 / refine against the lossless frame? Same steps, both encodings.
    python scripts/perception/bgperc/jpeg_check.py SEQ_DIR(png) [--steps 14,20,30] [--fold F]"""
import argparse
import io
import json
import sys
import time

import numpy as np


def jpeg(im, q=95):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(im).save(b, format="JPEG", quality=q, subsampling=0)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--steps", default="")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--q", type=int, default=95)
    a = ap.parse_args()
    import zlib

    from dojo.perception.eval_g1 import TOLERANCE, failure_class, score_record
    from dojo.perception.seq_data import Sequence
    from dojo.runtime.perception_service import _Perception, encode_frame
    seq = Sequence(a.seq)
    fold = a.fold if a.fold is not None else zlib.crc32(str(seq.layout_id).encode()) % 2
    P = _Perception()
    P.set_scene(seq.task)
    f0 = seq.frame(0, fmt="png")
    fj = {**f0, "images": {c: jpeg(im, a.q) for c, im in f0["images"].items()}}
    out = {"task": seq.task, "layout": seq.layout_id, "fold": fold}
    recs = {}
    for name, fr in (("png", f0), ("jpg", fj)):
        t = time.time()
        recs[name] = P.estimate([encode_frame(fr)], seq.task, fold)
        out[f"b1_s_{name}"] = round(time.time() - t, 2)
    gt = {"objects": seq.gt(0)}
    tol = TOLERANCE.get(seq.task, (15, None, ""))
    rows = {}
    for name, rec in recs.items():
        rs, _, _ = score_record(rec, gt, seq.task, {})
        rows[name] = {r["label"]: (failure_class(r, tol), round(r.get("xy_err_mm", -1), 1), r.get("pred_idx")) for r in rs}
    diff = []
    for lab in rows["png"]:
        p, j = rows["png"][lab], rows["jpg"].get(lab)
        if j is None or p[0] != j[0] or p[2] != j[2] or abs(p[1] - j[1]) > 2.0:
            diff.append((lab, p, j))
    out["b1_labels"] = len(rows["png"])
    out["b1_diff"] = diff
    out["b1_png"] = rows["png"]
    # refine: every GT object (prior = GT + 8 mm), every wrist, at the given steps
    from dojo.perception.refine import refine
    steps = [int(s) for s in a.steps.split(",") if s] or [seq.steps[len(seq.steps) // 2]]
    rd = []
    for s in steps:
        fr = seq.frame(s, fmt="png")
        for o in seq.gt(s):
            if o.get("clutter") or o.get("pos") is None or o.get("type") not in ("Rigid",):
                continue
            key = f"{o['type']}/{o['category']}/{o['category_idx']:05d}"
            prior = list(np.asarray(o["pos"]) + np.r_[0.008, 0.0, 0.0]) + list(o["quat"])
            for arm in ("left", "right"):
                res = {}
                for name in ("png", "jpg"):
                    img = fr["images"][f"cam_{arm}_wrist"]
                    if name == "jpg":
                        img = jpeg(img, a.q)
                    try:
                        r = refine(key, img, fr["state"], prior, arm=arm)
                    except Exception as e:  # noqa: BLE001
                        r = {"ok": False, "why": f"{type(e).__name__}: {e}"}
                    pose = r.get("pose")
                    err = None if pose is None else round(1000 * float(np.hypot(*(np.asarray(pose[:2]) - np.asarray(o["pos"][:2])))), 1)
                    res[name] = (bool(r.get("ok")), err, round(float(r.get("iou") or 0), 3), round(float(r.get("time_s") or 0), 3))
                if res["png"][0] or res["jpg"][0]:
                    rd.append({"step": s, "label": o["label"], "arm": arm, **res})
    out["refine"] = rd
    agree = [r for r in rd if r["png"][0] == r["jpg"][0] and (not r["png"][0] or abs(r["png"][1] - r["jpg"][1]) <= 1.5)]
    out["refine_agree"] = f"{len(agree)}/{len(rd)}"
    json.dump(out, sys.stdout, indent=1, default=str)


if __name__ == "__main__":
    main()
