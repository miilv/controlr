"""B1 on recorded frames of any task (belt_bg record layout: <root>/<task>/<id>/frame_<tag>*), scored against the frame's
GT with eval_g1.score_record (Hungarian within interchangeable label groups). Frame 0 alone, or with the scan-pose
frames as extra views (``--scan``: frame_A*). GT for scoring only.

    python scripts/perception/val_frames.py --root ROOT --task TASK --out OUT [--frame W0000] [--scan] [--ids ...]
(env DOJO_BG_FOLDS, DOJO_COLOR_CALIB, DOJO_CFG = EstimatorConfig overrides as JSON)
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _bgs(task):
    import cv2

    from dojo.perception.inverse_layout import task_map
    from dojo.perception.segment import load_backgrounds, with_scene
    d = Path(os.environ["DOJO_BG_FOLDS"]) / "fold0"
    b = load_backgrounds(d)
    m = d / "cam_head_clean_unseen.png"
    if m.exists():
        b["cam_head_clean_unseen"] = (cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    keys = sorted({k for dd in task_map()[task]["labels"].values() if dd.get("type") in ("Dynamic", "Geometry")
                   for k in dd.get("candidates") or []})
    return with_scene(b, d, keys)


def run(args):
    ep, task, frame, scan = args
    from dojo.perception import d0
    from dojo.perception.eval_g1 import score_record
    from dojo.perception.inverse_layout import EstimatorConfig, estimate
    ep = Path(ep)
    cal = json.loads(Path(os.environ["DOJO_COLOR_CALIB"]).read_text())["fold0"]
    cfg = EstimatorConfig(use_wrists=True, color_M=cal, **json.loads(os.environ.get("DOJO_CFG", "{}")))
    obs = d0.load_obs(ep, frame)
    extra = []
    if scan:
        for p in sorted(ep.glob("frame_A*.json")):
            o = d0.load_obs(ep, p.stem[len("frame_"):])
            extra.append((o["images"], o["state"]))
    t0 = time.time()
    rec = estimate(obs["images"], obs["state"], task, obs["instruction"], bgs=_bgs(task), cfg=cfg, extra_frames=extra)
    dt = time.time() - t0
    rows, n_gt, n_pred = score_record(rec, d0.load_gt(ep, frame), task, {"episode": ep.name, "frame": frame, "scan": scan})
    return ep.name, rec, rows, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", default="W0000")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.root) / a.task
    eps = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit() and (not a.ids or p.name in a.ids))
    allrows = []
    with ProcessPoolExecutor(a.workers) as ex, open(out / "objects.jsonl", "w") as f:
        for name, rec, rows, dt in ex.map(run, [(str(e), a.task, a.frame, a.scan) for e in eps]):
            (out / f"rec_{name}.json").write_text(json.dumps(rec, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
            for r in rows:
                f.write(json.dumps(r) + "\n")
            allrows += rows
            print(f"{name} {dt:.0f}s | " + " ".join(
                f"{r['label']}:{'-' if not r.get('found') else '%.0f/%+.0f' % (r['xy_err_mm'], r['z_err_mm'])}" for r in rows), flush=True)
    by: dict = {}
    for r in allrows:
        by.setdefault(r["category"], []).append(r)
    for c, rs in sorted(by.items()):
        f_ = [r for r in rs if r.get("found")]
        xy = sorted(r["xy_err_mm"] for r in f_)
        z = sorted(r["z_err_mm"] for r in f_)
        print(f"{c}: n {len(rs)} found {len(f_)} xy med {np.median(xy) if xy else float('nan'):.1f} p90 "
              f"{np.percentile(xy, 90) if xy else float('nan'):.1f} | within 15 mm {sum(x <= 15 for x in xy)} | "
              f"z med {np.median(z) if z else float('nan'):+.1f}")


if __name__ == "__main__":
    main()
