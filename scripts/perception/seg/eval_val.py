"""Twin-trained segmenter on its validation draws (seg_render dirs never trained on: the --val-ids layouts), per task,
camera and arm pose (home / hover / look / both): recall@0.5 and asset-class accuracy against Isaac's instance masks.
DEV ONLY. Answers "how well does it read X in a wrist at hover" (make_kong's glyph tiles), which the D0 frames lack.

    python scripts/perception/seg/eval_val.py --render ~/dojo-data/segtrain/render --ids 100049 --tasks make_kong \\
        --weights W --classes classes.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main(argv=None):
    from PIL import Image

    from dojo.perception import seg_twin as st

    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--ids", default="100049")
    ap.add_argument("--tasks", default="make_kong")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-px", type=int, default=30)
    a = ap.parse_args(argv)
    classes = json.load(open(os.path.expanduser(a.classes)))["names"]
    M = st._SegModel(os.path.expanduser(a.weights), classes, conf=a.conf)
    ids = {s.strip() for s in a.ids.split(",")}
    agg = defaultdict(lambda: [0, 0, 0])  # n, recalled, class right
    for task in a.tasks.split(","):
        for d in sorted(Path(a.render).expanduser().glob(f"{task}/*")):
            if not (d / "done.json").exists() or d.name.split("_")[0] not in ids:
                continue
            for fj in sorted(d.glob("frame_*.json")):
                rec = json.loads(fj.read_text())
                k = rec["frame"]
                kind = (rec.get("move") or {}).get("kind") or "home"
                objs = rec["gt"]["objects"]
                for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                    p = d / f"inst_{k}_{cam}.png"
                    if not p.exists():
                        continue
                    codes = np.asarray(Image.open(p)).astype(np.int64)
                    img = np.asarray(Image.open(d / f"frame_{k}_{cam}.png").convert("RGB"))
                    dets = [st.Det(m.astype(np.uint8), key, sc, tuple(b)) for m, key, sc, b in M.detect(img, a.conf)]
                    gt = {i: m for i, m in st.instances(codes, a.min_px).items() if i < len(objs) and not objs[i].get("clutter")}
                    match = st.match_greedy(gt, dets)
                    view = "head" if cam == "cam_head" else "wrist"
                    for i in gt:
                        key = st.asset_key(objs[i])
                        hit = i in match and match[i][1] >= 0.5
                        right = hit and dets[match[i][0]].key == key
                        for grp in ((task, view, kind), (task, view, "all")):
                            agg[grp][0] += 1
                            agg[grp][1] += hit
                            agg[grp][2] += right
    print("| task | view | pose | n | recall@0.5 | asset class right |")
    print("|---|---|---|---|---|---|")
    for (task, view, kind), (n, r, c) in sorted(agg.items()):
        print(f"| {task} | {view} | {kind} | {n} | {r / n:.2f} | {c / n:.2f} |")


if __name__ == "__main__":
    main()
