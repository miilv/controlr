"""DEV (stream C): B1's frame-0 read of deposit_coin's coin and stand on a recorded run's saved frame 0, per estimator
config, against the layout GT (scorer side).

    DOJO_BG_FOLDS=... DOJO_COLOR_CALIB=... python scripts/perception/coin_frame0_eval.py RUN_DIR [RUN_DIR ...] \
        --layouts LAYOUT_DIR [--cfg '{"thin_slot_keep": true}'] [--out rows.jsonl]

RUN_DIR holds ``*_look_frames/frame0_cam_{head,left_wrist,right_wrist}.png`` + ``frame0_state.json`` and
``*_episodes.jsonl`` (its layout id).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

TASK = "deposit_coin"
INSTRUCTION = "Pick up the coin from the holder and insert it precisely into the coin bank."


def _yaw(q):
    from dojo.perception.cameras import quat_wxyz_to_R
    R = quat_wxyz_to_R(q)
    return math.atan2(R[1, 0], R[0, 0])


def main():
    from dojo.perception.inverse_layout import EstimatorConfig, estimate, task_map
    from dojo.perception.segment import load_backgrounds, with_scene

    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--layouts", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    d = Path(os.environ["DOJO_BG_FOLDS"]) / f"fold{a.fold}"
    bgs = load_backgrounds(d)
    m = d / "cam_head_clean_unseen.png"
    if m.exists():
        bgs["cam_head_clean_unseen"] = (cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    keys = sorted({k for dd in task_map()[TASK]["labels"].values() if dd.get("type") in ("Dynamic", "Geometry")
                   for k in dd.get("candidates") or []})
    bgs = with_scene(bgs, d, keys)
    cal = json.loads(Path(os.environ["DOJO_COLOR_CALIB"]).read_text())
    cfg = EstimatorConfig(use_wrists=True, color_M=cal[f"fold{a.fold}"], **json.loads(a.cfg))
    fo = open(a.out, "a") if a.out else None
    for run in a.runs:
        lf = glob.glob(os.path.join(run, "*_look_frames"))[0]
        ep = [json.loads(x) for x in open(glob.glob(os.path.join(run, "*_episodes.jsonl"))[0])][-1]
        lid = int(ep["layout_id"])
        imgs = {c: cv2.cvtColor(cv2.imread(os.path.join(lf, f"frame0_{c}.png")), cv2.COLOR_BGR2RGB)
                for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
        state = {k: np.asarray(v, float) for k, v in json.load(open(os.path.join(lf, "frame0_state.json"))).items()}
        rec = estimate(imgs, state, TASK, INSTRUCTION, bgs=bgs, cfg=cfg)
        lay = json.load(open(os.path.join(a.layouts, f"{lid}.settled.json")))
        g = {"coin0": lay["Rigid"]["coin"][0], "vertical_coin_stand": lay["Geometry"]["vertical_coin_stand"][0]}
        row = {"run": os.path.basename(run), "layout_id": lid, "cfg": a.cfg}
        for lab, typ, cat in (("coin0", "Rigid", "coin"), ("vertical_coin_stand", "Geometry", "vertical_coin_stand")):
            r = next((x for x in (rec.get(typ) or {}).get(cat, []) if x.get("label") == lab), None)
            if r is None:
                nf = next((x for x in rec.get("_not_found", []) if x.get("label") == lab), {})
                row[lab] = {"found": False, "why": nf.get("reason")}
                continue
            e = 1000 * math.hypot(r["default_pos"][0] - g[lab]["default_pos"][0], r["default_pos"][1] - g[lab]["default_pos"][1])
            dy = math.degrees((_yaw(r["default_ori"]) - _yaw(g[lab]["default_ori"]) + math.pi / 2) % math.pi - math.pi / 2)
            p = r.get("_perception") or {}
            row[lab] = {"found": True, "xy_mm": round(e, 1), "yaw_deg_mod180": round(dy, 1),
                        "conf": p.get("confidence"), "unverified": bool(p.get("unverified"))}
        print(json.dumps(row), flush=True)
        if fo:
            fo.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
