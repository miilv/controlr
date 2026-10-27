"""DEV: B1 with the height check on deposit_coin's saved proxy frames (frame 0 and the scan read, as the runtime's
perception service computes them), stand / coin errors vs our settled layouts with ``stereo_reject`` off vs
"slot_bases" (applied to the same annotated record, so the pair differs only by the consumer).

    DOJO_GL=0 python scripts/perception/stereo_coin_e2e.py --out OUT [--ids ...]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stereo_coin import LAYOUTS, RUNS, SRC, load_set, objects_of  # noqa: E402


def bgs_for(fold: int, scan: bool) -> dict:
    from dojo.perception.segment import load_backgrounds

    d = Path(os.path.expanduser("~/dojo-perception-data/bg_folds_scan" if scan else "~/dojo-perception-data/bg_folds")) / f"fold{fold}"
    out = load_backgrounds(d)
    m = d / "cam_head_clean_unseen.png"
    if m.exists():
        out["cam_head_clean_unseen"] = (cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    return out


def errs(rec: dict, gts: list[dict]) -> dict:
    out = {}
    pred = {o["cat"]: o for o in objects_of(rec)}
    nf = {x["label"] for x in rec.get("_not_found", [])}
    for g in gts:
        if g["cat"] not in ("coin", "vertical_coin_stand"):
            continue
        p = pred.get(g["cat"])
        out[g["cat"]] = None if p is None else round(float(np.linalg.norm(np.subtract(p["pos"], g["pos"]))) * 1000, 1)
        if p is None:
            out[g["cat"] + "_nf"] = g["label"] in nf or g["cat"] in nf
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", nargs="*", type=int, default=list(range(100000, 100010)))
    a = ap.parse_args()
    from dojo.perception.inverse_layout import EstimatorConfig, estimate, reject_slot_bases
    from dojo.perception.scan import estimate_with_scan

    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cal = json.loads(Path(os.path.expanduser("~/dojo-perception-data/catalog/color_calib.json")).read_text())
    for lid in a.ids:
        d = RUNS / SRC[lid] / f"deposit_coin_perception_verify-bundle_{lid}_look_frames"
        i0, s0 = load_set(d, "frame0")
        iS, sS = load_set(d, "scan")
        fold = zlib.crc32(str(lid).encode()) % 2
        cfg = EstimatorConfig(use_wrists=True, color_M=cal[f"fold{fold}"], stereo_check=True, stereo_reject="off")
        t0 = time.time()
        rec0 = estimate(i0, s0, "deposit_coin", "", bgs=bgs_for(fold, False), cfg=cfg)
        t1 = time.time()
        recS = estimate_with_scan({"images": iS, "state": sS}, {"images": i0, "state": s0}, "deposit_coin", "",
                                  bgs=bgs_for(fold, True), cfg=cfg, rec0=copy.deepcopy(rec0))
        t2 = time.time()
        gts = objects_of(json.loads((LAYOUTS / f"{lid}.settled.json").read_text()))  # DEV-ONLY
        row = {"id": lid, "b1_s": [round(t1 - t0, 1), round(t2 - t1, 1)]}
        for name, rec in (("frame0", rec0), ("scan", recS)):
            on = copy.deepcopy(rec)
            dropped = reject_slot_bases(on)
            st = {o["cat"]: (o.get("_perception") or {}).get("stereo") for o in [{"cat": c, **r} for t in ("Geometry", "Rigid")
                                                                                 for c, l in (rec.get(t) or {}).items() for r in l]}
            row[name] = {"off": errs(rec, gts), "on": errs(on, gts), "dropped": dropped,
                         "stand_check": st.get("vertical_coin_stand"), "check": (rec.get("_perception") or {}).get("stereo_check")}
            (out / f"rec_{lid}_{name}.json").write_text(json.dumps({k: v for k, v in rec.items() if k != "_perception"} | {"_perception": {k: v for k, v in rec.get("_perception", {}).items() if k.startswith("stereo")}}))
        print(json.dumps(row), flush=True)
        with open(out / "rows.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
