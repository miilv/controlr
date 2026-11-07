"""CLASSIFY3 (dev only): the perception service's frame-0 estimate re-run offline on recorded camera rows, scored per
GT instance against the row's own layout (GT is read only after the estimate, for scoring).

    DOJO_BG_FOLDS=... DOJO_CALIB_SET=frozen_d0_100040_49 DOJO_CATALOG=... DOJO_SEG_ADDR=... [DOJO_SAM_ADDR=...] \\
    [DOJO_PERCEPTION_CFG='{"sam_task_routes": {...}}'] \\
        python scripts/perception/classify3/frame0_e2e.py --rows ROWS --layouts LAYOUT_ROOT --arm NAME --out OUT.jsonl

Per row (``c4b_<arm>_<id>/``): frame 0 as the episode saved it (``*_look_frames/frame0_cam_*.png`` +
``frame0_state.json``), the instruction and the vocabulary the episode used (its report), then
``perception_service._Perception.estimate`` (the service's own call: the frozen calibration set, its colour map,
``DOJO_PERCEPTION_CFG`` overrides). Scores: Rigid beliefs, GT instances found (<= 50 mm xy, one-to-one), category
right, exact asset right, phantoms; the series count trace; and the same scores of the recorded run's own frame-0
record (``records.initial``) for a reproduction check. Our slices only (100000-09, 100020-29).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.perception.classify3.stage_audit import SLICES, _cat, _idx, match  # noqa: E402

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def rigid_rows(rec: dict) -> dict:
    """{name: (key, pos)} of a perception record's Rigid objects (names made unique)."""
    out = {}
    for cat, lst in (rec.get("Rigid") or {}).items():
        for i, r in enumerate(lst):
            out[f"{r.get('label', cat)}#{cat}{i}"] = (f"Rigid/{cat}/{int(r['category_idx']):05d}", r["default_pos"])
    return out


def score(rec: dict, gt: dict) -> dict:
    bel = rigid_rows(rec)
    m = match({g: o["default_pos"] for g, o in gt.items()}, {b: v[1] for b, v in bel.items()})
    cat_ok = [g for g, b in m.items() if _cat(bel[b][0]) == gt[g]["category"]]
    exact = [g for g in cat_ok if _idx(bel[m[g]][0]) == gt[g]["category_idx"]]
    by_cat: dict = {}
    for g, o in gt.items():
        c = by_cat.setdefault(o["category"], {"n": 0, "found": 0, "cat_ok": 0})
        c["n"] += 1
        c["found"] += g in m
        c["cat_ok"] += g in cat_ok
    return {"n_belief": len(bel), "found": len(m), "cat_ok": len(cat_ok), "exact_ok": len(exact),
            "phantoms": sorted(bel[b][0] for b in bel if b not in m.values()), "by_category": by_cat,
            "missed": sorted(f"{gt[g]['category']}/{gt[g]['category_idx']}" for g in gt if g not in m),
            "wrong_cat": sorted(f"{gt[g]['category']}->{_cat(bel[m[g]][0])}" for g in m if g not in cat_ok)}


def load_frame(row: Path) -> dict:
    from PIL import Image
    lf = next(row.glob("*_look_frames"))
    imgs = {c: np.asarray(Image.open(lf / f"frame0_{c}.png").convert("RGB")) for c in CAMS if (lf / f"frame0_{c}.png").exists()}
    st = json.loads((lf / "frame0_state.json").read_text())
    return {"images": imgs, "state": {k: np.asarray(v, float) for k, v in st.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--layouts", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated row-name substrings")
    a = ap.parse_args(argv)
    from dojo.runtime.perception_service import _Perception, encode_frame
    P = _Perception()
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(ln)["row"] for ln in open(a.out) if ln.strip()}
    rows = sorted(Path(a.rows).glob("c4b_*_1000??"))
    if a.only:
        rows = [r for r in rows if any(s in r.name for s in a.only.split(","))]
    for row in rows:
        if row.name in done:
            continue
        ep = json.loads(open(next(row.glob("*_episodes.jsonl"))).readline())
        lid = int(ep["layout_id"])
        if lid not in SLICES:
            raise SystemExit(f"{row}: layout {lid} outside our slices")
        rep = json.load(gzip.open(next(row.glob("*_report.json.gz"))))
        vocab = rep.get("vocabulary") or ep["task"]
        instr = ""
        for e in rep.get("events") or []:
            if e.get("kind") == "ported" and e.get("details"):
                instr = json.loads(e["details"]).get("instruction") or ""
        if not instr:
            m = re.search(r'"instruction": "([^"]+)"', json.dumps(rep))
            instr = m.group(1) if m else ""
        fr = load_frame(row)
        fr["instruction"] = instr
        P.set_scene(vocab)
        t0 = time.time()
        rec = P.estimate([encode_frame(fr)], vocab, 0)
        dt = time.time() - t0
        lay = json.load(open(os.path.join(a.layouts, ep["task"], os.path.basename(ep["layout"]))))   # DEV: after the estimate
        gt = {o["label"]: o for lst in lay["Rigid"].values() for o in lst}
        per = rec.get("_perception") or {}
        out = {"row": row.name, "arm": a.arm, "task": ep["task"], "layout_id": lid, "n_gt": len(gt), "estimate_s": round(dt, 1),
               "new": score(rec, gt), "recorded_initial": score((rep.get("records") or {}).get("initial") or {}, gt),
               "series": per.get("series"), "sam": per.get("sam"), "seg": (per.get("seg_propose") or per.get("seg")),
               "cfg_env": os.environ.get("DOJO_PERCEPTION_CFG", "")}
        with open(a.out, "a") as f:
            f.write(json.dumps(out, default=str) + "\n")
        n = out["new"]
        print(f"[frame0_e2e] {a.arm} {row.name}: gt {len(gt)} belief {n['n_belief']} found {n['found']} cat {n['cat_ok']} "
              f"exact {n['exact_ok']} phantoms {len(n['phantoms'])} ({dt:.0f} s; recorded belief "
              f"{out['recorded_initial']['n_belief']} cat {out['recorded_initial']['cat_ok']})", flush=True)


if __name__ == "__main__":
    main()
