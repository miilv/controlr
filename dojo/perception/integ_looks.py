"""Offline analysis of integration's saved look frames (paired runs on our dev layouts).

For every ``*_look_frames`` dir (frame0_* / look_* PNGs + state JSONs): frame-0 estimate, conservative fuse
with the look frame, and (optionally) the full re-estimate, scored against D0's GT for the same layout
(simlab: the settled records replay bit-exact). The look variant ("plan", "d0.18/e70", ...) comes from the
episode rows / report. DEV-ONLY scoring.

    python -m dojo.perception.integ_looks --runs ~/dojo-data/integ/runs/B_stack_bundle_base3_* --out <json>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
from pathlib import Path

import cv2
import numpy as np


def _obs(d: Path, prefix: str) -> dict:
    imgs = {c: cv2.cvtColor(cv2.imread(str(d / f"{prefix}_{c}.png")), cv2.COLOR_BGR2RGB)
            for c in ("cam_head", "cam_left_wrist", "cam_right_wrist") if (d / f"{prefix}_{c}.png").exists()}
    st = json.loads((d / f"{prefix}_state.json").read_text())
    return {"images": imgs, "state": {k: np.asarray(v, dtype=float) for k, v in st.items() if isinstance(v, list)}}


def _look_meta(run: Path, task: str, lid: str) -> list:
    for rp in run.glob(f"{task}_*_{lid}_report.json.gz"):
        try:
            return json.load(gzip.open(rp)).get("looks", [])
        except Exception:
            return []
    return []


def analyse(run_dirs, d0_root: Path, full: bool = False, color_calib: str | None = None) -> list[dict]:
    import zlib

    from . import d0
    from .eval_g1 import fold_backgrounds, score_record
    from .inverse_layout import EstimatorConfig, estimate
    from .look import fuse

    cal = json.loads(Path(color_calib).read_text()) if color_calib and Path(color_calib).exists() else None
    rows = []
    for run in map(Path, run_dirs):
        for d in sorted(run.glob("*_look_frames")):
            m = re.match(r"(.+)_perception_.*_(\d+)_look_frames", d.name)
            if not m or not (d / "look_state.json").exists():
                continue
            task, lid = m.group(1), m.group(2)
            ep = d0_root / task / lid
            if not (ep / "frame_0.json").exists():
                continue
            kw = {"color_M": cal[f"fold{zlib.crc32(lid.encode()) % 2}"]} if cal else {}
            cfg = EstimatorConfig(**kw)
            bgs = fold_backgrounds(ep)
            o0, oL = _obs(d, "frame0"), _obs(d, "look")
            rec0 = estimate(o0["images"], o0["state"], task, bgs=bgs, cfg=cfg)
            recC = fuse([o0, oL], task, bgs=bgs, cfg=cfg, prior_record=rec0)
            recs = {"frame0": rec0, "conservative": recC}
            if full:
                recs["full"] = fuse([o0, oL], task, bgs=bgs, cfg=cfg, mode="full")
            gt = d0.load_gt(ep, 0)
            looks = _look_meta(run, task, lid)
            variant = ",".join(f"{lk.get('arm')}:{lk.get('pose')}" for lk in looks) or "?"
            for name, rec in recs.items():
                rs, _, _ = score_record(rec, gt, task, {"episode": lid, "run": run.name, "mode": name, "variant": variant})
                rows += rs
            print(run.name, task, lid, variant, flush=True)
    return rows


def main(argv=None):
    from collections import defaultdict

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--d0", default=str(Path("~/dojo-data/d0").expanduser()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--color-calib", default=None)
    a = ap.parse_args(argv)
    runs = [p for r in a.runs for p in glob.glob(r) if Path(p).is_dir()]
    rows = analyse(runs, Path(a.d0), a.full, a.color_calib)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows))
    by = defaultdict(list)
    for r in rows:
        by[(r["task"], r["mode"])].append(r)
    for (t, mname), rs in sorted(by.items()):
        f = [r for r in rs if r.get("found")]
        xy = np.array([r["xy_err_mm"] for r in f]) if f else np.array([np.nan])
        print(f"{t} {mname}: n {len(rs)} found {len(f)} xy med {np.nanmedian(xy):.1f} p90 {np.nanpercentile(xy, 90):.1f} "
              f"within 15 mm {np.mean([r.get('found') and r['xy_err_mm'] <= 15 for r in rs]):.2f}")
    # per object: conservative vs frame 0, split by the look variant
    key = lambda r: (r["run"], r["episode"], r["label"])  # noqa: E731
    f0 = {key(r): r for r in rows if r["mode"] == "frame0"}
    cons = {key(r): r for r in rows if r["mode"] == "conservative"}
    agg = defaultdict(lambda: [0, 0, 0])
    for k, a0 in f0.items():
        c = cons.get(k)
        if not c or not a0.get("found") or not c.get("found"):
            continue
        d = c["xy_err_mm"] - a0["xy_err_mm"]
        v = "plan" if "plan" in c["variant"] else ("fallback" if c["variant"] != "?" else "?")
        agg[v][0] += 1
        agg[v][1] += d > 5
        agg[v][2] += d < -5
    for v, (n, worse, better) in agg.items():
        print(f"conservative vs frame 0, look variant {v}: objects {n}, worse > 5 mm {worse}, better > 5 mm {better}")


if __name__ == "__main__":
    main()
