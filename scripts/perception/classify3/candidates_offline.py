"""Dev only: ``dojo.perception.evidence.candidate_regions`` on recorded camera rows' frame-0 records, scored against
the rows' layouts (GT read after, for scoring only).

    python scripts/perception/classify3/candidates_offline.py --rows ROWS --layouts LAYOUT_ROOT --catalog CATALOG [--out o.json]

Per candidate: **hit** = within 50 mm of a GT instance no frame-0 belief found (an unseen object the candidate would
recover after a confirming look), **duplicate** = within 50 mm of a found instance, **phantom** = no GT instance within
50 mm. Per row: missed instances and how many a candidate covers. Our slices only.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from dojo.perception.evidence import candidate_regions  # noqa: E402
from scripts.perception.classify3.stage_audit import SLICES, match  # noqa: E402

NEAR = 0.05


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--layouts", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    def half_extent(key):
        p = Path(a.catalog) / key / "info.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return (np.asarray(d["bbox_max"][:2], float) - np.asarray(d["bbox_min"][:2], float)) / 2

    rows, tot = [], {}
    for d in sorted(glob.glob(os.path.join(a.rows, "c4b_*_1000??"))):
        ep = json.loads(open(glob.glob(os.path.join(d, "*_episodes.jsonl"))[0]).readline())
        if int(ep["layout_id"]) not in SLICES:
            raise SystemExit(f"{d}: outside our slices")
        rec = json.load(gzip.open(glob.glob(os.path.join(d, "*_report.json.gz"))[0]))["records"]["initial"]
        cands = candidate_regions(rec, half_extent)
        lay = json.load(open(os.path.join(a.layouts, ep["task"], os.path.basename(ep["layout"]))))
        gt = {o["label"]: np.asarray(o["default_pos"][:2], float) for lst in lay["Rigid"].values() for o in lst}
        bel = {f"b{i}": r["default_pos"] for i, (_, r) in enumerate((c, r) for c, lst in (rec.get("Rigid") or {}).items()
                                                                       for r in lst)}
        found = set(match({g: list(p) for g, p in gt.items()}, bel))
        missed = [g for g in gt if g not in found]
        out = []
        covered = set()
        for c in cands:
            xy = np.asarray(c["xy"], float)
            dist = {g: float(np.linalg.norm(xy - p)) for g, p in gt.items()}
            g = min(dist, key=dist.get) if dist else None
            kind = "phantom" if g is None or dist[g] > NEAR else ("hit" if g in missed else "duplicate")
            if kind == "hit":
                covered.add(g)
            out.append({**c, "kind": kind, "gt": g, "dist_mm": round(1000 * dist[g]) if g else None})
        task = ep["task"]
        t = tot.setdefault(task, {"rows": 0, "gt": 0, "missed": 0, "covered": 0, "hit": 0, "duplicate": 0, "phantom": 0,
                                  "phantom_rows": 0, "merged": 0})
        t["rows"] += 1
        t["gt"] += len(gt)
        t["missed"] += len(missed)
        t["covered"] += len(covered)
        for k in ("hit", "duplicate", "phantom"):
            t[k] += sum(1 for c in out if c["kind"] == k)
        t["phantom_rows"] += any(c["kind"] == "phantom" for c in out)
        t["merged"] += sum(1 for c in out if c["merged"])
        rows.append({"row": os.path.basename(d), "n_gt": len(gt), "missed": missed, "covered": sorted(covered),
                     "candidates": out})
        print(f"{os.path.basename(d)}: gt {len(gt)} missed {len(missed)} covered {len(covered)} candidates "
              + ", ".join(f"{c['kind']}({c['area_px']}{'M' if c['merged'] else ''})" for c in out))
    print(json.dumps(tot, indent=1))
    if a.out:
        json.dump({"summary": tot, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
