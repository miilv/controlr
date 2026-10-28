"""DEV-ONLY: re-select E3 fusion candidates offline (e3_fusion_eval frames.jsonl) under other policies, eval_g1 each.

    python scripts/perception/mv/e3_policy.py FRAMES.jsonl --b1dump DIR
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np


def apply(rows, b1dump: Path, margin: float, sil_max: float, srcs: set, w_sil: float = 0.0) -> dict:
    from dojo.perception import d0
    from dojo.perception.mv3d_eval import SET_ALIAS, _b1_record, _score, root_dir

    tot = Counter()
    for r in rows:
        if "error" in r:
            continue
        meta = json.loads((b1dump / {v: k for k, v in SET_ALIAS.items()}.get(r["set"], r["set"]) / r["task"] / r["episode"] / "meta.json").read_text())
        rec0 = _b1_record(meta)
        rec1 = copy.deepcopy(rec0)
        taken = []
        for e in r["log"]:
            cs = e["cands"]
            if cs[0]["d"] is None:
                continue
            d0_ = cs[0]["d"]
            best, bs = None, None
            for c in cs[1:]:
                if c["src"] not in srcs or c["d"] is None or (c["sil"] is not None and c["sil"] > sil_max):
                    continue
                sc = c["d"] + w_sil * (c["sil"] or 0.0)
                if bs is None or sc < bs:
                    best, bs = c, sc
            if best is None or best["d"] >= d0_ - margin:
                continue
            if any(np.hypot(best["xy"][0] - t[0], best["xy"][1] - t[1]) < 0.015 for t in taken):
                continue
            ent = rec1[e["typ"]][e["cat"]][e["j"]]
            ent["default_pos"], ent["default_ori"] = best["pos"], best["ori"]
            taken.append(best["xy"])
            tot["switched"] += 1
        gt = d0.load_gt(root_dir(r["set"]) / r["task"] / r["episode"], 0)  # DEV-ONLY
        m = {"set": r["set"], "task": r["task"], "episode": r["episode"], "frame": 0}
        c0, p0 = _score(rec0, gt, r["task"], m)
        c1, p1 = _score(rec1, gt, r["task"], m)
        tot["phantoms_before"] += p0
        tot["phantoms_after"] += p1
        for k in c0:
            tot[("b", c0[k])] += 1
            tot[("a", c1.get(k))] += 1
            if c0[k] != c1.get(k):
                tot[f"{c0[k]}->{c1.get(k)}"] += 1
    within = lambda s: tot[(s, "ok")] + tot[(s, "identity")]  # noqa: E731
    return {"margin": margin, "sil_max": sil_max, "srcs": sorted(srcs), "w_sil": w_sil, "switched": tot["switched"],
            "within": f"{within('b')} -> {within('a')}", "gross": f"{tot[('b', 'gross')]} -> {tot[('a', 'gross')]}",
            "missed": f"{tot[('b', 'missed')]} -> {tot[('a', 'missed')]}",
            "phantoms": f"{tot['phantoms_before']} -> {tot['phantoms_after']}",
            "changes": {k: v for k, v in tot.items() if isinstance(k, str) and "->" in k}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--b1dump", required=True)
    a = ap.parse_args(argv)
    rows = [json.loads(line) for line in open(a.frames)]
    b1 = Path(a.b1dump).expanduser()
    for srcs in ({"swap", "raised", "fg"}, {"raised", "fg"}, {"swap", "raised", "fg", "b1_yaw"}, {"det"},
                 {"det", "swap", "raised", "fg"}):
        for margin in (0.02, 0.05, 0.1):
            for sil_max in (0.3, 0.6):
                print(json.dumps(apply(rows, b1, margin, sil_max, srcs)), flush=True)


if __name__ == "__main__":
    main()
