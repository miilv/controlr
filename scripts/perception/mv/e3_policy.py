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


_CACHE: dict = {}


def gated(e: dict, gate_d: float, flag_mm: float) -> bool:
    """The run's gate, re-applied: the height check flags it, or B1's own pose's distance exceeds gate_d."""
    dh, d0 = e.get("dh_mm"), e["cands"][0]["d"]
    return (dh is not None and abs(dh) > flag_mm) or (d0 is not None and d0 > gate_d)


def apply(rows, b1dump: Path, margin: float, sil_max: float, srcs: set, w_sil: float = 0.0, gate_d: float = 0.55,
          flag_mm: float = 15.0, protect_mm: float = 0.0) -> dict:
    """``protect_mm`` > 0: a candidate within that xy distance of a B1 object the gate did NOT doubt (its own pose
    looks and stands right) is not taken: a confident object keeps its blob."""
    from dojo.perception import d0
    from dojo.perception.mv3d_eval import SET_ALIAS, _b1_record, _score, root_dir

    tot = Counter()
    for r in rows:
        if "error" in r:
            continue
        key = (r["set"], r["task"], r["episode"])
        if key not in _CACHE:  # the B1 record, the GT and B1's own scores: once per scene for every policy
            meta = json.loads((b1dump / {v: k for k, v in SET_ALIAS.items()}.get(r["set"], r["set"]) / r["task"] / r["episode"] / "meta.json").read_text())
            gt_ = d0.load_gt(root_dir(r["set"]) / r["task"] / r["episode"], 0)  # DEV-ONLY
            m_ = {"set": r["set"], "task": r["task"], "episode": r["episode"], "frame": 0}
            _CACHE[key] = (meta, gt_, m_, _score(_b1_record(meta), gt_, r["task"], m_))
        meta, gt, m, (c0, p0) = _CACHE[key]
        rec0 = _b1_record(meta)
        rec1 = copy.deepcopy(rec0)
        taken = []
        switched_here = 0
        doubted_xy = [e["cands"][0]["xy"] for e in r["log"] if e["cands"][0]["d"] is not None and gated(e, gate_d, flag_mm)]
        confident = [p["pos"][:2] for p in meta["preds"]
                     if not any(np.hypot(p["pos"][0] - q[0], p["pos"][1] - q[1]) < 0.02 for q in doubted_xy)]
        for e in r["log"]:
            cs = e["cands"]
            if cs[0]["d"] is None or not gated(e, gate_d, flag_mm):
                continue
            d0_ = cs[0]["d"]
            best, bs = None, None
            for c in cs[1:]:
                if c["src"] not in srcs or c["d"] is None or (c["sil"] is not None and c["sil"] > sil_max):
                    continue
                if protect_mm > 0 and any(np.hypot(c["xy"][0] - q[0], c["xy"][1] - q[1]) * 1000 < protect_mm for q in confident):
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
            switched_here += 1
        c1, p1 = _score(rec1, gt, r["task"], m) if switched_here else (c0, p0)
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
    ap.add_argument("--heldout", action="store_true", help="only ids 100000-09 / 100020-24 (stream S's held-out split)")
    ap.add_argument("--gate-d", type=float, default=0.55)
    ap.add_argument("--flag-mm", type=float, default=15.0)
    ap.add_argument("--srcs", nargs="+", default=["swap,raised,fg", "det", "det,swap,raised,fg"])
    a = ap.parse_args(argv)
    rows = [json.loads(line) for line in open(a.frames)]
    if a.heldout:
        rows = [r for r in rows if "error" not in r and (100000 <= int(r["episode"]) <= 100009 or 100020 <= int(r["episode"]) <= 100024)]
    b1 = Path(a.b1dump).expanduser()
    for srcs in [set(x.split(",")) for x in a.srcs]:
        for margin in (0.05, 0.1, 0.15):
            for sil_max in (0.6, 0.8):
                for prot in (0.0, 25.0):
                    print(json.dumps(apply(rows, b1, margin, sil_max, srcs, gate_d=a.gate_d, flag_mm=a.flag_mm,
                                           protect_mm=prot) | {"protect_mm": prot}), flush=True)


if __name__ == "__main__":
    main()
