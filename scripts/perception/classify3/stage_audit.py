"""CLASSIFY3 stage audit of recorded classify camera rows (dev only; GT is used for scoring, never by a policy).

    python scripts/perception/classify3/stage_audit.py --rows ROWS_DIR --layouts LAYOUT_ROOT --replay REPLAY_DIR \
        [--out audit.json]

ROWS_DIR holds copied camera row dirs (``c4b_<arm>_<id>/``: ``*_episodes.jsonl``, ``*_report.json.gz``). LAYOUT_ROOT
holds the rows' layout files under ``classify_objects/`` and ``classify_objects_by_language/`` (the row's own layout
file name). REPLAY_DIR holds ``scripts/perception/m95_replay.py --full`` outputs named after the row dirs (physics-only
replays of the recorded actions: GT poses every few steps and at every gripper change).

Per row and per GT instance it splits the loss into stages:

* **frame 0** (the belief the skill planned on, the first ``dev_belief_probe`` snapshot): instance found (a belief
  within ``MATCH_MM`` in xy, one-to-one, nearest first), its category right, its exact asset right; phantoms (beliefs
  matching no GT instance);
* **target**: the basket the instance belongs in. Language rows: the instruction's map (left / middle / right).
  Free rows: the category -> basket map the skill chose for the category the instance was perceived as (its own
  category when that is right; a missed / miscategorised instance has none of its own);
* **outcome** (replay): lifted at all, final basket (or table), ``in_target``; a believed-placed instance (the skill's
  ``truth_in`` = its belief says inside) that physics has outside is a **false placement belief**.

Layout ids are restricted to our slices (100000-09, 100020-29).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
from collections import Counter

import numpy as np

MATCH_MM = 50.0
TABLE_Z = 0.765
BASKET_HALF = (0.126, 0.084)    # catalogue Geometry/basket/0000[234] bbox half extents (x, y)
BASKET_TOP = 0.842
POS = {"left": 0, "middle": 1, "center": 1, "centre": 1, "right": 2}
SLICES = set(range(100000, 100010)) | set(range(100020, 100030))


def _cat(key: str | None) -> str | None:
    p = (key or "").split("/")
    return p[1] if len(p) >= 3 else None


def _idx(key: str | None) -> int | None:
    p = (key or "").split("/")
    return int(p[2]) if len(p) >= 3 and p[2].isdigit() else None


def match(gt: dict, bel: dict, cap_mm: float = MATCH_MM) -> dict:
    """{gt label: belief label}: one-to-one by xy distance, nearest pairs first, within ``cap_mm``."""
    pairs = []
    for g, gp in gt.items():
        for b, bp in bel.items():
            d = 1000.0 * float(np.hypot(gp[0] - bp[0], gp[1] - bp[1]))
            if d <= cap_mm:
                pairs.append((d, g, b))
    out, used = {}, set()
    for d, g, b in sorted(pairs):
        if g in out or b in used:
            continue
        out[g] = b
        used.add(b)
    return out


def basket_of(pos, baskets: list) -> int | None:
    """Index (0 = leftmost) of the basket whose footprint holds ``pos`` below its rim, else None (table / elsewhere)."""
    x, y, z = (float(v) for v in pos[:3])
    for i, (bx, by) in enumerate(baskets):
        if abs(x - bx) <= BASKET_HALF[0] and abs(y - by) <= BASKET_HALF[1] and TABLE_Z - 0.01 <= z <= BASKET_TOP + 0.01:
            return i
    return None


def parse_map(instr: str) -> dict:
    """{category: basket index} from 'Put X objects into the left basket, ...' (empty for a free sort)."""
    out = {}
    for phrase, pos in re.findall(r"([a-z_]+)\s+objects?\s+into\s+the\s+(left|middle|center|centre|right)\s+basket",
                                  (instr or "").lower()):
        out[phrase] = POS[pos]
    return out


def audit_row(row_dir: str, layout_root: str, replay_dir: str | None) -> dict:
    ep = json.loads(open(glob.glob(os.path.join(row_dir, "*_episodes.jsonl"))[0]).readline())
    lid = int(ep["layout_id"])
    if lid not in SLICES:
        raise SystemExit(f"{row_dir}: layout {lid} is outside our slices")
    rep = json.load(gzip.open(glob.glob(os.path.join(row_dir, "*_report.json.gz"))[0]))
    lay = json.load(open(os.path.join(layout_root, ep["task"], os.path.basename(ep["layout"]))))
    gt0 = {o["label"]: o for lst in lay["Rigid"].values() for o in lst}
    baskets = sorted((tuple(o["default_pos"][:2]) for o in lay["Geometry"]["basket"]), key=lambda p: p[0])
    snaps = rep["dev_belief_probe"]["snaps"]
    s0 = snaps[0]
    bel0 = {k: v for k, v in s0["objects"].items() if v["key"].startswith("Rigid/")}
    m0 = match({g: o["default_pos"] for g, o in gt0.items()}, {b: v["pos"] for b, v in bel0.items()})
    ported = next((e for e in rep["events"] if e.get("kind") == "ported"), None)
    det = json.loads(ported["details"]) if ported and isinstance(ported.get("details"), str) else (ported or {}).get("details") or {}
    log = det.get("log") or []
    instr = det.get("instruction") or ""
    fixed = parse_map(instr)
    # the skill's names for belief objects: "<category>_<asset idx>_<n>"; basket names map to the leftmost-first order
    plan = next((x for x in log if "round" in x and "map" in x), {})
    gmap = plan.get("map") or {}

    def bidx(name):
        # basket_<idx>_0: catalogue indices 2 / 3 / 4 are basket0 / 1 / 2 left to right in both tasks' configs
        m = re.match(r"basket_(\d+)_\d+", name or "")
        return int(m.group(1)) - 2 if m else None
    free_map = {cat: bidx(c) for cat, c in gmap.items()} if not fixed else {}
    rp = None
    if replay_dir:
        f = os.path.join(replay_dir, os.path.basename(row_dir.rstrip("/")) + ".json.gz")
        if os.path.exists(f):
            rp = json.load(gzip.open(f))
    frames = {int(k): v for k, v in (rp or {}).get("frames", {}).items()}
    last = frames[max(frames)] if frames else None
    target_source = "instruction" if fixed else "skill_map"
    if not fixed and not free_map and frames:
        # an episode that ended on its success latch inside the skill logs no skill map: each category's basket is
        # where most of its instances ended (reporting only)
        target_source = "final_majority"
        fin: dict = {}
        for fo in frames[max(frames)]["objects"]:
            if fo["label"] in gt0:
                fin.setdefault(gt0[fo["label"]]["category"], []).append(basket_of(fo["pos"], baskets))
        free_map = {c: Counter(v).most_common(1)[0][0] for c, v in fin.items()}
    inst = []
    for g, o in sorted(gt0.items()):
        b = m0.get(g)
        bcat = _cat(bel0[b]["key"]) if b else None
        rec = {"gt": g, "category": o["category"], "asset": o["category_idx"], "found0": b is not None,
               "cat_ok0": bcat == o["category"], "exact_ok0": b is not None and bcat == o["category"]
               and _idx(bel0[b]["key"]) == o["category_idx"], "belief": b, "belief_cat": bcat}
        if fixed:
            rec["target"] = fixed.get(o["category"])
        else:
            rec["target"] = free_map.get(o["category"]) if rec["cat_ok0"] else None
        if frames:
            zs = [(k, fr_o["pos"]) for k, fr in sorted(frames.items()) for fr_o in fr["objects"] if fr_o["label"] == g]
            z0 = zs[0][1][2]
            rec["lifted"] = any(p[2] - z0 > 0.03 for _, p in zs)
            fin = next(fo for fo in last["objects"] if fo["label"] == g)["pos"]
            rec["final_basket"] = basket_of(fin, baskets)
            rec["final_xy"] = [round(float(fin[0]), 3), round(float(fin[1]), 3)]
            rec["in_target"] = rec["target"] is not None and rec["final_basket"] == rec["target"]
        inst.append(rec)
    # the skill's own per-object outcome records vs physics at their step
    claims = []
    gt_of_belief = {b: g for g, b in m0.items()}
    # HarnessRobot's instance names: "<category>_<asset idx>_<n>", given in (category, idx, x, y) order at first read
    name_of, used = {}, set()
    for b, v in sorted(bel0.items(), key=lambda kv: (_cat(kv[1]["key"]), _idx(kv[1]["key"]) or 0,
                                                       round(float(kv[1]["pos"][0]), 4), round(float(kv[1]["pos"][1]), 4))):
        stem, n = f"{_cat(v['key'])}_{_idx(v['key'])}", 0
        while f"{stem}_{n}" in used:
            n += 1
        name_of[f"{stem}_{n}"] = b
        used.add(f"{stem}_{n}")

    def gt_at(step, g):
        if not frames or g is None:
            return None
        k = min((k for k in frames if k >= (step or 0)), default=max(frames))
        return next((fo["pos"] for fo in frames[k]["objects"] if fo["label"] == g), None)
    for x in log:
        if "obj" not in x or "truth_in" not in x:
            continue
        g = gt_of_belief.get(name_of.get(x["obj"]))
        p = gt_at(x.get("step"), g)
        phys = None if p is None else basket_of(p, baskets) == bidx(x.get("container"))
        bl = name_of.get(x["obj"])
        sn = [sp for sp in snaps if sp["step"] <= (x.get("step") or 0) and bl in sp["objects"]]
        bp = sn[-1]["objects"][bl]["pos"] if sn else None
        bi = bidx(x.get("container"))
        bxy = None if bp is None or bi is None else bool(abs(bp[0] - baskets[bi][0]) <= BASKET_HALF[0]
                                                         and abs(bp[1] - baskets[bi][1]) <= BASKET_HALF[1])
        claims.append({"obj": x["obj"], "gt": g, "container": x.get("container"), "truth_in": x.get("truth_in"),
                       "belief_pos": None if bp is None else [round(float(v), 4) for v in bp], "belief_xy_in": bxy,
                       "physics_in": phys, "step": x.get("step"), "mode": x.get("mode"), "fallback": "fallback" in x,
                       "high_retry": "high_retry" in x, "why": x.get("why")})
    phantoms = [b for b in bel0 if b not in gt_of_belief]
    return {"row": os.path.basename(row_dir.rstrip("/")), "task": ep["task"], "layout_id": lid, "score": ep["score"],
            "replay_score": (rp or {}).get("score"), "steps": ep["steps_used"], "instruction": instr, "target_source": target_source,
            "n_gt": len(gt0), "n_belief0": len(bel0), "found0": sum(r["found0"] for r in inst),
            "cat_ok0": sum(r["cat_ok0"] for r in inst), "exact_ok0": sum(r["exact_ok0"] for r in inst),
            "phantoms0": [(b, bel0[b]["key"]) for b in phantoms], "lifted": sum(bool(r.get("lifted")) for r in inst),
            "in_target": sum(bool(r.get("in_target")) for r in inst), "claims": claims,
            "claimed_in": sum(1 for c in claims if c["truth_in"]),
            "false_in": sum(1 for c in claims if c["truth_in"] and c["physics_in"] is False),
            "false_out": sum(1 for c in claims if c["truth_in"] is False and c["physics_in"]), "map": gmap, "fixed_map": fixed, "instances": inst}


def summarize(rows: list) -> dict:
    out = {}
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == task]
        ins = [i for r in rs for i in r["instances"]]
        by_cat = {}
        for c in sorted({i["category"] for i in ins}):
            ci = [i for i in ins if i["category"] == c]
            by_cat[c] = {"n": len(ci), "found0": sum(i["found0"] for i in ci), "cat_ok0": sum(i["cat_ok0"] for i in ci),
                         "lifted": sum(bool(i.get("lifted")) for i in ci),
                         "in_target": sum(bool(i.get("in_target")) for i in ci)}
        # losses by first failing stage
        stage = Counter()
        for i in ins:
            if not i["found0"]:
                stage["missed_frame0"] += 1
            elif not i["cat_ok0"]:
                stage["wrong_category_frame0"] += 1
            elif i.get("target") is None:
                stage["no_target"] += 1
            elif not i.get("lifted"):
                stage["never_lifted"] += 1
            elif not i.get("in_target"):
                stage["lifted_not_in_target"] += 1
            else:
                stage["in_target"] += 1
        out[task] = {"rows": len(rs), "mean_score": round(100 * float(np.mean([r["score"] for r in rs])), 1),
                     "score_sd": round(100 * float(np.std([r["score"] for r in rs], ddof=1)), 1) if len(rs) > 1 else None,
                     "replay_matches": sum(1 for r in rs if r["replay_score"] is not None
                                           and abs(r["replay_score"] - r["score"]) < 1e-6),
                     "n_gt": sum(r["n_gt"] for r in rs), "n_belief0": sum(r["n_belief0"] for r in rs),
                     "found0": sum(r["found0"] for r in rs), "cat_ok0": sum(r["cat_ok0"] for r in rs),
                     "exact_ok0": sum(r["exact_ok0"] for r in rs), "phantoms0": sum(len(r["phantoms0"]) for r in rs),
                     "first_loss_stage": dict(stage),
                     "claims": sum(len(r["claims"]) for r in rs), "false_in": sum(r["false_in"] for r in rs),
                     "false_out": sum(r["false_out"] for r in rs), "by_category": by_cat,
                     "nine_object_rows": sum(1 for r in rs if r["n_gt"] == 9),
                     "nine_with_six_beliefs": sum(1 for r in rs if r["n_gt"] == 9 and r["n_belief0"] <= 6)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--layouts", required=True)
    ap.add_argument("--replay")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rows = [audit_row(d, a.layouts, a.replay) for d in sorted(glob.glob(os.path.join(a.rows, "c4b_*_1000??")))]
    res = {"summary": summarize(rows), "rows": rows}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))
    for r in rows:
        print(f"{r['row']}: score {r['score']} (replay {r['replay_score']}) gt {r['n_gt']} bel {r['n_belief0']} "
              f"found {r['found0']} cat {r['cat_ok0']} lifted {r['lifted']} in_target {r['in_target']} "
              f"claimed_in {r['claimed_in']}/{len(r['claims'])} false_in {r['false_in']} false_out {r['false_out']} phantoms {len(r['phantoms0'])}")


if __name__ == "__main__":
    main()
