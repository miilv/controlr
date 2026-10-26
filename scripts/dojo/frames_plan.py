"""Which perception checks of a set of dev-eval episodes to regenerate as frames (rendered replay), under a disk budget.

    python3 scripts/dojo/frames_plan.py plan --budget-mb N --out plan.json EVAL_DIR [EVAL_DIR ...]
    python3 scripts/dojo/frames_plan.py index --plan plan.json --replays DIR --layouts LAYOUT_ROOT --out index.json

``plan``: every ``held_look`` / ``post_place`` event in the episodes' reports (``<task>_perception_verify-bundle_<id>
_report.json.gz``). Negatives (held=False / ok=False) come first, then an equal number of positives (spread over the
episodes), as many as ``--budget-mb`` allows at ~0.9 MB per frame set. Writes per episode the steps for
``replay_kits.py run --save-obs-at``.
``index``: one entry per regenerated set with the check (verdict, confidence, source, iou, ...), the TRUE pose of
every task object from the replay, and the physical object the check was about. The match is by ASSET KEY + POSITION:
the belief label's asset (``category_idx`` in the episode's record) -> the truth objects with that asset -> the one
nearest the checking arm's flange at that step. Labels of same-asset objects are exchangeable, so matching by label
would be meaningless. ``in_jaws`` = within 0.20 m of that flange (the X5 holds an object ~0.15 m below it) and > 1 cm above its layout height.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

MB_PER_SET = 0.9


def _reports(d: Path):
    for f in sorted(d.glob("*_perception_verify-bundle_*_report.json.gz")):
        name = f.name
        task = name.split("_perception_verify-bundle_")[0]
        lid = int(name.split("_perception_verify-bundle_")[1].split("_")[0])
        yield task, lid, f


def is_negative(e: dict) -> bool:
    return (e.get("kind") == "held_look" and e.get("held") is False) or (e.get("kind") == "post_place" and e.get("ok") is False)


def plan_cmd(a) -> None:
    neg, pos = [], []
    for d in a.dirs:
        for task, lid, f in _reports(Path(d)):
            with gzip.open(f, "rt") as fh:
                rep = json.load(fh)
            for e in rep.get("events") or []:
                if e.get("kind") in ("held_look", "post_place") and e.get("step") is not None:
                    item = {"dir": str(d), "task": task, "layout": lid, "step": int(e["step"]), "kind": e["kind"],
                            "label": e.get("label"), "arm": e.get("arm"), "negative": is_negative(e)}
                    (neg if item["negative"] else pos).append(item)
    n_max = int(a.budget_mb / MB_PER_SET)
    take_neg = neg[:n_max]
    room = max(0, n_max - len(take_neg))
    k = min(len(take_neg), room, len(pos))
    stride = max(1, len(pos) // k) if k else 1
    take_pos = pos[::stride][:k]
    chosen = take_neg + take_pos
    eps: dict = {}
    for c in chosen:
        key = f"{c['dir']}|{c['task']}|{c['layout']}"
        eps.setdefault(key, set()).add(c["step"])
    out = {"budget_mb": a.budget_mb, "mb_per_set": MB_PER_SET, "n_checks_total": len(neg) + len(pos),
           "n_negatives_total": len(neg), "n_chosen_negatives": len(take_neg), "n_chosen_positives": len(take_pos),
           "checks": chosen, "episodes": [{"dir": k.split("|")[0], "task": k.split("|")[1], "layout": int(k.split("|")[2]),
                                           "steps": sorted(v)} for k, v in sorted(eps.items())]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[plan] {len(neg) + len(pos)} checks ({len(neg)} negatives); chosen {len(take_neg)} negatives + "
          f"{len(take_pos)} positives in {len(eps)} episodes, {len({(c['layout'], c['task'], c['step']) for c in chosen})} "
          f"frame sets (budget {a.budget_mb} MB)")


def _dist(p, q) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(p[:3], q[:3])) ** 0.5


def index_cmd(a) -> None:
    plan = json.loads(Path(a.plan).read_text())
    out = []
    for c in plan["checks"]:
        rp_path = Path(a.replays) / c["task"] / f"{c['task']}_{c['layout']}_replay.json"
        if not rp_path.exists():
            continue
        rp = json.loads(rp_path.read_text())
        saved = rp.get("saved_obs", {}).get(str(c["step"]))
        if saved is None:
            continue
        with gzip.open(Path(c["dir"]) / f"{c['task']}_perception_verify-bundle_{c['layout']}_report.json.gz", "rt") as fh:
            rep = json.load(fh)
        ev = next((e for e in rep["events"] if e.get("step") == c["step"] and e.get("kind") == c["kind"]
                   and e.get("label") == c["label"]), {})
        recs = rep.get("records") or {}
        rec = recs.get("fused") or recs.get("scan") or recs.get("initial") or {}
        bel_asset = {r.get("label"): (typ, cat, r.get("category_idx")) for typ, cats in rec.items()
                     if isinstance(cats, dict) and not typ.startswith("_") for cat, rows in cats.items()
                     if isinstance(rows, list) for r in rows if isinstance(r, dict)}
        lay = json.loads((Path(a.layouts) / c["task"] / f"{c['layout']}.settled.json").read_text())
        truth_asset = {r.get("label"): (typ, cat, r.get("category_idx")) for typ, cats in lay.items()
                       if isinstance(cats, dict) and not typ.startswith("_") for cat, rows in cats.items()
                       if isinstance(rows, list) for r in rows if isinstance(r, dict) and r.get("label")}
        layout_z = {r.get("label"): float(r["default_pos"][2]) for typ, cats in lay.items() if isinstance(cats, dict)
                    and not typ.startswith("_") for cat, rows in cats.items() if isinstance(rows, list)
                    for r in rows if isinstance(r, dict) and r.get("label") and r.get("default_pos")}
        fr = rp["frames"][c["step"]]
        flange = fr["flange"].get(c.get("arm") or "", None)
        key = bel_asset.get(c["label"])
        cands = [t for t, k in truth_asset.items() if k == key and t in fr["objects"]]
        phys = None
        if cands and flange:
            phys = min(cands, key=lambda t: _dist(fr["objects"][t], flange))
        elif len(cands) == 1:
            phys = cands[0]
        in_jaws = None
        if phys and flange:
            p = fr["objects"][phys]
            in_jaws = _dist(p, flange) < 0.20 and p[2] > layout_z.get(phys, p[2]) + 0.01
        out.append({"task": c["task"], "layout": c["layout"], "step": c["step"], "dir": saved["dir"],
                    "check": {k: ev.get(k) for k in ("kind", "label", "arm", "held", "confidence", "source", "dropped", "ok",
                                                     "iou", "visible_frac", "shift_mm", "why", "calls", "dark_risk")},
                    "negative": c["negative"], "belief_asset": key, "physical_object": phys, "physical_in_jaws": in_jaws,
                    "objects_true": fr["objects"], "grip_cmd": fr["grip_cmd"], "flange": fr["flange"]})
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[index] {len(out)} frame sets -> {a.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--budget-mb", type=float, required=True)
    p.add_argument("--out", required=True)
    i = sub.add_parser("index")
    i.add_argument("--plan", required=True)
    i.add_argument("--replays", required=True)
    i.add_argument("--layouts", required=True)
    i.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    (plan_cmd if a.cmd == "plan" else index_cmd)(a)


if __name__ == "__main__":
    main()
