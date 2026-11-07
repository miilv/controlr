#!/usr/bin/env python3
"""Read-only analysis of the registered private POUR panel; no simulator imports."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

EPS = 1e-8


def moments(values):
    return {"mean": statistics.mean(values) if values else None,
            "sd": statistics.stdev(values) if len(values) > 1 else None}


def scored_pairs(pairs, full_only=False):
    values = [[100 * int(r["success"]) if full_only else 100 * r["score"] for r in pair] for pair in pairs]
    delta = [b - a for a, b in values]
    wins = sum(d >= 10 - EPS for d in delta)
    losses = sum(d <= -10 + EPS for d in delta)
    return {"A": moments([v[0] for v in values]), "B": moments([v[1] for v in values]),
            "delta": moments(delta), "wins": wins, "losses": losses, "ties": len(pairs) - wins - losses}


def clean_success(row, task):
    fluids = (row.get("pour_final") or {}).get("fluids", [])
    return bool(row["success"] and len(fluids) == (3 if task == "pour_by_language" else 1)
                and all(f.get("predicate_now") for f in fluids))


def details(run):
    try:
        return json.loads(run.get("details") or "{}")
    except (ValueError, TypeError):
        return {}


def failure_stage(row, task):
    if row["success"]:
        return "success"
    final = row.get("pour_final") or {}
    pending = final.get("pending_upright_labels")
    first_pending = None
    if pending:
        first_pending = min(int(x.rsplit("_", 1)[1]) for x in pending) if task == "pour_by_language" else 0
    runs = row.get("ported") or []
    for i, run in enumerate(runs):
        d = details(run)
        error, stop = str(run.get("error") or ""), str(d.get("pour_stop") or "")
        if "no reachable" in error or (d.get("plan") or {}).get("none"):
            return "planning"
        if "neck grasp failed" in error or "pre-held" in error:
            return "grasp/lift"
        if stop == "in-hand slip":
            return "tilt/slip"
        if "budget" in stop:
            return "profile/budget"
        if stop.startswith("move refused"):
            return "profile/lease" if "lease" in stop else "profile/move"
        tilt = (d.get("final") or {}).get("bottle_tilt")
        if tilt is not None and tilt > 30:
            reasons = [x.get("plan_fail_reason") for x in (d.get("return") or {}).get("untilt", [])]
            if "budget" in reasons:
                return "untilt/budget"
            if "lease_too_short" in reasons:
                return "untilt/lease"
            return "upright/return"
        if i == first_pending:
            fluids = final.get("fluids", [])
            if i < len(fluids):
                f = fluids[i]
                if f.get("predicate_now"):
                    return "upright edge/checker"
                if f.get("raw_residual_fraction", 0) > (.1 if task == "pour_by_language" else .15):
                    return "fluid/residual"
                return "fluid/capture/checker"
    return "checker/other"


def paired_summary(a_rows, b_rows, ids, task):
    def keyed(rows):
        assert len({r["seed"] for r in rows}) == len(rows), "duplicate episode"
        return {r["seed"]: r for r in rows}
    a, b = keyed(a_rows), keyed(b_rows)
    pairs, excluded, missing = [], [], []
    for seed in ids:
        if seed not in a or seed not in b:
            missing.append(seed)
            continue
        pair = (a[seed], b[seed])
        if not all(r.get("stable") and not r.get("infra_failure") for r in pair):
            excluded.append(seed)
            continue
        pairs.append(pair)
    result = {"n": len(pairs), "expected": len(ids), "missing": missing, "excluded": excluded,
              "complete": not missing and not excluded and len(pairs) == len(ids),
              "official": scored_pairs(pairs), "full_only": scored_pairs(pairs, True)}
    for key, fn in (("successes", lambda r: int(r["success"])),
                    ("clean_successes", lambda r: int(clean_success(r, task)))):
        result[key] = {arm: sum(fn(p[i]) for p in pairs) for i, arm in enumerate(("A", "B"))}
    result["stages"] = {arm: dict(Counter(failure_stage(p[i], task) for p in pairs))
                        for i, arm in enumerate(("A", "B"))}
    result["steps"] = {arm: {"median": statistics.median([p[i]["steps"] for p in pairs]) if pairs else None,
                             "max": max([p[i]["steps"] for p in pairs], default=None)}
                       for i, arm in enumerate(("A", "B"))}
    return result


def decision(cells):
    if not all(c["complete"] for c in cells.values()):
        return {"complete": False, "passes": None}
    gain = cells["pour_by_language"]
    guard = cells["pour_liquid_into_cup"]
    gain_pass = all(gain[k]["delta"]["mean"] >= 5 - EPS
                    and gain[k]["wins"] >= 2 * gain[k]["losses"]
                    and gain[k]["wins"] - gain[k]["losses"] >= 2 for k in ("official", "full_only"))
    guard_pass = all(guard[k]["delta"]["mean"] >= -5 - EPS
                     and guard[k]["wins"] >= guard[k]["losses"] for k in ("official", "full_only"))
    clean = gain["clean_successes"]["B"] >= 1
    return {"complete": True, "gain_pass": gain_pass, "guard_pass": guard_pass,
            "clean_feasibility": clean, "passes": gain_pass and guard_pass and clean}


def analyze(root, registration):
    assert set(registration["cells"]) == {"pour_by_language", "pour_liquid_into_cup"}, "unexpected cells"
    assert set(registration["arms"]) == {"A", "B"}, "unexpected arms"
    result = {"registration": registration, "cells": {}}
    for task, cell in registration["cells"].items():
        assert all(100080 <= i < 100100 for i in cell["ids"]), "not the registered fresh dev block"
        arms = {}
        for arm in ("A", "B"):
            p = root / f"panel-{arm}-{task}" / f"{task}_episodes.jsonl"
            rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
            for row in rows:
                assert row["code"]["commit"] == registration["controller_commit"], "mixed controller source"
                expected = registration["arms"][arm]
                assert [row["skill_attrs"][f"ported_pour_{k}"] for k in ("pace", "aim", "profile")] == [expected[k] for k in ("pace", "aim", "profile")]
            arms[arm] = rows
        result["cells"][task] = paired_summary(arms["A"], arms["B"], cell["ids"], task)
    result["decision"] = decision(result["cells"])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "dojo-data/pour")
    ap.add_argument("--registration", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    reg = args.registration or args.root / "panel-registration.json"
    text = json.dumps(analyze(args.root, json.loads(reg.read_text())), indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
