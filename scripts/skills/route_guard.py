"""Routing guard for a new route (stream N): which tasks' instructions does it take, on real camera worlds?

    uv run python scripts/skills/route_guard.py --route pack --templates T.json --worlds W.json

``--layout-worlds TASK:GLOB`` adds our own dev layout records as worlds (a cell no camera report has). ``T.json``:
{task: [instruction templates]} (the public ``gen_instruction`` strings of RoboDojo's task files; ``{x}``
placeholders are filled with "object"). ``W.json``: [{"task", "record"}] (slim planning records of camera reports:
perception's belief, never GT). Two checks, CPU only, no network:
- own task: each world with its own task's instructions, through ``planner.plan`` with every new route on;
- cross: every instruction with one world of every task, through the route alone.
Prints the tasks the route fires on; exit 1 when it fires outside ``--expect`` (default: tasks named like the route's
cells)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

from dojo.runtime.estimate import objects_from_record
from dojo.skills import planner
from dojo.skills.routes import NEW_ROUTES, route
from dojo.skills.world import WorldState


def _world(rec: dict, instr: str, routes) -> WorldState:
    return WorldState("", instr, objects_from_record(rec, None, unique=True), {}, {}, notes={"routes": list(routes)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True)
    ap.add_argument("--skill", default=None, help="the route's skill name (default: whatever it returns)")
    ap.add_argument("--templates", required=True)
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--expect", default=None, help="regex of the task names it may fire on")
    ap.add_argument("--layout-worlds", action="append", default=[],
                    help="TASK:GLOB of our own dev layout records (the route's own cells, when no camera report has them)")
    a = ap.parse_args(argv)
    tmpl = {t: [re.sub(r"\{[^}]*\}", "object", s) for s in v] for t, v in json.load(open(a.templates)).items()}
    worlds = json.load(open(a.worlds))
    import glob
    for spec in a.layout_worlds:
        t, pat = spec.split(":", 1)
        worlds += [{"task": t, "record": json.load(open(f))} for f in sorted(glob.glob(pat))]
    expect = re.compile(a.expect or re.escape(a.route))
    own, n_own = Counter(), Counter()
    for w in worlds:
        for instr in tmpl.get(w["task"], []):
            n_own[w["task"]] += 1
            p = planner.plan(instr, _world(w["record"], instr, NEW_ROUTES))
            hit = route(instr.lower(), _world(w["record"], instr, [a.route]), [a.route])
            if hit is not None and (a.skill is None or p.skill == a.skill):
                own[w["task"]] += 1
    one = {}
    for w in worlds:
        one.setdefault(w["task"], w["record"])
    cross = Counter()
    for t, instrs in tmpl.items():
        for instr in instrs:
            for wt, rec in one.items():
                if route(instr.lower(), _world(rec, instr, [a.route]), [a.route]) is not None:
                    cross[(t, wt)] += 1
    print(f"worlds {len(worlds)} of {len(one)} tasks; templates {sum(len(v) for v in tmpl.values())} of {len(tmpl)} tasks")
    print("own-task fires:", {t: f"{own[t]}/{n_own[t]}" for t in sorted(own)} or "none")
    print("cross fires (instruction task, world task):", sorted(cross) or "none")
    bad = sorted({t for t in own if not expect.search(t)} | {t for t, _ in cross if not expect.search(t)})
    print("outside", a.expect or a.route, ":", bad or "none")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
