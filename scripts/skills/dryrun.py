"""Dry run of ``planner.solve`` on the kinematic env (CPU, no Isaac): a layout record + the assets'
metadata (+ mesh catalogue) -> ``KinematicEnv`` -> the same entry point G3 runs.

    DOJO_CATALOG=~/dojo-data/skills2/catalog RD_ROOT=~/dojo-robodojo/RoboDojo \\
      .venv/bin/python scripts/skills/dryrun.py LAYOUT.json --instruction "..." [--pose LABEL:x,y,z,qw,qx,qy,qz]

The kinematic env has no physics: a closed gripper carries the object nearest the TCP, an opened one
drops it straight down. It checks planning (roles, reach, collisions, step counts), not contacts.
``--pose`` overrides an object's pose (e.g. a scripted opponent's effect the kinematic env cannot play).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dojo.simlab.layout import iter_objects  # noqa: E402
from dojo.skills import planner  # noqa: E402
from dojo.skills.env import KinematicEnv  # noqa: E402
from dojo.skills.skills import GTProvider, Skills  # noqa: E402
from dojo.skills.world import AssetDB  # noqa: E402


def rows_from_layout(lay: dict, assets: AssetDB) -> list[dict]:
    rows = []
    for otype, cat, inst in iter_objects(lay):
        clutter = inst.get("type") == "cluttered"
        try:
            info = assets.get(otype, cat, int(inst["category_idx"]), clutter=clutter)
            half = (info.bbox_max - info.bbox_min) / 2
            ctr = (info.bbox_max + info.bbox_min) / 2
        except (OSError, KeyError):
            half, ctr = np.full(3, 0.02), np.zeros(3)
        rows.append({"label": inst.get("label") or f"{cat}{len(rows)}", "inst_name": inst.get("label"), "type": otype,
                     "category": cat, "category_idx": int(inst["category_idx"]), "clutter": clutter,
                     "pos": np.asarray(inst["default_pos"], float), "quat": np.asarray(inst["default_ori"], float),
                     "half": half, "center": ctr})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("layout")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--step-lim", type=int, default=1000)
    ap.add_argument("--pose", action="append", default=[], help="LABEL:x,y,z,qw,qx,qy,qz")
    ap.add_argument("--trace-poses", default=None, help="a g3 failure trace (.json.gz): start from its first frame's "
                    "object poses (the twin's settled state)")
    ap.add_argument("--objects-root", default=None, help="asset metadata root (default RoboDojo's)")
    ap.add_argument("--catalog", default=None, help="mesh catalogue (default DOJO_CATALOG / perception's)")
    a = ap.parse_args(argv)
    lay = json.loads(Path(a.layout).read_text())
    assets = AssetDB(a.objects_root, a.catalog)
    rows = rows_from_layout(lay, assets)
    if a.trace_poses:
        import gzip
        fr0 = json.load(gzip.open(a.trace_poses, "rt"))["frames"][0]["obj"]
        for r in rows:
            if r["label"] in fr0:
                v = fr0[r["label"]]
                r["pos"], r["quat"] = np.asarray(v[:3], float), np.asarray(v[3:7], float)
    for spec in a.pose:
        lab, vals = spec.split(":")
        v = [float(x) for x in vals.split(",")]
        r = next(r for r in rows if r["label"] == lab)
        r["pos"], r["quat"] = np.asarray(v[:3]), np.asarray(v[3:7])
    env = KinematicEnv(rows, step_lim=a.step_lim, instruction=a.instruction)
    logs = []
    prov = GTProvider(env, assets, instruction=a.instruction)
    sk = Skills(env, prov, reserve=30, log=lambda m: (logs.append(m), print(m, flush=True)))
    t0 = time.perf_counter()
    try:
        res = planner.solve(sk)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        res = {"ok": False, "error": str(e)}
    print(json.dumps({"result": {k: str(v)[:300] for k, v in res.items()}, "steps": env.steps_used,
                      "plan_s": round(time.perf_counter() - t0, 1),
                      "history": [(h.skill, h.status, h.steps, h.arm, h.note[:120]) for h in sk.history]},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
