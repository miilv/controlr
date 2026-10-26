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
        if otype == "Geometry":
            pegs = _pegs(assets, otype, cat, int(inst["category_idx"]), rows[-1]["pos"], rows[-1]["quat"])
            if pegs:
                rows[-1]["pegs"] = pegs
            hooks = _hooks(assets, otype, cat, int(inst["category_idx"]), rows[-1]["pos"], rows[-1]["quat"])
            if hooks:
                rows[-1]["hang_points"] = hooks
    return rows


def _hooks(assets: AssetDB, otype: str, cat: str, idx: int, pos, quat) -> list:
    """World points of a rack's hook frames (functional points named hook / support_point / hang), for the kinematic
    env: an object let go around one hangs there."""
    import re
    from dojo.skills.geometry import quat_to_R
    try:
        info = assets.get(otype, cat, idx)
    except (OSError, KeyError):
        return []
    R = quat_to_R(np.asarray(quat, float))
    out = []
    for tag, v in (info.functional or {}).items():
        if re.search(r"hook|support_point|hang", tag.lower()):
            for f in (v or {}).get("frame") or []:
                w = R @ np.asarray(f[:3], float) + np.asarray(pos, float)
                out.append([float(x) for x in w])
    return out


def _pegs(assets: AssetDB, otype: str, cat: str, idx: int, pos, quat) -> list:
    """[[x, y, plate_z], ...] (world) of a base's pegs (support frames named pole / peg), for the kinematic env: the
    plate height around each peg from the mesh (else the frame's height)."""
    import re
    from dojo.skills.geometry import quat_to_R
    try:
        info = assets.get(otype, cat, idx, mesh=True)
    except (OSError, KeyError):
        return []
    R = quat_to_R(np.asarray(quat, float))
    out = []
    for tag, frames in (info.support or {}).items():
        if not re.search(r"pole|peg", tag.lower()):
            continue
        for f in frames:
            c = np.asarray(f[0], float)[:3]
            z = float(c[2])
            if info.mesh is not None:
                V = info.mesh[0]
                d = np.hypot(V[:, 0] - c[0], V[:, 1] - c[1])
                ring = V[(d > 0.008) & (d < 0.02)]
                if len(ring):
                    z = float(np.percentile(ring[:, 2], 90))
            w = R @ np.array([c[0], c[1], z]) + np.asarray(pos, float)
            out.append([float(w[0]), float(w[1]), float(w[2])])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("layout")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--step-lim", type=int, default=1000)
    ap.add_argument("--pose", action="append", default=[], help="LABEL:x,y,z,qw,qx,qy,qz")
    ap.add_argument("--trace-poses", default=None, help="a g3 failure trace (.json.gz): start from its first frame's "
                    "object poses (the twin's settled state)")
    ap.add_argument("--trace-step", type=int, default=0, help="with --trace-poses: the first frame at or after this "
                    "env step (-1: the last frame)")
    ap.add_argument("--objects-root", default=None, help="asset metadata root (default RoboDojo's)")
    ap.add_argument("--catalog", default=None, help="mesh catalogue (default DOJO_CATALOG / perception's)")
    a = ap.parse_args(argv)
    lay = json.loads(Path(a.layout).read_text())
    assets = AssetDB(a.objects_root, a.catalog)
    rows = rows_from_layout(lay, assets)
    if a.trace_poses:
        import gzip
        frs = json.load(gzip.open(a.trace_poses, "rt"))["frames"]
        fr0 = (frs[-1] if a.trace_step < 0 else next((f for f in frs if f["step"] >= a.trace_step), frs[-1]))["obj"]
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
