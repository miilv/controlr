"""Pre-settled layout records: generate -> settle in RoboDojo (physics-only) -> write the settled poses back.

For every stored dev layout ``layouts/<task>/<id>.json`` (kept unchanged) this writes
``layouts/<task>/<id>.settled.json``: the same record with every Rigid / Dynamic / Articulation object's
``default_pos`` / ``default_ori`` replaced by where it actually came to rest after RoboDojo's own reset +
settle (+ ``extra_steps`` physics steps), and ``_simlab.settled`` with the provenance. Layouts are dropped
(no file; status in ``settled_index.json``) when

* pass 1: a task object still moves after the extra steps (> 1 mm or 1 deg over the last ``rest_window``
  physics steps = 0.4 s; velocity readouts are noise on resting contacts) or fell below the table, or
* pass 2: the settled record, loaded with a fresh RoboDojo reset, fails RoboDojo's stability check or moves
  more than ``drift_tol_mm``.

A layout whose objects moved during pass 1 but came to rest (fill_egg_holder: the egg basket is a
collision-less Geometry, so the eggs drop to the table) is kept: the settled record is what RoboDojo
would actually show at episode start.

    scripts/simlab/rdpy.sh -m dojo.simlab.presettle --task fill_egg_holder [--physics-only]
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path

import numpy as np

from dojo.simlab import paths
from dojo.simlab.layout import TABLE_TOP_Z, iter_objects, quat_angle_deg

SETTLER = "simlab/presettle v1"
MOVABLE = ("Rigid", "Dynamic", "Articulation")
ID_RE = re.compile(r"^\d+\.json$")


def layout_files(task: str, root: Path | None = None) -> list[Path]:
    d = (root or paths.LAYOUTS) / task
    return sorted((p for p in d.glob("*.json") if ID_RE.match(p.name)), key=lambda p: int(p.stem))


def _key(otype, cat, idx, label, clutter):
    return (otype, cat, int(idx), label, bool(clutter))


def write_back(layout: dict, live: list[dict]) -> tuple[dict, int]:
    """Copy of ``layout`` with movable objects at their live (settled) poses."""
    pool: dict = {}
    for o in live:
        if o["type"] in MOVABLE and o["pos"] is not None:
            pool.setdefault(_key(o["type"], o["category"], o["category_idx"], o["label"], o["clutter"]), []).append(o)
    out = copy.deepcopy(layout)
    n = 0
    for t, cat, inst in iter_objects(out):
        if t not in MOVABLE:
            continue
        cands = pool.get(_key(t, cat, inst["category_idx"], inst.get("label"), inst.get("type") == "cluttered"))
        if not cands:
            continue
        o = cands.pop(0)
        q = np.asarray(o["quat"], float)
        inst["default_pos"] = [float(v) for v in o["pos"]]
        inst["default_ori"] = [float(v) for v in q / np.linalg.norm(q)]
        n += 1
    return out, n


def at_rest(before: list[dict], after: list[dict], table: bool, tol_mm: float = 1.0,
            tol_deg: float = 1.0) -> tuple[bool, list[str]]:
    """Task objects moved < ``tol_mm`` / ``tol_deg`` between two samples (velocity readouts are useless here:
    PhysX reports mm/s and 0.1 rad/s on blocks resting still), and none fell below the table."""
    prev = {o["inst_name"]: o for o in before}
    bad = []
    for o in after:
        if o["type"] not in ("Rigid", "Dynamic") or o["clutter"] or o["pos"] is None:
            continue
        p = prev.get(o["inst_name"])
        if p is not None and p["pos"] is not None:
            mm = float(np.linalg.norm(np.subtract(o["pos"], p["pos"]))) * 1000
            deg = quat_angle_deg(o["quat"], p["quat"])
            if mm > tol_mm or deg > tol_deg:
                bad.append(f"{o['label']} still moving ({mm:.2f} mm, {deg:.2f} deg in the rest window)")
        if table and o["pos"][2] < TABLE_TOP_Z - 0.05:
            bad.append(f"{o['label']} fell (z={o['pos'][2]:.3f})")
    return not bad, bad


def drift(live: list[dict]) -> tuple[float, float]:
    mm = deg = 0.0
    for o in live:
        if o["type"] not in MOVABLE or o["clutter"] or o["pos"] is None:
            continue
        mm = max(mm, float(np.linalg.norm(np.subtract(o["pos"], o["layout_pos"]))) * 1000)
        deg = max(deg, quat_angle_deg(o["quat"], o["layout_quat"]))
    return mm, deg


def presettle_task(task: str, physics_only: bool = True, extra_steps: int = 250, rest_window: int = 100,
                   drift_tol_mm: float = 5.0, ids: list[int] | None = None) -> dict:
    from dojo.simlab.runner import RoboDojoRunner, jsonable
    files = [p for p in layout_files(task) if not ids or int(p.stem) in ids]
    run = RoboDojoRunner(task, physics_only=physics_only, need_planner=False,
                         workdir=str(paths.DATA / "simlab" / "work" / f"presettle_{task}"))
    index_p = paths.LAYOUTS / task / "settled_index.json"
    index = json.loads(index_p.read_text()) if index_p.exists() else {}
    for f in files:
        t0 = time.perf_counter()
        lay = json.loads(f.read_text())
        rec: dict = {"from": f.name}
        info = run.load(lay)
        rec["pass1_robodojo_stable"] = bool(info["stable"])
        for _ in range(extra_steps - rest_window):
            run.env.sim_step(render=False)
        before = run.objects_gt()
        for _ in range(rest_window):
            run.env.sim_step(render=False)
        live = run.objects_gt()
        rec["pass1_drift_mm"], rec["pass1_drift_deg"] = (round(v, 2) for v in drift(live))
        ok, why = at_rest(before, live, "Table" in lay)
        out = f.with_name(f"{f.stem}.settled.json")
        if not ok:
            rec.update({"status": "dropped: not at rest", "why": why})
            out.unlink(missing_ok=True)
        else:
            settled, n = write_back(lay, live)
            info2 = run.load(settled)
            live2 = run.objects_gt() if info2["stable"] else []
            d2 = drift(live2) if live2 else (float("nan"), float("nan"))
            rec.update({"objects_written": n, "pass2_robodojo_stable": bool(info2["stable"]),
                        "pass2_drift_mm": round(d2[0], 3), "pass2_drift_deg": round(d2[1], 3)})
            if info2["stable"] and d2[0] <= drift_tol_mm:
                meta = dict(settled.get("_simlab") or {})
                meta["settled"] = {"settler": SETTLER, "from": f.name, "generator": meta.get("generator"),
                                   "physics": "isaac-5.1 physics-only" if physics_only else "isaac-5.1",
                                   "extra_steps": extra_steps, **{k: rec[k] for k in (
                                       "pass1_robodojo_stable", "pass1_drift_mm", "pass2_drift_mm")}}
                settled["_simlab"] = meta
                out.write_text(json.dumps(jsonable(settled), indent=1))
                rec["status"] = "ok"
            else:
                rec["status"] = "dropped: settled record not stable"
                out.unlink(missing_ok=True)
        rec["s"] = round(time.perf_counter() - t0, 1)
        index[f.stem] = rec
        index_p.write_text(json.dumps(index, indent=1))
        print(f"[presettle] {task} {f.stem}: {rec['status']} pass1 stable={rec['pass1_robodojo_stable']} "
              f"drift {rec['pass1_drift_mm']}mm -> pass2 {rec.get('pass2_drift_mm')}mm ({rec['s']}s)", flush=True)
    run.close()
    n_ok = sum(1 for k, v in index.items() if v["status"] == "ok")
    print(f"[presettle] {task}: {n_ok}/{len(index)} settled records", flush=True)
    return index


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--extra-steps", type=int, default=250)
    ap.add_argument("--rendered", action="store_true", help="use the full (rendering) env, e.g. on compute2")
    a = ap.parse_args(argv)
    presettle_task(a.task, physics_only=not a.rendered, extra_steps=a.extra_steps, ids=a.ids)


if __name__ == "__main__":
    main()
