"""Pre-settled layout records: generate -> settle in RoboDojo (physics-only) -> write the settled poses back.

For every stored dev layout ``layouts/<task>/<id>.json`` (kept unchanged) this writes
``layouts/<task>/<id>.settled.json``: the same record with every Rigid / Dynamic object's
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

``--keep-raw-unmoved`` (off by default) re-tries only ids dropped in pass 2 that have no settled file: objects that did not move in pass 1 (<= ``UNMOVED_MM`` / ``UNMOVED_DEG`` from the record) keep their
generated pose instead of the live one. insert_key's key (friction 0) rests at its generated pose (0.03 mm in
pass 1), but written back with the 0.26 deg tilt PhysX reports it slides 6-37 mm on the reload and fails pass 2.
Such records carry ``_simlab.presettle = "keep_raw_unmoved"``; their index entry keeps the default run's
entry under ``previous``. Existing settled records are never rewritten.

``--raw-check`` (off by default) is the stability test for a task that RUNS ITS RAW RECORD (``settled: false``:
make_toast and the two conveyor tasks; pass 1 / pass 2 can never settle them: a belt moves its riders by design, the
toaster's slices re-drop 7-9 mm on every reload). It writes no record, only ``raw_index.json`` {id: status}; "raw ok"
iff the raw record as stored (1) passes RoboDojo's reset check, (2) runs ``extra_steps`` physics steps without an
exception, and (3) is at rest over the last ``rest_window`` steps (``at_rest``, 1 mm / 1 deg) after the belt's motion,
the median displacement of the Rigid task objects, is subtracted from those riders (only in a scene with no Table; the
Dynamic conveyor body itself is judged as is), and (4) no task object fell: below the table top - 5 cm, or (no table)
more than 5 cm below its own record z.

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

SETTLER = "simlab/presettle v2"   # v2: articulations not written back; optional re-settle passes
# Articulations keep their generated pose (v2): RoboDojo reports some of them (make_toast's toasters 0/1) with a
# constant ~7 mm root offset from the record's default_pos, so writing the reported pose back moved them 7 mm on
# every reload and failed pass 2 (24/24). They are fixtures that do not settle anyway.
MOVABLE = ("Rigid", "Dynamic")
KEEP_RAW_UNMOVED = "keep_raw_unmoved"   # _simlab.presettle of records made with --keep-raw-unmoved
UNMOVED_MM, UNMOVED_DEG = 1.0, 1.0      # "did not move in pass 1": the at_rest tolerances
RAW_OK = "raw ok"                       # raw_index.json status of a raw record that passes --raw-check
RAW_CHECK = "simlab/raw-check v1"       # the raw-check rule's version (in every raw_index.json entry)
FALL_M = 0.05                           # "fell": more than 5 cm below the table top / (no table) its record z
ID_RE = re.compile(r"^\d+\.json$")


def layout_files(task: str, root: Path | None = None) -> list[Path]:
    d = (root or paths.LAYOUTS) / task
    return sorted((p for p in d.glob("*.json") if ID_RE.match(p.name)), key=lambda p: int(p.stem))


def _key(otype, cat, idx, label, clutter):
    return (otype, cat, int(idx), label, bool(clutter))


def unmoved(live: list[dict], tol_mm: float = UNMOVED_MM, tol_deg: float = UNMOVED_DEG) -> set[str]:
    """``inst_name`` of the movable objects within ``tol_mm`` / ``tol_deg`` of the pose their record gave them."""
    out = set()
    for o in live:
        if o["type"] not in MOVABLE or o["pos"] is None or o.get("layout_pos") is None or o.get("layout_quat") is None:
            continue
        mm = float(np.linalg.norm(np.subtract(o["pos"], o["layout_pos"]))) * 1000
        if mm <= tol_mm and quat_angle_deg(o["quat"], o["layout_quat"]) <= tol_deg:
            out.add(o["inst_name"])
    return out


def write_back(layout: dict, live: list[dict], keep: set[str] | None = None) -> tuple[dict, int]:
    """Copy of ``layout`` with movable objects at their live (settled) poses; objects whose ``inst_name`` is in
    ``keep`` stay at the record's pose (not counted)."""
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
        if keep and o["inst_name"] in keep:
            continue
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


def raw_at_rest(before: list[dict], after: list[dict], table: bool, tol_mm: float = 1.0,
                tol_deg: float = 1.0) -> tuple[bool, list[str], list[float]]:
    """``--raw-check``'s rest test of a record that runs raw: ``at_rest`` after subtracting the belt (scenes with no
    Table: the median window displacement of the Rigid task objects, i.e. the riders, is taken off each rider; the
    Dynamic conveyor body is judged as is), plus the fall floor (no table: 5 cm below the object's record z).
    Returns (ok, why, belt displacement over the window in m)."""
    if table:
        ok, why = at_rest(before, after, True, tol_mm, tol_deg)
        return ok, why, [0.0, 0.0, 0.0]
    prev = {o["inst_name"]: o for o in before}
    riders = {o["inst_name"] for o in after if o["type"] == "Rigid" and not o["clutter"] and o["pos"] is not None
              and (prev.get(o["inst_name"]) or {}).get("pos") is not None}
    belt = (np.median([np.subtract(o["pos"], prev[o["inst_name"]]["pos"]) for o in after if o["inst_name"] in riders],
                      axis=0) if riders else np.zeros(3))
    corrected = [dict(o, pos=[float(v) for v in np.subtract(o["pos"], belt)]) if o["inst_name"] in riders else o
                 for o in after]
    ok, why = at_rest(before, corrected, False, tol_mm, tol_deg)
    for o in after:
        if o["type"] not in MOVABLE or o["clutter"] or o["pos"] is None or o.get("layout_pos") is None:
            continue
        if o["pos"][2] < o["layout_pos"][2] - FALL_M:
            why.append(f"{o['label']} fell (z={o['pos'][2]:.3f}, record z={o['layout_pos'][2]:.3f})")
            ok = False
    return ok, why, [float(v) for v in belt]


def drift(live: list[dict], top: list | None = None) -> tuple[float, float]:
    """Max task-object displacement from the record (mm, deg); ``top`` collects the 3 largest (label, mm, type)."""
    mm = deg = 0.0
    rows = []
    for o in live:
        if o["clutter"] or o["pos"] is None or o.get("layout_pos") is None:
            continue
        d = float(np.linalg.norm(np.subtract(o["pos"], o["layout_pos"]))) * 1000
        rows.append((o.get("label"), round(d, 2), o["type"]))
        if o["type"] not in MOVABLE:
            continue
        mm = max(mm, d)
        deg = max(deg, quat_angle_deg(o["quat"], o["layout_quat"]))
    if top is not None:
        top.extend(sorted(rows, key=lambda r: -r[1])[:3])
    return mm, deg


def select_files(task: str, ids: list[int] | None = None, reserved_slice: bool = False,
                 keep_raw_unmoved: bool = False, root: Path | None = None) -> list[Path]:
    """The raw records to settle. ``keep_raw_unmoved``: only ids the index lists as dropped in pass 2 (at rest in
    pass 1, settled record not stable) that have no settled file (an existing settled record is never rewritten;
    pass 1 is the same run, so a "not at rest" id is not re-tried)."""
    files = [p for p in layout_files(task, root) if not ids or int(p.stem) in ids]
    if not reserved_slice:                      # D-C10: ids >= paths.RESERVED_MIN_ID belong to joint paired runs
        files = [p for p in files if not paths.is_reserved_id(p.stem)]
    if keep_raw_unmoved:
        index_p = (root or paths.LAYOUTS) / task / "settled_index.json"
        index = json.loads(index_p.read_text()) if index_p.exists() else {}
        files = [p for p in files if index.get(p.stem, {}).get("status") == "dropped: settled record not stable"
                 and not p.with_name(f"{p.stem}.settled.json").exists()]
    return files


def presettle_task(task: str, physics_only: bool = True, extra_steps: int = 250, rest_window: int = 100,
                   drift_tol_mm: float = 5.0, ids: list[int] | None = None, extra_passes: int = 0,
                   reserved_slice: bool = False, keep_raw_unmoved: bool = False) -> dict:
    from dojo.simlab.runner import RoboDojoRunner, jsonable
    files = select_files(task, ids, reserved_slice, keep_raw_unmoved)
    run = RoboDojoRunner(task, physics_only=physics_only, need_planner=False,
                         workdir=str(paths.DATA / "simlab" / "work" / f"presettle_{task}"))
    index_p = paths.LAYOUTS / task / "settled_index.json"
    index = json.loads(index_p.read_text()) if index_p.exists() else {}
    for f in files:
        t0 = time.perf_counter()
        lay = json.loads(f.read_text())
        rec: dict = {"from": f.name}
        if keep_raw_unmoved:
            rec.update({"presettle": KEEP_RAW_UNMOVED, "previous": index.get(f.stem)})
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
        keep = unmoved(live) if keep_raw_unmoved else None
        if not ok:
            rec.update({"status": "dropped: not at rest", "why": why})
            if not keep_raw_unmoved:
                out.unlink(missing_ok=True)
        else:
            settled, n = write_back(lay, live, keep)
            if keep is not None:
                rec["kept_raw"] = sorted(o["label"] or o["inst_name"] for o in live if o["inst_name"] in keep)
            for k in range(max(1, extra_passes + 1)):   # pass 2 (+ re-settle passes for marginal rests)
                info2 = run.load(settled)
                live2 = run.objects_gt()
                top2: list = []
                d2 = drift(live2, top2) if info2["stable"] else (float("nan"), float("nan"))
                if not info2["stable"]:
                    drift(live2, top2)
                if not info2["stable"] or d2[0] <= drift_tol_mm or k == extra_passes:
                    break
                settled, n = write_back(settled, live2, unmoved(live2) if keep_raw_unmoved else None)   # leaning bread slices re-settle ~7 mm once
            rec["pass2_top"] = top2
            rec.update({"objects_written": n, "pass2_robodojo_stable": bool(info2["stable"]),
                        "pass2_drift_mm": round(d2[0], 3), "pass2_drift_deg": round(d2[1], 3), "passes": 2 + k})
            if info2["stable"] and d2[0] <= drift_tol_mm:
                meta = dict(settled.get("_simlab") or {})
                meta["settled"] = {"settler": SETTLER, "from": f.name, "generator": meta.get("generator"),
                                   "physics": "isaac-5.1 physics-only" if physics_only else "isaac-5.1",
                                   "extra_steps": extra_steps, **{k: rec[k] for k in (
                                       "pass1_robodojo_stable", "pass1_drift_mm", "pass2_drift_mm", "passes")}}
                if keep_raw_unmoved:
                    meta["presettle"] = KEEP_RAW_UNMOVED
                    meta["settled"].update({"mode": KEEP_RAW_UNMOVED, "kept_raw": rec["kept_raw"],
                                            "unmoved_tol": [UNMOVED_MM, UNMOVED_DEG]})
                    if out.exists():                    # never rewrite an existing settled record
                        raise FileExistsError(out)
                settled["_simlab"] = meta
                out.write_text(json.dumps(jsonable(settled), indent=1))
                rec["status"] = "ok"
            else:
                rec["status"] = "dropped: settled record not stable"
                if not keep_raw_unmoved:
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


def raw_check_task(task: str, physics_only: bool = True, extra_steps: int = 250, rest_window: int = 100,
                   ids: list[int] | None = None, reserved_slice: bool = False) -> dict:
    """``--raw-check``: judge each raw record as stored (it is what a ``settled: false`` task runs); writes only
    ``raw_index.json`` {id: {"status": "raw ok" | "dropped: ...", ...}}, never a record."""
    from dojo.simlab.runner import RoboDojoRunner
    files = select_files(task, ids, reserved_slice)
    run = RoboDojoRunner(task, physics_only=physics_only, need_planner=False,
                         workdir=str(paths.DATA / "simlab" / "work" / f"rawcheck_{task}"))
    index_p = paths.LAYOUTS / task / "raw_index.json"
    index = json.loads(index_p.read_text()) if index_p.exists() else {}
    for f in files:
        t0 = time.perf_counter()
        lay = json.loads(f.read_text())
        rec: dict = {"from": f.name, "check": RAW_CHECK, "extra_steps": extra_steps, "rest_window": rest_window}
        try:
            info = run.load(lay)
            rec["robodojo_stable"] = bool(info["stable"])
            if not info["stable"]:
                rec["status"] = "dropped: RoboDojo reset not stable"
            else:
                for _ in range(extra_steps - rest_window):
                    run.env.sim_step(render=False)
                before = run.objects_gt()
                for _ in range(rest_window):
                    run.env.sim_step(render=False)
                live = run.objects_gt()
                ok, why, belt = raw_at_rest(before, live, "Table" in lay)
                rec.update({"belt_mm_per_window": round(float(np.linalg.norm(belt)) * 1000, 2), "why": why,
                            "status": RAW_OK if ok else "dropped: raw record not at rest"})
        except Exception as e:  # noqa: BLE001 (any failure of a raw record's run drops it; the type is recorded)
            rec["status"] = f"dropped: exception {type(e).__name__}"
        rec["s"] = round(time.perf_counter() - t0, 1)
        index[f.stem] = rec
        index_p.write_text(json.dumps(index, indent=1))
        print(f"[raw-check] {task} {f.stem}: {rec['status']} ({rec['s']}s)", flush=True)
    run.close()
    print(f"[raw-check] {task}: {sum(1 for v in index.values() if v['status'] == RAW_OK)}/{len(index)} raw ok", flush=True)
    return index


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--extra-steps", type=int, default=250)
    ap.add_argument("--rendered", action="store_true", help="use the full (rendering) env, e.g. on compute2")
    ap.add_argument("--extra-passes", type=int, default=0,
                    help="re-settle passes: write the pass-k poses back and reload again (marginal rests)")
    ap.add_argument("--reserved-slice", action="store_true",
                    help="also settle ids >= paths.RESERVED_MIN_ID (D-C10: only the agreed joint-run generation)")
    ap.add_argument("--keep-raw-unmoved", action="store_true",
                    help="re-try dropped ids without a settled file: objects that did not move in pass 1 keep their "
                         "generated pose (records marked _simlab.presettle=keep_raw_unmoved)")
    ap.add_argument("--raw-check", action="store_true",
                    help="judge each RAW record as stored (settled: false tasks): raw_index.json only, no record "
                         "written (RoboDojo stable + extra steps + belt-subtracted at rest + fall floor)")
    a = ap.parse_args(argv)
    if a.raw_check:
        if a.keep_raw_unmoved or a.extra_passes:
            raise SystemExit("presettle: --raw-check excludes --keep-raw-unmoved / --extra-passes")
        raw_check_task(a.task, physics_only=not a.rendered, extra_steps=a.extra_steps, ids=a.ids,
                       reserved_slice=a.reserved_slice)
        return
    presettle_task(a.task, physics_only=not a.rendered, extra_steps=a.extra_steps, ids=a.ids,
                   extra_passes=a.extra_passes, reserved_slice=a.reserved_slice, keep_raw_unmoved=a.keep_raw_unmoved)


if __name__ == "__main__":
    main()
