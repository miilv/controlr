"""Dataset D0 for perception: our layouts, initial obs + obs after 3 scripted arm moves, with GT.

Per task, per layout id (seed >= 100000):

    <out>/<task>/<id>/layout.json                 the record loaded (RoboDojo schema + _simlab)
    <out>/<task>/<id>/frame_<k>_<cam>.png         k = 0..3, cam in cam_head/cam_left_wrist/cam_right_wrist
    <out>/<task>/<id>/frame_<k>.json              {"obs": official contract minus images,
                                                   "gt": DEV-ONLY ground truth, "move": what moved}

Moves (both grippers stay open): 1. left arm tilted ~50 deg down, wrist camera aimed at a task
object from ~14 cm; 2. the same with the right arm and another object; 3. both TCPs over the table
centre at 25 cm, tilted 25 deg (arms occlude the head view). A move the planner cannot reach is
retried with less tilt, then skipped (``move.ok`` false: the frame then equals the previous one).

    scripts/simlab/rdpy.sh -m dojo.simlab.make_d0 --task stack_blocks --n 25
    scripts/simlab/d0_ext.sh 10 make_kong ...     (pre-settled records, frames 0-1 -> ~/dojo-data/d0_ext)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from dojo.simlab import paths
from dojo.simlab.gen_layouts import LayoutError, generate
from dojo.simlab.layout import TABLE_TOP_Z
from dojo.simlab.runner import RoboDojoRunner, jsonable, save_png, settle_report

D0 = paths.DATA / "d0"


def obs_record(o: dict, frame: int) -> dict:
    return {"instruction": o["instruction"], "state": {k: v.tolist() for k, v in o["state"].items()},
            "images": {cam: f"frame_{frame}_{cam}.png" for cam in o["images"]},
            "image_shape": {cam: list(img.shape) for cam, img in o["images"].items()}}


def save_frame(run: RoboDojoRunner, d: Path, k: int, move: dict) -> None:
    o = run.obs()
    for cam, img in o["images"].items():
        save_png(img, d / f"frame_{k}_{cam}.png")
    ob = obs_record(o, k)
    if run.physics_only:        # images come later from an external renderer (render_d0_proxy.py)
        ob["images"] = {cam: f"frame_{k}_{cam}.png" for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
        ob["image_shape"] = {}
    rec = {"frame": k, "obs": ob, "gt": run.gt(), "move": move,
           "meta": {"renderer": None if run.physics_only else "isaac-5.1 (RoboDojo native)",
                    "physics": "isaac-5.1 physics-only" if run.physics_only else "isaac-5.1"}}
    (d / f"frame_{k}.json").write_text(json.dumps(jsonable(rec)))


def task_targets(gt_objects: list[dict]) -> list[dict]:
    return [o for o in gt_objects if not o["clutter"] and o["type"] in ("Rigid", "Dynamic", "Articulation")
            and o["pos"] is not None and o["pos"][2] > TABLE_TOP_Z - 0.05]


def tilted(home_quat, deg: float) -> np.ndarray:
    """Home flange orientation pitched about world x so the tool axis (flange +x) points down by
    ~``deg`` (sign chosen from the resulting axis)."""
    from dojo.simlab.motion import down_quat, quat_to_mat
    best = None
    for sgn in (1, -1):
        q = down_quat(home_quat, sgn * deg)
        z = (quat_to_mat(q) @ np.array([1.0, 0.0, 0.0]))[2]
        if best is None or z < best[0]:
            best = (z, q)
    return best[1]


def hover_move(mv, run: RoboDojoRunner, label: str, hover: float) -> dict:
    """Frame 1 for refine tests: a wrist ~``hover`` m above object ``label``, tool axis (near) straight down,
    whichever arm reaches (left first); falls back to a 50 deg look from 14 cm."""
    o = next((r for r in run.objects_gt() if r.get("label") == label and r["pos"] is not None), None)
    if o is None:
        return {"name": f"hover_{label}", "ok": False, "why": "label not in the scene"}
    for arm in ("left", "right"):
        for pitch in (85, 70, 55):
            q = tilted(mv.home_flange[arm][3:], pitch)
            tcp = np.asarray(o["pos"], float) + [0.0, 0.0, hover]
            if mv.tcp({arm: tcp}, quat={arm: q}, name=f"hover_{label}").get(arm):
                return {"name": f"hover_{label}", "arm": arm, "target_label": label, "target_inst": o["inst_name"],
                        "tcp_target": tcp.tolist(), "pitch_deg": pitch, "hover_m": hover, "ok": True}
    return {"name": f"hover_{label}", "ok": False, "why": "unreachable"}


def scripted_moves(run: RoboDojoRunner, d: Path, seed: int, max_frame: int = 3, look_label: str | None = None,
                   hover: float = 0.10) -> list[dict]:
    from dojo.simlab.motion import Mover, quat_to_mat
    rng = random.Random(seed)
    mv = Mover(run)
    objs = task_targets(run.objects_gt())
    log = []

    def look_at(arm, o, dist, pitch):
        q = tilted(mv.home_flange[arm][3:], pitch)
        axis = quat_to_mat(q) @ np.array([1.0, 0.0, 0.0])
        return np.asarray(o["pos"], float) - dist * axis, q

    def try_move(arm, cands, name):
        for c in cands:
            for pitch, dist in ((50, 0.14), (35, 0.18), (0, 0.0)):
                if pitch == 0:
                    tcp, q = np.asarray(c["pos"], float) + [0, 0, 0.15], mv.home_flange[arm][3:]
                else:
                    tcp, q = look_at(arm, c, dist, pitch)
                ok = mv.tcp({arm: tcp}, quat={arm: q}, name=name)
                if ok.get(arm):
                    return {"name": name, "arm": arm, "target_label": c["label"], "target_inst": c["inst_name"],
                            "tcp_target": tcp.tolist(), "pitch_deg": pitch, "ok": True}
        return {"name": name, "arm": arm, "ok": False}

    left_c = sorted(objs, key=lambda o: o["pos"][0])            # world -x is the robot's left
    right_c = sorted(objs, key=lambda o: -o["pos"][0])
    m1 = hover_move(mv, run, look_label, hover) if look_label else {"ok": False}
    if not m1["ok"]:
        m1 = {**try_move("left", left_c[:3], "left_look_at_object"),
              **({"hover_failed": look_label} if look_label else {})}
    log.append(m1)
    save_frame(run, d, 1, m1)
    if max_frame < 2:
        (d / "trajectory.json").write_text(json.dumps(mv.traj.to_json()))
        return log
    used = {m1.get("target_inst")}
    m2 = try_move("right", [o for o in right_c if o["inst_name"] not in used][:3] or right_c[:3],
                  "right_look_at_object")
    log.append(m2)
    save_frame(run, d, 2, m2)
    if max_frame < 3:
        (d / "trajectory.json").write_text(json.dumps(mv.traj.to_json()))
        return log
    cx, cy = rng.uniform(-0.05, 0.05), rng.uniform(-0.15, -0.05)
    tz = TABLE_TOP_Z + 0.25
    tgt = {"left": np.array([cx - 0.09, cy, tz]), "right": np.array([cx + 0.09, cy, tz])}
    quat = {a: tilted(mv.home_flange[a][3:], 25) for a in tgt}
    ok = mv.tcp(tgt, quat=quat, name="both_centre")
    if not all(ok.values()):
        ok = mv.tcp({a: t for a, t in tgt.items() if not ok[a]}, name="both_centre_retry") | {a: True for a in ok if ok[a]}
    m3 = {"name": "both_centre", "ok": ok, "tcp_targets": {a: t.tolist() for a, t in tgt.items()}}
    log.append(m3)
    save_frame(run, d, 3, m3)
    (d / "trajectory.json").write_text(json.dumps(mv.traj.to_json()))
    return log


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--start", type=int, default=paths.SEED_BASE)
    ap.add_argument("--out", default=str(D0))
    ap.add_argument("--physics-only", action="store_true",
                    help="no renderer: states + stage.usd + cameras.json for render_d0_proxy.py")
    ap.add_argument("--settled", action="store_true",
                    help="use the stored pre-settled records (<id>.settled.json, ids from --start) instead of "
                         "generating; LAYOUTS is not written")
    ap.add_argument("--max-frame", type=int, default=3, choices=(1, 2, 3), help="last frame (moves 1..K)")
    ap.add_argument("--index-name", default="index.json")
    ap.add_argument("--look-label", default=None, help="frame 1 = a wrist hovering over this object (refine tests)")
    ap.add_argument("--hover-cm", type=float, default=10.0)
    a = ap.parse_args(argv)
    out = Path(a.out) / a.task
    paths.assert_not_eval(out)
    out.mkdir(parents=True, exist_ok=True)
    lay_dir = paths.LAYOUTS / a.task
    lay_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    run = RoboDojoRunner(a.task, workdir=str(paths.DATA / "simlab" / "work" / f"d0_{a.task}"),
                         physics_only=a.physics_only)
    print(f"[d0] {a.task}: boot {time.perf_counter() - t0:.1f}s", flush=True)
    index, seed, made = [], a.start, 0
    while made < a.n and seed < a.start + 3 * a.n:
        s, seed = seed, seed + 1
        d = out / str(s)
        if (d / f"frame_{a.max_frame}.json").exists():
            made += 1
            index.append({"id": s, "status": "ok", "cached": True})
            continue
        if a.settled:
            sp = lay_dir / f"{s}.settled.json"
            if not sp.exists():
                index.append({"id": s, "status": "no settled record"})
                continue
            lay = json.loads(sp.read_text())
        else:
            try:
                lay = generate(a.task, s)
            except LayoutError as e:
                index.append({"id": s, "status": f"gen_fail: {e}"})
                continue
            (lay_dir / f"{s}.json").write_text(json.dumps(lay, indent=1))
        t1 = time.perf_counter()
        info = run.load(lay)
        if not info["stable"]:
            index.append({"id": s, "status": "unstable"})
            print(f"[d0] {a.task} {s}: unstable layout, skipped", flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        (d / "layout.json").write_text(json.dumps(lay, indent=1))
        settle = settle_report(lay, run.objects_gt())
        if a.physics_only:
            run.export_stage(d / "stage.usd")
            (d / "cameras.json").write_text(json.dumps(run.camera_spec(), indent=1))
        save_frame(run, d, 0, {"name": "initial"})
        moves = scripted_moves(run, d, s, a.max_frame, a.look_label, a.hover_cm / 100)
        made += 1
        index.append({"id": s, "status": "ok", "instruction": info["instruction"][0], "moves_ok":
                      [bool(m["ok"] if isinstance(m["ok"], bool) else all(m["ok"].values())) for m in moves],
                      "settle_max_task_drift_mm": settle["max_task_drift_mm"],
                      "load_s": round(info["reset_s"] + info["close_s"], 1),
                      "total_s": round(time.perf_counter() - t1, 1)})
        print(f"[d0] {a.task} {s}: ok moves={index[-1]['moves_ok']} load={index[-1]['load_s']}s "
              f"total={index[-1]['total_s']}s", flush=True)
        (out / a.index_name).write_text(json.dumps(index, indent=1))
    (out / a.index_name).write_text(json.dumps(index, indent=1))
    print(f"[d0] {a.task}: {made} layouts in {time.perf_counter() - t0:.0f}s", flush=True)
    run.close()


if __name__ == "__main__":
    main()
