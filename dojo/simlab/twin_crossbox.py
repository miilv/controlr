"""Does the twin verifier's nominal outcome depend on the machine? Cross-box rollouts at the verifier's decision points.

The bundle verifier (``dojo.runtime.verifier.BundleVerifier``) loads a world from the belief (object poses + proprio,
``Twin.load_from_estimate``: 50 settle steps), rolls a candidate segment and judges the outcome (final poses, the task's
checker). If PhysX gives different outcomes on different machines, the verdicts fitted on our boxes may not hold on
the leaderboard's.

    # 1. decision points (one box): replay recorded G3 action lists, snapshot the world at the start of every place /
    #    release segment where an object is in the jaws
    rdpy.sh -m dojo.simlab.twin_crossbox points --task T --actions A1.json.gz ... --layouts-dir DIR --out points_T.jsonl
    # 2. the same points on every box (identical inputs): Twin.verify, sample 0 (nominal), one candidate each
    rdpy.sh -m dojo.simlab.twin_crossbox roll --points points_T.jsonl --out roll_T_<box>.jsonl
    # 3. offline: per point, per object, qualitative flips between two boxes
    python -m dojo.simlab.twin_crossbox compare roll_*_<boxA>.jsonl --b roll_*_<boxB>.jsonl

Points are independent decision points (each rolled from its own loaded world), not steps of one accumulated replay.
Flips (``flips``): the checker's state (success, stages / final checks left, score), and per task object: upright vs
toppled (tilt > 45 deg), fell off the table (z < 0.70 m), held in the jaws, moved by the segment (> 20 mm) vs at rest.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

ARMS = ("left", "right")
PRE, POST = 60, 40                      # segment = up to PRE steps before a release .. POST steps after it
TILT_DEG, FELL_Z, MOVED_MM, HELD_M = 45.0, 0.70, 20.0, 0.15   # held: raised > 1 cm, centre within 15 cm of the TCP (a bowl by its rim)


def _grip(a: dict, arm: str) -> float:
    return float(np.asarray(a[f"{arm}_ee_joint_state"], float).reshape(-1)[0])


def grip_events(actions: list[dict]) -> list[tuple[int, str, str]]:
    """(step, arm, "close" | "open") for every commanded gripper transition (half open)."""
    out = []
    for arm in ARMS:
        prev = _grip(actions[0], arm) if actions else 1.0
        for t, a in enumerate(actions):
            g = _grip(a, arm)
            if prev >= 0.5 > g:
                out.append((t, arm, "close"))
            elif prev < 0.5 <= g:
                out.append((t, arm, "open"))
            prev = g
    return sorted(out)


def segments(actions: list[dict]) -> list[dict]:
    """One segment per release: from max(release - PRE, previous gripper event + 1) to min(release + POST, next gripper
    event, end)."""
    ev = grip_events(actions)
    out = []
    for i, (t, arm, kind) in enumerate(ev):
        if kind != "open":
            continue
        prev = max((s for s, _a, _k in ev[:i]), default=-1)
        nxt = min((s for s, _a, _k in ev[i + 1:] if s > t), default=len(actions))
        t0, t1 = max(t - PRE, prev + 1, 0), min(t + POST, nxt, len(actions))
        if t1 - t0 >= 5:
            out.append({"t0": t0, "t_rel": t, "t1": t1, "arm": arm})
    return out


def tilt_deg(quat) -> float:
    """Angle between the object's z axis and world up (wxyz quaternion)."""
    w, x, y, z = (float(v) for v in quat)
    zz = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def _with_poses(layout: dict, rows: list[dict]) -> dict:
    """The layout record with every object's pose replaced by the live one (GT rows, the root frame RoboDojo poses)."""
    import copy

    from dojo.simlab.layout import iter_objects
    out = copy.deepcopy(layout)
    by_label = {r["label"]: r for r in rows if r.get("label")}
    for _t, _c, inst in iter_objects(out):
        r = by_label.get(inst.get("label"))
        if r is not None:
            inst["default_pos"] = [float(v) for v in r["pos"]]
            inst["default_ori"] = [float(v) for v in r["quat"]]
    return out


def cmd_points(a) -> None:
    from dojo.simlab import paths
    from dojo.simlab.motion import tcp_of
    from dojo.simlab.twin import Twin
    tw = Twin(a.task, num_envs=1, physics_only=True, need_planner=False,
              workdir=str(paths.DATA / "simlab" / "work" / f"crossbox_{a.task}"))
    out = open(a.out, "a")
    n_pts = 0
    for af in a.actions:
        lid = int(Path(af).name.split("_actions")[0].rsplit("_", 1)[-1])
        if paths.is_offslice_rendered(lid) or paths.is_calibration_id(lid):
            print(f"[xbox] skip {lid}: not ours to run", flush=True)
            continue
        lf = Path(a.layouts_dir) / f"{lid}.settled.json"
        if not lf.exists():
            print(f"[xbox] skip {lid}: no {lf}", flush=True)
            continue
        paths.assert_not_eval(lf)
        layout = json.loads(lf.read_text())
        with gzip.open(af, "rt") as f:
            acts = json.load(f)["actions"]
        segs = segments(acts)
        if not segs:
            continue
        t_start = time.perf_counter()
        tw.load([layout])
        z0 = {lab: o["pos"][2] for lab, o in tw._task_labels(0).items()}
        want = {s["t0"]: [] for s in segs}
        for s in segs:
            want[s["t0"]].append(s)
        rel = {s["t_rel"]: s for s in segs}
        for t, act in enumerate(acts):
            if t in want:
                rows = tw.run.objects_gt(0)
                st = tw.run.obs_state(0)
                robot = {f"{arm}_q": np.asarray(st[f"{arm}_arm_joint_state"], float).reshape(-1)[:6].tolist() for arm in ARMS}
                robot.update({f"{arm}_grip": float(np.asarray(st[f"{arm}_ee_joint_state"], float).reshape(-1)[0])
                              for arm in ARMS})
                for s in want[t]:
                    s["layout"], s["robot"] = _with_poses(layout, rows), robot
            if t in rel:                                # held at the release: a raised task object in that arm's jaws
                st = tw.run.obs_state(0)
                tcp = tcp_of(st[f"{rel[t]['arm']}_ee_pose"])
                rel[t]["held"] = sorted(lab for lab, o in tw._task_labels(0).items()
                                        if o["pos"][2] - z0.get(lab, o["pos"][2]) > 0.01
                                        and float(np.linalg.norm(np.asarray(o["pos"]) - tcp)) < HELD_M)
            if tw.step([{k: np.asarray(v, np.float32) for k, v in act.items()}])[0]:   # as G3 / verify feed it
                break
        for i, s in enumerate(segs):
            if "layout" not in s or not s.get("held"):
                continue                               # the episode ended first, or nothing was in the jaws
            rec = {"id": f"{a.task}/{lid}/{i}", "task": a.task, "layout_id": lid, "src": str(af),
                   "t0": s["t0"], "t_rel": s["t_rel"], "t1": s["t1"], "arm": s["arm"], "held": s["held"],
                   "layout": s["layout"], "robot": s["robot"], "actions": acts[s["t0"]:s["t1"]]}
            out.write(json.dumps(rec) + "\n")
            n_pts += 1
        out.flush()
        print(f"[xbox] {a.task} {lid}: {len(segs)} releases, {n_pts} points so far "
              f"({time.perf_counter() - t_start:.0f} s)", flush=True)
    out.close()
    tw.close()


def cmd_roll(a) -> None:
    import platform

    from dojo.simlab import paths
    from dojo.simlab.runner import jsonable
    from dojo.simlab.twin import Twin
    pts = [json.loads(ln) for ln in open(a.points)]
    if not pts:
        return
    task = pts[0]["task"]
    tw = Twin(task, num_envs=1, physics_only=True, need_planner=False,
              workdir=str(paths.DATA / "simlab" / "work" / f"crossbox_roll_{task}"))
    host = platform.node()
    with open(a.out, "w") as out:
        for p in sorted(pts, key=lambda p: (p["layout_id"], p["t0"])):  # same object set back to back: fast loads
            t0 = time.perf_counter()
            try:
                o = tw.verify({"layout": p["layout"], "robot": p["robot"]}, [{"actions": p["actions"]}], k=1)[0]
                res = {k: o.get(k) for k in ("final_objects", "reward", "score", "held", "fell", "dropped", "lifted",
                                             "moved_mm", "success")}
            except Exception as e:  # noqa: BLE001 (record and go on: the same point fails on every box or it is a flip)
                res = {"error": f"{type(e).__name__}: {e}"[:300]}
            out.write(json.dumps(jsonable({"id": p["id"], "host": host, "s": round(time.perf_counter() - t0, 2),
                                           "held_at_release": p["held"], **res})) + "\n")
            out.flush()
    tw.close()


def classify(a: dict, b: dict) -> dict:
    """Per point: the quantitative divergence and the qualitative flips between two boxes' outcomes."""
    if a.get("error") or b.get("error"):
        return {"flips": ["error"] if bool(a.get("error")) != bool(b.get("error")) else [], "error": True}
    flips = []
    ra, rb = a.get("reward") or {}, b.get("reward") or {}
    for k in ("success_flag", "ended", "stages_left", "final_left"):
        if ra.get(k) != rb.get(k):
            flips.append(f"checker:{k}")
    if abs(float(a.get("score") or 0.0) - float(b.get("score") or 0.0)) > 1e-6:
        flips.append("checker:score")
    fa, fb = a.get("final_objects") or {}, b.get("final_objects") or {}
    ma, mb = a.get("moved_mm") or {}, b.get("moved_mm") or {}
    dmax = 0.0
    for lab in sorted(set(fa) & set(fb)):
        pa, pb = np.asarray(fa[lab][:3], float), np.asarray(fb[lab][:3], float)
        dmax = max(dmax, float(np.linalg.norm(pa - pb)) * 1000)
        if (tilt_deg(fa[lab][3:7]) > TILT_DEG) != (tilt_deg(fb[lab][3:7]) > TILT_DEG):
            flips.append(f"topple:{lab}")
        if (pa[2] < FELL_Z) != (pb[2] < FELL_Z):
            flips.append(f"fell:{lab}")
        if (float(ma.get(lab, 0.0)) > MOVED_MM) != (float(mb.get(lab, 0.0)) > MOVED_MM):
            flips.append(f"moved:{lab}")
    if sorted(a.get("held") or []) != sorted(b.get("held") or []):
        flips.append("held")
    return {"flips": flips, "max_mm": round(dmax, 2)}


def cmd_compare(a) -> None:
    def load(files):
        out = {}
        for f in files:
            for ln in open(f):
                r = json.loads(ln)
                out[r["id"]] = r
        return out
    A, B = load(a.a), load(a.b)
    ids = sorted(set(A) & set(B))
    rows, per_task = [], {}
    for i in ids:
        c = classify(A[i], B[i])
        rows.append({"id": i, **c})
        t = per_task.setdefault(i.split("/")[0], {"n": 0, "flips": 0, "max_mm": []})
        t["n"] += 1
        t["flips"] += bool(c["flips"])
        if "max_mm" in c:
            t["max_mm"].append(c["max_mm"])
    n, k = len(rows), sum(1 for r in rows if r["flips"])
    from math import comb
    # exact one-sided 95% upper bound (Clopper-Pearson) on the flip rate
    def upper(k, n, conf=0.95):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            p = (lo + hi) / 2
            cdf = sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k + 1))
            lo, hi = (p, hi) if cdf > 1 - conf else (lo, p)
        return hi
    summ = {"n": n, "flipped": k, "rate": round(k / n, 4) if n else None, "upper95": round(upper(k, n), 4) if n else None,
            "per_task": {t: {"n": v["n"], "flipped": v["flips"],
                             "max_mm_median": round(float(np.median(v["max_mm"])), 2) if v["max_mm"] else None,
                             "max_mm_p90": round(float(np.percentile(v["max_mm"], 90)), 2) if v["max_mm"] else None,
                             "max_mm_max": round(float(np.max(v["max_mm"])), 2) if v["max_mm"] else None}
                         for t, v in sorted(per_task.items())},
            "flip_examples": [r for r in rows if r["flips"]][:20],
            "only_in_a": len(set(A) - set(B)), "only_in_b": len(set(B) - set(A))}
    print(json.dumps(summ, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("points")
    p.add_argument("--task", required=True)
    p.add_argument("--actions", nargs="+", required=True)
    p.add_argument("--layouts-dir", required=True)
    p.add_argument("--out", required=True)
    r = sub.add_parser("roll")
    r.add_argument("--points", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a", nargs="+")
    c.add_argument("--b", nargs="+", required=True)
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    {"points": cmd_points, "roll": cmd_roll, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
