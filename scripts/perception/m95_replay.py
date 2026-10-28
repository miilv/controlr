"""GT at every grasp / release of saved integ episodes: replay each episode's executed actions physics-only on its dev
layout (one RoboDojoRunner per task, many episodes per process) and snapshot every labelled object's pose at the steps
the grasp-time metric needs (``dojo.perception.grasp_metric``, stream M, PERCEPTION_95.md).

    scripts/simlab/rdpy.sh -u scripts/perception/m95_replay.py --task T --jobs JOBS.jsonl --out-dir DIR [--every 25]

``JOBS.jsonl``: one ``{"name", "task", "layout", "actions"}`` per line (other tasks' lines are skipped). Per episode
``DIR/<name>.json.gz``: ``frames`` = {step: {"objects": [GT rows], "flange": {arm: pose7}}} for step 0, every
``--every`` steps, the state before and after every action that changes a gripper command (so every grasp / release
of ``grasp_metric.grip_events``, at its step k - 1) and the end; ``reward`` at the end (compare with the episode row: the replay reproduces a proxy episode bit for bit; a
genuine-5.1 rendered episode only approximately, see replay_kits.py). Episodes already in DIR are skipped. Dev layouts
only (the runner's asset firewall).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

ARMS = ("left", "right")


def snap(run) -> dict:
    rows = []
    for o in run.objects_gt(0):
        if o.get("pos") is None or not (o.get("label") or o.get("clutter")):
            continue
        r = {k: o.get(k) for k in ("label", "type", "category", "category_idx", "clutter", "pos", "quat")}
        r["label"] = r["label"] or f"clutter:{o.get('inst_name')}"     # a distractor: a neighbour, never a target
        rows.append(r)
    s = run.obs_state(0)
    return {"objects": rows,
            "flange": {arm: [float(v) for v in np.asarray(s[f"{arm}_ee_pose"]).reshape(-1)[:7]] for arm in ARMS},
            "grip": {arm: float(np.asarray(s[f"{arm}_ee_joint_state"]).reshape(-1)[0]) for arm in ARMS}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--full", action="store_true", help="replay every action (else stop after the last gripper change)")
    a = ap.parse_args(argv)
    from dojo.perception.grasp_metric import grip_events
    from dojo.simlab import paths
    from dojo.simlab.runner import RoboDojoRunner

    out = Path(a.out_dir)
    paths.assert_not_eval(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(ln) for ln in open(a.jobs) if ln.strip()]
    jobs = [j for j in jobs if j["task"] == a.task and not (out / f"{j['name']}.json.gz").exists()]
    if not jobs:
        print(f"[m95] {a.task}: nothing to do", flush=True)
        return
    run = RoboDojoRunner(a.task, workdir=str(out / f"work_{a.task}"), physics_only=True, need_planner=False)
    for j in jobs:
        t0 = time.perf_counter()
        paths.assert_not_eval(j["layout"])
        lid = int(Path(j["layout"]).name.split(".")[0])
        if paths.is_offslice_rendered(str(lid)):
            print(f"[m95] {j['name']}: off-slice layout {lid}, skipped", flush=True)
            continue
        with gzip.open(j["actions"], "rt") as f:
            acts = json.load(f)["actions"]
        need = {e["step"] - 1 for e in grip_events(acts)}
        prev = {}
        for k, act in enumerate(acts, 1):           # every gripper change: any later grasp rule finds its frame
            g = {arm: float(np.asarray(act.get(f"{arm}_ee_joint_state", [1.0]), float).reshape(-1)[0]) for arm in ARMS}
            if prev and any(abs(g[x] - prev[x]) > 1e-6 for x in ARMS):
                need |= {k - 1, k}
            prev = g
        # nothing after the last gripper change is scored: stop there (a loaded box replays ~2 steps/s); the final
        # reward then is the one at the stop (``stopped_at``): compare it with the episode's only when it ran out
        stop = len(acts) if a.full else min(len(acts), max(need | {0}) + 2)
        want = {0, stop} | need | set(range(0, stop + 1, max(1, a.every)))
        info = run.load(json.loads(Path(j["layout"]).read_text()))
        frames = {}
        if 0 in want:
            frames[0] = snap(run)
        ended_at = None
        for k, act in enumerate(acts, 1):
            if run.step({kk: np.asarray(v, np.float32) for kk, v in act.items()})[0] and ended_at is None:
                ended_at = k
            if k in want or ended_at is not None:
                frames[k] = snap(run)
            if ended_at is not None or k >= stop:
                break
        res = {"name": j["name"], "task": a.task, "layout": j["layout"], "layout_id": lid, "actions": j["actions"],
               "n_actions": len(acts), "stopped_at": min(stop, ended_at or stop), "stable": info["stable"], "ended_at": ended_at,
               "reward": run.reward(0),
               "score": run.score(0), "replay_s": round(time.perf_counter() - t0, 1), "frames": frames}
        with gzip.open(out / f"{j['name']}.json.gz", "wt") as f:
            json.dump(res, f)
        print(f"[m95] {j['name']}: {len(acts)} actions, {len(frames)} frames, ended_at {ended_at}, "
              f"score {res['score']:.2f}, {res['replay_s']} s", flush=True)
    sys.stdout.flush()
    try:
        run.close()
    finally:
        import os
        os._exit(0)


if __name__ == "__main__":
    main()
