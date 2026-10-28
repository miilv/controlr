"""Stream H: what RoboDojo's official gripper reading says about the jaws at every grasp / place (PERCEPTION_95.md).

RoboDojo's ``{arm}_ee_joint_state`` (strict server v2: ``ee_joint_state_official``) is the APPLIED gripper control: the
measured finger value at most 0.2 of the range past the target. Closed on nothing it reads the command (0.0); closed on
an object w wide it reads about ``(w / 2 + 0.01) / 0.054 - 0.2`` (``dojo.runtime.belief.expected_official``). Stream M's
physics-only replays (``scripts/perception/m95_replay.py``) record it per frame (``grip``), so every saved integ episode
can be re-judged offline, with no LLM and no GPU:

    python3 scripts/perception/h95_grip_offline.py --events SCORE/events.jsonl --replays REPLAYS --runs RUNS_ROOT \
        [--eps 0.001] [--json out.json]

``--events``: stream M's scored events (``m95_score.py score``); ``--replays``: its replays (``<run>_<layout>.json.gz``);
``--runs``: a dir with ``<run>/<task>_*_<layout>_report.json.gz`` and ``_actions.json.gz`` (the integ run dirs).

Per grasp (a close followed by a place / release of the same arm): the first replay frame at least ``--settle`` steps
after the close and before the release is the check point (the runtime checks at its next refresh, ``verify_pick``
after the lift). Truth: the GT object follows the flange (its position in the flange frame moved < ``--follow-mm`` since
the close) once the flange has moved >= 30 mm. Tables:

1. the check (official - command <= eps) against the truth: empty / held, per task;
2. M's place events whose held object is ``not_held`` at the release, by pattern: ``air`` (empty at the check and at the
   release: closed on nothing), ``slip`` (held at the check, empty at the release), ``on`` (the jaws on something at the
   release: another object), and by what the run's vision held check (``held_look``) said;
3. the same per run family (code generation), so old rows are not read as today's rate.

DEV-ONLY: GT is read here, never by a policy.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os

import numpy as np


def _qR(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / max(float(np.linalg.norm(q)), 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def family(run: str) -> str:
    return run.split("_")[0]


def load(path: str):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def episode_files(runs_root: str, run: str, task: str, lid) -> tuple[str | None, str | None]:
    rp = glob.glob(os.path.join(runs_root, run, f"{task}_*_{lid}_report.json.gz"))
    ap = glob.glob(os.path.join(runs_root, run, f"{task}_*_{lid}_actions.json.gz"))
    return (rp[0] if rp else None), (ap[0] if ap else None)


def rel_pos(frame: dict, arm: str, label: str):
    o = [r for r in frame["objects"] if r.get("label") == label]
    if not o:
        return None
    fl = frame["flange"][arm]
    return _qR(fl[3:7]).T @ (np.asarray(o[0]["pos"], float) - np.asarray(fl[:3], float)), np.asarray(fl[:3], float)


def grasps(events: list[dict], runs_root: str, replays: str, settle: int) -> list[dict]:
    """One row per (grasp, its place) of every scored episode."""
    by = collections.defaultdict(list)
    for e in events:
        by[(e["run"], e["layout_id"])].append(e)
    rows = []
    for (run, lid), evs in sorted(by.items()):
        task = evs[0]["task"]
        rpath = os.path.join(replays, f"{run}_{lid}.json.gz")
        rp, ap = episode_files(runs_root, run, task, lid)
        if not os.path.exists(rpath) or ap is None:
            continue
        frames = {int(k): v for k, v in load(rpath)["frames"].items()}
        acts = load(ap)["actions"]
        rep = load(rp) if rp else {}
        looks = [x for x in rep.get("events") or [] if x.get("kind") == "held_look"]
        evs = sorted(evs, key=lambda e: e["step"])
        for i, e in enumerate(evs):
            if e["kind"] != "place":
                continue
            arm = e["arm"]
            g = [x for x in evs[:i] if x["kind"] == "grasp" and x["arm"] == arm]
            if not g:
                continue
            k0 = g[-1]["step"]

            def cmd(k: int) -> float:
                return float(np.asarray(acts[k - 1][f"{arm}_ee_joint_state"], float).reshape(-1)[0])

            reads = [(k, float(frames[k]["grip"][arm]) - cmd(k)) for k in sorted(frames)
                     if k0 + settle <= k < e["step"] and 1 <= k <= len(acts)]
            gt_label = e["required"][0].get("gt_label") or e["target"]
            truth = None
            f0 = frames.get(k0) or frames.get(k0 - 1)
            a = rel_pos(f0, arm, gt_label) if f0 else None
            if a is not None and reads:
                b = rel_pos(frames[reads[0][0]], arm, gt_label)
                if b is not None and float(np.linalg.norm(b[1] - a[1])) >= 0.03:
                    truth = "held" if float(np.linalg.norm(b[0] - a[0])) < 0.025 else "empty"
            lk = [x for x in looks if x.get("arm") == arm and k0 <= int(x.get("step") or -1) <= e["step"]]
            rows.append({"task": task, "run": run, "family": family(run), "layout_id": lid, "arm": arm, "label": e["target"],
                         "grasp_step": k0, "release_step": e["step"], "cls": e["required"][0]["cls"],
                         "grasp_cls": g[-1]["required"][0]["cls"], "reads": reads, "truth": truth,
                         "look": None if not lk else {k: lk[0].get(k) for k in ("held", "confidence", "evidence", "kept",
                                                                                "dropped", "sure")}})
    return rows


def pattern(reads: list, eps: float) -> str:
    if not reads:
        return "no_read"
    early, late = reads[0][1] <= eps, reads[-1][1] <= eps
    return {(True, True): "air", (False, True): "slip", (True, False): "late_on", (False, False): "on"}[(early, late)]


def look_verdict(look: dict | None) -> str:
    if not look:
        return "no look"
    if look.get("held"):
        return "look: held"
    return "look: not held" + (" (kept)" if look.get("kept") else (" (dropped)" if look.get("dropped") else ""))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--replays", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--settle", type=int, default=6)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    events = [json.loads(ln) for ln in open(a.events) if ln.strip()]
    rows = grasps(events, a.runs, a.replays, a.settle)
    print(f"{len(rows)} grasp -> place pairs with a replay ({sum(1 for r in rows if r['reads'])} with a settled reading)")

    print("\n1. official empty (official - command <= %.4f at the first settled frame) vs physics truth" % a.eps)
    t = collections.Counter((r["truth"], r["reads"][0][1] <= a.eps) for r in rows if r["reads"] and r["truth"])
    print(f"   truth empty: check empty {t[('empty', True)]}, check held {t[('empty', False)]}")
    print(f"   truth held:  check empty {t[('held', True)]}, check held {t[('held', False)]}")
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["reads"] and r["truth"]:
            per[r["task"]][(r["truth"], r["reads"][0][1] <= a.eps)] += 1
    for task, c in sorted(per.items()):
        print(f"   {task:28s} empty->empty {c[('empty', True)]:3d}  empty->held {c[('empty', False)]:3d}  "
              f"held->empty {c[('held', True)]:3d}  held->held {c[('held', False)]:3d}")

    nh = [r for r in rows if r["cls"] == "not_held"]
    print(f"\n2. M's not_held places: {len(nh)}")
    for k, v in sorted(collections.Counter((pattern(r["reads"], a.eps), look_verdict(r["look"])) for r in nh).items()):
        print(f"   {v:3d} {k[0]:8s} {k[1]}")
    print("\n3. not_held / places per family (newest last: see PERCEPTION_95 ### M for the shas)")
    fam = collections.defaultdict(collections.Counter)
    for r in rows:
        fam[r["family"]]["places"] += 1
        if r["cls"] == "not_held":
            fam[r["family"]]["not_held"] += 1
            fam[r["family"]][pattern(r["reads"], a.eps)] += 1
    for f, c in fam.items():
        print(f"   {f:6s} {dict(c)}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f)


if __name__ == "__main__":
    main()
