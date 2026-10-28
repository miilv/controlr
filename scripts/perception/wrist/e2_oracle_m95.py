"""E2 ceiling (stream W): what a perfect post-release read is worth on stream M's whole replay set (no frames needed:
GT only). For every release, the released object's belief is set to its GT pose a few steps after the release (the
replay's first frame >= k + ``--delay``, else the frame at the release) and kept until the report's next update of
that label; every grasp / place is scored with M's metric exactly as ``m95_score.py score`` does (held drift / gap).

    python scripts/perception/wrist/e2_oracle_m95.py --jobs ~/dojo-data/m95/jobs_all.jsonl \
        --replays ~/dojo-data/m95/replays --out OUT.jsonl [--delay 5] [--limit N]
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _m95():
    spec = importlib.util.spec_from_file_location("m95_score", os.path.join(HERE, "..", "m95_score.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--replays", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delay", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, HERE)
    from e2_release import patched_tracker

    from dojo.perception import grasp_metric as G
    from dojo.skills.kinematics import X5Kin

    M = _m95()
    kin = X5Kin()
    fp = G.Footprints()
    groups: dict = {}
    jobs = {}
    for ln in open(os.path.expanduser(a.jobs)):
        if ln.strip():
            j = json.loads(ln)
            jobs[j["name"]] = j
    fo = open(a.out, "w")
    n = 0
    for name in sorted(jobs):
        j = jobs[name]
        if not M.on_slice(j.get("layout_id") or name.rsplit("_", 1)[-1]):
            continue
        rpath = os.path.join(os.path.expanduser(a.replays), f"{name}.json.gz")
        rep_path = j["actions"].replace("_actions.json.gz", "_report.json.gz")
        if not (os.path.exists(rpath) and os.path.exists(rep_path) and os.path.exists(j["actions"])):
            continue
        rep = json.load(gzip.open(rep_path, "rt"))
        acts = json.load(gzip.open(j["actions"], "rt"))["actions"]
        frames = json.load(gzip.open(rpath, "rt"))["frames"]
        task = j["task"]
        g = groups.setdefault(task, G.Groups(task))
        track = G.tcp_track(acts, kin)
        steps = sorted(int(s) for s in frames)

        def gt_after(k):
            s = next((s for s in steps if k + a.delay <= s <= k + 40), None)
            s = s if s is not None else next((s for s in steps if s >= k), None)
            return (s, frames[str(s)]["objects"]) if s is not None else (None, None)

        rel = [(int(e.step), e.target) for e, _ in G.episode_events(rep, acts, kin) if e.kind == "place" and e.target]
        res = {}
        for arm_name in ("rt", "oracle"):
            patches = []
            if arm_name == "oracle":
                for k, lab in rel:
                    s, objs = gt_after(k)
                    if s is not None:
                        patches.append((s, lab, objs))
            done = set()

            def patcher(bt, upto, patches=patches, done=done, g=g):
                for i, (s, lab, objs) in enumerate(patches):
                    if i in done or s > upto or lab not in bt.belief:
                        continue
                    done.add(i)
                    o = bt.snapshot(s).get(lab) or bt.belief[lab]
                    c = fp.centre(o.get("key"), o["pos"], o["quat"])
                    same = [x for x in G.gt_rows_asset(objs) if g.of(x["label"]) == g.of(lab)]
                    if not same:
                        continue
                    x = min(same, key=lambda x: float(np.linalg.norm(fp.centre(G.gt_key(x), x["pos"], x["quat"])[:2]
                                                                     - c[:2])))
                    bt.belief[lab].update(pos=list(x["pos"]), quat=list(x["quat"]), source="release_gt")

            out = {}
            last_grasp: dict = {}
            with patched_tracker(G, patcher):
                evs = G.episode_events(rep, acts, kin)
            for ev, belief in evs:
                if ev.kind not in ("grasp", "place"):
                    continue
                f = M.frame_at(frames, ev.step - 1)
                if f is None:
                    continue
                if ev.kind == "place" and ev.arm in last_grasp:
                    dr = M.held_drift(frames, track, ev.arm, *last_grasp[ev.arm], ev.step)
                    if dr is not None:
                        ev.meta["held_drift_mm"] = round(dr, 1)
                    gap = M.held_gap(f, track, ev.arm, last_grasp[ev.arm][1], ev.step, fp)
                    if gap is not None:
                        ev.meta["held_gap_mm"] = round(gap, 1)
                s = G.score_event(ev, belief, f["objects"], task, footprints=fp, groups=g)
                if ev.kind == "grasp" and s["required"][0].get("gt_label"):
                    last_grasp[ev.arm] = (ev.step, s["required"][0]["gt_label"])
                out[(int(ev.step), ev.arm)] = {"kind": ev.kind, "clean_r": s["clean_r"], "reasons": s["reasons"]["10"],
                                               "clean_required": s["clean_required"]}
            res[arm_name] = out
        keys = sorted(set(res["rt"]) | set(res["oracle"]))
        for key in keys:
            row = {"name": name, "task": task, "step": key[0], "arm": key[1],
                   "rt": res["rt"].get(key), "oracle": res["oracle"].get(key)}
            fo.write(json.dumps(row) + "\n")
        c_rt = sum(v["clean_r"]["10"] for v in res["rt"].values())
        c_or = sum(v["clean_r"]["10"] for v in res["oracle"].values())
        print(f"{name}: releases {len(rel)} events rt {len(res['rt'])} oracle {len(res['oracle'])} "
              f"clean@10 {c_rt} -> {c_or}", flush=True)
        n += 1
        if a.limit and n >= a.limit:
            break
    fo.close()
    summarize(a.out)


def summarize(path: str) -> None:
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    both = [r for r in rows if r["rt"] and r["oracle"]]
    by = defaultdict(lambda: Counter())
    for r in both:
        t = r["task"]
        for R in ("5", "10", "15"):
            a, b = r["rt"]["clean_r"][R], r["oracle"]["clean_r"][R]
            by[t][f"rt{R}"] += a
            by[t][f"or{R}"] += b
            by[t][f"fix{R}"] += b and not a
            by[t][f"brk{R}"] += a and not b
        by[t]["n"] += 1
    print(f"events scored in both arms: {len(both)} of {len(rows)} (attribution changed: {len(rows) - len(both)})")
    print("| task | events | rt R10 | oracle R10 | fixed / broken |")
    print("|---|---|---|---|---|")
    tot = Counter()
    for t in sorted(by, key=lambda t: -(by[t]["fix10"])):
        c = by[t]
        tot.update(c)
        print(f"| {t} | {c['n']} | {c['rt10']} | {c['or10']} | +{c['fix10']} / -{c['brk10']} |")
    print(f"| all | {tot['n']} | {tot['rt10']} | {tot['or10']} | +{tot['fix10']} / -{tot['brk10']} |")
    for R in ("5", "15"):
        print(f"R{R}: rt {tot['rt' + R]} oracle {tot['or' + R]} (+{tot['fix' + R]} / -{tot['brk' + R]})")


if __name__ == "__main__":
    main()
