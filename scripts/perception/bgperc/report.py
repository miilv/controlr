"""Stream C: aggregate replay_eval outputs (one JSON per sequence) into per-task tables.
    python scripts/perception/bgperc/report.py EVAL_DIR [--base rt] [--md]"""
import argparse
import glob
import json
import os
from collections import defaultdict

RS = ("5", "10", "15")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--base", default="rt")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--lag", default=None, help="background arms: score at this extra lag (replay_eval --extra-lags)")
    a = ap.parse_args()
    per_task = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    pairs = defaultdict(lambda: defaultdict(lambda: [0, 0]))        # task -> arm -> [fixed, broken] vs base at R10
    looks = defaultdict(lambda: [0, 0, 0, 0])                      # arm -> [fires on unclean, fires on clean, unclean, clean]
    looks_d = defaultdict(lambda: [0, 0, 0, 0])                    # the same, "doubt" requests only (absence evidence)
    cost = defaultdict(lambda: defaultdict(float))
    reasons = defaultdict(lambda: defaultdict(int))
    n_ep = defaultdict(int)
    for f in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        d = json.load(open(f))
        task = d["task"]
        n_ep[task] += 1
        base = {(e["kind"], e["step"]): e for e in d["arms"].get(a.base, {}).get("events", [])}
        for arm, r in d["arms"].items():
            t = per_task[task][arm]
            for e in r["events"]:
                if e["kind"] not in ("grasp", "place"):
                    continue
                if a.lag is not None and a.lag in (e.get("by_lag") or {}):
                    e = {**e, **e["by_lag"][a.lag]}
                t["n"] += 1
                for R in RS:
                    t[f"R{R}"] += int(bool(e["clean_r"][R]))
                t["target"] += int(e["clean_target"])
                t["required"] += int(e["clean_required"])
                for x in e["reasons"]["10"]:
                    reasons[arm][x.split(" ")[0] + " " + (x.split(" ")[2] if x.startswith(("target", "held", "reference")) else
                                                          ("not found" if "not found" in x else "off" if x.startswith("neighbour") else ""))] += 1
                b = base.get((e["kind"], e["step"]))
                if b is not None and arm != a.base:
                    if e["clean_r"]["10"] and not b["clean_r"]["10"]:
                        pairs[task][arm][0] += 1
                    if b["clean_r"]["10"] and not e["clean_r"]["10"]:
                        pairs[task][arm][1] += 1
                if "look_requests" in e:
                    for lk, fired in ((looks[arm], bool(e["look_requests"])),
                                      (looks_d[arm], any(x.get("why") == "doubt" for x in e["look_requests"]))):
                        if e["clean_r"]["10"]:
                            lk[3] += 1
                            lk[1] += int(fired)
                        else:
                            lk[2] += 1
                            lk[0] += int(fired)
            st = r.get("stats") or {}
            if st:
                c = cost[arm]
                c["steps"] += st.get("steps", 0)
                c["refines"] += st.get("refines", 0)
                c["refine_s"] += st.get("refine_s", 0.0)
                c["vis_s"] += st.get("vis_s", 0.0)
                c["moves"] += st.get("moves", 0)
                c["confirms"] += st.get("confirms", 0)
                c["absent"] += st.get("absent", 0)
                c["searches"] += st.get("searches", 0)
                c["rereads"] += st.get("rereads", 0)
                c["reread_s"] += st.get("reread_s", 0.0)
    arms = sorted({arm for t in per_task.values() for arm in t})
    tot = defaultdict(lambda: defaultdict(int))
    lines = ["| task | episodes | arm | events | clean R5 | clean R10 | clean R15 | target ok | required ok | fixed / broken vs "
             f"{a.base} (R10) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for task in sorted(per_task):
        for arm in arms:
            t = per_task[task].get(arm)
            if not t:
                continue
            for k, v in t.items():
                tot[arm][k] += v
            fb = pairs[task].get(arm)
            lines.append(f"| {task} | {n_ep[task]} | {arm} | {t['n']} | {t['R5']} | {t['R10']} | {t['R15']} | {t['target']} | "
                         f"{t['required']} | {'' if arm == a.base else f'{fb[0]} / {fb[1]}' if fb else '0 / 0'} |")
    for arm in arms:
        t = tot[arm]
        n = max(1, t["n"])
        fb = [sum(pairs[k].get(arm, [0, 0])[0] for k in pairs), sum(pairs[k].get(arm, [0, 0])[1] for k in pairs)]
        lines.append(f"| **all** | {sum(n_ep.values())} | {arm} | {t['n']} | {t['R5']} ({t['R5'] / n:.0%}) | {t['R10']} "
                     f"({t['R10'] / n:.0%}) | {t['R15']} ({t['R15'] / n:.0%}) | {t['target']} | {t['required']} | "
                     f"{'' if arm == a.base else f'{fb[0]} / {fb[1]}'} |")
    out = "\n".join(lines)
    out += "\n\nlook trigger (needs_look on a required object at the event), per arm: fires on unclean / unclean events, " \
           "fires on clean / clean events\n"
    for arm, lk in looks.items():
        d = looks_d[arm]
        out += (f"- {arm}: any request {lk[0]}/{lk[2]} unclean, {lk[1]}/{lk[3]} clean; doubt only {d[0]}/{d[2]} unclean, "
                f"{d[1]}/{d[3]} clean\n")
    out += "\ncost (background arms): per step = refine + visibility CPU seconds / replayed steps\n"
    for arm, c in cost.items():
        s = max(1.0, c["steps"])
        out += (f"- {arm}: {int(c['refines'])} refines ({c['refine_s'] / max(1, c['refines']):.2f} s each), "
                f"{int(c['rereads'])} re-reads ({c['reread_s'] / max(1, c['rereads']):.2f} s each), "
                f"{1000 * (c['refine_s'] + c['vis_s'] + c['reread_s']) / s:.0f} ms/step over {int(s)} steps; "
                f"moves {int(c['moves'])}, confirms {int(c['confirms'])}, absent {int(c['absent'])}, "
                f"searches {int(c['searches'])}\n")
    out += "\nR10 failure reasons per arm (count of reason strings)\n"
    for arm in arms:
        out += f"- {arm}: " + ", ".join(f"{k.strip()} {v}" for k, v in sorted(reasons[arm].items(), key=lambda x: -x[1])) + "\n"
    print(out)


if __name__ == "__main__":
    main()
