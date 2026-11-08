"""Jobs for m95_replay with exact frames at every belief write step (and the step after); compute3 paths."""
import glob, gzip, json, os, sys
sys.path.insert(0, sys.argv[1] + "/scripts/perception")
from m95_score import belief_writes, episodes
RUNS, LAYROOT, OUTRUNS = sys.argv[2], sys.argv[3], sys.argv[4]
n = 0
for d, r, rp, ap in episodes([RUNS + "/c6b_*"]):
    rep = json.load(gzip.open(rp, "rt"))
    steps = sorted({int(w["step"]) for w in belief_writes(rep)} | {int(w["step"]) + 1 for w in belief_writes(rep)})
    name = f"{os.path.basename(d)}_{r['layout_id']}"
    job = {"name": name, "task": r["task"], "layout": f"{LAYROOT}/{r['task']}/{os.path.basename(r['layout'])}",
           "actions": f"{OUTRUNS}/{os.path.basename(d)}/{os.path.basename(ap)}", "snapshot_steps": steps}
    print(json.dumps(job)); n += 1
print(n, "jobs", file=sys.stderr)
