"""Truth for runtime held_look reads (dev layouts only): was the object really in the jaws when held_look said "not held"?

1. Collect the negatives from integ run reports (event kind "held_look", held False) into EVENTS.json:
       python3 scripts/dojo/held_truth.py collect RUNS_DIR 'B_*_breadth2' 'S79_*' > EVENTS.json
2. Replay each episode's executed actions physics-only on its layout (proxy physics = the twin's; one light slot):
       scripts/simlab/rdpy.sh -u scripts/dojo/replay_kits.py run --task T --layout L --actions A --out OUT/<run>_<id>.json \
           --physics-only
3. Judge: python3 scripts/dojo/held_truth.py judge EVENTS.json OUT
   Held = the flange rose > 20 mm since the jaws closed, the object rose with it (> half) and kept its offset to the
   flange (< 20 mm drift). 2026-10-27 (journal dojo-integ3 §4): 52 of 52 negatives were empty jaws.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys

import numpy as np


def collect(runs: str, patterns: list[str]) -> list[dict]:
    out = []
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(runs, pat, "*_report.json.gz"))):
            rep = json.load(gzip.open(f))
            d = os.path.basename(os.path.dirname(f))
            rows = [json.loads(ln) for jf in glob.glob(os.path.join(os.path.dirname(f), "*_episodes.jsonl")) for ln in open(jf)]
            lid = int(f.split("_")[-2])
            row = next((r for r in rows if r["layout_id"] == lid), {})
            for e in rep.get("events", []):
                if e.get("kind") == "held_look" and e.get("held") is False:
                    op = e.get("opening") or {}
                    out.append({"run": d, "layout_id": lid, "task": row.get("task"), "layout": row.get("layout"),
                                "actions": f.replace("_report.json.gz", "_actions.json.gz"), "step": e["step"],
                                "label": e["label"], "arm": e["arm"], "conf": e.get("confidence"), "evidence": e.get("evidence"),
                                "score": op.get("score"), "confirm": (e.get("confirm") or {}).get("held")})
    return out


def judge(events: list[dict], replays: str) -> list[tuple]:
    res = []
    for e in events:
        f = os.path.join(replays, f"{e['run']}_{e['layout_id']}.json")
        if not os.path.exists(f):
            continue
        fr = json.load(open(f))["frames"]
        s, arm, lab = min(e["step"], len(fr) - 1), e["arm"], e["label"]
        c = next((k for k in range(s, 0, -1) if fr[k]["grip_cmd"][arm] < 0.5
                  and fr[k - 1]["grip_cmd"][arm] > fr[k]["grip_cmd"][arm] + 1e-6), None)
        if c is None or lab not in fr[s]["objects"] or lab not in fr[c]["objects"]:
            res.append((e, None, "no close / label"))
            continue
        rel = lambda k: np.asarray(fr[k]["objects"][lab][:3]) - np.asarray(fr[k]["flange"][arm][:3])  # noqa: E731
        dfz = fr[s]["flange"][arm][2] - fr[c]["flange"][arm][2]
        doz = fr[s]["objects"][lab][2] - fr[c]["objects"][lab][2]
        drift = float(np.linalg.norm(rel(s) - rel(c)))
        held = dfz > 0.02 and doz > 0.5 * dfz and drift < 0.02
        res.append((e, held, f"flange +{dfz * 1000:.0f} mm, object +{doz * 1000:.0f} mm, drift {drift * 1000:.0f} mm"))
    return res


def main(argv=None) -> None:
    a = sys.argv[1:] if argv is None else argv
    if a[0] == "collect":
        print(json.dumps(collect(a[1], a[2:])))
        return
    res = judge(json.load(open(a[1])), a[2])
    for e, held, why in res:
        print(f"{e['run'][:28]:28s} {e['layout_id']} s{e['step']:<4d} {e['label']:14s} {e.get('evidence')} conf {e.get('conf')} "
              f"confirm {e.get('confirm')} | held={held} ({why})")
    known = [(e, h) for e, h, _ in res if h is not None]
    print(f"{len(known)} judged: {sum(not h for _, h in known)} empty jaws, {sum(bool(h) for _, h in known)} held")


if __name__ == "__main__":
    main()
