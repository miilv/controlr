"""ORG2 registered panel report (docs/experiments/2026-11-07-org2-g3-prereg.md); reads G3 episode records only.

    python3 scripts/skills/org2_panel_report.py RUN_A RUN_B [--bar 85.0] [--json OUT]

Pairs by layout id, pooled means and sample SDs (points), paired deltas, score wins / losses (>= 10 points), success
flips, the registered decision, and B's 95 % bootstrap interval (10000 resamples, seed 0) against the bar."""
import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path


def rows(run):
    out = {}
    for line in (Path(run) / "organize_table_episodes.jsonl").read_text().splitlines():
        r = json.loads(line)
        out[int(r["seed"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--bar", type=float, default=85.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    A, B = rows(a.a), rows(a.b)
    ids = sorted(set(A) & set(B))
    ok = [i for i in ids if A[i].get("stable") and B[i].get("stable")
          and not A[i].get("infra_failure") and not B[i].get("infra_failure")]
    sa = [100 * A[i]["score"] for i in ok]
    sb = [100 * B[i]["score"] for i in ok]
    d = [y - x for x, y in zip(sa, sb)]
    wins = sum(x >= 10 for x in d)
    losses = sum(x <= -10 for x in d)
    flips_w = sum(1 for i in ok if B[i]["success"] and not A[i]["success"])
    flips_l = sum(1 for i in ok if A[i]["success"] and not B[i]["success"])
    rng = random.Random(0)
    boots = sorted(st.mean(rng.choice(sb) for _ in sb) for _ in range(10000))
    rep = {"n_pairs": len(ok), "missing": sorted(set(A) ^ set(B)), "excluded": [i for i in ids if i not in ok],
           "A_mean": round(st.mean(sa), 2), "A_sd": round(st.stdev(sa), 2) if len(sa) > 1 else None,
           "B_mean": round(st.mean(sb), 2), "B_sd": round(st.stdev(sb), 2) if len(sb) > 1 else None,
           "delta": round(st.mean(d), 2), "wins": wins, "losses": losses, "ties": len(d) - wins - losses,
           "A_success": sum(A[i]["success"] for i in ok), "B_success": sum(B[i]["success"] for i in ok),
           "flip_wins": flips_w, "flip_losses": flips_l,
           "B_ci95": [round(boots[249], 1), round(boots[9749], 1)], "bar": a.bar,
           "B_steps_median": st.median(B[i]["steps"] for i in ok), "A_steps_median": st.median(A[i]["steps"] for i in ok),
           "pairs": [{"id": i, "A": 100 * A[i]["score"], "B": 100 * B[i]["score"], "A_steps": A[i]["steps"],
                      "B_steps": B[i]["steps"], "A_success": A[i]["success"], "B_success": B[i]["success"]} for i in ok]}
    rep["passes"] = (len(ok) >= 10 and rep["delta"] >= 5 and wins >= 2 * losses and wins - losses >= 2)
    print(json.dumps(rep, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
