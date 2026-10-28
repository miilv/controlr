"""Per-cell table of the agent-fallback ceiling runs: success / score / calls / tokens, Average points, failure modes.
    python3 scripts/agent_fallback/summarize.py ROOT [ROOT ...]"""
import collections
import glob
import json
import sys

W = {"cover_blocks": 1 / 30, "swap_T": 1 / 30, "pour_balls_into_vase": 1 / 40, "classify_objects": 1 / 40,
     "align_blocks": 1 / 40, "general_pickup": 1 / 40, "solve_equation": 1 / 40, "classify_objects_by_language": 1 / 40,
     "pour_by_language": 1 / 40}
for t in ("push_T", "pack_objects_into_box", "pour_liquid_into_cup", "arrange_largest_number", "sort_nesting_dolls_by_size"):
    W[t] = W[t + "_random"] = 1 / 120

rows = []
for root in sys.argv[1:]:
    for f in glob.glob(f"{root}/*/*_episodes.jsonl"):
        rows += [json.loads(l) for l in open(f)]
by = collections.defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
tot_pts = 0.0
modes = collections.Counter()
print("| cell | n | success | score | planner skill | agent calls/ep | prompt k (cached k) / out k per ep | wall s/ep | pts |")
print("|---|---|---|---|---|---|---|---|---|")
for t in sorted(W, key=lambda t: -W[t]):
    rs = [r for r in by.get(t, []) if not r.get("unstable_episode")]
    if not rs:
        print(f"| {t} | 0 | - | - | - | - | - | - | - |")
        continue
    n = len(rs)
    s = sum(bool(r["success"]) for r in rs)
    sc = sum(float(r["score"]) for r in rs) / n
    ag = [r.get("agent") or {} for r in rs]
    calls = [int((a.get("usage") or {}).get("calls") or 0) for a in ag]
    pt = [(a.get("usage") or {}).get("prompt_tokens", 0) for a in ag]
    ct = [(a.get("usage") or {}).get("cached_tokens", 0) or (a.get("usage") or {}).get("cache_read_tokens", 0) for a in ag]
    ot = [(a.get("usage") or {}).get("completion_tokens", 0) for a in ag]
    ps = collections.Counter(str(a.get("planner_skill")) for a in ag)
    pts = 100 * W[t] * sc
    tot_pts += pts
    for r, a in zip(rs, ag):
        if not r["success"]:
            modes[(r.get("category"), str(a.get("outcome")))] += 1
    print(f"| {t} | {n} | {s}/{n} | {100 * sc:.0f} | {dict(ps)} | {sum(calls) / n:.1f} | {sum(pt) / n / 1e3:.0f} ({sum(ct) / n / 1e3:.0f}) / "
          f"{sum(ot) / n / 1e3:.1f} | {sum(r['wall_s'] for r in rs) / n:.0f} | {pts:.2f} |")
print(f"\nAverage points: {tot_pts:.2f} (vs 0 today; 32.5 available)")
print("failure modes (category, agent outcome):", modes.most_common(8))
