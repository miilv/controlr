"""Summarise PyRUA-Lean runs (run_task.sh output dirs) against PyRUA's official per-task numbers.

  python3 summarize.py /root/dojo-pyrua/runs/<stamp>_<task>_s0 [...] [--csv episodes.csv]

The verdict is RoboDojo's own (`robodojo_result/_result.json`: per-layout success and score).
Cost and behaviour come from PyRUA-Lean's per-episode record (`episodes/*/result.json`,
`openai_events.jsonl`). Stdlib only.
"""
import argparse, csv, glob, json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DIM = {"stack_blocks": "G", "put_bottles_into_dustbin": "L", "build_tower": "P", "insert_tubes": "P",
       "play_tic_tac_toe": "L", "make_toast": "G", "swap_blocks": "M", "cover_blocks": "M",
       "general_pickup": "O", "solve_equation": "O"}


def episodes(run):
    rd = json.load(open(os.path.join(run, "robodojo_result", "_result.json")))
    recs = {}
    for f in glob.glob(os.path.join(run, "episodes", "*", "result.json")):
        r = json.load(open(f))
        recs[int(r["seed"])] = (r, os.path.dirname(f))
    out = []
    for d in rd["details"].values():
        lay = int(d["layout_id"])
        r, edir = recs.get(lay, ({}, None))
        g = r.get("generation") or {}
        n429 = nerr = 0
        if edir:
            for line in open(os.path.join(edir, "openai_events.jsonl")):
                e = json.loads(line)
                if e.get("type") == "request" and not e.get("ok"):
                    n429 += e.get("status_code") == 429
                    nerr += e.get("status_code") != 429
        out.append({
            "task": r.get("task") or os.path.basename(run).split("_", 1)[1].rsplit("_s", 1)[0],
            "layout": lay, "success": bool(d["success"]), "score": 100.0 * float(d["score"]),
            "pyrua_success": r.get("environment_success"), "status": r.get("status"),
            "reason": r.get("reason"), "finish": (r.get("finish") or {}).get("status") if r.get("finish") else None,
            "llm_calls": g.get("requests"), "cells": r.get("tool_turns"),
            "input_tokens": g.get("input_tokens"), "cached_tokens": g.get("cached_input_tokens"),
            "output_tokens": g.get("output_tokens"), "reasoning_tokens": g.get("reasoning_output_tokens"),
            "images_sent": g.get("images_sent"), "env_steps": r.get("env_steps_total"),
            "step_limit": r.get("max_episode_steps"), "wall_s": r.get("total_wall_s"),
            "http_429": n429, "http_other_err": nerr, "dir": edir,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--csv")
    a = ap.parse_args()
    off = json.load(open(os.path.join(HERE, "official_pyrua_per_task.json")))
    rows = [e for run in a.runs for e in episodes(run)]
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    print("| Task | Dim | n | SR % | Score | Official SR / Score | LLM calls/ep | input tok/ep (cached %) | output tok/ep | env steps/ep (limit) | wall min/ep | 429s | failures |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    tot = {"n": 0, "calls": 0, "in": 0, "cached": 0, "out": 0, "wall": 0.0}
    for t in tasks:
        rs = [r for r in rows if r["task"] == t]
        n = len(rs)
        m = lambda k: st.mean([r[k] or 0 for r in rs])
        o = off.get(t, {})
        fails = ", ".join(f"L{r['layout']}:{r['status']}{'/' + r['finish'] if r['finish'] else ''}"
                          for r in rs if not r["success"]) or "-"
        print(f"| {t} | {DIM.get(t, '?')} | {n} | {100 * sum(r['success'] for r in rs) / n:.0f} | {m('score'):.1f} | "
              f"{o.get('successRate', '?')} / {o.get('score', '?')} | {m('llm_calls'):.1f} | "
              f"{m('input_tokens') / 1e3:.0f}k ({100 * sum(r['cached_tokens'] or 0 for r in rs) / max(1, sum(r['input_tokens'] or 0 for r in rs)):.0f}%) | "
              f"{m('output_tokens') / 1e3:.1f}k | {m('env_steps'):.0f} ({rs[0]['step_limit']}) | {m('wall_s') / 60:.1f} | "
              f"{sum(r['http_429'] for r in rs)} | {fails} |")
        tot["n"] += n; tot["calls"] += sum(r["llm_calls"] or 0 for r in rs)
        tot["in"] += sum(r["input_tokens"] or 0 for r in rs); tot["cached"] += sum(r["cached_tokens"] or 0 for r in rs)
        tot["out"] += sum(r["output_tokens"] or 0 for r in rs); tot["wall"] += sum(r["wall_s"] or 0 for r in rs)
    n = max(1, tot["n"])
    sr = 100 * sum(r["success"] for r in rows) / n
    print(f"\nEpisodes {tot['n']}, SR {sr:.1f}%, mean score {st.mean(r['score'] for r in rows):.1f}; "
          f"LLM calls {tot['calls']} ({tot['calls'] / n:.1f}/ep); input {tot['in'] / 1e6:.2f}M "
          f"({100 * tot['cached'] / max(1, tot['in']):.0f}% cached, {tot['in'] / n / 1e3:.0f}k/ep); "
          f"output {tot['out'] / 1e3:.0f}k ({tot['out'] / n / 1e3:.1f}k/ep); policy wall {tot['wall'] / 3600:.2f} h "
          f"({tot['wall'] / n / 60:.1f} min/ep)")


if __name__ == "__main__":
    main()
