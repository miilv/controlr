"""Phase 2 summary: per (model, task) success, score, LLM calls / tokens / time per episode, from integ eval runs.

    python -m dojo.agent.e2e_report RUN_DIR [RUN_DIR ...] [--md]

Reads every ``*_episodes.jsonl`` row (``dojo.runtime.eval``; ``row["agent"]`` = AgentPolicy's report). Rows of the
same (model, task, layout) in several dirs: the last one wins (a rerun replaces a crashed episode).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(dirs: list[Path]) -> list[dict]:
    rows = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*_episodes.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("skipped") or r.get("unstable"):
                    continue
                ag = r.get("agent") or {}
                key = (ag.get("model") or r.get("policy") or "rules", r["task"], r.get("layout_id"))
                rows[key] = r
    return list(rows.values())


def summarize(rows: list[dict]) -> list[dict]:
    g = defaultdict(list)
    for r in rows:
        ag = r.get("agent") or {}
        g[(ag.get("model") or "rules", r["task"])].append(r)
        g[(ag.get("model") or "rules", "ALL")].append(r)
    out = []
    for (model, task), rs in sorted(g.items()):
        us = [(r.get("agent") or {}).get("usage") or {} for r in rs]

        def m(k):
            v = [u.get(k) or 0 for u in us]
            return float(np.mean(v)) if v else 0.0
        out.append({"model": model, "task": task, "n": len(rs), "success": sum(bool(r["success"]) for r in rs),
                    "score": float(np.mean([r["score"] for r in rs])), "calls": m("calls"), "prompt": m("prompt_tokens"),
                    "cached": m("cached_tokens"), "out": m("completion_tokens"), "reasoning": m("reasoning_tokens"),
                    "llm_s": m("llm_s"), "wall_s": float(np.mean([r.get("wall_s") or 0 for r in rs])),
                    "steps": float(np.mean([r.get("steps_used") or 0 for r in rs])),
                    "calls_total": int(sum(u.get("calls") or 0 for u in us)),
                    "outcomes": dict(sorted(_count([(r.get("agent") or {}).get("outcome") for r in rs]).items())),
                    "skills": dict(sorted(_count([s for r in rs for s in ((r.get("agent") or {}).get("skills_called") or [])]).items())),
                    "fail": dict(sorted(_count([r.get("category") for r in rs if not r["success"]]).items()))})
    return out


def _count(xs):
    c: dict = {}
    for x in xs:
        c[str(x)] = c.get(str(x), 0) + 1
    return c


def first_decision(row: dict) -> str:
    """'skill <name>' when the first cell calls robo.skill, else 'code' (from the episode's cells.jsonl)."""
    import re
    d = (row.get("agent") or {}).get("log_dir")
    f = Path(d) / "cells.jsonl" if d else None
    if f is None or not f.exists():
        return "?"
    lines = f.read_text().splitlines()
    if not lines:
        return "none"
    code = json.loads(lines[0]).get("code", "")
    m = re.search(r"robo\.skill\(\s*['\"]([a-z_]+)['\"]", code)
    return f"skill {m.group(1)}" if m else "code"


def episodes_md(rows: list[dict]) -> str:
    out = ["| model | task | layout | success | score | calls | first cell | skills called | outcome | env steps | wall s |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: ((r.get("agent") or {}).get("model") or "", r["task"], r.get("layout_id"))):
        ag = r.get("agent") or {}
        out.append(f"| {ag.get('model')} | {r['task']} | {r.get('layout_id')} | {'yes' if r['success'] else 'no'} | "
                   f"{r['score']:.0f} | {(ag.get('usage') or {}).get('calls')} | {first_decision(r)} | "
                   f"{','.join(ag.get('skills_called') or []) or '-'} | {ag.get('outcome')} | {r.get('steps_used')} | "
                   f"{r.get('wall_s')} |")
    return "\n".join(out)


def to_md(summary: list[dict]) -> str:
    lines = ["| model | task | n | success | score | LLM calls / ep | tokens in / cached / out / reasoning per ep | "
             "LLM s / ep | wall s / ep | env steps / ep | agent outcomes | skills called | failure categories |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['model']} | {s['task']} | {s['n']} | {s['success']}/{s['n']} | {s['score']:.1f} | {s['calls']:.1f} | "
                     f"{s['prompt']:.0f} / {s['cached']:.0f} / {s['out']:.0f} / {s['reasoning']:.0f} | {s['llm_s']:.0f} | "
                     f"{s['wall_s']:.0f} | {s['steps']:.0f} | {s['outcomes']} | {s['skills']} | {s['fail']} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--episodes", action="store_true")
    a = ap.parse_args(argv)
    rows = load([Path(d) for d in a.dirs])
    if a.episodes:
        print(episodes_md(rows))
        return
    s = summarize(rows)
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=1))
    print(to_md(s) if a.md else json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
