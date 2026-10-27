"""The official path on our dev layouts (``scripts/dojo/official_eval.sh``): result rows, and the per-layout pairing
with ``integ_eval`` (``dojo.runtime.eval``) on the same layouts.

    python -m dojo.bridge.official rows OUT                                  # -> OUT/rows.jsonl (+ a table)
    python -m dojo.bridge.official compare --official OUT [OUT ...] --integ DIR [DIR ...] [--md FILE]

``rows`` joins, per episode, RoboDojo's own verdict (``OUT/_result.json``, written by the eval client after the
episode was booked: ``details[i].layout_id`` is the index of the dev layout in id order, ``dojo/simlab/dev_eval.py``)
with what our side recorded (the driver's ``bridge.json`` / ``report.json.gz`` under ``OUT/eps/<layout>/<server id>/``).
Read after the run only; the driver never reads a verdict.

``compare`` pairs official rows with integ_eval rows by (task, layout): success, score, steps, failure category, and
the first executed action where the two paths differ. Physics is deterministic (bitwise, journal 2026-10-26-bridge-isr
A2), so two runs that executed the same actions reached the same states: an outcome disagreement with identical
actions is a scoring / accounting difference (step accounting, success latch, the end-of-episode rule); one after a
divergence is a decision difference, whose cause is at the divergence step (an observation difference, timing, an
LLM's non-determinism, ...).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def staged(out: Path) -> tuple[str, list[int]]:
    """(task, layout ids in the eval client's order) from ``OUT/layouts/<task>/<id>.settled.json``."""
    tasks = [d for d in (out / "layouts").iterdir() if d.is_dir()]
    if len(tasks) != 1:
        raise ValueError(f"{out}/layouts: expected one task dir, found {[d.name for d in tasks]}")
    ids = sorted(int(p.name.split(".")[0]) for p in tasks[0].glob("*.json") if p.name.split(".")[0].isdigit())
    return tasks[0].name, ids


def official_results(out: Path, ids: list[int]) -> dict[int, dict]:
    """{layout id: {success, score, index}} from RoboDojo's ``_result.json`` (absent: {})."""
    p = out / "_result.json"
    if not p.is_file():
        return {}
    res = json.loads(p.read_text())
    got = {}
    for idx, d in (res.get("details") or {}).items():
        i = int(d["layout_id"])
        if 0 <= i < len(ids):
            got[ids[i]] = {"success": bool(d.get("success")), "score": float(d.get("score") or 0.0), "index": int(idx)}
    return got


def _ep_dir(out: Path, lid: int) -> Path | None:
    hits = sorted((out / "eps" / str(lid)).glob("*/bridge.json"))
    return hits[-1].parent if hits else None


def _report(d: Path | None) -> dict:
    if d is None or not (d / "report.json.gz").is_file():
        return {}
    with gzip.open(d / "report.json.gz", "rt") as f:
        return json.load(f)


def load_actions(path: Path | str | None) -> list[dict] | None:
    if not path or not Path(path).is_file():
        return None
    with gzip.open(path, "rt") as f:
        return json.load(f).get("actions")


def episode_row(out: Path, task: str, lid: int, verdict: dict | None) -> dict:
    from dojo.runtime.eval import classify
    d = _ep_dir(out, lid)
    s = json.loads((d / "bridge.json").read_text()) if d else {}
    rep = _report(d)
    succ = (verdict or {}).get("success")
    res = {"success": bool(succ), "steps_used": s.get("steps_used", 0), "step_lim": s.get("step_lim", 1)}
    cat, why = classify(res, {}, rep, None, "perception") if verdict is not None else (None, "")
    if verdict is not None and not succ and s.get("end") == "policy_done" and cat == "other":
        cat, why = "policy_done", str(rep.get("result"))[:200]
    if verdict is None and d is not None and (out / "_result.json").is_file():
        # the client booked the run but not this episode: the task marked it unstable (mark_env_unstable: make_kong's
        # scripted discard, play_tic_tac_toe's opponent, imitate's demonstrator) and RoboDojo dropped it from the eval
        cat, why = "unstable_episode", "no row in RoboDojo's _result.json: dropped from the eval (unstable_nums)"
    return {"path": "official", "task": task, "layout_id": lid, "success": succ,
            "score": (verdict or {}).get("score"), "steps_used": s.get("steps_used"), "step_lim": s.get("step_lim"),
            "end": s.get("end"), "wall_s": s.get("wall_s"), "policy_s": s.get("policy_s"), "exec_s": s.get("exec_s"),
            "observe_s": s.get("observe_s"), "twin_scene": s.get("twin_scene"), "calib_fold": s.get("calib_fold"),
            "policy_error": s.get("policy_error") or s.get("policy_error_handled"), "category": cat, "why": why,
            "skill": rep.get("skill"), "agent": rep.get("agent"), "agentic_n": len(rep.get("agentic") or []),
            "joint_steps_chunks": len(s.get("joint_steps_chunks") or []),
            "grip_change_while_moving": (s.get("backend") or {}).get("grip_change_while_moving"),
            "tracking_max_rad": (s.get("backend") or {}).get("tracking_max_rad"),
            "actions_path": str(d / "actions.json.gz") if d else None, "dir": str(d) if d else None}


def rows(out: Path) -> list[dict]:
    task, ids = staged(out)
    verdicts = official_results(out, ids)
    return [episode_row(out, task, lid, verdicts.get(lid)) for lid in ids]


def load_official(dirs: list[str]) -> dict[tuple, dict]:
    got = {}
    for d in dirs:
        p = Path(d) / "rows.jsonl"
        rs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.is_file() else rows(Path(d))
        for r in rs:
            got[(r["task"], int(r["layout_id"]))] = r
    return got


def load_integ(dirs: list[str]) -> dict[tuple, dict]:
    got = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*_episodes.jsonl")):
            tag = f.name[: -len("_episodes.jsonl")]
            for ln in f.read_text().splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if "success" not in r:
                    continue
                lid = int(r["layout_id"])
                r = {**r, "path": "integ", "actions_path": str(Path(d) / f"{tag}_{lid}_actions.json.gz")}
                got[(r["task"], lid)] = r
    return got


def first_divergence(a: list[dict] | None, b: list[dict] | None, tol: float = 1e-4) -> int | None:
    """Index of the first executed action that differs (beyond ``tol`` on any key), the shorter length when one list
    is a prefix of the other, None when identical (or either is missing: then -1). ``tol`` 1e-4 rad: the strict
    server rounds the measured joints / flange it reports to 1e-5 (and the commanded gripper to 1e-4), so plans
    started from them differ from the in-process ones by that much without being a different decision."""
    if a is None or b is None:
        return -1
    for i, (x, y) in enumerate(zip(a, b)):
        if set(x) != set(y) or any(np.max(np.abs(np.asarray(x[k], float) - np.asarray(y[k], float))) > tol for k in x):
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def pair(off: dict[tuple, dict], integ: dict[tuple, dict], actions: bool = True) -> list[dict]:
    out = []
    for key in sorted(set(off) & set(integ)):
        o, g = off[key], integ[key]
        div = first_divergence(load_actions(o.get("actions_path")), load_actions(g.get("actions_path"))) if actions else -1
        so, sg = o.get("score"), g.get("score")
        agree = (o.get("success") == g.get("success")) and so is not None and sg is not None and abs(so - sg) < 1e-6
        if o.get("category") == "unstable_episode" or g.get("unstable_episode"):    # not a sample on that path
            agree = o.get("category") == "unstable_episode" and bool(g.get("unstable_episode"))
        out.append({"task": key[0], "layout_id": key[1], "agree": agree, "first_divergence": div,
                    "official": {k: o.get(k) for k in ("success", "score", "steps_used", "end", "category", "why",
                                                       "wall_s", "policy_error")},
                    "integ": {k: g.get(k) for k in ("success", "score", "steps_used", "category", "why", "wall_s",
                                                    "unstable_episode")}})
    return out


def _fmt(side: dict) -> str:
    if side.get("success") is None:
        return "-"
    sc = side.get("score")
    return (f"{'yes' if side['success'] else 'no'} / {100 * sc:.0f}" if sc is not None else "?") + f" / {side.get('steps_used')}"


def table_md(pairs: list[dict]) -> str:
    lines = ["| task | layout | official (success / score / steps) | integ_eval | agree | first action divergence | "
             "official end, category | integ category |", "|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        o, g = p["official"], p["integ"]
        div = {None: "none (identical)", -1: "n/a"}.get(p["first_divergence"], f"step {p['first_divergence']}")
        lines.append(f"| {p['task']} | {p['layout_id']} | {_fmt(o)} | {_fmt(g)} | {'yes' if p['agree'] else '**no**'} | "
                     f"{div} | {o.get('end')}, {o.get('category') or '-'} | {g.get('category') or '-'} |")
    n = len(pairs)
    lines.append("")
    lines.append(f"n = {n}; agree {sum(p['agree'] for p in pairs)}/{n}; official success "
                 f"{sum(bool(p['official'].get('success')) for p in pairs)}/{n}, integ success "
                 f"{sum(bool(p['integ'].get('success')) for p in pairs)}/{n}")
    return "\n".join(lines)


def rows_md(rs: list[dict]) -> str:
    lines = ["| task | layout | success / score | steps | end | category | wall s | skill | why |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        sc = r.get("score")
        lines.append(f"| {r['task']} | {r['layout_id']} | {r.get('success')} / {'-' if sc is None else round(100 * sc)} | "
                     f"{r.get('steps_used')}/{r.get('step_lim')} | {r.get('end')} | {r.get('category') or '-'} | "
                     f"{r.get('wall_s')} | {r.get('skill')} | {str(r.get('why') or '')[:100]} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows")
    r.add_argument("out")
    c = sub.add_parser("compare")
    c.add_argument("--official", nargs="+", required=True)
    c.add_argument("--integ", nargs="+", required=True)
    c.add_argument("--md", default=None)
    c.add_argument("--json", default=None)
    c.add_argument("--no-actions", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "rows":
        out = Path(a.out)
        rs = rows(out)
        (out / "rows.jsonl").write_text("".join(json.dumps(x, default=str) + "\n" for x in rs))
        print(rows_md(rs))
        return
    ps = pair(load_official(a.official), load_integ(a.integ), actions=not a.no_actions)
    md = table_md(ps)
    print(md)
    if a.md:
        Path(a.md).write_text(md + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(ps, indent=1, default=str))


if __name__ == "__main__":
    main()
