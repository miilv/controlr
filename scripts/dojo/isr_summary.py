"""One row per CLOSED practice bridge episode on isr-lab (evaluation only, after the episode's close):

    .venv/bin/python scripts/dojo/isr_summary.py ID [ID ...] [--bench ~/data2/ilia/bench] [--runs ...] [--jsonl out]

Reads OUR outputs (``<runs>/<id>/bridge.json``, ``report.json.gz``) and, because the episode is closed and practice
(seed 0 layout 9), the server's dev-side files in OUR bench run dir: RoboDojo's ``robodojo_result/_result.json``
(the score), ``dev.json`` (task / layout / seed: never visible to the policy) and ``dev.jsonl``'s ``hello`` (the
objects' start poses) for the perception error at planning. Never used during an episode.
Perception error: per category, each true object is matched to the nearest estimate of the same category (Hungarian
on xy); "gross" = > 30 mm off or a different asset index, or not found.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

GROSS_MM = 30.0


def _load(p: Path):
    if not p.exists():
        return None
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as f:
            return json.load(f)
    return json.loads(p.read_text())


def record_objects(rec: dict) -> list[dict]:
    out = []
    for typ, cats in (rec or {}).items():
        if typ.startswith("_") or not isinstance(cats, dict):
            continue
        for cat, rows in cats.items():
            for r in rows if isinstance(rows, list) else []:
                if isinstance(r, dict) and "default_pos" in r:
                    out.append({"category": cat, "idx": r.get("category_idx"), "label": r.get("label"),
                                "pos": np.asarray(r["default_pos"][:3], float)})
    return out


def truth_objects(hello: dict, est_categories: set | None = None) -> list[dict]:
    """Labelled task objects; static Geometry fixtures (a camera stand) only when the estimate has that category."""
    pos = hello.get("objects_pos") or {}
    out = []
    for o in hello.get("objects") or []:
        if not o.get("label"):
            continue                                   # clutter / background: not scored here
        if o.get("type") == "Geometry" and est_categories is not None and o.get("category") not in est_categories:
            continue
        inst = o.get("inst") or ""
        p = pos.get(inst)
        if isinstance(p, dict):
            p = p.get("pos")
        if p is None:
            continue
        idx = None
        parts = inst.split("_")
        if len(parts) >= 3 and parts[-2].isdigit():
            idx = int(parts[-2])
        out.append({"category": o.get("category"), "idx": idx, "label": o["label"], "pos": np.asarray(p[:3], float)})
    return out


def perception_errors(rec: dict, truth: list[dict]) -> dict:
    from scipy.optimize import linear_sum_assignment

    est = record_objects(rec)
    rows, gross = [], 0
    for cat in sorted({t["category"] for t in truth}):
        T = [t for t in truth if t["category"] == cat]
        E = [e for e in est if e["category"] == cat]
        # identity is assessed only where the estimate names assets: a category whose estimates all carry one index
        # while the truth has several (make_kong's tiles: grouped by label, not by face) is position-only
        judge_id = not (len({e["idx"] for e in E}) == 1 and len({t["idx"] for t in T}) > 1)
        if E:
            C = np.array([[np.linalg.norm(t["pos"][:2] - e["pos"][:2]) for e in E] for t in T])
            ti, ei = linear_sum_assignment(C)
            pairs = dict(zip(ti, ei))
        else:
            pairs = {}
        for i, t in enumerate(T):
            if i not in pairs:
                rows.append({"label": t["label"], "found": False})
                gross += 1
                continue
            e = E[pairs[i]]
            err = float(np.linalg.norm(t["pos"] - e["pos"]) * 1000)
            ident = (not judge_id) or t["idx"] is None or e["idx"] is None or int(t["idx"]) == int(e["idx"])
            g = err > GROSS_MM or not ident
            gross += int(g)
            rows.append({"label": t["label"], "found": True, "err_mm": round(err, 1), "identity_ok": ident, "gross": g})
    errs = [r["err_mm"] for r in rows if r.get("found")]
    return {"n_true": len(truth), "n_est": len(est), "gross": gross,
            "err_mm_med": round(float(np.median(errs)), 1) if errs else None,
            "err_mm_max": round(float(max(errs)), 1) if errs else None, "objects": rows}


def agent_usage(rep: dict) -> dict:
    ag = rep.get("agent") or {}
    u = ag.get("usage") or {}
    return {"model": ag.get("model"), "calls": u.get("calls", ag.get("calls")),
            "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": u.get("cached_tokens"),
            "completion_tokens": u.get("completion_tokens"), "outcome": ag.get("outcome")}


def row(rid: str, bench: Path, runs: Path) -> dict:
    rd, od = bench / "run" / rid, runs / rid
    b = _load(od / "bridge.json") or {}
    rep = _load(od / "report.json.gz") or {}
    res = _load(rd / "robodojo_result" / "_result.json") or {}
    dev = _load(rd / "dev.json") or {}
    hello = None
    if (rd / "dev.jsonl").exists():
        for line in (rd / "dev.jsonl").read_text().splitlines():
            d = json.loads(line)
            if d.get("kind") == "hello":
                hello = d
    det = (res.get("details") or {}).get("0") or {}
    recs = rep.get("records") or {}
    out = {"id": rid, "task": dev.get("task"), "seed": dev.get("seed"), "layout": dev.get("layouts"),
           "policy": (b.get("policy") or "").split(":")[-1], "perception": b.get("perception"),
           "agentic_mode": (b.get("config") or {}).get("agentic"), "verify": b.get("verify"),
           "success": det.get("success"), "score": det.get("score"),
           "steps": b.get("steps_used"), "step_lim": b.get("step_lim"), "end": b.get("end"),
           "wall_s": b.get("wall_s"), "policy_s": b.get("policy_s"), "exec_s": b.get("exec_s"),
           "twin_boot_s": b.get("twin_boot_s"), "skill": rep.get("skill"), "result": rep.get("result"),
           "error": rep.get("error") or b.get("policy_error"),
           "verify_stats": {k: (rep.get("verify") or {}).get(k) for k in ("segments", "calls", "rollouts", "wall_s",
                                                                           "disabled", "first_ok", "none_ok")},
           "tracking_max_rad": (b.get("backend") or {}).get("tracking_max_rad"),
           "grip_change_while_moving": (b.get("backend") or {}).get("grip_change_while_moving"),
           "agent": agent_usage(rep), "agentic_calls": rep.get("agentic") or [],
           "agentic_gate": [e for e in rep.get("events") or [] if e.get("kind") == "agentic_gate"],
           "env": b.get("env")}
    if hello is not None:
        cats = {e["category"] for k in ("initial", "fused", "searched", "agentic2") for e in record_objects(recs.get(k) or {})}
        truth = truth_objects(hello, cats)
        out["perception_initial"] = perception_errors(recs.get("initial") or {}, truth)
        last = next((recs[k] for k in ("agentic2", "searched", "fused") if recs.get(k)), None)
        if last is not None:
            out["perception_final"] = perception_errors(last, truth)
    return out


def _triggers(r: dict) -> str:
    n: dict[str, int] = {}
    for c in r.get("agentic_calls") or []:
        t = c.get("tag") or "?"
        n[t] = n.get(t, 0) + int(c.get("calls") or 0)
    return ", ".join(f"{k} {v}" for k, v in n.items()) or "-"


def markdown(rows: list[dict]) -> str:
    out = ["| run | task | policy | perception | success / score | steps | wall s | perception gross at planning "
           "(err med / max mm) | verify used | agent calls | agentic calls by trigger |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pi = r.get("perception_initial") or {}
        vs = r.get("verify_stats") or {}
        used = "no (disabled)" if vs.get("disabled") else ("yes" if vs.get("calls") else "no")
        ag = r.get("agent") or {}
        pol = r["policy"] + (" + agentic" if r.get("agentic_mode") == "budget" else "")
        sc = "-" if r.get("score") is None else f"{'yes' if r.get('success') else 'no'} / {100 * float(r['score']):.0f}"
        out.append(f"| {r['id']} | {r['task']} | {pol} | {r['perception']} | {sc} | {r['steps']}/{r['step_lim']} | "
                   f"{r['wall_s']} | {pi.get('gross')}/{pi.get('n_true')} ({pi.get('err_mm_med')} / {pi.get('err_mm_max')}) | "
                   f"{used} | {ag.get('calls') or 0} | {_triggers(r)} |")
    return "\n".join(out)


def main(argv=None) -> None:
    ilia = Path("/home/anywherevla/data2/ilia")
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--bench", default=str(ilia / "bench"))
    ap.add_argument("--runs", default=str(ilia / "dojo-data" / "bridge" / "runs"))
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--md", action="store_true", help="a markdown table instead of the one-line summaries")
    a = ap.parse_args(argv)
    rows = [row(i, Path(a.bench), Path(a.runs)) for i in a.ids]
    if a.md:
        print(markdown(rows))
        return
    if a.jsonl:
        with open(a.jsonl, "a") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
    for r in rows:
        pi, pf = r.get("perception_initial") or {}, r.get("perception_final") or {}
        print(f"{r['id']}: {r['task']} s{r['seed']} L{r['layout']} {r['policy']} perc={r['perception']} "
              f"agentic={r['agentic_mode']} -> success={r['success']} score={r['score']} steps={r['steps']}/{r['step_lim']} "
              f"wall={r['wall_s']}s twin_boot={r['twin_boot_s']}s verify={r['verify_stats']} "
              f"perc gross init={pi.get('gross')}/{pi.get('n_true')} (med {pi.get('err_mm_med')} max {pi.get('err_mm_max')} mm) "
              f"final={pf.get('gross')} | agent={r['agent']} agentic_calls={len(r['agentic_calls'])}")


if __name__ == "__main__":
    main()
