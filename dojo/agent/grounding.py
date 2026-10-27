"""Phase 1 grounding benchmark: instruction + scene -> {skill, args} or {code, plan}, per model.

    python -m dojo.agent.grounding run   --dataset runs/agent/dataset.jsonl --arms astra-high,sol-6.1 \\
                                         --out runs/agent/grounding_<stamp> [--limit N] [--workers 6]
    python -m dojo.agent.grounding score runs/agent/grounding_<stamp> [--md]

One system prompt for every item (the manual + the decision appendix; cached prefix), one user turn per item
(``RUN <id>`` nonce: omniroute replays byte-identical requests). ``calls.jsonl`` keeps every call (reply,
parsed answer, usage, timings); a rerun skips (arm, item) pairs already answered, so no call is paid twice.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from dojo.agent.gold import gold_for
from dojo.agent.llmio import call_record, make_client
from dojo.agent.manual import render_manual

ARMS = {
    "astra-high": {"model": "cx/gpt-6-astra-high"},
    "astra-medium": {"model": "cx/gpt-6-astra-medium"},
    "sol-6.1": {"model": "cx/gpt-6.1-sol"},
    "sonnet-5.5": {"model": "claude/claude-sonnet-5-5"},                       # no field ~ effort high
    "sonnet-5.5-low": {"model": "claude/claude-sonnet-5-5", "extra_body": {"reasoning_effort": "low"}},
}
MAX_TOKENS = 16000


def system_prompt(manual: str = "manual_v1", appendix: str = "grounding_v1") -> str:
    return render_manual(manual) + (Path(__file__).with_name("prompts") / f"{appendix}.md").read_text()


def user_turn(item: dict, run_id: str) -> str:
    return f"RUN {run_id}\nINSTRUCTION: {item['instruction']}\n\n{item['scene']}"


def parse_answer(text: str) -> dict | None:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except ValueError:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, dict) else None
        except ValueError:
            return None
    return None


# ------------------------------------------------------------------ run

def run(dataset: Path, arms: list[str], out: Path, limit: int | None, workers: int, only: list[str] | None,
        manual: str = "manual_v1", appendix: str = "grounding_v1") -> None:
    items = [json.loads(l) for l in dataset.read_text().splitlines() if l.strip()]
    if only:
        items = [it for it in items if it["task"] in only or it["id"] in only]
    if limit:
        items = items[:limit]
    out.mkdir(parents=True, exist_ok=True)
    calls_f = out / "calls.jsonl"
    done = set()
    if calls_f.exists():
        for l in calls_f.read_text().splitlines():
            r = json.loads(l)
            if r.get("call", {}).get("ok"):
                done.add((r["arm"], r["id"]))
    sys_text = system_prompt(manual, appendix)
    (out / "system_prompt.md").write_text(sys_text)
    meta = {"dataset": str(dataset), "arms": {a: ARMS[a] for a in arms}, "n_items": len(items),
            "manual": manual, "appendix": appendix,
            "max_tokens": MAX_TOKENS, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    lock = threading.Lock()
    run_id = uuid.uuid4().hex[:10]

    def one(client, arm, it):
        spec = ARMS[arm]
        msgs = [{"role": "system", "content": sys_text}, {"role": "user", "content": user_turn(it, run_id)}]
        res = client.complete(spec["model"], msgs, max_tokens=MAX_TOKENS, extra_body=spec.get("extra_body"))
        rec = {"arm": arm, "id": it["id"], "task": it["task"], "reply": res.text, "answer": parse_answer(res.text),
               "call": call_record(res, spec["model"]), "reasoning_chars": len(res.reasoning_text or "")}
        with lock:
            with calls_f.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        a = rec["answer"] or {}
        print(f"[grounding] {arm:14s} {it['id']:40s} {('skill ' + str(a.get('skill'))) if 'skill' in a else ('code' if a.get('code') else 'PARSE?')}"
              f"  {rec['call']['t_wall'] or 0:.1f}s {rec['call']['error'] or ''}", flush=True)

    for arm in arms:
        todo = [it for it in items if (arm, it["id"]) not in done]
        if not todo:
            continue
        client = make_client(timeout_s=600, max_retries=3)
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(lambda it: one(client, arm, it), todo))
        finally:
            client.close()


# ------------------------------------------------------------------ score

def _cmp(spec: dict, val, args: dict | None = None) -> bool:
    """``args``: every argument of the answer (``pegmap`` pairs this one with ``spec["with"]``)."""
    if val is None and "default" in spec:
        val = spec["default"]
    op, ref = spec["op"], spec.get("value")
    try:
        if op == "any":
            return True
        if op == "in":
            return val in ref
        if op == "k_of":           # exactly ``k`` distinct ids, all from ``allowed`` (any ``k`` of the slices)
            return (isinstance(val, list) and len(val) == spec["k"] and len(set(map(str, val))) == len(val)
                    and set(map(str, val)) <= set(spec["allowed"]))
        if op == "pegmap":         # groups[i] goes onto with[i]: the same {frame: set of ids} as ``value``
            tags = (args or {}).get(spec["with"])
            if not (isinstance(val, list) and isinstance(tags, list) and len(val) == len(tags)):
                return False
            got = {str(t): sorted(map(str, g)) for g, t in zip(val, tags) if isinstance(g, list)}
            return got == {k: sorted(v) for k, v in ref.items()}
        if op == "eq":
            return val == ref or (isinstance(ref, bool) and isinstance(val, str) and val.lower() == str(ref).lower())
        if op == "approx":
            return abs(float(val) - float(ref)) <= float(spec.get("tol", 1e-6))
        if op == "set":
            norm = lambda xs: sorted(json.dumps(x) for x in xs)    # noqa: E731
            return isinstance(val, list) and norm(val) == norm(ref)
        if op == "seq":
            return isinstance(val, list) and [list(x) if isinstance(x, (list, tuple)) else x for x in val] == ref
        if op == "has":
            return isinstance(val, list) and set(ref) <= set(val) and set(val) <= set(spec.get("allowed", ref))
    except (TypeError, ValueError):
        return False
    return False


def judge(answer: dict | None, gold: dict) -> dict:
    a = answer or {}
    is_code = bool(a.get("code")) and "skill" not in a
    skill = a.get("skill") if not is_code else None
    out = {"parsed": answer is not None, "answer_kind": "code" if is_code else ("skill" if skill else "none"),
           "gold_kind": gold["kind"], "decision_ok": False, "skill_ok": None, "args_ok": None, "need_ok": None,
           "bad_args": []}
    if gold["kind"] == "skill":
        out["decision_ok"] = skill is not None
        out["skill_ok"] = skill == gold["skill"]
        if out["skill_ok"]:
            args = a.get("args") or {}
            bad = [k for k, spec in gold["args"].items() if not _cmp(spec, args.get(k), args)]
            out["args_ok"] = not bad
            out["bad_args"] = bad
        else:
            out["args_ok"] = False
        out["full_ok"] = bool(out["args_ok"])
    else:
        alt_ok = skill is not None and any(skill == g["skill"] and all(_cmp(sp, (a.get("args") or {}).get(k),
                                                                            a.get("args") or {})
                                                                     for k, sp in g["args"].items())
                                           for g in gold.get("alt", []))
        out["decision_ok"] = is_code or alt_ok
        out["alt_ok"] = alt_ok
        if gold.get("need"):
            named = set(map(str, a.get("objects") or [])) | set(re.findall(r"[a-z_]+_\d+", str(a.get("plan", ""))))
            out["need_ok"] = set(gold["need"]) <= named
        out["full_ok"] = (is_code and out["need_ok"] is not False) or alt_ok
    # what a wrong answer costs: a skill call that will not satisfy the checker fails the task (harmful); code
    # where one skill would do costs calls but can still succeed (missed)
    out["harmful"] = skill is not None and not out["full_ok"]
    out["missed_skill"] = gold["kind"] == "skill" and is_code
    if is_code and gold.get("home"):
        out["plan_home"] = bool(re.search(r"\bhome|finish|reset", str(a.get("plan", "")), re.I))
    return out


def planner_answer(item: dict) -> dict:
    p = item["planner"]
    if p["skill"] in ("none", "error"):
        return {"code": True, "plan": p.get("why", ""), "objects": []}
    return {"skill": p["skill"], "args": p["args"]}


def _pct(xs):
    xs = [x for x in xs if x is not None]
    return (100.0 * sum(map(bool, xs)) / len(xs), len(xs)) if xs else (float("nan"), 0)


def score(run_dir: Path, dataset: Path | None, md: bool) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    dataset = dataset or Path(meta["dataset"])
    items = {it["id"]: it for it in (json.loads(l) for l in dataset.read_text().splitlines() if l.strip())}
    golds = {i: gold_for(it) for i, it in items.items()}
    recs = {}
    for l in (run_dir / "calls.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r["id"] in items:
            if r["call"]["ok"] or (r["arm"], r["id"]) not in recs:
                recs[(r["arm"], r["id"])] = r
    arms = list(meta["arms"]) + ["planner(rules)"]
    rows, per_task = [], {}
    for arm in arms:
        js, calls = [], []
        for iid, it in items.items():
            if arm == "planner(rules)":
                ans, c = planner_answer(it), None
            else:
                r = recs.get((arm, iid))
                if r is None:
                    continue
                ans, c = r["answer"], r["call"]
                calls.append(c)
            j = judge(ans, golds[iid])
            j["id"], j["task"] = iid, it["task"]
            js.append(j)
            per_task.setdefault(it["task"], {}).setdefault(arm, []).append(j["full_ok"])
        if not js:
            continue
        sk = [j for j in js if j["gold_kind"] == "skill"]
        co = [j for j in js if j["gold_kind"] == "code"]
        lat = [c["t_wall"] for c in calls if c and c.get("t_wall") is not None and c["ok"]]
        tok = lambda k: [c[k] for c in calls if c and c.get(k) is not None]  # noqa: E731
        row = {"arm": arm, "n": len(js),
               "full": _pct([j["full_ok"] for j in js]),
               "decision": _pct([j["decision_ok"] for j in js]),
               "skill_choice": _pct([j["skill_ok"] for j in sk]),
               "args": _pct([j["args_ok"] for j in sk]),
               "args_given_skill": _pct([j["args_ok"] for j in sk if j["skill_ok"]]),
               "code_recall": _pct([j["decision_ok"] for j in co]),
               "false_skill": _pct([j["answer_kind"] == "skill" and not j.get("alt_ok") for j in co]),
               "harmful": _pct([j["harmful"] for j in js]),
               "missed_skill": _pct([j["missed_skill"] for j in sk]),
               "target_grounding": _pct([j["need_ok"] for j in co]),
               "plan_home": _pct([j.get("plan_home") for j in co]),
               "parse_fail": sum(not j["parsed"] for j in js),
               "errors": sum(1 for c in calls if c and not c["ok"]),
               "calls": len(calls)}
        if lat:
            row["lat_p50"], row["lat_p90"] = float(np.percentile(lat, 50)), float(np.percentile(lat, 90))
        for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens"):
            v = tok(k)
            row[f"{k}_mean"] = float(np.mean(v)) if v else None
            row[f"{k}_sum"] = int(np.sum(v)) if v else 0
        row["wrong"] = [(j["id"], j["answer_kind"], j["bad_args"]) for j in js if not j["full_ok"]]
        rows.append(row)
    res = {"rows": rows, "per_task": {t: {a: (sum(v), len(v)) for a, v in d.items()} for t, d in per_task.items()}}
    (run_dir / "score.json").write_text(json.dumps(res, indent=1, default=str))
    if md:
        print(to_markdown(res))
    return res


def to_markdown(res: dict) -> str:
    f = lambda p: "–" if p[1] == 0 else f"{p[0]:.0f}% ({p[1]})"  # noqa: E731
    lines = ["| arm | full correct | harmful (skill that fails the checker) | decision | skill choice | args, right "
             "skill | missed skill (code instead) | wrong skill on code items | target grounding | lat p50 / p90 s | "
             "in / cached / out / reasoning tokens (mean) | calls (err) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        lat = f"{r['lat_p50']:.1f} / {r['lat_p90']:.1f}" if "lat_p50" in r else "–"
        tok = ("–" if r["prompt_tokens_mean"] is None else
               f"{r['prompt_tokens_mean']:.0f} / {r['cached_tokens_mean'] or 0:.0f} / {r['completion_tokens_mean']:.0f} / "
               f"{r['reasoning_tokens_mean'] or 0:.0f}")
        lines.append(f"| {r['arm']} | {f(r['full'])} | {f(r['harmful'])} | {f(r['decision'])} | {f(r['skill_choice'])} | "
                     f"{f(r['args_given_skill'])} | {f(r['missed_skill'])} | {f(r['false_skill'])} | "
                     f"{f(r['target_grounding'])} | {lat} | {tok} | {r['calls']} ({r['errors']}) |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True)
    r.add_argument("--arms", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--only", nargs="*", default=None, help="task names or item ids")
    r.add_argument("--manual", default="manual_v1")
    r.add_argument("--appendix", default="grounding_v1")
    s = sub.add_parser("score")
    s.add_argument("run_dir")
    s.add_argument("--dataset", default=None)
    s.add_argument("--md", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(Path(a.dataset), a.arms.split(","), Path(a.out), a.limit, a.workers, a.only, a.manual, a.appendix)
    else:
        score(Path(a.run_dir), Path(a.dataset) if a.dataset else None, a.md)


if __name__ == "__main__":
    main()
