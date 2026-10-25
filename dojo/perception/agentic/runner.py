"""The agent loop: one fresh conversation per call, JSON replies, tool results fed back as user turns.

Reply contract (every turn, one JSON object; text-only so it works on every route, no provider tool-calling):
    {"commands": ["<command>", ...], "note": "<optional>"}     run up to ``max_cmds`` tools, results come back
    {"answer": {...}}                                         final answer (schema in the runner preamble)

The system prompt is fixed per pin (``prompt.system_prompt``); nothing is carried between calls.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .llm import ChatReply, image_part

FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_reply(text: str, keys=("answer", "commands")) -> dict | None:
    """The reply's JSON object (one holding any of ``keys``): whole text, a ```json fence, or the outermost {...}."""
    t = (text or "").strip()
    for cand in [t] + FENCE.findall(t) + ([t[t.index("{"): t.rindex("}") + 1]] if "{" in t and "}" in t else []):
        try:
            d = json.loads(cand)
        except Exception:
            continue
        if isinstance(d, dict) and any(k in d for k in keys):
            return d
    # some routes write a function call as markup in the text (<invoke name=...><parameter name="commands">[...]</parameter>)
    alt = "|".join(re.escape(k) for k in keys)
    for m in re.finditer(r'<parameter name="(%s)">\s*(.*?)\s*</parameter>' % alt, t, re.S):  # first non-empty, in order
        try:
            v = json.loads(m.group(2))
        except Exception:
            continue
        if v:
            return {m.group(1): v}
    return None


def run_agent(chat, model: str, system: str, first_user: list[dict], tools, *, max_turns: int = 8, max_cmds: int = 6,
              max_images: int = 6, max_tokens: int = 6000, extra_body: dict | None = None, log_dir: Path | None = None,
              wall_budget_s: float | None = None) -> dict:
    """-> {"answer": dict | None, "turns", "calls", "tokens": {...}, "wall_s", "commands", "errors", "stop"}."""
    t0 = time.time()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": first_user}]
    stats = {"answer": None, "turns": 0, "calls": 0, "commands": 0, "errors": [], "stop": None,
             "tokens": {"prompt": 0, "completion": 0, "reasoning": 0, "cache_read": 0}, "call_s": []}
    logf = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logf = open(log_dir / "transcript.jsonl", "a")
        logf.write(json.dumps({"event": "start", "model": model, "system_chars": len(system),
                               "first_user": [p.get("text", "<image>") if p.get("type") == "text" else "<image>" for p in first_user]}) + "\n")
    try:
        for turn in range(max_turns):
            stats["turns"] = turn + 1
            r: ChatReply = chat(model, messages, max_tokens=max_tokens, extra_body=extra_body)
            stats["calls"] += 1
            stats["call_s"].append(round(r.wall_s, 2))
            for k, a in (("prompt", r.prompt_tokens), ("completion", r.completion_tokens), ("reasoning", r.reasoning_tokens),
                         ("cache_read", r.cache_read_tokens)):
                stats["tokens"][k] += a or 0
            if logf:
                logf.write(json.dumps({"event": "reply", "turn": turn, "text": r.text, "error": r.error, "wall_s": round(r.wall_s, 2),
                                       "tokens": [r.prompt_tokens, r.completion_tokens, r.reasoning_tokens, r.cache_read_tokens],
                                       "request_bytes": r.request_bytes, "finish": r.finish_reason}) + "\n")
                logf.flush()
            if r.error and not r.text:
                stats["errors"].append(f"turn {turn}: {r.error}")
                if len(stats["errors"]) >= 2:
                    stats["stop"] = "llm_error"
                    break
                continue
            # never an empty assistant turn: Anthropic's endpoint rejects one (omniroute tolerated it)
            messages.append({"role": "assistant", "content": r.text if r.text.strip() else "(empty reply)"})
            d = parse_reply(r.text)
            last = turn == max_turns - 2
            if d is None:
                stats["errors"].append(f"turn {turn}: no JSON object")
                messages.append({"role": "user", "content": [{"type": "text", "text":
                                 'Your reply had no parsable JSON object. Reply with exactly one JSON object: '
                                 '{"commands": [...]} or {"answer": {...}}.' + (" This is your last turn: answer now." if last else "")}]})
                continue
            if "answer" in d:
                stats["answer"] = d["answer"] if isinstance(d["answer"], dict) else {"objects": d["answer"]}
                stats["stop"] = "answer"
                break
            cmds = [c for c in (d.get("commands") or []) if isinstance(c, str)][:max_cmds]
            parts, n_img = [], 0
            for i, c in enumerate(cmds, 1):
                text, imgs = tools.run(c)
                stats["commands"] += 1
                parts.append({"type": "text", "text": f"[{i}] $ {c}\n{text}"})
                for name, png in imgs:
                    if n_img >= max_images:
                        parts.append({"type": "text", "text": f"(image {name} not attached: {max_images} images per turn)"})
                        continue
                    parts.append({"type": "text", "text": f"image {name}:"})
                    parts.append(image_part(png))
                    n_img += 1
                if logf:
                    logf.write(json.dumps({"event": "tool", "turn": turn, "cmd": c, "out": text[:2000], "images": [n for n, _ in imgs]}) + "\n")
            if not cmds:
                parts.append({"type": "text", "text": "No commands given."})
            left = max_turns - turn - 1
            over = wall_budget_s is not None and time.time() - t0 > wall_budget_s
            if left <= 1 or over:
                parts.append({"type": "text", "text": "This is your last turn: reply with {\"answer\": {...}} now."})
            else:
                parts.append({"type": "text", "text": f"({left} turns left including the answer.) Reply with ONE JSON object, "
                                                      "starting with '{' (plain text, no markup)."})
            messages.append({"role": "user", "content": parts})
        else:
            stats["stop"] = stats["stop"] or "max_turns"
    finally:
        stats["wall_s"] = round(time.time() - t0, 2)
        if logf:
            logf.write(json.dumps({"event": "end", **{k: v for k, v in stats.items() if k != "answer"}, "answer": stats["answer"]}) + "\n")
            logf.close()
    return stats
