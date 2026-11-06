"""Teacher-forced route probe: replay stored dev requests, never tools or a robot.

Only rows named *-100000..04 / *-100030..49 are accepted. Requires original prompt
hash, untruncated tool output, immutable PNGs and matching historical request size.
Default is dry-run; --live permits ONLY cx/ routes, with zero automatic retries.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from controlr.keyguard import redact
from dojo.perception.agentic.llm import OmniChat, image_part
from dojo.perception.agentic.prompt import pin, system_prompt
from dojo.perception.agentic.runner import parse_reply


def requests(ws: Path):
    events = [json.loads(s) for s in (ws / "transcript.jsonl").read_text().splitlines()]
    start = events[0]
    assert start["model"] == "claude/claude-sonnet-5-5", "unexpected baseline"
    calls = [json.loads(s) for s in (ws.parent / "calls.jsonl").read_text().splitlines()]
    call = next(c for c in calls if Path(c.get("ws", "")).name == ws.name)
    kind = call["pin"]["kind"]
    assert pin(kind, start["model"])["system_md5"] == call["pin"]["system_md5"], "prompt changed"
    first = []
    for part in start["first_user"]:
        if part == "<image>":
            first.append(image_part((ws / first[-1]["text"].removesuffix(":")).read_bytes()))
        else:
            first.append({"type": "text", "text": part})
    messages = [{"role": "system", "content": system_prompt(kind)}, {"role": "user", "content": first}]
    pending = []
    tool_images = set()
    max_turns = 8  # pinned candidate default; byte-size verification rejects other schedules
    for event in events[1:]:
        if event["event"] == "tool":
            assert len(event["out"]) < 2000, "truncated tool output"
            i = sum(p["type"] == "text" and p["text"].startswith("[") for p in pending) + 1
            pending.append({"type": "text", "text": f'[{i}] $ {event["cmd"]}\n{event["out"]}'})
            for name in event["images"]:
                assert name not in tool_images, "tool image overwritten"
                tool_images.add(name)
                assert sum(p["type"] == "image_url" for p in pending) < 6, "image cap ambiguity"
                pending.extend([{"type": "text", "text": f"image {name}:"}, image_part((ws / name).read_bytes())])
        elif event["event"] == "reply":
            turn = event["turn"]
            if pending:
                left = max_turns - turn
                pending.append({"type": "text", "text": 'This is your last turn: reply with {"answer": {...}} now.' if left <= 1 else
                                f"({left} turns left including the answer.) Reply with ONE JSON object, starting with '{{' (plain text, no markup)."})
                messages.append({"role": "user", "content": pending})
                pending = []
            body = {"model": start["model"], "messages": messages, "max_tokens": 6000, "stream": True,
                    "stream_options": {"include_usage": True}}
            wire = json.dumps(body).encode()
            assert len(wire) == event["request_bytes"], f"request size mismatch at turn {turn}"
            yield copy.deepcopy(messages), event, hashlib.sha256(wire).hexdigest()
            if event.get("error") and not event["text"]:
                continue
            messages.append({"role": "assistant", "content": event["text"] if event["text"].strip() else "(empty reply)"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", default="cx/gpt-6-sol")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.model.startswith("cx/") or not 1 <= args.limit <= 200:
        ap.error("only cx/ routes and 1..200 calls permitted")
    selected = []
    for row in sorted(args.root.iterdir()):
        m = re.fullmatch(r".+-(100\d{3})", row.name)
        if not m or int(m[1]) not in {*range(100000, 100005), *range(100030, 100050)}:
            continue
        for ws in sorted(row.glob("call_*/*/transcript.jsonl")):
            try:
                # Validate ALL preceding tool outputs before selecting the final answer request.
                reqs = list(requests(ws.parent))
                finals = [r for r in reqs if "answer" in (parse_reply(r[1]["text"]) or {})]
                if finals:
                    selected.append((ws, finals[-1]))
            except (AssertionError, KeyError, StopIteration, FileNotFoundError) as e:
                print(f"skip {ws}: {e}")
    chat = OmniChat(max_retries=0) if args.live else None
    try:
        with args.out.open("w") as out:
            for ws, (msgs, old, sha) in selected[:args.limit]:
                rec = {"source": str(ws), "turn": old["turn"], "request_sha256": sha, "source_model": "claude/claude-sonnet-5-5",
                       "model": args.model, "old": old, "n_images": sum(p.get("type") == "image_url" for m in msgs
                        if isinstance(m["content"], list) for p in m["content"])}
                if chat:
                    rep = chat(args.model, msgs, max_tokens=6000)
                    rec["new"] = vars(rep)
                out.write(redact(json.dumps(rec), patterns=False) + "\n")
                out.flush()
                print(f'{ws.parents[2].name} turn={old["turn"]} images={rec["n_images"]} live={bool(chat)}', flush=True)
    finally:
        if chat:
            chat.close()


if __name__ == "__main__":
    main()
