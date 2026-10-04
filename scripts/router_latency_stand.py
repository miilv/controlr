#!/usr/bin/env python3
"""Router latency test stand: per-stage timestamps of omniroute calls (⚠️ live calls, except
`models` / `badmodel`). One JSON line per call on stdout.

Stages: httpx trace (connect / TLS / body sent / response headers — through a local HTTP proxy
"body sent" only means "handed to the proxy"), first stream byte, first content token, end, and the
SSE event marks (chat chunks: keepalive / content / usage; Responses: response.created, deltas, ...).
Note: omniroute holds a Codex stream back until the first text delta, so `response.created` is NOT
when the Codex backend accepted the request (journal/2026-10-04-luna-latency.md).

Cases: models | badmodel (router only, no LLM) | chat_tiny | chat_tiny_nostream | resp_tiny |
chat_turn | resp_turn (replay a recorded turn of --run: all images, as the loop sends it) |
episode (replay turns 0..--turn of --run in ONE session, in order, with the recorded replies — what
the loop sends; adds `t_complete` = the STATUS word, when the loop starts the robot).

  set -a; . ./.env; set +a
  uv run python scripts/router_latency_stand.py resp_tiny --reps 10 --tag local > out.jsonl
  uv run python scripts/router_latency_stand.py chat_turn --run runs/<run> --turn 12 --reps 5
  uv run python scripts/router_latency_stand.py chat_tiny --model claude/claude-sonnet-5-5 --extra '{"reasoning_effort":"low"}'
"""
import argparse, base64, json, os, socket, time
from pathlib import Path
import httpx

from controlr.protocol.grammar import is_complete as _is_complete

ap = argparse.ArgumentParser()
ap.add_argument("case"); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--tag", default="")
ap.add_argument("--model", default="cx/gpt-6-luna"); ap.add_argument("--extra", default='{"reasoning_effort":"none"}')
ap.add_argument("--run", help="run dir for *_turn cases"); ap.add_argument("--turn", type=int, default=12)
ap.add_argument("--fresh-conn", action="store_true", help="new TCP/TLS connection per call")
ap.add_argument("--max-tokens", type=int, default=60)
ap.add_argument("--gap", type=float, default=0.0, help="episode: seconds between turns")
a = ap.parse_args()
BASE = os.environ["OMNIROUTE_BASE_URL"].rstrip("/"); KEY = os.environ["OMNIROUTE_API_KEY"]
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

def turn_messages(turn):
    run = Path(a.run); recs = [json.loads(l) for l in open(run / "messages.jsonl")]
    out = []
    for r in recs[: 1 + 2 * turn + 1]:
        if isinstance(r["content"], str): out.append(dict(r)); continue
        parts = []
        for p in r["content"]:
            if p["type"] == "image":
                b = (run / "images" / f"{p['image_sha']}.jpg").read_bytes()
                parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}})
            else: parts.append(dict(p))
        out.append({"role": r["role"], "content": parts})
    out[1]["content"][0]["text"] = out[1]["content"][0]["text"].replace("RUN ", f"RUN st{time.time_ns()} ", 1)
    return out

def to_responses_input(msgs):
    """chat messages -> Responses API (instructions + input items)."""
    instr = msgs[0]["content"] if msgs[0]["role"] == "system" else None
    items = []
    for m in msgs[1:] if instr is not None else msgs:
        if isinstance(m["content"], str):
            parts = [{"type": "input_text" if m["role"] == "user" else "output_text", "text": m["content"]}]
        else:
            parts = []
            for p in m["content"]:
                if p["type"] == "text": parts.append({"type": "input_text" if m["role"] == "user" else "output_text", "text": p["text"]})
                else: parts.append({"type": "input_image", "image_url": p["image_url"]["url"]})
        items.append({"role": m["role"], "content": parts})
    return instr, items

client = None
def get_client():
    global client
    if client is None or a.fresh_conn:
        if client is not None: client.close()
        client = httpx.Client(timeout=120)
    return client

def call(method, url, body=None, stream=True):
    ev = {}; t0 = time.perf_counter()
    def trace(name, info):
        k = name.replace("http11.", "").replace("connection.", "")
        if k not in ev: ev[k] = round(time.perf_counter() - t0, 3)
    c = get_client()
    data = None if body is None else json.dumps(body).encode()
    req = c.build_request(method, url, headers=H, content=data, extensions={"trace": trace})
    r = c.send(req, stream=True)
    out = {"status": r.status_code, "bytes_up": len(data or b""), "phases": ev, "t_headers": round(time.perf_counter() - t0, 3)}
    marks = []; first_any = None; first_content = None; text = ""; usage = None; n = 0; t_status = None
    for line in r.iter_lines():
        if not line.strip(): continue
        now = round(time.perf_counter() - t0, 3); n += 1
        if first_any is None: first_any = now
        if line.startswith("event:"):
            marks.append((now, line[6:].strip())); continue
        if not line.startswith("data:"):
            if len(marks) < 3: marks.append((now, line[:60]))
            continue
        d = line[5:].strip()
        if d == "[DONE]": marks.append((now, "[DONE]")); continue
        try: j = json.loads(d)
        except Exception: marks.append((now, d[:60])); continue
        typ = j.get("type")
        if typ:  # responses event
            if typ not in [m[1] for m in marks]: marks.append((now, typ))
            if typ == "response.output_text.delta" and first_content is None: first_content = now
            if typ == "response.output_text.delta":
                text += j.get("delta", "")
                if t_status is None and _is_complete(text): t_status = now
            if typ == "response.completed": usage = j.get("response", {}).get("usage")
        else:  # chat chunk
            ch = (j.get("choices") or [{}])[0]; delta = ch.get("delta") or {}
            tag = "keepalive" if j.get("model") == "keepalive" else ("content" if delta.get("content") else ("reasoning" if any(delta.get(k) for k in ("reasoning_content", "reasoning")) else ("usage" if j.get("usage") else "chunk")))
            if tag not in [m[1] for m in marks]: marks.append((now, tag))
            if tag == "content" and first_content is None: first_content = now
            text += delta.get("content") or ""
            if t_status is None and _is_complete(text): t_status = now
            if j.get("usage"): usage = j["usage"]
    if not stream and n == 1:
        pass
    out.update({"t_first_byte": first_any, "t_first_content": first_content, "t_complete": t_status, "t_end": round(time.perf_counter() - t0, 3),
                "marks": marks[:14], "text": text[:60], "usage": usage, "hdr_provider": r.headers.get("x-omniroute-provider"),
                "server_timing": r.headers.get("server-timing")})
    r.close()
    return out

if a.case == "episode":
    from controlr.protocol.grammar import is_complete
    run_nonce = f"ep{time.time_ns()}"
    recs_all = [json.loads(l) for l in open(Path(a.run) / "messages.jsonl")]
    for rep in range(a.reps):
        run_nonce = f"ep{time.time_ns()}"
        for k in range(a.turn + 1):
            msgs = turn_messages(k)
            msgs[1]["content"][0]["text"] = msgs[1]["content"][0]["text"].replace("RUN st", f"RUN {run_nonce} st", 1)
            # same nonce for the whole episode (turn_messages adds a fresh one; strip it)
            t1 = msgs[1]["content"][0]["text"]
            import re as _re
            msgs[1]["content"][0]["text"] = _re.sub(r"RUN (ep\d+) st\d+ ", r"RUN \1 ", t1, count=1)
            body = {"model": a.model, "messages": msgs, "max_tokens": 2000, "stream": True,
                    "stream_options": {"include_usage": True}, **json.loads(a.extra)}
            res = call("POST", BASE + "/chat/completions", body)
            # time of the STATUS word: re-stream marks are coarse; recompute from text arrival is not
            # possible here, so call() records it via a hook below
            res.update({"case": "episode", "tag": a.tag, "host": socket.gethostname()[:10], "model": a.model,
                        "rep": rep, "turn": k})
            print(json.dumps(res), flush=True)
            if a.gap:
                time.sleep(a.gap)
    raise SystemExit(0)

for rep in range(a.reps):
    m = a.model; extra = json.loads(a.extra)
    if a.case == "models":
        res = call("GET", BASE + "/models")
    elif a.case == "badmodel":
        res = call("POST", BASE + "/chat/completions", {"model": "cx/does-not-exist-xyz", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    elif a.case in ("chat_tiny", "chat_tiny_nostream"):
        st = a.case == "chat_tiny"
        body = {"model": m, "messages": [{"role": "user", "content": f"Reply with exactly: OK ({time.time_ns()})"}], "max_tokens": a.max_tokens, "stream": st, **extra}
        if st: body["stream_options"] = {"include_usage": True}
        res = call("POST", BASE + "/chat/completions", body)
    elif a.case == "chat_turn":
        body = {"model": m, "messages": turn_messages(a.turn), "max_tokens": 2000, "stream": True, "stream_options": {"include_usage": True}, **extra}
        res = call("POST", BASE + "/chat/completions", body)
    elif a.case in ("resp_tiny", "resp_turn"):
        if a.case == "resp_tiny":
            instr, items = None, f"Reply with exactly: OK ({time.time_ns()})"
        else:
            instr, items = to_responses_input(turn_messages(a.turn))
        body = {"model": m, "input": items, "stream": True}
        if instr: body["instructions"] = instr
        if "reasoning_effort" in extra: body["reasoning"] = {"effort": extra["reasoning_effort"]}
        if "service_tier" in extra: body["service_tier"] = extra["service_tier"]
        res = call("POST", BASE + "/responses", body)
    else:
        raise SystemExit("bad case")
    res.update({"case": a.case, "tag": a.tag, "host": socket.gethostname()[:10], "model": m, "rep": rep, "fresh_conn": a.fresh_conn})
    print(json.dumps(res), flush=True)
