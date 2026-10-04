#!/usr/bin/env python3
"""Direct-to-Codex latency stand (⚠️ live calls on the ChatGPT subscription of the own login,
scripts/codex_login.py). Replays an episode the way the loop runs it — the manual, the recorded
user turns (feedback + frame) and the MODEL'S OWN replies — straight to chatgpt.com, bypassing
the router, over HTTP (whole conversation every turn) or WebSocket (one connection; from turn 2 on
only the new user turn is sent, with previous_response_id). One JSON line per turn.

  uv run --with websockets python scripts/codex_direct_stand.py --mode ws --run runs/<run> --turns 12
  uv run python scripts/codex_direct_stand.py --mode http --run runs/<run> --turns 12

Protocol from the Codex CLI v0.160.0 source (journal/2026-10-04-turn-latency.md).
"""
import argparse, base64, json, platform, time, uuid
from pathlib import Path

import httpx

from controlr.llm import codex_auth as ca
from controlr.protocol.grammar import is_complete

URL = "https://chatgpt.com/backend-api/codex/responses"
VERSION = "0.160.0"

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["http", "ws"], default="http")
ap.add_argument("--run", required=True); ap.add_argument("--turns", type=int, default=12)
ap.add_argument("--model", default="gpt-6-luna"); ap.add_argument("--effort", default="none")
ap.add_argument("--tier", default="priority", help="'' = no service_tier")
ap.add_argument("--manual-as", choices=["instructions", "developer"], default="instructions")
ap.add_argument("--tag", default="")
ap.add_argument("--prewarm", action="store_true", help="ws: after each reply send generate:false (as if during execution)")
ap.add_argument("--detail", default="", help="input_image detail: low|high|auto (default: unset)")
ap.add_argument("--gap", type=float, default=0.0, help="seconds between turns (robot execution time)")
a = ap.parse_args()

run = Path(a.run)
recs = [json.loads(l) for l in open(run / "messages.jsonl")]
manual = recs[0]["content"]
nonce = f"direct{time.time_ns()}"


def user_item(r, first):
    parts = []
    for p in r["content"]:
        if p["type"] == "image":
            b = (run / "images" / f"{p['image_sha']}.jpg").read_bytes()
            parts.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(b).decode(),
                          **({"detail": a.detail} if a.detail else {})})
        else:
            t = p["text"].replace("RUN ", f"RUN {nonce} ", 1) if first else p["text"]
            parts.append({"type": "input_text", "text": t})
    return {"type": "message", "role": "user", "content": parts}


users = [user_item(r, i == 0) for i, r in enumerate(x for x in recs if x["role"] == "user")]
http = httpx.Client(timeout=120, limits=httpx.Limits(keepalive_expiry=120))
tok = ca.ensure_fresh(http)
sid = str(uuid.uuid4())
H = {"Authorization": f"Bearer {tok.access_token}", "ChatGPT-Account-ID": tok.account_id or "",
     "version": VERSION, "originator": "codex_cli_rs",
     "User-Agent": f"codex_cli_rs/{VERSION} ({platform.system()} {platform.release()}; {platform.machine()}) controlr",
     "session-id": sid, "thread-id": sid, "x-openai-internal-codex-responses-lite": "true"}


def body(inp):
    b = {"model": a.model, "input": inp, "store": False, "stream": True, "parallel_tool_calls": False,
         "reasoning": {"effort": a.effort, "context": "all_turns"}, "prompt_cache_key": sid, "text": {"verbosity": "low"}, "include": []}
    if a.manual_as == "instructions":
        b["instructions"] = manual
    else:
        b["instructions"] = ""
    if a.tier:
        b["service_tier"] = a.tier
    return b


def consume(events, t0):
    """events: iterator of (now, event dict). Returns timing/usage record + output text + response id."""
    out = {"created": None, "first_token": None, "complete": None, "end": None}
    text, usage, rid, err = "", None, None, None
    for now, ev in events:
        typ = ev.get("type", "")
        if typ == "response.created":
            out["created"] = now; rid = (ev.get("response") or {}).get("id")
        elif typ == "response.output_text.delta":
            if out["first_token"] is None:
                out["first_token"] = now
            text += ev.get("delta", "")
            if out["complete"] is None and is_complete(text):
                out["complete"] = now
        elif typ in ("response.completed", "response.done"):
            resp = ev.get("response") or {}
            usage = resp.get("usage"); rid = resp.get("id") or rid; out["end"] = now
            break
        elif typ in ("error", "response.failed") or "error" in typ:
            err = json.dumps(ev)[:300]; out["end"] = now
            break
    return out, text, usage, rid, err


def http_turn(inp):
    t0 = time.perf_counter()
    h = dict(H, **{"x-client-request-id": str(uuid.uuid4()), "Accept": "text/event-stream"})
    with http.stream("POST", URL, json=body(inp), headers=h) as r:
        hdr = time.perf_counter() - t0
        if r.status_code != 200:
            return {"status": r.status_code, "hdr": hdr, "err": r.read().decode()[:300]}, "", None, None

        def evs():
            for line in r.iter_lines():
                if line.startswith("data:"):
                    try:
                        yield round(time.perf_counter() - t0, 3), json.loads(line[5:])
                    except json.JSONDecodeError:
                        pass
        o, text, usage, rid, err = consume(evs(), t0)
    o.update(status=200, hdr=round(hdr, 3), err=err)
    return o, text, usage, rid


ws = None


def ws_turn(payload):
    global ws
    from websockets.sync.client import connect
    t0 = time.perf_counter()
    if ws is None:
        ws = connect(URL.replace("https://", "wss://"),  # noqa

                     additional_headers=dict(H, **{"OpenAI-Beta": "responses_websockets=2026-02-06",
                                                   "x-client-request-id": str(uuid.uuid4())}),
                     max_size=None, open_timeout=30)
    t_conn = time.perf_counter() - t0
    t0 = time.perf_counter()
    ws.send(json.dumps(dict(payload, type="response.create")))

    def evs():
        while True:
            m = ws.recv(timeout=120)
            yield round(time.perf_counter() - t0, 3), json.loads(m)
    o, text, usage, rid, err = consume(evs(), t0)
    o.update(status=200 if not err else "err", connect=round(t_conn, 3), err=err)
    return o, text, usage, rid


history = []
prev_id, prev_len = None, 0
for k in range(min(a.turns, len(users))):
    history.append(users[k])
    if a.mode == "http":
        o, text, usage, rid = http_turn(list(history))
        sent = len(json.dumps(body(history)))
    else:
        b = body(list(history))
        if prev_id is not None:   # only what is new since the last response (its output is server-side)
            b["input"] = history[prev_len:]
            b["previous_response_id"] = prev_id
        sent = len(json.dumps(b))
        o, text, usage, rid = ws_turn(b)
    history.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
    prev_id, prev_len = rid, len(history)
    if o.get("err") and a.mode == "ws":
        # the failed response may still be streaming on this socket: drop the connection and
        # resend the whole conversation on a new one (what the loop would have to do)
        ws.close(); ws = None
        prev_id, prev_len = None, 0
    pw = None
    if a.prewarm and a.mode == "ws" and prev_id:
        # during "execution": let the server ingest everything so far; the next turn only adds the new frame
        tp = time.perf_counter()
        ws.send(json.dumps(dict(body([]), type="response.create", previous_response_id=rid, generate=False)))
        while True:
            ev = json.loads(ws.recv(timeout=60))
            if ev.get("type") in ("response.completed", "response.done", "error", "response.failed"):
                break
        pw = round(time.perf_counter() - tp, 3)
        new_id = (ev.get("response") or {}).get("id")
        if ev.get("type") in ("response.completed", "response.done") and new_id:
            prev_id = new_id
        else:
            pw = f"err {json.dumps(ev)[:200]}"
    if a.gap:
        time.sleep(a.gap)
    cached = ((usage or {}).get("input_tokens_details") or {}).get("cached_tokens")
    print(json.dumps({"tag": a.tag, "mode": a.mode, "turn": k, "bytes_sent": sent, **o,
                      "in": (usage or {}).get("input_tokens"), "cached": cached,
                      "out": (usage or {}).get("output_tokens"),
                      "reas": ((usage or {}).get("output_tokens_details") or {}).get("reasoning_tokens"),
                      "text": text[:70], "prewarm_s": pw if a.prewarm else None}), flush=True)
    if o.get("err") and a.mode == "http":
        break
if ws is not None:
    ws.close()
