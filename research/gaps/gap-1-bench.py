# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic", "openai", "google-genai", "pillow"]
# ///
"""Gap 1 latency/cost benchmark for robot-harness backbones (UNTESTED against live APIs).

Status: written 2026-10-02 for research/gaps/gap-1.md, byte-compiled only. Not run against any API.
It spends money. Run it only after the owner approves the budget printed by `--dry-run`.

Request shape (one "robot turn"):
  * a cached prefix: system text padded to ~10k tokens plus 8 strict tools
  * one 1920x480 JPEG tile per user turn (made from --frames-dir images, or synthetic noise)
  * a short proprioception text block
  * a tool call is required by the prompt, but tool_choice stays "auto" (Opus 5.5 rejects any/tool)
Each rollout is a live multi-turn conversation of --depth turns. History is append-only:
genuine assistant turns (with thinking or reasoning items) are replayed unchanged, and a
canned tool_result is appended. Latency is therefore measured at every depth from 1 to --depth,
and the analysis buckets depths 1-5, 16-20 and 31-35.

Per call it logs (JSONL): t_first_event, t_first_block, t_tool_start, t_total, status/error,
n_tool_calls, input/cached/cache-write/output/thinking tokens, cost_usd, request_bytes.

Usage:
  uv run gap-1-bench.py --dry-run
  uv run gap-1-bench.py --configs opus55-low,opus55-medium --rollouts 10 --depth 35 --out bench.jsonl
"""
from __future__ import annotations

import argparse, base64, io, json, os, random, sys, time

# $/MTok: (uncached input, cache write 5m, cache read, output). Sources are in gap-1.md.
PRICES = {
    "claude-opus-5-5": (4.0, 5.0, 0.20, 20.0),
    "claude-opus-5-5:fast": (8.0, 10.0, 0.40, 40.0),
    "claude-sonnet-5-5": (2.0, 2.5, 0.20, 10.0),  # cache multipliers assumed 1.25x / 0.1x: UNVERIFIED
    "claude-haiku-4-5": (1.0, 1.25, 0.10, 5.0),
    "gpt-6-astra": (10.0, 12.5, 1.0, 50.0),
    "gpt-6-astra:ultrafast": (60.0, 75.0, 6.0, 300.0),
    "gemini-robotics-er-2-preview": (1.0, 1.0, 0.10, 5.0),  # 2026 promo price; doubles 2027-01-01
    "gemini-3.8-flash": (0.75, 0.75, 0.075, 3.75),
}

# name -> (provider, model, options)
CONFIGS = {
    "opus55-low": ("anthropic", "claude-opus-5-5", {"effort": "low"}),
    "opus55-medium": ("anthropic", "claude-opus-5-5", {"effort": "medium"}),
    "opus55-high": ("anthropic", "claude-opus-5-5", {"effort": "high"}),
    # starts at medium, then drops to low through an effort-only system message after turn 1
    "opus55-permsg-low": ("anthropic", "claude-opus-5-5", {"effort": "medium", "per_message_effort": "low"}),
    "opus55-medium-fast": ("anthropic", "claude-opus-5-5", {"effort": "medium", "speed": "fast"}),
    "sonnet55-between-low": ("anthropic", "claude-sonnet-5-5", {"effort": "low", "thinking": {"type": "between_tools"}}),
    "haiku45": ("anthropic", "claude-haiku-4-5", {}),
    "astra-low": ("openai", "gpt-6-astra", {"effort": "low"}),
    "astra-medium": ("openai", "gpt-6-astra", {"effort": "medium"}),
    "astra-low-ultrafast": ("openai", "gpt-6-astra", {"effort": "low", "service_tier": "ultrafast"}),
    "er2-point-low-k1": ("gemini_point", "gemini-robotics-er-2-preview", {"thinking_level": "low", "k": 1}),
    "er2-point-medium-k1": ("gemini_point", "gemini-robotics-er-2-preview", {"thinking_level": "medium", "k": 1}),
    "er2-point-low-k3": ("gemini_point", "gemini-robotics-er-2-preview", {"thinking_level": "low", "k": 3}),
    "gemini38flash-low": ("gemini", "gemini-3.8-flash", {"thinking_level": "low"}),
}

SYSTEM_HEAD = (
    "You control a bimanual tabletop robot through tools. Each user turn has one 1920x480 image tile "
    "(top, left wrist, right wrist views side by side) and the arm state. Respond with exactly one tool "
    "call per turn. Coordinates are metres in the left arm base frame.\n\n# Rig facts\n"
)


def tools_json_schema():
    num = {"type": "number"}
    vec3 = {"type": "array", "items": num, "description": "exactly 3 numbers"}
    px = {"type": "array", "items": {"type": "integer"}, "description": "exactly 2 integers [x, y]"}
    def t(name, desc, props):
        return {"name": name, "description": desc, "strict": True,
                "input_schema": {"type": "object", "properties": props, "required": list(props),
                                 "additionalProperties": False}}
    return [
        t("move_to", "Move one arm's grasp point to xyz_m with yaw.", {"arm": {"type": "string", "enum": ["left", "right"]}, "xyz_m": vec3, "yaw_rad": num, "note": {"type": "string"}}),
        t("grasp", "Close the gripper of one arm.", {"arm": {"type": "string", "enum": ["left", "right"]}, "note": {"type": "string"}}),
        t("release", "Open the gripper of one arm.", {"arm": {"type": "string", "enum": ["left", "right"]}, "note": {"type": "string"}}),
        t("look", "Return a zoomed crop of one camera.", {"camera": {"type": "string", "enum": ["top", "left", "right"]}, "center_px": px, "note": {"type": "string"}}),
        t("point", "Mark a target pixel for the skill layer.", {"camera": {"type": "string", "enum": ["top", "left", "right"]}, "center_px": px, "label": {"type": "string"}}),
        t("wait", "Wait without moving.", {"seconds": num, "note": {"type": "string"}}),
        t("done", "Declare the task complete.", {"evidence": {"type": "string"}}),
        t("give_up", "Stop because the task cannot be completed.", {"reason": {"type": "string"}}),
    ]


def system_text(target_tokens: int) -> str:
    rng = random.Random(0)
    lines, n = [SYSTEM_HEAD], 0
    while n < target_tokens * 4:  # ~4 chars per token; check with count_tokens before a real run
        a, b = rng.randint(1, 40), rng.randint(1, 40)
        line = f"- Calibrated fact {a}-{b}: offset {rng.uniform(-0.05, 0.05):+.4f} m, sag {rng.uniform(0, 0.3):.3f} rad at reach {rng.uniform(0.2, 0.7):.2f} m.\n"
        lines.append(line); n += len(line)
    return "".join(lines)


def make_tile(frames: list[str], i: int) -> bytes:
    from PIL import Image
    if frames:
        im = Image.open(frames[i % len(frames)]).convert("RGB")
    else:
        rng = random.Random(i)
        im = Image.effect_noise((480, 480), 64).convert("RGB")
        im = Image.merge("RGB", [im.getchannel(0).point(lambda v: (v + rng.randint(0, 80)) % 256)] * 3)
    w = 1920; im = im.resize((w // 3, 480))
    tile = Image.new("RGB", (w, 480))
    for k in range(3):
        tile.paste(im, (k * w // 3, 0))
    buf = io.BytesIO(); tile.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def state_text(i: int) -> str:
    rng = random.Random(1000 + i)
    v = [round(rng.uniform(-1, 1), 4) for _ in range(14)]
    return f"Turn {i}. state[eef]: {v}. Last motion: executed over {rng.randint(3, 40)} steps. Respond with exactly one tool call."


def cost(price, unc, cw, cr, out):
    return (unc * price[0] + cw * price[1] + cr * price[2] + out * price[3]) / 1e6


# ---------------------------------------------------------------- Anthropic
def run_anthropic(model, opt, sysmsg, tile_fn, depth, log):
    import anthropic
    client = anthropic.Anthropic(max_retries=0, timeout=180)
    betas = []
    if opt.get("speed") == "fast": betas.append("fast-mode-2026-02-01")
    if opt.get("per_message_effort"): betas.append("mid-conversation-output-config-2026-07-01")
    tools = tools_json_schema()
    msgs = []
    pk = f"{model}:fast" if opt.get("speed") == "fast" else model
    for d in range(1, depth + 1):
        img = base64.b64encode(tile_fn(d)).decode()
        user = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}},
                {"type": "text", "text": state_text(d)}]
        if msgs and msgs[-1]["role"] == "assistant":
            tid = [b for b in msgs[-1]["content"] if b.get("type") == "tool_use"]
            user = [{"type": "tool_result", "tool_use_id": t["id"], "content": "ok"} for t in tid] + user
        msgs.append({"role": "user", "content": user})
        if d == 2 and opt.get("per_message_effort"):
            msgs.insert(len(msgs) - 1, {"role": "system", "content": [], "output_config": {"effort": opt["per_message_effort"]}})
        kw = dict(model=model, max_tokens=16000, system=[{"type": "text", "text": sysmsg, "cache_control": {"type": "ephemeral"}}],
                  tools=tools, messages=msgs, cache_control={"type": "ephemeral"})
        if "claude-haiku" not in model:
            kw["output_config"] = {"effort": opt.get("effort", "medium")}
            kw["thinking"] = opt.get("thinking", {"type": "adaptive"})
        if opt.get("speed"): kw["speed"] = opt["speed"]
        rec = {"depth": d, "request_bytes": len(json.dumps(kw["messages"]))}
        t0 = time.perf_counter(); rec.update(t_first_event=None, t_first_block=None, t_tool_start=None)
        try:
            api = client.beta.messages if betas else client.messages
            extra = {"betas": betas} if betas else {}
            with api.stream(**kw, **extra) as s:
                for ev in s:
                    now = time.perf_counter() - t0
                    if rec["t_first_event"] is None: rec["t_first_event"] = now
                    if ev.type == "content_block_start":
                        if rec["t_first_block"] is None: rec["t_first_block"] = now
                        if ev.content_block.type == "tool_use" and rec["t_tool_start"] is None: rec["t_tool_start"] = now
                final = s.get_final_message()
            rec["t_total"] = time.perf_counter() - t0
            u = final.usage
            unc, cw, cr, out = u.input_tokens, u.cache_creation_input_tokens or 0, u.cache_read_input_tokens or 0, u.output_tokens
            rec.update(status=200, stop_reason=final.stop_reason, input=unc, cache_write=cw, cache_read=cr, output=out,
                       n_tool_calls=sum(b.type == "tool_use" for b in final.content), cost_usd=cost(PRICES[pk], unc, cw, cr, out))
            msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in final.content]})
        except Exception as e:  # log and end the rollout; a 400 here usually means the request shape is wrong
            rec.update(status=getattr(e, "status_code", None), error=repr(e)[:300], t_total=time.perf_counter() - t0)
            log(rec); return
        log(rec)
        if rec["n_tool_calls"] == 0:  # mirror Inspect Robots: nudge once, count it
            msgs.append({"role": "user", "content": [{"type": "text", "text": "Respond with exactly one tool call."}]})


# ---------------------------------------------------------------- OpenAI Responses
def run_openai(model, opt, sysmsg, tile_fn, depth, log):
    from openai import OpenAI
    client = OpenAI(max_retries=0, timeout=180)
    tools = [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["input_schema"], "strict": True}
             for t in tools_json_schema()]
    items, pending = [], []
    pk = f"{model}:{opt['service_tier']}" if opt.get("service_tier") else model
    for d in range(1, depth + 1):
        img = base64.b64encode(tile_fn(d)).decode()
        items += [{"type": "function_call_output", "call_id": c, "output": "ok"} for c in pending]
        items.append({"role": "user", "content": [{"type": "input_image", "image_url": f"data:image/jpeg;base64,{img}"},
                                                  {"type": "input_text", "text": state_text(d)}]})
        kw = dict(model=model, instructions=sysmsg, input=items, tools=tools, tool_choice="auto", store=False,
                  include=["reasoning.encrypted_content"], reasoning={"effort": opt.get("effort", "medium")})
        if opt.get("service_tier"): kw["service_tier"] = opt["service_tier"]
        rec = {"depth": d, "request_bytes": len(json.dumps(items)), "t_first_event": None, "t_first_block": None, "t_tool_start": None}
        t0 = time.perf_counter()
        try:
            final = None
            for ev in client.responses.create(stream=True, **kw):
                now = time.perf_counter() - t0
                if rec["t_first_event"] is None: rec["t_first_event"] = now
                if ev.type == "response.output_item.added":
                    if rec["t_first_block"] is None: rec["t_first_block"] = now
                    if ev.item.type == "function_call" and rec["t_tool_start"] is None: rec["t_tool_start"] = now
                if ev.type == "response.completed": final = ev.response
            rec["t_total"] = time.perf_counter() - t0
            u = final.usage
            cr = (u.input_tokens_details.cached_tokens or 0) if u.input_tokens_details else 0
            out = u.output_tokens
            calls = [o for o in final.output if o.type == "function_call"]
            rec.update(status=200, input=u.input_tokens - cr, cache_write=None, cache_read=cr, output=out,
                       thinking=getattr(u.output_tokens_details, "reasoning_tokens", None), n_tool_calls=len(calls),
                       service_tier=getattr(final, "service_tier", None), cost_usd=cost(PRICES[pk], u.input_tokens - cr, 0, cr, out))
            items += [o.model_dump(exclude_none=True) for o in final.output]
            pending = [c.call_id for c in calls]
        except Exception as e:
            rec.update(status=getattr(e, "status_code", None), error=repr(e)[:300], t_total=time.perf_counter() - t0)
            log(rec); return
        log(rec)


# ---------------------------------------------------------------- Gemini (agent loop and single-point query)
def run_gemini(model, opt, sysmsg, tile_fn, depth, log, point_only=False):
    from google import genai
    from google.genai import types
    client = genai.Client()
    cfg = types.GenerateContentConfig(system_instruction=None if point_only else sysmsg,
                                      thinking_config=types.ThinkingConfig(thinking_level=opt["thinking_level"]))
    hist = []
    k = opt.get("k", 1)
    for d in range(1, depth + 1):
        part_img = types.Part.from_bytes(data=tile_fn(d), mime_type="image/jpeg")
        if point_only:
            contents = [part_img, 'Point to the handle of the mug. Answer as JSON [{"point": [y, x], "label": "handle"}], coordinates normalised to 0-1000.']
        else:
            hist.append(types.Content(role="user", parts=[part_img, types.Part.from_text(text=state_text(d))]))
            contents = hist
        for rep in range(k):
            rec = {"depth": d, "rep": rep, "t_first_event": None, "t_first_block": None, "t_tool_start": None}
            t0 = time.perf_counter()
            try:
                last = None
                for ch in client.models.generate_content_stream(model=model, contents=contents, config=cfg):
                    if rec["t_first_event"] is None: rec["t_first_event"] = time.perf_counter() - t0
                    last = ch
                rec["t_total"] = time.perf_counter() - t0
                um = last.usage_metadata
                cr = um.cached_content_token_count or 0
                out = (um.candidates_token_count or 0) + (um.thoughts_token_count or 0)
                rec.update(status=200, input=(um.prompt_token_count or 0) - cr, cache_read=cr, output=out, thinking=um.thoughts_token_count,
                           cost_usd=cost(PRICES[model], (um.prompt_token_count or 0) - cr, 0, cr, out))
            except Exception as e:
                rec.update(status=getattr(e, "code", None), error=repr(e)[:300], t_total=time.perf_counter() - t0)
                log(rec); return
            log(rec)
        if not point_only:
            hist.append(last.candidates[0].content)  # keeps thought signatures for the next turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--depth", type=int, default=35)
    ap.add_argument("--prefix-tokens", type=int, default=10000)
    ap.add_argument("--frames-dir", default="")
    ap.add_argument("--out", default="gap1-bench.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    names = a.configs.split(",")
    if a.dry_run:
        # rough budget: per call, prefix plus history read from cache and ~2k new tokens written; Tier 1 output/call as a guide
        out_guess = {"claude-opus-5-5": 424, "gpt-6-astra": 134}
        for n in names:
            prov, model, opt = CONFIGS[n]
            calls = a.rollouts * (a.depth if prov != "gemini_point" else min(a.depth, 5) * opt.get("k", 1))
            pk = model + (":fast" if opt.get("speed") == "fast" else "") + (":ultrafast" if opt.get("service_tier") else "")
            hist = a.prefix_tokens + 2000 * a.depth / 2 if prov != "gemini_point" else 1300
            est = calls * cost(PRICES[pk], 0, 2000, hist, out_guess.get(model, 400))
            print(f"{n:24s} calls={calls:5d} est_cost=${est:7.2f}")
        return
    frames = sorted(os.path.join(a.frames_dir, f) for f in os.listdir(a.frames_dir)) if a.frames_dir else []
    tiles: dict[int, bytes] = {}
    tile_fn = lambda i: tiles.setdefault(i, make_tile(frames, i))
    sysmsg = system_text(a.prefix_tokens)
    with open(a.out, "a") as fh:
        for n in names:
            prov, model, opt = CONFIGS[n]
            for r in range(a.rollouts):
                def log(rec, n=n, r=r):
                    rec.update(config=n, model=model, rollout=r, ts=time.time())
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    print(n, r, rec.get("depth"), rec.get("status"), round(rec.get("t_total", 0), 2), file=sys.stderr)
                if prov == "anthropic": run_anthropic(model, opt, sysmsg, tile_fn, a.depth, log)
                elif prov == "openai": run_openai(model, opt, sysmsg, tile_fn, a.depth, log)
                elif prov == "gemini": run_gemini(model, opt, sysmsg, tile_fn, a.depth, log)
                else: run_gemini(model, opt, sysmsg, tile_fn, min(a.depth, 5), log, point_only=True)


if __name__ == "__main__":
    main()
