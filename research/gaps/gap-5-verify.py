"""Run independent success verifiers over the episode corpus.

Usage: verify.py CONDITION [id-prefix-filter]
CONDITION in CONDS below. Results appended to results/<COND>.jsonl (resumable).
"""
import base64, json, os, re, sys, time, glob, random, urllib.request, concurrent.futures as cf, threading

BASE = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
TOK = os.environ["ANTHROPIC_AUTH_TOKEN"]
W = "/dev/shm/gap5"
os.makedirs(f"{W}/results", exist_ok=True)

# name: (api, model, images, effort)
CONDS = {
    "o55_cs_low": ("anthropic", "claude/claude-opus-5-5", ["CS"], "low"),
    "o55_ef_low": ("anthropic", "claude/claude-opus-5-5", ["EF"], "low"),
    "o55_csef_low": ("anthropic", "claude/claude-opus-5-5", ["CS", "EF"], "low"),
    "o55_csef_high": ("anthropic", "claude/claude-opus-5-5", ["CS", "EF"], "high"),
    "astra_csef_low": ("openai", "cx/gpt-6-astra", ["CS", "EF"], "low"),
    "sol_csef_low": ("openai", "cx/gpt-5.6-sol-low", ["CS", "EF"], None),
}
PRICE = {"claude/claude-opus-5-5": (4.0, 20.0), "cx/gpt-6-astra": (10.0, 50.0), "cx/gpt-5.6-sol-low": (None, None)}

SYSTEM = """You are an independent success verifier for a robot manipulation episode. You did not control the robot. You do not see the controlling agent's reasoning or its claims, and you must not assume the agent succeeded.

Judge failure by what you can see. Do not infer success from the task description. An object that is near, touching, or resting on the edge of its target is not "in" or "on" it. A held object is not a placed object. If the outcome depends on a region you cannot actually see in the images (occluded, out of frame, too small to resolve), say so and answer "uncertain" rather than guessing.

Return only a JSON object, no other text."""

DESC = {
    "CS": "IMAGE {k}: a contact sheet of 20 frames sampled evenly across the whole episode, left to right then top to bottom. Each tile shows three synchronized cameras side by side ({views}). The yellow stamp under each tile is the episode time; the last tile (marked FINAL) is the final state.",
    "EF": "IMAGE {k}: the final frame of the episode at native resolution, three synchronized cameras side by side ({views}).",
}


def build_prompt(meta, imgs):
    lines = [f"TASK INSTRUCTION GIVEN TO THE ROBOT: {meta['instruction']}",
             f"SUCCESS CRITERION (must hold in the final state): {meta['success_criterion']}"]
    if meta.get("rubric"):
        lines.append("PROGRESS RUBRIC (award the highest level whose condition holds at the end):")
        for s, dsc in meta["rubric"]:
            lines.append(f"  {s}: {dsc}")
        prog = 'one of the rubric scores above'
    elif meta["corpus"] == "robocurve_astra_report":
        lines.append("PROGRESS STAGES: 0 no purposeful approach; 1 made contact with the object; 2 lifted the object clear of the table; 3 positioned it above the deposit point; 4 placed it in its final position (success).")
        prog = 'the highest stage 0-4 reached'
    else:
        prog = 'estimated fraction of the task completed, 0.0-1.0'
    for k, im in enumerate(imgs, 1):
        lines.append(DESC[im].format(k=k, views=meta["views"]))
    lines.append(f"""Respond with JSON exactly of this form:
{{"final_state": "<what the final images show about the objects named in the success criterion>",
 "decisive_region_visible": true or false,
 "verdict": "success" | "failure" | "uncertain",
 "p_success": <probability 0.0-1.0 that the success criterion holds in the final state>,
 "progress": <{prog}>}}""")
    return "\n".join(lines)


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def call(cond, meta):
    api, model, imgs, effort = CONDS[cond]
    d = f"{W}/frames/{meta['id']}"
    text = build_prompt(meta, imgs)
    t0 = time.time()
    if api == "anthropic":
        content = []
        for im in imgs:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(f"{d}/{im}.jpg")}})
        content.append({"type": "text", "text": text})
        body = {"model": model, "max_tokens": 16000, "system": SYSTEM, "output_config": {"effort": effort},
                "messages": [{"role": "user", "content": content}]}
        url = BASE + "/messages"
        hdr = {"content-type": "application/json", "x-api-key": TOK, "authorization": "Bearer " + TOK, "anthropic-version": "2023-06-01"}
    else:
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(f"{d}/{im}.jpg")}} for im in imgs]
        content.append({"type": "text", "text": text})
        body = {"model": model, "max_tokens": 16000, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
        if effort:
            body["reasoning_effort"] = effort
        url = BASE + "/chat/completions"
        hdr = {"content-type": "application/json", "authorization": "Bearer " + TOK}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=hdr, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        js = json.loads(r.read())
    lat = time.time() - t0
    if api == "anthropic":
        txt = "".join(b.get("text", "") for b in js.get("content", []) if b.get("type") == "text")
        u = js.get("usage", {})
        tin, tout = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0), u.get("output_tokens", 0)
        think = (u.get("output_tokens_details") or {}).get("thinking_tokens")
        stop = js.get("stop_reason")
    else:
        txt = js["choices"][0]["message"].get("content") or ""
        u = js.get("usage", {})
        tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        think = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        stop = js["choices"][0].get("finish_reason")
    m = re.search(r"\{.*\}", txt, re.S)
    parsed = None
    if m:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            parsed = None
    pin, pout = PRICE.get(model, (None, None))
    cost = (tin * pin + tout * pout) / 1e6 if pin else None
    return {"id": meta["id"], "cond": cond, "model_returned": js.get("model"), "latency_s": round(lat, 2), "in_tok": tin, "out_tok": tout,
            "thinking_tok": think, "stop": stop, "cost_usd": cost, "parsed": parsed, "raw": txt[-3000:]}


lock = threading.Lock()


def main():
    cond = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    nt = int(os.environ.get("NT", "6"))
    out = f"{W}/results/{cond}.jsonl"
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            if r.get("parsed") is not None:
                done.add(r["id"])
    metas = []
    for p in sorted(glob.glob(f"{W}/frames/*/meta.json")):
        m = json.load(open(p))
        if m["id"] in done or not m["id"].startswith(tuple(filt.split(","))):
            continue
        metas.append(m)
    random.Random(0).shuffle(metas)
    print(cond, "todo", len(metas), flush=True)

    def work(m):
        for attempt in range(4):
            try:
                r = call(cond, m)
                r["attempt"] = attempt
                with lock:
                    open(out, "a").write(json.dumps(r) + "\n")
                if r["parsed"] is not None:
                    return "ok"
            except Exception as e:
                err = repr(e)[:300]
                with lock:
                    open(f"{W}/results/{cond}.err", "a").write(json.dumps({"id": m["id"], "attempt": attempt, "err": err}) + "\n")
                time.sleep(5 * (attempt + 1))
        return "fail"

    with cf.ThreadPoolExecutor(nt) as ex:
        res = list(ex.map(work, metas))
    print(cond, "ok", res.count("ok"), "fail", res.count("fail"), flush=True)


if __name__ == "__main__":
    main()
