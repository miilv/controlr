"""Router smoke test for PyRUA-Lean's request format through omniroute (stdlib only).

Sends the body shape PyRUA-Lean's openai runtime sends (openai_runtime.py `chat()`): messages
with an image_url PNG data URI, `tools` = python/finish function specs, tool_choice=auto,
max_tokens, plus extra_body fields merged at the top level. Prints status, latency, usage and the
tool call; never prints credentials.

  set -a; . /root/controlr/.env; set +a
  python3 router_smoke.py --model cx/gpt-6-astra --effort high [--user NONCE] [--n 1]
"""
import argparse, base64, json, os, struct, time, urllib.request, urllib.error, zlib


def png(w=64, h=48):
    # a red square on white: the model must say "red" in its tool call
    rows = []
    for y in range(h):
        row = b"\x00"
        for x in range(w):
            row += b"\xff\x00\x00" if 16 <= x < 48 and 12 <= y < 36 else b"\xff\xff\xff"
        rows.append(row)
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


TOOLS = [
    {"type": "function", "function": {"name": "python", "description": "Run a Python cell.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "finish", "description": "End the episode.",
     "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"]},
                    "summary": {"type": "string"}}, "required": ["status", "summary"]}}},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cx/gpt-6-astra")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--user", default=None)
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    url = os.environ["OMNIROUTE_BASE_URL"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    img = "data:image/png;base64," + base64.b64encode(png()).decode()
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Look at the attached image. Call the python tool with code that prints "
                                     "the colour of the square, e.g. print('blue'). Do not answer in text."},
            {"type": "image_url", "image_url": {"url": img}}]}],
        "tools": TOOLS, "tool_choice": "auto", "max_tokens": 8192,
    }
    if a.effort != "none-field":
        body["reasoning_effort"] = a.effort
    if a.user:
        body["user"] = a.user
    for i in range(a.n):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OMNIROUTE_API_KEY"]})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                code, payload = r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            code, payload = e.code, e.read().decode()[:600]
        dt = time.time() - t0
        if code != 200:
            print(json.dumps({"i": i, "status": code, "latency_s": round(dt, 2), "error": payload}))
            continue
        msg = payload["choices"][0]["message"]
        print(json.dumps({"i": i, "status": code, "latency_s": round(dt, 2), "model": payload.get("model"),
                          "finish_reason": payload["choices"][0].get("finish_reason"),
                          "usage": payload.get("usage"), "content": (msg.get("content") or "")[:200],
                          "tool_calls": [(t["function"]["name"], t["function"]["arguments"][:200])
                                         for t in (msg.get("tool_calls") or [])]}))


if __name__ == "__main__":
    main()
