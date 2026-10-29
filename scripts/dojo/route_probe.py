"""Pre-flight probe of the agentic perceiver's model route, before a scored block (robot-harness D-C19 tranche A).

ONE tiny chat call through the same client the perceiver uses (``dojo.perception.agentic.llm.make_chat``: omniroute,
the key from the file named by AGENTIC_ENV_FILE), on the model it will use (AGENTIC_MODEL, else the code default).
Prints JSON: ok, the (redacted) error if any (a 429 / weekly-limit message is the headroom signal), wall time, tokens.
Never prints a key, a URL or a header. Costs one call (a few tokens).

    AGENTIC_ENV_FILE=~/controlr/.env python scripts/dojo/route_probe.py [--model M] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def default_model() -> str | None:
    src = (ROOT / "dojo/perception/agentic/j1.py").read_text()
    m = re.search(r'os\.environ\.get\("AGENTIC_MODEL", "([^"]+)"\)', src)
    return os.environ.get("AGENTIC_MODEL") or (m.group(1) if m else None)


def probe(model: str, chat=None) -> dict:
    from dojo.perception.agentic.llm import make_chat
    t0 = time.time()
    try:
        chat = chat or make_chat()
        r = chat(model, [{"role": "user", "content": "Reply with the single word OK."}], max_tokens=16)
        err = r.error
        out = {"text_ok": "ok" in (r.text or "").lower(), "finish_reason": r.finish_reason,
               "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens}
    except Exception as e:  # noqa: BLE001 (a probe reports, never raises)
        from controlr import keyguard
        err, out = keyguard.clip(f"{type(e).__name__}: {e}", 300), {}
    limited = bool(err) and bool(re.search(r"\b429\b|rate.?limit|quota|weekly|limit reached", err, re.I))
    return {"model": model, "ok": not err and out.get("text_ok", False), "error": err, "rate_limited": limited,
            "wall_s": round(time.time() - t0, 2), "probed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=None, help="default: AGENTIC_MODEL, else the perceiver's code default")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = probe(a.model or default_model())
    text = json.dumps(res, indent=1, sort_keys=True)
    if a.json:
        Path(a.json).write_text(text + "\n")
    print(text)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
