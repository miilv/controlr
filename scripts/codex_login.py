#!/usr/bin/env python3
"""Own ChatGPT/Codex login for controlr (device code). Prints a URL and a code; a human with the
ChatGPT account opens the URL, enters the code and approves. The token file goes to
~/.controlr/codex_auth.json (0600; override: CONTROLR_CODEX_AUTH). Never commit or copy it —
one holder only, refresh tokens rotate (controlr/llm/codex_auth.py).

  uv run python scripts/codex_login.py            # waits up to 15 min for the approval
"""
import sys

import httpx

from controlr.llm import codex_auth as ca


def show(url: str, code: str) -> None:
    print(f"Open {url} and enter the code:  {code}\n(waiting up to 15 min ...)", flush=True)


if __name__ == "__main__":
    with httpx.Client(timeout=30) as http:
        t = ca.device_login(http, show)
    print(f"logged in: account {t.account_id}, token file {ca.DEFAULT_PATH} (0600)")
    sys.exit(0)
