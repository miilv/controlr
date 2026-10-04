"""Own ChatGPT login for the Codex backend (chatgpt.com/backend-api/codex), as the
open-source Codex CLI does it (codex-rs/login, v0.160.0; journal/2026-10-04-turn-latency.md).

Why our OWN login: refresh tokens are single-use and rotate (the server answers
``refresh_token_reused`` to the second holder), so sharing the router's login would
break the router for every client. A device-code login gives controlr a separate grant.

The token file (default ``~/.controlr/codex_auth.json``, mode 0600) is a secret: never
print it, never copy it into the repo, runs or logs. Exactly ONE machine may hold and
refresh it — a copy that refreshes invalidates the other.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

AUTH_BASE = "https://auth.openai.com"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"          # the Codex CLI's public OAuth client id
DEVICE_PAGE = AUTH_BASE + "/codex/device"
REDIRECT_URI = AUTH_BASE + "/deviceauth/callback"
DEFAULT_PATH = Path(os.environ.get("CONTROLR_CODEX_AUTH", "~/.controlr/codex_auth.json")).expanduser()
REFRESH_MARGIN_S = 300                               # refresh 5 min before the access token expires


def jwt_claims(token: str) -> dict:
    """Payload of a JWT without verifying it (we only read exp / account id)."""
    try:
        part = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except Exception:  # noqa: BLE001
        return {}


def account_id_from(id_token: str) -> str | None:
    c = jwt_claims(id_token)
    return (c.get("https://api.openai.com/auth") or {}).get("chatgpt_account_id") or c.get("chatgpt_account_id")


@dataclass
class CodexTokens:
    access_token: str
    refresh_token: str
    id_token: str
    account_id: str | None
    last_refresh: float

    def expires_at(self) -> float:
        return float(jwt_claims(self.access_token).get("exp") or 0)

    def __repr__(self) -> str:   # never leak tokens
        return f"CodexTokens(account_id={self.account_id!r}, expires_at={self.expires_at():.0f})"


def save(tokens: CodexTokens, path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"tokens": {"access_token": tokens.access_token, "refresh_token": tokens.refresh_token,
                              "id_token": tokens.id_token, "account_id": tokens.account_id},
                   "last_refresh": tokens.last_refresh}, f)
    os.replace(tmp, path)


def load(path: Path = DEFAULT_PATH) -> CodexTokens:
    d = json.loads(path.read_text())
    t = d["tokens"]
    return CodexTokens(t["access_token"], t["refresh_token"], t["id_token"],
                       t.get("account_id") or account_id_from(t["id_token"]), float(d.get("last_refresh") or 0))


def _tokens_from(resp: dict, old: CodexTokens | None = None) -> CodexTokens:
    id_token = resp.get("id_token") or (old.id_token if old else "")
    return CodexTokens(
        access_token=resp["access_token"],
        refresh_token=resp.get("refresh_token") or (old.refresh_token if old else ""),
        id_token=id_token,
        account_id=account_id_from(id_token) or (old.account_id if old else None),
        last_refresh=time.time())


def refresh(tokens: CodexTokens, http: httpx.Client, path: Path = DEFAULT_PATH) -> CodexTokens:
    """Refresh and persist at once (the old refresh token is dead after this call)."""
    r = http.post(AUTH_BASE + "/oauth/token", json={"grant_type": "refresh_token", "client_id": CLIENT_ID,
                                                     "refresh_token": tokens.refresh_token})
    if r.status_code != 200:
        raise RuntimeError(f"token refresh failed: HTTP {r.status_code} {r.text[:200]}")
    new = _tokens_from(r.json(), tokens)
    save(new, path)
    return new


def ensure_fresh(http: httpx.Client, path: Path = DEFAULT_PATH, *, now: float | None = None) -> CodexTokens:
    t = load(path)
    if t.expires_at() - (now if now is not None else time.time()) < REFRESH_MARGIN_S:
        t = refresh(t, http, path)
    return t


def device_login(http: httpx.Client, show: Callable[[str, str], None], *, path: Path = DEFAULT_PATH,
                 timeout_s: float = 900, sleep: Callable[[float], None] = time.sleep) -> CodexTokens:
    """Device-code flow: ``show(url, code)`` tells the human where to approve; polls until done."""
    r = http.post(AUTH_BASE + "/api/accounts/deviceauth/usercode", json={"client_id": CLIENT_ID})
    r.raise_for_status()
    d = r.json()
    dev_id, code = d["device_auth_id"], d.get("user_code") or d.get("usercode")
    interval = float(d.get("interval") or 5)
    show(DEVICE_PAGE, code)
    t0 = time.time()
    while True:
        p = http.post(AUTH_BASE + "/api/accounts/deviceauth/token", json={"device_auth_id": dev_id, "user_code": code})
        if p.status_code == 200:
            g = p.json()
            break
        if p.status_code not in (403, 404):
            raise RuntimeError(f"device auth failed: HTTP {p.status_code} {p.text[:200]}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError("device code not approved in time")
        sleep(interval)
    tok = http.post(AUTH_BASE + "/oauth/token", data={
        "grant_type": "authorization_code", "client_id": CLIENT_ID, "code": g["authorization_code"],
        "code_verifier": g["code_verifier"], "redirect_uri": REDIRECT_URI})
    if tok.status_code != 200:
        raise RuntimeError(f"code exchange failed: HTTP {tok.status_code} {tok.text[:200]}")
    t = _tokens_from(tok.json())
    save(t, path)
    return t
