"""Own Codex login: device flow and refresh rotation, on httpx.MockTransport (no network)."""

from __future__ import annotations

import base64
import json
import os
import stat

import httpx

from controlr.llm import codex_auth as ca


def _jwt(claims: dict) -> str:
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
    return f"{enc({'alg': 'none'})}.{enc(claims)}.sig"


ID = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}})


def test_device_login_saves_a_private_token_file(tmp_path):
    seen, polls = [], []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/deviceauth/usercode"):
            return httpx.Response(200, json={"device_auth_id": "d1", "user_code": "ABCD-1234", "interval": "1"})
        if req.url.path.endswith("/deviceauth/token"):
            polls.append(1)
            return httpx.Response(404) if len(polls) < 3 else httpx.Response(
                200, json={"authorization_code": "c", "code_verifier": "v", "code_challenge": "x"})
        if req.url.path == "/oauth/token":
            assert b"grant_type=authorization_code" in req.content and b"code_verifier=v" in req.content
            return httpx.Response(200, json={"access_token": _jwt({"exp": 2_000_000_000}), "refresh_token": "r1",
                                             "id_token": ID})
        return httpx.Response(500)

    p = tmp_path / "auth.json"
    t = ca.device_login(httpx.Client(transport=httpx.MockTransport(handler)), lambda u, c: seen.append((u, c)),
                        path=p, sleep=lambda s: None)
    assert seen == [(ca.DEVICE_PAGE, "ABCD-1234")] and len(polls) == 3
    assert t.account_id == "acct-1" and "r1" not in repr(t)
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600 and ca.load(p).refresh_token == "r1"


def test_refresh_rotates_and_persists_before_returning(tmp_path):
    p = tmp_path / "auth.json"
    ca.save(ca.CodexTokens(_jwt({"exp": 1000}), "old", ID, "acct-1", 0), p)

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body == {"grant_type": "refresh_token", "client_id": ca.CLIENT_ID, "refresh_token": "old"}
        return httpx.Response(200, json={"access_token": _jwt({"exp": 9999}), "refresh_token": "new"})

    t = ca.ensure_fresh(httpx.Client(transport=httpx.MockTransport(handler)), p, now=900)
    assert t.refresh_token == "new" and ca.load(p).refresh_token == "new" and t.account_id == "acct-1"
    # still fresh -> no request
    t2 = ca.ensure_fresh(httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))), p, now=1000)
    assert t2.refresh_token == "new"
