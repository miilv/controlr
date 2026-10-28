"""SEC: provider error bodies, exception texts and response headers never carry a credential into an LLMResult
(controlr.keyguard; robot-harness docs/notes/SEC-VERIFICATION-2026-10-28.md findings 1 and 5). Fake keys only,
httpx.MockTransport, no network. Assertions compare names and fragments, never print an environment."""

from __future__ import annotations

import json

import httpx
import pytest

from controlr import keyguard
from controlr.config import load_config
from controlr.llm.client import KEEP_HEADER_PREFIXES, LLMClient, _error_message
from controlr.llm.decisions import DecisionsClient

FAKE = "sk-" + "ant-FAKE-0000-q7Wm2Lx9Rt4Yb8Nc3Vd6Hj1Kp5"   # sk-shaped (split so CI's secret scan sees no key literal)
OPAQUE = "FAKEopaque0000zzzz1111qqqq2222wwww"            # no pattern matches it: only the registered value


def fragments(key: str, text: str, n: int = 4) -> list[str]:
    """Every n-character piece of ``key`` found in ``text`` (empty = no fragment survived)."""
    return [key[i:i + n] for i in range(len(key) - n + 1) if key[i:i + n] in text]


def no_key(text: str, *keys: str) -> bool:
    return not any(fragments(k, text) for k in keys) and "sk-a" not in text and "FAKE" not in text


# ---------------------------------------------------------------------------
# keyguard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    f"Authorization: Bearer {FAKE}",
    f'{{"headers": {{"authorization": "Bearer {FAKE}"}}}}',
    f"x-api-key: {OPAQUE}",
    f'{{\\"api_key\\": \\"{OPAQUE}\\"}}',                  # JSON-escaped, as in a raw body
    f"OMNIROUTE_API_KEY={OPAQUE}",
    f"Authorization: Basic {OPAQUE}",
    f"proxy http://user:{OPAQUE}@proxy.test:3128 refused",
    f"GET https://api.test/v1?key={OPAQUE}&x=1",
    f"sk-or-v1-{OPAQUE}",
])
def test_redact_catches_credential_shapes(text):
    out = keyguard.redact(text)
    assert keyguard.R in out and no_key(out, FAKE, OPAQUE)


def test_redact_registered_and_extra_values_are_exact_matches():
    bare = f"received {OPAQUE} in the body"
    assert keyguard.redact(bare, extra=(OPAQUE,)) == f"received {keyguard.R} in the body"
    assert keyguard.redact("max_tokens is too large; anthropic-ratelimit-tokens-remaining: 10") == \
        "max_tokens is too large; anthropic-ratelimit-tokens-remaining: 10"   # no false positives on "tokens"


@pytest.mark.parametrize("key", [FAKE, OPAQUE])
@pytest.mark.parametrize("start", [390, 395, 398, 400])
def test_clip_redacts_before_cutting(key, start):
    """The padded case: the key starts just before the 400-character cut; cutting first would keep its head."""
    text = "x" * start + key + " tail"
    out = keyguard.clip(text, 400, extra=(key,))
    assert len(out) <= 403 and not fragments(key, out, 4) and "sk-a" not in out


def test_safe_label():
    assert keyguard.safe_label("claude/claude-sonnet-5-5") == "claude/claude-sonnet-5-5"
    assert keyguard.safe_label("stop") == "stop" and keyguard.safe_label(None) is None
    assert keyguard.safe_label(FAKE) == keyguard.R
    assert keyguard.safe_label("model with spaces") == "<invalid>"
    assert keyguard.safe_label({"x": 1}) == "<invalid>" and keyguard.safe_label("a" * 121) == "<invalid>"


def test_error_message_bounds_and_redacts():
    body = json.dumps({"error": {"message": "x" * 372 + f" Authorization: Bearer {FAKE}"}}).encode()   # key at 395
    msg = _error_message(400, body)
    assert msg.startswith("HTTP 400: ") and no_key(msg, FAKE) and len(msg) <= len("HTTP 400: ") + 403


# ---------------------------------------------------------------------------
# LLMClient (chat): error bodies, retries, stream errors, transport errors, headers
# ---------------------------------------------------------------------------

def _client(handler, key: str = FAKE, retries: int = 0) -> LLMClient:
    return LLMClient("https://example.test/v1", key, timeout_s=5, max_retries=retries,
                     transport=httpx.MockTransport(handler), sleep=lambda s: None)


def _echo(status: int, padded: bool = False, raw: bool = False):
    """A gateway that echoes the request's headers in its error body (the audit's finding 1)."""
    def h(req: httpx.Request) -> httpx.Response:
        auth = req.headers["authorization"]
        if raw:                                      # the bare key, no "Bearer", no JSON: only registration catches it
            return httpx.Response(status, text="y" * 385 + " received " + auth.split(" ", 1)[1])
        if padded:                                   # the key starts 5 characters before the 400-character cut
            return httpx.Response(status, text="x" * (395 - len("Authorization: Bearer ")) + f"Authorization: {auth}")
        return httpx.Response(status, json={"error": {"message": f"bad request; headers were {{'authorization': '{auth}'}}",
                                                      "type": "invalid_request_error"}})
    return h


@pytest.mark.parametrize("case", ["echo", "padded", "raw"])
def test_400_echoing_the_key_never_reaches_the_error(case):
    key = OPAQUE if case == "raw" else FAKE
    with _client(_echo(400, padded=case == "padded", raw=case == "raw"), key=key) as c:
        r = c.complete("m", [{"role": "user", "content": "x"}], max_tokens=5)
    assert r.http_status == 400 and r.error.startswith("HTTP 400") and no_key(r.error, key)


def test_retried_5xx_and_gave_up_error_is_redacted():
    with _client(_echo(503), retries=2) as c:
        r = c.complete("m", [{"role": "user", "content": "x"}], max_tokens=5)
    assert r.attempts == 3 and "gave up after 3 attempts" in r.error and no_key(r.error, FAKE)


def test_stream_error_chunk_is_redacted():
    body = (f'data: {json.dumps({"error": {"message": f"upstream said Bearer {FAKE}", "code": 400}})}\n\n').encode()
    h = lambda req: httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})  # noqa: E731
    with _client(h) as c:
        r = c.complete("m", [{"role": "user", "content": "x"}], max_tokens=5)
    assert r.error.startswith("stream error:") and no_key(r.error, FAKE)


def test_transport_error_text_is_redacted():
    def h(req):
        raise httpx.ProxyError(f"proxy http://user:{OPAQUE}@proxy.test:3128 refused; Bearer {FAKE}")
    with _client(h, retries=1) as c:
        r = c.complete("m", [{"role": "user", "content": "x"}], max_tokens=5)
    assert "ProxyError" in r.error and no_key(r.error, FAKE, OPAQUE)


def test_only_allow_listed_response_headers_are_kept():
    sse = b'data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}\n\ndata: [DONE]\n\n'
    hdrs = {"content-type": "text/event-stream", "x-cache": "MISS", "x-omniroute-cache": "MISS",
            "x-request-id": "req-1", "x-ratelimit-remaining-requests": "99", "anthropic-ratelimit-tokens-remaining": "5",
            "set-cookie": "s=1", "server": "gw", "cf-ray": "abc", "x-omniroute-api-key": OPAQUE,
            "x-echo-authorization": f"Bearer {FAKE}", "x-omniroute-upstream": f"Bearer {FAKE}"}
    with _client(lambda req: httpx.Response(200, content=sse, headers=hdrs)) as c:
        r = c.complete("m", [{"role": "user", "content": "x"}], max_tokens=5)
    assert r.text == "ok" and r.finish_reason == "stop"
    assert sorted(r.headers) == sorted(["x-cache", "x-omniroute-cache", "x-request-id", "x-ratelimit-remaining-requests",
                                        "anthropic-ratelimit-tokens-remaining", "x-omniroute-upstream"])
    assert all(k.lower().startswith(KEEP_HEADER_PREFIXES) for k in r.headers)
    assert r.headers["x-omniroute-upstream"] == f"Bearer {keyguard.R}"       # a kept value is redacted
    assert no_key(json.dumps(r.headers), FAKE, OPAQUE)


def test_finish_reason_is_a_safe_label():
    sse = (f'data: {json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": FAKE}]})}\n\n'
           "data: [DONE]\n\n").encode()
    with _client(lambda req: httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})) as c:
        r = c.complete("m", [{"role": "user", "content": "x"}], max_tokens=5)
    assert r.finish_reason == keyguard.R


# ---------------------------------------------------------------------------
# DecisionsClient: the same error path, headers and the served ids in the decision record
# ---------------------------------------------------------------------------

def _dcfg():
    return load_config(None, ["llm.backend=decisions", "llm.model=openai/gpt-6-luna-decisions", "planner.enabled=false",
                              "robot.backend=mock", "task.name=reach", "llm.base_url=http://fake",
                              "llm.overlap_tail=false", "observation.renderers=[raw]"], dotenv=None)


def _dclient(handler) -> DecisionsClient:
    return DecisionsClient("https://example.test/api/alpha/", FAKE, _dcfg(), 5, 0,
                           transport=httpx.MockTransport(handler), sleep=lambda s: None)


MSGS = [{"role": "system", "content": [{"type": "text", "text": "MANUAL"}]},
        {"role": "user", "content": [{"type": "text", "text": "TASK: reach"}]}]


def test_decisions_error_and_headers_are_clean():
    res = _dclient(_echo(400, padded=True)).complete("m", MSGS)
    assert res.http_status == 400 and no_key(res.error, FAKE) and res.headers == {}


def test_decisions_served_ids_are_safe_labels():
    zero = {str(i): (1.0 if i == 3 else 0.0) for i in range(7)}
    answers = {k: {"type": "score", "probabilities": zero, "score": 3.0} for k in ("dx", "dy", "dz")}
    answers["grip"] = {"type": "choice", "choice": "keep", "probabilities": {"keep": 0.9}}
    answers["status"] = {"type": "choice", "choice": "CONTINUE", "probabilities": {"CONTINUE": 0.9}}

    def h(req):
        return httpx.Response(200, json={"id": "gen-1", "model": FAKE, "provider": "Open AI <script>", "answers": answers,
                                         "usage": {"input_tokens": 1, "output_tokens": 0}},
                              headers={"x-request-id": "r1", "x-leak": f"Bearer {FAKE}"})
    res = _dclient(h).complete("m", MSGS)
    assert res.error is None and res.decisions["id"] == "gen-1"
    assert res.decisions["model"] == keyguard.R and res.decisions["provider"] == "<invalid>"
    assert res.headers == {"x-request-id": "r1"}


# ---------------------------------------------------------------------------
# one source per process; child environments
# ---------------------------------------------------------------------------

def _envfile(tmp_path, text: str):
    f = tmp_path / "agentic.env"
    f.write_text(text)
    return str(f)


def test_router_credentials_one_source(tmp_path):
    both = _envfile(tmp_path, f"OMNIROUTE_BASE_URL=https://router.test/v1\nexport OMNIROUTE_API_KEY={OPAQUE}\nOTHER=1\n")
    assert keyguard.router_credentials(both, "AGENTIC_ENV_FILE", env={}) == ("https://router.test/v1", OPAQUE)
    env_only = {"OMNIROUTE_BASE_URL": "https://router.test/v1", "OMNIROUTE_API_KEY": OPAQUE}
    assert keyguard.router_credentials(None, "AGENTIC_ENV_FILE", env=env_only)[1] == OPAQUE
    # the old merge: the file holds the key, the environment the URL -> refused, values never in the message
    key_only = _envfile(tmp_path, f"OMNIROUTE_API_KEY={OPAQUE}\n")
    with pytest.raises(RuntimeError, match="one source") as e:
        keyguard.router_credentials(key_only, "AGENTIC_ENV_FILE", env={"OMNIROUTE_BASE_URL": "https://elsewhere.test/v1"})
    assert OPAQUE not in str(e.value) and "elsewhere" not in str(e.value)
    with pytest.raises(RuntimeError, match="OMNIROUTE_BASE_URL not set in AGENTIC_ENV_FILE"):
        keyguard.router_credentials(key_only, "AGENTIC_ENV_FILE", env={})
    with pytest.raises(RuntimeError, match="no such file"):
        keyguard.router_credentials(str(tmp_path / "missing.env"), "AGENTIC_ENV_FILE", env={})


def test_child_env_drops_credential_names():
    parent = {"PATH": "/usr/bin", "HOME": "/h", "CUDA_VISIBLE_DEVICES": "0", "DISPLAY": ":1", "XAUTHORITY": "/h/.Xa",
              "PYOPENGL_PLATFORM": "egl", "RD_HOME": "/rd", "DOJO_CATALOG": "/c", "SIMLAB_GPU": "0", "J1_GL": "egl",
              "MAX_TOKENS": "5", "OMNIROUTE_BASE_URL": "https://router.test/v1", "OMNIROUTE_API_KEY": FAKE,
              "AGENTIC_ENV_FILE": "/k/agentic.env", "DOJO_AGENT_DOTENV": "/k/.env", "OPENROUTER_API_KEY": FAKE,
              "ANTHROPIC_AUTH_TOKEN": FAKE, "ANTHROPIC_BASE_URL": "https://a.test", "OPENAI_API_KEY": FAKE,
              "HF_TOKEN": "hf_FAKE", "GITHUB_TOKEN": "gh_FAKE", "AWS_SECRET_ACCESS_KEY": "FAKE",
              "SSH_AUTH_SOCK": "/tmp/agent", "CONTROLR_CODEX_AUTH": "/k/codex.json", "SOME_PASSWORD": "FAKE",
              "SIMLAB_TWIN_AUTHKEY": "ab", "DOJO_PERCEPTION_AUTHKEY": "cd", "UNRELATED_SETTING": OPAQUE}
    keyguard.register(OPAQUE)
    kept = {"PATH", "HOME", "CUDA_VISIBLE_DEVICES", "DISPLAY", "XAUTHORITY", "PYOPENGL_PLATFORM", "RD_HOME",
            "DOJO_CATALOG", "SIMLAB_GPU", "J1_GL", "MAX_TOKENS", "SIMLAB_TWIN_AUTHKEY", "DOJO_PERCEPTION_AUTHKEY"}
    assert set(keyguard.child_env(base=parent)) == kept
    assert set(keyguard.child_env({"SIMLAB_TWIN_PORT": 1}, base=parent)) == kept | {"SIMLAB_TWIN_PORT"}
    model = set(keyguard.child_env(base=parent, model_calls=True))
    assert model == kept | set(keyguard.MODEL_ENV)
