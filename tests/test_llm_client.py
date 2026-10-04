"""LLMClient: SSE parsing, usage normalisation, early stop, retries (no network:
httpx.MockTransport)."""

from __future__ import annotations

import json

import httpx
import pytest

from controlr.llm.client import (LLMClient, StreamState, iter_sse_data, normalize_usage,
                                 _parse_retry_after)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def chunk(content: str | None = None, *, reasoning: str | None = None,
          finish: str | None = None, usage: dict | None = None) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    c: dict = {"id": "x", "object": "chat.completion.chunk",
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        c = {"id": "x", "choices": [], "usage": usage}
    return c


def sse_body(chunks: list[dict], done: bool = True) -> bytes:
    out = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    if done:
        out += "data: [DONE]\n\n"
    return out.encode()


USAGE = {"prompt_tokens": 4900, "completion_tokens": 12, "total_tokens": 5012,
         "prompt_tokens_details": {"cached_tokens": 4900},
         "cache_read_input_tokens": 4900, "cache_creation_input_tokens": 100,
         "completion_tokens_details": {"reasoning_tokens": 0}}


def make_client(handler, *, max_retries: int = 3, sleeps: list | None = None, **kw) -> LLMClient:
    sl = sleeps if sleeps is not None else []
    return LLMClient("https://example.test/v1/", "sk-secret", timeout_s=5, max_retries=max_retries,
                     transport=httpx.MockTransport(handler), sleep=sl.append, **kw)


def stream_handler(body: bytes, status: int = 200, seen: list | None = None):
    def h(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, content=body, headers={"content-type": "text/event-stream",
                                                             "x-cache": "MISS", "set-cookie": "s=1"})
    return h


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------

def test_iter_sse_data_spec_cases():
    lines = [
        ": keep-alive",
        "event: message",
        'data: {"a": 1}',
        "",
        "data:no-space",
        "",
        "data: line1",
        "data: line2",
        "id: 7",
        "",
        "",
        "data: [DONE]",
        "data: trailing-without-blank",
    ]
    assert list(iter_sse_data(lines)) == [
        '{"a": 1}', "no-space", "line1\nline2", "[DONE]\ntrailing-without-blank"]


def test_iter_sse_data_crlf():
    assert list(iter_sse_data(["data: x\r\n", "\r\n", "data: y\r"])) == ["x", "y"]


def test_stream_state_accumulates_content_reasoning_finish_usage():
    st = StreamState()
    st.feed(chunk(reasoning="think "))
    st.feed({"choices": [{"delta": {"reasoning": "more"}}]})
    st.feed(chunk("MOVE ee_delta 1 2 3\n"))
    st.feed(chunk("STATUS OK", finish="stop"))
    st.feed(chunk(usage=USAGE))
    assert st.text == "MOVE ee_delta 1 2 3\nSTATUS OK"
    assert st.reasoning == "think more"
    assert st.finish_reason == "stop"
    assert st.usage_raw == USAGE


def test_stream_state_list_content_and_error_chunk():
    st = StreamState()
    st.feed({"choices": [{"delta": {"content": [{"type": "text", "text": "ab"}]}}]})
    assert st.text == "ab"
    st.feed({"error": {"message": "overloaded", "code": 529}})
    assert st.error == "stream error: overloaded" and st.error_status == 529


# ---------------------------------------------------------------------------
# usage normalisation
# ---------------------------------------------------------------------------

def test_usage_openai_style_with_anthropic_extras():
    u = normalize_usage(USAGE)
    assert (u.prompt_tokens, u.completion_tokens, u.cache_read_tokens,
            u.cache_write_tokens, u.reasoning_tokens) == (5000, 12, 4900, 100, 0)
    assert u.raw is USAGE


def test_usage_only_cached_tokens_and_top_level_reasoning():
    u = normalize_usage({"prompt_tokens": 300, "completion_tokens": 50,
                         "prompt_tokens_details": {"cached_tokens": 256}, "reasoning_tokens": 40})
    assert (u.prompt_tokens, u.cache_read_tokens, u.cache_write_tokens, u.reasoning_tokens) == (300, 256, 0, 40)


def test_usage_completion_details_reasoning():
    u = normalize_usage({"prompt_tokens": 10, "completion_tokens": 9,
                         "completion_tokens_details": {"reasoning_tokens": 7}})
    assert u.reasoning_tokens == 7


def test_usage_omniroute_semantics_measured():
    # write-only turn: prompt_tokens excludes the creation
    u = normalize_usage({"prompt_tokens": 223, "completion_tokens": 24, "total_tokens": 247,
                         "reasoning_tokens": 0, "completion_tokens_details": {"reasoning_tokens": 0},
                         "cache_creation_input_tokens": 22174})
    assert (u.prompt_tokens, u.cache_read_tokens, u.cache_write_tokens) == (22397, 0, 22174)
    # read turn: prompt_tokens = uncached + read, creation on top
    u = normalize_usage({"prompt_tokens": 22397, "completion_tokens": 24,
                         "prompt_tokens_details": {"cached_tokens": 22174},
                         "cache_read_input_tokens": 22174, "cache_creation_input_tokens": 245})
    assert (u.prompt_tokens, u.cache_read_tokens, u.cache_write_tokens) == (22642, 22174, 245)


def test_usage_prompt_excluding_reads_is_made_total():
    u = normalize_usage({"prompt_tokens": 100, "completion_tokens": 5,
                         "cache_read_input_tokens": 4900, "cache_creation_input_tokens": 50})
    assert u.prompt_tokens == 5050


def test_usage_anthropic_native_keys():
    u = normalize_usage({"input_tokens": 100, "output_tokens": 5,
                         "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0})
    assert (u.prompt_tokens, u.completion_tokens, u.cache_read_tokens) == (1000, 5, 900)


def test_usage_missing_or_garbage():
    assert normalize_usage(None) is None
    assert normalize_usage({}) is None
    u = normalize_usage({"prompt_tokens": None, "prompt_tokens_details": None,
                         "completion_tokens": "x"})
    assert u.prompt_tokens == 0 and u.completion_tokens == 0


# ---------------------------------------------------------------------------
# complete(): streaming, request body, timings
# ---------------------------------------------------------------------------

def test_complete_streams_and_builds_request():
    seen: list[httpx.Request] = []
    body = sse_body([chunk(reasoning="hmm"), chunk("MOVE ee_delta 1 0 0\n"),
                     chunk("STATUS OK", finish="stop"), chunk(usage=USAGE)])
    with make_client(stream_handler(body, seen=seen)) as c:
        r = c.complete("claude/x", [{"role": "user", "content": "hi"}], max_tokens=50,
                       temperature=0.0, extra_body={"reasoning_effort": "low"})
    assert r.error is None and r.http_status == 200 and r.attempts == 1
    assert r.text == "MOVE ee_delta 1 0 0\nSTATUS OK"
    assert r.reasoning_text == "hmm"
    assert r.headers.get("x-cache") == "MISS" and "set-cookie" not in r.headers
    assert r.finish_reason == "stop" and not r.stopped_early
    assert r.usage is not None and r.usage.cache_read_tokens == 4900
    t = r.timings
    assert t.ttft_any is not None and t.ttft is not None and t.ttft_any <= t.ttft <= t.t_end
    assert t.t_complete is None and t.t_wall >= t.t_end
    req = seen[0]
    assert str(req.url) == "https://example.test/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-secret"
    sent = json.loads(req.content)
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert sent["max_tokens"] == 50 and sent["temperature"] == 0.0
    assert sent["reasoning_effort"] == "low" and sent["model"] == "claude/x"


def test_temperature_omitted_when_none():
    seen: list[httpx.Request] = []
    with make_client(stream_handler(sse_body([chunk("a")]), seen=seen)) as c:
        c.complete("m", [], max_tokens=5)
    assert "temperature" not in json.loads(seen[0].content)


def test_repr_hides_key():
    c = make_client(stream_handler(b""))
    assert "sk-secret" not in repr(c)


def test_json_fallback_when_gateway_ignores_stream():
    def h(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": "STATUS DONE"},
                                                      "finish_reason": "stop"}], "usage": USAGE})
    with make_client(h) as c:
        r = c.complete("m", [], max_tokens=5, stop_when=lambda s: "STATUS" in s)
    assert r.text == "STATUS DONE" and r.usage.prompt_tokens == 5000 and r.timings.t_complete is not None


# ---------------------------------------------------------------------------
# early stop
# ---------------------------------------------------------------------------

class CountingStream(httpx.SyncByteStream):
    """Yields SSE events one by one and records how many were consumed and
    whether the stream was closed."""

    def __init__(self, events: list[bytes]):
        self.events = events
        self.consumed = 0
        self.closed = False

    def __iter__(self):
        for e in self.events:
            self.consumed += 1
            yield e

    def close(self):
        self.closed = True


def _events(chunks: list[dict]) -> list[bytes]:
    return [f"data: {json.dumps(c)}\n\n".encode() for c in chunks] + [b"data: [DONE]\n\n"]


def test_early_stop_closes_stream():
    stream = CountingStream(_events(
        [chunk("MOVE ee_delta 1 0 0\n"), chunk("STATUS OK"), chunk("\nand some prose"),
         chunk(" more prose", finish="stop"), chunk(usage=USAGE)]))

    def h(req):
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    with make_client(h) as c:
        r = c.complete("m", [], max_tokens=50, stop_when=lambda s: "STATUS" in s)
    assert r.stopped_early and r.text == "MOVE ee_delta 1 0 0\nSTATUS OK"
    assert r.usage is None and r.finish_reason is None
    assert r.timings.t_complete is not None and r.timings.t_complete <= r.timings.t_end
    assert stream.closed and stream.consumed == 2


def test_early_stop_with_usage_grace_keeps_frozen_text_but_gets_usage():
    stream = CountingStream(_events(
        [chunk("STATUS OK"), chunk(" trailing", finish="stop"), chunk(usage=USAGE)]))

    def h(req):
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    with make_client(h, usage_grace_s=5.0) as c:
        r = c.complete("m", [], max_tokens=50, stop_when=lambda s: "STATUS" in s)
    # the STATUS line's note is completed during the grace period (stream ended -> complete)
    assert r.stopped_early and r.text == "STATUS OK trailing" and not r.truncated
    assert r.usage is not None and r.usage.prompt_tokens == 5000
    assert r.finish_reason == "stop"
    assert r.request_bytes > 0 and r.timings.t_headers is not None


def test_grace_cuts_the_note_at_the_end_of_the_status_line():
    """Review caching #9 / contracts #19: the stored note must not depend on chunking."""
    for split in ([chunk("STATUS OK mov"), chunk("ing above\nprose"), chunk(usage=USAGE)],
                  [chunk("STATUS OK moving above\npr"), chunk("ose"), chunk(usage=USAGE)],
                  [chunk("STATUS OK moving a"), chunk("bove"), chunk("\nprose", finish="stop"), chunk(usage=USAGE)]):
        stream = CountingStream(_events(split))

        def h(req, stream=stream):
            return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

        with make_client(h, usage_grace_s=5.0) as c:
            r = c.complete("m", [], max_tokens=50, stop_when=lambda s: "STATUS OK" in s)
        assert r.text == "STATUS OK moving above" and not r.truncated, split


def test_no_grace_marks_a_cut_note_as_truncated():
    stream = CountingStream(_events([chunk("STATUS OK mov"), chunk("ing\n"), chunk(usage=USAGE)]))

    def h(req):
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    with make_client(h, usage_grace_s=0.0) as c:
        r = c.complete("m", [], max_tokens=50, stop_when=lambda s: "STATUS OK" in s)
    assert r.text == "STATUS OK mov" and r.truncated


def test_per_call_timeout_and_retries_and_extra_headers():
    seen = []

    def h(req):
        seen.append(req)
        return httpx.Response(503, json={"error": {"message": "busy"}})

    with make_client(h, max_retries=3, headers={"X-Test-Lease": "run-1"}) as c:
        r = c.complete("m", [], max_tokens=5, timeout_s=600, max_retries=1)
    assert r.error and r.attempts == 2 and len(seen) == 2
    assert seen[0].headers["X-Test-Lease"] == "run-1"
    assert seen[0].extensions["timeout"]["read"] == 600


def test_stop_when_only_checked_on_content():
    body = sse_body([chunk(reasoning="STATUS in thoughts"), chunk("MOVE"), chunk(" x", finish="stop")])
    with make_client(stream_handler(body)) as c:
        r = c.complete("m", [], max_tokens=5, stop_when=lambda s: "STATUS" in s)
    assert not r.stopped_early and r.text == "MOVE x"


# ---------------------------------------------------------------------------
# retries / errors
# ---------------------------------------------------------------------------

def test_retry_on_429_respects_retry_after_then_succeeds():
    calls = {"n": 0}

    def h(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}},
                                  headers={"retry-after": "2"})
        if calls["n"] == 2:
            return httpx.Response(503, text="upstream busy")
        return httpx.Response(200, content=sse_body([chunk("ok", finish="stop")]),
                              headers={"content-type": "text/event-stream"})

    sleeps: list[float] = []
    with make_client(h, sleeps=sleeps, backoff_base=0.5) as c:
        r = c.complete("m", [], max_tokens=5)
    assert r.text == "ok" and r.error is None and r.attempts == 3
    assert sleeps[0] == 2.0
    assert 0.5 <= sleeps[1] <= 1.0          # base*2**1 with [50%,100%] jitter


def test_retry_on_connection_error_and_give_up():
    def h(req):
        raise httpx.ConnectError("connection refused", request=req)

    sleeps: list[float] = []
    with make_client(h, max_retries=2, sleeps=sleeps, backoff_base=1.0) as c:
        r = c.complete("m", [], max_tokens=5)
    assert r.attempts == 3 and len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 2.0
    assert "ConnectError" in r.error and "gave up after 3 attempts" in r.error
    assert r.text == "" and r.usage is None


def test_backoff_is_capped():
    c = make_client(stream_handler(b""), backoff_base=1.0, backoff_max=3.0, retry_after_max=10)
    assert all(c._backoff(k, None) <= 3.0 for k in range(1, 20))
    assert c._backoff(1, 999.0) == 10


def test_no_retry_on_400_and_clear_error():
    calls = {"n": 0}

    def h(req):
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "unsupported_image_block",
                                                   "type": "invalid_request_error"}})

    with make_client(h) as c:
        r = c.complete("dva/gemini", [], max_tokens=5)
    assert calls["n"] == 1 and r.attempts == 1 and r.http_status == 400
    assert r.error == "HTTP 400: unsupported_image_block"
    assert "sk-secret" not in r.error


def test_in_stream_error_before_content_is_retried():
    calls = {"n": 0}

    def h(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, content=sse_body([{"error": {"message": "overloaded"}}]),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, content=sse_body([chunk("fine")]),
                              headers={"content-type": "text/event-stream"})

    with make_client(h) as c:
        r = c.complete("m", [], max_tokens=5)
    assert r.text == "fine" and r.attempts == 2


def test_in_stream_error_after_content_returns_partial():
    body = sse_body([chunk("MOVE ee"), {"error": {"message": "cut", "code": 500}}])
    with make_client(stream_handler(body)) as c:
        r = c.complete("m", [], max_tokens=5)
    assert r.text == "MOVE ee" and r.error == "stream error: cut" and r.attempts == 1


def test_read_error_mid_stream_retried():
    calls = {"n": 0}

    class Broken(httpx.SyncByteStream):
        def __iter__(self):
            yield f"data: {json.dumps(chunk('MO'))}\n\n".encode()
            raise httpx.ReadError("reset")

    def h(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, stream=Broken(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, content=sse_body([chunk("MOVE")]),
                              headers={"content-type": "text/event-stream"})

    with make_client(h) as c:
        r = c.complete("m", [], max_tokens=5)
    assert r.text == "MOVE" and r.attempts == 2


def test_parse_retry_after():
    assert _parse_retry_after("3") == 3.0
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("garbage") is None
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0


def test_http_client_reused_across_calls():
    with make_client(stream_handler(sse_body([chunk("a")]))) as c:
        h1 = c._http
        c.complete("m", [], max_tokens=5)
        c.complete("m", [], max_tokens=5)
        assert c._http is h1 and not h1.is_closed
    assert h1.is_closed


def test_idle_connections_are_kept_longer_than_a_slow_turn():
    from controlr.llm.client import KEEPALIVE_S, LLMClient

    c = LLMClient("http://localhost:1", "k", timeout_s=5, max_retries=0)
    try:
        pool = getattr(getattr(c._http, "_transport", None), "_pool", None)
        if pool is not None and hasattr(pool, "_keepalive_expiry"):     # httpx internals; skip if they move
            assert pool._keepalive_expiry == KEEPALIVE_S >= 60
    finally:
        c.close()
