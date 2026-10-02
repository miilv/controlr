"""FakeLLM: scripted replies with the real client's result shape."""

from __future__ import annotations

import inspect

from controlr.llm.client import LLMClient
from controlr.llm.fake import FakeLLM


def test_same_signature_as_client():
    assert inspect.signature(FakeLLM.complete) == inspect.signature(LLMClient.complete)


def test_list_replies_in_order_last_repeats():
    f = FakeLLM(["A\nSTATUS OK", "B\nSTATUS DONE"])
    texts = [f.complete("m", [{"role": "user", "content": "x"}], max_tokens=100).text for _ in range(3)]
    assert texts == ["A\nSTATUS OK", "B\nSTATUS DONE", "B\nSTATUS DONE"]
    assert len(f.calls) == 3 and f.calls[0]["model"] == "m"


def test_callable_replies_and_usage_grows_cache():
    f = FakeLLM(lambda msgs: f"n={len(msgs)}\nSTATUS OK")
    msgs = [{"role": "system", "content": "s" * 400}, {"role": "user", "content": "u"}]
    r1 = f.complete("m", msgs, max_tokens=100)
    r2 = f.complete("m", msgs + [{"role": "assistant", "content": "a"},
                                 {"role": "user", "content": [
                                     {"type": "image_url", "image_url": {"url": "data:"}}]}],
                    max_tokens=100)
    assert r1.text == "n=2\nSTATUS OK" and r2.text == "n=4\nSTATUS OK"
    assert r1.usage.cache_read_tokens == 0
    assert r2.usage.cache_read_tokens == r1.usage.prompt_tokens < r2.usage.prompt_tokens
    assert r2.timings.ttft is not None and r2.timings.t_end > r2.timings.ttft


def test_early_stop_truncates_like_real_client():
    f = FakeLLM(["MOVE ee_delta 1 0 0\nSTATUS OK\nthen some trailing prose"], chunk_chars=4)
    r = f.complete("m", [], max_tokens=100, stop_when=lambda s: "STATUS OK" in s)
    assert r.stopped_early and r.text.startswith("MOVE ee_delta 1 0 0\nSTATUS OK")
    assert "prose" not in r.text and r.timings.t_complete is not None
    assert r.usage is not None and not r.truncated      # like the real client with usage grace


def test_exception_reply_is_error_result():
    f = FakeLLM([RuntimeError("boom"), "STATUS OK"])
    r = f.complete("m", [], max_tokens=10)
    assert r.error == "boom" and r.text == ""
    assert f.complete("m", [], max_tokens=10).text == "STATUS OK"
