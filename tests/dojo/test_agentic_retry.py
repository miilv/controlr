"""All HTTP responses come from MockTransport; urllib is adapted without sockets."""
import io
import json
import urllib.error
import urllib.request
from email.utils import formatdate

import httpx
import pytest

from dojo.perception.agentic import llm


@pytest.mark.parametrize("status,header,expected", [
    (429, "7", [7]), (503, "120", [60]), (501, "garbage", [2]),
    (429, None, [2]), (400, "7", []),
])
def test_stdlib_retry_policy(monkeypatch, status, header, expected):
    sent, sleeps = [], []

    def handle(req):
        sent.append(req.content)
        if len(sent) == 1:
            return httpx.Response(status, headers={"Retry-After": header} if header else {}, text="busy")
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        def open_request(req, timeout):
            r = client.post(req.full_url, content=req.data)
            if r.status_code >= 400:
                raise urllib.error.HTTPError(req.full_url, r.status_code, "error", r.headers, io.BytesIO(r.content))
            return io.BytesIO(r.content)

        monkeypatch.setattr(urllib.request, "urlopen", open_request)
        monkeypatch.setattr(llm, "router_credentials", lambda: ("http://router.invalid/v1", "test-key"))
        monkeypatch.setattr(llm.time, "sleep", sleeps.append)
        r = llm.UrllibChat(max_retries=1)("model", [], max_tokens=10)
    assert sleeps == expected
    assert len(sent) == (2 if expected else 1)
    assert all(body == sent[0] for body in sent)
    assert json.loads(sent[0])["model"] == "model"
    assert (r.text == "ok") if expected else r.error.startswith("HTTP 400")


def test_exhaustion_never_sleeps_after_last_attempt(monkeypatch):
    monkeypatch.setattr(llm, "router_credentials", lambda: ("http://router.invalid/v1", "test-key"))
    sleeps, sent = [], []
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429, text="busy"))) as client:
        def open_request(req, timeout):
            sent.append(req.data)
            r = client.post(req.full_url, content=req.data)
            raise urllib.error.HTTPError(req.full_url, r.status_code, "error", r.headers, io.BytesIO(r.content))
        monkeypatch.setattr(urllib.request, "urlopen", open_request)
        monkeypatch.setattr(llm.time, "sleep", sleeps.append)
        r = llm.UrllibChat(max_retries=2)("model", [])
    assert len(sent) == 3 and sleeps == [2, 4]
    assert r.error.startswith("HTTP 429")


def test_retry_after_date_and_malformed(monkeypatch):
    monkeypatch.setattr(llm.time, "time", lambda: 1000)
    assert llm.retry_delay(0, formatdate(1009, usegmt=True)) == 9
    assert llm.retry_delay(0, formatdate(900, usegmt=True)) == 0
    assert llm.retry_delay(1, "nan") == 4
    assert llm.retry_delay(1, "inf") == 4
    assert llm.retry_delay(1, "bad date") == 4
