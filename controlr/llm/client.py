"""Streaming OpenAI-compatible chat client (one endpoint: omniroute).

WHY streaming even though we only act on the full reply: (1) it gives us the
timings we study (time to first token, time until the reply grammar is
complete), and (2) it lets us *early stop* — once the reply contains a STATUS
line nothing after it matters, so we close the stream instead of paying latency
for trailing prose. The robot is holding still the whole time, so every second
saved here is a second of robot idle time saved.

WHY our own SSE parser instead of the openai SDK: the dependency budget is
numpy + pillow + httpx + pyyaml, omniroute passes through provider-specific
usage keys (Anthropic cache counters) that SDK models would drop, and we need
exact control over when the connection is closed.

Usage normalisation: different routes report the same quantities under
different keys (see ``normalize_usage``). ``Usage.prompt_tokens`` is always
the TOTAL input (uncached + cache read + cache write) so cache-read share is
``cache_read_tokens / prompt_tokens`` regardless of route. ``raw`` keeps the
original dict for audits.
"""

from __future__ import annotations

import email.utils
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

import httpx


# ---------------------------------------------------------------------------
# result types (contract: docs/ARCHITECTURE.md "LLM client")
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    """Normalised across routes. ``prompt_tokens`` = total input tokens
    including cached ones (OpenAI semantics), whatever the route reported."""
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    raw: dict


@dataclass
class Timings:
    """Seconds, perf_counter relative to the start of the (final) attempt.

    ttft:       first non-empty *content* delta (the reply text proper)
    t_complete: first moment ``stop_when(text)`` returned True (None if never)
    t_end:      stream closed (normally or early)
    ttft_any:   first non-empty content OR reasoning delta (thinking routes
                start "talking" long before the visible reply) — addition
    t_wall:     whole call including failed attempts and backoff sleeps — addition
    t_headers:  response headers received (request upload + router queue; the
                rest of ``ttft`` is the model) — addition

    NB: on thinking routes ``ttft`` includes the thinking time (the first
    CONTENT delta comes after it); compare ``ttft_any`` and the reasoning tokens.
    """
    ttft: float | None
    t_complete: float | None
    t_end: float
    ttft_any: float | None = None
    t_wall: float | None = None
    t_headers: float | None = None


@dataclass
class LLMResult:
    text: str
    usage: Usage | None
    timings: Timings
    stopped_early: bool
    finish_reason: str | None
    error: str | None
    http_status: int | None
    attempts: int
    reasoning_text: str = ""   # addition: streamed thinking deltas, if the route exposes them
    headers: dict = field(default_factory=dict)   # addition: response headers (diagnostics, e.g.
                                                  # gateway response-cache hits); no cookies
    request_bytes: int = 0      # addition: serialised request size (all images are re-uploaded each turn)
    # addition: True when an early stop cut the STATUS line mid-note (the stream was closed
    # before that line ended); the loop then stores the reply without the partial note
    truncated: bool = False


# ---------------------------------------------------------------------------
# usage normalisation
# ---------------------------------------------------------------------------

def _int(x: Any) -> int:
    try:
        return int(x or 0)
    except (TypeError, ValueError):
        return 0


def normalize_usage(raw: dict | None) -> Usage | None:
    """Map every usage-key variant seen through omniroute onto ``Usage``.

    Keys handled: ``prompt_tokens`` / ``input_tokens``, ``completion_tokens`` /
    ``output_tokens``, ``prompt_tokens_details.cached_tokens``,
    ``cache_read_input_tokens``, ``cache_creation_input_tokens`` (also
    ``prompt_tokens_details.cache_creation_tokens``), ``reasoning_tokens``,
    ``completion_tokens_details.reasoning_tokens``.

    Total-input rule (measured through omniroute on claude/* routes,
    2026-10): ``prompt_tokens`` = uncached input + cache READ, but EXCLUDES
    cache CREATION (turn with only a write: prompt_tokens=223,
    cache_creation_input_tokens=22174; next turn: prompt_tokens=22397 =
    223 + cached_tokens 22174, creation 245 on top). So total =
    prompt_tokens + cache_write. OpenAI routes report no creation counter, so
    this reduces to prompt_tokens. If ``prompt_tokens`` is smaller than the
    read counter it cannot contain the reads either, so they are added too.
    Anthropic-native ``input_tokens`` excludes both read and write.
    """
    if not raw or not isinstance(raw, dict):
        return None
    ptd = raw.get("prompt_tokens_details") or {}
    ctd = raw.get("completion_tokens_details") or {}
    if not isinstance(ptd, dict):
        ptd = {}
    if not isinstance(ctd, dict):
        ctd = {}

    cache_read = max(_int(ptd.get("cached_tokens")), _int(raw.get("cache_read_input_tokens")))
    cache_write = max(_int(raw.get("cache_creation_input_tokens")),
                      _int(ptd.get("cache_creation_tokens")),
                      _int(ptd.get("cache_write_tokens")))
    reasoning = max(_int(raw.get("reasoning_tokens")), _int(ctd.get("reasoning_tokens")))

    if "prompt_tokens" in raw:
        prompt = _int(raw.get("prompt_tokens")) + cache_write
        if prompt - cache_write < cache_read:           # reads not included either
            prompt += cache_read
    else:
        prompt = _int(raw.get("input_tokens")) + cache_read + cache_write
    completion = _int(raw.get("completion_tokens", raw.get("output_tokens")))
    return Usage(prompt_tokens=prompt, completion_tokens=completion,
                 cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                 reasoning_tokens=reasoning, raw=raw)


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------

DONE = "[DONE]"


def iter_sse_data(lines: Iterable[str]) -> Iterator[str]:
    """Yield the ``data`` payload of each server-sent event.

    Follows the SSE spec as far as chat streams need: ``data:`` lines of one
    event are joined with newlines, a blank line dispatches, ``:`` lines are
    comments (keep-alives), ``event:``/``id:``/``retry:`` are ignored. A final
    event without a trailing blank line is still dispatched. ``[DONE]`` is
    yielded as-is; the consumer decides to stop.
    """
    buf: list[str] = []
    for line in lines:
        line = line.rstrip("\r\n")
        if line == "":
            if buf:
                yield "\n".join(buf)
                buf = []
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if name == "data":
            buf.append(value)
    if buf:
        yield "\n".join(buf)


def _delta_text(v: Any) -> str:
    """Content deltas are normally strings; some gateways send a list of parts."""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "".join(p.get("text", "") for p in v if isinstance(p, dict))
    return ""


@dataclass
class StreamState:
    """Accumulates parsed chunks of one streamed completion."""
    text: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    usage_raw: dict | None = None
    error: str | None = None
    error_status: int | None = None
    n_chunks: int = 0

    def feed(self, chunk: dict) -> tuple[str, str]:
        """Apply one decoded chunk; return (content_delta, reasoning_delta)."""
        self.n_chunks += 1
        if isinstance(chunk.get("error"), (dict, str)):
            err = chunk["error"]
            if isinstance(err, dict):
                self.error = f"stream error: {err.get('message') or err}"
                code = err.get("code") or err.get("status")
                self.error_status = code if isinstance(code, int) else None
            else:
                self.error = f"stream error: {err}"
            return "", ""
        if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
            self.usage_raw = chunk["usage"]
        content, reasoning = "", ""
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            content += _delta_text(delta.get("content"))
            for key in ("reasoning_content", "reasoning", "thinking"):
                reasoning += _delta_text(delta.get(key))
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        self.text += content
        self.reasoning += reasoning
        return content, reasoning


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------

RETRY_STATUS = {408, 409, 425, 429}   # + every 5xx


class _Retryable(Exception):
    def __init__(self, msg: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(msg)
        self.status = status
        self.retry_after = retry_after


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(value)
        return max(0.0, dt.timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def _error_message(status: int, body: bytes) -> str:
    """Short, key-free error string from an error response body."""
    msg = ""
    try:
        data = json.loads(body)
        err = data.get("error", data) if isinstance(data, dict) else data
        if isinstance(err, dict):
            msg = str(err.get("message") or err.get("type") or err)
        else:
            msg = str(err)
    except (ValueError, AttributeError):
        msg = body.decode("utf-8", "replace")
    msg = " ".join(msg.split())
    if len(msg) > 400:
        msg = msg[:400] + "..."
    return f"HTTP {status}: {msg}" if msg else f"HTTP {status}"


class LLMClient:
    """Reusable client: one ``httpx.Client`` (connection pool, keep-alive) for
    all calls, so turn N+1 skips the TLS handshake.

    ``max_retries`` = extra attempts after the first. Backoff is exponential
    (``backoff_base * 2**k``, capped at ``backoff_max``) with jitter in
    [50%, 100%]; a server ``Retry-After`` (capped at ``retry_after_max``)
    takes precedence. ``transport``/``sleep`` are injection points for tests.

    ``usage_grace_s``: after ``stop_when`` fires, keep reading for up to this
    long to catch the trailing usage chunk — early stop otherwise loses
    token/cache accounting for that turn. During the grace period the text keeps
    growing only until the line that completed the reply (the STATUS line) ends,
    so the stored note does not depend on where the router cut its chunks; if
    the grace period ends first, ``LLMResult.truncated`` is set. 0 = close
    immediately (usage is then None). The deadline is checked when a chunk
    arrives; a router that stalls after STATUS can still hold the call up to the
    read timeout (not observed; see docs/FIXLOG.md).

    ``headers``: extra request headers (e.g. a router lease header).
    ``complete(..., timeout_s=, max_retries=)`` override the client defaults
    per call (the planner needs a long read timeout and few retries: a retry
    re-pays the whole think).
    """

    def __init__(self, base_url: str, api_key: str, timeout_s: float, max_retries: int, *,
                 backoff_base: float = 0.5, backoff_max: float = 20.0,
                 retry_after_max: float = 60.0, usage_grace_s: float = 0.0,
                 headers: dict | None = None,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 rng: random.Random | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.retry_after_max = retry_after_max
        self.usage_grace_s = usage_grace_s
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.timeout_s = timeout_s
        hdrs = {str(k): str(v) for k, v in (headers or {}).items()}
        hdrs.update({"Authorization": f"Bearer {api_key}", "Accept": "text/event-stream",
                     "Content-Type": "application/json"})
        self._http = httpx.Client(
            timeout=self._timeout(timeout_s),
            headers=hdrs,
            transport=transport,
        )

    @staticmethod
    def _timeout(t: float) -> httpx.Timeout:
        return httpx.Timeout(t, connect=min(15.0, t))

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:   # never leak the key
        return f"LLMClient(base_url={self.base_url!r}, max_retries={self.max_retries})"

    # -- public --------------------------------------------------------------
    def complete(self, model: str, messages: list[dict], *, max_tokens: int,
                 temperature: float | None = None, extra_body: dict | None = None,
                 stop_when: Callable[[str], bool] | None = None,
                 timeout_s: float | None = None, max_retries: int | None = None) -> LLMResult:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if temperature is not None:
            body["temperature"] = temperature
        if extra_body:
            body.update(extra_body)
        payload = json.dumps(body).encode()       # serialised once; size logged per turn
        retries = self.max_retries if max_retries is None else max(0, int(max_retries))
        timeout = self._timeout(timeout_s) if timeout_s is not None else None

        t_wall0 = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                res = self._attempt(payload, stop_when, timeout)
                res.attempts = attempt
                res.timings.t_wall = time.perf_counter() - t_wall0
                res.request_bytes = len(payload)
                return res
            except _Retryable as e:
                if attempt > retries:
                    return LLMResult(
                        text="", usage=None,
                        timings=Timings(None, None, time.perf_counter() - t_wall0,
                                        t_wall=time.perf_counter() - t_wall0),
                        stopped_early=False, finish_reason=None,
                        error=f"{e} (gave up after {attempt} attempts)",
                        http_status=e.status, attempts=attempt)
                self._sleep(self._backoff(attempt, e.retry_after))

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.retry_after_max)
        d = min(self.backoff_max, self.backoff_base * 2 ** (attempt - 1))
        return d * (0.5 + 0.5 * self._rng.random())

    # -- one HTTP attempt ------------------------------------------------------
    def _attempt(self, payload: bytes, stop_when: Callable[[str], bool] | None,
                 timeout: httpx.Timeout | None = None) -> LLMResult:
        """Run one request. Raises ``_Retryable`` for transient failures that
        happened before a usable reply existed; returns an LLMResult (possibly
        with ``error``) otherwise."""
        t0 = time.perf_counter()
        st = StreamState()
        ttft = ttft_any = t_complete = t_headers = None
        stopped = False
        line_done = False          # the line that completed the reply has ended (newline or end of stream)
        frozen = ("", "")
        stop_len = 0
        headers: dict = {}
        status: int | None = None
        kw = {"timeout": timeout} if timeout is not None else {}
        try:
            with self._http.stream("POST", f"{self.base_url}/chat/completions", content=payload, **kw) as r:
                t_headers = time.perf_counter() - t0
                status = r.status_code
                headers = {k: v for k, v in r.headers.items()
                           if k.lower() not in ("set-cookie", "authorization")}
                if status >= 400:
                    raw = r.read()
                    msg = _error_message(status, raw)
                    if status in RETRY_STATUS or status >= 500:
                        raise _Retryable(msg, status, _parse_retry_after(r.headers.get("retry-after")))
                    return LLMResult("", None, Timings(None, None, time.perf_counter() - t0),
                                     False, None, msg, status, 1, headers=headers)
                ctype = r.headers.get("content-type", "")
                if "text/event-stream" not in ctype and "json" in ctype:
                    # gateway ignored stream=true: treat the JSON body as one chunk
                    data = json.loads(r.read())
                    st.feed(data)
                    ttft = ttft_any = time.perf_counter() - t0 if (st.text or st.reasoning) else None
                    if stop_when is not None and stop_when(st.text):
                        t_complete = ttft
                else:
                    for data in iter_sse_data(r.iter_lines()):
                        if data.strip() == DONE:
                            line_done = True
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue          # tolerate junk keep-alives
                        if not isinstance(chunk, dict):
                            continue
                        dc, dr = st.feed(chunk)
                        now = time.perf_counter() - t0
                        if (dc or dr) and ttft_any is None:
                            ttft_any = now
                        if dc and ttft is None:
                            ttft = now
                        if st.error:
                            break
                        if stopped:
                            if dc and not line_done:
                                txt = frozen[0] + dc
                                nl = txt.find("\n", stop_len)
                                if nl >= 0:
                                    txt, line_done = txt[:nl], True
                                frozen = (txt, frozen[1])
                            if st.finish_reason or st.usage_raw is not None:
                                line_done = True
                            if st.usage_raw is not None or now - t_complete > self.usage_grace_s:
                                break
                            continue
                        if dc and stop_when is not None and stop_when(st.text):
                            t_complete = now
                            stopped = True
                            stop_len = len(st.text)
                            frozen = (st.text, st.reasoning)
                            nl = st.text.find("\n", max(0, len(st.text) - len(dc)))
                            if nl >= 0 and stop_when(st.text[:nl]):
                                frozen, line_done = (st.text[:nl], st.reasoning), True
                            if self.usage_grace_s <= 0:
                                break     # leaving the `with` closes the connection
        except (httpx.TransportError,) as e:   # connect/read/protocol errors, timeouts
            if not stopped:
                raise _Retryable(f"{type(e).__name__}: {e}", status) from None
        except json.JSONDecodeError as e:
            return LLMResult(st.text, None, Timings(ttft, None, time.perf_counter() - t0),
                             False, None, f"bad JSON body: {e}", status, 1, st.reasoning)
        t_end = time.perf_counter() - t0
        if stopped:
            st.text, st.reasoning = frozen
            st.error = None          # anything after the stop point is irrelevant

        if st.error and not st.text:
            es = st.error_status
            if es is None or es in RETRY_STATUS or es >= 500:
                raise _Retryable(st.error, es or status)
        return LLMResult(
            text=st.text, usage=normalize_usage(st.usage_raw),
            timings=Timings(ttft, t_complete, t_end, ttft_any=ttft_any, t_headers=t_headers),
            stopped_early=stopped, finish_reason=st.finish_reason,
            error=st.error, http_status=status, attempts=1, reasoning_text=st.reasoning,
            headers=headers, truncated=stopped and not line_done)
