"""Scripted stand-in for ``LLMClient`` — loop tests run without network.

Same ``complete()`` signature and the same ``LLMResult`` shape, including
early stop: the scripted reply is "streamed" in small chunks; once ``stop_when``
turns True the text is completed to the end of that line, as the real client
does during its usage grace period. Timings and usage are synthetic but
plausible (cache reads grow with the transcript like a warm Anthropic cache)
and usage is returned on early stops too (like the real client with grace), so
run logs / summaries written from fake runs have every field populated.
``is_fake = True`` lets the loop mark dry runs in the run log.
"""

from __future__ import annotations

import json
from typing import Callable

from .client import LLMResult, Timings, Usage

Reply = str | Exception


class FakeLLM:
    """``replies``: a list consumed in order (the last one repeats when the
    list runs out; an ``Exception`` instance yields an error result), or a
    callable ``f(messages) -> str``. Every call is recorded in ``calls``."""

    is_fake = True

    def __init__(self, replies: list[Reply] | Callable[[list[dict]], str], *,
                 chunk_chars: int = 8, ttft_s: float = 0.4, s_per_chunk: float = 0.01) -> None:
        if isinstance(replies, list) and not replies:
            raise ValueError("FakeLLM needs at least one reply")
        self.replies = replies
        self.chunk_chars = max(1, chunk_chars)
        self.ttft_s = ttft_s
        self.s_per_chunk = s_per_chunk
        self.calls: list[dict] = []
        self._i = 0
        self._prev_prompt = 0

    def close(self) -> None:
        pass

    def _next(self, messages: list[dict]) -> Reply:
        if callable(self.replies):
            return self.replies(messages)
        r = self.replies[min(self._i, len(self.replies) - 1)]
        self._i += 1
        return r

    def complete(self, model: str, messages: list[dict], *, max_tokens: int,
                 temperature: float | None = None, extra_body: dict | None = None,
                 stop_when: Callable[[str], bool] | None = None,
                 timeout_s: float | None = None, max_retries: int | None = None) -> LLMResult:
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature, "extra_body": extra_body,
                           "timeout_s": timeout_s, "max_retries": max_retries})
        reply = self._next(messages)
        if isinstance(reply, Exception):
            return LLMResult(text="", usage=None, timings=Timings(None, None, 0.05, t_wall=0.05),
                             stopped_early=False, finish_reason=None, error=str(reply),
                             http_status=None, attempts=1)

        text, t, t_complete, stopped = "", self.ttft_s, None, False
        for k in range(0, len(reply), self.chunk_chars):
            text += reply[k:k + self.chunk_chars]
            t += self.s_per_chunk
            if stop_when is not None and stop_when(text):
                t_complete, stopped = t, True
                nl = reply.find("\n", len(text) - self.chunk_chars)
                while nl >= 0 and not stop_when(reply[:nl]):
                    nl = reply.find("\n", nl + 1)
                text = reply[:nl] if nl >= 0 else reply      # finish the STATUS line
                break
        finish = None if stopped else "stop"
        if not stopped and max_tokens and len(text) > 4 * max_tokens:
            text, finish = text[:4 * max_tokens], "length"

        # ~4 chars/token for text, a flat 300 tokens per image; everything
        # seen in the previous call counts as a cache read.
        n_img = sum(1 for m in messages if isinstance(m.get("content"), list)
                    for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url")
        n_chars = sum(len(json.dumps(m.get("content"))) for m in messages
                      if not (isinstance(m.get("content"), list)
                              and any(isinstance(p, dict) and p.get("type") == "image_url"
                                      for p in m["content"])))
        prompt = n_chars // 4 + 300 * n_img + 50 * len(messages)
        read = min(self._prev_prompt, prompt)
        self._prev_prompt = prompt
        usage = Usage(
            prompt_tokens=prompt, completion_tokens=max(1, len(text) // 4),
            cache_read_tokens=read, cache_write_tokens=prompt - read, reasoning_tokens=0,
            raw={"fake": True})
        return LLMResult(text=text, usage=usage,
                         timings=Timings(self.ttft_s, t_complete, t, ttft_any=self.ttft_s, t_wall=t),
                         stopped_early=stopped, finish_reason=finish, error=None,
                         http_status=200, attempts=1, request_bytes=n_chars + 40_000 * n_img)
