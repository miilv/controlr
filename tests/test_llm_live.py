"""Live checks against omniroute (CONTROLR_LIVE=1 only; reads .env).

Budget: 8 real calls per full run —
  * 3-turn growing image transcript on no-think Haiku 4.5 (system padded past
    Haiku's 4096-token minimum cacheable prefix) -> cache reads on turns 2-3;
  * the same on Sonnet 5 (1024-token minimum);
  * early stop (immediate close) and early stop with a usage grace period.
Each user turn ends with an IMAGE part, so this also verifies that a
cache_control marker on an image_url part is honoured through omniroute.
Run with ``-s`` to see the numbers.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import numpy as np
import pytest

from controlr.config import load_dotenv
from controlr.llm import LLMClient, Transcript, apply_cache_markers, cache_style_for, encode_image

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(os.environ.get("CONTROLR_LIVE") != "1",
                                 reason="live API test; set CONTROLR_LIVE=1")]

HAIKU = "no-think/claude/claude-haiku-4-5-20251001"
SONNET = "claude/claude-sonnet-5"
STATUS_RE = re.compile(r"^\s*STATUS\s+(OK|DONE|FAIL|STUCK|LIMIT)\b", re.M | re.I)
CALLS: list[str] = []


@pytest.fixture(scope="module")
def client():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    base, key = os.environ.get("OMNIROUTE_BASE_URL"), os.environ.get("OMNIROUTE_API_KEY")
    if not base or not key:
        pytest.skip("OMNIROUTE_BASE_URL / OMNIROUTE_API_KEY not set")
    c = LLMClient(base, key, timeout_s=90, max_retries=1)
    yield c
    c.close()
    print(f"\n[live] real calls made: {len(CALLS)} -> {CALLS}")


def _hdr(r) -> dict:
    return {k: v for k, v in r.headers.items() if "cache" in k.lower() or k.lower().startswith("x-")}


RUN_ID = uuid.uuid4().hex[:12]


def system_text(n_rules: int = 450) -> str:
    """Operating-manual-like padding (~17k tokens) so the system prefix alone
    exceeds Haiku's 4096-token minimum cacheable length.

    The per-run id at the top matters: omniroute also has an exact-match
    RESPONSE cache (measured 2026-10-02: re-sending an identical transcript
    returned the earlier replies in ~0.4 s with the cache counters stripped
    from usage). A unique prefix per run defeats it while keeping prompt
    caching within the run intact."""
    head = (f"[run {RUN_ID}]\nYou control a UR3 robot arm. Reply ONLY with lines of the form\n"
            "MOVE ee_delta <dx_mm> <dy_mm> <dz_mm>\nSTATUS OK|DONE|FAIL\n"
            "Exactly one STATUS line, last. No other text.\n\nManual:\n")
    rules = [f"Rule {i:04d}: when the gripper is {('open', 'closed')[i % 2]} and the target is "
             f"{i % 37} mm to the {('left', 'right', 'front', 'back')[i % 4]}, prefer steps of "
             f"{5 + i % 11} mm and re-check the camera view." for i in range(n_rules)]
    return head + "\n".join(rules)


def frame(k: int) -> np.ndarray:
    """Deterministic synthetic 448x336 'scene' with a block that moves per turn."""
    h, w = 336, 448
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx * 255 // w), (yy * 255 // h), np.full_like(xx, 90)], -1).astype(np.uint8)
    x0 = 60 + 40 * k
    img[150:200, x0:x0 + 50] = (220, 30, 30)
    return img


def _f(x: float | None) -> str:
    return "None" if x is None else f"{x:.2f}s"


def run_growing(client: LLMClient, model: str) -> list:
    style = cache_style_for(model, "auto")
    assert style == "anthropic"
    tr = Transcript(system_text())
    results = []
    for k in range(3):
        tr.add_user([f"TURN {k}. Task: move the TCP towards the red block. Camera:",
                     encode_image(frame(k), 85, "scene")])
        msgs = apply_cache_markers(tr.to_messages(), style)
        r = client.complete(model, msgs, max_tokens=400, temperature=0.0)
        CALLS.append(model)
        assert r.error is None, r.error
        u = r.usage
        assert u is not None, "no usage chunk"
        print(f"[live] {model} turn {k}: ttft_any={_f(r.timings.ttft_any)} ttft={_f(r.timings.ttft)} "
              f"t_end={_f(r.timings.t_end)} prompt={u.prompt_tokens} read={u.cache_read_tokens} "
              f"write={u.cache_write_tokens} out={u.completion_tokens} finish={r.finish_reason} "
              f"raw={u.raw} text={r.text!r} reasoning={len(r.reasoning_text)} chars hdr={_hdr(r)}")
        results.append(r)
        tr.add_assistant(r.text)
    return results


CACHE_KEYS = ("cache_read_input_tokens", "cache_creation_input_tokens", "prompt_tokens_details")


def _check_cache(results: list) -> None:
    if not any(k in r.usage.raw for r in results for k in CACHE_KEYS):
        # Measured 2026-10-02: the no-think/claude/* route returns only
        # prompt/completion/total — cache counters are dropped, so cache reads
        # cannot be verified from usage on that route (latency suggests hits).
        pytest.xfail("route reports no cache counters in usage")
    for r in results[1:]:
        u = r.usage
        assert u.cache_read_tokens > 0.8 * u.prompt_tokens, u.raw
        assert u.cache_write_tokens < 0.2 * u.prompt_tokens, u.raw
    assert results[2].usage.cache_read_tokens > results[1].usage.cache_read_tokens


def test_haiku_growing_image_transcript_cache_reads(client):
    _check_cache(run_growing(client, HAIKU))


def test_sonnet_growing_image_transcript_cache_reads(client):
    _check_cache(run_growing(client, SONNET))


EARLY_PROMPT = [
    {"role": "system", "content": f"[run {RUN_ID}] Follow the output format exactly."},
    {"role": "user", "content": "Output exactly this first line:\nSTATUS OK\n"
                                "then write a 150-word paragraph about robot arms."},
]


def test_early_stop_closes_stream(client):
    r = client.complete(HAIKU, EARLY_PROMPT, max_tokens=400, temperature=0.0,
                        stop_when=lambda s: bool(STATUS_RE.search(s)))
    CALLS.append(HAIKU)
    print(f"[live] early stop: text={r.text!r} t_complete={r.timings.t_complete} "
          f"t_end={r.timings.t_end:.2f}s usage={r.usage}")
    assert r.error is None and r.stopped_early
    assert STATUS_RE.search(r.text) and len(r.text) < 60
    assert r.timings.t_complete is not None and r.timings.t_end - r.timings.t_complete < 0.5


def test_early_stop_with_usage_grace(client):
    c = LLMClient(client.base_url, os.environ["OMNIROUTE_API_KEY"], timeout_s=90, max_retries=1,
                  usage_grace_s=0.5)
    try:
        r = c.complete(HAIKU, EARLY_PROMPT, max_tokens=400, temperature=0.0,
                       stop_when=lambda s: bool(STATUS_RE.search(s)))
    finally:
        c.close()
    CALLS.append(HAIKU)
    print(f"[live] early stop + grace: text={r.text!r} t_complete={r.timings.t_complete} "
          f"t_end={r.timings.t_end:.2f}s usage={r.usage}")
    assert r.error is None and r.stopped_early and len(r.text) < 60
    assert r.timings.t_end - r.timings.t_complete <= 1.0
