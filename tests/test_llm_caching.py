"""Cache-marker placement and routing."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from controlr.llm.caching import MAX_MARKERS, apply_cache_markers, cache_style_for
from controlr.llm.transcript import Transcript, encode_image


def markers(msgs):
    out = []
    for i, m in enumerate(msgs):
        c = m.get("content")
        if isinstance(c, list):
            for j, p in enumerate(c):
                if isinstance(p, dict) and "cache_control" in p:
                    out.append((i, j, p["cache_control"]))
    return out


@pytest.mark.parametrize("model,style", [
    ("claude/claude-sonnet-5", "anthropic"),
    ("cc/claude-sonnet-5", "anthropic"),
    ("no-think/claude/claude-haiku-4-5-20251001", "anthropic"),
    ("claude/claude-opus-5-xhigh", "anthropic"),
    ("gpt-5.6-luna-low", "none"),
    ("cx/gpt-6-astra", "none"),
    ("dva/gemini-3-pro", "none"),
])
def test_cache_style_auto(model, style):
    assert cache_style_for(model, "auto") == style


def test_cache_style_configured_overrides():
    assert cache_style_for("gpt-5.6", "anthropic") == "anthropic"
    assert cache_style_for("claude/x", "none") == "none"
    with pytest.raises(ValueError):
        cache_style_for("claude/x", "bogus")


def _transcript(n_turns: int) -> Transcript:
    tr = Transcript("SYSTEM manual")
    img = encode_image(np.full((16, 16, 3), 100, np.uint8), 90, "scene")
    for k in range(n_turns):
        tr.add_user([f"TURN {k}", img])
        tr.add_assistant(f"MOVE ee_delta {k} 0 0\nSTATUS OK")
    tr.add_user([f"TURN {n_turns}", img, "feedback last"])
    return tr


def test_anthropic_markers_positions_and_count():
    msgs = _transcript(3).to_messages()
    orig = copy.deepcopy(msgs)
    out = apply_cache_markers(msgs, "anthropic")
    assert msgs == orig                       # input untouched (deep copy)
    mk = markers(out)
    assert len(mk) == 3 <= MAX_MARKERS
    user_idx = [i for i, m in enumerate(out) if m["role"] == "user"]
    # system string converted to a single text part with the marker
    assert out[0]["content"] == [{"type": "text", "text": "SYSTEM manual",
                                  "cache_control": {"type": "ephemeral"}}]
    last, prev = user_idx[-1], user_idx[-2]
    assert (last, len(out[last]["content"]) - 1) in [(i, j) for i, j, _ in mk]
    assert (prev, len(out[prev]["content"]) - 1) in [(i, j) for i, j, _ in mk]
    assert out[prev]["content"][-1]["type"] == "image_url"   # image parts can carry markers


def test_ttl_1h_and_bad_ttl():
    out = apply_cache_markers(_transcript(1).to_messages(), "anthropic", ttl="1h")
    assert all(cc == {"type": "ephemeral", "ttl": "1h"} for _, _, cc in markers(out))
    with pytest.raises(ValueError):
        apply_cache_markers(_transcript(1).to_messages(), "anthropic", ttl="2d")


def test_none_style_is_plain_copy():
    msgs = _transcript(2).to_messages()
    out = apply_cache_markers(msgs, "none")
    assert out == msgs and out is not msgs and markers(out) == []


def test_first_turn_only_two_markers():
    tr = Transcript("sys")
    tr.add_user(["task"])
    out = apply_cache_markers(tr.to_messages(), "anthropic")
    assert len(markers(out)) == 2
    assert out[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_string_user_content_converted_and_empty_skipped():
    msgs = [{"role": "system", "content": ""},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"}]
    out = apply_cache_markers(msgs, "anthropic")
    assert out[0]["content"] == ""            # empty text can't carry a marker
    assert out[1]["content"] == [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}]
    assert out[3]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert out[2]["content"] == "b"           # assistant never marked


def test_never_more_than_four_markers_with_preexisting():
    msgs = [{"role": "system", "content": [{"type": "text", "text": "s"}]}]
    for k in range(6):
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": f"u{k}", "cache_control": {"type": "ephemeral"}} if k < 3
            else {"type": "text", "text": f"u{k}"}]})
        msgs.append({"role": "assistant", "content": "ok"})
    msgs.append({"role": "user", "content": [{"type": "text", "text": "last"}]})
    out = apply_cache_markers(msgs, "anthropic")
    assert len(markers(out)) == 4
    # priority: system marker first
    assert "cache_control" in out[0]["content"][-1]


def test_markers_move_but_prefix_stays_stable():
    """Between two consecutive requests, the only differences in the shared
    prefix are cache_control keys (the moving marker)."""
    tr = _transcript(2)
    a = apply_cache_markers(tr.to_messages(), "anthropic")
    tr.add_assistant("STATUS OK")
    tr.add_user(["next", encode_image(np.zeros((8, 8, 3), np.uint8), 90, "x")])
    b = apply_cache_markers(tr.to_messages(), "anthropic")

    def strip(ms):
        ms = copy.deepcopy(ms)
        for m in ms:
            if isinstance(m["content"], list):
                for p in m["content"]:
                    p.pop("cache_control", None)
        return ms

    k = len(a)
    assert json.dumps(strip(a)[1:]) == json.dumps(strip(b)[1:k])
    # the system message is byte-identical including its marker
    assert json.dumps(a[0]) == json.dumps(b[0])
