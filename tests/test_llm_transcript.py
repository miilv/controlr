"""Transcript: append-only, byte-stable prefix, deterministic image encoding."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from controlr.llm.transcript import (EMPTY_ASSISTANT_PLACEHOLDER, ImagePart, Transcript,
                                     encode_image)


def frame(seed: int, h: int = 48, w: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_encode_image_deterministic_and_valid_jpeg():
    a = encode_image(frame(0), 85, "scene")
    b = encode_image(frame(0), 85, "scene")
    assert a.jpeg == b.jpeg and a.sha == b.sha == hashlib.sha256(a.jpeg).hexdigest()
    assert (a.width, a.height, a.label) == (64, 48, "scene")
    im = Image.open(io.BytesIO(a.jpeg))
    assert im.format == "JPEG" and im.size == (64, 48)
    assert a.data_url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(a.data_url.split(",", 1)[1]) == a.jpeg
    assert a.data_url is a.data_url           # computed once, stored


def test_encode_image_quality_and_content_change_bytes():
    assert encode_image(frame(0), 50, "").sha != encode_image(frame(0), 95, "").sha
    assert encode_image(frame(0), 90, "").sha != encode_image(frame(1), 90, "").sha


def test_encode_image_accepts_gray_rgba_float():
    assert encode_image(np.zeros((4, 5), np.uint8), 90, "g").width == 5
    assert encode_image(np.zeros((4, 5, 4), np.uint8), 90, "a").height == 4
    assert encode_image(np.full((4, 5, 3), 300.0), 90, "f").jpeg
    with pytest.raises(ValueError):
        encode_image(np.zeros((4, 5, 2), np.uint8), 90, "bad")


def test_image_part_frozen_and_from_jpeg():
    p = encode_image(frame(2), 90, "s")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.label = "x"  # type: ignore[misc]
    q = ImagePart.from_jpeg(p.jpeg, "s")
    assert q == p and q.data_url == p.data_url


def test_to_messages_format():
    tr = Transcript("manual")
    img = encode_image(frame(3), 90, "scene")
    tr.add_user(["task", img, ""])
    tr.add_assistant("MOVE ee_delta 1 2 3\nSTATUS OK")
    msgs = tr.to_messages()
    assert msgs[0] == {"role": "system", "content": "manual"}
    assert msgs[1] == {"role": "user", "content": [
        {"type": "text", "text": "task"},
        {"type": "image_url", "image_url": {"url": img.data_url}}]}   # empty text dropped
    assert msgs[2] == {"role": "assistant", "content": "MOVE ee_delta 1 2 3\nSTATUS OK"}
    assert "cache_control" not in json.dumps(msgs)
    assert tr.n_images() == 1 and len(tr) == 3 and tr.images() == [img]


def test_prefix_byte_stable_across_growth():
    tr = Transcript("manual " * 50)
    prev: list[str] = []
    for k in range(6):
        tr.add_user([f"TURN {k}", encode_image(frame(k), 90, "scene"), f"STATE {k}"])
        cur = [json.dumps(m) for m in tr.to_messages()]
        assert cur[:len(prev)] == prev
        tr.add_assistant(f"MOVE ee_delta {k} 0 0\nSTATUS OK" if k % 2 else "MOVE ee_del")  # truncated
        prev = [json.dumps(m) for m in tr.to_messages()]
        assert prev[:len(cur)] == cur
    assert tr.n_images() == 6


def test_mutating_returned_messages_does_not_touch_transcript():
    tr = Transcript("s")
    tr.add_user(["a"])
    m = tr.to_messages()
    m[1]["content"][0]["cache_control"] = {"type": "ephemeral"}
    m[0]["content"] = "changed"
    assert tr.to_messages() == [{"role": "system", "content": "s"},
                                {"role": "user", "content": [{"type": "text", "text": "a"}]}]


def test_assistant_stored_exactly_empty_serialised_as_placeholder():
    tr = Transcript("s")
    tr.add_user(["a"])
    tr.add_assistant("")
    tr.add_user(["b"])
    assert tr.to_messages()[2]["content"] == EMPTY_ASSISTANT_PLACEHOLDER
    assert tr.to_log_records()[2]["content"] == ""


def test_log_records_replace_images_with_sha():
    tr = Transcript("s")
    img = encode_image(frame(4), 90, "wrist")
    tr.add_user(["obs", img])
    recs = tr.to_log_records()
    assert recs[1]["content"][1] == {"type": "image", "image_sha": img.sha, "label": "wrist",
                                     "width": 64, "height": 48}
    assert "base64" not in json.dumps(recs)


def test_add_user_validation():
    tr = Transcript("s")
    with pytest.raises(ValueError):
        tr.add_user([])
    with pytest.raises(ValueError):
        tr.add_user([""])
    with pytest.raises(TypeError):
        tr.add_user([b"bytes"])  # type: ignore[list-item]
