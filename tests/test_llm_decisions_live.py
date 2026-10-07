"""Live probe of the decision head (CONTROLR_LIVE=1 only; OPENROUTER_API_KEY from .env).

Budget: 4 real calls. The Decisions schema does not say how images go into ``state``; this checks,
for each ``decisions.image_mode``, that the model SEES the frame: a red square on the left vs the
right half of a synthetic image must flip a left/right choice. Run with ``-s`` for the numbers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest

from controlr.config import load_config, load_dotenv
from controlr.llm import encode_image
from controlr.llm.decisions import build_state

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(os.environ.get("CONTROLR_LIVE") != "1",
                                 reason="live API test; set CONTROLR_LIVE=1")]

MODEL = "openai/gpt-6-luna-decisions"
QUESTION = {"side": {"type": "choice", "instructions": "Where is the red square in the image?",
                     "criteria": {"left": "in the left half of the image",
                                  "right": "in the right half of the image"}}}


def _frame(left: bool) -> str:
    img = np.full((240, 320, 3), 235, np.uint8)
    x0 = 30 if left else 220
    img[90:160, x0:x0 + 70] = (220, 30, 30)
    return encode_image(img, 90, "scene").data_url


@pytest.mark.parametrize("mode", ["parts", "field"])
def test_the_model_sees_the_frame(mode):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    cfg = load_config(None, [f"decisions.image_mode={mode}", "decisions.include_manual=false"], dotenv=None)
    got = {}
    with httpx.Client(timeout=60, headers={"Authorization": f"Bearer {key}"}) as http:
        for left in (True, False):
            msgs = [{"role": "user", "content": [{"type": "text", "text": "TASK: look"},
                                                 {"type": "image_url", "image_url": {"url": _frame(left)}}]}]
            r = http.post(f"{cfg.decisions.base_url}/decisions",
                          json={"model": MODEL, "state": build_state(msgs, cfg.decisions), "questions": QUESTION})
            print(f"\n[{mode} left={left}] {r.status_code} {r.elapsed.total_seconds():.2f}s {r.text[:400]}")
            assert r.status_code == 200, r.text[:300]
            got[left] = r.json()["answers"]["side"]
    assert got[True]["choice"] == "left" and got[False]["choice"] == "right", json.dumps(got)
