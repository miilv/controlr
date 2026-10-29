"""``Skills.arrange_order`` / ``RuntimeConfig.arrange_order``: "near" (today) | "big_first" (the largest object placed
first)."""

from __future__ import annotations

import numpy as np
import pytest

from dojo.skills import library
from dojo.skills.env import KinematicEnv
from dojo.skills.skills import GTProvider, SkillResult, Skills

Z = 0.765


def _rows():
    return [dict(label="small", category="mouse", category_idx=0, pos=[0.0, -0.2, Z + 0.015], quat=[1, 0, 0, 0],
                 half=[0.03, 0.02, 0.015]),
            dict(label="big", category="garage", category_idx=0, pos=[-0.35, -0.05, Z + 0.09], quat=[1, 0, 0, 0],
                 half=[0.05, 0.04, 0.09]),
            dict(label="pad_a", category="mousemat", category_idx=0, type="Geometry", pos=[0.1, -0.1, Z + 0.002],
                 quat=[1, 0, 0, 0], half=[0.08, 0.08, 0.002]),
            dict(label="pad_b", category="cube_cushion", category_idx=0, type="Geometry", pos=[-0.2, -0.1, Z + 0.02],
                 quat=[1, 0, 0, 0], half=[0.05, 0.05, 0.02])]


@pytest.mark.parametrize("order,expected", [("near", ["small", "big"]), ("big_first", ["big", "small"]),
                                            ("push_big_first", ["kb", "big", "small"])])
def test_arrange_order(monkeypatch, order, expected):
    env = KinematicEnv(_rows(), step_lim=300)
    sk = Skills(env, GTProvider(env, None), reserve=30)
    sk.arrange_order = order
    done = []
    monkeypatch.setattr(library, "container_obstacle", lambda sk_, sup: None)
    monkeypatch.setattr(library, "_on_targets", lambda sk_, obj, sup: ([np.eye(4)], False))
    monkeypatch.setattr(library, "place_oriented", lambda sk_, obj, tg, **kw: done.append(obj) or SkillResult(
        True, "place", "ok", 0, None))
    monkeypatch.setattr(library, "push_to", lambda sk_, obj, xy: done.append("kb") or SkillResult(True, "push", "ok", 0, None))
    r = library.arrange(sk, [("place", "small", "pad_a"), ("push", "kbd", "pad_a"), ("place", "big", "pad_b")])
    assert r["ok"] and [d for d in done if order == "push_big_first" or d != "kb"] == expected


def test_arrange_order_default_near():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().arrange_order == "near" and SKILL_ATTRS["arrange_order"][0] == "near"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(arrange_order="random"))


def _kb_rows():
    return [dict(label="keyboard", category="keyboard", category_idx=0, pos=[0.0, -0.22, Z + 0.0125], quat=[1, 0, 0, 0],
                 half=[0.22, 0.065, 0.0125]),
            dict(label="wall", category="drawer", category_idx=0, pos=[-0.27, -0.22, Z + 0.1], quat=[1, 0, 0, 0],
                 half=[0.03, 0.12, 0.1])]


@pytest.mark.parametrize("stroke", ["face", "press"])
def test_press_stroke_when_no_face_stroke_plans(stroke):
    """The sideways face stroke would start 0.24 m out from the keyboard's centre, inside a box: "face" (today) has no
    stroke; "press" drags it with the fingertips on its top."""
    env = KinematicEnv(_kb_rows(), step_lim=400)
    logs = []
    sk = Skills(env, GTProvider(env, None), reserve=30, log=logs.append)
    sk.push_stroke = stroke
    r = library.push_to(sk, "keyboard", np.array([0.08, -0.22]))
    if stroke == "face":
        assert r.status == "ik_fail" and "no push stroke" in r.note and r.steps == 0
    else:                                   # the kinematic env does not drag the keyboard: the stroke plans and runs
        assert r.steps > 0 and any("press stroke" in m for m in logs), (r.status, r.note)


def test_push_stroke_default_face():
    from dojo.runtime.policy import RuntimeConfig
    assert RuntimeConfig().push_stroke == "face"
