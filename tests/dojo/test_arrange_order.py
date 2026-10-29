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


@pytest.mark.parametrize("order,expected", [("near", ["small", "big"]), ("big_first", ["big", "small"])])
def test_arrange_order(monkeypatch, order, expected):
    env = KinematicEnv(_rows(), step_lim=300)
    sk = Skills(env, GTProvider(env, None), reserve=30)
    sk.arrange_order = order
    done = []
    monkeypatch.setattr(library, "container_obstacle", lambda sk_, sup: None)
    monkeypatch.setattr(library, "_on_targets", lambda sk_, obj, sup: ([np.eye(4)], False))
    monkeypatch.setattr(library, "place_oriented", lambda sk_, obj, tg, **kw: done.append(obj) or SkillResult(
        True, "place", "ok", 0, None))
    r = library.arrange(sk, [("place", "small", "pad_a"), ("place", "big", "pad_b")])
    assert r["ok"] and done == expected


def test_arrange_order_default_near():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().arrange_order == "near" and SKILL_ATTRS["arrange_order"][0] == "near"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(arrange_order="random"))
