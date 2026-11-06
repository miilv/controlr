"""Reachable face pushes preserve contact geometry across approach orientations."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills import library
from dojo.skills.kinematics import TIP_BEYOND_TCP


@pytest.mark.parametrize("mode,n", [("off", 2), ("tilted", 6)])
def test_face_candidates_keep_tip_path(mode, n):
    calls = []

    class Motion:
        def to_pose(self, arm, q, T, *args, **kw):
            calls.append((arm, T.copy(), kw))
            return SimpleNamespace(ok=False)

    o = SimpleNamespace(center_w=np.array([0.0, 0.1, 0.78]), R=np.eye(3),
                        asset=SimpleNamespace(mesh=None, extents=np.array([0.4, 0.12, 0.02])),
                        top_z=0.79, bottom_z=0.77)
    sk = SimpleNamespace(push_approach=mode, motion=Motion(),
                         kin=SimpleNamespace(base=lambda a: np.eye(4)),
                         ex=SimpleNamespace(q={a: np.zeros(6) for a in ("left", "right")}))
    assert library._push_stroke(sk, o, "wide", np.array([1., 0.]), 0.05) is None
    assert len(calls) == n * 2
    expected_tip = np.array([-0.22, 0.1, o.top_z + 0.03])
    for _, T, kw in calls:
        np.testing.assert_allclose(T[:3, 3] + T[:3, 0] * TIP_BEYOND_TCP, expected_tip, atol=1e-14)
        assert kw["ignore"] == ("wide",) and kw["gripper"] == library.CLOSED
    if mode == "tilted":
        assert any(T[2, 0] > -0.9 for _, T, _ in calls)


def test_push_approach_config():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().push_approach == "off"
    assert SKILL_ATTRS["push_approach"] == ("off", "tilted")
    RuntimePolicy(RuntimeConfig(push_approach="tilted"))
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(push_approach="bad"))


@pytest.mark.parametrize("mode,expected", [("off", {"pre_h": 0.04}),
                                          ("low", {"pre_h": 0.005, "pre_clear": 0.003})])
def test_arrange_hover(monkeypatch, mode, expected):
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills, SkillResult
    rows = [dict(label=lab, category=cat, category_idx=0, type=typ, pos=[x, -0.2, 0.785],
                 quat=[1, 0, 0, 0], half=[0.02, 0.02, 0.02])
            for lab, cat, typ, x in (("item", "mouse", "Rigid", 0.), ("pad", "mousemat", "Geometry", 0.1))]
    env = KinematicEnv(rows, step_lim=300)
    sk = Skills(env, GTProvider(env, None), reserve=30)
    sk.arrange_hover = mode
    seen = []
    monkeypatch.setattr(library, "container_obstacle", lambda *a: None)
    monkeypatch.setattr(library, "_on_targets", lambda *a: ([np.eye(4)], False))
    monkeypatch.setattr(library, "place_oriented", lambda *a, **kw: seen.append(kw) or
                        SkillResult(True, "place", "ok", 0, None))
    library.arrange(sk, [("place", "item", "pad")])
    assert seen[0]["place_kw"] == expected
    assert seen[0]["ignore"] == ("pad",)


def test_arrange_hover_config():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().arrange_hover == "off"
    assert SKILL_ATTRS["arrange_hover"] == ("off", "low")
    RuntimePolicy(RuntimeConfig(arrange_hover="low"))
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(arrange_hover="bad"))
