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
@pytest.mark.parametrize("route", ["off", "lateral"])
def test_arrange_hover(monkeypatch, mode, expected, route):
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills, SkillResult
    rows = [dict(label=lab, category=cat, category_idx=0, type=typ, pos=[x, -0.2, 0.785],
                 quat=[1, 0, 0, 0], half=[0.02, 0.02, 0.02])
            for lab, cat, typ, x in (("item", "mouse", "Rigid", 0.), ("pad", "mousemat", "Geometry", 0.1))]
    env = KinematicEnv(rows, step_lim=300)
    sk = Skills(env, GTProvider(env, None), reserve=30)
    sk.arrange_hover = mode
    sk.arrange_route = route
    seen = []
    monkeypatch.setattr(library, "container_obstacle", lambda *a: None)
    monkeypatch.setattr(library, "_on_targets", lambda *a: ([np.eye(4)], False))
    monkeypatch.setattr(library, "place_oriented", lambda *a, **kw: seen.append(kw) or
                        SkillResult(True, "place", "ok", 0, None))
    library.arrange(sk, [("place", "item", "pad")])
    assert seen[0]["place_kw"] == {**expected, **({"approach_lateral": True} if route == "lateral" else {})}
    assert seen[0]["ignore"] == ("pad",)


def test_arrange_hover_config():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().arrange_hover == "off"
    assert SKILL_ATTRS["arrange_hover"] == ("off", "low")
    RuntimePolicy(RuntimeConfig(arrange_hover="low"))
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(arrange_hover="bad"))


def test_tilted_face_reaches_where_vertical_does_not():
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills
    rows = [dict(label="wide", category="keyboard", category_idx=0, pos=[0., 0.1, 0.78],
                 quat=[1, 0, 0, 0], half=[0.2, 0.06, 0.015])]
    env = KinematicEnv(rows, step_lim=1000)
    sk = Skills(env, GTProvider(env, None))
    o = sk.world.obj("wide")
    assert library._push_stroke(sk, o, "wide", np.array([1., 0.]), 0.05) is None
    sk.push_approach = "tilted"
    plan = library._push_stroke(sk, o, "wide", np.array([1., 0.]), 0.05)
    assert plan is not None and plan.steps > 0


@pytest.mark.parametrize("status", ["reached", "ik_fail", "partial"])
def test_lateral_keeps_direct_result(monkeypatch, status):
    from dojo.skills.motion import Motion, MoveResult
    calls = []
    motion = Motion.__new__(Motion)
    direct = MoveResult(status, "left", [])
    monkeypatch.setattr(motion, "to_pose", lambda *a, **kw: calls.append((a, kw)) or direct)
    assert motion.lateral("left", np.zeros(6), np.eye(4)) is direct
    assert len(calls) == 1


@pytest.mark.parametrize("success", [False, True])
def test_lateral_checks_whole_route(monkeypatch, success):
    from dojo.skills.motion import Motion, MoveResult, Segment
    motion = Motion.__new__(Motion)
    calls = []
    target = np.eye(4)
    direct = MoveResult("collision", "left", [])
    reached = MoveResult("reached", "left", [Segment(np.stack([np.zeros(6), np.ones(6)]), .03)])
    def transit(arm, q, T, *args, **kw):
        calls.append((q, T.copy(), args, kw))
        if len(calls) == 1:
            return direct
        return reached if success or not np.array_equal(T, target) else direct
    monkeypatch.setattr(motion, "to_pose", transit)
    held = object()
    result = motion.lateral("left", np.zeros(6), target, .03, "carry", held=held, ignore=("item",), ease=5)
    assert result.ok == success
    assert len(calls) == (3 if success else 9)
    if success:
        assert len(result.segments) == 2
    else:
        assert result is direct
    np.testing.assert_array_equal(target, np.eye(4))
    for _, _, args, kw in calls:
        assert args == (.03, "carry") and kw == dict(held=held, ignore=("item",), ease=5)


def test_arrange_route_config():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().arrange_route == "off"
    assert SKILL_ATTRS["arrange_route"] == ("off", "lateral")
    RuntimePolicy(RuntimeConfig(arrange_route="lateral"))
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(arrange_route="bad"))


@pytest.mark.parametrize("enabled", [False, True])
def test_lateral_place_dispatch(enabled):
    from unittest.mock import Mock
    from dojo.skills.skills import Skills, SkillResult
    sk = object.__new__(Skills)
    h = SimpleNamespace(label="item", T_tcp_obj=np.eye(4), box=object())
    sk.held = {"left": h}
    sk.ex = SimpleNamespace(q={"left": np.zeros(6)}, g={"left": 0.0})
    sk.world = SimpleNamespace(obj=lambda _: SimpleNamespace(asset=SimpleNamespace(center=np.zeros(3))))
    bad = SimpleNamespace(ok=False, note="collision", segments=[])
    sk.motion = SimpleNamespace(to_pose=Mock(return_value=bad), lateral=Mock(return_value=bad))
    result = sk.plan_place("left", np.eye(4), approach_lateral=enabled)
    assert isinstance(result, SkillResult) and not result.ok
    chosen = sk.motion.lateral if enabled else sk.motion.to_pose
    other = sk.motion.to_pose if enabled else sk.motion.lateral
    chosen.assert_called_once()
    other.assert_not_called()
    kw = chosen.call_args.kwargs
    assert kw['held'] is h.box and kw['ignore'] == ('item',) and 'check' not in kw
