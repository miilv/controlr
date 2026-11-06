"""Slot carry experiments must leave default transits unchanged and checked."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dojo.skills.skills import Skills, SkillResult


@pytest.mark.parametrize('linear', [False, True])
def test_place_approach_selects_checked_cartesian_path(linear):
    sk = object.__new__(Skills)
    h = SimpleNamespace(label='tool', T_tcp_obj=np.eye(4), box=object())
    sk.held = {'left': h}
    sk.ex = SimpleNamespace(q={'left': np.zeros(6)}, g={'left': 0.0})
    sk.world = SimpleNamespace(obj=lambda _: SimpleNamespace(asset=SimpleNamespace(center=np.zeros(3))))
    # A refused transit must remain refused: never fall back to an orientation-changing path.
    bad = SimpleNamespace(ok=False, note='ik_fail: blocked waypoint', segments=[])
    sk.motion = SimpleNamespace(line=Mock(return_value=bad), to_pose=Mock(return_value=bad))
    result = sk.plan_place('left', np.eye(4), approach_line=linear)
    assert isinstance(result, SkillResult) and not result.ok
    chosen = sk.motion.line if linear else sk.motion.to_pose
    other = sk.motion.to_pose if linear else sk.motion.line
    chosen.assert_called_once()
    other.assert_not_called()
    kw = chosen.call_args.kwargs
    assert kw['held'] is h.box and kw['ignore'] == ('tool',)
    assert kw['gripper'] == 0.0
    assert 'check' not in kw  # both planners' collision checking defaults stay on


@pytest.mark.parametrize('mode', ['off', 'linear'])
@pytest.mark.parametrize('fallback', [False, True])
def test_slot_switch_reaches_both_level_and_reorienting_transfers(monkeypatch, mode, fallback):
    from dojo.skills import library
    obj = SimpleNamespace(R=np.eye(3), center_w=np.zeros(3), asset=SimpleNamespace(extents=np.ones(3)))
    world = SimpleNamespace(objects={'tool': obj}, obj=lambda _: obj)
    sk = SimpleNamespace(refresh=lambda: world, slot_approach=mode, log=lambda _: None,
                         held={'left': None, 'right': None}, bank=lambda: None)
    monkeypatch.setattr(library, 'container_obstacle', lambda *a: None)
    monkeypatch.setattr(library, 'slot_targets', lambda *a: [np.eye(4)])
    monkeypatch.setattr(library, '_level_grasps', lambda *a: None)
    calls = []
    def transfer(*args, **kw):
        calls.append(kw)
        if fallback and len(calls) == 1:
            return SkillResult(False, 'transfer', 'ik_fail', 0, None, note='no path')
        return SkillResult(True, 'place', 'ok', 1, 'left')
    monkeypatch.setattr(library, 'place_oriented', transfer)
    assert library.place_in_slots(sk, ['tool'], 'box')['ok']
    assert len(calls) == (2 if fallback else 1)
    for kw in calls:
        assert kw['place_kw'].get('approach_line', False) == (mode == 'linear')
        assert kw['place_kw']['drop_ok'] is True


def test_runtime_default_and_invalid_switch():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    assert RuntimeConfig().slot_approach == 'off'
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(slot_approach='typo'))


def test_linear_place_keeps_easing_and_tail_options_separate():
    from dojo.skills.executor import ArmPlan
    from dojo.skills.motion import MoveResult, Segment
    sk = object.__new__(Skills)
    h = SimpleNamespace(label='tool', T_tcp_obj=np.eye(4), box=object())
    sk.held = {'left': h}
    sk.ex = SimpleNamespace(q={'left': np.zeros(6)}, g={'left': 0.0})
    sk.world = SimpleNamespace(obj=lambda _: SimpleNamespace(asset=SimpleNamespace(center=np.zeros(3))))
    segment = Segment(np.array([np.zeros(6), np.full(6, 0.1)]), 0.05)
    transit = MoveResult('reached', 'left', [segment], np.eye(4))
    sk.motion = SimpleNamespace(line=Mock(return_value=transit), to_pose=Mock())
    sk._place_tail = Mock(return_value=ArmPlan('left', segment.q[-1], 0.0))
    plan = sk.plan_place('left', np.eye(4), approach_line=True, ease=3, held_read=True)
    assert plan.approach is transit and segment.ease == 3
    assert plan.tail_kw['held_read'] is True
    assert 'approach_line' not in plan.tail_kw
    assert plan.plan.g0 == 0.0
    sk.motion.to_pose.assert_not_called()
