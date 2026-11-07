"""Headset experiment plans a clear turn and fixed-orientation carry, CPU only."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills.geometry import T_from, rot_z
from dojo.skills.headset import staged_carry
from dojo.skills.motion import MoveResult, Segment


def test_staged_waypoint_preserves_object_xy_and_uses_measured_attachment():
    start = T_from([-.2, -.1, .9], rot_z(.3))
    rel = T_from([.04, -.02, -.03], rot_z(.4))
    goal = T_from([-.3, .05, 1.0], rot_z(-.6))
    calls = []
    def turn(arm, q, T, speed, name, **kw):
        calls.append(('turn', T, kw))
        return MoveResult('reached', arm, [Segment(np.array([q, np.ones(6)]), speed)])
    def line(arm, q, T, speed, name, **kw):
        assert np.array_equal(q, np.ones(6))
        calls.append(('line', T, kw))
        return MoveResult('reached', arm, [Segment(np.array([q, np.ones(6)*2]), speed)])
    motion = SimpleNamespace(kin=SimpleNamespace(tcp=lambda a,q: start), to_pose=turn, line=line)
    held = SimpleNamespace(T_tcp_obj=rel)
    result = staged_carry(motion, 'left', np.zeros(6), goal, .03, 'carry', gripper=0, held=held, ignore=('headset',))
    assert result.ok and len(result.segments) == 2
    mid_obj, initial_obj, goal_obj = calls[0][1] @ rel, start @ rel, goal @ rel
    assert np.allclose(mid_obj[:2, 3], initial_obj[:2, 3])
    assert mid_obj[2, 3] == max(initial_obj[2, 3], goal_obj[2, 3])
    assert np.allclose(mid_obj[:3,:3], goal_obj[:3,:3])
    assert np.allclose(calls[0][1][:3,:3], goal[:3,:3])
    assert np.array_equal(calls[1][1], goal)
    assert calls[1][2]['check'] is True
    assert all(c[2]['held'] is held and c[2]['ignore'] == ('headset',) for c in calls)


@pytest.mark.parametrize('failed_leg', ['turn', 'line'])
def test_incomplete_staged_carry_is_refused(failed_leg):
    calls = []
    def leg(which):
        def plan(arm,q,*a,**kw):
            calls.append(which)
            return MoveResult('collision' if which == failed_leg else 'reached', arm,
                              [] if which == failed_leg else [Segment(np.array([q,np.ones(6)]), .03)])
        return plan
    motion = SimpleNamespace(kin=SimpleNamespace(tcp=lambda *a: np.eye(4)),
                             to_pose=leg('turn'), line=leg('line'))
    result = staged_carry(motion, 'left', np.zeros(6), np.eye(4), .03, 'carry',
                          gripper=0, held=SimpleNamespace(T_tcp_obj=np.eye(4)))
    assert not result.ok
    assert calls == (['turn'] if failed_leg == 'turn' else ['turn', 'line'])


def test_headset_carry_default_and_config_validation():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    from dojo.skills.skills import Skills
    from dojo.skills.env import KinematicEnv
    from dojo.skills.world import WorldState
    assert RuntimeConfig().headset_carry == 'legacy'
    assert SKILL_ATTRS['headset_carry'] == ('legacy', 'staged')
    assert Skills(KinematicEnv([]),SimpleNamespace(refresh=lambda:WorldState('','',{}, {}, {}))).headset_carry == 'legacy'
    assert RuntimePolicy(RuntimeConfig(headset_carry='staged'),assets=object()).cfg.headset_carry == 'staged'
    with pytest.raises(ValueError,match='bad runtime config'):
        RuntimePolicy(RuntimeConfig(headset_carry='nope'),assets=object())


def test_dense_grasps_only_changes_proposals_and_filters_width(monkeypatch):
    from dojo.skills import grasps as G
    from dojo.skills.headset import dense_grasps
    seen = []
    thin = SimpleNamespace(width=.026)
    wide = SimpleNamespace(width=.041)
    monkeypatch.setattr(G,'box_grasps',lambda *a:[thin])
    monkeypatch.setattr(G,'slice_grasps',lambda *a:[wide])
    def antipodal(*a,**kw):
        seen.append(kw)
        return [thin,wide]
    monkeypatch.setattr(G,'antipodal_grasps',antipodal)
    assert dense_grasps(SimpleNamespace(kin=object()),object(),'left') == [wide,wide]
    assert seen == [dict(n_samples=400,n_approach=13)]


def test_dense_grasp_switch_default_and_validation():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    from dojo.skills.skills import Skills
    from dojo.skills.env import KinematicEnv
    from dojo.skills.world import WorldState
    assert RuntimeConfig().headset_grasp == 'legacy'
    assert SKILL_ATTRS['headset_grasp'] == ('legacy','dense')
    sk=Skills(KinematicEnv([]),SimpleNamespace(refresh=lambda:WorldState('','',{}, {}, {})))
    assert sk.headset_grasp == 'legacy'
    with pytest.raises(ValueError,match='bad runtime config'):
        RuntimePolicy(RuntimeConfig(headset_grasp='other'),assets=object())
