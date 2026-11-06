"""Bounded carry reads: sensor gates and checked motion, without camera/oracle substitution."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from dojo.skills.motion import MoveResult, Segment
from dojo.skills.skills import Held
from dojo.skills.slot_carry import measured_approach


def rig(reads, fail=False):
    q = np.zeros(6)
    ex = SimpleNamespace(q={'left': q}, steps_used=0)
    def tcp(arm, values):
        T = np.eye(4)
        T[:3, 3] = values[:3]
        return T
    asset = SimpleNamespace(key='Rigid/tool/00000', center=np.zeros(3), extents=np.full(3, 0.02))
    obj = SimpleNamespace(asset=asset, T=np.eye(4))
    world = SimpleNamespace(obj=lambda _: obj, arms={'left': SimpleNamespace(q=q)})
    def move(arm, q0, target, *a, **kw):
        if fail:
            return MoveResult('collision', arm, note='blocked')
        q1 = np.r_[target[:3, 3], np.zeros(3)]
        return MoveResult('reached', arm, [Segment(np.array([q0, q1]), 0.05)], target)
    def run(plans, **kw):
        ex.q['left'] = plans['left'].q_end
    ex.run = Mock(side_effect=run)
    sk = SimpleNamespace(held={'left': Held('tool', np.eye(4), None)}, remeasure=False,
                         provider=SimpleNamespace(held_pose=object()), refresh=lambda: world,
                         kin=SimpleNamespace(tcp=tcp), ex=ex,
                         motion=SimpleNamespace(line=Mock(side_effect=move)),
                         _held_protocol=Mock(side_effect=reads))
    pp = SimpleNamespace(arm='left', label='tool', T_obj_target=np.eye(4), tail_kw={'servo': True},
                         approach=SimpleNamespace(segments=[Segment(np.zeros((2, 6)), 0.05, ease=3)]))
    commands = [(np.array([i * 0.01, 0, 0, 0, 0, 0]), 0.0) for i in range(1, 5)]
    return sk, pp, commands


def test_missing_provider_keeps_legacy_transit():
    sk, pp, commands = rig([])
    sk.provider = object()
    assert measured_approach(sk, pp, commands, 0.02, 10) == (False, None)
    sk.ex.run.assert_not_called()


def test_rejected_first_read_does_not_move_or_change_attachment():
    sk, pp, commands = rig([None])
    old = sk.held['left']
    assert measured_approach(sk, pp, commands, 0.02, 10) == (False, None)
    assert sk.held['left'] is old
    sk.ex.run.assert_not_called()


def test_four_hops_use_accepted_transform_and_retain_it_for_missing_later_reads():
    T = np.eye(4)
    T[0, 3] = 0.005
    sk, pp, commands = rig([T, None, None, None])
    assert measured_approach(sk, pp, commands, 0.02, 10) == (True, None)
    assert sk.ex.run.call_count == sk._held_protocol.call_count == 4
    np.testing.assert_allclose(sk.ex.q['left'][:3], [0.035, 0, 0])
    for call in sk.motion.line.call_args_list:
        assert call.kwargs['ignore'] == ('tool',)
        assert call.kwargs['gripper'] == 0.0
        assert 'check' not in call.kwargs
        assert call.kwargs['held'] is not None
    request = sk._held_protocol.call_args.args[0]
    assert request.tail_kw['servo'] is False and request.tail_kw['held_read'] is True
    assert pp.tail_kw == {'servo': True}


def test_infeasible_hop_does_not_execute_partial_move_or_release():
    sk, pp, commands = rig([np.eye(4)], fail=True)
    handled, failure = measured_approach(sk, pp, commands, 0.02, 10)
    assert handled and 'blocked' in failure
    sk.ex.run.assert_not_called()
    assert sk.held['left'] is not None


def test_camera_path_uses_shared_uncertainty_gate_without_reading_true_object_pose():
    from dojo.skills.skills import Skills
    sk, pp, commands = rig([])
    asset = sk.refresh().obj('tool').asset
    class CameraObject:
        @property
        def T(self):
            raise AssertionError('camera controller read true object pose')
    obj = CameraObject()
    obj.asset = asset
    world = SimpleNamespace(obj=lambda _: obj, arms={'left': SimpleNamespace(q=np.zeros(6))})
    sk.refresh = lambda: world
    sk.provider.held_pose = Mock(return_value={'ok': True, 'T_tcp_obj': np.eye(4),
                                             'sigma': {'lead_lat_mm': 99, 'axis_deg': 1}})
    sk.held_events = []
    sk.log = lambda _: None
    sk.held_looks = False
    sk._held_read = lambda *a: Skills._held_read(sk, *a)
    sk.held_sigma = Skills.held_sigma
    sk._held_protocol = lambda *a: Skills._held_protocol(sk, *a)
    assert measured_approach(sk, pp, commands, 0.02, 10) == (False, None)
    sk.ex.run.assert_not_called()
    assert len(sk.held_events) == 1 and not sk.held_events[0]['adopted']
    sk.provider.held_pose.return_value['sigma']['lead_lat_mm'] = 1
    assert measured_approach(sk, pp, commands, 0.02, 10) == (True, None)
    assert sum(e['adopted'] for e in sk.held_events) == 4


def test_carry_orientation_starts_at_first_accepted_pose_not_stale_attachment():
    from dojo.skills.geometry import rot_z
    measured = np.eye(4)
    measured[:3, :3] = rot_z(0.4)
    sk, pp, commands = rig([measured, None, None, None])
    assert measured_approach(sk, pp, commands, 0.02, 10) == (True, None)
    first_tcp = sk.motion.line.call_args_list[0].args[2]
    np.testing.assert_allclose(first_tcp[:3, :3] @ measured[:3, :3], rot_z(0.3), atol=1e-8)
