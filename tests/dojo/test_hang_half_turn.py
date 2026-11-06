"""An alternate handle orientation must preserve the physical threading line."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills import library as L
from dojo.skills.geometry import T_from, rot_x, rot_z
from dojo.skills.skills import SkillResult
from dojo.skills.world import AssetInfo, ObjectState, WorldState


@pytest.mark.parametrize('slow,rate', [(False, L.HANG_CARRY), ('on', L.HANG_CARRY / 2)])
@pytest.mark.parametrize('flag,count', [(False, 1), ('off', 1), (True, 2), ('on', 2)])
def test_half_turn_fallback_preserves_hole_line(monkeypatch, flag, count, slow, rate):
    def obj(label, category):
        return ObjectState(label, 'Rigid', category, 0, np.array([0., 0., .8]), np.array([1., 0, 0, 0]),
                           AssetInfo(key=category, bbox_min=np.full(3, -.03), bbox_max=np.full(3, .03)))
    mug, rack = obj('mug', 'mug'), obj('rack', 'cup_holder')
    w = WorldState('', '', {'mug': mug, 'rack': rack}, {}, {})
    H = T_from((.1, .2, 1.1), rot_z(.3))
    handle = T_from((.02, .01, .03), rot_x(.7))
    monkeypatch.setattr(L, 'hook_frames', lambda *a: [(0, H, .03)])
    monkeypatch.setattr(L, 'functional_frames', lambda o, *a, **kw: [handle] if o is mug else [H])
    monkeypatch.setattr(L, 'on_hook', lambda *a: None)
    monkeypatch.setattr(L, 'free_spots', lambda *a, **kw: [])
    targets = []

    def transfer(lab, poses, **kw):
        assert kw["place_kw"]["approach_rad"] == rate
        assert kw["place_kw"]["release"] is False
        targets.append(poses[0])
        return SkillResult(False, 'transfer', 'ik_fail', 0, None, note='0 grasp-place pairs')

    sk = SimpleNamespace(world=w, refresh=lambda: w, log=lambda m: None, hang_half_turn=flag, hang_carry_slow=slow,
                         coll=SimpleNamespace(include=set(), parts={}, held_skip={}), plan_transfer=transfer)
    L.hang_on_hooks(sk, ['mug'], 'rack', 'handle_support_point')
    assert len(targets) == count  # A failed original orientation must not suppress the alternate.
    for target in targets:
        actual_handle = target @ handle
        hole = actual_handle[:3, 3] - actual_handle[:3, 2] * .008
        np.testing.assert_allclose(hole, H[:3, 3] + H[:3, 0] * (.03 + L.HANG_PRE))
        np.testing.assert_allclose(actual_handle[:3, 2], H[:3, 2], atol=1e-12)
        assert abs(actual_handle[:3, 0] @ H[:3, 0]) == pytest.approx(1)
    if count == 2:
        assert (targets[0] @ handle)[:3, 0] @ (targets[1] @ handle)[:3, 0] == pytest.approx(-1)


def test_half_turn_defaults_off():
    from dojo.runtime.policy import RuntimeConfig
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().hang_half_turn is False
    assert SKILL_ATTRS['hang_half_turn'] == ('off', 'on')


def test_slow_carry_defaults_off():
    from dojo.runtime.policy import RuntimeConfig
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().hang_carry_slow is False
    assert SKILL_ATTRS['hang_carry_slow'] == ('off', 'on')
