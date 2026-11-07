"""Loaded-hold proof contracts, using only measured-motion mocks."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills import headset_hold
from dojo.skills.geometry import T_from, rot_z
from dojo.skills.motion import MoveResult, Segment
from dojo.skills.skills import SkillResult


def fixture(*, follow=True, clearance=.10, slip=0., turn=0., fail_leg=None, final_ok=True):
    obj = SimpleNamespace(category="headset", pos=np.array([0., 0., .765 + clearance]), R=np.eye(3),
                          asset=SimpleNamespace(mesh=(np.zeros((1, 3)), np.zeros((0, 3), int))))
    arm = SimpleNamespace(q=np.zeros(6))
    world = SimpleNamespace(obj=lambda label: obj, arms={"left": arm})
    calls = []
    base = obj.pos.copy()
    ex = SimpleNamespace(steps_left=500, reserve=30, on_step=lambda ex: calls.append("observer"), used=0)

    def refresh():
        obj.T = T_from(obj.pos, obj.R)
        return world

    def step(q):
        ex.used += 1
        arm.q = np.asarray(q)
        obj.pos = base + [slip * ex.used, 0., q[0] if follow else 0.]
        obj.R = rot_z(turn * ex.used)
        ex.on_step(ex)

    def hold(n, **kw):
        for _ in range(n):
            step(arm.q.copy())

    def run(plans, **kw):
        for q, _ in plans['left'].compile()[0]:
            step(q)

    ex.hold, ex.run = hold, run
    def tcp(a, q):
        return T_from([0, 0, base[2] + q[0]])
    def line(a, q, T, speed, name, **kw):
        calls.append((name, kw))
        end = np.zeros(6)
        end[0] = T[2, 3] - base[2]
        return MoveResult("collision" if name == fail_leg else "reached", a,
                          [] if name == fail_leg else [Segment(np.array([q, end]), speed)])
    held = SimpleNamespace(box=object())
    def verify(pp, before):
        calls.append(("verify", before.copy()))
        if not final_ok:
            sk.held['left'] = None
        return SkillResult(final_ok, 'pick', 'loose' if final_ok else 'grasp_slip', note='raw drift 17 mm')
    sk = SimpleNamespace(world=world, refresh=refresh, kin=SimpleNamespace(tcp=tcp), ex=ex,
                         motion=SimpleNamespace(line=line), held={'left': held}, verify_pick=verify)
    refresh()
    pp = SimpleNamespace(arm='left', label='headset')
    result = SkillResult(True, 'pick', 'loose', 40, 'left', note='raw drift 16 mm')
    return sk, pp, result, calls


def test_loaded_proof_requires_unsupported_following_and_retains_nominal_evidence():
    sk, pp, raw, calls = fixture()
    observer = sk.ex.on_step
    before = np.array([0, 0, .8])
    out = headset_hold.loaded_hold(sk, pp, raw, before)
    assert out.ok and out.status == 'ok' and out is not raw
    assert raw.status == 'loose' and 'raw drift 16 mm' in out.note and 'raw drift 17 mm' in out.note
    assert 'rise 20.0 mm' in out.note and 'relative 0.00 mm/0.00 deg' in out.note
    assert sk.ex.on_step is observer and calls.count('observer') == sk.ex.used
    legs = [c for c in calls if isinstance(c, tuple) and c[0].startswith('headset_load')]
    assert [c[0] for c in legs] == ['headset_load_up', 'headset_load_down']
    assert all(c[1] == dict(gripper=0., held=sk.held['left'].box, ignore=('headset',), check=True) for c in legs)
    assert len([c for c in calls if isinstance(c, tuple) and c[0] == 'verify']) == 2


@pytest.mark.parametrize('kw', [dict(follow=False), dict(slip=.0002), dict(turn=.01),
                                dict(clearance=.002), dict(clearance=-.01),
                                dict(fail_leg='headset_load_up'), dict(fail_leg='headset_load_down')])
def test_stale_pose_sliding_rotating_supported_or_unchecked_hold_never_promotes(kw):
    sk, pp, raw, calls = fixture(**kw)
    observer = sk.ex.on_step
    assert headset_hold.loaded_hold(sk, pp, raw, np.zeros(3)) is raw
    assert sk.ex.on_step is observer
    if 'fail_leg' in kw:
        assert sk.ex.used == 30  # both legs must plan before any challenge motion


def test_original_hard_slip_is_not_overridden():
    sk, pp, raw, _ = fixture(final_ok=False)
    result = headset_hold.loaded_hold(sk, pp, raw, np.zeros(3))
    assert not result.ok and result.status == 'grasp_slip'
    assert sk.held['left'] is None


@pytest.mark.parametrize('condition', ['ok', 'slip', 'other_object', 'no_mesh', 'invalid_mesh', 'no_hold', 'budget'])
def test_unqualified_inputs_do_not_run_a_proof(condition):
    sk, pp, raw, calls = fixture()
    if condition == 'ok':raw.status = 'ok'
    if condition == 'slip':raw.ok, raw.status = False, 'grasp_slip'
    if condition == 'other_object':sk.world.obj('').category = 'mug'
    if condition == 'no_mesh':sk.world.obj('').asset.mesh = None
    if condition == 'invalid_mesh':sk.world.obj('').asset.mesh = ('invalid', [])
    if condition == 'no_hold':sk.held['left'] = None
    if condition == 'budget':sk.ex.steps_left = 129
    assert headset_hold.loaded_hold(sk, pp, raw, np.zeros(3)) is raw
    assert sk.ex.used == 0 and not calls


@pytest.mark.parametrize('kind', ['budget', 'ended'])
def test_observer_is_restored_after_execution_stops(kind):
    from dojo.skills.executor import BudgetExceeded, EpisodeEnded
    sk, pp, raw, _ = fixture()
    observer = sk.ex.on_step
    def stop(*a, **kw):
        raise (BudgetExceeded if kind == 'budget' else EpisodeEnded)('stop')
    sk.ex.hold = stop
    out = headset_hold.loaded_hold(sk, pp, raw, np.zeros(3))
    assert not out.ok and out.status == ('step_budget' if kind == 'budget' else 'ended')
    assert sk.ex.on_step is observer


def test_loaded_hold_defaults_validation_and_other_transfers_unchanged():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.env import KinematicEnv
    from dojo.skills.g3 import SKILL_ATTRS
    from dojo.skills.skills import Skills
    from dojo.skills.world import WorldState
    assert RuntimeConfig().headset_hold == 'legacy'
    assert SKILL_ATTRS['headset_hold'] == ('legacy', 'loaded')
    sk = Skills(KinematicEnv([]), SimpleNamespace(refresh=lambda: WorldState('', '', {}, {}, {})))
    assert sk.headset_hold == 'legacy'
    with pytest.raises(ValueError, match='bad runtime config'):
        RuntimePolicy(RuntimeConfig(headset_hold='invalid'), assets=object())
    assert RuntimePolicy(RuntimeConfig(headset_hold='loaded'), assets=object()).cfg.headset_hold == 'loaded'


def test_a_hard_slip_after_the_challenge_still_rejects():
    sk, pp, raw, _ = fixture()
    original = sk.verify_pick
    calls = []
    def verify(p, before):
        calls.append(True)
        if len(calls) == 2:
            sk.held['left'] = None
            return SkillResult(False, 'pick', 'grasp_slip', note='31 mm nominal drift')
        return original(p, before)
    sk.verify_pick = verify
    out = headset_hold.loaded_hold(sk, pp, raw, np.zeros(3))
    assert not out.ok and out.status == 'grasp_slip' and len(calls) == 2
    assert sk.held['left'] is None


@pytest.mark.parametrize('mode,category,prove', [('legacy', 'headset', False), ('loaded', 'mug', False),
                                               ('loaded', 'headset', True)])
def test_transfer_proof_is_scoped_and_uses_the_actual_pre_pick_root(monkeypatch, mode, category, prove):
    from dojo.skills.skills import Skills, PickPlan, PlacePlan
    obj = SimpleNamespace(category=category, pos=np.array([.1, .2, .8]))
    world = SimpleNamespace(obj=lambda l: obj)
    pp = PickPlan('left', object(), object(), np.eye(4), 40, 'item')
    pl = PlacePlan('left', object(), np.eye(4), 50, 'item')
    calls = []
    sk = SimpleNamespace(headset_hold=mode, headset_retry='sites', world=world, last_pick=pp,
                         refine=None, held={'left':object()}, _record=lambda r:r, refresh=lambda:world)
    def pick(p):
        obj.pos = obj.pos + [0, 0, .08]
        return SkillResult(True, 'pick', 'loose', note='16 mm nominal')
    def proof(s, p, raw, before):
        assert s is sk and p is pp
        np.testing.assert_allclose(before, [.1, .2, .8])
        calls.append('proof')
        return SkillResult(True, 'pick', 'ok', note='loaded proof')
    def place(*a, **kw):
        calls.append('place')
        return SkillResult(False, 'place', 'ik_fail', note='native refusal')
    def regrasp(a):
        calls.append('put_down')
        sk.held[a] = None
        return SkillResult(True, 'regrasp')
    sk.run_pick, sk.plan_place, sk.regrasp = pick, place, regrasp
    monkeypatch.setattr(headset_hold, 'loaded_hold', proof)
    out = Skills.run_transfer(sk, pp, pl)
    assert calls == (['proof', 'place', 'put_down'] if prove else ['put_down'])
    assert not out.ok and out.status == ('grasp_unusable' if prove else 'loose')


@pytest.mark.parametrize('mode', ['legacy', 'loaded'])
def test_tolerant_relay_has_identical_legacy_execution_without_a_load_proof(monkeypatch, mode):
    from dojo.skills.skills import Skills, PickPlan, PlacePlan
    obj = SimpleNamespace(category='headset', pos=np.array([0., 0., .8]),
                          asset=SimpleNamespace(extents=np.full(3, .1)))
    pp = PickPlan('left', object(), object(), np.eye(4), 40, 'headset')
    pl = PlacePlan('left', object(), np.eye(4), 40, 'headset')
    events = []
    sk = SimpleNamespace(headset_hold=mode, headset_retry='sites', refine=None, last_pick=pp,
                         world=SimpleNamespace(obj=lambda label:obj, objects={}),
                         held={'left':object(), 'right':None}, _record=lambda r:r,
                         run_pick=lambda p: events.append('pick') or SkillResult(True, 'pick', 'loose'),
                         refresh=lambda:None, plan_place=lambda *a,**kw:events.append('plan_place') or pl,
                         _park_if_blocking=lambda *a,**kw:events.append('park'),
                         run_place=lambda *a,**kw:events.append('place') or SkillResult(True, 'place'))
    monkeypatch.setattr(headset_hold, 'loaded_hold', lambda *a:pytest.fail('relay must keep its old path'))
    out = Skills.run_transfer(sk, pp, pl, tol_xy=.03)
    assert out.ok and events == ['pick', 'plan_place', 'park', 'place']


def test_strict_transfer_puts_down_after_refused_proof(monkeypatch):
    from dojo.skills.skills import Skills, PickPlan, PlacePlan
    pp = PickPlan('left', object(), object(), np.eye(4), 40, 'headset')
    raw = SkillResult(True, 'pick', 'loose')
    events = []
    sk = SimpleNamespace(headset_hold='loaded', world=SimpleNamespace(obj=lambda l:SimpleNamespace(
                         category='headset', pos=np.zeros(3))), last_pick=pp, refine=None,
                         _record=lambda r:r, run_pick=lambda p:raw,
                         regrasp=lambda a:events.append('put_down') or SkillResult(True, 'regrasp'),
                         plan_place=lambda *a,**kw:pytest.fail('refused proof cannot carry'))
    monkeypatch.setattr(headset_hold, 'loaded_hold', lambda *a:events.append('proof_refused') or raw)
    out = Skills.run_transfer(sk, pp, PlacePlan('left', object(), np.eye(4), 40, 'headset'), tol_xy=.015)
    assert not out.ok and out.status == 'loose' and events == ['proof_refused', 'put_down']
