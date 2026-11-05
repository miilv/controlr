"""Wide belt riders must be offered a waist before a shallow scoop pinch, with fallback.

The mesh is a cone with a wide cap, independent of task name or asset identity.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills import interactive as I
from dojo.skills.kinematics import X5Kin
from dojo.skills.world import AssetInfo, ObjectState


def _rider():
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    rings = [(0.0, 0.012), (0.09, 0.034), (0.10, 0.052), (0.14, 0.052)]
    vertices = np.array([[r * np.cos(t), r * np.sin(t), z] for z, r in rings for t in angles])
    faces = []
    for k in range(3):
        for j in range(32):
            a, b = k * 32 + j, k * 32 + (j + 1) % 32
            faces.extend(((a, b, b + 32), (a, b + 32, a + 32)))
    asset = AssetInfo('Rigid/test/00000', np.array([-.052, -.052, 0]), np.array([.052, .052, .14]),
                      mesh=(vertices, np.array(faces)))
    return ObjectState('rider', 'Rigid', 'test', 0, np.array([-.4, -.05, .77]), np.array([1., 0, 0, 0]), asset)


def test_cone_waist_has_mesh_contacts_below_wide_cap():
    from dojo.skills import grasps as G
    o = _rider()
    sk = SimpleNamespace(kin=X5Kin())
    assert not G.candidates(o, sk.kin, 'left', 'top')
    grasps = I.waist_grasps(sk, o, 'left')
    assert grasps
    assert all(0.002 < g.width <= G.MAX_WIDTH for g in grasps)
    assert all(g.T[2, 3] < o.pos[2] + .10 for g in grasps)


@pytest.mark.parametrize('enabled,reachable,expected', [
    ('off', True, 'surface'), ('on', True, 'waist'), ('on', False, 'surface')])
def test_wide_interception_prefers_waist_but_keeps_surface_fallback(monkeypatch, enabled, reachable, expected):
    from dojo.skills import grasps as G
    o = _rider()
    before = o.pos.copy()
    sk = SimpleNamespace(kin=X5Kin(), world=SimpleNamespace(obj=lambda _: o), belt_waist_first=enabled)
    # Surface grasps represent the shallow slice/rim alternatives. Keep actual mesh waist generation.
    T = np.eye(4)
    T[:3, :3] = G._frame(np.array([0., 0, -1]), np.array([1., 0, 0]))
    T[:3, 3] = o.center_w
    surface = G.Grasp('surface', T, .04)
    monkeypatch.setattr(G, 'slice_grasps', lambda *a: [surface])
    monkeypatch.setattr(G, 'antipodal_grasps', lambda *a: [])
    monkeypatch.setattr(G, 'rim_grasps', lambda *a: [])

    def search(sk, label, obj, velocity, rel, arms, *args):
        gs = [g for a in arms for g, _, _ in rel[a]]
        if not gs or (gs[0].method == 'waist' and not reachable):
            return None
        return gs[0]

    monkeypatch.setattr(I, '_intercept_search', search)
    assert I.plan_intercept(sk, 'rider', np.array([.004, 0]), arms=['left']).method == expected
    np.testing.assert_array_equal(o.pos, before)


def test_waist_switch_default_and_validation():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().belt_waist_first == 'off'
    assert SKILL_ATTRS['belt_waist_first'] == ('off', 'on')
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(belt_waist_first='typo'))


def test_narrow_top_never_uses_new_search(monkeypatch):
    o = _rider()
    o.asset.bbox_min[:2] /= 2
    o.asset.bbox_max[:2] /= 2
    o.asset.mesh = (o.asset.mesh[0] * np.array([.5, .5, 1]), o.asset.mesh[1])
    sk = SimpleNamespace(kin=X5Kin(), world=SimpleNamespace(obj=lambda _: o), belt_waist_first='on')
    monkeypatch.setattr(I, 'waist_grasps', lambda *a: pytest.fail('narrow top used waist search'))
    monkeypatch.setattr(I, '_intercept_search', lambda *a: 'reachable top')
    assert I.plan_intercept(sk, 'rider', np.array([.004, 0]), arms=['left']) == 'reachable top'


def test_retry_does_not_reintroduce_skipped_waist(monkeypatch):
    from dojo.skills import grasps as G
    o = _rider()
    sk = SimpleNamespace(kin=X5Kin(), world=SimpleNamespace(obj=lambda _: o), belt_waist_first='on')
    for name in ('slice_grasps', 'antipodal_grasps', 'rim_grasps'):
        monkeypatch.setattr(G, name, lambda *a: [])
    offered = []

    def unreachable(sk, label, obj, velocity, rel, arms, *args):
        offered.extend(g.method for a in arms for g, _, _ in rel[a])
        return None

    monkeypatch.setattr(I, '_intercept_search', unreachable)
    assert I.plan_intercept(sk, 'rider', np.array([.004, 0]), arms=['left'], skip_methods=('waist',)) is None
    assert 'waist' not in offered
