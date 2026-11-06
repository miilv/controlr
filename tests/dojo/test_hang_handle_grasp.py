from types import SimpleNamespace

import numpy as np

from dojo.skills.hanging import handle_grasps
from dojo.skills.world import AssetInfo, ObjectState


def _bar():
    lo, hi = np.array([.058, -.004, -.018]), np.array([.062, .004, .018])
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    # Frame x across the bar, z out of the loop. Its quaternion is the cyclic xyz permutation.
    asset = AssetInfo(key='Rigid/mug/799', bbox_min=lo, bbox_max=hi, mesh=(V, F),
                      active_functional={'handle': {'frame': [[.062, 0., 0., .5, .5, .5, .5]]}})
    return ObjectState('mug', 'Rigid', 'mug', 799, np.array([0., 0., .9]), np.array([1., 0, 0, 0]), asset)


def test_handle_pinch_has_opposed_faces_and_downward_approach():
    o = _bar()
    gs = handle_grasps(o)
    assert gs
    for g in gs:
        assert g.method == 'handle'
        np.testing.assert_allclose(g.width, .008, atol=1e-12)
        np.testing.assert_allclose(g.T[:3, 0], [0., 0., -1.])
        assert abs(g.T[1, 1]) == 1.
    o.asset.mesh = None
    assert handle_grasps(o) == []


def test_default_grasp_set_is_unchanged(monkeypatch):
    from dojo.skills import library as L, grasps as G
    monkeypatch.setattr(G, 'box_grasps', lambda *a: ['box'])
    monkeypatch.setattr(G, 'wall_grasps', lambda *a, **kw: ['wall'])
    o = _bar()
    assert L.G_hang(SimpleNamespace(kin=None), o, 'left') == ['box', 'wall']
    enabled = L.G_hang(SimpleNamespace(kin=None, hang_handle_grasp=True), o, 'left')
    assert enabled[-2:] == ['box', 'wall']
    assert all(g.method == 'handle' for g in enabled[:-2])


def test_default_off():
    from dojo.runtime.policy import RuntimeConfig
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().hang_handle_grasp is False
    assert SKILL_ATTRS['hang_handle_grasp'] == ('off', 'on')
