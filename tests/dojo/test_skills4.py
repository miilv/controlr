"""skills4 pieces without Isaac: physics link poses for the checker, mesh openings, ring axis."""

import sys
import types

import numpy as np

from dojo.simlab import runner as R


class _PV:
    def __init__(self, T):
        self.T = T

    def get_link_transforms(self):
        return self.T


class _AV:
    def __init__(self, names, T):
        self.body_names = names
        self._physics_view = _PV(T)


def test_physics_link_pose_converts_xyzw_to_wxyz():
    T = np.zeros((1, 2, 7))
    T[0, 1] = [0.1, -0.2, 0.9, 0.0, 0.0, 0.7071, 0.7071]        # x y z, qx qy qz qw
    art = types.SimpleNamespace(_articulation_view=_AV(["base", "link_1"], T))
    p = R.physics_link_pose(art, "link_1")
    assert np.allclose(p, [0.1, -0.2, 0.9, 0.7071, 0.0, 0.0, 0.7071], atol=1e-4)
    assert R.physics_link_pose(art, "nope") is None
    assert R.physics_link_pose(types.SimpleNamespace(), "base") is None


def test_install_physics_link_poses_keeps_the_usd_fallback(monkeypatch):
    class ArticulationObject:
        def __init__(self, av):
            self._articulation_view = av

        def get_link_pose(self, link_name):
            return np.array([9.0, 9, 9, 1, 0, 0, 0])

    mod = types.ModuleType("env.scene_manager.objects.articulation")
    mod.ArticulationObject = ArticulationObject
    for name in ("env", "env.scene_manager", "env.scene_manager.objects"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "env.scene_manager.objects.articulation", mod)
    sys.modules["env.scene_manager.objects"].articulation = mod
    R.install_physics_link_poses()
    T = np.zeros((1, 1, 7))
    T[0, 0] = [1.0, 2.0, 3.0, 0, 0, 0, 1.0]
    moved = ArticulationObject(_AV(["E_body_1"], T))
    assert np.allclose(moved.get_link_pose("E_body_1")[:3], [1, 2, 3])          # physics, not the spawn pose
    assert np.allclose(moved.get_link_pose("other")[:3], [9, 9, 9])             # unknown link: USD as before
    R.install_physics_link_poses()                                              # idempotent
    assert getattr(ArticulationObject.get_link_pose, "_simlab_physics", False)


def _box(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    V = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def test_usd_sync_links_defaults_on_in_physics_only(monkeypatch):
    # the official rendered kit writes physics to USD; the physics-only twin must read link poses from physics
    monkeypatch.delenv("SIMLAB_USD_SYNC_LINKS", raising=False)
    assert R.resolve_usd_sync_links(None, physics_only=True) is True
    assert R.resolve_usd_sync_links(None, physics_only=False) is False
    monkeypatch.setenv("SIMLAB_USD_SYNC_LINKS", "0")
    assert R.resolve_usd_sync_links(None, physics_only=True) is False
    monkeypatch.setenv("SIMLAB_USD_SYNC_LINKS", "1")
    assert R.resolve_usd_sync_links(None, physics_only=False) is True
    assert R.resolve_usd_sync_links(False, physics_only=True) is False


def test_openings_finds_two_slots_and_their_floor():
    from dojo.skills.insertion import openings
    parts = [_box((-0.1, -0.06, 0.0), (0.1, 0.06, 0.03)),                          # floor block
             _box((-0.1, -0.06, 0.03), (0.1, -0.035, 0.15)),                       # walls between / beside slots
             _box((-0.1, -0.01, 0.03), (0.1, 0.01, 0.15)),
             _box((-0.1, 0.035, 0.03), (0.1, 0.06, 0.15)),
             _box((-0.1, -0.06, 0.03), (-0.08, 0.06, 0.15)),
             _box((0.08, -0.06, 0.03), (0.1, 0.06, 0.15))]
    V = np.vstack([p[0] for p in parts])
    F = np.vstack([p[1] + 8 * i for i, p in enumerate(parts)])
    ops = sorted(openings(V, F, res=0.002), key=lambda o: o["center"][1])
    assert len(ops) == 2
    for op, y in zip(ops, (-0.0225, 0.0225)):
        assert abs(op["center"][1] - y) < 0.003 and abs(op["floor"] - 0.03) < 0.002
        assert 0.02 < op["width"] < 0.03 and op["length"] > 0.14
        assert abs(abs(op["along"][0]) - 1.0) < 1e-3


def test_ring_center_ignores_a_handle():
    from dojo.skills.grasps import ring_center
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ring = np.c_[0.01 + 0.04 * np.cos(th), 0.04 * np.sin(th), np.full_like(th, 0.076)]   # the wall 4 mm below the rim
    handle = np.array([[0.07, 0.0, 0.076], [0.072, 0.003, 0.075], [0.072, -0.003, 0.075]])
    V = np.vstack([ring, handle, [[0.01, 0.04, 0.08], [0.0, 0.0, 0.0]]])
    asset = types.SimpleNamespace(mesh=(V, np.zeros((0, 3), int)), center=np.array([0.02, 0.0, 0.04]))
    c = ring_center(types.SimpleNamespace(asset=asset))
    assert np.allclose(c, [0.01, 0.0], atol=1e-3)
