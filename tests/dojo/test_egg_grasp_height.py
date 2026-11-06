"""Upper-shell pinches retain the wall guard and the original off path."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills import holders as H
from dojo.skills.grasps import Grasp
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy


def test_raised_pinches_rescue_a_corridor_without_relaxing_wall_clearance(monkeypatch):
    T = np.eye(4)
    from dojo.skills.kinematics import down_R
    T[:3, :3] = down_R(0.0)
    g = Grasp("round", T.copy(), 0.042, note="level")
    monkeypatch.setattr(H, "round_grasps", lambda *a, **kw: [g])
    monkeypatch.setattr(H, "container_of", lambda *a: SimpleNamespace(R=np.eye(3), pos=np.zeros(3)))
    monkeypatch.setattr(H, "wall_profile", lambda c: {"center": [0, 0], "rim": 0.01})
    monkeypatch.setattr(H, "finger_samples", lambda kin, T, go: T[:3, 3][None])
    # At the bottom, 6 mm clears the wall. Approaching from above also clears.
    monkeypatch.setattr(H, "wall_clearance_v", lambda c, prof, P: float(P[0, 2] - 0.015))
    kin = SimpleNamespace(flange_from_tcp=lambda T: T)
    sk = SimpleNamespace(kin=kin, world=SimpleNamespace(obj=lambda label: None), egg_grasp_height="off")
    assert H.egg_grasps(sk, "egg", "left") == []
    sk.egg_grasp_height = "raised"
    gs = H.egg_grasps(sk, "egg", "left")
    assert gs
    assert {round(float(x.T[2, 3]), 3) for x in gs} == {0.006, 0.010}
    assert all("raised" in x.note for x in gs)
    assert np.array_equal(g.T, T)  # shared generator poses remain immutable


def test_off_preserves_candidates_and_raised_is_explicit(monkeypatch):
    g = Grasp("round", np.eye(4), 0.042, note="level")
    monkeypatch.setattr(H, "round_grasps", lambda *a, **kw: [g])
    monkeypatch.setattr(H, "container_of", lambda *a: None)
    sk = SimpleNamespace(kin=None, world=SimpleNamespace(obj=lambda label: None))
    assert H.egg_grasps(sk, "egg", "left") == [g, g]
    assert RuntimeConfig().egg_grasp_height == "off"
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["egg_grasp_height"] == ("off", "raised")
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(egg_grasp_height="typo"), assets=None)


def test_wide_angles_are_opt_in_and_retain_the_round_grasp_geometry(monkeypatch):
    calls = []
    def rounds(*a, **kw):
        calls.append(kw)
        return [Grasp("round", np.eye(4), 0.042, note="test")]
    monkeypatch.setattr(H, "round_grasps", rounds)
    monkeypatch.setattr(H, "container_of", lambda *a: None)
    sk = SimpleNamespace(kin=None, world=SimpleNamespace(obj=lambda label: None))
    H.egg_grasps(sk, "egg", "left")
    assert len(calls) == 2
    sk.egg_grasp_angles = "wide"
    calls.clear()
    H.egg_grasps(sk, "egg", "left")
    assert calls[-1]["tilts"] == (0.0, 40.0, -40.0, 55.0, -55.0)
    assert calls[-1]["ends"] is False
    assert RuntimeConfig().egg_grasp_angles == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(egg_grasp_angles="typo"), assets=None)


def test_side_entry_checks_its_whole_corridor_and_keeps_the_wall_threshold(monkeypatch):
    from dojo.skills.kinematics import down_R
    T = np.eye(4)
    T[:3, :3] = H._rot([0, 1, 0], np.deg2rad(70)) @ down_R(0.0)
    g = Grasp("round", T, 0.042, note="side")
    assert -0.5 < g.approach[2] < -0.2
    monkeypatch.setattr(H, "round_grasps", lambda *a, **kw: [g])
    monkeypatch.setattr(H, "container_of", lambda *a: SimpleNamespace(R=np.eye(3), pos=np.zeros(3)))
    monkeypatch.setattr(H, "wall_profile", lambda c: {"center": [0, 0], "rim": 0.1})
    monkeypatch.setattr(H, "finger_samples", lambda kin, T, go: T[:3, 3][None])
    monkeypatch.setattr(H, "wall_clearance_v", lambda c, prof, P: 0.0)
    sk = SimpleNamespace(kin=SimpleNamespace(flange_from_tcp=lambda T: T),
                         world=SimpleNamespace(obj=lambda label: None))
    assert not H.egg_grasps(sk, "egg", "left")
    sk.egg_grasp_angles = "wide"
    gs = H.egg_grasps(sk, "egg", "left")
    assert gs and all(x.pre_dist == g.pre_dist for x in gs)
    monkeypatch.setattr(H, "wall_clearance_v", lambda c, prof, P: -0.02 if np.linalg.norm(P[0]) > 0.02 else 0.0)
    assert not H.egg_grasps(sk, "egg", "left")


@pytest.mark.parametrize("mode,expected", [("off", 0.05), ("low", 0.02)])
def test_low_drop_changes_only_the_approach_height(monkeypatch, mode, expected):
    # Stop at the first place planner call: no execution or simulator required.
    class Planned(Exception):
        pass
    obj = SimpleNamespace(pos=np.zeros(3), R=np.eye(3))
    world = SimpleNamespace(obj=lambda label: obj)
    monkeypatch.setattr(H, "in_hole", lambda *a: False)
    monkeypatch.setattr(H, "_hole_free", lambda *a: True)
    monkeypatch.setattr(H, "container_of", lambda *a: None)
    monkeypatch.setattr(H, "egg_grasps", lambda *a: [SimpleNamespace(note="level")])
    monkeypatch.setattr(H, "egg_drop_candidates", lambda *a: [(0.012, 0, 1, np.eye(4))])
    def place(arm, target, **kw):
        assert arm == "left"
        assert kw == dict(yaw_sym=1, release_gap=0.0, pre_h=expected, ignore=("holder#body",),
                          descend_rad=0.03, settle=2, retreat=0.04)
        raise Planned()
    sk = SimpleNamespace(world=world, refresh=lambda: world, egg_drop_approach=mode,
                         kin=SimpleNamespace(base=lambda a: np.eye(4)), log=lambda msg: None,
                         plan_pick=lambda *a, **kw: SimpleNamespace(arm="left", grasp=SimpleNamespace(
                             method="round", note="level", pre_dist=0.07)),
                         run_pick=lambda pp: SimpleNamespace(ok=True, status="ok"), _record=lambda r: r,
                         plan_place=place)
    with pytest.raises(Planned):
        H._fill_drop(sk, ["egg"], "holder", obj, [{"top": 0.8}], ("holder#body",), {"done": []})
    assert RuntimeConfig().egg_drop_approach == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(egg_drop_approach="typo"), assets=None)


def test_shell_envelope_frees_air_under_flare_and_still_blocks_material():
    prof = dict(center=np.zeros(2), r_out=0.1, bin=0.01,
                surf=np.full(11, 0.06), lower=np.full(11, 0.04))
    c = SimpleNamespace(pos=np.array([0.1, -0.2, 0.8]), R=np.eye(3))
    points = np.array([[0.15, -0.2, 0.82], [0.15, -0.2, 0.85], [0.15, -0.2, 0.88]])
    assert H.wall_clearance_v(c, prof, points[:1]) < -0.03
    assert H.wall_shell_clearance(c, prof, points[:1]) == pytest.approx(0.02)
    assert H.wall_shell_clearance(c, prof, points[1:2]) == pytest.approx(-0.01)
    assert H.wall_shell_clearance(c, prof, points[2:]) == pytest.approx(0.02)
    assert H.wall_shell_clearance(c, prof, [[1, 1, 1]]) == float("inf")
    assert RuntimeConfig().egg_wall_model == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(egg_wall_model="typo"), assets=None)


def test_drop_spin_preserves_the_long_axis_and_off_candidates():
    from dojo.skills.world import AssetInfo, ObjectState
    from dojo.skills.geometry import R_to_quat
    info = AssetInfo(key="Rigid/egg/00000", bbox_min=np.array([-.02, -.02, -.03]),
                     bbox_max=np.array([.02, .02, .03]))
    R = H._rot([0, 1, 0], np.pi / 2)
    o = ObjectState("e", "Rigid", "egg", 0, np.zeros(3), R_to_quat(R), info)
    hole = dict(center=np.array([0.0, 0.0]), radius=.025, top=.81, floor=.76)
    base = H.egg_drop_targets(o, hole, .81, o.R)
    off = list(H.egg_drop_candidates(SimpleNamespace(), o, hole, .81))
    on = list(H.egg_drop_candidates(SimpleNamespace(egg_drop_spin="axis"), o, hole, .81))
    assert len(on) == 7 * len(base)
    for a, b in zip(base, off):
        assert a[:3] == b[:3] and np.array_equal(a[3], b[3])
    for index, (_, _, _, T) in enumerate(on):
        orig = base[index % len(base)][3]
        assert np.allclose(T[:3, 2], orig[:3, 2])
        assert np.allclose(T[:3, 3], orig[:3, 3])
    assert RuntimeConfig().egg_drop_spin == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(egg_drop_spin="typo"), assets=None)
