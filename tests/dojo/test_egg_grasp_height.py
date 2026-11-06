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
    assert calls[-1]["tilts"] == (40.0, -40.0, 55.0, -55.0)
    assert calls[-1]["ends"] is False
    assert RuntimeConfig().egg_grasp_angles == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(egg_grasp_angles="typo"), assets=None)
