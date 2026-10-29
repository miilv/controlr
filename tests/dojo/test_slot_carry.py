"""Stream H: ``RuntimeConfig.slot_carry`` / ``Skills.slot_carry`` ("v1" = today | "gentle": ``place_in_slots``'
lift and carry at 0.05 rad/step with an 8-step ease, ``library.SLOT_GENTLE``). G3 store_tools v15 draws: 7 of 11 lost
tools moved 45-394 mm in the hand on the way to their slots. The effect is screened in G3; here: what reaches the
transfers, the default, validation. CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import library
from dojo.skills.skills import SkillResult


def _run(monkeypatch, carry, level):
    seen = []

    def fake(sk, lab, targets, **kw):
        seen.append(kw)
        return SkillResult(True, "place", "ok", 1, "left")
    monkeypatch.setattr(library, "place_oriented", fake)
    monkeypatch.setattr(library, "container_obstacle", lambda sk, c: None)
    monkeypatch.setattr(library, "slot_targets", lambda sk, l, c: [np.eye(4) if level else library.T_from((0, 0, 0), library.rot_z(1.5))])
    monkeypatch.setattr(library, "_level_grasps", lambda sk, l: [])
    obj = SimpleNamespace(asset=SimpleNamespace(extents=np.full(3, 0.1)), center_w=np.zeros(3), R=np.eye(3))
    w = SimpleNamespace(objects={"hammer": obj, "box": obj}, obj=lambda l: obj)
    sk = SimpleNamespace(refresh=lambda: w, log=lambda m: None, held={"left": None, "right": None}, bank=lambda: None,
                         slot_carry=carry, _record=lambda r: r)
    library.place_in_slots(sk, ["hammer"], "box")
    return seen[0]


@pytest.mark.parametrize("level", [True, False])
def test_gentle_reaches_the_slot_transfers(monkeypatch, level):
    v1 = _run(monkeypatch, "v1", level)
    g = _run(monkeypatch, "gentle", level)
    assert "ease" not in v1["place_kw"] and g["place_kw"]["ease"] == library.SLOT_GENTLE[1]
    assert g["place_kw"]["approach_rad"] == library.SLOT_GENTLE[0]
    if level:                                   # the level carry was already 0.05: only the ease changes
        assert v1["place_kw"]["approach_rad"] == library.LEVEL_CARRY
    else:                                       # the re-orienting transfer also lifts slower
        assert "transfer_kw" not in v1 and g["transfer_kw"]["pick_kw"]["lift_rad"] == library.SLOT_GENTLE[0]


def test_default_and_validation():
    assert RuntimeConfig().slot_carry == "v1"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", slot_carry="slow"), assets=None)
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["slot_carry"][0] == RuntimeConfig().slot_carry
