"""Stream H: ``RuntimeConfig.slot_loose`` / ``Skills.slot_loose`` ("carry" = today | "regrasp"): inside
``place_in_slots`` a loose pick is put down and the leg re-planned instead of carried to the slot (G3 store_tools v15
draws: 7 of 11 lost tools moved 45-394 mm in the hand during the carry, most after a loose pick). CPU only."""

from types import SimpleNamespace

import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import library
from dojo.skills.skills import SkillResult, Skills


class _WentOn(Exception):
    pass


def _sk(monkeypatch, regrasp_flag):
    sk = Skills.__new__(Skills)
    sk._loose_regrasp = regrasp_flag
    sk.refine = None
    calls = []
    monkeypatch.setattr(sk, "_record", lambda r: r, raising=False)
    monkeypatch.setattr(sk, "run_pick", lambda pp: SkillResult(True, "pick", "loose", 5, "left", note="drift 9 mm"),
                        raising=False)
    monkeypatch.setattr(sk, "regrasp", lambda arm: calls.append(arm) or SkillResult(True, "regrasp", "ok", 3, arm),
                        raising=False)

    def went_on():
        raise _WentOn()
    monkeypatch.setattr(sk, "refresh", went_on, raising=False)
    return sk, calls


def test_a_loose_slot_pick_is_put_down_only_with_regrasp(monkeypatch):
    pp = SimpleNamespace(arm="left", label="hammer")
    sk, calls = _sk(monkeypatch, False)
    with pytest.raises(_WentOn):                       # today: carried on to the (re-planned) place
        Skills.run_transfer(sk, pp, None, tol_xy=0.02)
    assert calls == []
    sk, calls = _sk(monkeypatch, True)
    pp.grasp = SimpleNamespace(note="box close 2 approach +1 tilt +15 off -11mm")
    r = Skills.run_transfer(sk, pp, None, tol_xy=0.02)
    assert r.status == "loose" and not r.ok and calls == ["left"]
    assert sk._loose_notes == {"hammer": {"box close 2 approach +1 tilt +15 off -11mm"}}    # not offered again


def test_the_flag_is_scoped_to_place_in_slots_and_reset(monkeypatch):
    seen = []

    def fake_slot_place(sk, lab, level, tg, kw):
        seen.append(sk._loose_regrasp)
        return SkillResult(True, "place", "ok", 1, "left")
    monkeypatch.setattr(library, "_slot_place", fake_slot_place)
    monkeypatch.setattr(library, "container_obstacle", lambda sk, c: None)
    monkeypatch.setattr(library, "slot_targets", lambda sk, l, c: [__import__("numpy").eye(4)])
    import numpy as np
    obj = SimpleNamespace(asset=SimpleNamespace(extents=np.full(3, 0.1)), center_w=np.zeros(3), R=np.eye(3))
    w = SimpleNamespace(objects={"hammer": obj, "box": obj}, obj=lambda l: obj)
    sk = SimpleNamespace(refresh=lambda: w, log=lambda m: None, held={"left": None, "right": None}, bank=lambda: None,
                         _loose_regrasp=False, slot_loose="regrasp", _record=lambda r: r)
    library.place_in_slots(sk, ["hammer"], "box")
    assert seen == [True] and sk._loose_regrasp is False
    sk.slot_loose = "carry"
    library.place_in_slots(sk, ["hammer"], "box")
    assert seen == [True, False]


def test_default_and_validation():
    assert RuntimeConfig().slot_loose == "carry" and Skills.__init__.__code__.co_varnames  # attribute set in __init__
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", slot_loose="drop"), assets=None)
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["slot_loose"][0] == RuntimeConfig().slot_loose
