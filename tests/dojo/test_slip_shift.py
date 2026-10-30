"""Stream H: ``RuntimeConfig.slip_shift`` / ``Skills.slip_shift`` ("off" = today | "across"): ``library._run_leg``
retries a thin object's pick that closed on air 7 mm to one side across its thinnest axis, then to the other. CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import library as L
from dojo.skills.skills import SkillResult


def _world(ext):
    o = SimpleNamespace(asset=SimpleNamespace(extents=np.asarray(ext, float)), R=np.eye(3))
    return SimpleNamespace(objects={"bread_1": o}, obj=lambda l: o)


def _slip(rose):
    return SkillResult(False, "pick", "grasp_slip", 40, "left", note=f"object rose {rose} mm (planned 178), drift 190 mm")


def test_no_rise_slip_only_for_thin_objects_that_did_not_lift():
    sk = SimpleNamespace(world=_world([0.117, 0.116, 0.012]))
    assert L._no_rise_slip(sk, "bread_1", _slip(-13))
    assert not L._no_rise_slip(sk, "bread_1", _slip(40))
    assert not L._no_rise_slip(SimpleNamespace(world=_world([0.06, 0.06, 0.05])), "bread_1", _slip(0))
    assert not L._no_rise_slip(sk, "bread_1", SkillResult(True, "transfer", "ok", 40, "left"))


def test_run_leg_retries_across_both_sides(monkeypatch):
    calls, logs = [], []
    results = iter([_slip(-13), _slip(0), SkillResult(True, "transfer", "ok", 40, "left")])

    def plan_transfer(label, targets, **kw):
        calls.append(kw.get("grasps"))
        return ("pp", "pl")
    sk = SimpleNamespace(world=_world([0.117, 0.116, 0.012]), slip_shift="across", held={"left": None, "right": None},
                         plan_transfer=plan_transfer, run_transfer=lambda *a, **k: next(results),
                         log=logs.append, refresh=lambda: None, _record=lambda r: r)
    monkeypatch.setattr(L, "_shifted_grasps", lambda sk, label, d, tilts: ("shift", d))
    r = L._run_leg(sk, "bread_1", ["T"], 2, first=("pp0", "pl0"))       # the caller's plan first, then re-planned
    assert r.ok and calls == [("shift", L.SLIP_SHIFT), ("shift", -L.SLIP_SHIFT)]
    assert sum("closed on air" in m for m in logs) == 2


def test_off_returns_the_slip():
    results = iter([_slip(-13)])
    sk = SimpleNamespace(world=_world([0.117, 0.116, 0.012]), slip_shift="off", held={"left": None, "right": None},
                         plan_transfer=lambda *a, **k: ("pp", "pl"), run_transfer=lambda *a, **k: next(results),
                         log=lambda m: None, refresh=lambda: None, _record=lambda r: r)
    assert L._run_leg(sk, "bread_1", ["T"], 2).status == "grasp_slip"


def test_default_and_validation():
    assert RuntimeConfig().slip_shift == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", slip_shift="on"), assets=None)
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["slip_shift"] == ("off", "across")


def test_sibling_order_default_and_validation():
    assert RuntimeConfig().sibling_order == "given"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", sibling_order="near"), assets=None)
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["sibling_order"] == ("given", "far")
