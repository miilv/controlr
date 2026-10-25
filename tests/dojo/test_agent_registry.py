"""J2 skill registry: the real file loads and matches the library; strict filtering; P4 doc check."""

import inspect

import pytest

from dojo.agent import registry as R
from dojo.agent.loop import AgentConfig, system_text
from dojo.skills import library


def test_registry_loads_and_matches_the_library():
    reg = R.load()
    dojo = {e.name: e for e in reg.entries if e.backend == "dojo_skills"}
    assert set(dojo) == set(library.SKILLS), "every library skill has one registry entry (and no stale ones)"
    for name, e in dojo.items():
        fn = R.resolve(e)
        assert fn is library.SKILLS[name]
        params = list(inspect.signature(fn).parameters.values())[1:]          # drop sk
        known = {p.name for p in params}
        assert set(e.args) <= known, (name, set(e.args) - known)
        required = {p.name for p in params if p.default is inspect.Parameter.empty}
        assert required <= set(e.args), (name, required - set(e.args))
        assert e.obs_needs.get("held_look", False) in R.HELD_LOOK


def _reg(**over):
    base = dict(name="x", side="harness", doc="does x", args={}, obs_needs={"contract_only": False},
                backend="robot_server", entry="skills/x/skill_robodojo.py:run")
    base.update(over)
    return R.SkillEntry(**base)


def test_strict_runs_keep_only_contract_only_entries_and_drop_fallbacks():
    reg = R.load()
    reg.entries += [_reg(name="place_on_rgb"), _reg(name="place_on_rgb_strict", obs_needs={"contract_only": True}),
                    _reg(name="old_place", obs_needs={"contract_only": True}, fallback_for="place_on_rgb_strict")]
    R.validate(reg)
    strict = {e.name for e in reg.select(strict=True)}
    assert "place_on_rgb_strict" in strict and "place_on_rgb" not in strict and "old_place" not in strict
    assert "place_on_rgb" in {e.name for e in reg.select(strict=False)}
    assert not any(e.backend == "robot_server" for e in reg.select(backends=("dojo_skills",)))


def test_docs_must_not_name_tasks():
    reg = R.Registry(1, [_reg(doc="the best skill for stack_blocks_random")])
    with pytest.raises(R.RegistryError, match="stack_blocks"):
        R.validate(reg)
    with pytest.raises(R.RegistryError, match="duplicate"):
        R.validate(R.Registry(1, [_reg(), _reg()]))


def test_manual_from_the_registry_lists_the_prompted_skills():
    t = system_text(AgentConfig(registry="default"))
    for e in R.load().select(backends=("dojo_skills",)):
        assert e.signature() in t
    assert "stack_objects(objects: [id], nest: bool = false, ordered: bool = false)" in t


def test_scene_v3_lists_named_frames():
    import numpy as np
    from dojo.agent.scene import scene_text
    from dojo.skills.world import ArmState, AssetInfo, ObjectState, WorldState
    a = AssetInfo(key="Geometry/toy_base/00000", bbox_min=np.full(3, -0.05), bbox_max=np.full(3, 0.05),
                  support={"peg_0": [], "peg_1": []}, functional={"slot_left": {}}, active_functional={"lever": {}})
    o = ObjectState("base", "Geometry", "toy_base", 0, np.zeros(3), np.array([1.0, 0, 0, 0]), a)
    w = WorldState("", "", {"base": o}, {s: ArmState(np.zeros(6), np.zeros(7), 1.0) for s in ("left", "right")}, {})
    assert "frames peg_0, peg_1, slot_left, lever" in scene_text(w, version=3)[0]
    assert "frames" not in scene_text(w, version=2)[0]
