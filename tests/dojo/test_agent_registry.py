"""J2 skill registry (schema j2-registry/1): our file, robot-harness's file (pinned copy), the merged J4 file, strict
filtering, P4 checks, and the manual built from it.

tests/fixtures/robot_harness_registry_2efef1d55294.yaml = github.com/msannikov03/robot-harness skills/registry.yaml
at commit 2efef1d55294400434a8b819cce90f3276d25bbd (main d279be1, 2026-10-25), byte for byte.
"""

import inspect
from pathlib import Path

import pytest
import yaml

from dojo.agent import registry as R
from dojo.agent.loop import AgentConfig, system_text
from dojo.skills import library

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "robot_harness_registry_2efef1d55294.yaml"
MARKER = "  # ---- ilia side: added by J4 import ----"


def test_ilia_registry_loads_and_matches_the_library():
    reg = R.load()
    assert reg.schema == "j2-registry/1"
    assert {e.raw["side"] for e in reg.entries} == {"ilia"} and {e.raw["benchmark"] for e in reg.entries} == {"robodojo"}
    fns = {R.resolve(e) for e in reg.entries}
    assert fns == set(library.SKILLS.values()), "one registry entry per library skill, none stale"
    for e in reg.entries:
        fn = R.resolve(e)
        params = list(inspect.signature(fn).parameters.values())[1:]          # drop sk
        known = {p.name for p in params}
        assert set(e.raw["args"]) <= known, (e.name, set(e.raw["args"]) - known)
        required = {p.name for p in params if p.default is inspect.Parameter.empty}
        assert required <= set(e.raw["args"]), (e.name, required - set(e.raw["args"]))
        assert "catalog_mesh" in e.raw["obs_needs"]["reads"]
        assert e.derived_contract_only is True and e.contract_only        # contract_only derived from reads


def test_robot_harness_registry_loads_unchanged():
    reg = R.load(FIXTURE)
    assert len(reg.entries) == 83 and {e.raw["side"] for e in reg.entries} == {"misha"}
    strict = {e.name for e in reg.select(strict=True, benchmark="robodojo")}
    # docs/REGISTRY.md: the eight *_strict skills, put_into_bin_rgb_strict_v2 and return_home_budget
    assert len(strict) == 10 and "put_into_bin_rgb_strict_v2" in strict and "return_home_budget" in strict
    assert all(n.endswith(("_strict", "_strict_v2")) or n == "return_home_budget" for n in strict)
    assert reg.select(strict=True, benchmark="libero_pro") == []
    assert reg.select(backends=("dojo_skills",)) == []          # nothing of theirs runs in-process on the dojo runtime


def merged_text() -> str:
    """robot-harness's file with our entries appended under its ilia marker (the J4 shape)."""
    theirs = FIXTURE.read_text()
    ours = R.REGISTRY.read_text()
    body = ours.split(MARKER + "\n", 1)[1]
    assert theirs.rstrip().endswith(MARKER.strip())
    return theirs.rstrip("\n") + "\n" + body


def test_merged_j4_file_validates(tmp_path):
    p = tmp_path / "registry.yaml"
    p.write_text(merged_text())
    reg = R.load(p)
    assert len(reg.entries) == 83 + len(R.load().entries)
    names = [e.name for e in reg.entries]
    assert len(names) == len(set(names))
    strict = reg.select(strict=True, benchmark="robodojo")
    assert {e.raw["side"] for e in strict} == {"misha", "ilia"}
    assert yaml.safe_load(p.read_text())["schema"] == "j2-registry/1"


def _e(**over):
    base = dict(name="x", side="misha", benchmark="robodojo", doc="does x\n", args={}, measured="none",
                obs_needs={"contract_only": False, "held_look": False, "reads": ["joints", "privileged_poses"]},
                backend="robot_server", entry="skills/x/skill_robodojo.py:run")
    base.update(over)
    return R.SkillEntry(base)


def test_validation_mirrors_harness_rules():
    ok = [_e(), _e(name="y", obs_needs={"contract_only": True, "held_look": "wrist", "reads": ["joints", "held_look"]})]
    assert R.validate(ok)[0] == []
    bad = {
        "P4 task": _e(doc="Use for stack_blocks.\n"),
        "P4 spaced": _e(doc="Pour balls into vase.\n"),
        "P4 seed": _e(doc="Practised on seed 1.\n"),
        "contract": _e(obs_needs={"contract_only": True, "held_look": False, "reads": ["joints", "privileged_poses"]}),
        "held_look": _e(obs_needs={"contract_only": True, "held_look": "wrist", "reads": ["joints"]}),
        "side": _e(side="dojo"),
        "doc lines": _e(doc="a\nb\nc\nd\ne\nf\ng\n"),
        "ilia reads": _e(side="ilia", backend="dojo_skills", entry="m.n:f",
                         obs_needs={"contract_only": True, "held_look": False}),
    }
    for why, e in bad.items():
        assert R.validate([e])[0], why
    assert R.validate([_e(), _e()])[0]                          # duplicate name


def test_strict_selection_and_fallbacks():
    reg = R.Registry("j2-registry/1", [_e(name="place_rgb"),
                                       _e(name="place_rgb_strict", obs_needs={"contract_only": True, "held_look": False,
                                                                              "reads": ["joints"]}),
                                       _e(name="old_place", fallback=True,
                                          obs_needs={"contract_only": True, "held_look": False, "reads": ["joints"]}),
                                       _e(name="lp", benchmark="libero_pro")])
    assert [e.name for e in reg.select(strict=True)] == ["place_rgb_strict"]
    assert [e.name for e in reg.select()] == ["place_rgb", "place_rgb_strict"]


def test_chooser_block_and_manual_from_the_registry():
    reg = R.load()
    block = R.chooser_docs(reg.entries)
    assert block.startswith("### stack_objects\nargs:\n  objects: ")
    for hidden in ("measured", "stats", "notes", "dojo.skills.library", "journal"):
        assert hidden not in block
    t = system_text(AgentConfig(registry="default", strict=True))
    for e in reg.select(strict=True, backends=("dojo_skills",)):
        assert f"### {e.name}" in t


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
