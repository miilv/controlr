"""Unpromoted skill versions never reach a chooser (robot-harness S2 routing-integrity finding 1).

robot-harness's ``skills/registry.yaml`` marks a new version that is not promoted yet ``status: candidate``;
``harness/registry.py``'s choosers leave candidates out unless the caller opts in (``d918f85``), and their copy of this
loader got ``Registry.select(..., candidates=False)`` (``a4f4d5e``, docs/notes/S2-VERIFICATION-2026-10-27.md). This is
the same rule here, for every backend and both strict and non-strict selection, plus the validator's status rules.

tests/fixtures/robot_harness_registry_excerpt_9b424334ac66.yaml = the header and eight entries of
github.com/msannikov03/robot-harness ``skills/registry.yaml`` at main 9b424334ac666d4db499e5e233621a6e6d2cb8be
(2026-10-28), each entry byte for byte with its section marker: lift_strict, place_on_rgb_strict,
put_into_bin_rgb_strict_v2 (``default: true``), pour_bottle, pour_bottle_v2 (candidate, non-strict), stack_objects
(the J4-imported ilia entry, ``catalog_mesh``), lift_strict_v2 and place_on_rgb_strict_v2 (strict candidates).
"""

import copy
import os
from pathlib import Path

import pytest
import yaml

from dojo.agent import registry as R
from dojo.agent.loop import AgentConfig, registry_skills, system_text

EXCERPT = Path(__file__).resolve().parents[1] / "fixtures" / "robot_harness_registry_excerpt_9b424334ac66.yaml"
EXCERPT_CANDIDATES = {"pour_bottle_v2", "lift_strict_v2", "place_on_rgb_strict_v2"}


def _e(name, backend="robot_server", contract_only=True, **over):
    reads = ["joints", "flange"] if contract_only else ["joints", "privileged_poses"]
    base = dict(name=name, side="misha" if backend == "robot_server" else "ilia", benchmark="robodojo",
                doc="does a thing\n", args={}, measured="none",
                obs_needs={"contract_only": contract_only, "held_look": False, "reads": reads}, backend=backend,
                entry="skills/x/skill_robodojo.py:run" if backend == "robot_server" else "dojo.skills.library:x")
    base.update(over)
    return R.SkillEntry(base)


def _mixed() -> R.Registry:
    """A promoted and a candidate entry for every backend x contract_only, plus a candidate J5 loser."""
    es = []
    for be in R.BACKENDS:
        for co in (True, False):
            tag = f"{be}_{'c' if co else 'n'}"
            es += [_e(f"{tag}", be, co), _e(f"{tag}_promoted", be, co, status="promoted"),
                   _e(f"{tag}_v2", be, co, status="candidate")]
    es.append(_e("old_v2", "dojo_skills", True, status="candidate", fallback=True))
    return R.Registry(R.SCHEMA, es)


def test_the_runtime_select_leaves_candidates_out_for_every_backend():
    reg = _mixed()
    assert R.validate(reg.entries)[0] == []
    for strict in (False, True):
        for benchmark in (None, "robodojo"):
            for backends in (R.BACKENDS, ("robot_server",), ("dojo_skills",)):
                got = reg.select(strict=strict, benchmark=benchmark, backends=backends)
                assert got, (strict, benchmark, backends)
                assert not any(e.is_candidate for e in got), (strict, benchmark, backends)
                assert {e.raw["backend"] for e in got} == set(backends)
                assert all(e.contract_only for e in got) or not strict
    # absent and "promoted" are both promoted
    assert {e.name for e in reg.select(backends=("dojo_skills",))} == {
        "dojo_skills_c", "dojo_skills_c_promoted", "dojo_skills_n", "dojo_skills_n_promoted"}


def test_opt_in_lists_candidates_and_keeps_the_other_filters():
    reg = _mixed()
    for strict in (False, True):
        for backends in (R.BACKENDS, ("robot_server",), ("dojo_skills",)):
            base = {e.name for e in reg.select(strict=strict, backends=backends)}
            opt = {e.name for e in reg.select(strict=strict, backends=backends, candidates=True)}
            want = {f"{be}_{co}_v2" for be in backends for co in (("c",) if strict else ("c", "n"))}
            assert opt - base == want, (strict, backends)
    # the opt-in does not lift strict, the backend filter or the J5 losers
    assert "robot_server_n_v2" not in {e.name for e in reg.select(strict=True, candidates=True)}
    assert "old_v2" not in {e.name for e in reg.select(candidates=True)}


def test_validator_status_rules(tmp_path):
    for status in (None, "candidate", "promoted"):
        e = _e("x") if status is None else _e("x", status=status)
        assert R.validate([e])[0] == [], status
    for bad in ("maybe", "Candidate", "", True):
        assert any("status must be one of" in x for x in R.validate([_e("x", status=bad)])[0]), bad
    assert any("candidate cannot be the default" in x for x in R.validate([_e("x", status="candidate", default=True)])[0])
    assert R.validate([_e("x", status="promoted", default=True)])[0] == []
    assert R.validate([_e("x", default=True)])[0] == []
    with pytest.raises(R.RegistryError, match="status must be one of"):
        R.load(_written(tmp_path / "registry.yaml", [dict(_e("x", status="maybe").raw)]))


def _written(p: Path, entries: list[dict]) -> Path:
    p.write_text(yaml.safe_dump({"schema": R.SCHEMA, "entries": entries}, sort_keys=False))
    return p


def test_robot_harness_excerpt_selects_like_filter_strict():
    """harness/registry.filter_strict / filter_benchmark on the same eight entries (computed there at 9b42433)."""
    reg = R.load(EXCERPT)
    assert {e.name for e in reg.entries if e.is_candidate} == EXCERPT_CANDIDATES
    strict = ["lift_strict", "place_on_rgb_strict", "put_into_bin_rgb_strict_v2", "stack_objects"]
    assert [e.name for e in reg.select(strict=True)] == strict
    assert [e.name for e in reg.select(strict=True, candidates=True)] == strict + ["lift_strict_v2",
                                                                                   "place_on_rgb_strict_v2"]
    assert [e.name for e in reg.select()] == strict[:3] + ["pour_bottle", "stack_objects"]
    assert {e.name for e in reg.select(candidates=True)} == {e.name for e in reg.entries}
    # a pour chooser offers only the original pour_bottle (D-C14 there)
    assert {e.name for e in reg.select() if e.name.startswith("pour_bottle")} == {"pour_bottle"}
    # catalog_mesh is a contract read (harness/registry.py f7f9017): the imported ilia entry stays strict
    assert reg.by_name()["stack_objects"].derived_contract_only is True


def test_registry_skills_prompts_no_candidate(tmp_path):
    """loop.registry_skills (the prompt's skill list and Robo.skill's callable set) never carries a candidate, also
    when one is on the dojo_skills backend it calls."""
    data = yaml.safe_load(EXCERPT.read_text())
    ours = next(e for e in data["entries"] if e["name"] == "stack_objects")
    cand = copy.deepcopy(ours)
    cand.update(name="stack_objects_v2", status="candidate")
    data["entries"].append(cand)
    p = tmp_path / "registry.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    for path in (EXCERPT, p):
        for strict in (False, True):
            got = registry_skills(AgentConfig(registry=str(path), strict=strict))
            assert [e.name for e in got] == ["stack_objects"], (path.name, strict)
    assert R.load(p).select(backends=("dojo_skills",), candidates=True)[-1].name == "stack_objects_v2"
    t = system_text(AgentConfig(registry=str(p)))
    assert "### stack_objects\n" in t and "### stack_objects_v2" not in t


@pytest.mark.skipif(not (Path(os.environ.get("HARNESS_ROOT", "/nonexistent")) / "skills" / "registry.yaml").is_file(),
                    reason="set HARNESS_ROOT to a robot-harness checkout to check its current registry")
def test_current_robot_harness_registry_if_present():
    """The live file from a robot-harness checkout (HARNESS_ROOT): it loads unchanged and no candidate is selected."""
    path = Path(os.environ["HARNESS_ROOT"]) / "skills" / "registry.yaml"
    reg = R.load(path)
    cands = {e.name for e in reg.entries if e.is_candidate}
    assert cands
    for strict in (False, True):
        for backends in (R.BACKENDS, ("robot_server",), ("dojo_skills",)):
            assert not {e.name for e in reg.select(strict=strict, backends=backends)} & cands
        got = registry_skills(AgentConfig(registry=str(path), strict=strict))
        assert got and not any(e.is_candidate for e in got)
