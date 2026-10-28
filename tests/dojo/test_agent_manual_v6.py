"""manual_v6 (the default): the agent's prompt carries no per-task recipes, no checker-derived numbers and no judging
language (robot-harness RULES-robodojo §14.5; journal 2026-10-28-dojo-agent-manual-v6)."""
import re

from dojo.agent.loop import AgentConfig, system_text
from dojo.agent.manual import SKILL_DOCS, render_manual

NUMBER_UNIT = re.compile(r"\d+(\.\d+)?\s*(cm|mm|degrees?|deg|%|steps?)\b|\b0\.[0-9]+\s*m\b|opening above")
JUDGING = re.compile(r"fails? the task|\bchecked\b|\bjudged\b|\bscor(e|es|ed|ing)\b|\breward", re.I)


def test_no_instruction_pattern_recipes():
    text = render_manual("manual_v6")
    assert "Instruction patterns" not in text
    assert "how the benchmark checks results" not in text
    for s in ("## 1. World", "## 2. Ending and banking", "## 3. Skill library", "## 4. Code", "## 5. Working rules"):
        assert s in text


def _section2_and_skills(version: str) -> str:
    text = system_text(AgentConfig(manual=version))
    return text.split("## 2. ")[1].split("## 4. Code")[0]     # §2 (rules) + §3 (skill library and its docs)


def test_v6_rules_and_skill_docs_are_number_free_and_never_judge():
    part = _section2_and_skills("manual_v6")
    assert [m.group(0) for m in NUMBER_UNIT.finditer(part)] == []
    assert [m.group(0) for m in JUDGING.finditer(part)] == []
    for name, (_sig, doc) in SKILL_DOCS.items():
        assert not NUMBER_UNIT.search(doc), name
        assert not JUDGING.search(doc), name


def test_v6_keeps_the_generic_rules():
    text = render_manual("manual_v6")
    for s in ("`finish()`", "Open both grippers after each completed stage", "Counts and orders given in the "
              "instruction are exact", "Hold still while another arm moves", "Do not release an object unless the "
              "instruction asks you to put it somewhere", "record it in code before the scene changes"):
        assert s in text, s


def test_v6_is_the_default_manual():
    assert AgentConfig().manual == "manual_v6"


def test_v6_with_the_registry_skill_list_is_number_free_and_never_judges():
    # the J2 registry's docs reach the same prompt slot when AgentConfig.registry is set
    text = system_text(AgentConfig(manual="manual_v6", registry="default"))
    part = text.split("## 2. ")[1].split("## 4. Code")[0]
    assert [m.group(0) for m in NUMBER_UNIT.finditer(part)] == []
    assert [m.group(0) for m in JUDGING.finditer(part)] == []
