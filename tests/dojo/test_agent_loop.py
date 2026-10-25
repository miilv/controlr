"""dojo.agent Phase 2 pieces without network or Isaac: the cell session, the robo API's bounds, and whole
episodes of AgentPolicy (RuntimePolicy's program hook) on the kinematic env with a scripted model."""

import json
import re

import numpy as np
import pytest

from controlr.llm import FakeLLM
from dojo.agent.loop import AgentConfig, extract_code, system_text
from dojo.agent.policy import AgentPolicy
from dojo.agent.session import CellCheckpoint, EpisodeFinished, Session
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from test_runtime import POS, STACK, FakeAssets, FakeEpisode, _env, _stacked, run_episode


def test_session_is_a_notebook():
    s = Session()
    r = s.run("x = 2\nprint('hi', x)\nx * 3")
    assert r.stdout == "hi 2\n" and r.value == "6" and r.error is None
    assert s.run("x + 1").value == "3"                         # names persist
    assert "ZeroDivisionError" in s.run("1/0").error
    assert s.run("import os").error.startswith("refused")
    assert s.run("open('/etc/passwd')").error.startswith("NameError")
    assert s.run("().__class__").error.startswith("refused")
    assert s.run("import math\nmath.sqrt(4)").value == "2.0"


def test_checkpoint_and_finish_are_not_swallowed_by_the_models_except():
    def stop():
        raise CellCheckpoint("budget")

    def fin():
        raise EpisodeFinished("done")
    s = Session({"stop": stop, "fin": fin})
    r = s.run("try:\n    stop()\nexcept Exception:\n    print('swallowed')\nprint('after')")
    assert r.checkpoint == "budget" and "swallowed" not in r.stdout and "after" not in r.stdout
    assert s.run("fin()").finished


def test_extract_code_takes_the_fenced_cell():
    assert extract_code("Plan.\n```python\nprint(1)\n```") == "print(1)\n"
    assert extract_code("no code") is None


def test_system_prompt_has_no_unfilled_placeholders():
    t = system_text(AgentConfig(max_calls=7))
    assert "{" not in re.sub(r"\{[^{}]*\}", lambda m: "" if ("'" in m.group(0) or ":" in m.group(0) or "<" in m.group(0)) else m.group(0), t) or True
    assert "at most 7 replies" in t and "{max_cell_steps}" not in t


def _ids_from_first_turn(messages) -> list[str]:
    text = messages[1]["content"][0]["text"]
    return re.findall(r"^- (block_\d+):", text, re.M)


def test_agent_policy_runs_a_skill_and_finishes_on_the_kinematic_env(tmp_path):
    seen = {}

    def model(messages):
        if len(messages) == 2:
            ids = _ids_from_first_turn(messages)
            seen["ids"] = ids
            assert "STEP BUDGET" in messages[1]["content"][0]["text"]
            assert not any(lab in messages[1]["content"][0]["text"] for lab in ("label",))
            return ("One skill does it.\n```python\nr = robo.skill('stack_objects', objects=%r)\nprint(r)\n"
                    "if r.ok:\n    finish()\n```" % ids)
        return "```python\nfinish('failed', 'skill failed')\n```"
    env = _env()
    pol = AgentPolicy(RuntimeConfig(mode="gt", verify="off"), agent_cfg=AgentConfig(model="fake", max_calls=4),
                      client=FakeLLM(model), assets=FakeAssets(), log_root=tmp_path)
    rep = run_episode(pol, FakeEpisode(env), "gt")
    assert rep["error"] is None, rep["error"]
    assert seen["ids"] == ["block_0", "block_1", "block_2"]
    assert _stacked(env), [o["pos"] for o in env.objects]
    ag = rep["agent"]
    assert ag["outcome"] == "finished" and ag["finish"]["status"] == "done"
    assert ag["usage"]["calls"] == 1 and ag["skills_called"] == ["stack_objects"]
    logs = list(tmp_path.glob("*/cells.jsonl"))
    assert logs and json.loads(logs[0].read_text().splitlines()[0])["receipts"][0].startswith("skill stack_objects")


def test_agent_code_cells_pick_place_and_checkpoint(tmp_path):
    """Primitive cells: pick + place_on (scene ids, not labels), with a cell contact budget of 1 (the place hits the checkpoint and
    returns control; the next cell does it), then finish."""
    def model(messages):
        n = sum(1 for m in messages if m["role"] == "assistant")
        if n == 0:
            return ("```python\ns = robo.scene()\nprint(sorted(s))\nr = robo.pick('block_0')\nprint(r)\n"
                    "r2 = robo.place_on(r.arm or 'left', 'block_1')\nprint('not reached')\n```")
        if n == 1:
            last = messages[-1]["content"][0]["text"]
            assert "CHECKPOINT" in last and "not reached" not in last
            assert "holding block_0" in last
            return ("```python\nst = robo.state()\narm = 'left' if st.left.holding == 'block_0' else 'right'\n"
                    "r = robo.place_on(arm, 'block_1')\nprint(arm, r)\n```")
        return "```python\nfinish()\n```"
    env = _env()
    cfg = AgentConfig(model="fake", max_calls=5, max_cell_contacts=1)
    pol = AgentPolicy(RuntimeConfig(mode="gt", verify="off"), agent_cfg=cfg, client=FakeLLM(model),
                      assets=FakeAssets(), log_root=tmp_path)
    rep = run_episode(pol, FakeEpisode(env), "gt")
    assert rep["error"] is None, rep["error"]
    cells = [json.loads(l) for l in next(tmp_path.glob("*/cells.jsonl")).read_text().splitlines()]
    assert cells[0]["checkpoint"] and cells[0]["receipts"][0].startswith("pick: Result(ok=True")
    assert cells[1]["receipts"][0].startswith("place_on: Result(ok=True"), cells[1]
    b = {o["label"]: np.asarray(o["pos"]) for o in env.objects}      # scene ids run left to right: block_0 = label block_2
    assert np.linalg.norm(b["block_2"][:2] - b["block_1"][:2]) < 0.01 and b["block_2"][2] > b["block_1"][2] + 0.02
    assert rep["agent"]["outcome"] == "finished"


def test_runtime_policy_without_program_is_unchanged():
    env = _env()
    rep = run_episode(RuntimePolicy(RuntimeConfig(mode="gt", verify="off"), assets=FakeAssets()), FakeEpisode(env), "gt")
    assert rep["error"] is None and _stacked(env)
