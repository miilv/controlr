"""Full turn loop with FakeLLM + MockRobot (no network, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from controlr.config import Config, load_config
from controlr.llm.fake import FakeLLM
from controlr.robot.mock import MockRobot
from controlr.runlog import read_jsonl

HOME_TCP = (-0.300, -0.150, 0.150)       # MockRobot home TCP (base frame, m)
NEAR = [-0.300, -0.150, 0.100]           # 50 mm below home: reachable with two -25 mm moves
FAR = [-0.300, 0.150, 0.100]


class RecMock(MockRobot):
    """MockRobot that records lifecycle calls."""

    def __init__(self, params=None):
        super().__init__({"width": 160, "height": 120, "settle_s": 0.0, **(params or {})})
        self.calls: list[str] = []

    def hold(self):
        self.calls.append("hold")

    def close(self):
        self.calls.append("close")


def _cfg(tmp_path: Path, target=NEAR, **over) -> Config:
    sets = [f"log.root={tmp_path}", "planner.enabled=false", "robot.backend=mock", "task.name=reach",
            f"task.params={{target: {list(target)}}}", "llm.base_url=http://fake", "episode.max_turns=6",
            "observation.renderers=[raw]"]
    sets += [f"{k}={v}" for k, v in over.items()]
    return load_config(None, sets, dotenv=None)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_success_after_moves(tmp_path):
    robot = RecMock()
    llm = FakeLLM(["MOVE ee_delta 0 0 -25\nSTATUS OK", "MOVE ee_delta 0 0 -25\nSTATUS OK",
                   "STATUS DONE reached"])
    from controlr.loop import run_episode

    # early_stop off: the fake (like the real stream) drops usage when cut early
    res = run_episode(_cfg(tmp_path, **{"llm.early_stop": "false"}), robot=robot, llm=llm)
    assert res.outcome == "success", res.error
    assert res.success and res.turns == 3
    assert robot.calls[-2:] == ["hold", "close"]
    run = Path(res.run_dir)
    for f in ("config.yaml", "system_prompt.md", "messages.jsonl", "turns.jsonl", "summary.json"):
        assert (run / f).exists(), f
    turns = read_jsonl(run / "turns.jsonl")
    assert [t["turn"] for t in turns] == [0, 1, 2]
    assert turns[0]["actions"][0]["raw"].startswith("MOVE")
    assert turns[0]["actions"][0]["values_si"][2] == pytest.approx(-0.025)
    assert turns[0]["executed"] and "llm" in turns[0] and "timings" in turns[0]
    msgs = read_jsonl(run / "messages.jsonl")
    assert msgs[0]["role"] == "system"
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant"] * 3
    imgs = list((run / "images").glob("*.jpg"))
    assert imgs
    summary = json.loads((run / "summary.json").read_text())
    assert summary["outcome"] == "success" and summary["turns"] == 3
    assert summary["cache_read_share"] is not None
    # transcript prefix is append-only: every request extends the previous one
    calls = llm.calls
    for a, b in zip(calls, calls[1:]):
        n = len(a["messages"])
        strip = lambda ms: json.dumps(ms, sort_keys=True).replace('"cache_control"', "")  # noqa: E731
        assert len(b["messages"]) == n + 2
        assert [m["role"] for m in b["messages"][:n]] == [m["role"] for m in a["messages"]]
        assert strip(b["messages"][:1]) == strip(a["messages"][:1])


def test_done_not_verified_continues_then_max_turns(tmp_path):
    robot = RecMock()
    llm = FakeLLM(["STATUS DONE"])
    from controlr.loop import run_episode

    res = run_episode(_cfg(tmp_path, FAR, **{"episode.max_turns": 3}), robot=robot, llm=llm)
    assert res.outcome == "max_turns" and not res.success and res.turns == 3


def test_trust_done(tmp_path):
    from controlr.loop import run_episode

    res = run_episode(_cfg(tmp_path, FAR, **{"episode.trust_done": "true"}),
                      robot=RecMock(), llm=FakeLLM(["STATUS DONE"]))
    assert res.outcome == "done_unverified" and res.turns == 1


def test_fail_ends(tmp_path):
    from controlr.loop import run_episode

    res = run_episode(_cfg(tmp_path), robot=RecMock(), llm=FakeLLM(["STATUS FAIL cannot"]))
    assert res.outcome == "fail" and res.turns == 1


def test_parse_error_streak(tmp_path):
    from controlr.loop import run_episode

    res = run_episode(_cfg(tmp_path, **{"episode.max_parse_errors": 2}), robot=RecMock(),
                      llm=FakeLLM(["I think I should move left."]))
    assert res.outcome == "parse_errors" and res.turns == 2
    turns = read_jsonl(Path(res.run_dir) / "turns.jsonl")
    assert turns[0]["parse_errors"]
    assert "PARSE ERROR" in turns[0]["feedback"]


def test_llm_error_ends_episode(tmp_path):
    from controlr.loop import run_episode

    robot = RecMock()
    res = run_episode(_cfg(tmp_path), robot=robot,
                      llm=FakeLLM(["MOVE ee_delta 0 0 -10\nSTATUS OK", RuntimeError("503 retries exhausted")]))
    assert res.outcome == "llm_error" and res.turns == 2
    assert "retries" in res.error
    assert "close" in robot.calls
    assert json.loads((Path(res.run_dir) / "summary.json").read_text())["outcome"] == "llm_error"


def test_planner_plan_in_turn0(tmp_path):
    from controlr.loop import run_episode

    llm = FakeLLM(["1. go down 50 mm\n2. say DONE", "MOVE ee_delta 0 0 -50\nSTATUS OK", "STATUS DONE"])
    cfg = _cfg(tmp_path, **{"planner.enabled": "true"})
    res = run_episode(cfg, robot=RecMock(), llm=llm)
    assert res.outcome == "success", res.error
    run = Path(res.run_dir)
    assert (run / "plan.md").read_text().startswith("1. go down")
    assert llm.calls[0]["model"] == cfg.planner.model
    assert llm.calls[0]["extra_body"] is None   # effort lives in the model id (…-xhigh)
    # planner request carries no cache markers; control turn 0 contains the plan text
    assert "cache_control" not in json.dumps(llm.calls[0]["messages"])
    turn0 = json.dumps(llm.calls[1]["messages"][1])
    assert "go down 50 mm" in turn0 and llm.calls[1]["model"] == cfg.llm.model


def test_planner_excluded_from_context(tmp_path):
    from controlr.loop import run_episode

    llm = FakeLLM(["SECRET PLAN", "STATUS FAIL"])
    run_episode(_cfg(tmp_path, **{"planner.enabled": "true", "planner.include_in_context": "false"}),
                robot=RecMock(), llm=llm)
    assert "SECRET PLAN" not in json.dumps(llm.calls[1]["messages"])


def test_cache_markers_on_claude_routes(tmp_path):
    from controlr.loop import run_episode

    llm = FakeLLM(["MOVE ee_delta 0 0 -10\nSTATUS OK", "STATUS FAIL"])
    run_episode(_cfg(tmp_path), robot=RecMock(), llm=llm)
    assert "cache_control" in json.dumps(llm.calls[1]["messages"])
    llm2 = FakeLLM(["STATUS FAIL"])
    run_episode(_cfg(tmp_path, **{"llm.model": "gpt-5.6-luna-low"}), robot=RecMock(), llm=llm2)
    assert "cache_control" not in json.dumps(llm2.calls[0]["messages"])


def test_early_stop_passed(tmp_path):
    from controlr.loop import run_episode

    llm = FakeLLM(["STATUS FAIL\nand then a lot of rambling text that must not be kept"])
    res = run_episode(_cfg(tmp_path), robot=RecMock(), llm=llm)
    rec = res.records[0]
    assert rec["stopped_early"] and "rambling" not in rec["reply"]


def test_keyboard_interrupt_is_safe(tmp_path):
    from controlr.loop import run_episode

    robot = RecMock()

    def boom(messages):
        raise KeyboardInterrupt

    res = run_episode(_cfg(tmp_path), robot=robot, llm=FakeLLM(boom))
    assert res.outcome == "interrupted"
    assert robot.calls[-2:] == ["hold", "close"]
    assert (Path(res.run_dir) / "summary.json").exists()


def test_backend_exception_recorded(tmp_path):
    from controlr.loop import run_episode

    class Broken(RecMock):
        def execute(self, actions):
            raise RuntimeError("sim crashed")

    res = run_episode(_cfg(tmp_path), robot=Broken(), llm=FakeLLM(["MOVE ee_delta 0 0 -10\nSTATUS OK"]))
    assert res.outcome == "error" and "sim crashed" in res.error
    assert (Path(res.run_dir) / "error.txt").exists()


def test_with_real_mock_robot(tmp_path):
    """The robot built by make_robot from the config (backend=mock)."""
    pytest.importorskip("controlr.robot", reason="make_robot")
    from controlr import robot as robot_pkg
    if not hasattr(robot_pkg, "make_robot"):
        pytest.skip("controlr.robot.make_robot not available yet")
    from controlr.loop import run_episode

    llm = FakeLLM(["MOVE ee_delta 10 0 0\nSTATUS OK", "STATUS FAIL giving up"])
    res = run_episode(_cfg(tmp_path), llm=llm)
    assert res.outcome == "fail", res.error
    assert res.turns == 2
    turns = read_jsonl(Path(res.run_dir) / "turns.jsonl")
    assert turns[0]["state"]["tcp_pos"] is not None
