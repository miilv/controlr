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
    _assert_prefix_stable(llm.calls)


def _strip_markers(messages):
    """Remove cache_control structurally (markers move every turn by design)."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = [{k: v for k, v in p.items() if k != "cache_control"} if isinstance(p, dict) else p for p in c]
        out.append({**m, "content": c})
    return json.dumps(out, sort_keys=True)


def _assert_prefix_stable(calls):
    """Every request = previous request (byte-identical without markers) + reply + user turn."""
    for a, b in zip(calls, calls[1:]):
        n = len(a["messages"])
        assert len(b["messages"]) == n + 2
        assert _strip_markers(b["messages"][:n]) == _strip_markers(a["messages"])


def test_prefix_is_byte_stable_with_overlays_and_diff(tmp_path):
    """Review contracts #25: the old check compared only the system message."""
    from controlr.loop import run_episode

    llm = FakeLLM(["MOVE ee_delta 10 0 0\nSTATUS OK", "MOVE ee_delta 0 10 0\nSTATUS OK",
                   "MOVE ee_delta 0 0 -10\nSTATUS OK", "MOVE ee_delta -10 0 0\nSTATUS OK", "STATUS FAIL"])
    res = run_episode(_cfg(tmp_path, **{"observation.renderers": "[grid,ee_marker,diff]"}),
                      robot=RecMock(), llm=llm)
    assert res.outcome == "fail" and len(llm.calls) == 5
    _assert_prefix_stable(llm.calls)


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
    assert "WARN: reply not understood" in turns[0]["feedback"]


def test_length_cutoff_is_explained(tmp_path):
    """A reply cut by max_tokens (thinking routes) gets a specific PARSE ERROR, not only
    'missing STATUS' (smoke test: Sonnet spent all 2000 tokens thinking)."""
    from controlr.loop import run_episode

    res = run_episode(_cfg(tmp_path, **{"episode.max_parse_errors": 1, "llm.max_tokens": 5}),
                      robot=RecMock(), llm=FakeLLM(["I am thinking very hard about the next move"]))
    turns = read_jsonl(Path(res.run_dir) / "turns.jsonl")
    assert turns[0]["finish_reason"] == "length"
    assert "cut off by the output limit" in turns[0]["parse_errors"][0]


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


class StopMock(RecMock):
    """Every execute ends in a STOP event of the given kind (contact stop / diverged sim)."""

    def __init__(self, kind="collision"):
        super().__init__()
        self.kind = kind

    def execute(self, actions):
        from controlr.types import EventLevel, SafetyEvent
        rep = super().execute(actions)
        rep.stopped = True
        rep.events.append(SafetyEvent(EventLevel.STOP, self.kind, "motion stopped: contact force 120 N > 80 N"))
        return rep


def test_stops_end_episode_after_max_stops(tmp_path):
    from controlr.loop import run_episode

    llm = FakeLLM(["MOVE ee_delta 0 0 -10\nSTATUS OK"])
    res = run_episode(_cfg(tmp_path, **{"episode.max_stops": "2"}), robot=StopMock(), llm=llm)
    assert res.outcome == "safety_stop" and res.turns == 2
    turns = read_jsonl(Path(res.run_dir) / "turns.jsonl")
    assert "STOP: motion stopped" in turns[0]["feedback"]      # the model saw the first stop


def test_unstable_stop_ends_immediately(tmp_path):
    from controlr.loop import run_episode

    res = run_episode(_cfg(tmp_path, **{"episode.max_stops": "5"}), robot=StopMock("unstable"),
                      llm=FakeLLM(["MOVE ee_delta 0 0 -10\nSTATUS OK"]))
    assert res.outcome == "safety_stop" and res.turns == 1


def test_run_nonce_in_turn0_only(tmp_path):
    from controlr.loop import run_episode

    llm = FakeLLM(["STATUS FAIL x"])
    res = run_episode(_cfg(tmp_path), robot=RecMock(), llm=llm)
    first = llm.calls[0]["messages"][1]["content"][0]["text"]
    assert first.startswith(f"RUN {Path(res.run_dir).name}\nTASK:")
    assert "RUN " not in llm.calls[0]["messages"][0]["content"][0]["text"]   # system prompt untouched


# ---------------------------------------------------------------------------
# review fixes: run-log completeness, fake marking, plan file, end_on_goal, ...
# ---------------------------------------------------------------------------

def test_turn_records_rebuild_obs_action_next_obs(tmp_path):
    """Review contracts #10: (obs_t, action_t, obs_t+1) must be reconstructable from
    turns.jsonl alone, including the terminal turn's final frame."""
    from controlr.loop import run_episode

    llm = FakeLLM(["MOVE ee_delta 0 0 -25\nSTATUS OK", "MOVE ee_delta 0 0 -25\nSTATUS OK", "STATUS DONE"])
    res = run_episode(_cfg(tmp_path, **{"log.save_raw_frames": "true"}), robot=RecMock(), llm=llm)
    run = Path(res.run_dir)
    turns = read_jsonl(run / "turns.jsonl")
    setup = json.loads((run / "setup.json").read_text())
    assert turns[0]["obs_images"] == setup["obs0_images"]
    for a, b in zip(turns, turns[1:]):
        assert b["obs_images"] == a["next_obs_images"]
        assert b["state_before"]["tcp_pos"] == pytest.approx(a["state"]["tcp_pos"])
    for t in turns:
        assert t["obs_images"] and t["next_obs_images"] and t["state_before"] and t["state"]
        for sha in t["obs_images"] + t["next_obs_images"]:
            assert (run / "images" / f"{sha}.jpg").exists()
        assert (run / t["next_raw_frames"][0]).exists()
    assert turns[0]["executed"][0]["q_target"] and "reasoning" in turns[0]
    assert "feedback" not in turns[-1]                    # terminal turn: no next user turn
    cams = json.loads((run / "cameras.json").read_text())
    assert cams["scene"]["K"] and len(cams["scene"]["T_cam_base"]) == 4
    spec = json.loads((run / "spec.json").read_text())
    assert spec["workspace_lo"] and spec["table_z"] is not None


def test_fake_runs_are_marked(tmp_path):
    """Review contracts #11: dry runs were indistinguishable from live runs."""
    from controlr.loop import run_episode

    res = run_episode(_cfg(tmp_path), robot=RecMock(), llm=FakeLLM(["STATUS FAIL"]))
    run = Path(res.run_dir)
    assert run.name.split("_", 1)[1].startswith("fake_")
    assert json.loads((run / "summary.json").read_text())["llm_backend"] == "fake"
    assert json.loads((run / "setup.json").read_text())["llm_backend"] == "fake"


def test_plan_file_replaces_the_planner_call(tmp_path):
    from controlr.loop import run_episode

    plan = tmp_path / "plan.txt"
    plan.write_text("PINNED PLAN: go down")
    llm = FakeLLM(["STATUS FAIL"])
    res = run_episode(_cfg(tmp_path, **{"planner.enabled": "true", "planner.plan_file": str(plan)}),
                      robot=RecMock(), llm=llm)
    assert len(llm.calls) == 1 and "PINNED PLAN" in json.dumps(llm.calls[0]["messages"][1])
    assert res.plan == "PINNED PLAN: go down"


def test_planner_gets_long_timeout_and_few_retries(tmp_path):
    from controlr.loop import run_episode

    llm = FakeLLM(["a plan", "STATUS FAIL"])
    run_episode(_cfg(tmp_path, **{"planner.enabled": "true"}), robot=RecMock(), llm=llm)
    assert llm.calls[0]["timeout_s"] == 600 and llm.calls[0]["max_retries"] == 1
    assert llm.calls[1]["timeout_s"] is None


def test_end_on_goal_and_success_fields(tmp_path):
    from controlr.loop import run_episode

    llm = FakeLLM(["MOVE ee_delta 0 0 -25\nSTATUS OK", "MOVE ee_delta 0 0 -25\nSTATUS OK", "STATUS OK"])
    res = run_episode(_cfg(tmp_path, **{"episode.end_on_goal": "true"}), robot=RecMock(), llm=llm)
    assert res.outcome == "goal_reached" and res.success and not res.success_verified and res.turns == 2
    s = json.loads((Path(res.run_dir) / "summary.json").read_text())
    assert s["success_verified"] is False and s["success"] is True
    assert s["planner"] is None and "cache_regressions" in s


def test_missing_camera_fails_early_with_a_clear_message(tmp_path):
    from controlr.loop import run_episode

    res = run_episode(_cfg(tmp_path, **{"observation.cameras": "[scene,top]"}), robot=RecMock(),
                      llm=FakeLLM(["STATUS FAIL"]))
    assert res.outcome == "error" and "not provided by backend" in res.error


def test_reference_state_fixes_the_tool_orientation(tmp_path):
    """Review control-safety #8 / caching #13: the envelope and the manual use the
    backend's nominal reset state when it has one."""
    from controlr.loop import run_episode

    seen = []

    class RefMock(RecMock):
        def reference_state(self):
            seen.append(1)
            return self.state()

    run_episode(_cfg(tmp_path), robot=RefMock(), llm=FakeLLM(["STATUS FAIL"]))
    assert seen


def test_truncated_note_is_stripped_from_the_stored_reply(tmp_path):
    from controlr.llm.client import LLMResult, Timings
    from controlr.loop import run_episode

    class Cut(FakeLLM):
        def complete(self, *a, **kw):
            r = super().complete(*a, **kw)
            return LLMResult("MOVE ee_delta 0 0 -5\nSTATUS OK moving abo", r.usage, r.timings, True, None, None,
                             200, 1, truncated=True)

    res = run_episode(_cfg(tmp_path, **{"episode.max_turns": 1}), robot=RecMock(), llm=Cut(["x"]))
    rec = res.records[0]
    assert rec["reply"] == "MOVE ee_delta 0 0 -5\nSTATUS OK" and rec["reply_streamed"].endswith("abo")


def test_setup_records_the_sampled_scene(tmp_path):
    """Rotation round: setup.json carries the task scene the backend sampled (here the mock's
    reach target and start pose; Isaac: packet pose + yaw offset, box, start yaw, marker)."""
    from controlr.loop import run_episode

    res = run_episode(_cfg(tmp_path), robot=RecMock(), llm=FakeLLM(["STATUS FAIL"]))
    scene = json.loads((Path(res.run_dir) / "setup.json").read_text())["scene"]
    assert scene["task"] == "reach"
    assert scene["target_mm"] == pytest.approx([v * 1000 for v in NEAR])
    assert scene["start"]["tcp_mm"] and scene["start"]["tcp_yaw_deg"] is not None


class BoxMock(RecMock):
    """RecMock with a known obstacle (a box that moves after the first execute) and backend
    diagnostics in its ExecReport — the Isaac contract."""

    def __init__(self, params=None):
        super().__init__(params)
        self.n_exec = 0

    def obstacles(self):
        from controlr.robot.obstacles import box_from_bin_info
        dx = 0.02 * self.n_exec
        return [box_from_bin_info({"lower": [-0.5705 + dx, 0.0, -0.0055], "upper": [-0.2105 + dx, 0.26, 0.1805],
                                   "center": [-0.3905 + dx, 0.13, -0.0095], "yaw": 0.0, "wall": 0.02,
                                   "dynamic": True})]

    def execute(self, actions):
        rep = super().execute(actions)
        self.n_exec += 1
        rep.backend = {"profile": {"physics_s": 0.5, "contacts_s": 0.1}, "wall_s": 0.7, "sim_s": 0.4,
                       "contacts_peak_n": {"gripper-object": 14.0}}
        return rep


def test_obstacles_reach_manual_envelope_and_log_and_backend_profile_is_logged(tmp_path):
    """contacts-and-speed: the backend's obstacles (current pose) go to the manual (setup.json
    records them) and to the envelope every turn; ExecReport.backend lands in turns.jsonl."""
    from controlr.loop import run_episode

    robot = BoxMock()
    res = run_episode(_cfg(tmp_path, **{"episode.max_turns": 2}), robot=robot,
                      llm=FakeLLM(["MOVE ee_delta 0 0 -10\nSTATUS OK", "MOVE ee_delta 0 0 -10\nSTATUS OK"]))
    run = Path(res.run_dir)
    setup = json.loads((run / "setup.json").read_text())
    assert setup["obstacles"][0]["name"] == "the blue box" and setup["obstacles"][0]["movable"]
    assert "slides when pushed" in (run / "system_prompt.md").read_text()
    turns = read_jsonl(run / "turns.jsonl")
    assert turns[0]["backend"]["profile"]["physics_s"] == 0.5
    assert turns[0]["backend"]["contacts_peak_n"]["gripper-object"] == 14.0
    summary = json.loads((run / "summary.json").read_text())
    assert summary["sim_time"]["sim_s"] == pytest.approx(0.8) and summary["sim_time"]["server_wall_s"] == pytest.approx(1.4)
    # default feedback: STATE only (no TURN / EXEC / holding)
    assert turns[0]["feedback"].startswith("STATE: tcp ") and "holding" not in turns[0]["feedback"]
    assert "TURN" not in turns[0]["feedback"] and "EXEC" not in turns[0]["feedback"]


def test_envelope_sees_the_box_where_it_is_now(tmp_path, monkeypatch):
    from controlr.loop import run_episode
    from controlr.robot import safety as safety_mod

    centres = []
    orig = safety_mod.SafetyEnvelope.set_obstacles

    def spy(self, obstacles):
        centres.append(obstacles[0].center[0])
        return orig(self, obstacles)
    monkeypatch.setattr(safety_mod.SafetyEnvelope, "set_obstacles", spy)
    run_episode(_cfg(tmp_path, **{"episode.max_turns": 3}), robot=BoxMock(),
                llm=FakeLLM(["MOVE ee_delta 0 0 -5\nSTATUS OK"] * 3))
    assert centres == pytest.approx([-0.3905, -0.3705, -0.3505])


# ---------------------------------------------------------------------------
# llm.overlap_tail: execute on the STATUS word, finish the note + usage meanwhile
# ---------------------------------------------------------------------------

class _TailLLM(FakeLLM):
    """Streams the reply; after stop_when fires it does not return until ``release`` is
    set — like a router still sending the note and the usage chunk."""

    def __init__(self, replies, release, **kw):
        super().__init__(replies, **kw)
        self.release = release
        self.waited: list[bool] = []

    def complete(self, *a, stop_when=None, **kw):
        fired = []

        def sw(text):
            done = stop_when(text) if stop_when else False
            if done:
                fired.append(text)
            return done

        r = super().complete(*a, stop_when=sw, **kw)
        if fired:
            self.waited.append(self.release.wait(5.0))
            self.release.clear()
        return r


class _SignalRobot(RecMock):
    def __init__(self, release):
        super().__init__()
        self.release = release

    def execute(self, actions):
        self.release.set()          # the call may only return after the robot started moving
        return super().execute(actions)


def test_overlap_executes_before_the_call_returns_and_stores_the_full_reply(tmp_path):
    import threading

    from controlr.loop import run_episode

    release = threading.Event()
    replies = ["MOVE ee_delta 0 0 -25\nSTATUS OK lowering to the target", "MOVE ee_delta 0 0 -25\nSTATUS OK again",
               "STATUS DONE reached"]
    llm = _TailLLM(replies, release)
    robot = _SignalRobot(release)
    # max_turns=3: the DONE turn has no action, so nothing releases it -> its call waits 5 s;
    # keep it out by ending on the second move
    res = run_episode(_cfg(tmp_path, **{"episode.max_turns": 2}), robot=robot, llm=llm)
    assert llm.waited == [True, True]                     # execution started while the call was open
    recs = res.records
    assert recs[0]["reply"] == replies[0]                 # the note streamed after the STATUS word is kept
    assert recs[0]["timings"]["exec_overlapped"] is True and "llm_tail" in recs[0]["timings"]
    assert recs[0]["usage"] is not None and "overlap_mismatch" not in recs[0]
    assert [a["raw"] for a in recs[0]["executed"]] == ["MOVE ee_delta 0 0 -25"]
    msgs = read_jsonl(Path(res.run_dir) / "messages.jsonl")
    assert msgs[2]["content"] == replies[0]


def test_overlap_and_classic_paths_give_the_same_transcript(tmp_path):
    from controlr.loop import run_episode

    replies = ["MOVE ee_delta 0 0 -25\nSTATUS OK one", "MOVE ee_delta 0 0 -25\nSTATUS OK two", "STATUS DONE reached"]
    runs = {}
    for flag in ("true", "false"):
        res = run_episode(_cfg(tmp_path / flag, **{"llm.overlap_tail": flag, "llm.request_nonce": "false"}),
                          robot=RecMock(), llm=FakeLLM(list(replies)))
        assert res.outcome == "success"
        runs[flag] = (Path(res.run_dir) / "messages.jsonl").read_text(), [r["executed"] for r in res.records[:2]]
        assert all(("exec_overlapped" in r["timings"]) == (flag == "true") for r in res.records[:2])
    assert runs["true"] == runs["false"]


def test_overlap_falls_back_when_the_reply_never_completes(tmp_path):
    from controlr.loop import run_episode

    # no STATUS -> stop_when never fires -> classic path: parse error, nothing executed early
    res = run_episode(_cfg(tmp_path, **{"episode.max_turns": 1}), robot=RecMock(),
                      llm=FakeLLM(["MOVE ee_delta 0 0 -10"]))
    rec = res.records[0]
    assert "exec_overlapped" not in rec["timings"] and rec["parse_errors"]
    res = run_episode(_cfg(tmp_path, **{"episode.max_turns": 2}), robot=RecMock(),
                      llm=FakeLLM(["MOVE ee_delta 0 0 -10\nSTATUS OK", RuntimeError("503 retries exhausted")]))
    assert res.outcome == "llm_error" and res.turns == 2


def test_overlap_reraises_client_exceptions_in_the_loop(tmp_path):
    from controlr.loop import run_episode

    class Boom(FakeLLM):
        def complete(self, *a, **kw):
            raise ValueError("client bug")

    res = run_episode(_cfg(tmp_path), robot=RecMock(), llm=Boom(["x"]))
    assert res.outcome == "error" and "client bug" in res.error
