"""Decision head (llm.backend=decisions): questions, state, answer reduction, the HTTP client
(httpx.MockTransport) and a whole episode on the mock robot. No network."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from controlr.config import load_config
from controlr.llm.decisions import (DecisionsClient, build_questions, build_state, reduce_score,
                                    render_reply)
from controlr.llm.fake import FakeLLM
from controlr.protocol.grammar import parse_reply
from controlr.robot.mock import MockRobot
from controlr.robot.spec import ur3_cb3_spec
from controlr.runlog import read_jsonl

LEVELS = [-30.0, -10.0, -3.0, 0.0, 3.0, 10.0, 30.0]
ZERO = {str(i): (1.0 if i == 3 else 0.0) for i in range(7)}     # all mass on level 0


def _cfg(tmp_path: Path | None = None, *sets: str):
    base = ["llm.backend=decisions", "llm.model=openai/gpt-6-luna-decisions", "planner.enabled=false",
            "robot.backend=mock", "task.name=reach", "llm.base_url=http://fake", "llm.overlap_tail=false",
            "observation.renderers=[raw]", "episode.max_turns=6"]
    if tmp_path is not None:
        base.append(f"log.root={tmp_path}")
    return load_config(None, base + list(sets), dotenv=None)


def _score(probs: dict) -> dict:
    return {"type": "score", "probabilities": probs, "score": 3.0}


def _answers(dx=None, dy=None, dz=None, grip="keep", status="CONTINUE", dyaw=None) -> dict:
    a = {k: _score(v or ZERO) for k, v in (("dx", dx), ("dy", dy), ("dz", dz))}
    if dyaw is not None:
        a["dyaw"] = _score(dyaw)
    a["grip"] = {"type": "choice", "choice": grip, "probabilities": {grip: 0.9}}
    a["status"] = {"type": "choice", "choice": status, "probabilities": {status: 0.9}}
    return a


def _messages(n_turns: int) -> list[dict]:
    img = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}}
    m = [{"role": "system", "content": [{"type": "text", "text": "MANUAL"}]},
         {"role": "user", "content": [{"type": "text", "text": "TASK: reach"}, img]}]
    for i in range(n_turns):
        m.append({"role": "assistant", "content": f"MOVE ee_delta {i} 0 0\nSTATUS OK"})
        m.append({"role": "user", "content": [{"type": "text", "text": f"STATE: {i}"},
                                              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,I{i}"}}]})
    return m


# ---------------------------------------------------------------------------
# questions and state
# ---------------------------------------------------------------------------

def test_questions_per_axis_and_yaw_only_with_rotation():
    from controlr.llm.decisions import load_questions

    tpl = load_questions("decisions_v0")
    cfg = _cfg()
    q = build_questions(cfg.decisions, cfg.action, tpl)
    assert list(q) == ["dx", "dy", "dz", "grip", "status"]
    assert q["dx"]["type"] == "score" and len(q["dx"]["criteria"]) == len(LEVELS)
    assert q["dx"]["criteria"][0] == "move the tool -30 mm along x"
    assert q["dz"]["criteria"][3] == "do not move along z in this step"
    assert set(q["status"]["criteria"]) == {"CONTINUE", "DONE", "FAIL"}
    cfg = _cfg(None, "action.rotation=yaw")
    q = build_questions(cfg.decisions, cfg.action, tpl)
    assert list(q) == ["dx", "dy", "dz", "dyaw", "grip", "status"]
    assert q["dyaw"]["criteria"][-1] == "turn the tool +20 deg about base z"


def test_state_carries_task_window_and_only_the_newest_frame():
    cfg = _cfg(None, "decisions.history=2")
    state = build_state(_messages(4), cfg.decisions)
    obj = json.loads(state[0]["text"])
    assert obj["manual"] == "MANUAL" and obj["task"] == "TASK: reach" and obj["turn"] == 4
    assert [s["turn"] for s in obj["recent_steps"]] == [2, 3]
    assert obj["recent_steps"][-1] == {"turn": 3, "action": "MOVE ee_delta 3 0 0\nSTATUS OK",
                                       "feedback": "STATE: 3"}
    assert [p["image_url"]["url"] for p in state[1:]] == ["data:image/jpeg;base64,I3"]
    # turn 0: no steps, the first frame
    s0 = build_state(_messages(0), cfg.decisions)
    assert json.loads(s0[0]["text"])["recent_steps"] == [] and len(s0) == 2


def test_state_field_mode_and_no_manual():
    cfg = _cfg(None, "decisions.image_mode=field", "decisions.include_manual=false", "decisions.history=0")
    state = build_state(_messages(2), cfg.decisions)
    assert "manual" not in state and state["recent_steps"] == []
    assert state["images"] == ["data:image/jpeg;base64,I1"]


def test_state_is_deterministic():
    cfg = _cfg()
    assert json.dumps(build_state(_messages(3), cfg.decisions)) == json.dumps(build_state(_messages(3), cfg.decisions))


# ---------------------------------------------------------------------------
# answers -> reply
# ---------------------------------------------------------------------------

def test_reduce_expected_argmax_deadband_and_fallback():
    p = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.5, "4": 0.0, "5": 0.5, "6": 0.0}
    assert reduce_score({"probabilities": p}, LEVELS, "expected", 1.0) == 5.0
    assert reduce_score({"probabilities": {"5": 0.6, "3": 0.4}}, LEVELS, "argmax", 1.0) == 10.0
    assert reduce_score({"probabilities": {"3": 0.8, "4": 0.2}}, LEVELS, "expected", 1.0) == 0.0  # 0.6 < deadband
    neg = reduce_score({"probabilities": {"2": 0.1, "3": 0.9}}, LEVELS, "expected", 0.0)
    assert neg == -0.3
    assert str(reduce_score({"probabilities": {"3": 1.0}}, LEVELS, "expected", 0.0)) == "0.0"   # no -0.0
    # no probabilities: interpolate the expected index between levels
    assert reduce_score({"score": 4.5}, LEVELS, "expected", 0.0) == 6.5
    assert reduce_score({"score": 4.6}, LEVELS, "argmax", 0.0) == 10.0
    assert reduce_score({}, LEVELS, "expected", 0.0) == 0.0          # a missing answer never moves


def test_render_move_parses_to_si():
    cfg = _cfg()
    text, rec = render_reply(_answers(dx={"5": 1.0}, dz={"1": 0.5, "2": 0.5}), cfg.decisions, cfg.action)
    assert text == "MOVE ee_delta 10.0 0.0 -6.5\nSTATUS OK"
    parsed = parse_reply(text, cfg.action, ur3_cb3_spec())
    assert not parsed.errors and parsed.status.value == "OK"
    assert parsed.actions[0].values == pytest.approx((0.010, 0.0, -0.0065))
    assert rec["steps"] == {"dx": 10.0, "dy": 0.0, "dz": -6.5} and rec["status"] == "OK"


def test_render_yaw_grip_hold_and_terminal():
    cfg = _cfg(None, "action.rotation=yaw")
    text, _ = render_reply(_answers(dyaw={"3": 1.0}, dx={"4": 1.0}), cfg.decisions, cfg.action)
    assert text == "MOVE ee_delta 3.0 0.0 0.0 5.0\nSTATUS OK"
    assert not parse_reply(text, cfg.action, ur3_cb3_spec()).errors
    text, _ = render_reply(_answers(dx={"6": 1.0}, grip="close", dyaw={"2": 1.0}), cfg.decisions, cfg.action)
    assert text == "GRIP close\nSTATUS OK"                      # gripper changes happen in place
    text, _ = render_reply(_answers(dyaw={"2": 1.0}), cfg.decisions, cfg.action)
    assert text == "HOLD\nSTATUS OK"
    text, _ = render_reply(_answers(dx={"6": 1.0}, status="DONE", dyaw={"2": 1.0}), cfg.decisions, cfg.action)
    assert text == "STATUS DONE"                                # no motion on a terminal turn
    for t in ("GRIP close\nSTATUS OK", "HOLD\nSTATUS OK", "STATUS DONE"):
        assert not parse_reply(t, cfg.action, ur3_cb3_spec()).errors


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

def _response(answers: dict, status: int = 200, **headers) -> httpx.Response:
    return httpx.Response(status, json={"id": "gen-dec-1", "model": "openai/gpt-6-luna-decisions-20261006",
                                        "provider": "OpenAI", "answers": answers,
                                        "usage": {"input_tokens": 1200, "output_tokens": 0, "cost": 0.00012}},
                          headers=headers)


def _client(handler, cfg=None, sleeps=None, **kw) -> DecisionsClient:
    sl = sleeps if sleeps is not None else []
    return DecisionsClient("https://example.test/api/alpha/", "sk-or-secret", cfg or _cfg(), 5, 3,
                           transport=httpx.MockTransport(handler), sleep=sl.append, **kw)


def test_client_request_and_result():
    seen = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return _response(_answers(dy={"0": 1.0}))
    c = _client(h)
    res = c.complete("openai/gpt-6-luna-decisions", _messages(1), max_tokens=2000,
                     extra_body={"provider": {"allow_fallbacks": False}})
    req = seen[0]
    assert str(req.url) == "https://example.test/api/alpha/decisions"
    assert req.headers["authorization"] == "Bearer sk-or-secret"
    body = json.loads(req.content)
    assert body["model"] == "openai/gpt-6-luna-decisions" and body["provider"] == {"allow_fallbacks": False}
    assert set(body["questions"]) == {"dx", "dy", "dz", "grip", "status"} and "max_tokens" not in body
    assert res.error is None and res.text == "MOVE ee_delta 0.0 -30.0 0.0\nSTATUS OK"
    assert res.usage.prompt_tokens == 1200 and res.usage.raw["cost"] == 0.00012
    assert res.decisions["id"] == "gen-dec-1" and res.decisions["steps"]["dy"] == -30.0
    assert res.request_bytes == len(req.content) and res.timings.t_complete is not None
    assert "sk-or-secret" not in repr(c)


def test_client_retries_429_then_succeeds_and_does_not_retry_400():
    calls, sleeps = [], []

    def h(req):
        calls.append(1)
        return _response({}, 429, **{"retry-after": "2"}) if len(calls) == 1 else _response(_answers())
    res = _client(h, sleeps=sleeps).complete("m", _messages(0))
    assert res.error is None and res.attempts == 2 and sleeps == [2.0] and res.text == "HOLD\nSTATUS OK"

    calls.clear()

    def bad(req):
        calls.append(1)
        return httpx.Response(400, json={"error": {"code": 400, "message": "Invalid request parameters"}})
    res = _client(bad).complete("m", _messages(0))
    assert len(calls) == 1 and res.http_status == 400 and "Invalid request" in res.error


def test_client_gives_up_and_reports_malformed_bodies():
    res = _client(lambda r: httpx.Response(503, text="down")).complete("m", _messages(0))
    assert res.error and "gave up after 4 attempts" in res.error and res.attempts == 4
    res = _client(lambda r: httpx.Response(200, json={"nope": 1})).complete("m", _messages(0))
    assert res.error and "bad decisions response" in res.error


# ---------------------------------------------------------------------------
# a whole episode
# ---------------------------------------------------------------------------

def test_episode_on_the_mock_robot_with_a_separate_planner(tmp_path):
    from controlr.loop import run_episode

    script = [_answers(dz={"1": 1.0}), _answers(dz={"1": 0.5, "2": 0.5}), _answers(status="DONE")]
    seen = []

    def h(req):
        seen.append(json.loads(req.content))
        return _response(script[min(len(seen), len(script)) - 1])
    cfg = _cfg(tmp_path, "planner.enabled=true", "task.params={target: [-0.300, -0.150, 0.124]}",
               "episode.trust_done=true")
    planner = FakeLLM(["1. go down 26 mm\n2. stop"])
    robot = MockRobot({"width": 160, "height": 120, "settle_s": 0.0})
    res = run_episode(cfg, robot=robot, llm=_client(h, cfg), planner_llm=planner)
    assert res.outcome == "success", res.error
    assert res.turns == 3 and len(planner.calls) == 1 and len(seen) == 3
    assert "go down 26 mm" in json.loads(seen[0]["state"][0]["text"])["task"]   # the plan reaches the head
    turns = read_jsonl(Path(res.run_dir) / "turns.jsonl")
    assert turns[0]["reply"] == "MOVE ee_delta 0.0 0.0 -10.0\nSTATUS OK"
    assert turns[0]["executed"][0]["values_si"][2] == pytest.approx(-0.010)
    assert turns[1]["decisions"]["steps"]["dz"] == -6.5
    assert turns[0]["usage"]["prompt_tokens"] == 1200
    # the feedback after turn 0 is in turn 1's state
    assert json.loads(seen[1]["state"][0]["text"])["recent_steps"][0]["feedback"].startswith("STATE")


def test_decisions_backend_without_key_fails_fast(tmp_path, monkeypatch):
    from controlr.loop import run_episode

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    res = run_episode(_cfg(tmp_path))
    assert res.outcome == "error" and "OPENROUTER_API_KEY" in res.error


def test_a_refused_question_is_dropped_not_retried():
    bodies = []

    def h(req):
        b = json.loads(req.content)
        bodies.append(b)
        if "dx" in b["questions"]:
            return httpx.Response(502, json={"error": {"code": 502,
                                                       "message": 'OpenAI refused to answer question "dx"'}})
        return _response(_answers(dy={"6": 1.0}))
    res = _client(h).complete("m", _messages(0))
    assert len(bodies) == 2 and "dx" not in bodies[1]["questions"]            # no backoff retries
    assert res.error is None and res.text == "MOVE ee_delta 0.0 30.0 0.0\nSTATUS OK"
    assert res.decisions["refused"] == ["dx"]


def test_split_head_defers_to_the_wide_view_when_the_crop_says_level():
    from controlr.llm.decisions import reduce_split

    mags = [3.0, 10.0, 30.0]
    far = {"probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}}
    crop_level = {"probabilities": {"neg": 0.1, "zero": 0.8, "pos": 0.1}}
    wide_pos = {"probabilities": {"neg": 0.0, "zero": 0.0, "pos": 1.0}}
    assert reduce_split(crop_level, far, mags, "expected", 0.0) == 0.0
    assert reduce_split(crop_level, far, mags, "expected", 0.0, wide_pos) == 8.0     # (0 + 0.8 * 1) * 10
    assert reduce_split({"probabilities": {"neg": 1.0, "zero": 0.0, "pos": 0.0}}, far, mags, "expected", 0.0,
                        wide_pos) == -10.0                                          # a sure crop wins
    assert reduce_split({"choice": "zero"}, far, mags, "argmax", 0.0, {"choice": "pos"}) == 10.0


# ---------------------------------------------------------------------------
# split head: image-relation questions, views, labels, fovea, target
# ---------------------------------------------------------------------------

def _mock3():
    from controlr.robot.mock import d435_camera, side_camera, top_camera

    return {"scene": d435_camera(640, 480, "scene"), "top": top_camera(640, 480), "side": side_camera(640, 480)}


def test_axis_views_pick_an_unambiguous_image_per_axis():
    import numpy as np

    from controlr.llm.decisions import axis_views

    o = np.array([-0.3, -0.1, 0.1])
    v = axis_views(_mock3(), ["scene", "top", "side"], o)
    assert v["x"]["image"] == "scene" and v["x"]["pos"] == "to the right of"
    assert v["y"]["image"] == "side" and v["y"]["pos"] == "to the left of"
    assert v["z"]["image"] == "side" and v["z"]["pos"] == "nearer the top edge of the image than"
    # the angled D435 alone: +y and +z both point up -> neither reads on its own
    v1 = axis_views({"scene": _mock3()["scene"]}, ["scene"], o)
    assert v1["x"] is not None and v1["y"] is None and v1["z"] is None


def test_split_questions_fill_target_views_and_fovea():
    from controlr.llm.decisions import load_questions

    cfg = _cfg(None, "decisions.head=split", "decisions.questions=decisions_v5", "decisions.target=the red ball",
               "robot.params={cameras: [top, side]}", "observation.cameras=[scene,top,side]",
               "observation.renderers=[raw,fovea]")
    c = DecisionsClient("http://x", "k", cfg, 5, 0)
    import numpy as np
    c.set_scene(_mock3(), np.array([-0.3, -0.1, 0.1]), task="reach it")
    q = build_questions(cfg.decisions, cfg.action, load_questions("decisions_v5"), c.hints, c.views, c.task)
    assert list(q) == ["dir_x", "wide_x", "far_x", "dir_y", "wide_y", "far_y", "dir_z", "wide_z", "far_z",
                       "grip", "status"]
    assert "`scene fovea`" in q["dir_x"]["instructions"] and "the red ball" in q["dir_x"]["instructions"]
    assert "magenta cross" in q["dir_x"]["criteria"]["pos"]
    assert "`scene` image only" in q["wide_x"]["instructions"] and "fingertips" in q["wide_x"]["criteria"]["pos"]
    assert "the red ball" in q["far_y"]["instructions"]
    assert "3 px" not in c.hints["z"]                     # perspective drift is not offered as a cue
    body = c.request_body("m", _messages(0))      # 1 image but 6 configured: unlabeled, not mislabeled
    assert [p["text"] for p in body["state"] if p["type"] == "text"][1:] == []


def test_target_task_placeholder_and_image_labels():
    from controlr.llm.decisions import load_questions
    from controlr.observation.renderers import image_labels

    cfg = _cfg(None, "decisions.head=split", "decisions.questions=decisions_v5", "decisions.target=the spot named in {task}")
    q = build_questions(cfg.decisions, cfg.action, load_questions("decisions_v5"), {}, {}, " grab the cup ")
    assert "the spot named in grab the cup" in q["dir_z"]["instructions"]
    o = _cfg(None, "observation.cameras=[scene,top]", "observation.renderers=[raw,fovea]").observation
    assert image_labels(o) == ["scene", "scene fovea", "top", "top fovea"]
    assert image_labels(_cfg(None, "observation.renderers=[raw,diff]").observation) is None


def test_state_labels_each_image_when_counts_match():
    cfg = _cfg()
    m = _messages(0)
    m[-1]["content"].append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBB"}})
    st = build_state(m, cfg.decisions, ["scene", "top"])
    assert [p.get("text") for p in st[1:]] == ["image `scene`:", None, "image `top`:", None]
    st = build_state(m, cfg.decisions, ["scene"])                 # mismatch: no labels
    assert [p["type"] for p in st[1:]] == ["image_url", "image_url"]


# ---------------------------------------------------------------------------
# state_layout=transcript: the whole episode, append-only
# ---------------------------------------------------------------------------

def test_transcript_state_keeps_every_step_and_is_prefix_stable():
    cfg = _cfg(None, "decisions.state_layout=transcript")
    s3, s4 = build_state(_messages(3), cfg.decisions), build_state(_messages(4), cfg.decisions)
    texts = [p["text"] for p in s4 if p["type"] == "text"]
    assert texts[0] == "OPERATING MANUAL\nMANUAL" and texts[1].startswith("EPISODE START (turn 0)\nTASK: reach")
    assert [t.split("\n")[0] for t in texts[2:6]] == ["TURN 0 ACTION", "TURN 1 ACTION", "TURN 2 ACTION", "TURN 3 ACTION"]
    assert "TURN 4 FEEDBACK\nSTATE: 3" in texts[5] and texts[6].startswith("NOW: turn 4")
    assert [p["image_url"]["url"] for p in s4 if p["type"] == "image_url"] == ["data:image/jpeg;base64,I3"]
    # everything before the moving tail (NOW line + newest images) is a byte-identical prefix
    tail = 2                                                  # NOW + 1 image
    assert json.dumps(s4[:len(s3) - tail]) == json.dumps(s3[:len(s3) - tail])


def test_transcript_state_keeps_past_images_every_k_turns():
    cfg = _cfg(None, "decisions.state_layout=transcript", "decisions.history_image_every=2")
    s = build_state(_messages(4), cfg.decisions, ["scene"])
    urls = [p["image_url"]["url"] for p in s if p["type"] == "image_url"]
    assert urls == ["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,I1", "data:image/jpeg;base64,I3"]
    labels = [p["text"] for p in s if p["type"] == "text" and "image `" in p["text"]]
    assert labels == ["turn 0 (past) image `scene`:", "turn 2 (past) image `scene`:", "image `scene`:"]
    with pytest.raises(ValueError):
        _cfg(None, "decisions.state_layout=transcript", "decisions.image_mode=field")
