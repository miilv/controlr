"""FK agent: dataset builder, prompt assembly, constraints, the loop on a fake LLM + fake server,
and the offline evaluation plumbing. No network, no hardware."""

from __future__ import annotations

import json
import threading
from multiprocessing.connection import Listener

import numpy as np
import pytest
from PIL import Image

from controlr.fkagent import dataset as D
from controlr.fkagent import geometry as G
from controlr.fkagent import offline as O
from controlr.fkagent import prompt as Pm
from controlr.fkagent.agent import AgentConfig, FkAgent, ServerArm
from controlr.llm import FakeLLM
from controlr.robot.so101 import kinematics as K
from controlr.robot.so101 import server as S

ARM = list(K.ARM_JOINTS)
LO, HI = 1000, 3000          # one calibrated range for every joint in the synthetic recordings
START = np.array([0.0, -100.0, 95.0, 80.0, -90.0])       # measured deg (folded, as in the demos)
GRASP = np.array([2.0, -11.0, 59.0, 38.0, -94.0])        # measured deg (ball_demo_4's grasp)


def _norm(deg):
    raw = np.asarray(deg, float) * 4095 / 360 + (LO + HI) / 2
    return (raw - LO) / (HI - LO) * 200 - 100


def _row(t, q_deg, gF, gL, frame=None):
    f = dict(zip(ARM, _norm(q_deg).round(4).tolist()), gripper=gF)
    lead = dict(zip(ARM, _norm(q_deg).round(4).tolist()), gripper=gL)
    r = {"t": round(t, 4), "leader": lead, "follower": f}
    if frame:
        r["frame"] = frame
    return r


def make_recording(d, *, miss: bool = False, extra_take: bool = False):
    """Start pose 0.5 s -> move to the grasp pose (2 s) -> [an empty close and reopen] -> close on an
    object (F stalls at 25 while L goes to 15) -> lift 60 mm-ish -> back."""
    (d / "frames").mkdir(parents=True)
    (d / "calib_so101_follower.json").write_text(json.dumps(
        {j: {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": LO, "range_max": HI}
         for i, j in enumerate(ARM + ["gripper"], 1)}))
    lifted = GRASP + np.array([0.0, -15.0, 0.0, 0.0, 0.0])
    rows, t, dt = [], 0.0, 1 / 60
    k = 0

    def add(q, gF, gL):
        nonlocal t, k
        frame = None
        if k % 6 == 0:
            frame = f"{t:.3f}.jpg"
            shade = int(np.clip(60 + 40 * t, 0, 255))
            img = np.full((72, 128, 3), shade, np.uint8)
            img[30:70, 30:90] = 200 if gF > 20 and gL < gF - 1.5 else shade      # "object" between the jaws
            Image.fromarray(img).save(d / "frames" / frame)
        rows.append(_row(t, q, gF, gL, frame))
        t += dt
        k += 1

    for _ in range(30):
        add(START, 2.0, 1.0)
    for s in np.linspace(0, 1, 120):
        add(START + s * (GRASP - START), 2.0 + 33 * s, 1.0 + 34 * s)
    if miss:
        for g in np.linspace(35, 2, 30):
            add(GRASP, g, g - 0.3)
        for _ in range(20):
            add(GRASP, 2.0, 1.7)
        for g in np.linspace(2, 35, 20):
            add(GRASP, g, g)
    for g in np.linspace(35, 15, 30):          # close: the follower stalls at 25
        add(GRASP, max(25.0, g), g)
    for _ in range(30):
        add(GRASP, 25.0, 15.0)
    for s in np.linspace(0, 1, 60):
        add(GRASP + s * (lifted - GRASP), 25.0, 15.0)
    for g in np.linspace(15, 40, 20):          # release
        add(lifted, g, g)
    for s in np.linspace(0, 1, 60):
        add(lifted + s * (START - lifted), 40.0, 40.0)
    lines = [json.dumps({"t": 0.0, "event": "start"})]
    if extra_take:                              # a short aborted take before the real one
        lines += [json.dumps(_row(0.0, START, 2, 1)), json.dumps({"t": 0.1, "event": "stop"}),
                  json.dumps({"t": 0.0, "event": "start"})]
    lines += [json.dumps(r) for r in rows] + [json.dumps({"t": t, "event": "stop"})]
    (d / "joints.jsonl").write_text("\n".join(lines) + "\n")
    return d


# ----------------------------------------------------------------------------- dataset

def test_load_converts_to_degrees_and_keeps_the_longest_take(tmp_path):
    rec = D.load(make_recording(tmp_path / "r", extra_take=True))
    assert len(rec.t) > 300 and np.all(np.diff(rec.t) > 0)
    np.testing.assert_allclose(rec.q_meas[0], START, atol=0.05)
    np.testing.assert_allclose(rec.q_model[0], START + G.JOINT_OFFSETS_DEG, atol=0.05)
    assert len(rec.frame_t) > 40


def test_grasp_detection_keypoints_and_fk(tmp_path):
    rec = D.load(make_recording(tmp_path / "r"))
    holds = D.detect_holds(rec)
    assert len(holds) == 1 and holds[0].stall > 5 and not holds[0].weak
    ss = D.build_sessions(rec, labels={"r": "ball"})
    assert len(ss) == 1
    s = ss[0]
    assert s["object"] == "ball" and not s["regrip"] and s["misses"] == []
    names = [k["name"] for k in s["keypoints"]]
    assert names[0] == "start" and "pre_grasp" in names and "grasp" in names and names[-1] == "lift"
    g = s["grasp"]
    # the grasp pose's TCP is FK of measured + offsets
    np.testing.assert_allclose(g["tcp_mm"], G.tcp_mm(GRASP + G.JOINT_OFFSETS_DEG), atol=1.0)
    assert g["gripper_stall"] == pytest.approx(25.0, abs=1.5) and g["gripper_open"] >= 34
    assert s["keypoints"][0]["t"] == 0.0 and s["keypoints"][0]["motion_deg"] is None
    kp = {k["name"]: k for k in s["keypoints"]}
    np.testing.assert_allclose(np.add(kp["pre_grasp"]["q_deg"], kp["grasp"]["motion_deg"]), kp["grasp"]["q_deg"],
                               atol=0.02)
    pre = np.linalg.norm(np.subtract(kp["pre_grasp"]["tcp_mm"], g["tcp_mm"]))
    assert pre >= 60
    assert g["frame_check"]["ok"] is True       # the "object" moves with the camera


def test_empty_close_is_a_miss_not_a_grasp(tmp_path):
    rec = D.load(make_recording(tmp_path / "r", miss=True))
    ss = D.build_sessions(rec)
    assert len(ss) == 1 and len(ss[0]["misses"]) == 1
    assert "miss" in [k["name"] for k in ss[0]["keypoints"]]


def test_export_writes_json_images_and_sheets(tmp_path):
    rec = D.load(make_recording(tmp_path / "r"))
    ss = D.build_sessions(rec)
    p = D.export(ss, {rec.name: rec}, tmp_path / "out", long_edge=64)
    sessions, root = D.load_sessions(tmp_path / "out")
    assert p.exists() and len(sessions) == 1
    for k in sessions[0]["keypoints"]:
        assert k["photo"] and (root / k["photo"]).exists()
        assert max(Image.open(root / k["photo"]).size) == 64
    assert list((tmp_path / "out" / "sheets").glob("*.jpg"))
    assert "r#1" in "\n".join(D.summary_lines(sessions))


# ----------------------------------------------------------------------------- constraints

def test_survey_start_pose_is_inside_the_limits():
    lim = G.Limits()
    assert G.check_path([G.SURVEY_Q_DEG], lim) == ""
    s = G.pose_summary(G.SURVEY_Q_DEG)
    assert 140 <= s["tcp_mm"][0] <= 270 and s["tcp_mm"][2] > 100


def test_plan_ok_comes_down_from_above():
    lim = G.Limits()
    target = GRASP + G.JOINT_OFFSETS_DEG             # TCP ~ (237, 2, 14)
    p = G.plan_target(G.SURVEY_Q_DEG, target, lim)
    assert p.ok and not p.clamped and p.scale == 1.0
    np.testing.assert_allclose(p.target_q_deg, target)
    above = G.tcp_mm(p.approach[1])
    np.testing.assert_allclose(above, G.tcp_mm(target) + [0, 0, lim.approach_dz_mm], atol=3.0)
    assert G.check_path(p.approach, lim) == ""


def test_plan_clamps_a_target_outside_the_box_and_below_the_table():
    lim = G.Limits()
    far = np.array([0.0, 40.0, -20.0, 70.0, -90.0])    # x ~ 330 mm, beyond the box
    assert G.tcp_mm(far)[0] > lim.box_hi_mm[0]
    p = G.plan_target(G.SURVEY_Q_DEG, far, lim)
    assert p.ok and p.clamped and 0.05 < p.scale < 1.0
    assert G.in_box(G.tcp_mm(p.target_q_deg), lim)
    assert any("workspace" in n for n in p.notes)
    low = GRASP + G.JOINT_OFFSETS_DEG + np.array([0, 8, 0, 0, 0])
    assert G.tcp_mm(low)[2] < lim.box_lo_mm[2]
    p2 = G.plan_target(G.SURVEY_Q_DEG, low, lim)
    assert p2.ok and p2.clamped and G.tcp_mm(p2.target_q_deg)[2] >= lim.box_lo_mm[2] - 1e-6


def test_plan_clips_joint_limits_and_limits_the_step():
    lim = G.Limits()
    q = GRASP + G.JOINT_OFFSETS_DEG
    p = G.plan_target(G.SURVEY_Q_DEG, q + [0, 0, 0, 0, -200], lim)
    assert p.ok and p.target_q_deg[4] == pytest.approx(lim.joint_lo_deg[4])
    assert any("clipped" in n for n in p.notes)
    nxt = q + np.array([-20.0, 0, 0, 0, 0])              # ~80+ mm sideways
    p = G.plan_target(G.SURVEY_Q_DEG, nxt, G.Limits(max_step_mm=30.0), prev_target_q=q)
    assert p.ok and p.clamped
    assert np.linalg.norm(G.tcp_mm(p.target_q_deg) - G.tcp_mm(q)) <= 30.0 + 1e-6
    rej = G.plan_target(G.SURVEY_Q_DEG, nxt, G.Limits(max_step_mm=0.5), prev_target_q=q)
    assert not rej.ok and "step" in rej.reason


def test_check_path_catches_a_dip_below_the_table():
    lim = G.Limits(approach_dz_mm=0)
    a = GRASP + G.JOINT_OFFSETS_DEG
    b = a + np.array([0, 12, 0, 0, 0])                    # ends below the table
    assert "workspace" in G.check_path([a, b], lim)
    assert G.check_path([G.SURVEY_Q_DEG, a], lim) == ""


def test_lift_path_goes_straight_up():
    lim = G.Limits()
    q = GRASP + G.JOINT_OFFSETS_DEG
    up = G.lift_path(q, 50.0, lim)
    assert len(up) > 5
    d = G.tcp_mm(up[-1]) - G.tcp_mm(q)
    assert d[2] == pytest.approx(50.0, abs=2.5) and abs(d[0]) < 3 and abs(d[1]) < 3


def test_jacobian_signs():
    J = G.jacobian_mm_per_deg(G.SURVEY_Q_DEG)
    assert J[1, 0] < -1.0          # +pan -> toward -y (the robot's right)
    assert J[2, 1] < 0             # +shoulder_lift -> down at the survey pose


# ----------------------------------------------------------------------------- prompt

def _dataset(tmp_path, n=2):
    recs, ss = {}, []
    for i in range(n):
        rec = D.load(make_recording(tmp_path / f"rec{i}"))
        recs[rec.name] = rec
        ss += D.build_sessions(rec, labels={rec.name: f"thing{i}"})
    D.export(ss, recs, tmp_path / "ds", long_edge=64)
    return D.load_sessions(tmp_path / "ds")


def test_system_text_states_the_limits():
    lim = G.Limits()
    s = Pm.system_text(lim, max_steps=5)
    assert "x 140..270" in s and "z 5..260" in s and "at most 5 steps" in s and "80 mm" in s
    assert "gripper_open_pct 10..35" in s
    assert "{" in s and "q_deg" in s


def test_fewshot_and_history_assembly(tmp_path):
    sessions, root = _dataset(tmp_path)
    parts = Pm.fewshot_parts(sessions, root, long_edge=64)
    imgs = [p for p in parts if not isinstance(p, str)]
    assert len(imgs) == sum(1 for s in sessions for k in s["keypoints"] if k.get("photo"))
    text = "\n".join(p for p in parts if isinstance(p, str))
    assert "EXAMPLE 1: pick up the thing0" in text and "EXAMPLE 2: pick up the thing1" in text
    assert "=> grasped the thing0 with q_deg" in text
    jpeg = (root / sessions[0]["keypoints"][0]["photo"]).read_bytes()
    steps = [Pm.Step(0, "observe", list(G.SURVEY_Q_DEG), 35.0, jpeg),
             Pm.Step(1, "grip", list(GRASP), 30.0, jpeg, requested_q_deg=list(GRASP + 1), executed_q_deg=list(GRASP),
                     motion_deg=[1, 2, 3, 4, 5], notes=["clamped"], result="the jaws closed EMPTY")]
    hp = Pm.history_parts("pick up the thing0", steps, long_edge=64, max_steps=4)
    msgs = Pm.build_messages("SYS", parts + hp)
    assert msgs[0] == {"role": "system", "content": "SYS"}
    content = msgs[1]["content"]
    assert Pm.count_images(msgs) == len(imgs) + 2
    # text parts are merged between images, never two text parts in a row
    kinds = [c["type"] for c in content]
    assert all(not (a == b == "text") for a, b in zip(kinds, kinds[1:]))
    full = "\n".join(c["text"] for c in content if c["type"] == "text")
    assert "TASK: pick up the thing0" in full and "Step 1 - grip" in full and "executed" in full
    assert "NOTE: clamped" in full and "closed EMPTY" in full and "Steps left: 3" in full
    assert "d TCP per +1 deg" in full
    # deterministic: same inputs -> byte-identical request (prompt cache)
    again = Pm.build_messages("SYS", Pm.fewshot_parts(sessions, root, long_edge=64) + hp)
    assert json.dumps(again) == json.dumps(msgs)


@pytest.mark.parametrize("text", [
    'thinking...\n```json\n{"object": {"xyz_mm": [230, 10, 20], "uncertainty_mm": 30, "visible": true},'
    ' "action": "grip", "q_deg": [1, -16, 49, 36, -94], "gripper_open_pct": 40,'
    ' "expected_tcp_mm": [237, 2, 14], "reason": "a {brace} in a string"}\n```',
    'An example {"x": 1} first. Final: {"action": "GRIP", "q_deg": [1, -16, 49, 36, -94, 0],'
    ' "object": {"xyz_mm": [230, 10, 20]}}',
])
def test_parse_reply_ok(text):
    d = Pm.parse_reply(text)
    assert d.action == "grip" and d.q_deg == [1, -16, 49, 36, -94] and d.object_xyz_mm == [230, 10, 20]


@pytest.mark.parametrize("text,err", [
    ("no json here", "no JSON"),
    ('{"action": "jump", "q_deg": [0, 0, 0, 0, 0]}', "unknown action"),
    ('{"action": "move"}', "q_deg is missing"),
    ('{"action": "move", "q_deg": [0, 0, 0]}', "5 numbers"),
])
def test_parse_reply_errors(text, err):
    with pytest.raises(Pm.ReplyError, match=err):
        Pm.parse_reply(text)


def test_grasp_q_is_parsed_and_defaults_to_q_on_grip():
    d = Pm.parse_reply('{"action": "move", "q_deg": [0, 0, 0, 0, 0], "grasp_q_deg": [1, 2, 3, 4, 5]}')
    assert d.grasp_q_deg == [1, 2, 3, 4, 5]
    assert Pm.parse_reply('{"action": "grip", "q_deg": [0, 1, 0, 0, 0]}').grasp_q_deg == [0, 1, 0, 0, 0]
    assert Pm.parse_reply('{"action": "move", "q_deg": [0, 0, 0, 0, 0], "grasp_q_deg": "x"}').grasp_q_deg is None


def test_give_up_needs_no_joints():
    assert Pm.parse_reply('{"action": "give_up"}').q_deg is None


# ----------------------------------------------------------------------------- loop

class StallArm(S.FakeArm):
    """Fake arm whose jaws stall at ``stall`` when closing (an object between them)."""

    def __init__(self, stall=None, **kw):
        super().__init__(**kw)
        self.stall = stall

    def _advance(self):
        super()._advance()
        if self.stall is not None and self.goal_g < self.stall:
            self.g = max(self.g, self.stall)


@pytest.fixture
def server_factory():
    threads = []

    def make(stall=None):
        arm = StallArm(stall=stall, speed_rad_s=30.0)
        srv = S.Server(arm, S.Camera(None, 160, 90, 30), control_hz=200.0, max_joint_speed=20.0,
                       settle_s=0.02, settle_max_s=0.2)
        lst = Listener(("127.0.0.1", 0), authkey=b"test")
        t = threading.Thread(target=S.serve, args=(srv, "", 0, b"test"),
                             kwargs={"listener": lst, "max_clients": 1, "handle_sigint": False}, daemon=True)
        t.start()
        threads.append(t)
        return arm, lst.address

    yield make
    for t in threads:
        t.join(5)


def _reply(action, q, obj=(237, 2, 30), g=40):
    return json.dumps({"object": {"xyz_mm": list(obj), "uncertainty_mm": 30, "visible": True}, "action": action,
                       "q_deg": None if q is None else [float(v) for v in q], "gripper_open_pct": g,
                       "expected_tcp_mm": None if q is None else G.tcp_mm(q).round(0).tolist(), "reason": "test"})


def _arm(address):
    # the fake arm reports current ~ lag (4000 mA/rad): on a loaded CI box a scheduling hiccup
    # would trip the real stop thresholds, so the tests relax them (stops are the server's tests)
    return ServerArm(address[0], address[1], tcp_speed_mm_s=2000.0, track_stop_deg=180.0, current_stop_ma=None)


def _agent(address, monkeypatch, tmp_path, replies, **cfg):
    monkeypatch.setenv("CONTROLR_SO101_AUTHKEY", "test")
    arm = _arm(address)
    llm = FakeLLM(replies)
    agent = FkAgent(arm, llm, AgentConfig(task="pick up the ball", max_steps=4, reset_duration_s=0.2, fold=False, **cfg),
                    tmp_path / "ep", limits=arm.limits())
    return agent, arm, llm


def test_loop_move_empty_grip_bad_reply_give_up(server_factory, monkeypatch, tmp_path):
    fake, address = server_factory()
    q = GRASP + G.JOINT_OFFSETS_DEG
    far = np.array([0.0, 40.0, -20.0, 70.0, -90.0])
    agent, arm, llm = _agent(address, monkeypatch, tmp_path,
                             [_reply("move", q + [0, -10, 0, 0, 0]), _reply("grip", q), "garbage",
                              _reply("move", far), _reply("give_up", None)],
                             object_xyz_mm=(237, 2, 30))
    try:
        summary = agent.run()
    finally:
        arm.close()
    assert summary["outcome"] == "give_up" and not summary["success"]
    recs = [json.loads(x) for x in (agent.dir / "steps.jsonl").read_text().splitlines()]
    assert [r["action"] for r in recs] == ["observe", "move", "grip", "move", "give_up"]
    assert recs[2]["result"].startswith("the jaws closed EMPTY")
    assert recs[3]["llm"][0].get("parse_error") and len(recs[3]["llm"]) == 2      # re-asked once
    assert recs[3]["plan"]["clamped"]                                             # far target pulled back
    assert all(r["home"] and not r["home"]["stopped"] for r in recs[1:4])
    # the arm is back at the start pose after every motion (server frame = model - offsets)
    np.testing.assert_allclose(np.degrees(fake.q), np.asarray(G.SURVEY_Q_DEG) - G.JOINT_OFFSETS_DEG, atol=0.5)
    # reached the target, photo saved, metrics against the operator's object position
    np.testing.assert_allclose(recs[2]["reached_q_deg"], q, atol=0.5)
    assert (agent.dir / recs[1]["photo"]).exists() and recs[2]["tcp_to_object_mm"] < 25
    assert len(list((agent.dir / "prompts").glob("*.json"))) == 5
    # the model saw its history: step 3's request mentions the empty grip and has 3 photos
    last = llm.calls[-1]["messages"]
    assert Pm.count_images(last) == 4
    assert "closed EMPTY" in json.dumps(last)
    assert json.loads((agent.dir / "config.json").read_text())["limits"]["box_hi_mm"][0] == 270.0


def test_loop_grasp_lifts_and_ends(server_factory, monkeypatch, tmp_path):
    fake, address = server_factory(stall=22.0)
    q = GRASP + G.JOINT_OFFSETS_DEG
    agent, arm, _ = _agent(address, monkeypatch, tmp_path, [_reply("grip", q)])
    try:
        summary = agent.run()
    finally:
        arm.close()
    assert summary["outcome"] == "grasped" and summary["success"] and summary["steps"] == 1
    rec = [json.loads(x) for x in (agent.dir / "steps.jsonl").read_text().splitlines()][-1]
    assert rec["result"].startswith("GRASPED")
    assert rec["after_lift"]["tcp_mm"][2] == pytest.approx(G.tcp_mm(q)[2] + 50, abs=4)
    assert (agent.dir / rec["photo_after_lift"]).exists()
    assert rec["placed_back"] and rec["place_down"] and not rec["place_down"]["stopped"]
    np.testing.assert_allclose(rec["place_down"]["q_deg"], q, atol=0.5)       # put down where it was
    np.testing.assert_allclose(np.degrees(fake.q), np.asarray(G.SURVEY_Q_DEG) - G.JOINT_OFFSETS_DEG, atol=0.5)
    assert fake.g == pytest.approx(35.0, abs=1.0)        # reopened (requested 40 %, capped at 35 %)


def test_wait_does_not_move_and_people_rule_is_in_the_prompt(server_factory, monkeypatch, tmp_path):
    fake, address = server_factory()
    agent, arm, llm = _agent(address, monkeypatch, tmp_path, ['{"action": "wait"}', _reply("give_up", None)])
    try:
        summary = agent.run()
    finally:
        arm.close()
    recs = [json.loads(x) for x in (agent.dir / "steps.jsonl").read_text().splitlines()]
    assert [r["action"] for r in recs] == ["observe", "wait", "give_up"] and summary["motions"] == 0
    assert (agent.dir / recs[1]["photo"]).exists()
    sysmsg = llm.calls[0]["messages"][0]["content"]
    assert "a hand or an arm is visible" in sysmsg and '"wait"' in sysmsg and "10..35" in sysmsg


def test_repeated_boundary_clamps_stop_the_episode(server_factory, monkeypatch, tmp_path):
    fake, address = server_factory()
    far = np.array([0.0, 40.0, -20.0, 70.0, -90.0])          # beyond x = 270 every time
    agent, arm, _ = _agent(address, monkeypatch, tmp_path, [_reply("move", far)] * 4)
    try:
        summary = agent.run()
    finally:
        arm.close()
    recs = [json.loads(x) for x in (agent.dir / "steps.jsonl").read_text().splitlines()]
    assert summary["outcome"] == "boundary"
    assert [r["action"] for r in recs] == ["observe", "move", "boundary_stop"]
    np.testing.assert_allclose(np.degrees(fake.q), np.asarray(G.SURVEY_Q_DEG) - G.JOINT_OFFSETS_DEG, atol=0.5)


def test_loop_rejected_target_does_not_move(server_factory, monkeypatch, tmp_path):
    fake, address = server_factory()
    q = GRASP + G.JOINT_OFFSETS_DEG
    monkeypatch.setenv("CONTROLR_SO101_AUTHKEY", "test")
    arm = _arm(address)
    lim = arm.limits(max_step_mm=0.5)           # after the first target nothing may move further
    agent = FkAgent(arm, FakeLLM([_reply("move", q), _reply("move", q + [-15, 0, 0, 0, 0]), _reply("give_up", None)]),
                    AgentConfig(max_steps=3, reset_duration_s=0.2, fold=False), tmp_path / "ep", limits=lim)
    try:
        agent.run()
    finally:
        arm.close()
    recs = [json.loads(x) for x in (agent.dir / "steps.jsonl").read_text().splitlines()]
    assert [r["action"] for r in recs] == ["observe", "move", "rejected", "give_up"]
    assert "move" not in recs[2] and not recs[2]["plan"]["ok"]
    assert "refused" in agent.steps[2].notes[0]
    np.testing.assert_allclose(np.degrees(fake.q), np.asarray(G.SURVEY_Q_DEG) - G.JOINT_OFFSETS_DEG, atol=0.5)


# ----------------------------------------------------------------------------- offline

def test_offline_eval_with_fake_llm(tmp_path):
    sessions, root = _dataset(tmp_path, n=2)
    g = sessions[0]["grasp"]
    llm = FakeLLM([_reply("grip", g["q_deg"], obj=g["tcp_mm"])])
    res = O.run(sessions, root, tmp_path / "off", llm, model="fake", workers=1, views=("pre_grasp",))
    assert len(res) == 2 * len(O.CONDITIONS)
    by = {(r["session"], r["condition"]): r for r in res}
    r0 = by[(sessions[0]["session_id"], "zero")]
    assert r0["est_err_mm"] == 0.0 and r0["target_err_mm"] == pytest.approx(0.0, abs=0.1)
    assert r0["grasp_err_mm"] == pytest.approx(0.0, abs=0.1)        # grip: grasp_q = q_deg
    assert by[(sessions[0]["session_id"], "loo")]["n_fewshot"] == 1
    assert by[(sessions[0]["session_id"], "loo_rec")]["n_fewshot"] == 1
    assert r0["baseline_stay_mm"] >= 60
    # zero-shot requests carry 1 photo, few-shot ones the other session's photos too
    n_imgs = sorted(Pm.count_images(c["messages"]) for c in llm.calls)
    assert n_imgs[0] == 1 and n_imgs[-1] > 1
    rep = O.report(res)
    assert "| target TCP 3D |" in rep and "| proposed grasp TCP 3D |" in rep and "zero" in rep
    assert (tmp_path / "off" / "results.jsonl").exists()


def test_fewshot_split_excludes_the_held_out_session():
    ss = [{"session_id": "a#1", "recording": "a"}, {"session_id": "a#2", "recording": "a"},
          {"session_id": "b#1", "recording": "b"}]
    assert [s["session_id"] for s in O.fewshot_for("loo", ss[0], ss)] == ["a#2", "b#1"]
    assert [s["session_id"] for s in O.fewshot_for("loo_rec", ss[0], ss)] == ["b#1"]
    assert O.fewshot_for("zero", ss[0], ss) == []


# ----------------------------------------------------------------------------- continuous loop

def _cagent(address, monkeypatch, tmp_path, replies, **cfg):
    monkeypatch.setenv("CONTROLR_SO101_AUTHKEY", "test")
    arm = _arm(address)
    llm = FakeLLM(replies)
    cfg = {"loop": "continuous", "fold": False, **cfg}
    agent = FkAgent(arm, llm, AgentConfig(task="pick and place", max_steps=5, reset_duration_s=0.2, **cfg),
                    tmp_path / "ep", limits=arm.limits())
    return agent, arm, llm


def _steps(agent):
    return [json.loads(x) for x in (agent.dir / "steps.jsonl").read_text().splitlines()]


def test_continuous_pick_and_place(server_factory, monkeypatch, tmp_path):
    fake, address = server_factory(stall=22.0)
    q = GRASP + G.JOINT_OFFSETS_DEG
    place = q + np.array([-12.0, 0, 0, 0, 0])            # ~50 mm to the left
    agent, arm, llm = _cagent(address, monkeypatch, tmp_path, [_reply("grip", q), _reply("place", place)],
                              object_xyz_mm=(237, 2, 30), place_xyz_mm=tuple(G.tcp_mm(place)[:2]))
    try:
        summary = agent.run()
    finally:
        arm.close()
    assert summary["outcome"] == "placed" and summary["success"] and summary["grasped"]
    recs = _steps(agent)
    assert [r["action"] for r in recs] == ["observe", "grip", "place", "home"]
    # no return to the start pose between the grasp and the place: step 2 starts at the lifted pose
    lifted = G.tcp_mm(recs[2]["start_q_deg"])
    assert lifted[2] == pytest.approx(G.tcp_mm(q)[2] + 50, abs=4)
    assert recs[1]["holding"] and not recs[2]["holding"]
    assert summary["place_error_xy_mm"] < 5 and summary["closest_tcp_to_object_mm"] < 25
    np.testing.assert_allclose(np.degrees(fake.q), np.asarray(G.SURVEY_Q_DEG) - G.JOINT_OFFSETS_DEG, atol=0.5)
    sysmsg = llm.calls[0]["messages"][0]["content"]
    assert "(continuous)" in sysmsg and '"place"' in sysmsg and "FROM WHERE IT IS NOW" in sysmsg
    assert "HOLDING the object" in json.dumps(llm.calls[1]["messages"])


def test_continuous_empty_grip_stays_low_revisit_refused_then_home(server_factory, monkeypatch, tmp_path):
    fake, address = server_factory()
    q = GRASP + G.JOINT_OFFSETS_DEG
    agent, arm, llm = _cagent(address, monkeypatch, tmp_path,
                              [_reply("grip", q), _reply("place", q + [0, -5, 0, 0, 0]),
                               _reply("move", q + [0.3, 0, 0, 0, 0]), _reply("give_up", None)])
    try:
        summary = agent.run()
    finally:
        arm.close()
    recs = _steps(agent)
    assert [r["action"] for r in recs] == ["observe", "grip", "rejected", "rejected", "give_up"]
    assert recs[1]["result"].startswith("the jaws closed EMPTY")
    assert "needs the object" in recs[2]["refused"] and "revisit" in recs[3]["refused"]
    # stayed at the grasp pose after the empty close (no return to start in between)
    np.testing.assert_allclose(recs[2]["start_q_deg"], q, atol=0.5)
    assert summary["outcome"] == "give_up" and "returned to the start pose" in summary["end"]
    np.testing.assert_allclose(np.degrees(fake.q), np.asarray(G.SURVEY_Q_DEG) - G.JOINT_OFFSETS_DEG, atol=0.5)


def test_continuous_path_lifts_first():
    lim = G.Limits()
    q = GRASP + G.JOINT_OFFSETS_DEG
    rows = G.continuous_path(q, q + np.array([-10.0, 0, 0, 0, 0]), lim)
    assert G.tcp_mm(rows[1])[2] > G.tcp_mm(q)[2]
    up_end = max(G.tcp_mm(r)[2] for r in rows)
    assert up_end >= G.tcp_mm(q)[2] + lim.approach_dz_mm - 3
    assert G.check_path(rows, lim) == ""


def _plan(grasp, place, action="plan", g=30, rel=35):
    return json.dumps({"object": {"xyz_mm": [237, 2, 30]}, "action": action,
                       "grasp_q_deg": None if grasp is None else [float(v) for v in grasp], "gripper_open_pct": g,
                       "place_q_deg": None if place is None else [float(v) for v in place], "release_open_pct": rel,
                       "reason": "test"})


def test_plan_once_stages_checks_and_fold(server_factory, monkeypatch, tmp_path):
    fake, address = server_factory(stall=22.0)
    q = GRASP + G.JOINT_OFFSETS_DEG
    place = q + np.array([-30.0, 0, 0, 0, 0])            # ~120 mm away: split by the 80 mm step limit
    agent, arm, llm = _cagent(address, monkeypatch, tmp_path, [_plan(q, place)], loop="plan_once", fold=True)
    try:
        summary = agent.run()
    finally:
        arm.close()
    recs = _steps(agent)
    stages = [r.get("stage") or r.get("action") for r in recs]
    assert stages == ["observe", "plan", "grasp", "lift", "carry", "release", "home"]
    assert all(r["check"]["ok"] for r in recs if "check" in r)
    carry = next(r for r in recs if r.get("stage") == "carry")
    assert len(carry["moves"]) >= 2 and any("split" in n for n in carry["notes"])
    assert summary["outcome"] == "placed" and summary["replans"] == 0 and len(llm.calls) == 1
    assert summary["fold"]["folded"]
    np.testing.assert_allclose(np.degrees(fake.q), S.REST_DEG, atol=0.5)       # folded at the end
    assert "(plan once)" in llm.calls[0]["messages"][0]["content"]


def test_plan_once_empty_grasp_replans(server_factory, monkeypatch, tmp_path):
    fake, address = server_factory()                      # nothing between the jaws
    q = GRASP + G.JOINT_OFFSETS_DEG
    agent, arm, llm = _cagent(address, monkeypatch, tmp_path,
                              [_plan(q, q + [-10, 0, 0, 0, 0]), _plan(q + [3, 0, 0, 0, 0], q + [-10, 0, 0, 0, 0]),
                               _plan(None, None, action="give_up")], loop="plan_once")
    try:
        summary = agent.run()
    finally:
        arm.close()
    recs = _steps(agent)
    checks = [r["check"] for r in recs if "check" in r]
    assert [c["ok"] for c in checks] == [False, False]
    assert [r["action"] for r in recs if r.get("action") in ("plan", "replan")] == ["plan", "replan", "replan"]
    assert summary["outcome"] == "give_up" and summary["replans"] == 2 and not summary["grasped"]
    assert "closed EMPTY" in json.dumps(llm.calls[1]["messages"])


def test_history_keeps_25_states_and_downscales_old_photos(tmp_path):
    img = np.zeros((360, 640, 3), np.uint8)
    import io
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG")
    steps = [Pm.Step(i, "observe" if i == 0 else "move", [0, 0, 0, 0, 0], 30.0, buf.getvalue(),
                     requested_q_deg=None if i == 0 else [0, 0, 0, 0, 0]) for i in range(30)]
    parts = Pm.history_parts("t", steps, long_edge=640)
    imgs = [p for p in parts if not isinstance(p, str)]
    assert len(imgs) == 25 and "(5 older steps omitted)" in parts
    assert [p.width for p in imgs].count(640) == 1 + 5 and [p.width for p in imgs].count(320) == 19


def test_temperature_guard(server_factory, monkeypatch, tmp_path):
    fake, address = server_factory()
    agent, arm, llm = _cagent(address, monkeypatch, tmp_path, [_reply("move", GRASP + G.JOINT_OFFSETS_DEG)],
                              max_temp_c=20.0)          # the fake reports 25 C
    try:
        summary = agent.run()
    finally:
        arm.close()
    assert summary["outcome"] == "overheat" and len(llm.calls) == 0


def test_continuous_cut_place_does_not_release(server_factory, monkeypatch, tmp_path):
    fake, address = server_factory(stall=22.0)
    q = GRASP + G.JOINT_OFFSETS_DEG
    far = q + np.array([-40.0, 0, 0, 0, 0])               # > 80 mm away: the place move is cut
    agent, arm, llm = _cagent(address, monkeypatch, tmp_path,
                              [_reply("grip", q), _reply("place", far), _reply("give_up", None)])
    try:
        summary = agent.run()
    finally:
        arm.close()
    recs = _steps(agent)
    assert recs[2]["requested_action"] == "place" and recs[2]["action"] == "move"
    assert "CUT" in recs[2]["result"] and recs[2]["holding"]
    assert summary["outcome"] == "give_up" and summary["placed_tcp_mm"] is None
    assert "put the object back down" in summary["end"]


# ----------------------------------------------------------------------------- pixel loop

def _eye(ag, cam, ball, target, *, blind: int = 0, calls=[0]):  # noqa: B006 (counter shared per test via reset)
    """A perfect fake 'model': projects the true ball / target into the newest photo's camera."""
    q = ag.steps[-1].q_deg
    holding = bool(ag.steps[-1].holding)
    if any(s.action == "place" for s in ag.steps):
        return json.dumps({"object_px": None, "target_px": None, "held": False, "done": True, "note": "on target"})
    out = {"object_px": None, "target_px": None, "held": holding, "done": False, "person": False, "note": "t"}
    if len([s for s in ag.steps]) > blind:
        for key, pt in (("object_px", ball), ("target_px", target)):
            if key == "object_px" and holding:
                continue
            u, v, z = cam.project(q, pt)
            if z > 0 and 0 <= u <= 1280 and 0 <= v <= 720:
                out[key] = [u / 2, v / 2]
    return json.dumps(out)


def test_pixel_geometry_roundtrip_and_fit():
    from controlr.fkagent import visual as V
    true = V.Camera(R_fix=V.rotvec_to_matrix(np.radians([-10.0, -16.0, -3.0])))
    q = np.asarray(G.SURVEY_Q_DEG)
    P = np.array([230.0, -60.0, 40.0])
    u, v, _ = true.project(q, P)
    np.testing.assert_allclose(V.pixel_to_point(true, q, u, v, 40.0), P, atol=0.5)
    # the stock mount misses it; a fit from 4 observations recovers the rotation
    assert np.linalg.norm(V.pixel_to_point(V.Camera(), q, u, v, 40.0)[:2] - P[:2]) > 40
    obs = []
    for qq, pp in [(q, P), (q + [10, 0, 0, 0, 0], [250, 40, 30]), (q + [0, 15, -10, 0, 0], [240, -20, 0]),
                   (q + [-15, 0, 0, 0, 0], [210, 90, 20])]:
        uu, vv, _ = true.project(qq, pp)
        obs.append((qq, uu, vv, np.asarray(pp, float)))
    fit = V.fit_rotation(obs)
    np.testing.assert_allclose(V.pixel_to_point(fit, q, u, v, 40.0), P, atol=1.0)
    qg, pitch = V.tool_pose_q((237.0, 2.0, 18.0), G.Limits(), q)
    assert qg is not None and pitch >= 60
    np.testing.assert_allclose(G.tcp_mm(qg), [237, 2, 18], atol=2.0)


def test_parse_pixel_reply():
    r = Pm.parse_pixel_reply('{"object_px": [320, 100], "target_px": null, "held": false, "done": false, '
                             '"person": false, "note": "ball upper middle"}', 640, 360)
    assert r.object_px == [320.0, 100.0] and r.target_px is None and not r.done
    with pytest.raises(Pm.ReplyError, match="outside"):
        Pm.parse_pixel_reply('{"object_px": [900, 100], "done": false}', 640, 360)
    assert Pm.pixel_json_complete('{"object_px": null, "done": true}') and not Pm.pixel_json_complete('{"object_px": [1,')


def test_pixel_loop_pick_place_done_and_fold(server_factory, monkeypatch, tmp_path):
    from controlr.fkagent import visual as V
    fake, address = server_factory(stall=22.0)
    cam = V.Camera()
    ball, target = np.array([237.0, 2.0, 40.0]), np.array([200.0, -60.0, 0.0])
    state = {"agent": None}

    def eye(messages):
        return _eye(state["agent"], cam, ball, target)

    monkeypatch.setenv("CONTROLR_SO101_AUTHKEY", "test")
    arm = _arm(address)
    llm = FakeLLM(eye)
    agent = FkAgent(arm, llm, AgentConfig(task="put the ball on the target", max_steps=8, loop="pixel", camera=cam,
                                          fold=True), tmp_path / "ep", limits=arm.limits())
    state["agent"] = agent
    try:
        summary = agent.run()
    finally:
        arm.close()
    recs = _steps(agent)
    acts = [r.get("action") for r in recs]
    assert acts[:4] == ["observe", "look", "grasp", "place"] and acts[-1] == "done", acts
    grasp = next(r for r in recs if r.get("action") == "grasp")
    np.testing.assert_allclose(grasp["object_estimate_mm"], ball, atol=2.0)
    assert grasp["holding"] and "stalled" in grasp["result"]
    place = next(r for r in recs if r.get("action") == "place")
    np.testing.assert_allclose(place["target_estimate_mm"], target, atol=2.0)
    assert summary["success"] and summary["model_judged_done"] and summary["fold"]["folded"]
    assert summary["motions"] == 3                       # look, grasp, place
    np.testing.assert_allclose(np.degrees(fake.q), S.REST_DEG, atol=0.5)
    # the model never sees a request for joint angles; it gets the code's estimate as text
    sysmsg = llm.calls[0]["messages"][0]["content"]
    assert "never give" in sysmsg and "object_px" in sysmsg
    assert "object estimate from your pixel" in json.dumps(llm.calls[2]["messages"])


def test_cli_live_pixel_defaults_on_the_fake_server(server_factory, monkeypatch, tmp_path, capsys):
    import importlib.util
    from controlr.fkagent import visual as V
    spec = importlib.util.spec_from_file_location("fkcli", "scripts/so101_fk_agent.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    fake, address = server_factory()
    monkeypatch.setenv("CONTROLR_SO101_AUTHKEY", "test")
    seen = {}
    real = cli.FkAgent

    def spy(arm, llm, cfg, log_dir, **kw):
        seen["cfg"] = cfg
        return real(arm, llm, cfg, log_dir, **kw)

    monkeypatch.setattr(cli, "FkAgent", spy)
    rc = cli.main(["live", "--host", address[0], "--port", str(address[1]), "--loop", "pixel", "--fake-llm",
                   "--task", "put the beige rope ball on the plush carrot", "--max-steps", "4",
                   "--tcp-speed", "2000", "--log-root", str(tmp_path)])
    assert rc == 0
    cfg = seen["cfg"]
    assert cfg.loop == "pixel" and cfg.target_z_mm == 50.0 and cfg.object_z_mm == 40.0 and cfg.grasp_z_mm == 18.0
    np.testing.assert_allclose(cfg.camera.R_fix, V.Camera.fitted().R_fix)
    assert cfg.gripper_open_max == 35.0 and cfg.extra_body == {"reasoning_effort": "low"}
    out = capsys.readouterr().out
    assert "built-in fit" in out and '"outcome": "done"' in out
    np.testing.assert_allclose(np.degrees(fake.q), S.REST_DEG, atol=0.5)        # folded at the end
    with pytest.raises(SystemExit, match="no such file"):
        cli.main(["live", "--host", address[0], "--port", str(address[1]), "--loop", "pixel", "--fake-llm",
                  "--task", "x", "--camera-fit", str(tmp_path / "missing.json"), "--log-root", str(tmp_path)])


def _pixel_agent(address, monkeypatch, tmp_path, eye_fn, cam, **cfg):
    monkeypatch.setenv("CONTROLR_SO101_AUTHKEY", "test")
    arm = _arm(address)
    agent = FkAgent(arm, FakeLLM(eye_fn), AgentConfig(task="ball on target", max_steps=12, loop="pixel", camera=cam,
                                                      fold=False, **cfg), tmp_path / "ep", limits=arm.limits())
    return agent, arm


def test_pixel_search_looks_around_until_both_are_located(server_factory, monkeypatch, tmp_path):
    from controlr.fkagent import visual as V
    fake, address = server_factory(stall=22.0)
    cam = V.Camera()
    ball, target = np.array([237.0, 2.0, 40.0]), np.array([200.0, -60.0, 0.0])
    holder = {}
    # "not visible" in the first 3 photos (start pose + 2 look poses)
    agent, arm = _pixel_agent(address, monkeypatch, tmp_path, lambda m: _eye(holder["a"], cam, ball, target, blind=3), cam)
    holder["a"] = agent
    try:
        summary = agent.run()
    finally:
        arm.close()
    recs = _steps(agent)
    acts = [r.get("action") for r in recs]
    assert acts[1:4] == ["search", "search", "search"], acts       # step 0 + 2 looks were blind
    assert summary["searches"] == 3 and summary["success"]
    np.testing.assert_allclose(summary["found"]["object"]["xyz_mm"], ball, atol=2.0)
    np.testing.assert_allclose(summary["found"]["target"]["xyz_mm"], target, atol=2.0)
    assert "place" in acts and acts[-1] == "done"
    assert any("LOCATED" in n for r in recs for n in (r.get("notes") or []))


def test_pixel_out_of_reach_is_reported_not_chased(server_factory, monkeypatch, tmp_path):
    from controlr.fkagent import visual as V
    fake, address = server_factory()
    cam = V.Camera()
    ball, target = np.array([300.0, 20.0, 40.0]), np.array([200.0, -60.0, 0.0])   # x 300 > 270
    holder = {}
    agent, arm = _pixel_agent(address, monkeypatch, tmp_path, lambda m: _eye(holder["a"], cam, ball, target), cam)
    holder["a"] = agent
    try:
        summary = agent.run()
    finally:
        arm.close()
    assert summary["outcome"] == "out_of_reach" and not summary["success"]
    far = summary["out_of_reach"]["object"]
    np.testing.assert_allclose(far["xyz_mm"][:2], ball[:2], atol=2.0)
    assert "outside the workspace" in far["why"]
    acts = [r.get("action") for r in _steps(agent)]
    assert "grasp" not in acts and "look" not in acts


def test_search_poses_are_inside_the_limits(server_factory, monkeypatch, tmp_path):
    fake, address = server_factory()
    agent, arm = _pixel_agent(address, monkeypatch, tmp_path, lambda m: "{}", None)
    try:
        poses = agent._search_poses()
    finally:
        arm.close()
    names = [n for n, _ in poses]
    assert names[:4] == ["pan -20", "pan +20", "pan -40", "pan +40"] and "far" in names and "near" in names
    for _, q in poses:
        assert G.check_path([q], agent.lim) == ""
    far_q = dict(poses)["far"]
    assert G.tool_pitch_deg(far_q) < 60                   # tilted forward: sees farther


# ----------------------------------------------------------------------------- tool-calling dialog

def _tool_script(holder, cam, ball, target, *, blind_frames=0):
    """A fake model that 'sees' the true scene through the same camera and calls tools."""
    st = {"phase": "find", "looks": 0}

    def px(pt):
        ag = holder["a"]
        u, v, z = cam.project(ag._frame_q, pt)
        return (u / 2, v / 2) if (z > 0 and 0 <= u <= 1280 and 0 <= v <= 720) else None

    def script(messages):
        ag = holder["a"]
        assert messages[-1]["role"] in ("user", "tool")
        blind = ag._n_frame < blind_frames
        b, t = (None, None) if blind else (px(ball), px(target))
        ph = st["phase"]
        if ph == "find":
            if b is None or t is None:
                st["looks"] += 1
                return "", [("look", {"pose": ["pan+20", "pan-20", "far", "near"][min(st["looks"] - 1, 3)]})]
            st["phase"] = "above"
            return "", [("locate", {"u": b[0], "v": b[1], "height_mm": 40}),
                        ("move_to", {"u": b[0], "v": b[1], "height_mm": 40, "above_mm": 80})]
        if ph == "above":
            st["phase"] = "down"
            return "", [("move_to", {"u": b[0], "v": b[1], "height_mm": 40, "z": 18})]
        if ph == "down":
            st["phase"] = "close"
            return "", [("grip", {"action": "close"})]
        if ph == "close":
            st["phase"] = "lift"
            return "", [("lift", {"dz_mm": 50}), ("move_to", {"x": 1, "y": 1, "z": 1})]   # 2nd motion is refused
        if ph == "lift":
            if t is None:
                return "", [("look", {"pose": "survey"})]
            st["phase"] = "release"
            return "", [("move_to", {"u": t[0], "v": t[1], "height_mm": 0, "above_mm": 40})]
        if ph == "release":
            st["phase"] = "up"
            return "", [("grip", {"action": "open"})]
        if ph == "up":
            st["phase"] = "end"
            return "", [("lift", {})]
        return "ball is on the target", [("done", {"summary": "ball on target"})]

    return script


def _tool_agent(address, monkeypatch, tmp_path, script, cam, **cfg):
    from controlr.fkagent.toolagent import ToolAgent
    from controlr.fkagent.toolchat import FakeToolChat
    monkeypatch.setenv("CONTROLR_SO101_AUTHKEY", "test")
    arm = _arm(address)
    chat = FakeToolChat(script)
    ag = ToolAgent(arm, chat, AgentConfig(task="put the ball on the target", max_steps=20, loop="tools", camera=cam,
                                          **cfg), tmp_path / "ep", limits=arm.limits())
    return ag, arm, chat


def test_tool_dialog_pick_place_with_search(server_factory, monkeypatch, tmp_path):
    from controlr.fkagent import visual as V
    fake, address = server_factory(stall=22.0)
    cam = V.Camera()
    ball, target = np.array([237.0, 2.0, 40.0]), np.array([200.0, -60.0, 0.0])
    holder = {}
    ag, arm, chat = _tool_agent(address, monkeypatch, tmp_path, _tool_script(holder, cam, ball, target, blind_frames=2), cam)
    holder["a"] = ag
    try:
        summary = ag.run()
    finally:
        arm.close()
    assert summary["outcome"] == "done" and summary["success"] and summary["grasped"], summary
    assert summary["fold"]["folded"]
    np.testing.assert_allclose(np.degrees(fake.q), S.REST_DEG, atol=0.5)
    recs = _steps(ag)
    names = [t["name"] for r in recs for t in r.get("tool_calls", [])]
    assert names[:2] == ["look", "look"] and "grip" in names and names[-1] == "done"
    loc = next(t for r in recs for t in r.get("tool_calls", []) if t["name"] == "locate")
    np.testing.assert_allclose(loc["result"]["xyz_mm"], ball, atol=2.0)
    grip = next(t for r in recs for t in r.get("tool_calls", []) if t["name"] == "grip")
    assert grip["result"]["blocked"]
    refused = [t for r in recs for t in r.get("tool_calls", []) if "one motion per turn" in str(t["result"])]
    assert len(refused) == 1
    np.testing.assert_allclose(summary["released_at_mm"][:2], target[:2] + 0, atol=5.0)
    # the dialog is one append-only conversation: system, frame, (assistant, tool..., frame)*
    msgs = chat.calls[-1]["messages"]
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    for i, m in enumerate(msgs):
        if m["role"] == "tool":
            j = i - 1
            while msgs[j]["role"] == "tool":
                j -= 1
            assert msgs[j]["role"] == "assistant" and msgs[j]["tool_calls"]
    imgs = sum(1 for m in msgs if m["role"] == "user" and isinstance(m["content"], list)
               for c in m["content"] if c["type"] == "image_url")
    assert imgs <= ag.keep_photos
    assert any("omitted" in json.dumps(m) for m in msgs)
    prefix = json.dumps(chat.calls[1]["messages"][:2])
    assert json.dumps(chat.calls[2]["messages"][:2]) == prefix                 # append-only head (cache)


def test_tool_dialog_out_of_reach_and_bad_calls(server_factory, monkeypatch, tmp_path):
    from controlr.fkagent import visual as V
    fake, address = server_factory()
    cam = V.Camera()
    holder = {}
    turns = iter([
        ("", [("move_to", {"x": 330, "y": 0, "z": 20})]),              # outside the box: refused, no motion
        ("", [("locate", {"u": 900, "v": 50, "height_mm": 40})]),      # outside the frame
        ("", [("frobnicate", {})]),
        ("nothing", []),                                                # no tool call: nudged
        ("", [("give_up", {"reason": "ball at x 330 is outside the workspace"})]),
    ])
    ag, arm, chat = _tool_agent(address, monkeypatch, tmp_path, lambda m: next(turns), cam)
    holder["a"] = ag
    try:
        summary = ag.run()
    finally:
        arm.close()
    assert summary["outcome"] == "give_up" and not summary["success"] and "outside" in summary["end_text"]
    recs = _steps(ag)
    res = [t["result"] for r in recs for t in r.get("tool_calls", [])]
    assert "outside the workspace" in res[0]["why"] and "outside the 640x360 frame" in res[1]["why"]
    assert "unknown tool" in res[2]["why"]
    assert any(r.get("action") == "no_tool_call" for r in recs)
    assert summary["frames"] == 0                                       # nothing moved, no new frame


def test_side_extension_of_the_box():
    lim = G.Limits(side_y_mm=175.0, side_x_max_mm=230.0)
    assert G.in_box(np.array([193.0, 161.0, 40.0]), lim)              # the ball after t1
    assert G.in_box(np.array([193.0, -170.0, 40.0]), lim)
    assert not G.in_box(np.array([250.0, 161.0, 40.0]), lim)           # beyond x 230 on the side
    assert not G.in_box(np.array([193.0, 180.0, 40.0]), lim)
    assert not G.in_box(np.array([193.0, 161.0, 40.0]), G.Limits())     # off by default
    assert "where |y|" in G.box_violation(np.array([250.0, 161.0, 40.0]), lim)


def test_tool_grip_wide_only_for_the_carrot(server_factory, monkeypatch, tmp_path):
    from controlr.fkagent import visual as V
    fake, address = server_factory()
    turns = iter([("", [("grip", {"action": "open", "open_pct": 50})]),
                  ("", [("grip", {"action": "open", "open_pct": 50, "object": "carrot"})]),
                  ("", [("grip", {"action": "open", "open_pct": 70, "object": "carrot"})]),
                  ("", [("give_up", {"reason": "test"})])])
    ag, arm, chat = _tool_agent(address, monkeypatch, tmp_path, lambda m: next(turns), V.Camera(), gripper_wide_pct=50.0,
                                fold=False)
    try:
        ag.run()
    finally:
        arm.close()
    res = [t["result"] for r in _steps(ag) for t in r.get("tool_calls", [])]
    assert res[0]["gripper_pct"] == pytest.approx(35, abs=1.5) and "capped at 35" in res[0]["note"]
    assert res[1]["gripper_pct"] == pytest.approx(50, abs=1.5) and "note" not in res[1]
    assert res[2]["gripper_pct"] == pytest.approx(50, abs=1.5) and "capped at 50" in res[2]["note"]
    assert "up to 50" in chat.calls[0]["messages"][0]["content"]


def test_tool_nudge_moves_along_the_image_axes_and_lift_down(server_factory, monkeypatch, tmp_path):
    from controlr.fkagent import visual as V
    fake, address = server_factory()
    cam = V.Camera()
    q = GRASP + G.JOINT_OFFSETS_DEG + np.array([0, -8, 0, 0, 0])       # some height above the table
    turns = iter([("", [("move_joints", {"q_deg": [float(v) for v in q]})]),
                  ("", [("nudge", {"right_mm": 20})]),
                  ("", [("nudge", {"up_mm": -15})]),
                  ("", [("lift", {"dz_mm": -10})]),
                  ("", [("nudge", {"right_mm": 500})]),
                  ("", [("give_up", {"reason": "test"})])])
    ag, arm, chat = _tool_agent(address, monkeypatch, tmp_path, lambda m: next(turns), cam, fold=False)
    try:
        ag.run()
    finally:
        arm.close()
    res = [t["result"] for r in _steps(ag) for t in r.get("tool_calls", [])]
    p0 = np.array(res[0]["reached_tcp_mm"])
    p1, p2, p3 = (np.array(r["reached_tcp_mm"]) for r in res[1:4])
    assert np.linalg.norm(p1[:2] - p0[:2]) == pytest.approx(20, abs=2.5) and abs(p1[2] - p0[2]) < 2.5
    assert np.linalg.norm(p2[:2] - p1[:2]) == pytest.approx(15, abs=2.5)
    Rc = cam.T_base_cam(q)[:3, :3]
    right = np.array([Rc[0, 0], Rc[1, 0]]) / np.linalg.norm(Rc[:2, 0])
    assert (p1[:2] - p0[:2]) @ right > 15                                 # along the image's right
    assert p3[2] == pytest.approx(p2[2] - 10, abs=2.5)                    # lift(-10) went down
    assert res[4]["moved_mm"] == [40.0, 0.0] or not res[4]["ok"]          # clipped to 40 mm (or refused)
