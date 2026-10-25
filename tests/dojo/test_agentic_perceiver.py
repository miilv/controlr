"""Agentic perceiver runner (dojo.perception.agentic): pins, reply parsing, tools, J1 interface on FakeChat.

No network, no GPU, no catalogue: numpy + pillow only.
"""
import json
import math

import numpy as np
import pytest

from dojo.perception.agentic import j1, prompt, runner
from dojo.perception.agentic.llm import FakeChat
from dojo.perception.agentic.tools_rd import RoboDojoTools, project_points, write_calib


def _calib():
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    K_h = [[288.1325, 0, 320], [0, 288.1325, 240], [0, 0, 1]]
    T_h = np.eye(4)
    T_h[:3, :3] = [[1, 0, 0], [0, -c, s], [0, -s, -c]]
    T_h[:3, 3] = [0, -0.41, 1.308]
    T_w = np.eye(4)  # a wrist camera 0.25 m above the table looking straight down
    T_w[:3, :3] = np.c_[[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    T_w[:3, 3] = [0.0, -0.1, 1.015]
    K_w = [[397.04, 0, 320], [0, 397.04, 240], [0, 0, 1]]
    return {"cam_head": {"K": K_h, "T_world_cam": T_h.tolist()}, "cam_left_wrist": {"K": K_w, "T_world_cam": T_w.tolist()}}


FLANGE = {"left": [-0.3, -0.2, 0.9, 1, 0, 0, 0], "right": [0.3, -0.2, 0.9, 1, 0, 0, 0]}


def _frames():
    img = np.full((480, 640, 3), 90, np.uint8)
    img[230:250, 310:330] = (200, 30, 30)
    return {"cam_head": img, "cam_left_wrist": img.copy()}


def test_vendor_is_pinned_and_prompt_contains_protocol():
    v = prompt.verify_vendor()
    assert "PERCEIVER-PROTOCOL.md" in v and "robodojo/zoom.py" in v and "j1_schema.py" in v
    s = prompt.system_prompt("robodojo")
    assert s.startswith("# Runner preamble (agentic perceiver, RoboDojo, v1)")
    assert "# Perceiver protocol" in s and "zoom.py" in s
    p1, p2 = prompt.pin("robodojo", "m"), prompt.pin("robodojo", "m")
    assert p1["system_md5"] == p2["system_md5"] and p1["protocol_md5"] == v["PERCEIVER-PROTOCOL.md"]
    assert prompt.system_prompt("held").startswith("# Runner preamble (agentic perceiver, held_look, v1)")


def test_parse_reply_forms():
    assert runner.parse_reply('{"commands": ["a"]}') == {"commands": ["a"]}
    assert runner.parse_reply('sure:\n```json\n{"answer": {"objects": []}}\n```') == {"answer": {"objects": []}}
    assert runner.parse_reply('text {"answer": {"objects": [1]}} tail') == {"answer": {"objects": [1]}}
    assert runner.parse_reply("no json") is None
    assert runner.parse_reply('{"other": 1}') is None
    assert runner.parse_reply('{"held": "yes"}', keys=("held",)) == {"held": "yes"}


def test_tools_locate_project_roundtrip_and_refusals(tmp_path):
    cal = _calib()
    write_calib(tmp_path / "calib0.json", cal, FLANGE)
    t = RoboDojoTools(tmp_path)
    X = [0.05, -0.12, 0.78]
    px = project_points(cal, [X])
    uh, uw = px["cam_head"][0], px["cam_left_wrist"][0]
    out, _ = t.run(f"locate cam=cam_head px=[{uh[0]},{uh[1]}] cam2=cam_left_wrist px2=[{uw[0]},{uw[1]}]")
    d = json.loads(out)
    assert np.allclose(d["xyz"], X, atol=4e-3) and d["ray_gap_mm"] < 5  # pixels rounded to whole px
    out, _ = t.run(f"locate cam=cam_head px=[{uh[0]},{uh[1]}] z=0.78")
    assert np.allclose(json.loads(out)["xyz"][:2], X[:2], atol=3e-3)
    out, _ = t.run(f"bin/robot r1 locate cam=cam_head px=[{uh[0]},{uh[1]}] z=0.78")  # the protocol's server-verb form
    assert "xyz" in out
    out, _ = t.run("project pts=[[0.05,-0.12,0.78]]")
    assert json.loads(out)["cam_head"][0] is not None
    assert t.run("rm -rf /")[0].startswith("error: 'rm'")
    assert "only workspace file names" in t.run("zoom.py /etc/passwd 1 1 1 x.png")[0]


def test_tools_zoom_returns_image_and_protects_frames(tmp_path):
    from PIL import Image

    Image.fromarray(_frames()["cam_head"]).save(tmp_path / "cam_head.png")
    write_calib(tmp_path / "calib0.json", _calib(), FLANGE)
    t = RoboDojoTools(tmp_path)
    out, imgs = t.run("python3 zoom.py cam_head.png 320 240 30 z1.png 6")
    assert [n for n, _ in imgs] == ["z1.png"] and imgs[0][1][:4] == b"\x89PNG"
    out, imgs = t.run("zoom.py cam_head.png 320 240 30 cam_head.png 6")
    assert out.startswith("error") and not imgs


def _b1():
    return {"objects": [{"label": "block_0", "category": "block", "asset": "Rigid/block/00007", "pos": [0.05, -0.12, 0.7825],
                         "quat": [1, 0, 0, 0], "size": [0.035, 0.035, 0.035], "confidence": 0.9, "source": "head"},
                        {"label": "block_1", "category": "block", "asset": "Rigid/block/00010", "pos": [-0.2, -0.1, 0.7825],
                         "quat": [1, 0, 0, 0], "size": [0.035, 0.035, 0.035], "confidence": 0.3, "source": "head",
                         "flags": {"occluded": True, "hidden_frac": 0.4}}],
            "regions": [], "needs_look": [{"cam": "cam_left_wrist", "reason": "partly_hidden", "label": "block_1",
                                           "target": [-0.2, -0.1, 0.78], "flange_pose": [-0.2, -0.2, 0.95, 1, 0, 0, 0]}]}


def test_perceive_impl_fake_session_validates(tmp_path):
    replies = [
        json.dumps({"commands": ["zoom.py cam_head.png 320 240 30 z1.png 6", "locate cam=cam_head px=[320,240] z=0.80"]}),
        lambda msgs: json.dumps({"answer": {"objects": [{"keep": 0},
                                                        {"label": "block", "pos": [-0.21, -0.09, 0.7825], "yaw_deg": 30,
                                                         "top_z": 0.80, "size": [0.035, 0.035, 0.035], "confidence": 0.7}],
                                            "needs_look": [{"keep": 0}], "notes": "block_1 re-located"}}),
    ]
    fc = FakeChat(replies)
    out = j1.perceive_impl(_frames(), _calib(), FLANGE, {"left": 1.0, "right": 1.0}, "Stack the three blocks.", ["block"],
                           b1=_b1(), chat=fc, model="fake", refine=False, log_dir=str(tmp_path))
    assert len(fc.sent) == 2
    first = fc.sent[0][1]["content"]
    assert "Hint (B1" in first[0]["text"] and "[1] label='block_1'" in first[0]["text"] and "hidden 0.40" in first[0]["text"]
    assert sum(p["type"] == "image_url" for p in first) == 2
    tool_turn = fc.sent[1][-1]["content"]
    assert any(p.get("text", "").startswith("[1] $ zoom.py") for p in tool_turn)
    assert any(p["type"] == "image_url" for p in tool_turn)
    objs = out["objects"]
    assert objs[0]["label"] == "block_0" and objs[0]["pos"] == [0.05, -0.12, 0.7825]
    q = objs[1]["quat"]
    assert q[0] == pytest.approx(math.cos(math.radians(15)), abs=1e-6) and q[3] == pytest.approx(math.sin(math.radians(15)), abs=1e-6)
    assert out["needs_look"][0]["flange_pose"] == [-0.2, -0.2, 0.95, 1, 0, 0, 0]
    assert "result=agent" in out["notes"] and "system_md5=" in out["notes"]
    clean, rep = j1._validate(out)
    assert rep["ok"], rep
    assert len(clean["objects"]) == 2
    calls = [json.loads(x) for x in open(tmp_path / "calls.jsonl")]
    assert calls[-1]["result"] == "agent" and calls[-1]["calls"] == 2 and calls[-1]["commands"] == 2


def test_perceive_impl_falls_back_to_b1_without_answer(tmp_path):
    fc = FakeChat(["not json"] * 3)
    out = j1.perceive_impl(_frames(), _calib(), FLANGE, {"left": 1.0, "right": 1.0}, "Stack the three blocks.", ["block"],
                           b1=_b1(), chat=fc, model="fake", max_turns=3, log_dir=str(tmp_path))
    assert "result=b1_fallback" in out["notes"] and len(out["objects"]) == 2
    fc = FakeChat(["not json"] * 3)
    out = j1.perceive_impl(_frames(), _calib(), FLANGE, {}, "x", None, b1=_b1(), chat=fc, model="fake", max_turns=3,
                           fallback="none", log_dir=str(tmp_path))
    assert out["objects"] == []


def test_answer_to_output_rejects_bad_entries():
    objs, looks, probs = j1.answer_to_output({"objects": [{"keep": 7}, {"label": "x"}, "junk", {"label": "T", "pos": [0, 0, 0.78]}],
                                              "needs_look": [{"keep": 3}]}, _b1())
    assert [o["label"] for o in objs] == ["T"] and len(probs) == 4 and looks == []


def test_held_look_fake(tmp_path):
    fc = FakeChat(['{"held": "yes", "note": "cube between the pads"}'])
    out = j1.held_look_impl(_frames(), _calib(), FLANGE, {"left": 0.0, "right": 1.0}, chat=fc, model="fake", held_mode="agent",
                            log_dir=str(tmp_path))
    assert out["held"] is True and out["left"]["held"] is True and out["confidence"] > 0.5
    parts = fc.sent[0][1]["content"]
    assert "left gripper" in parts[0]["text"] and sum(p["type"] == "image_url" for p in parts) == 2
    fc = FakeChat(["maybe", "dunno"])
    out = j1.held_look_impl(_frames(), _calib(), FLANGE, {"left": 0.0}, chat=fc, model="fake", held_mode="agent", log_dir=str(tmp_path))
    assert out["held"] is False and out["confidence"] == 0.0


def test_libero_cell_fake_helpers(tmp_path, monkeypatch):
    from PIL import Image

    from dojo.perception.agentic import libero

    cap = tmp_path / "cap"
    cap.mkdir()
    Image.fromarray(np.full((512, 512, 3), 120, np.uint8)).save(cap / "agentview.png")
    Image.fromarray(np.full((512, 512, 3), 60, np.uint8)).save(cap / "wrist.png")
    h = tmp_path / "harness"
    h.mkdir()
    (h / "libero_rgbd.py").write_text(
        "import json,sys\nuv=[a for i,a in enumerate(sys.argv) if sys.argv[i-1]=='--uv']\n"
        "print(json.dumps({'floor_z':0.9,'points':[{'uv':[int(x) for x in u.split(',')],'world':[0.1,0.2,0.95],'spread_3x3_m':0.002} for u in uv]}))\n")
    (h / "libero_rgbd_v3.py").write_text(
        "import json,sys\nfrom PIL import Image\na=sys.argv\nout=a[a.index('--overlay')+1]\nImage.new('RGB',(64,64)).save(out)\n"
        "p=json.load(open(a[a.index('--picks')+1]))\nprint(json.dumps({'floor_z':0.9,'target':{'centre_xy':[0.1,0.2],'top_z':0.95},'picks_seen':sorted(p)}))\n")
    monkeypatch.setenv("LIBERO_PY", __import__("sys").executable)
    monkeypatch.setenv("LIBERO_HARNESS", str(h))
    picks = {"cam": "agentview", "target": {"name": "alphabet_soup", "kind": "round", "center": [216, 209], "box": [196, 198, 236, 254]},
             "container": {"name": "basket", "rim": [[412, 262], [412, 211]], "box": [355, 207, 478, 328]}}
    fc = FakeChat([json.dumps({"commands": ["grid.py crops agentview.png c1.png --box soup:190,190,240,260", "probe 216,209 412,262",
                                            "locate " + json.dumps(json.dumps(picks))]}),
                   json.dumps({"answer": {"picks": picks, "notes": "soup into basket"}})])
    r = libero.perceive_cell(str(cap), "put the alphabet soup in the basket", "object_task", str(tmp_path / "picks.json"),
                             reply={"robot_base": [0, 0, 0.9], "eef": {"single": {"pos": [0.1, 0.0, 1.1]}}}, model="fake", chat=fc,
                             log_dir=str(tmp_path / "logs"))
    assert r["stats"]["ok"] and r["stats"]["calls"] == 2 and r["stats"]["commands"] == 3
    tool_turn = fc.sent[1][-1]["content"]
    texts = "\n".join(p.get("text", "") for p in tool_turn)
    assert "table z 0.9" in texts and "spread 0.002" in texts and "centre_xy" in texts
    assert sum(p["type"] == "image_url" for p in tool_turn) == 2  # the crop and the overlay
    saved = json.loads((tmp_path / "picks.json").read_text())
    assert saved["target"]["name"] == "alphabet_soup" and saved["robot_base"] == [0, 0, 0.9] and saved["eef"] == [0.1, 0.0, 1.1]
    assert prompt.system_prompt("libero").startswith("# Runner preamble (agentic perceiver, LIBERO, v1)")


def test_urllib_chat_streams_sse(monkeypatch):
    import http.server
    import threading

    from dojo.perception.agentic.llm import UrllibChat

    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers["Content-Length"])
            seen["body"] = json.loads(self.rfile.read(n))
            seen["auth"] = self.headers["Authorization"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for ch in ({"choices": [{"delta": {"content": '{"answer": '}}]}, {"choices": [{"delta": {"content": '{"objects": []}}'}, "finish_reason": "stop"}]},
                       {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 5}}):
                self.wfile.write(b"data: " + json.dumps(ch).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OMNIROUTE_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("OMNIROUTE_API_KEY", "test-not-a-key")
    monkeypatch.delenv("AGENTIC_ENV_FILE", raising=False)
    r = UrllibChat(timeout_s=10)("m", [{"role": "user", "content": "x"}], max_tokens=10)
    srv.shutdown()
    assert r.text == '{"answer": {"objects": []}}' and r.prompt_tokens == 11 and r.completion_tokens == 5 and r.error is None
    assert seen["body"]["stream"] is True and seen["auth"] == "Bearer test-not-a-key"


def test_budget_gate(tmp_path):
    b1 = _b1()
    b1["want_match"] = {"block": "block_0"}
    risky = j1.risk(b1["objects"][1])
    assert 0 <= j1.risk(b1["objects"][0]) < risky <= 1
    cfg = {"gate": "budget", "max_per_episode": 2, "gate_thr": 0.2}
    assert j1.gate_decision(b1, ["block"], None, cfg) == (True, "first_call")
    j1._EPISODE["agent_calls"] = 1
    use, why = j1.gate_decision(b1, ["block"], {"objects": []}, cfg)
    assert use == (risky >= 0.2) and why.startswith("risk" if use else "b1_confident")
    j1._EPISODE["agent_calls"] = 2
    assert j1.gate_decision(b1, ["block"], {"objects": []}, cfg) == (False, "episode_budget_spent")
    calm = {"objects": [b1["objects"][0]], "want_match": {"block": "block_0"}}
    j1._EPISODE["agent_calls"] = 1
    assert j1.gate_decision(calm, ["block"], {"objects": []}, cfg) == (False, "b1_confident")
    out = j1.perceive_impl(_frames(), _calib(), FLANGE, {"left": 1.0}, "Stack the three blocks.", ["block"], prior={"objects": []},
                           b1=calm, chat=FakeChat([]), model="fake", gate="budget", log_dir=str(tmp_path))
    assert "result=b1_gated" in out["notes"] and out["objects"][0]["label"] == "block_0"


def test_unmentioned_b1_objects_are_kept_at_half_confidence():
    objs = [{"label": "block", "pos": [0.051, -0.121, 0.78], "confidence": 0.8}]
    out, n = j1.add_unmentioned(objs, _b1())
    assert n == 1 and out[1]["label"] == "block_1" and out[1]["confidence"] == pytest.approx(0.15)
    assert "flags" not in out[1]
    # the agent moved block_1 under a new label and reports both blocks: no duplicate comes back
    moved = objs + [{"label": "cube left", "category": "block", "pos": [-0.26, -0.15, 0.78]}]
    assert j1.add_unmentioned(moved, _b1())[1] == 0
    # same label = mentioned, even far away
    assert j1.add_unmentioned([{"label": "block_1", "pos": [0.3, 0.0, 0.78]}], _b1())[1] == 1


def _rec():
    return {"task": None, "instruction": "Stack the three blocks.",
            "Rigid": {"block": [
                {"category": "block", "category_idx": 7, "label": "block_0", "default_pos": [0.05, -0.12, 0.7825],
                 "default_ori": [1, 0, 0, 0], "scale": [1, 1, 1], "_perception": {"confidence": 0.9, "residual": 0.1}},
                {"category": "block", "category_idx": 10, "label": "block_1", "default_pos": [-0.2, -0.1, 0.7825],
                 "default_ori": [1, 0, 0, 0], "scale": [1, 1, 1],
                 "_perception": {"confidence": 0.2, "residual": 0.7, "occluded": True, "hidden_frac": 0.5}}]},
            "_not_found": [{"label": "block_2", "candidates": ["Rigid/block/00013"], "confidence": 0.0}]}


def test_record_merge_moves_finds_and_keeps():
    pytest.importorskip("scipy")
    from dojo.perception.agentic import record

    objs = [{"label": "block_0", "asset": "Rigid/block/00007", "pos": [0.05, -0.12, 0.7825], "quat": [1, 0, 0, 0], "source": "head"},
            {"label": "white cube", "category": "block", "pos": [-0.23, -0.14, 0.7825], "confidence": 0.7, "source": "head"},
            {"label": "cube near edge", "category": "block", "pos": [0.40, -0.05, 0.7825], "confidence": 0.6, "source": "head"}]
    new, log = record.merge_into_record(_rec(), objs)
    labs = {r["label"]: r for r in new["Rigid"]["block"]}
    assert labs["block_0"]["default_pos"] == [0.05, -0.12, 0.7825] and "source" not in labs["block_0"]["_perception"]
    assert labs["block_1"]["default_pos"] == [-0.23, -0.14, 0.7825] and labs["block_1"]["_perception"]["source"] == "agent"
    assert labs["block_1"]["_perception"]["b1_pos"] == [-0.2, -0.1, 0.7825]
    assert labs["block_2"]["default_pos"] == [0.40, -0.05, 0.7825] and labs["block_2"]["category_idx"] == 13
    assert new["_not_found"] == [] and {x["action"] for x in log} == {"moved", "found"}
    ok, why = record.needs_agent(_rec())
    assert ok and why.startswith("not_found")


def test_agentic_estimate_on_a_b1_record(tmp_path):
    pytest.importorskip("scipy")
    from dojo.perception.agentic import record

    fc = FakeChat([json.dumps({"answer": {"objects": [{"keep": 0}, {"label": "block", "pos": [-0.23, -0.14, 0.7825]},
                                                      {"label": "block", "pos": [0.40, -0.05, 0.7825]}]}})])
    st = {"left_ee_pose": FLANGE["left"], "right_ee_pose": FLANGE["right"], "left_ee_joint_state": [1.0], "right_ee_joint_state": [1.0]}
    import unittest.mock as um

    with um.patch.object(record, "_calib_from_state", lambda s: _calib()):
        new = record.agentic_estimate(_frames(), st, "stack_blocks", "Stack the three blocks.", bgs={}, b1_record=_rec(),
                                      chat=fc, model="fake", refine=False, log_dir=str(tmp_path))
    ag = new["_perception"]["agentic"]
    assert ag["result"] == "agent" and ag["calls"] == 1
    labs = {r["label"]: r for r in new["Rigid"]["block"]}
    assert labs["block_1"]["default_pos"][0] == -0.23 and labs["block_2"]["default_pos"][0] == 0.40


def test_budget_gate_tracks_the_agent_scene(tmp_path):
    first = FakeChat([json.dumps({"answer": {"objects": [{"keep": 0}, {"label": "block_1", "pos": [-0.23, -0.14, 0.7825],
                                                                       "confidence": 0.9}]}})])
    calm = {"objects": [dict(_b1()["objects"][0], flags={}), dict(_b1()["objects"][1], pos=[-0.2, -0.1, 0.7825], confidence=0.85,
                                                                  flags={})], "want_match": {"block": "block_0"}}
    calm["objects"][1]["residual"] = 0.05
    calm["objects"][0]["residual"] = 0.05
    out1 = j1.perceive_impl(_frames(), _calib(), FLANGE, {"left": 1.0}, "Stack.", ["block"], prior=None, b1=_b1(), chat=first,
                            model="fake", gate="budget", refine=False, log_dir=str(tmp_path))
    assert "result=agent" in out1["notes"]
    # second call: B1 re-reads block_1 at its old (wrong) place 50 mm away, a bit less confident, no gripper near it
    out2 = j1.perceive_impl(_frames(), _calib(), FLANGE, {"left": 1.0}, "Stack.", ["block"], prior=out1, b1=calm,
                            chat=FakeChat([]), model="fake", gate="budget", gate_thr=0.99, log_dir=str(tmp_path))
    assert "result=b1_gated" in out2["notes"]
    b = next(o for o in out2["objects"] if o["label"] == "block_1")
    assert b["pos"][0] == -0.23 and "implausible_move" in out2["notes"]  # untouched: the agent's pose stands
    assert len(out2["objects"]) == 2


def test_empty_reply_never_becomes_an_empty_assistant_turn(tmp_path):
    fc = FakeChat(["", json.dumps({"answer": {"objects": [{"keep": 0}]}})])
    out = j1.perceive_impl(_frames(), _calib(), FLANGE, {}, "x", None, b1=_b1(), chat=fc, model="claude-sonnet-5-5",
                           refine=False, keep_unmentioned=False, log_dir=str(tmp_path))
    assert "result=agent" in out["notes"]
    assert all(m["content"] for m in fc.sent[1] if m["role"] == "assistant")
    assert fc.sent[0][0]["role"] == "system" and not any("cache_control" in json.dumps(m) for m in fc.sent[1])


def test_urllib_chat_reads_a_non_streamed_body(monkeypatch):
    import http.server
    import threading

    from dojo.perception.agentic.llm import UrllibChat

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": '{"held": "no"}'}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 7, "completion_tokens": 3}}).encode())

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OMNIROUTE_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("OMNIROUTE_API_KEY", "test-not-a-key")
    monkeypatch.delenv("AGENTIC_ENV_FILE", raising=False)
    r = UrllibChat(timeout_s=10)("claude-sonnet-5-5", [{"role": "user", "content": "x"}], max_tokens=10)
    srv.shutdown()
    assert r.text == '{"held": "no"}' and r.prompt_tokens == 7
