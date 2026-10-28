"""Legality, "For Ilia" items of robot-harness's verification notes (G1F 2026-10-28, P1S 2026-10-27): an unknown stays
unknown. Each test drives the real function with fakes: no Isaac, no GPU, no LLM (FakeChat), no catalogue.

- P1S I1-3: the agent's held_look without a yes / no (and B1 failing in mode b1) is ``held: None`` with ``error``, at
  the top level and per arm; the J1 schema refuses it (the server's B1-failure path), the runtime keeps the grasp.
- G1F-4 / G2-3: a uniform wrist frame or a silhouette fit with tied scores is no measurement.
- G1F-11 / G2-10: a held answer for the other arm never stands for the asked arm.
- G1F-1 / G1-1: ``merge_prior`` never invents a confidence (nor a pose) for a prior entry without one.
- P1S I1-5: an object without a measured quat never becomes an identity-oriented planning row.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dojo.bridge.perception import ROTATION_FREE_CATEGORIES, ServerPerception, record_from_perceived
from dojo.perception import j1_adapter as J
from dojo.perception.agentic import j1
from dojo.perception.agentic.llm import FakeChat


def _schema():
    sys.path.insert(0, str(Path(j1.__file__).resolve().parent / "vendor"))
    try:
        import j1_schema  # the strict server's own validator, vendored byte-identical
    finally:
        sys.path.pop(0)
    return j1_schema


def _calib():
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    T_h = np.eye(4)
    T_h[:3, :3] = [[1, 0, 0], [0, -c, s], [0, -s, -c]]
    T_h[:3, 3] = [0, -0.41, 1.308]
    T_w = np.eye(4)
    T_w[:3, :3] = np.c_[[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    T_w[:3, 3] = [0.0, -0.1, 1.015]
    K_h = [[288.1325, 0, 320], [0, 288.1325, 240], [0, 0, 1]]
    K_w = [[397.04, 0, 320], [0, 397.04, 240], [0, 0, 1]]
    return {"cam_head": {"K": K_h, "T_world_cam": T_h.tolist()}, "cam_left_wrist": {"K": K_w, "T_world_cam": T_w.tolist()}}


FLANGE = {"left": [-0.3, -0.2, 0.9, 1, 0, 0, 0], "right": [0.3, -0.2, 0.9, 1, 0, 0, 0]}


def _frames():
    img = np.full((480, 640, 3), 90, np.uint8)
    img[230:250, 310:330] = (200, 30, 30)
    return {"cam_head": img, "cam_left_wrist": img.copy()}


# ------------------------------------------------------------------ P1S I1-3: no answer is unknown, not "no"

def test_agent_without_a_yes_no_is_unknown_and_the_schema_refuses_it(tmp_path):
    out = j1.held_look_impl(_frames(), _calib(), FLANGE, {"left": 0.0}, arm="left", chat=FakeChat(["maybe", "dunno"]),
                            model="fake", held_mode="agent", log_dir=str(tmp_path))
    assert out["held"] is None and out["confidence"] == 0.0 and out["error"] == "no answer"
    assert out["left"]["held"] is None and out["left"]["error"] == "no answer"
    clean, rep = _schema().validate_held(out, "left")
    assert clean is None and "held must be" in rep["error"]           # never a "no": the B1-failure path
    # a real answer still validates
    yes = j1.held_look_impl(_frames(), _calib(), FLANGE, {"left": 0.0}, arm="left", chat=FakeChat(['{"held": "no"}']),
                            model="fake", held_mode="agent", log_dir=str(tmp_path))
    clean, _ = _schema().validate_held(yes, "left")
    assert clean["held"] is False and clean["confidence"] == 0.9


def test_b1_error_in_mode_b1_is_unknown_not_an_error_key(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise FileNotFoundError("X5A.urdf")
    monkeypatch.setattr(j1._b1, "_held_look_impl", broken)
    out = j1.held_look_impl(_frames(), _calib(), FLANGE, {"left": 0.0}, arm="left", chat=FakeChat([]), model="fake",
                            held_mode="b1", log_dir=str(tmp_path))
    assert "_error" not in out                                        # (the worker client would raise on it)
    assert out["held"] is None and out["error"].startswith("B1 error: FileNotFoundError")
    assert out["left"]["held"] is None and out["left"]["error"] == out["error"]
    assert _schema().validate_held(out, "left")[0] is None


# ------------------------------------------------------------------ G1F-11 / G2-10: the asked arm's own answer only

def test_b1_summary_never_answers_for_the_other_arm():
    right_yes = {"right": {"held": True, "confidence": 1.0, "evidence": "opening"}}
    s = J.held_summary(right_yes, "left", {"left": "no wrist frame"})
    assert s["held"] is None and s["confidence"] == 0.0 and "left arm: no wrist frame" in s["error"]
    assert s["left"] == {"held": None, "confidence": 0.0, "error": "no answer for the left arm: no wrist frame"}
    assert s["right"]["held"] is True                                 # the other arm's verdict is still reported
    assert J.held_summary(right_yes, "right")["held"] is True
    both = {"left": {"held": False, "confidence": 0.7}, "right": {"held": True, "confidence": 0.9}}
    assert (J.held_summary(both, "left")["held"], J.held_summary(both, "left")["confidence"]) == (False, 0.7)
    assert (J.held_summary(both)["held"], J.held_summary(both)["confidence"]) == (True, 0.7)    # no arm asked: as before
    none = J.held_summary({}, None, {"left": "gripper commanded open: no grasp to check"})
    assert none["held"] is None and "gripper commanded open" in none["error"]
    assert _schema().validate_held(s, "left")[0] is None


@pytest.mark.parametrize("polarity", [True, False])
def test_agentic_gate_does_not_take_the_other_arms_b1_answer(tmp_path, monkeypatch, polarity):
    # the shape before the fix: B1 answered for the right arm only, its top level the right arm's verdict
    old = {"right": {"held": polarity, "confidence": 1.0, "evidence": "opening"}, "held": polarity, "confidence": 1.0}
    monkeypatch.setattr(j1._b1, "_held_look_impl", lambda *a, **k: dict(old))
    out = j1.held_look_impl(_frames(), _calib(), FLANGE, {"left": 0.0, "right": 0.0}, arm="left", chat=FakeChat([]),
                            model="fake", held_mode="b1", log_dir=str(tmp_path))
    assert out["held"] is None and out["left"]["held"] is None and "left" in out["error"]
    fc = FakeChat(['{"held": "yes"}'])                                # b1_then_agent: escalates to the agent instead
    out = j1.held_look_impl(_frames(), _calib(), FLANGE, {"left": 0.0, "right": 0.0}, arm="left", chat=fc, model="fake",
                            held_mode="b1_then_agent", log_dir=str(tmp_path))
    assert len(fc.sent) == 1 and "left gripper" in fc.sent[0][1]["content"][0]["text"]
    assert out["left"]["held"] is True and out["confidence"] == 0.9
    # a confident answer for the asked arm still stands without an agent call
    monkeypatch.setattr(j1._b1, "_held_look_impl", lambda *a, **k: {"left": {"held": polarity, "confidence": 1.0},
                                                                   "held": polarity, "confidence": 1.0})
    fc = FakeChat([])
    out = j1.held_look_impl(_frames(), _calib(), FLANGE, {"left": 0.0}, arm="left", chat=fc, model="fake",
                            held_mode="b1_then_agent", log_dir=str(tmp_path))
    assert fc.sent == [] and out["held"] is polarity and out["left"]["held"] is polarity


def test_runtime_held_look_answers_for_the_requested_arm_or_unknown(monkeypatch):
    from dojo.perception.agentic import record
    from dojo.runtime.perception_service import _Perception, encode_frame

    monkeypatch.setattr(record, "_calib_from_state", lambda st: _calib())
    st = {"left_ee_pose": FLANGE["left"], "right_ee_pose": FLANGE["right"], "left_ee_joint_state": [0.0],
          "right_ee_joint_state": [0.0], "left_arm_joint_state": [0.0] * 6, "right_arm_joint_state": [0.0] * 6}
    frame = encode_frame({"images": {"cam_head": _frames()["cam_head"]}, "state": st})
    reply = {"right": {"held": True, "confidence": 1.0, "evidence": "opening"}, "held": True, "confidence": 1.0}
    monkeypatch.setattr(j1, "held_look_impl", lambda *a, **k: dict(reply))
    r = _Perception().held_look(frame, "left", mode="b1")
    assert r["held"] is None and r["confidence"] == 0.0 and "left" in r["error"]
    r = _Perception().held_look(frame, "right", mode="b1")
    assert r["held"] is True and r["confidence"] == 1.0 and "error" not in r
    monkeypatch.setattr(j1, "held_look_impl", lambda *a, **k: {"held": None, "confidence": 0.0, "error": "no answer",
                                                              "left": {"held": None, "confidence": 0.0, "error": "no answer"}})
    r = _Perception().held_look(frame, "left", mode="agent")
    assert r["held"] is None and r["error"] == "no answer"


# ------------------------------------------------------------------ G1F-4 / G2-3: no fit is no measurement

def test_tied_or_flat_silhouette_scores_are_rejected():
    openings = [round(0.05 * i, 2) for i in range(21)]
    assert J.opening_fit_rejected([0.0] * 21, openings).startswith("no fit")          # a uniform frame
    peaked = [max(0.0, 0.8 - 3 * abs(o - 0.3)) for o in openings]
    assert J.opening_fit_rejected(peaked, openings) is None                           # one clear opening
    twin = [max(0.0, 0.8 - 3 * abs(o - 0.3)) + max(0.0, 0.79 - 3 * abs(o - 0.8)) for o in openings]
    assert J.opening_fit_rejected(twin, openings).startswith("tie")                   # two far openings explain it
    near = [0.8 if o in (0.3, 0.35) else 0.1 for o in openings]
    assert J.opening_fit_rejected(near, openings) is None                             # neighbours are one fit


class _Fingers:
    """Two boxes on either side of the tool axis, 0.01 + 0.03 * opening m from it, 0.12 m below a wrist camera
    looking straight down: FK and meshes for measure_opening / held_look without the X5 catalogue."""

    def fk_world(self, arm, q6, opening=0.0):
        d = 0.01 + 0.03 * float(opening)
        T7, T8, T6 = np.eye(4), np.eye(4), np.eye(4)
        T7[:3, 3], T8[:3, 3], T6[:3, 3] = [-d, -0.1, 0.895], [d, -0.1, 0.895], [0.0, -0.1, 0.99]
        return {"link6": T6, "link7": T7, "link8": T8}

    @staticmethod
    def meshes():
        h = np.array([0.003, 0.008, 0.02])
        V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * h
        F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        return {"link7": (V, F), "link8": (V, F)}


def _wrist_cam():
    from dojo.perception.cameras import Camera
    T = np.eye(4)
    T[:3, :3] = np.c_[[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    T[:3, 3] = [0.0, -0.1, 1.015]
    return Camera("cam_left_wrist", np.array([[397.04, 0, 320], [0, 397.04, 240], [0, 0, 1]], float), T)


def test_measure_opening_rejects_a_uniform_frame_and_keeps_a_visible_jaw():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception.render import silhouette

    cam, m, meshes = _wrist_cam(), _Fingers(), _Fingers.meshes()
    for v in (0, 255):                                                # G2-3's probe: black and white frames
        r = J.measure_opening(cam, "left", np.zeros(6), np.full((480, 640, 3), v, np.uint8), meshes, m)
        assert r["rejected"].startswith("no fit")
    img = np.full((480, 640, 3), 200, np.uint8)                       # bright table, dark fingers at opening 0.6
    P, F = m.fk_world("left", None, 0.6), np.zeros((480, 640), np.uint8)
    for ln in ("link7", "link8"):
        V, Fc = meshes[ln]
        silhouette(cam, V @ P[ln][:3, :3].T + P[ln][:3, 3], Fc, out=F)
    img[F > 0] = 25
    r = J.measure_opening(cam, "left", np.zeros(6), img, meshes, m)
    assert "rejected" not in r and abs(r["opening"] - 0.6) < 1e-9 and r["score"] > 0.5


def test_held_look_on_a_uniform_wrist_frame_is_unknown(monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import render, robot_model

    monkeypatch.setattr(robot_model.X5Model, "default", classmethod(lambda cls: _Fingers()))
    monkeypatch.setattr(render, "robot_link_meshes", lambda: _Fingers.meshes())
    cal = {"cam_left_wrist": {"K": _wrist_cam().K.tolist(), "T_world_cam": _wrist_cam().T_wc.tolist()}}
    joints = {"left": [0.0] * 6, "right": [0.0] * 6}
    for v in (0, 255):
        frames = {"cam_left_wrist": np.full((480, 640, 3), v, np.uint8)}
        out = J._held_look_impl(frames, cal, FLANGE, {"left": 0.0, "right": 1.0}, joints=joints, arm="left")
        assert out["held"] is None and out["left"]["held"] is None and "uniform" in out["left"]["error"]
        assert _schema().validate_held(out, "left")[0] is None        # the server's B1-failure path, not "no" at 1.0
    from dojo.perception.render import silhouette                     # visibly open jaws (0.6 vs a close): measured
    cam, m, meshes = _wrist_cam(), _Fingers(), _Fingers.meshes()
    img, P, F = np.full((480, 640, 3), 200, np.uint8), m.fk_world("left", None, 0.6), np.zeros((480, 640), np.uint8)
    for ln in ("link7", "link8"):
        silhouette(cam, meshes[ln][0] @ P[ln][:3, :3].T + P[ln][:3, 3], meshes[ln][1], out=F)
    img[F > 0] = 25
    out = J._held_look_impl({"cam_left_wrist": img}, cal, FLANGE, {"left": 0.0}, joints=joints, arm="left")
    assert out["left"]["evidence"] == "opening" and out["held"] is True and abs(out["left"]["opening"]["measured"] - 0.6) < 1e-9


# ------------------------------------------------------------------ G1F-1 / G1-1: no invented prior confidence

def test_merge_prior_never_invents_a_confidence_or_a_pose():
    pad = {"label": "target_t", "category": "t_cushion", "asset": "Geometry/t_cushion/00000",
           "pos": [-0.107, -0.224, 0.765], "quat": [1, 0, 0, 0]}                          # no confidence
    block = {"label": "block_1", "category": "block", "asset": "Rigid/block/00006", "pos": [0.216, -0.078, 0.78],
             "confidence": 0.95}                                                           # no quat
    agent_pt = {"label": "block_2", "category": "block", "pos": [0.3, -0.06, 0.78], "confidence": "high"}
    cur = [{"label": "block_1", "category": "block", "asset": "Rigid/block/00006", "pos": [0.219, -0.059, 0.78],
            "quat": [1, 0, 0, 0], "confidence": 0.20}]
    out, last_seen = J.merge_prior([dict(o) for o in cur], [pad, block, agent_pt])
    by = {o["label"]: o for o in out}
    assert "target_t" not in by                                       # not carried at an invented 0.45
    assert by["block_1"]["pos"] == cur[0]["pos"] and by["block_1"]["confidence"] == 0.20   # the reading stands
    assert by["block_1"].get("source") != "prior" and "new_reading" not in by["block_1"]
    ls = {o["label"]: o for o in last_seen}
    assert set(ls) == {"target_t", "block_2"} and "confidence" not in ls["target_t"]
    assert ls["target_t"]["prior_unusable"] == "no valid confidence" and ls["block_2"]["source"] == "prior"
    clean, rep = _schema().validate_perceive({"objects": out, "last_seen": last_seen, "regions": [], "needs_look": []})
    assert rep["ok"] and [o["label"] for o in clean["objects"]] == ["block_1"] and len(clean["last_seen"]) == 2
    assert J._carry_static([], [pad]) == []                           # the old carry path too
    assert J._carry_static([], [dict(pad, confidence=0.9)])[0]["confidence"] == 0.81
    # a prior with its own confidence and pose still works as before
    out, _ = J.merge_prior([dict(o) for o in cur], [dict(block, quat=[1, 0, 0, 0])])
    assert out[0]["pos"] == block["pos"] and out[0]["source"] == "prior" and out[0]["confidence"] == 0.855
    assert J.prior_unusable({"confidence": True, "pos": [0, 0, 1], "quat": [1, 0, 0, 0]}) == "no valid confidence"
    assert J.prior_unusable({"confidence": np.float32(0.5), "pos": np.zeros(3), "quat": np.r_[1.0, 0, 0, 0]}) is None
    assert J.prior_unusable({"confidence": 0.5, "pos": [0, 0, 1], "quat": [0, 0, 0, 0]}) == "no valid quat"


# ------------------------------------------------------------------ P1S I1-5: no invented orientation

def test_an_object_without_a_quat_is_not_a_planning_row():
    per = [{"label": "block_0", "category": "block", "asset": "Rigid/block/00005", "pos": [0.1, -0.1, 0.78],
            "confidence": 0.8, "source": "head"},                                          # no quat
           {"label": "block_1", "category": "block", "asset": "Rigid/block/00006", "pos": [0.2, -0.1, 0.78],
            "quat": [0.0, 0.0, 0.0, 0.0], "confidence": 0.8},                               # not a rotation
           {"label": "ball_0", "category": "ball", "asset": "Rigid/ball/00000", "pos": [0.0, -0.2, 0.79],
            "confidence": 0.9},                                                             # rotation-free
           {"label": "block_2", "category": "block", "asset": "Rigid/block/00007", "pos": [-0.1, -0.1, 0.78],
            "quat": [0, 0, 0, 1.05], "confidence": 0.7}]                                    # measured (normalised)
    rec = record_from_perceived(per, [{"reason": "not_found", "label": "block_0"}])
    rows = {r["label"]: r for lst in rec["Rigid"].values() for r in lst}
    assert set(rows) == {"ball_0", "block_2"}
    assert "ball" in ROTATION_FREE_CATEGORIES and rows["ball_0"]["default_ori"] == [1.0, 0.0, 0.0, 0.0]
    assert rows["ball_0"]["_perception"]["orientation"] == "rotation_free"
    assert rows["block_2"]["default_ori"] == [0.0, 0.0, 0.0, 1.0] and rows["block_2"]["_perception"]["orientation_known"]
    assert rec["_orientation_unknown"] == ["block_0", "block_1"]
    assert rec["_not_found"] == [{"label": "block_0", "candidates": ["Rigid/block/00005"], "why": "orientation unknown"},
                                 {"label": "block_1", "candidates": ["Rigid/block/00006"], "why": "orientation unknown"}]
    assert {u["label"]: u["pos"] for u in rec["_unusable"]} == {"block_0": [0.1, -0.1, 0.78], "block_1": [0.2, -0.1, 0.78]}

    class Backend:                                                    # the server's perceive verb
        def perceive(self, want=None, prior_n=None):
            return {"perceived": per[:1], "needs_look": [], "n": 1}
    rec = ServerPerception(Backend()).estimate([{}], "stack_blocks", 0)
    assert "Rigid" not in rec and rec["_not_found"][0]["label"] == "block_0"
