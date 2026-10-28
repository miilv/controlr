"""Stream M's scorer driver (scripts/perception/m95_score.py): root-cause classes, plan events, the summary numbers.
numpy only; the script is loaded by path (scripts/ is not a package)."""

import importlib.util
from pathlib import Path


_p = Path(__file__).resolve().parents[2] / "scripts" / "perception" / "m95_score.py"
_spec = importlib.util.spec_from_file_location("m95_score", _p)
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)


def _ev(kind, step, target, clean=True, required=None, neighbours=(), task="t", arm="left"):
    return {"kind": kind, "step": step, "target": target, "arm": arm, "task": task, "run": "r", "layout_id": 1,
            "success": False, "clean_r": {"5": clean, "10": clean, "15": clean}, "clean_required": clean,
            "clean_target": clean, "required": required or [{"role": "target" if kind == "grasp" else "held",
                                                             "label": target, "ok": True, "cls": "ok"}],
            "neighbours": list(neighbours), "phantoms": [], "reasons": {"10": [] if clean else ["x"]}, "meta": {}}


def _frames(pos_by_step):
    return {str(k): {"objects": [{"label": lab, "pos": list(p), "quat": [1, 0, 0, 0]} for lab, p in objs.items()]}
            for k, objs in pos_by_step.items()}


def test_root_causes():
    frames = _frames({0: {"a": (0, 0, 0.78), "b": (0.2, 0, 0.78)}, 50: {"a": (0, 0, 0.78), "b": (0.26, 0, 0.78)}})
    bad = lambda role, lab, cls: [{"role": role, "label": lab, "ok": False, "cls": cls, "xy_mm": 40.0}]   # noqa: E731
    # held state
    r = _ev("place", 40, "a", False, bad("held", "a", "not_held"))
    assert ms.root_cause(r, ms.failure_rows([r])[0], {}, frames, [], {}, None) == "A not_held"
    r = _ev("place", 40, "a", False, bad("held", "a", "gross"))
    assert ms.root_cause(r, ms.failure_rows([r])[0], {}, frames, [], {}, None) == "B in-hand"
    # touched: a retry after a missed close; after a never-held place
    prev_g = _ev("grasp", 10, "a")
    r = _ev("grasp", 30, "a", False, bad("target", "a", "gross"))
    assert ms.root_cause(r, ms.failure_rows([r])[0], {"a": [prev_g]}, frames, [], {}, None) == "C1 stale retry"
    prev_p = _ev("place", 20, "a", False, bad("held", "a", "not_held"))
    assert ms.root_cause(r, ms.failure_rows([r])[0], {"a": [prev_g, prev_p]}, frames, [], {}, None) == "C2 never-held placed"
    # untouched: never moved = a misread; moved = knocked, pushed on purpose, or carried under another label
    r = _ev("grasp", 30, "a", False, bad("target", "a", "gross"))
    assert ms.root_cause(r, ms.failure_rows([r])[0], {}, frames, [], {}, None) == "D misread"
    r = _ev("grasp", 60, "b", False, bad("target", "b", "gross"))
    iv_push = [{"arm": "left", "grasp": 20, "release": 45, "rode": [], "pushed": ["b"]}]
    assert ms.root_cause(r, ms.failure_rows([r])[0], {}, frames, iv_push, {}, r"^b$") == "G1 intended push"
    assert ms.root_cause(r, ms.failure_rows([r])[0], {}, frames, iv_push, {}, r"^cube") == "G2 knocked"   # not a push target
    assert ms.root_cause(r, ms.failure_rows([r])[0], {}, frames, iv_push, {}, None) == "G2 knocked"
    iv_ride = [{"arm": "left", "grasp": 20, "release": 45, "rode": ["b"], "pushed": []}]
    assert ms.root_cause(r, ms.failure_rows([r])[0], {}, frames, iv_ride, {("left", 20): "a"}, None) == \
        "G carried under another label"
    # clutter and the scripted opponent
    nb = [{"label": "junk", "dist_m": 0.05, "found": False, "clutter": True}]
    r = _ev("grasp", 30, "a", False, neighbours=nb)
    r["required"][0]["ok"] = True
    assert ms.root_cause(r, ms.failure_rows([r])[0], {}, frames, [], {}, None) == "E clutter"
    r = _ev("grasp", 30, "a", False, neighbours=[{"label": "opponent_piece0", "dist_m": 0.05, "found": False}],
            task="play_tic_tac_toe")
    assert ms.root_cause(r, ms.failure_rows([r])[0], {}, frames, [], {}, None) == "F actor"


def test_carried_intervals_tell_riding_from_pushing():
    from dojo.skills.kinematics import X5Kin
    kin = X5Kin()
    q0 = [0.0, 0.6, 0.6, 0.0, 0.0, 0.0]
    q1 = [0.3, 0.6, 0.6, 0.0, 0.0, 0.0]
    a = lambda g, q: {"left_arm_joint_state": q, "left_ee_joint_state": [g], "right_arm_joint_state": q0,   # noqa: E731
                      "right_ee_joint_state": [1.0]}
    acts = [a(1.0, q0), a(0.0, q0), a(0.0, q1), a(1.0, q1)]
    T0, T1 = kin.tcp("left", q0), kin.tcp("left", q1)
    held1 = T1[:3, :3] @ (T0[:3, :3].T @ (T0[:3, 3] - T0[:3, 3])) + T1[:3, 3]
    frames = _frames({2: {"cube": T0[:3, 3], "other": (0.3, 0.3, 0.78)},
                      3: {"cube": held1, "other": (0.33, 0.3, 0.78)}})
    ivs = ms.carried_intervals(frames, acts, ms_track(acts, kin))
    assert ivs == [{"arm": "left", "grasp": 2, "release": 4, "rode": ["cube"], "pushed": ["other"]}]


def ms_track(acts, kin):
    from dojo.perception.grasp_metric import tcp_track
    return tcp_track(acts, kin)


def test_summary_counts_blocked_episodes_and_retries():
    rows = [dict(_ev("grasp", 10, "a"), attempt=1), dict(_ev("grasp", 30, "a", False), attempt=2)]
    plans = [{"task": "t", "clean": True, "blocked": False}, {"task": "t", "clean": False, "blocked": True},
             {"task": "t", "clean": None, "blocked": False}]
    md = ms.summary_table(rows, plans)
    line = next(ln for ln in md.splitlines() if ln.startswith("| t |"))
    assert "50 % (1/2)" in line and "33 % (1/3)" in line                          # executed / blocked
    assert "| 100 % (1/1) | 0 % (0/1) | 50 % |" in line                          # first / retries / episode average


def test_belief_writes_take_every_logged_before_and_after():
    rep = {"events": [
        {"kind": "refine", "label": "a", "for": "pick", "step": 5, "ok": True, "prior7": [0, 0, 0.8, 1, 0, 0, 0],
         "pose7": [0.01, 0, 0.8, 1, 0, 0, 0]},
        {"kind": "refine", "label": "a", "for": "pick", "step": 6, "ok": False, "prior7": [0, 0, 0.8, 1, 0, 0, 0],
         "pose7": [0.05, 0, 0.8, 1, 0, 0, 0]},                                  # not applied: not a write
        {"kind": "post_place", "label": "nut0", "step": 177, "ok": True, "from": [0.34, -0.19, 0.8204],
         "to": [0.34, -0.19, 0.823], "pose7": [0.34, -0.19, 0.823, 1, 0, 0, 0]},
        {"kind": "discrepancy", "label": "b", "step": 9, "before7": [0, 0, 0, 1, 0, 0, 0], "after7": [0, 0, 0.01, 1, 0, 0, 0]}]}
    w = ms.belief_writes(rep)
    assert [(x["writer"], x["step"], x["label"]) for x in w] == [("refine_pick", 5, "a"), ("post_place", 177, "nut0"),
                                                                  ("discrepancy", 9, "b")]
    assert w[1]["after"][2] == 0.823
    # a provisional recentre is judged by its outcome: reverted by the next view = back to the prior
    rep2 = {"events": [
        {"kind": "refine", "label": "a", "for": "pick", "step": 5, "ok": True, "rehover": "recentre",
         "prior7": [0, 0, 0.8, 1, 0, 0, 0], "pose7": [0.03, 0, 0.8, 1, 0, 0, 0]},
        {"kind": "refine", "label": "a", "for": "pick", "step": 8, "ok": False, "why": " (recentre found nothing: "
         "correction reverted)", "prior7": [0.03, 0, 0.8, 1, 0, 0, 0], "pose7": [0.03, 0, 0.8, 1, 0, 0, 0]}]}
    w2 = ms.belief_writes(rep2)
    assert w2[0]["writer"] == "refine_pick+confirm" and w2[0]["after"] == [0, 0, 0.8]


def test_consume_verdict_checks_only_the_axes_the_action_needs():
    assert ms.consume_verdict({"xy_mm": 3.0, "z_mm": 9.0, "yaw_deg": None}, {"xy_mm": 4.0, "z_mm": None, "yaw_deg": None})
    assert not ms.consume_verdict({"xy_mm": 5.0, "z_mm": 1.0}, {"xy_mm": 4.0, "z_mm": 2.0})
    assert not ms.consume_verdict({"xy_mm": 1.0, "yaw_deg": None}, {"xy_mm": 4.0, "yaw_deg": 5.0})   # not measured
    assert ms.consume_verdict({}, {"xy_mm": None})
