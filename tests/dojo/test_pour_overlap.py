"""Stream H: ``ported_pour_pace`` "overlap": the other arm grasps and lifts the next pour's bottle during each pour
(``pour_bottle_lean``'s ``companion`` / ``pre_held``), through the adapter's two-arm move verb, which exists for that
pace only. The liquid is screened in G3; here: the verb's gate, the end-to-end overlap on the kinematic env, the
orchestration. CPU only."""

from __future__ import annotations

import json

import numpy as np

from dojo.skills import planner, pour_order
from dojo.skills.ported import harness_robot as hr
from dojo.skills.ported import ported

from test_port_adapter import Z, gt_skills, rows_bottle


def _two_pours():
    return [dict(label="bottle_a", category="wuliangye", category_idx=0, pos=[0.15, -0.02, Z + 0.11], quat=[1, 0, 0, 0],
                 half=[0.04, 0.04, 0.11]),
            dict(label="bottle_b", category="wuliangye", category_idx=0, pos=[-0.15, -0.02, Z + 0.11], quat=[1, 0, 0, 0],
                 half=[0.04, 0.04, 0.11]),
            dict(label="cup_a", category="goblet", category_idx=6, pos=[0.05, -0.17, Z + 0.05], quat=[1, 0, 0, 0],
                 half=[0.04, 0.04, 0.05]),
            dict(label="cup_b", category="goblet", category_idx=6, pos=[-0.07, -0.17, Z + 0.05], quat=[1, 0, 0, 0],
                 half=[0.04, 0.04, 0.05])]


def test_the_two_arm_verb_exists_only_when_switched_on():
    _, sk = gt_skills(rows_bottle(), "Pour the liquid from the bottle into the cup.")
    off = hr.HarnessRobot(sk)
    moves = {a: {"pos": [x, -0.1, 0.95], "yaw_deg": 0, "down": True} for a, x in (("left", -0.2), ("right", 0.2))}
    rep = off.call("move_pair", moves=moves)
    assert rep["ok"] is False and "unknown command" in rep["error"] and sk.ex.steps_used == 0
    on = hr.HarnessRobot(sk, pair_moves=True)
    q0 = {a: np.asarray(sk.ex.q[a], float).copy() for a in ("left", "right")}
    rep = on.call("move_pair", moves=moves)
    assert rep["ok"] and rep["moved"] and set(rep["arms"]) == {"left", "right"} and rep["steps"] > 0
    assert all(np.abs(np.asarray(sk.ex.q[a]) - q0[a]).max() > 1e-3 for a in ("left", "right"))   # both, in one move


def test_only_the_overlap_pace_builds_the_verb(monkeypatch):
    seen = []
    init = hr.HarnessRobot.__init__

    def spy(self, *a, **k):
        seen.append(bool(k.get("pair_moves", False)))
        init(self, *a, **k)
    monkeypatch.setattr(hr.HarnessRobot, "__init__", spy)
    for pace, skill in (("v1", "pour_bottle"), ("lean", "pour_bottle_lean"), ("overlap", "pour_bottle"),
                        ("overlap", "pour_bottle_lean")):
        _, sk = gt_skills(rows_bottle(), "Pour the liquid from the bottle into the cup.", step_lim=50)
        sk.ported_pour_pace = pace
        ported(sk, skill, {})
    assert seen == [False, False, False, True]


def test_overlap_grasps_the_next_bottle_during_the_pour():
    steps = {}
    for pace in ("v1", "overlap"):
        env, sk = gt_skills(_two_pours(), "Pour.", step_lim=1200)
        sk.ported_pour_pace, sk.ported_pour_aim = pace, "b10"
        r = pour_order.pour_in_order(sk, [["bottle_a", "cup_a"], ["bottle_b", "cup_b"]])
        assert r["ok"]
        runs = [json.loads(x["details"]) for x in sk.ported_runs[-2:]]
        assert all(d["pour_stop"] == "profile end" for d in runs)
        steps[pace] = env.steps_used
        if pace == "overlap":
            comp = runs[0]["companion"]
            assert comp["ok"] and comp["arm"] != runs[0]["plan"]["arm"]
            kinds = [next(iter(c)) for c in comp["log"]]          # pre, in, close, lift, then the carry over its bowl
            assert kinds == ["move", "move", "close", "move", "move"] and not any(c.get("alone") for c in comp["log"])
            assert comp["carried"]
            assert runs[1]["plan"]["arm"] == comp["arm"] and runs[1]["log"][0]["grasp"]["pre"] == {}  # no second grasp
            assert runs[1]["pour"][0]["phi"] != 0                                  # it starts over its bowl
    assert steps["overlap"] <= steps["v1"] - 140                     # kinematic env: 753 -> 584


def test_a_companion_only_for_a_side_bottle_on_its_own_side():
    from dojo.skills.ported.skills.pour_bottle_lean import skill_robodojo as lean
    objs = {"b": {"pos": [0.03, 0.0, 0.8]}, "w": {"pos": [0.0, -0.15, 0.78]}}

    def side_ok(x, arm):
        objs["b"]["pos"][0] = x
        try:
            lean.companion_plan(None, None, None, objs, "", "b", "w", arm, None, 0.72, 0.1, 0.04, (0,), (0,), "auto",
                                (0,), 0.012, 0.012, 0.0, 0.015, 0.55)
        except Exception:            # past the side rule: it went on to plan (no robot here)
            return True
        return False
    assert not side_ok(0.03, "left") and not side_ok(0.03, "right") and side_ok(0.2, "right") and not side_ok(0.2, "left")
    assert side_ok(-0.2, "left") and not side_ok(-0.2, "right")


def test_pour_order_hands_the_held_bottle_on(monkeypatch):
    calls = []

    def fake(sk, skill, args, reserve=None):
        calls.append((skill, dict(args)))
        return {"ok": True, "status": "ok", "verified": True, "steps": 10,
                "companion": {"arm": "left", "pitch": 0, "yaw": 0, "rise_m": 0.06, "tilt": 1.0, "ok": True}}
    import dojo.skills.ported as ported_mod
    monkeypatch.setattr(ported_mod, "ported", fake)

    class Sk:
        ported_pour_aim, ported_pour_pace = "b10", "overlap"

        def refresh(self):
            return None

        def log(self, m):
            pass
    assert pour_order.pour_in_order(Sk(), [["b0", "w0"], ["b1", "w1"], ["b2", "w2"]])["ok"]
    assert [c[1].get("companion") for c in calls] == [["@b1", "@w1"], ["@b2", "@w2"], None]
    assert ["pre_held" in c[1] for c in calls] == [False, True, True]
    assert [c[1].get("home", True) for c in calls] == [False, False, True]
    assert calls[0][1]["tilt_speed"] == planner.POUR_PACE["overlap"]["tilt_speed"]


def test_alternate_arms():
    assert pour_order.alternate_arms([0.0, 0.2, -0.2]) == ["left", "right", "left"]
    assert pour_order.alternate_arms([0.2, -0.2, 0.0]) == ["right", "left", "right"]
    assert pour_order.alternate_arms([0.2, 0.0, -0.2]) in (["right", "left", "left"], ["right", "right", "left"])
    assert pour_order.alternate_arms([0.2, 0.0, -0.2]) == ["right", "right", "left"]      # tie: the vendored choice


def test_only_the_last_pours_untilt_spends_the_reserve(monkeypatch):
    seen = []
    op_move = hr.HarnessRobot.op_move

    def spy(self, a):
        seen.append(int(a.get("reserve", 0)))
        return op_move(self, a)
    monkeypatch.setattr(hr.HarnessRobot, "op_move", spy)
    runs = {}
    for res in (None, 0):
        seen.clear()
        _, sk = gt_skills(rows_bottle(), "Pour the liquid from the bottle into the cup.", step_lim=800)
        sk.ported_pour_pace = "overlap"
        args = dict(planner.POUR_PACE["overlap"], **({} if res is None else {"untilt_reserve": res}))
        ported(sk, "pour_bottle_lean", args)
        runs[res] = (list(seen), json.loads(sk.ported_runs[-1]["details"]))
    (r_none, d_none), (r_zero, d_zero) = runs[None], runs[0]
    assert len(r_none) == len(r_zero)                                   # the same moves either way
    n_untilt = len(d_zero["return"]["untilt"])
    diff = [i for i, (a, b) in enumerate(zip(r_none, r_zero)) if a != b]
    assert len(diff) == n_untilt and all(r_zero[i] == 0 and r_none[i] > 0 for i in diff)   # only the untilt moves


def test_pour_order_spends_the_reserve_on_the_last_pour_only(monkeypatch):
    calls = []

    def fake(sk, skill, args, reserve=None):
        calls.append(dict(args))
        return {"ok": True, "status": "ok", "verified": True, "steps": 10}
    import dojo.skills.ported as ported_mod
    monkeypatch.setattr(ported_mod, "ported", fake)

    class Sk:
        ported_pour_aim, ported_pour_pace = "b10", "overlap"

        def refresh(self):
            return None

        def log(self, m):
            pass
    pour_order.pour_in_order(Sk(), [["b0", "w0"], ["b1", "w1"], ["b2", "w2"]])
    assert [c.get("untilt_reserve") for c in calls] == [None, None, 0]
    calls.clear()
    Sk.ported_pour_pace = "lean"
    pour_order.pour_in_order(Sk(), [["b0", "w0"], ["b1", "w1"]])
    assert all("untilt_reserve" not in c for c in calls)                 # overlap only
