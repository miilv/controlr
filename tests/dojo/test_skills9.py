"""skills9: a refined pick keeps the grasp its place was planned with; the mode-B held-pose protocol at the pre-insert
hover (read, look / show, servo on the measured transform, safe fallback); the G3 held-belief / held-pose test
provider (CPU only)."""

from __future__ import annotations

import numpy as np

from dojo.skills import grasps as G
from dojo.skills.env import KinematicEnv
from dojo.skills.g3 import OffsetBeliefProvider
from dojo.skills.kinematics import TABLE_Z
from dojo.skills.skills import GTProvider, PickPlan, PlacePlan, Skills

HALF = [0.03, 0.012, 0.01]


def _scene(offset=(0.003, -0.002), refine_fixes=True):
    """A block whose believed pose is ``offset`` off until the pick's refine re-measures it (then the truth)."""
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.22, -0.15, TABLE_Z + 0.01],
                         "quat": [1, 0, 0, 0], "half": HALF}])
    prov = OffsetBeliefProvider(GTProvider(env, None), {"b": offset})

    def refine(sk_, label, arm, kind):
        if kind == "pick" and refine_fixes:
            prov.offsets.pop(label, None)
    sk = Skills(env, prov, remeasure=False, refine=refine)
    return env, prov, sk


def _tilted(sk):
    """Only 25-deg tilted box grasps (the transfer's pairing; the default pick would come straight down)."""
    return lambda a: [g for g in G.box_grasps(sk.world.obj("b"), sk.kin, a, tilts=(25.0,)) if -g.T[2, 0] < 0.95]


def _target(sk):
    T = np.asarray(sk.world.obj("b").T, float).copy()
    T[:2, 3] = (-0.17, -0.22)
    return T


def test_a_refined_pick_keeps_the_planned_grasp_in_the_object_frame():
    env, prov, sk = _scene()
    pp, pl = sk.plan_transfer("b", [_target(sk)], grasps=_tilted(sk))
    assert callable(pp.repair)
    default = sk.plan_pick("b", pp.arm)
    assert abs(float(default.grasp.T[2, 0])) > 0.99          # the method's own grasp: straight down (the old re-plan)
    r = sk.run_transfer(pp, pl)
    assert r.ok, r.note
    ex = sk.last_pick
    assert ex is not pp and np.allclose(ex.T_tcp_obj, pp.T_tcp_obj, atol=1e-9)   # same grasp in the object frame
    assert -float(ex.grasp.T[2, 0]) < 0.95                    # still the tilted pinch
    # ... at the refined (true) pose: the grasp moved with the object by the 3 mm belief error
    assert np.allclose(ex.grasp.T[:2, 3] - pp.grasp.T[:2, 3], [-0.003, 0.002], atol=1e-6)
    assert np.linalg.norm(sk.refresh().obj("b").pos[:2] - [-0.17, -0.22]) < 0.002


def test_an_unmoved_refine_runs_the_original_descent():
    env, prov, sk = _scene(offset=(0.0, 0.0), refine_fixes=False)
    pp, pl = sk.plan_transfer("b", [_target(sk)], grasps=_tilted(sk))
    assert sk.run_pick(pp).ok
    assert sk.last_pick is pp


def test_an_infeasible_grasp_re_pairs_pick_and_place(monkeypatch):
    env, prov, sk = _scene()
    pp, pl = sk.plan_transfer("b", [_target(sk)], grasps=_tilted(sk))
    monkeypatch.setattr(sk, "_same_grasp_pick", lambda p, kw: None)
    calls = []
    orig = pp.repair

    def repair(q_s):
        calls.append(q_s)
        return orig(q_s)
    pp.repair = repair
    r = sk.run_transfer(pp, pl)
    assert calls and r.ok, r.note
    ex = sk.last_pick
    assert ex is not pp and isinstance(ex.paired_place, PlacePlan)
    assert np.allclose(calls[0][pp.arm], ex.plan.q0)           # re-planned from the hover
    assert np.linalg.norm(sk.refresh().obj("b").pos[:2] - [-0.17, -0.22]) < 0.002


def test_a_plain_pick_falls_back_to_the_method_grasps(monkeypatch):
    env, prov, sk = _scene()
    pp = sk.plan_pick("b", "left", grasps=_tilted(sk)("left"))
    assert isinstance(pp, PickPlan) and pp.repair is None
    monkeypatch.setattr(sk, "_same_grasp_pick", lambda p, kw: None)
    assert sk.run_pick(pp).ok
    assert abs(float(sk.last_pick.grasp.T[2, 0])) > 0.99       # no place to keep: the method's best grasp, as before


def test_a_pick_without_refine_is_unchanged():
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.22, -0.15, TABLE_Z + 0.01],
                         "quat": [1, 0, 0, 0], "half": HALF}])
    sk = Skills(env, GTProvider(env, None))
    pp, pl = sk.plan_transfer("b", [_target(sk)], grasps=_tilted(sk))
    assert sk.run_pick(pp).ok and sk.last_pick is pp


# ---------------------------------------------------------------------------------------------- (2) held-pose protocol
AMB = {"ok": 0, "why": "ambiguous"}
GOOD = {"ok": 1, "why": "", "err": [0.0] * 6, "sig": [0.1, 0.1, 0.1, 0.1, 0.3]}


def _held_scene(table, offset=(0.003, 0.0), mode="noise"):
    from dojo.skills.heldsim import HeldBeliefProvider
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.22, -0.15, TABLE_Z + 0.01],
                         "quat": [1, 0, 0, 0], "half": HALF}])
    prov = HeldBeliefProvider(OffsetBeliefProvider(GTProvider(env, None), {"b": offset}), env, mode=mode, table=table)
    sk = Skills(env, prov, remeasure=False, refine=lambda *a: None)
    assert sk.run_pick(sk.plan_pick("b", "left")).ok
    T = np.asarray(sk.world.obj("b").T, float).copy()
    T[:3, 3] = (-0.18, -0.22, TABLE_Z + 0.01)
    return env, prov, sk, T


def _true_pos(env):
    return np.asarray(next(o for o in env.objects if o["label"] == "b")["pos"], float)


def test_the_belief_attaches_at_the_close_with_the_believed_pose():
    env, prov, sk, T = _held_scene({}, mode=None)
    assert getattr(prov, "held_pose", None) is None             # belief only: no read
    assert "left" in prov.att
    # the in-hand transform the skills hold is the belief's (3 mm off the truth along world x)
    T_true = prov._true_T_rel("left", "b")
    d = sk.kin.tcp("left", sk.ex.q["left"])[:3, :3] @ (sk.held["left"].T_tcp_obj[:3, 3] - T_true[:3, 3])
    assert np.allclose(d[:2], [0.003, 0.0], atol=3e-4)


def test_an_ambiguous_hover_read_looks_then_inserts_with_the_measured_transform():
    env, prov, sk, T = _held_scene({"block": {"hover": [AMB], "look": [GOOD], "show": [GOOD]}})
    q_obs = sk.ex.q["right"].copy()
    pl = sk.plan_place("left", T, servo=True, pre_h=0.04)
    r = sk.run_place(pl)
    views = [e["view"] for e in sk.held_events]
    assert views[0] == "hover" and not sk.held_events[0]["ok"]
    assert views[1] in ("look", "show") and sk.held_events[1]["adopted"] and sk.held_events[1]["steps"] > 0
    assert "held read adopted" in r.note and "servo" in r.note
    assert np.abs(sk.ex.q["right"] - q_obs).max() < 1e-9         # the observer is back where it was
    assert np.linalg.norm(_true_pos(env)[:2] - T[:2, 3]) < 0.0005   # the true object at the target (belief was 3 mm off)


def test_a_good_hover_read_is_used_without_a_look():
    env, prov, sk, T = _held_scene({"block": {"hover": [GOOD], "look": [GOOD], "show": [GOOD]}})
    r = sk.run_place(sk.plan_place("left", T, servo=True, pre_h=0.04))
    assert [e["view"] for e in sk.held_events] == ["hover"] and sk.held_events[0]["adopted"]
    assert np.linalg.norm(_true_pos(env)[:2] - T[:2, 3]) < 0.0005, r.note


def test_held_tol_zero_looks_during_the_approach():
    table = {"block": {"hover": [AMB], "look": [GOOD], "show": [GOOD]}}
    env, prov, sk, T = _held_scene(table)
    n0 = sk.ex.steps_used
    sk.run_place(sk.plan_place("left", T, servo=True, pre_h=0.04))           # hover read, then a look
    seq = sk.ex.steps_used - n0
    env2, prov2, sk2, T2 = _held_scene(table)
    q_obs = sk2.ex.q["right"].copy()
    n0 = sk2.ex.steps_used
    r = sk2.run_place(sk2.plan_place("left", T2, servo=True, pre_h=0.04, held_tol=0.0))
    ev = sk2.held_events
    assert [e["view"] for e in ev] == ["look"] and ev[0]["adopted"], ev    # no hover read: the observer was there
    assert sk2.ex.steps_used - n0 < seq - 5, (sk2.ex.steps_used - n0, seq)  # the look's way out overlapped the approach
    assert np.abs(sk2.ex.q["right"] - q_obs).max() < 1e-9
    assert np.linalg.norm(_true_pos(env2)[:2] - T2[:2, 3]) < 0.0005, r.note


def test_held_tol_zero_without_a_pre_look_reads_at_the_hover_then_looks(monkeypatch):
    env, prov, sk, T = _held_scene({"block": {"hover": [GOOD], "look": [GOOD], "show": [GOOD]}})
    monkeypatch.setattr(sk, "_pre_look", lambda pp, a1: None)
    sk.run_place(sk.plan_place("left", T, servo=True, pre_h=0.04, held_tol=0.0))
    ev = sk.held_events
    assert ev[0]["view"] == "hover" and len(ev) == 2 and ev[1]["adopted"] and not ev[0]["adopted"]


def test_no_look_without_the_budget():
    env, prov, sk, T = _held_scene({"block": {"hover": [AMB], "look": [GOOD], "show": [GOOD]}})
    pl = sk.plan_place("left", T, servo=True, pre_h=0.04)
    env.step_lim = sk.ex.steps_used + pl.steps + 45            # the tail fits, a look would not leave it its steps
    sk.run_place(pl)
    assert [e["view"] for e in sk.held_events] == ["hover"]


def test_failed_reads_fall_back_to_the_guarded_tail_no_deeper_than_blind_up():
    env, prov, sk, T = _held_scene({"block": {"hover": [AMB], "look": [AMB], "show": [AMB]}})
    pl = sk.plan_place("left", T, servo=True, pre_h=0.04, blind_up=0.01)
    z_full = float((T @ np.linalg.inv(sk.held["left"].T_tcp_obj))[2, 3])     # the TCP at the full-depth target
    zs = []
    step = env.step

    def spy(action):
        zs.append(float(sk.kin.tcp("left", np.asarray(action["left_arm_joint_state"], float))[2, 3]))
        return step(action)
    env.step = spy
    r = sk.run_place(pl)
    assert "probing tail" in r.note and not any(e["adopted"] for e in sk.held_events)
    assert len(sk.held_events) >= 2                              # it looked before giving up
    assert min(zs) > z_full + 0.009                              # never below the raised (threaded) target


def test_mode_a_never_reads_held_pose():
    from dojo.skills.heldsim import HeldBeliefProvider
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.22, -0.15, TABLE_Z + 0.01],
                         "quat": [1, 0, 0, 0], "half": HALF}])
    prov = HeldBeliefProvider(GTProvider(env, None), env, mode="gt")
    sk = Skills(env, prov)                                       # remeasure (G3)
    assert sk.run_pick(sk.plan_pick("b", "left")).ok
    T = np.asarray(sk.world.obj("b").T, float).copy()
    T[:3, 3] = (-0.18, -0.22, TABLE_Z + 0.01)
    sk.run_place(sk.plan_place("left", T, servo=True, pre_h=0.04))
    assert sk.held_events == [] and sum(prov.stats["reads"].values()) == 0


def test_the_noise_read_moves_the_leading_point_by_the_tables_error():
    err = {"ok": 1, "why": "", "err": [0.0, 0.5, -0.3, 0.0, 0.0, 0.0], "sig": [0.2, 0.2, 0.2, 0.2, 0.5]}
    env, prov, sk, T = _held_scene({"block": {"hover": [err]}}, offset=(0.0, 0.0))
    T_true = prov._true_T_rel("left", "b")
    r = prov.held_pose("left", prior_T_tcp_obj=sk.held["left"].T_tcp_obj, asset_key=sk.world.obj("b").asset.key,
                       cfg_={"T_obj_target": T.tolist(), "view": "hover"})
    assert r["ok"] and r["sigma"]["lead_lat_mm"] == 0.2
    d = (np.asarray(r["T_tcp_obj"]) - T_true)[:3, 3] * 1000
    assert np.allclose(d, [0.0, 0.5, -0.3], atol=1e-6)          # a pure shift: the origin moves with the leading point
    assert np.allclose(prov.att["left"][1], r["T_tcp_obj"])     # re-anchored, as the runtime belief does


def test_pegs_seat_on_a_good_read_and_thread_blind_without_one(monkeypatch):
    from dojo.skills import library, pegs
    seen = {}
    monkeypatch.setattr(pegs, "side_bias", lambda o, peg: 0.0)
    monkeypatch.setattr(pegs, "stack_top", lambda sk, on, peg: 0.80)
    monkeypatch.setattr(pegs, "piece_target", lambda sk, lab, peg, yaw, z, bias=None: (np.eye(4), np.zeros(3)))
    monkeypatch.setattr(library, "place_oriented", lambda sk, lab, tg, **kw: seen.update(kw["place_kw"]) or "r")
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.22, -0.15, TABLE_Z + 0.01],
                         "quat": [1, 0, 0, 0], "half": HALF}])
    peg = {"top": 0.85}
    for remeasure in (True, False):
        prov = GTProvider(env, None)
        prov.held_pose = lambda arm, **kw: {"ok": False}
        sk = Skills(env, prov, remeasure=remeasure)
        seen.clear()
        pegs._one_piece(sk, "b", sk.world.obj("b"), "b", peg, [], True, pegs.THREAD, 0.003)
        if remeasure:                                            # G3: unchanged
            assert not {"held_tol", "held_gate", "blind_up", "guarded"} & set(seen)
        else:                                                    # mode B: threaded, guarded (on the read if any)
            assert seen["guarded"] and not {"held_tol", "held_gate", "blind_up"} & set(seen)
    monkeypatch.setattr(pegs, "PEG_SEAT_MODE_B", True)
    prov = GTProvider(env, None)
    prov.held_pose = lambda arm, **kw: {"ok": False}
    sk = Skills(env, prov, remeasure=False)
    seen.clear()
    pegs._one_piece(sk, "b", sk.world.obj("b"), "b", peg, [], True, pegs.THREAD, 0.003)
    assert seen["held_gate"] == pegs.SEAT_SIGMA and seen["guarded"]
    assert abs(seen["blind_up"] - (0.85 - pegs.THREAD - (0.80 + pegs.SEAT_GAP))) < 1e-9


def test_the_pressed_spiral_covers_its_disc_in_short_steps():
    from dojo.skills.skills import SPIRAL_PITCH, SPIRAL_STEP, spiral_points
    p = np.asarray(spiral_points(0.0005))
    assert np.allclose(p[0], 0.0) and np.linalg.norm(p, axis=1).max() <= 0.0005 + 1e-12
    assert np.linalg.norm(np.diff(p, axis=0), axis=1).max() <= SPIRAL_STEP * 1.01
    # every point of the disc is within half a pitch of the path
    g = np.array([(x, y) for x in np.linspace(-4.5e-4, 4.5e-4, 10) for y in np.linspace(-4.5e-4, 4.5e-4, 10)
                  if x * x + y * y <= 0.00045 ** 2])
    d = np.min(np.linalg.norm(g[:, None, :] - p[None, :, :], axis=2), axis=1)
    assert d.max() <= 0.5 * SPIRAL_PITCH + SPIRAL_STEP / 2 and len(p) < 40


def test_a_seat_on_a_read_threads_first_with_the_guarded_descent():
    env, prov, sk, T = _held_scene({"block": {"hover": [GOOD], "look": [GOOD], "show": [GOOD]}})
    r = sk.run_place(sk.plan_place("left", T, servo=True, pre_h=0.04, blind_up=0.01))
    assert sk.held_events[0]["adopted"] and "threaded first" in r.note and "servo" in r.note, r.note
    assert np.linalg.norm(_true_pos(env) - T[:3, 3]) < 0.0008, r.note        # then down to the full target


def test_a_guarded_place_runs_the_probing_tail_on_the_adopted_read():
    env, prov, sk, T = _held_scene({"block": {"hover": [GOOD], "look": [GOOD], "show": [GOOD]}})
    r = sk.run_place(sk.plan_place("left", T, servo=True, pre_h=0.04, guarded=True))
    assert sk.held_events[0]["adopted"] and "held read adopted" in r.note and "probing tail" in r.note, r.note
    assert np.linalg.norm(_true_pos(env)[:2] - T[:2, 3]) < 0.0005, r.note    # aimed with the read (belief 3 mm off)
# ------------------------------------------------------------------------- non-grasp outcomes (confirm_pose, #67)
def _kong_scene(freeze, confirm, offset=(0.002, 0.0)):
    from dojo.skills.heldsim import HeldBeliefProvider
    from tests.dojo.test_skills2 import kong_rows
    env = KinematicEnv(kong_rows(0)[:3], step_lim=400)
    base = OffsetBeliefProvider(GTProvider(env, None), {"mahjong0_1": offset})
    prov = HeldBeliefProvider(base, env, mode=None, freeze=freeze, confirm=confirm)
    return env, prov, Skills(env, prov, remeasure=False, refine=lambda *a: None)


def test_a_topple_the_belief_does_not_see_fails_without_confirm_pose():
    from dojo.skills.library import topple, up_body_axis
    env, prov, sk = _kong_scene(freeze=True, confirm=False)
    r = topple(sk, "mahjong0_1", up_body_axis(sk.world.obj("mahjong0_1")))
    from dojo.skills.geometry import quat_to_R
    assert quat_to_R(next(o for o in env.objects if o["label"] == "mahjong0_1")["quat"])[2, 2] > 0.95   # it did tip
    assert not r.ok and prov.stats["frozen"] >= 1                 # ... but the belief keeps it standing (mode B)


def test_confirm_pose_adopts_the_toppled_pose():
    from dojo.skills.library import topple, up_body_axis
    env, prov, sk = _kong_scene(freeze=True, confirm=True)
    r = topple(sk, "mahjong0_1", up_body_axis(sk.world.obj("mahjong0_1")))
    assert r.ok, r.note
    assert sk.confirm_events and sk.confirm_events[0]["adopted"] and prov.stats["confirm_adopted"] == 1
    assert sk.refresh().obj("mahjong0_1").R[2, 2] > 0.95
    for lab in ("mahjong0_0", "mahjong0_2"):                     # the neighbours were not confirmed into anything
        assert abs(sk.world.obj(lab).R[2, 1]) > 0.99


def test_a_wrong_expectation_is_not_confirmed():
    env, prov, sk = _kong_scene(freeze=True, confirm=True)
    from dojo.skills.library import topple, up_body_axis
    r = topple(sk, "mahjong0_1", up_body_axis(sk.world.obj("mahjong0_1")))
    assert r.ok
    T = sk.world.obj("mahjong0_1").T.copy()
    T[0, 3] += 0.05                                               # 5 cm off the truth
    ans = sk.confirm_moved("mahjong0_1", T, "test")
    assert ans is not None and not ans["adopted"]


def test_without_a_confirm_provider_a_topple_reads_the_world():
    from dojo.skills.library import topple, up_body_axis
    from tests.dojo.test_skills2 import kong_rows
    env = KinematicEnv(kong_rows(0)[:3], step_lim=300)
    sk = Skills(env, GTProvider(env, None))
    assert topple(sk, "mahjong0_1", up_body_axis(sk.world.obj("mahjong0_1"))).ok and sk.confirm_events == []


def test_toppled_pose_turns_about_the_far_bottom_edge():
    from tests.dojo.test_skills2 import FRONT, HALF as KH, _obj
    o = _obj("t", "mahjong", 0, (0.0, -0.2, TABLE_Z + KH[1]), KH, quat=FRONT)
    from dojo.skills.library import up_body_axis
    ax = np.asarray(up_body_axis(o), float)
    n = o.R @ ax                                                   # horizontal now; tips the other way
    T = Skills.toppled_pose(o, -n)
    from dojo.skills.geometry import quat_to_R, T_to_pose7
    R = quat_to_R(T_to_pose7(T)[3:])
    assert float((R @ ax)[2]) > 0.999                              # that axis points up
    lo = (R @ (np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * KH).T).T + T[:3, 3]
    assert abs(lo[:, 2].min() - TABLE_Z) < 1e-6                    # resting on the table


def test_declare_kong_in_mode_b_needs_confirm_pose():
    from dojo.skills import planner
    from dojo.skills.heldsim import HeldBeliefProvider
    from tests.dojo.test_skills2 import KONG, kong_rows
    out = {}
    for confirm in (False, True):
        env = KinematicEnv(kong_rows(0), step_lim=600, instruction=KONG)
        prov = HeldBeliefProvider(GTProvider(env, None, instruction=KONG), env, mode=None, freeze=True, confirm=confirm)
        sk = Skills(env, prov, remeasure=False, refine=lambda *a: None)
        res = planner.solve(sk)
        from dojo.skills.geometry import quat_to_R
        rows = {o["label"]: o for o in env.objects}
        out[confirm] = (bool(res.get("ok")), sum(quat_to_R(rows[f"mahjong0_{i}"]["quat"])[2, 2] > 0.85 for i in range(3)))
    assert out[True] == (True, 3), out
    assert not out[False][0], out                               # the belief never sees the tiles tip
