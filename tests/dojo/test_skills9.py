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


def test_held_tol_zero_always_looks_and_uses_the_last_read():
    env, prov, sk, T = _held_scene({"block": {"hover": [GOOD], "look": [GOOD], "show": [GOOD]}})
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
            assert not {"held_tol", "held_gate", "blind_up"} & set(seen)
        else:
            assert seen["held_gate"] == pegs.SEAT_SIGMA
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
