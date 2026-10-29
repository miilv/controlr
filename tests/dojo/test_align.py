"""Stream N: identical blocks slid into one aligned row (``dojo.skills.align``; align_blocks' instruction): the
checker's row measure, the 1-D spread, plan choice, the off-by-default route, a whole episode on the kinematic env.
CPU only."""

from __future__ import annotations

import math

import numpy as np

from dojo.skills import align, planner
from dojo.skills.env import KinematicEnv
from dojo.skills.skills import GTProvider, Skills
from dojo.skills.world import AssetInfo, ObjectState, WorldState

ALIGN = "Use the set square to push the three blocks into a straight, aligned row, then reset the robot arm."
CUBE = [0.0175, 0.0175, 0.0175]
BAR = [0.275, 0.0135, 0.01]                  # the set square (triangular_prism/00000)
Z = 0.765


def _obj(label, cat, xyz, half, yaw_deg=0.0, typ="Rigid"):
    info = AssetInfo(key=f"{typ}/{cat}/00000", bbox_min=-np.asarray(half, float), bbox_max=np.asarray(half, float))
    y = math.radians(yaw_deg)
    return ObjectState(label, typ, cat, 0, np.asarray(xyz, float), np.array([math.cos(y / 2), 0, 0, math.sin(y / 2)]), info)


def _scene(yaws=(20.0, -25.0, 10.0)):
    return [_obj("cube0", "cube", (-0.10, -0.08, Z + 0.0175), CUBE, yaws[0]),
            _obj("cube1", "cube", (0.02, -0.13, Z + 0.0175), CUBE, yaws[1]),
            _obj("cube2", "cube", (0.11, -0.07, Z + 0.0175), CUBE, yaws[2]),
            _obj("target", "triangular_prism", (0.0, -0.19, Z + 0.01), BAR, 8.0)]


def _world(objs, routes=()):
    return WorldState("", "", {o.label: o for o in objs}, {}, {}, notes={"routes": list(routes)} if routes else {})


def test_row_error_is_the_checkers_measure():
    pts = np.array([[0.0, 0.0], [0.06, 0.0], [0.12, 0.0]])
    assert align.row_error(pts, [0.0, 90.0, -180.0]) == (0.0, 0.0)
    d, a = align.row_error(np.array([[0.0, 0.0], [0.06, 0.01], [0.12, 0.0]]), [0.0, 5.0, 44.0])
    assert 0.006 < d < 0.007 and 43.0 < a < 45.0
    assert align.row_error(pts, [9.0, 0.0, 0.0])[1] > 8.0


def test_spread_keeps_order_side_and_spacing():
    out = align._spread({"a": 0.0, "b": 0.01, "c": 0.03, "d": -0.2}, "a", 0.06)
    assert out == {"a": 0.0, "b": 0.06, "c": 0.12, "d": -0.2}


def test_row_plans_anchor_and_targets():
    w = _world(_scene())
    assert len(align.row_plans(w, ["cube0", "cube1", "cube2"], slots=False, free_rows=False)) == 6   # 3 anchors x 2
    plans = align.row_plans(w, ["cube0", "cube1", "cube2"])
    assert len(plans) > 6 and plans[0]["cost"] <= plans[-1]["cost"]
    for p in plans:
        ts = p["targets"]
        pts = np.array([ts[k][0] for k in ("cube0", "cube1", "cube2")])
        d, a = align.row_error(pts, [ts[k][1] for k in ts])
        assert d < 1e-9 and a < 1e-9                                  # on one line, all along it
        if p["anchor"] is not None:
            assert np.allclose(ts[p["anchor"]][0], w.obj(p["anchor"]).center_w[:2])
        else:
            assert p["cost"] >= align.FREE_ROW_COST                 # a free row (every block moves) costs more
        assert min(np.linalg.norm(pts[i] - pts[j]) for i in range(3) for j in range(i)) >= align.SPACING - 1e-9


def test_route_off_by_default_and_ignores_the_set_square():
    w = _world(_scene())
    assert planner.plan(ALIGN, w).skill == "none"
    p = planner.plan(ALIGN, _world(_scene(), routes=["align"]))
    assert p.skill == "align_in_row" and p.args == {"objects": ["cube0", "cube1", "cube2"]}


def _rows(objs):
    return [dict(label=o.label, category=o.category, category_idx=o.idx, type=o.obj_type, pos=o.pos.tolist(),
                 quat=o.quat.tolist(), half=(o.asset.extents / 2).tolist()) for o in objs]


def test_align_in_row_on_the_kinematic_env():
    objs = _scene()
    env = KinematicEnv(_rows(objs), step_lim=200, instruction=ALIGN)
    prov = GTProvider(env, None, instruction=ALIGN)
    prov.notes["routes"] = ["align"]
    sk = Skills(env, prov)
    z0 = {o.label: float(o.pos[2]) for o in objs}
    rise = {k: 0.0 for k in z0}

    def track(ex):
        for r in env.objects:
            if r["label"] in rise:
                rise[r["label"]] = max(rise[r["label"]], float(r["pos"][2]) - z0[r["label"]])
    sk.ex.on_step = track
    res = planner.solve(sk)
    assert res["ok"], res
    w = sk.refresh()
    labs = ["cube0", "cube1", "cube2"]
    d, a = align.row_error(np.array([w.obj(k).center_w for k in labs]), [align.yaw_deg(w.obj(k)) for k in labs])
    assert d <= 0.008 and a <= 8.0, (d, a)
    assert max(rise.values()) < 0.007, rise                             # never lifted (the checker's 7 mm)
    assert env.steps_used <= 200


def test_mode_b_measures_each_block_before_its_slide():
    """With a refine hook (mode B) the slides run one at a time, a pick refine over each block before its slide; the
    row still comes out straight within 200 steps (the hook here observes nothing)."""
    objs = _scene()
    env = KinematicEnv(_rows(objs), step_lim=200, instruction=ALIGN)
    prov = GTProvider(env, None, instruction=ALIGN)
    prov.notes["routes"] = ["align"]
    calls = []
    sk = Skills(env, prov, refine=lambda sk_, label, arm, kind: calls.append((label, kind)) or None)
    res = planner.solve(sk)
    assert res["ok"], res
    anchor = sk.world.notes["align"]["anchor"]
    assert set(calls) == {(k, "pick") for k in ("cube0", "cube1", "cube2") if k != anchor}
    w = sk.refresh()
    labs = ["cube0", "cube1", "cube2"]
    d, a = align.row_error(np.array([w.obj(k).center_w for k in labs]), [align.yaw_deg(w.obj(k)) for k in labs])
    assert d <= 0.008 and a <= 8.0 and env.steps_used <= 200, (d, a, env.steps_used)


def _run_mode_b(measure: str):
    objs = _scene()
    env = KinematicEnv(_rows(objs), step_lim=200, instruction=ALIGN)
    prov = GTProvider(env, None, instruction=ALIGN)
    prov.notes["routes"] = ["align"]
    calls = []
    sk = Skills(env, prov, refine=lambda sk_, label, arm, kind: calls.append((label, arm, kind)) or None)
    sk.align_measure = measure
    res = planner.solve(sk)
    w = sk.refresh()
    labs = ["cube0", "cube1", "cube2"]
    d, a = align.row_error(np.array([w.obj(k).center_w for k in labs]), [align.yaw_deg(w.obj(k)) for k in labs])
    return res, calls, env.steps_used, d, a, sk


def test_mode_b_parallel_hovers_save_steps_and_keep_the_row():
    """align_measure "parallel" (default "seq"): every hover is the slide's pre-grasp; here the two hovers are too close
    to share the steps (``ARMS_APART``), so one block at a time, still one refine per mover; the row is as straight and
    the episode needs fewer steps (the camera arm ran out of its 200)."""
    from dojo.skills.skills import Skills as _S
    assert getattr(_S.__new__(_S), "align_measure", "seq") == "seq"
    r0, c0, n0, d0, a0, _ = _run_mode_b("seq")
    r1, c1, n1, d1, a1, sk = _run_mode_b("parallel")
    assert r0["ok"] and r1["ok"]
    assert d1 <= 0.008 and a1 <= 8.0
    anchor = sk.world.notes["align"]["anchor"]
    movers = {k for k in ("cube0", "cube1", "cube2") if k != anchor}
    assert {c[0] for c in c1} == movers and {c[1] for c in c1} == {"left", "right"} and all(c[2] == "pick" for c in c1)
    assert n1 < n0, (n0, n1)


def test_runtime_config_align_measure():
    import pytest as _pytest

    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    assert RuntimeConfig().align_measure == "seq"
    with _pytest.raises(ValueError):
        RuntimePolicy(cfg=RuntimeConfig(align_measure="bogus"))


def test_an_unconfirmed_fix_slide_is_measured_and_slid_again():
    """align_measure "parallel": the head view did not confirm the fixing slide (camera c4b_alignsg_par_100020: the row
    read 0.0 mm / 0.0 deg with a block 30 deg off). The block is measured from the arm's hover and slid once more from
    where it is."""
    objs = _scene(yaws=(0.0, 0.0, 30.0))
    env = KinematicEnv(_rows(objs), step_lim=400, instruction=ALIGN)
    calls = []
    sk = Skills(env, GTProvider(env, None, instruction=ALIGN),
                refine=lambda sk_, label, arm, kind: calls.append((label, arm, kind)) or None)
    sk.align_measure = "parallel"
    o = sk.refresh().obj("cube2")
    xy, th = o.center_w[:2] + np.array([0.0, -0.02]), 0.0
    d0, a0 = align._off(o, xy, th)
    assert a0 > 20.0
    arm = align._arm_for(sk, "cube2", xy)[0]
    align._fix_unconfirmed(sk, "cube2", xy, th, arm)
    d1, a1 = align._off(sk.refresh().obj("cube2"), xy, th)
    assert calls == [("cube2", arm, "pick")]
    assert d1 <= align.FIX_TOL[0] and a1 <= align.FIX_TOL[1], (d1, a1)
