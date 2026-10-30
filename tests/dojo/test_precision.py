"""Precision stream (2026-10-27) pieces without Isaac: the as-is relay of an object lying on the table (insert_each's
fallback when no arm has a one-arm plan into a hole)."""

import math
from types import SimpleNamespace

import numpy as np

from dojo.skills import library as L
from dojo.skills import precision as P
from dojo.skills.kinematics import TABLE_Z, X5Kin
from dojo.skills.skills import SkillResult
from dojo.skills.world import AssetInfo, ObjectState, WorldState


def _tube_mesh(r=0.014, length=0.11, n=16):
    """A tube along body y: a narrow tip ring at y 0, the body ring at ``length``."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    tip = np.stack([0.003 * np.cos(a), np.zeros(n), 0.003 * np.sin(a)], axis=1)
    body = np.stack([r * np.cos(a), np.full(n, length), r * np.sin(a)], axis=1)
    mid = np.stack([r * np.cos(a), np.full(n, 0.02), r * np.sin(a)], axis=1)
    return np.vstack([tip, mid, body]), np.zeros((0, 3), int)


def _obj(label, typ, cat, pos, quat=(1, 0, 0, 0), lo=(-0.014, 0, -0.014), hi=(0.014, 0.11, 0.014), mesh=None, **asset):
    info = AssetInfo(key=f"{typ}/{cat}/00000", bbox_min=np.asarray(lo, float), bbox_max=np.asarray(hi, float),
                     mesh=mesh, **asset)
    return ObjectState(label, typ, cat, 0, np.asarray(pos, float), np.asarray(quat, float), info)


def _quat_x(deg):
    t = math.radians(deg) / 2
    return (math.cos(t), math.sin(t), 0.0, 0.0)


def test_lies_on_table_wants_a_level_long_axis_resting_on_the_table():
    V, F = _tube_mesh()
    lying = _obj("t", "Rigid", "test_tube", (0.2, -0.1, TABLE_Z + 0.014), mesh=(V, F))     # long axis (y) level
    assert L.lies_on_table(lying)
    assert not L.rests_stably(lying, L.level_R(lying.R))     # its lowest points are a line: no footprint
    lifted = _obj("t", "Rigid", "test_tube", (0.2, -0.1, TABLE_Z + 0.03), mesh=(V, F))
    assert not L.lies_on_table(lifted)
    standing = _obj("t", "Rigid", "test_tube", (0.2, -0.1, TABLE_Z + 0.014), _quat_x(90), mesh=(V, F))
    assert not L.lies_on_table(standing)
    assert not L.lies_on_table(_obj("t", "Rigid", "test_tube", (0.2, -0.1, TABLE_Z + 0.014)))    # no mesh


def test_insert_each_relays_a_lying_tube_through_up_to_three_holes(monkeypatch):
    V, F = _tube_mesh()
    rack = _obj("rack", "Geometry", "test_tube_slot", (0.0, -0.05, TABLE_Z + 0.03), lo=(-0.1, -0.05, -0.03),
                hi=(0.1, 0.05, 0.03),
                support={f"s/{k}": [(np.array([-0.072 + 0.036 * k, 0.0, 0.028, 1, 0, 0, 0]), 0.015)] for k in range(5)})
    tube = _obj("tube0", "Rigid", "test_tube", (-0.35, -0.2, TABLE_Z + 0.014), mesh=(V, F))
    w = WorldState(task="", instruction="", objects={"rack": rack, "tube0": tube}, arms={}, home_q={}, step=0,
                   step_lim=500)
    logs = []
    sk = SimpleNamespace(world=w, kin=X5Kin(), held={"left": None, "right": None}, log=logs.append,
                         coll=SimpleNamespace(include=set(), heightmaps={}, margin_of={}),
                         refresh=lambda: w, bank=lambda: None,
                         plan_transfer=lambda *a, **k: SkillResult(False, "transfer", "ik_fail", 0, None,
                                                                   note="left: 0/9 grasps reachable, 0 grasp-place pairs"))
    calls = []

    def fake_place_oriented(sk_, lab, tg, **kw):
        calls.append(kw)
        return SkillResult(False, "place_oriented", "ik_fail", 0, None, note="no relay")
    monkeypatch.setattr(L, "place_oriented", fake_place_oriented)
    r = P.insert_each(sk, ["tube0"], "rack")
    assert not r["ok"]
    assert len(calls) == P.EACH_RELAY_HOLES
    assert all(c["as_is_lying"] and c["spot_avoid"] == ("rack",) and not c["air_handover"] for c in calls)
    assert sum(m.endswith(": relay") for m in logs) == P.EACH_RELAY_HOLES

    # a relay that moved something (steps > 0) is not followed by another hole
    calls.clear()
    monkeypatch.setattr(L, "place_oriented", lambda sk_, lab, tg, **kw: calls.append(kw) or SkillResult(
        False, "place", "place_miss", 40, "right", note="missed"))
    w.notes.clear()
    P.insert_each(sk, ["tube0"], "rack")
    assert len(calls) == 1


def test_insert_each_tries_a_relay_before_a_plan_that_holds_the_tube_off_its_centre_of_mass(monkeypatch):
    V, F = _tube_mesh()
    rack = _obj("rack", "Geometry", "test_tube_slot", (0.0, -0.05, TABLE_Z + 0.03), lo=(-0.1, -0.05, -0.03),
                hi=(0.1, 0.05, 0.03),
                support={f"s/{k}": [(np.array([-0.072 + 0.036 * k, 0.0, 0.028, 1, 0, 0, 0]), 0.015)] for k in range(5)})
    tube = _obj("tube0", "Rigid", "test_tube", (-0.35, -0.2, TABLE_Z + 0.014), mesh=(V, F))
    w = WorldState(task="", instruction="", objects={"rack": rack, "tube0": tube}, arms={}, home_q={}, step=0,
                   step_lim=500)
    plan = (SimpleNamespace(arm="left", grasp=SimpleNamespace(note="box close 0 off +21mm (off the CoM)")), None)
    sk = SimpleNamespace(world=w, kin=X5Kin(), held={"left": None, "right": None}, log=lambda m: None,
                         coll=SimpleNamespace(include=set(), heightmaps={}, margin_of={}),
                         refresh=lambda: w, bank=lambda: None,
                         plan_transfer=lambda *a, **k: seen.append(k.get("place_reach")) or plan)
    seen = []
    order = []
    monkeypatch.setattr(L, "place_oriented", lambda sk_, lab, tg, **kw: order.append("relay") or SkillResult(
        False, "place_oriented", "ik_fail", 0, None, note="no relay"))
    monkeypatch.setattr(L, "_run_leg", lambda sk_, lab, tg, ys, ign, pk, first=None, **kw: order.append(
        ("leg", first is plan)) or SkillResult(False, "place", "place_miss", 50, "left", note="x"))
    P.insert_each(sk, ["tube0"], "rack")
    assert order == ["relay"] * P.EACH_RELAY_HOLES + [("leg", True)]
    assert seen and all(r == P.EACH_REACH for r in seen)        # every one-arm screen keeps to the tubes' reach


def test_insert_each_takes_an_object_only_one_arm_reaches_first(monkeypatch):
    V, F = _tube_mesh()
    rack = _obj("rack", "Geometry", "test_tube_slot", (0.0, -0.05, TABLE_Z + 0.03), lo=(-0.1, -0.05, -0.03),
                hi=(0.1, 0.05, 0.03),
                support={f"s/{k}": [(np.array([-0.072 + 0.036 * k, 0.0, 0.028, 1, 0, 0, 0]), 0.015)] for k in range(5)})
    near = _obj("tubeA", "Rigid", "test_tube", (0.05, -0.2, TABLE_Z + 0.014), mesh=(V, F))     # both arms, by the rack
    far = _obj("tubeB", "Rigid", "test_tube", (-0.40, -0.12, TABLE_Z + 0.014), mesh=(V, F))     # the left arm only
    kin = X5Kin()
    d = lambda o: [float(np.linalg.norm(kin.base(a)[:2, 3] - o.center_w[:2])) for a in ("left", "right")]
    assert max(d(far)) > P.EACH_PICK_REACH >= max(d(near))
    w = WorldState(task="", instruction="", objects={"rack": rack, "tubeA": near, "tubeB": far}, arms={}, home_q={},
                   step=0, step_lim=500)
    order = []
    sk = SimpleNamespace(world=w, kin=kin, held={"left": None, "right": None}, log=lambda m: None,
                         coll=SimpleNamespace(include=set(), heightmaps={}, margin_of={}), refresh=lambda: w,
                         bank=lambda: None,
                         plan_transfer=lambda lab, *a, **k: order.append(lab) or SkillResult(
                             False, "transfer", "ik_fail", 0, None, note="0 grasp-place pairs"))
    monkeypatch.setattr(L, "place_oriented", lambda *a, **k: SkillResult(False, "place_oriented", "ik_fail", 0, None,
                                                                          note="x"))
    P.insert_each(sk, ["tubeA", "tubeB"], "rack")
    assert order[0] == "tubeB" and order[-1] == "tubeA"


def _counted(monkeypatch, short):
    """insert_each with a fake transfer that leaves the tube upright over hole 0, ``short`` (m) above its full depth."""
    V, F = _tube_mesh()
    rack = _obj("rack", "Geometry", "test_tube_slot", (0.0, -0.05, TABLE_Z + 0.03), lo=(-0.1, -0.05, -0.03),
                hi=(0.1, 0.05, 0.03),
                support={f"s/{k}": [(np.array([-0.072 + 0.036 * k, 0.0, 0.028, 1, 0, 0, 0]), 0.015)] for k in range(5)})
    tube = _obj("tube0", "Rigid", "test_tube", (0.2, -0.2, TABLE_Z + 0.014), mesh=(V, F))
    w = WorldState(task="", instruction="", objects={"rack": rack, "tube0": tube}, arms={}, home_q={}, step=0,
                   step_lim=500)
    mouth = P.hole_frames(rack)[0][1][:3, 3]
    plan = (SimpleNamespace(arm="right", grasp=SimpleNamespace(note="box close 0 off +9mm")), None)
    sk = SimpleNamespace(world=w, kin=X5Kin(), held={"left": None, "right": None}, log=lambda m: None,
                         coll=SimpleNamespace(include=set(), heightmaps={}, margin_of={}), refresh=lambda: w,
                         bank=lambda: None, plan_transfer=lambda *a, **k: plan)

    def run_leg(sk_, lab, tg, *a, **k):            # tip down (body y up), the tip `short` above mouth - 0.5 L
        o = w.objects[lab]
        o.pos = np.array([mouth[0], mouth[1], mouth[2] - 0.5 * 0.11 + short])
        o.quat = np.array(_quat_x(90))
        return SkillResult(True, "place", "ok", 50, "right", note="probing tail")
    monkeypatch.setattr(L, "_run_leg", run_leg)
    return P.insert_each(sk, ["tube0"], "rack")["ok"]


def test_insert_each_counts_a_tube_only_at_its_depth(monkeypatch):
    assert _counted(monkeypatch, 0.0)
    assert not _counted(monkeypatch, 0.012)        # believed held by the rack 12 mm short (a probing tail stopped)


def test_relay_spots_are_also_sought_toward_a_far_target(monkeypatch):
    V, F = _tube_mesh()
    tube = _obj("tube0", "Rigid", "test_tube", (-0.1, -0.25, TABLE_Z + 0.014), mesh=(V, F))
    w = WorldState(task="", instruction="", objects={"tube0": tube}, arms={}, home_q={}, step=0, step_lim=500)
    sk = SimpleNamespace(world=w, kin=X5Kin(), log=lambda m: None, refresh=lambda: w,
                         plan_transfer=lambda *a, **k: SkillResult(False, "transfer", "ik_fail", 0, None, note="none"),
                         _record=lambda r: r)
    seen = []
    monkeypatch.setattr(L, "free_spots", lambda sk_, lab, near_xy=None, **k: seen.append(tuple(near_xy)) or [])
    T = np.eye(4)
    T[:3, 3] = (0.39, -0.10, TABLE_Z + 0.03)                    # the target far to the right
    r = L.place_oriented(sk, "tube0", [T], air_handover=False)
    assert not r.ok and seen == [(0.0, -0.28), (L.RELAY_SHIFT, -0.28)]
    seen.clear()
    T[0, 3] = 0.02                                              # central: one round
    L.place_oriented(sk, "tube0", [T], air_handover=False)
    assert seen == [(0.0, -0.28)]


def _guarded_steps(kw):
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.25, -0.15, TABLE_Z + 0.015],
                         "quat": [1, 0, 0, 0], "half": [0.015] * 3}])
    sk = Skills(env, GTProvider(env, None))
    assert sk.run_pick(sk.plan_pick("b", "left")).ok
    T = np.asarray(sk.world.obj("b").T, float).copy()
    T[2, 3] = TABLE_Z + 0.07                                 # 5.5 cm of free descent onto the target
    T0 = np.asarray(sk.world.obj("b").T, float).copy()
    T0[2, 3] = TABLE_Z + 0.015
    n0 = sk.ex.steps_used
    sk._guarded_descent("left", T0, sk.held["left"], dict(kw))
    return sk.ex.steps_used - n0


def test_a_longer_guard_stride_takes_fewer_env_steps():
    base = _guarded_steps({})
    quick = _guarded_steps({"guard_step": P.EACH_PLACE_KW["guard_step"], "guard_rate": P.EACH_PLACE_KW["guard_rate"]})
    assert quick < 0.6 * base, (base, quick)
