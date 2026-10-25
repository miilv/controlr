"""skills2: re-orienting transfers, topple, kong, peg-in-hole fit, new planner rules (CPU only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dojo.skills.env import KinematicEnv
from dojo.skills.skills import GTProvider, Skills

HALF = [0.02245, 0.03235, 0.0164]                     # a mahjong tile's half extents
FRONT = [0.7071, 0.7071, 0, 0]                        # standing, face towards the robot
BACK = [0, 0, 0.7071, 0.7071]                         # standing, face away
DOWN = [0, 0.7071, 0.7071, 0]                         # lying face down
KONG = "Wait for the opponent to discard a tile, then declare a kong with the matching tiles."


def _box_mesh(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def _merge(*meshes):
    Vs, Fs, n = [], [], 0
    for V, F in meshes:
        Vs.append(V)
        Fs.append(F + n)
        n += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def _obj(label, cat, idx, xyz, half, typ="Rigid", quat=(1.0, 0, 0, 0), mesh=None, functional=None, place=None):
    from dojo.skills.world import AssetInfo, ObjectState
    info = AssetInfo(key=f"{typ}/{cat}/{idx:05d}", bbox_min=-np.asarray(half, float), bbox_max=np.asarray(half, float),
                     mesh=mesh, functional=functional or {}, place=place or {})
    return ObjectState(label, typ, cat, idx, np.asarray(xyz, float), np.asarray(quat, float), info)


def _world(objs):
    from dojo.skills.world import WorldState
    return WorldState("", "", {o.label: o for o in objs}, {}, {})


def kong_rows(discard: int = 0) -> list[dict]:
    rows = []
    ids = [37, 25, 40, 31]
    xs = [[-0.325, -0.279, -0.233], [-0.187, -0.141, -0.095], [-0.049, -0.003, 0.043], [0.089, 0.135, 0.181]]
    for g in range(4):
        for i, x in enumerate(xs[g]):
            rows.append(dict(label=f"mahjong{g}_{i}", category="mahjong", category_idx=ids[g], pos=[x, -0.15, 0.7974],
                             quat=FRONT, half=HALF))
    for i, x in enumerate([0.227, 0.273]):
        rows.append(dict(label=f"mahjong4_{i}", category="mahjong", category_idx=11, pos=[x, -0.15, 0.7974],
                         quat=FRONT, half=HALF))
    for g, x in enumerate([0.0, 0.046, -0.046, -0.092]):
        rows.append(dict(label=f"mahjong{5 + g}_0", category="mahjong", category_idx=ids[g], pos=[x, 0.05, 0.7974],
                         quat=BACK, half=HALF))
    for lab, idx, y, z in (("other0", 2, -0.05, 0.7815), ("other1", 36, -0.002, 0.7815), ("other2", 14, -0.002, 0.8145),
                           ("mahjong9_0", 11, -0.05, 0.8145)):
        rows.append(dict(label=lab, category="mahjong", category_idx=idx, pos=[-0.4, y, z], quat=DOWN, half=HALF))
    d = next(r for r in rows if r["label"] == f"mahjong{5 + discard}_0")
    d["pos"], d["quat"] = [d["pos"][0], 0.0, 0.7814], [1, 0, 0, 0]          # the opponent's discard, face up
    return rows


# ----------------------------------------------------------------------------------------- planner rules
def test_planner_new_rules():
    from dojo.skills.planner import plan
    tiles = [_obj(f"t{i}", "mahjong", i % 3, (0.05 * i - 0.3, -0.15, 0.797), HALF) for i in range(6)]
    p = plan(KONG.lower(), _world(tiles))
    assert p.skill == "declare_kong" and len(p.args["tiles"]) == 6

    key = _obj("key", "key", 0, (0.16, -0.2, 0.77), (0.011, 0.005, 0.031))
    slot = _obj("slot", "key_slot", 0, (-0.14, -0.155, 0.765), (0.05, 0.05, 0.0275), "Geometry")
    p = plan("pick up the key, hand it over to the other hand, insert it into the keyhole, then turn it.",
             _world([key, slot]))
    assert p.skill == "insert_into" and p.args["obj"] == "key" and p.args["receptacles"] == ["slot"]
    assert p.args["turn_deg"] == -60.0
    p = plan("plug the charger into the power strip.",
             _world([_obj("charger", "charger", 0, (0.3, -0.2, 0.78), (0.023, 0.035, 0.015)),
                     _obj("socket", "socket", 0, (-0.25, -0.06, 0.764), (0.078, 0.018, 0.01))]))
    assert p.skill == "insert_into" and p.args["turn_deg"] == 0.0

    box = _obj("toolbox", "toolbox", 0, (0.03, -0.02, 0.79), (0.137, 0.138, 0.025), "Geometry",
               functional={"hammer_slot": {"frame": [[0, 0, 0, 1, 0, 0, 0]]}, "wrench_slot": {"frame": [[0, 0, 0, 1, 0, 0, 0]]}})
    tools = [_obj("hammer", "hammer", 5, (-0.2, -0.05, 0.776), (0.11, 0.013, 0.038)),
             _obj("wrench", "wrench", 0, (-0.28, -0.12, 0.772), (0.08, 0.008, 0.02))]
    p = plan("place each tool into its matching position in the toolbox, then reset the robot arm.", _world([box] + tools))
    assert p.skill == "place_in_slots" and sorted(p.args["objects"]) == ["hammer", "wrench"]

    pens = [_obj("p0", "pen", 4, (-0.1, -0.1, 0.77), (0.09, 0.01, 0.01)),
            _obj("p1", "oil_pen", 1, (-0.2, -0.1, 0.77), (0.02, 0.012, 0.07)),
            _obj("h", "pen_holder", 0, (0.2, -0.1, 0.765), (0.04, 0.035, 0.06))]
    p = plan("hold the pen holder with one hand, place all pens into it with the other hand, then put it back down.",
             _world(pens))
    assert p.skill == "fill_container" and p.args["container"] == "h" and sorted(p.args["objects"]) == ["p0", "p1"]

    objs = [_obj("mouse", "mouse", 6, (0.4, -0.27, 0.78), (0.05, 0.03, 0.02)),
            _obj("alarm", "alarm", 0, (-0.3, -0.2, 0.8), (0.04, 0.025, 0.04)),
            _obj("keyboard", "keyboard", 2, (-0.04, -0.18, 0.77), (0.16, 0.07, 0.01)),
            _obj("garage", "garage", 11, (-0.3, -0.12, 0.86), (0.05, 0.05, 0.095)),
            _obj("cube_cushion", "cube_cushion", 2, (-0.15, 0.1, 0.765), (0.025, 0.025, 0.00005), "Geometry"),
            _obj("frame", "frame", 0, (0.1, 0, 0.765), (0.12, 0.05, 0.0002), "Geometry"),
            _obj("drawer", "drawer", 0, (-0.37, 0.1, 0.765), (0.13, 0.17, 0.11), "Geometry"),
            _obj("monitor", "monitor", 1, (0.14, 0.24, 0.97), (0.25, 0.08, 0.22), "Geometry"),
            _obj("mousemat", "mousemat", 0, (0.47, 0, 0.768), (0.12, 0.1, 0.004), "Geometry")]
    p = plan("place the alarm clock on the drawer, put the figurine on the stand, place the mouse on the mouse pad, "
             "and push the keyboard into the frame.", _world(objs))
    assert p.skill == "arrange"
    assert p.args["steps"] == [("place", "alarm", "drawer"), ("place", "garage", "cube_cushion"),
                               ("place", "mouse", "mousemat"), ("push", "keyboard", "frame")]


# ----------------------------------------------------------------------------------------- geometry
def test_fit_insertion_finds_the_hole():
    """A 10 x 10 mm square hole, 20 mm deep, 6 mm off the block's centre; an 8 x 8 mm peg goes in tip
    (narrow end) first, centred in the hole, ~1 mm clearance, to the floor."""
    from dojo.skills.insertion import fit_insertion
    c = np.array([0.006, 0.0])
    walls = [_box_mesh((-0.04, -0.04, 0.0), (0.04, 0.04, 0.01)),                       # floor
             _box_mesh((-0.04, -0.04, 0.01), (c[0] - 0.005, 0.04, 0.03)),
             _box_mesh((c[0] + 0.005, -0.04, 0.01), (0.04, 0.04, 0.03)),
             _box_mesh((c[0] - 0.005, -0.04, 0.01), (c[0] + 0.005, c[1] - 0.005, 0.03)),
             _box_mesh((c[0] - 0.005, c[1] + 0.005, 0.01), (c[0] + 0.005, 0.04, 0.03))]
    hole = _obj("blk", "block", 0, (0.1, -0.2, 0.765), (0.04, 0.04, 0.015), "Geometry", mesh=_merge(*walls))
    peg_mesh = _merge(_box_mesh((-0.004, -0.004, -0.04), (0.004, 0.004, 0.0)),          # shaft, tip at -z
                      _box_mesh((-0.012, -0.012, 0.0), (0.012, 0.012, 0.01)))           # wide head
    peg = _obj("peg", "peg", 0, (0.3, -0.2, 0.78), (0.012, 0.012, 0.025), mesh=peg_mesh, quat=(0.7071, 0.7071, 0, 0))
    f = fit_insertion(peg, hole, res=0.0005)
    assert f is not None
    tip_xy = f.T_obj[:2, 3]
    assert np.linalg.norm(tip_xy - (hole.pos[:2] + c)) < 0.0015
    assert 0.0005 < f.clearance < 0.0015
    assert abs(f.depth - 0.02) < 0.002
    assert float(f.T_obj[2, 2]) > 0.99                    # the shaft points down, head up
    assert abs(float(f.T_obj[2, 3]) - 0.04 - (0.765 + 0.01)) < 0.002   # tip on the floor


def test_box_grasps_keep_fingertips_above_the_table():
    from dojo.skills import grasps as G
    from dojo.skills.kinematics import TABLE_Z, X5Kin
    kin = X5Kin()
    o = _obj("w", "wrench", 0, (-0.25, -0.15, TABLE_Z + 0.008), (0.08, 0.02, 0.008))
    gs = G.box_grasps(o, kin, "left")
    down = [g for g in gs if g.T[2, 0] < -0.8]
    assert down
    for g in down:
        tip = g.T[:3, 3] + g.T[:3, 0] * 0.0126
        assert tip[2] >= TABLE_Z + 0.004


# ----------------------------------------------------------------------------------------- skills on the kinematic env
def test_topple_tips_a_standing_tile_face_up():
    from dojo.skills.library import topple, up_body_axis
    env = KinematicEnv(kong_rows(0)[:3], step_lim=300)
    sk = Skills(env, GTProvider(env, None))
    r = topple(sk, "mahjong0_1", up_body_axis(sk.world.obj("mahjong0_1")))
    assert r.ok
    w = sk.refresh()
    assert w.obj("mahjong0_1").R[2, 2] > 0.95                       # face up
    for lab in ("mahjong0_0", "mahjong0_2"):                       # neighbours 1 mm away still standing
        assert abs(w.obj(lab).R[2, 1]) > 0.99


@pytest.mark.parametrize("discard", [0, 3])
def test_declare_kong_on_the_kinematic_env(discard):
    from dojo.skills import planner
    env = KinematicEnv(kong_rows(discard), step_lim=600, instruction=KONG)
    sk = Skills(env, GTProvider(env, None, instruction=KONG))
    res = planner.solve(sk)
    assert res["ok"], res
    assert env.steps_used < 600
    rows = {o["label"]: o for o in env.objects}
    from dojo.skills.geometry import quat_to_R
    for i in range(3):                                              # the matching triple face up
        assert quat_to_R(rows[f"mahjong{discard}_{i}"]["quat"])[2, 2] > 0.85
    drawn = rows["mahjong9_0"]
    assert abs(quat_to_R(drawn["quat"])[2, 1]) > 0.99               # standing
    assert np.linalg.norm(np.asarray(drawn["pos"][:2]) - [0.319, -0.15]) < 0.015
    assert sk.world.notes["kong"]["discard"] == f"mahjong{5 + discard}_0"


def test_rests_stably():
    from dojo.skills.library import rests_stably
    tile = _obj("t", "mahjong", 0, (0, -0.2, 0.8), HALF, mesh=_box_mesh(-np.asarray(HALF), np.asarray(HALF)))
    assert rests_stably(tile, np.eye(3))                                # lying on its face
    R_end = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)          # standing on its end
    assert rests_stably(tile, R_end)
    key = _merge(_box_mesh((-0.0025, -0.0025, -0.053), (0.0025, 0.0025, -0.01)),
                 _box_mesh((-0.01, -0.005, -0.01), (0.01, 0.005, 0.01)))
    k = _obj("k", "key", 0, (0, -0.2, 0.8), (0.01, 0.005, 0.031), mesh=key)
    assert not rests_stably(k, np.eye(3))                               # upright on its 5 mm tip
