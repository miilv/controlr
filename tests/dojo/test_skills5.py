"""skills5 pieces without Isaac: strike order and tool grasps, discs into slots, height maps, insert_each holes,
screw_on pairing, the kinematic disc fall."""

import math

import numpy as np

from dojo.skills import planner
from dojo.skills import precision as P
from dojo.skills.env import KinematicEnv
from dojo.skills.geometry import T_from, rot_z
from dojo.skills.kinematics import TABLE_Z
from dojo.skills.skills import GTProvider, Skills
from dojo.skills.world import AssetInfo, ObjectState, WorldState


def _box_mesh(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    V = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def _cyl_mesh(r, y0, y1, n=24):
    """A closed cylinder along y (vertices only at its two end rings)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(a), np.zeros(n), r * np.sin(a)], axis=1)
    V = np.vstack([ring + [0, y0, 0], ring + [0, y1, 0], [[0, y0, 0], [0, y1, 0]]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return V, np.asarray(F)


def _obj(label, typ, cat, pos, quat=(1, 0, 0, 0), lo=(-0.02, -0.02, 0), hi=(0.02, 0.02, 0.04), mesh=None, **asset):
    info = AssetInfo(key=f"{typ}/{cat}/00000", bbox_min=np.asarray(lo, float), bbox_max=np.asarray(hi, float),
                     mesh=mesh, **asset)
    return ObjectState(label, typ, cat, 0, np.asarray(pos, float), np.asarray(quat, float), info)


def _world(objs, instruction=""):
    return WorldState(task="", instruction=instruction, objects={o.label: o for o in objs}, arms={},
                      home_q={}, step=0, step_lim=500)


# ---------------------------------------------------------------------------------------------- strike order


def _instrument(yaw=math.pi):
    fp = {f"hit_{k}": {"frame": [[0.15 - 0.043 * k, -0.015, 0.014, 1, 0, 0, 0]]} for k in range(8)}
    q = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    return _obj("xyl", "Geometry", "xylophone", (0.0, 0.0, TABLE_Z), q, lo=(-0.21, -0.12, 0), hi=(0.21, 0.09, 0.023),
                functional=fp)


def test_strike_points_follow_the_words_not_the_numbering():
    ins = _instrument(yaw=math.pi)                       # hit_0 ends up at the world's -x side
    tags = planner.strike_points(ins)
    assert tags == [f"hit_{k}" for k in range(8)]
    l2r = planner.order_points(ins, tags, "strike all keys from left to right")
    xs = [float(P.frame_points_w(ins, t).mean(0)[0]) for t in l2r]
    assert xs == sorted(xs) and l2r[0] == "hit_0"
    r2l = planner.order_points(ins, tags, "strike all keys from right to left")
    assert r2l == l2r[::-1]
    ins0 = _instrument(yaw=0.0)                          # the same instrument turned: the words still decide
    assert planner.order_points(ins0, tags, "from left to right")[0] == "hit_7"
    assert planner.order_points(ins0, tags, "strike every key") == tags     # no order in the words: own numbering


def test_planner_picks_strike_in_order_from_words_and_frames():
    V, F = _cyl_mesh(0.004, 0.014, 0.2)
    mallet = _obj("m", "Rigid", "mallet", (-0.25, -0.1, TABLE_Z + 0.013), lo=(-0.013, -0.013, -0.013),
                  hi=(0.013, 0.2, 0.013), mesh=(V, F),
                  active_functional={"beat": {"frame": [[0, 0, -0.01, 1, 0, 0, 0], [0, 0, 0.01, 1, 0, 0, 0]]}})
    w = _world([mallet, _instrument()], "Pick up the mallet and strike all xylophone keys from left to right.")
    p = planner.plan(w.instruction, w)
    assert p.skill == "strike_in_order" and p.args["tool"] == "m" and p.args["instrument"] == "xyl"
    assert p.args["points"][0] == "hit_0" and len(p.args["points"]) == 8


def test_section_points_and_handle_grasps_on_a_bare_cylinder():
    V, F = _cyl_mesh(0.004, 0.014, 0.2)
    sec = P.section_points((V, F), np.array([0, 1.0, 0]), 0.1)
    assert len(sec) >= 12 and np.allclose(np.linalg.norm(sec[:, [0, 2]], axis=1), 0.004, atol=4e-4)
    env = KinematicEnv([], instruction="")
    sk = Skills(env, GTProvider(env, None))
    o = _obj("m", "Rigid", "mallet", (-0.25, -0.15, TABLE_Z + 0.03), lo=(-0.013, -0.013, -0.013), hi=(0.013, 0.2, 0.013),
             mesh=(V, F))
    gs = P.handle_grasps(sk, o, "left", np.zeros(3))
    assert gs and all(g.method == "handle" and g.width < 0.01 for g in gs)
    assert abs(float(gs[0].T[:3, 1] @ np.array([0, 1.0, 0]))) < 1e-6           # closing across the handle
    assert gs[0].T[2, 0] < -0.99                                                # from above


def test_strike_orientations_point_the_gripper_down():
    R_tool_tcp = np.eye(3)
    ors = P.strike_orientations(np.array([0, 1.0, 0]), R_tool_tcp)
    assert ors and all(k[3] < -0.2 for _, k in ors) and ors[0][1][3] <= ors[-1][1][3]


# ---------------------------------------------------------------------------------------------- discs


def _coin(pos=(0.0, -0.2, TABLE_Z), quat=(1, 0, 0, 0)):
    a = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    ring = np.stack([0.0142 * np.cos(a), 0.0142 * np.sin(a), np.zeros(36)], axis=1)
    V = np.vstack([ring, ring + [0, 0, 0.0019]])
    F = np.array([[i, (i + 1) % 36, 36 + i] for i in range(36)])
    return _obj("coin", "Rigid", "coin", pos, quat, lo=(-0.0142, -0.0142, 0), hi=(0.0142, 0.0142, 0.0019), mesh=(V, F))


def test_disc_axis_and_spins_keep_the_centre():
    c = _coin()
    assert P.disc_axis(c) == 2
    assert P.disc_axis(_obj("b", "Rigid", "block", (0, 0, 0))) is None
    T = T_from([0.1, -0.2, 0.9], rot_z(0.3))
    sp = P.spin_about_axis(c, T, 2, n=6)
    cen = [S[:3, :3] @ c.asset.center + S[:3, 3] for S in sp]
    assert len(sp) == 6 and np.allclose(cen, cen[0], atol=1e-9)


def test_edge_grasps_keep_the_fingers_over_the_holder_and_pinch_grasps_hug_the_table():
    env = KinematicEnv([], instruction="")
    sk = Skills(env, GTProvider(env, None))
    q_edge = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]           # standing on its edge (normal along -y)
    c = _coin(pos=(-0.25, -0.12, TABLE_Z + 0.0152), quat=q_edge)
    floor = TABLE_Z + 0.02
    gs = P.edge_grasps(sk, c, "left", floor)
    assert gs
    for g in gs:
        low = float(sk.kin.finger_points(sk.kin.flange_from_tcp(g.T), P.G.gap_to_grip(0.024))[:, 2].min())
        assert low >= floor - 1e-6 and abs(float(g.T[:3, 1] @ c.R[:, 2])) > 0.99    # pads on the faces
    flat = _coin()
    ps = P.pinch_grasps(sk, flat, "right")
    assert ps and all(abs(float(sk.kin.finger_points(sk.kin.flange_from_tcp(g.T), 0.0)[:, 2].min()) - TABLE_Z
                          - P.LOW_TIP) < 1e-4 for g in ps)


def test_kinematic_env_lets_a_leaning_disc_fall_flat():
    q = [math.cos(math.radians(65) / 2), math.sin(math.radians(65) / 2), 0, 0]   # 25 deg off standing
    env = KinematicEnv([{"label": "coin", "pos": [0.0, -0.2, TABLE_Z + 0.014], "quat": q,
                         "half": [0.0142, 0.0142, 0.00095], "center": [0, 0, 0.00095]}])
    env._drop("coin")
    o = env._obj("coin")
    from dojo.skills.geometry import quat_to_R
    assert abs(quat_to_R(o["quat"])[2, 2]) > 0.99 and abs(o["pos"][2] - TABLE_Z) < 0.003


def test_height_map_lets_points_pass_over_a_low_part_of_a_box():
    V1, F1 = _box_mesh((-0.05, -0.05, 0.0), (0.05, 0.05, 0.02))        # a low slab ...
    V2, F2 = _box_mesh((-0.05, 0.02, 0.0), (0.05, 0.05, 0.1))          # ... with a tall back
    V, F = np.vstack([V1, V2]), np.vstack([F1, F2 + len(V1)])
    o = _obj("bank", "Geometry", "bank", (0, 0, TABLE_Z), lo=(-0.05, -0.05, 0), hi=(0.05, 0.05, 0.1), mesh=(V, F))
    hm = P.height_map(o)
    from dojo.skills.motion import below_height_map
    loc = np.array([[0.0, -0.03, 0.05], [0.0, 0.035, 0.05]]) - o.asset.center    # over the slab / in the back
    inside = below_height_map(hm, loc, 0.003)
    assert not inside[0] and inside[1]


# ---------------------------------------------------------------------------------------------- holes, screws


def test_hole_frames_and_upright_targets_put_the_narrow_end_down():
    rack = _obj("rack", "Geometry", "rack", (0, 0, TABLE_Z + 0.03), lo=(-0.1, -0.05, -0.03), hi=(0.1, 0.05, 0.03),
                support={"rack/0": [(np.array([0.0, -0.018, 0.028, 1, 0, 0, 0]), 0.015)],
                         "rack/1": [(np.array([0.036, -0.018, 0.028, 1, 0, 0, 0]), 0.015)]})
    holes = P.hole_frames(rack)
    assert len(holes) == 2 and np.allclose(holes[0][1][:3, 3], [0, -0.018, TABLE_Z + 0.058])
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    tip = np.stack([0.003 * np.cos(a), np.zeros(16), 0.003 * np.sin(a)], axis=1)
    body = np.stack([0.014 * np.cos(a), np.full(16, 0.11), 0.014 * np.sin(a)], axis=1)
    V = np.vstack([tip, body, tip + [0, 0.02, 0] + (body - [0, 0.11, 0]) * 0.9])
    tube = _obj("t", "Rigid", "test_tube", (0.3, -0.1, TABLE_Z + 0.015), lo=(-0.014, 0, -0.014), hi=(0.014, 0.11, 0.014),
                mesh=(V, np.zeros((0, 3), int)))
    p = holes[0][1][:3, 3]
    tg = P.upright_targets(tube, p, TABLE_Z + 0.006, 0.04, n_yaw=4)
    assert len(tg) == 4
    for T in tg:
        W = V @ T[:3, :3].T + T[:3, 3]
        assert abs(float(W[:, 2].min()) - (p[2] - 0.04)) < 1e-6           # the tip 4 cm into the hole
        assert np.allclose(W[np.argmin(W[:, 2])][:2], p[:2], atol=0.004)


def test_planner_insert_each_and_screw_pairs_by_colour():
    rack = _obj("rack", "Geometry", "test_tube_slot", (0, 0, TABLE_Z + 0.03),
                support={f"s/{k}": [(np.array([0.036 * k, 0, 0.028, 1, 0, 0, 0]), 0.015)] for k in range(4)})
    tubes = [_obj(f"tube{i}", "Rigid", "test_tube", (0.2 + 0.05 * i, -0.1, TABLE_Z + 0.015)) for i in range(3)]
    w = _world([rack] + tubes)
    p = planner.plan("Insert the three tubes into the rack one by one.", w)
    assert p.skill == "insert_each" and p.args["receptacle"] == "rack" and sorted(p.args["objects"]) == ["tube0", "tube1",
                                                                                                       "tube2"]
    cols = [(255, 0, 0), (0, 0, 255), (250, 180, 0)]
    nuts = [_obj(f"nut{i}", "Rigid", "factory_nut", (0.1 * i, -0.1, TABLE_Z), rgb=np.asarray(c, float))
            for i, c in enumerate(cols)]
    bolts = [_obj(f"bolt{i}", "Geometry", "factory_bolt", (0.3 - 0.1 * i, -0.2, TABLE_Z), rgb=np.asarray(c, float))
             for i, c in enumerate(cols[::-1])]
    w = _world(nuts + bolts)
    p = planner.plan("Insert and tighten each screw into the nut of the same color.", w)
    assert p.skill == "screw_on"
    assert dict(map(tuple, p.args["pairs"])) == {"nut0": "bolt2", "nut1": "bolt1", "nut2": "bolt0"}


def test_n_fold_of_a_hex_nut_and_a_square():
    hexn = _obj("n", "Rigid", "nut", (0, 0, 0), lo=(-0.0176, -0.0203, 0), hi=(0.0176, 0.0203, 0.019))
    sq = _obj("s", "Rigid", "block", (0, 0, 0), lo=(-0.02, -0.02, 0), hi=(0.02, 0.02, 0.02))
    assert P.n_fold(hexn) == 6 and P.n_fold(sq) == 4
    assert abs(P._flat_yaw(hexn)) < 1e-9                               # across the flats = its narrower x axis


def test_guarded_tail_descends_and_lets_go_on_the_kinematic_env():
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.25, -0.15, TABLE_Z + 0.015],
                         "quat": [1, 0, 0, 0], "half": [0.015] * 3}])
    sk = Skills(env, GTProvider(env, None))
    pp = sk.plan_pick("b", "left")
    r = sk.run_pick(pp)
    assert r.ok, r
    h = sk.held["left"]
    T = np.asarray(sk.world.obj("b").T, float).copy()
    T[2, 3] = TABLE_Z + 0.03
    note = sk._guarded_tail("left", T, h, {"release_gap": 0.0})
    assert "reached" in note and sk.ex.g["left"] > 0.9            # no contact on the kinematic env: straight down
    assert abs(sk.refresh().obj("b").pos[2] - (TABLE_Z + 0.015)) < 0.005   # let go: it dropped onto the table


def test_offset_belief_provider_shifts_only_unheld_objects():
    from dojo.skills.g3 import OffsetBeliefProvider
    env = KinematicEnv([{"label": "b", "pos": [-0.25, -0.15, TABLE_Z], "quat": [1, 0, 0, 0]}])
    prov = OffsetBeliefProvider(GTProvider(env, None), {"b": (0.004, -0.003)})
    w = prov.refresh()
    assert np.allclose(w.obj("b").pos[:2], [-0.246, -0.153])
    w = prov.refresh({"left": "b"})
    assert np.allclose(w.obj("b").pos[:2], [-0.25, -0.15])


def test_a_place_planned_with_its_pick_keeps_the_gripper_closed():
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.25, -0.15, TABLE_Z + 0.015],
                         "quat": [1, 0, 0, 0], "half": [0.015] * 3}])
    sk = Skills(env, GTProvider(env, None))
    T = np.asarray(sk.world.obj("b").T, float).copy()
    T[:2, 3] = (-0.2, -0.2)
    r = sk.plan_transfer("b", [T])
    assert not hasattr(r, "note"), r
    pp, pl = r
    assert pl.plan.g0 == 0.0                        # planned before the pick, while the gripper is open
    comp = pl.plan.compile()[0]
    assert all(g == 0.0 for _, g in comp[:5])       # the approach does not open it
