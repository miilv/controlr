"""skills6: observation-driven skills (board game vs a scripted opponent, copying a demonstration, picking off a
conveyor) and holders with holes and a hinged lid (CPU only)."""

from __future__ import annotations

import math

import numpy as np

from tests.dojo.test_skills2 import _box_mesh, _obj, _world


def _cells():
    return {f"cell/{i}": [(np.array([(-0.074, 0.0, 0.074)[i % 3], (-0.074, 0.0, 0.074)[i // 3], 0.022, 1, 0, 0, 0]),
                           0.032)] for i in range(9)}


def _board_world():
    b = _obj("board", "checkerboard", 0, (0, -0.075, 0.765), (0.11, 0.11, 0.008), typ="Geometry")
    b.asset.bbox_min, b.asset.bbox_max = np.array([-0.11, -0.11, 0.0]), np.array([0.11, 0.11, 0.016])
    b.asset.support = _cells()
    objs = [b]
    for i, x in enumerate((-0.2, -0.1, 0.0, 0.1, 0.2)):
        objs.append(_obj(f"mine{i}", "chessman", 0, (x, -0.25, 0.765), (0.025, 0.025, 0.005)))
    for i, x in enumerate((-0.15, -0.05, 0.05, 0.15)):
        objs.append(_obj(f"theirs{i}", "chessman", 1, (x, 0.2, 0.765), (0.025, 0.025, 0.005)))
    return _world(objs)


def test_tic_tac_toe_rules():
    from dojo.skills.interactive import board_cells, choose_cell, grid_lines
    w = _board_world()
    cells = board_cells(w.obj("board"))
    lines = grid_lines(cells)
    assert len(cells) == 9 and len(lines) == 8
    assert choose_cell({}, lines, 9) == 4                                         # centre first
    assert choose_cell({4: ("own", "a"), 0: ("opp", "b")}, lines, 9) == 8          # the corner opposite theirs
    assert choose_cell({4: ("own", "a"), 0: ("opp", "b"), 8: ("own", "c"), 2: ("opp", "d")}, lines, 9) == 1  # block
    assert choose_cell({4: ("own", "a"), 3: ("own", "c"), 0: ("opp", "b"), 8: ("opp", "d")}, lines, 9) == 5  # win
    assert choose_cell({4: ("own", "a")}, lines, 9, skip={0, 2, 6, 8}) in (1, 3, 5, 7)
    # a random opponent never leaves us without a legal move
    rng = np.random.default_rng(0)
    for _ in range(300):
        occ, who = {}, "own"
        while len(occ) < 9:
            k = choose_cell(occ, lines, 9) if who == "own" else int(rng.choice([c for c in range(9) if c not in occ]))
            assert k is not None and k not in occ
            occ[k] = (who, "x")
            who = "opp" if who == "own" else "own"


def test_board_state_reads_pieces_in_cells():
    from dojo.skills.interactive import board_cells, board_state
    w = _board_world()
    cells = board_cells(w.obj("board"))
    w.obj("mine2").pos = np.array([0.003, -0.077, 0.781])                          # on the centre cell
    w.obj("theirs0").pos = np.array([-0.074, -0.001, 0.781])                       # on cell 6
    w.obj("theirs1").pos = np.array([0.074, -0.149, 0.765])                        # beside the board, on the table
    occ = board_state(w, "board", cells, [f"mine{i}" for i in range(5)], [f"theirs{i}" for i in range(4)])
    assert occ == {4: ("own", "mine2"), 6: ("opp", "theirs0")}


def test_planner_picks_the_game_and_our_pieces():
    from dojo.skills.planner import plan
    w = _board_world()
    p = plan("Play tic-tac-toe as the first player and fill the board with the opponent.", w)
    assert p.skill == "play_board_game"
    assert p.args["board"] == "board"
    assert p.args["pieces"] == [f"mine{i}" for i in range(5)]                    # nearer the robot, left to right
    assert sorted(p.args["opponent_pieces"]) == [f"theirs{i}" for i in range(4)]


def test_planner_copy_demonstration_and_conveyor():
    from dojo.skills.planner import plan
    objs = [_obj("b0", "basket", 5, (-0.4, -0.15, 0.805), (0.1, 0.15, 0.04), typ="Geometry"),
            _obj("b1", "basket", 5, (0.35, 0.25, 0.805), (0.1, 0.15, 0.04), typ="Geometry")]
    for i, (cat, idx) in enumerate((("toy_car", 1), ("phone", 0), ("watch", 2))):
        objs.append(_obj(f"t{i}", cat, idx, (0.1 * i - 0.1, -0.1, 0.78), (0.02, 0.02, 0.01)))
        objs.append(_obj(f"a{i}", cat, idx, (0.1 * i - 0.1, 0.2, 0.78), (0.02, 0.02, 0.01)))
    p = plan("Observe the object placement order, remember it, then place the corresponding objects into the basket "
             "in the same order.", _world(objs))
    assert p.skill == "copy_demonstration"
    assert (p.args["container"], p.args["demo_container"]) == ("b0", "b1")
    assert set(p.args["objects"]) == {f"t{i}" for i in range(3)} | {f"a{i}" for i in range(3)}
    belt = [_obj("conveyor", "conveyor", 0, (-2.08, 0.175, 0.05), (2.0, 0.6, 0.46), typ="Dynamic"),
            _obj("x0", "puppet", 0, (0.2, 0.0, 0.79), (0.03, 0.03, 0.03)),
            _obj("x1", "puppet", 0, (-1.6, 0.0, 0.79), (0.03, 0.03, 0.03))]
    p = plan("Remember the first object on the conveyor, then pick the matching object when it appears again.",
             _world(belt))
    assert p.skill == "match_and_pick" and p.args["conveyor"] == "conveyor"
    assert set(p.args["objects"]) == {"x0", "x1"}


def test_planner_fill_holes_and_lid():
    from dojo.skills.planner import plan
    h = _obj("egg_holder", "egg_holder", 0, (0.0, -0.19, 0.766), (0.06, 0.06, 0.08), typ="Articulation",
             functional={"GetParentLink": {"base_link": "lid", "parent_joint": "hinge", "frame": [[0, 0, 0, 1, 0, 0, 0]]},
                         "bottom": {"base_link": "body", "parent_joint": None, "frame": [[0, 0, 0.005, 1, 0, 0, 0]]}})
    h.asset.joints = {"hinge": {"type": "revolute", "body0": "body", "body1": "lid", "axis": [1, 0, 0],
                                "origin": [0, 0.06, 0.046], "lower": -90.0, "upper": 0.0, "position": -90.0}}
    eggs = [_obj(f"target{i}", "egg", 0, (-0.3 + 0.05 * i, -0.15, 0.80), (0.021, 0.021, 0.029)) for i in range(4)]
    basket = _obj("egg_basket", "egg_basket", 0, (-0.28, -0.15, 0.82), (0.11, 0.11, 0.044), typ="Geometry")
    p = plan("Place the four eggs from the basket into the egg holder, then close the lid.", _world([h, basket] + eggs))
    assert p.skill == "fill_holes"
    assert p.args == {"objects": [f"target{i}" for i in range(4)], "holder": "egg_holder", "lid_tag": "GetParentLink"}


def test_lid_ratio_reads_radians_against_degree_limits():
    from types import SimpleNamespace

    from dojo.skills.holders import joint_angle, joint_ratio
    J = {"type": "revolute", "lower": -90.0, "upper": 0.0, "position": -90.0}
    assert abs(joint_angle(SimpleNamespace(joints=np.array([-math.pi / 2])), J) + 90.0) < 1e-6
    assert abs(joint_ratio(SimpleNamespace(joints=np.array([-0.05])), J) - (1 - math.degrees(0.05) / 90)) < 1e-6
    assert joint_ratio(SimpleNamespace(joints=None), J) == 0.0


def test_standing_targets_put_the_long_axis_up():
    from dojo.skills.holders import standing_targets
    egg = _obj("e", "egg", 0, (0, 0, 0.8), (0.021, 0.021, 0.029), quat=(0.7071, 0.7071, 0, 0))
    T = standing_targets(egg, (0.1, -0.2), 0.81, tilts=(0.0, 30.0))
    assert len(T) == 2 * 2 * 2
    for Ti in T:
        up = abs(float(Ti[2, 2]))                                                 # body z (the long axis) vs world z
        assert up > math.cos(math.radians(30.5))
        corners = np.array([[x, y, z] for x in (-0.021, 0.021) for y in (-0.021, 0.021) for z in (-0.029, 0.029)])
        cw = corners @ Ti[:3, :3].T + Ti[:3, 3]
        assert abs(cw[:, 2].min() - 0.81) < 1e-6
        assert np.allclose((cw.min(0) + cw.max(0))[:2] / 2, (0.1, -0.2), atol=1e-6)


def _bowl_mesh(r0=0.06, r1=0.1, h=0.08, n=48, k=12):
    """A bowl: flat floor of radius r0 at z = 0 and a straight flared wall to radius r1 at z = h (object frame), with
    rings every few mm (a real catalogue mesh is dense)."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    radii = list(np.linspace(0.0, r0, 7)[1:]) + list(np.linspace(r0, r1, k)[1:])
    zs = [0.0] * 6 + list(np.linspace(0.0, h, k)[1:])
    rings = [np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z)] for r, z in zip(radii, zs)]
    V = np.vstack(rings + [np.zeros((1, 3))])
    c = len(V) - 1
    F = [[c, (i + 1) % n, i] for i in range(n)]
    for m in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            F += [[m * n + i, m * n + j, (m + 1) * n + j], [m * n + i, (m + 1) * n + j, (m + 1) * n + i]]
    return V, np.asarray(F)


def test_wall_clearance_of_a_bowl():
    from dojo.skills.holders import inside_wall, wall_clearance, wall_profile
    V, F = _bowl_mesh()
    b = _obj("bowl", "basket", 0, (0.2, -0.1, 0.77), (0.1, 0.1, 0.04), typ="Geometry", mesh=(V, F))
    prof = wall_profile(b)
    assert prof is not None and abs(prof["rim"] - 0.08) < 1e-6 and abs(prof["floor"]) < 1e-6
    centre = np.array([[0.2, -0.1, 0.785]])
    assert abs(wall_clearance(b, prof, centre) - 0.015) < 1e-3                      # 15 mm over the floor
    assert inside_wall(b, prof, centre)
    assert not inside_wall(b, prof, np.array([[0.2 + 0.095, -0.1, 0.80]]))          # in the wall near the rim
    assert inside_wall(b, prof, np.array([[0.2 + 0.095, -0.1, 0.86]]))              # above the rim
    assert inside_wall(b, prof, np.array([[0.2 + 0.3, -0.1, 0.70]]))                # outside the bowl


def test_link_slabs_follow_the_points():
    from dojo.skills.motion import link_slabs
    # an L: a vertical shell 4 cm deep plus a lip that sticks out at the top
    P = np.vstack([np.c_[np.zeros(200), np.random.default_rng(0).uniform(0.06, 0.1, 200), np.linspace(0.05, 0.16, 200)],
                   np.c_[np.zeros(40), np.linspace(0.045, 0.08, 40), np.full(40, 0.17)]])
    lk = {"bbox_min": P.min(0), "bbox_max": P.max(0), "points": P}
    slabs = link_slabs(lk)
    assert len(slabs) >= 5
    low = [s for s in slabs if s[1][2] < 0.15]
    assert all(s[0][1] >= 0.06 - 1e-9 for s in low)                                # the lip only widens the top slab


def test_actors_moving_from_gt_joints():
    from dojo.skills.skills import GTProvider

    class Env:
        def __init__(self):
            self.q = {"support_arm0": [0.0] * 7}

        def actors_gt(self):
            return {k: list(v) for k, v in self.q.items()}

    env = Env()
    p = GTProvider(env, None)
    assert p.actors_moving() is True                                               # first call: unknown -> moving
    assert p.actors_moving() is False
    env.q["support_arm0"][3] = 0.01
    assert p.actors_moving() is True
    assert p.actors_moving() is False
    assert GTProvider(object(), None).actors_moving() is None                      # cannot tell -> fixed wait


def test_inside_xy_is_the_checkers_footprint():
    from dojo.skills.interactive import inside_xy
    c = _obj("basket", "basket", 5, (-0.4, -0.15, 0.805), (0.15, 0.1, 0.0385), typ="Geometry",
             quat=(0.7071, 0, 0, 0.7071))                                          # turned 90 deg: 0.2 x 0.3 m in world
    assert inside_xy(_obj("a", "phone", 0, (-0.4, -0.27, 0.78), (0.02, 0.02, 0.01)), c)
    assert not inside_xy(_obj("a", "phone", 0, (-0.27, -0.15, 0.78), (0.02, 0.02, 0.01)), c)
    assert not inside_xy(_obj("a", "phone", 0, (-0.4, -0.15, 0.70), (0.02, 0.02, 0.01)), c)


def test_body_holes_from_the_mesh():
    from dojo.skills.holders import body_holes
    from tests.dojo.test_skills2 import _merge
    # a 12 x 12 x 4.8 cm block with a 4 x 4 cm square hole down to a 2 mm floor
    parts = [_box_mesh((-0.06, -0.06, 0.0), (0.06, 0.06, 0.002)),
             _box_mesh((-0.06, -0.06, 0.0), (-0.02, 0.06, 0.048)), _box_mesh((0.02, -0.06, 0.0), (0.06, 0.06, 0.048)),
             _box_mesh((-0.02, -0.06, 0.0), (0.02, -0.02, 0.048)), _box_mesh((-0.02, 0.02, 0.0), (0.02, 0.06, 0.048))]
    V, F = _merge(*parts)
    h = _obj("h", "holder", 0, (0.1, -0.2, 0.766), (0.06, 0.06, 0.024), typ="Articulation", mesh=(V, F))
    holes = body_holes(h, None)
    assert len(holes) == 1
    assert np.allclose(holes[0]["center"], (0.1, -0.2), atol=0.003)
    assert abs(holes[0]["top"] - (0.766 + 0.048)) < 1e-6 and abs(holes[0]["floor"] - (0.766 + 0.002)) < 0.002


def test_gt_watch_read_shapes_and_arrival_order():
    from dojo.skills.watch import ArrivalOrder, gt_watch_read, region_cells
    w = _board_world()
    region = {"kind": "cells", "board": region_cells(w.obj("board")),
              "pieces": {"own": "Rigid/chessman/00000", "opponent": "Rigid/chessman/00001"}}
    r0 = gt_watch_read(w.objects, region)
    assert len(r0["cells"]) == 9 and not any(c["occupant"] for c in r0["cells"]) and not r0["stable_vs_prior"]
    w.obj("theirs0").pos = np.array([-0.074, -0.001, 0.781])
    r1 = gt_watch_read(w.objects, region, r0)
    assert r1["new"] == [6] and r1["cells"][6]["occupant"] == "opponent" and not r1["stable_vs_prior"]
    r2 = gt_watch_read(w.objects, region, r1)
    assert r2["new"] == [] and r2["stable_vs_prior"]
    # container reads -> arrival order
    c = _obj("b1", "basket", 5, (0.35, 0.25, 0.805), (0.15, 0.1, 0.04), typ="Geometry")
    a = _obj("a", "phone", 0, (0.0, 0.2, 0.78), (0.02, 0.02, 0.01))
    b = _obj("b", "watch", 1, (0.1, 0.2, 0.78), (0.02, 0.02, 0.01))
    objs = {o.label: o for o in (c, a, b)}
    reg = {"kind": "container", "container": {"asset": c.asset.key, "pos": c.pos.tolist(), "quat": c.quat.tolist()},
           "candidates": [a.asset.key, b.asset.key]}
    ao = ArrivalOrder(n_consecutive=2, expected=2)
    seq = [(None, None), ("b", None), ("b", None), ("b", "a"), ("b", "a")]
    for first, second in seq:
        for o in (a, b):
            o.pos = np.array([0.0 if o is a else 0.1, 0.2, 0.78])
        for lab in (first, second):
            if lab:
                objs[lab].pos = np.array([0.35, 0.25, 0.80])
        res = ao.update(gt_watch_read(objs, reg))
    assert res["order"] == [b.asset.key, a.asset.key] and res["done"]
    assert ao.update({"occluded": True, "inside": []})["order"] == res["order"]          # occluded reads change nothing


def test_container_read_is_stable_with_identical_twins():
    """Two objects of one asset (one demonstrated, one ours): a still scene must read as stable."""
    from dojo.skills.watch import gt_watch_read
    c = _obj("b1", "basket", 5, (0.35, 0.25, 0.805), (0.15, 0.1, 0.04), typ="Geometry")
    a1 = _obj("a1", "phone", 0, (0.35, 0.25, 0.80), (0.02, 0.02, 0.01))
    a2 = _obj("a2", "phone", 0, (-0.1, -0.1, 0.78), (0.02, 0.02, 0.01))
    objs = {o.label: o for o in (c, a1, a2)}
    reg = {"kind": "container", "container": {"asset": c.asset.key, "pos": c.pos.tolist(), "quat": c.quat.tolist()},
           "candidates": [a1.asset.key]}
    r0 = gt_watch_read(objs, reg)
    r1 = gt_watch_read(objs, reg, r0)
    assert r1["stable_vs_prior"] and len(r1["inside"]) == 1 and len(r1["outside"]) == 1
    a2.pos = a2.pos + np.array([0.01, 0, 0])
    assert not gt_watch_read(objs, reg, r1)["stable_vs_prior"]


def test_skill_reads_go_through_the_runtime_provider():
    """The waits and the belt read call RuntimeProvider's own methods (integration-2 #34 / #39) with the signatures
    they have: actors_moving(), watch_read(region, prior) (set by the policy), belt_velocity()."""
    from types import SimpleNamespace

    from dojo.runtime.belief import RuntimeProvider
    from dojo.runtime.estimate import objects_from_gt_rows
    from dojo.skills import interactive as I
    from dojo.skills.watch import watch
    from tests.dojo.test_runtime import STACK, Bridge, FakeAssets, FakeEpisode, _env
    env = _env()
    b = Bridge()
    ep = FakeEpisode(env)
    b.obs, b.fresh = ep.observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    clock = {"step": 0}

    class Ex:
        def hold(self, n, **kw):
            clock["step"] += n
            b.obs, b.fresh = dict(ep.observe(), steps_used=clock["step"]), True
            return n
    sk = SimpleNamespace(provider=prov, ex=Ex(), refresh=prov.refresh)
    # (a) no motion reader: unknown -> the fixed hold
    assert I.wait_actors_still(sk, fallback=7) == "fallback" and clock["step"] == 7
    # (a) a scripted arm that stops at step 30: still after a 12-step window of no motion
    prov.motion_reader = lambda a, b_: {"moving": b_["steps_used"] < 30}
    r = I.wait_actors_still(sk, still_steps=12, max_steps=200)
    assert r.startswith("still") and clock["step"] >= 30 + 11              # 12 still reads: steps 30..41
    # (b) the policy's watch_read attribute, called the way the skills call it
    calls = []
    prov.watch_read = lambda region, prior=None: calls.append((region["kind"], prior)) or {
        "occluded": False, "visible_frac": 0.9, "stable_vs_prior": True, "new": [], "inside": [], "outside": []}
    out = watch(sk, {"kind": "container", "container": {"asset": "Geometry/basket/00005", "pos": [0, 0, 0.77],
                                                      "quat": [1, 0, 0, 0]}, "candidates": []}, {"inside": []})
    assert calls == [("container", {"inside": []})] and out["visible_frac"] == 0.9
    # (c) the tracked belt velocity is used as soon as it reads as moving
    prov.velocity = {"block_0": np.array([0.004, 0.0])}
    v = I.belt_velocity(sk, ["block_0", "block_1"])
    assert np.allclose(v, [0.004, 0.0])
