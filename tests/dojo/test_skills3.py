"""skills3: hold quality of grasps, fine insertion fit, closed-loop insertion tail, refine guards (CPU only)."""

from __future__ import annotations

import math

import numpy as np

from tests.dojo.test_skills2 import _box_mesh, _merge, _obj


def _cyl_mesh(r, y0, y1, n=24):
    """A cylinder along y (a key's round bow)."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.c_[r * np.cos(a), np.zeros(n), r * np.sin(a)]
    V = np.vstack([ring + [0, y0, 0], ring + [0, y1, 0], [[0, y0, 0], [0, y1, 0]]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return V, np.asarray(F)


def test_mesh_com_is_the_volume_centroid():
    from dojo.skills.grasps import mesh_com
    from dojo.skills.world import AssetInfo
    # a heavy head (20 x 10 x 20 mm) on a thin shaft (5 x 5 x 40 mm): the volume centroid sits in the head
    m = _merge(_box_mesh((-0.0025, -0.0025, -0.05), (0.0025, 0.0025, -0.01)),
               _box_mesh((-0.01, -0.005, -0.01), (0.01, 0.005, 0.01)))
    info = AssetInfo("Rigid/key/00000", m[0].min(0), m[0].max(0), mesh=m)
    com = mesh_com(info)
    vol_shaft, vol_head = 5 * 5 * 40, 20 * 10 * 20
    exp = (vol_shaft * -0.03 + vol_head * 0.0) / (vol_shaft + vol_head)
    assert abs(com[2] - exp) < 5e-4
    assert com[2] > info.center[2] + 0.005          # well above the bbox centre


def test_pad_contact_flat_faces_beat_a_round_rim():
    """Pads on a box's flat faces touch over a patch; across a cylinder's diameter they touch a line, which is
    what lets a key turn in the pads (twin: the bow held on its rim swung 50 deg and fell)."""
    from dojo.skills import grasps as G
    from dojo.skills.geometry import T_from
    box = _obj("b", "block", 0, (0, -0.2, 0.8), (0.01, 0.01, 0.01), mesh=_box_mesh((-0.01,) * 3, (0.01,) * 3))
    cyl = _obj("c", "key", 0, (0, -0.2, 0.8), (0.01, 0.005, 0.01), mesh=_cyl_mesh(0.01, -0.005, 0.005))
    # approach straight down (-z), closing along world x: on the box two flat faces; on the cylinder (axis
    # along y) the curved side
    R = np.stack([[0, 0, -1.0], [1.0, 0, 0], np.cross([0, 0, -1.0], [1.0, 0, 0])], axis=1)
    T = T_from((0, -0.2, 0.8), R)
    pb, cb = G.pad_contact(box, T)
    pc, cc = G.pad_contact(cyl, T)
    assert pb > 0.4 and cb is not None
    assert pc < 0.6 * pb
    assert G.hold_quality(box, T) < G.hold_quality(cyl, T)


def test_refine_insertion_snaps_blades_into_slots_and_lowers_them():
    """Two 1.0 mm slots 12 mm apart, 15 mm deep; a plug with two 0.8 mm blades 12 mm apart, started 0.6 mm off
    across the slots with its blade tips at the surface: the fine fit shifts it onto the slots and lowers it to
    the floor."""
    from dojo.skills.geometry import T_from
    from dojo.skills.insertion import refine_insertion
    top, floor = 0.02, 0.005
    xs = (-0.0065, -0.0055, 0.0055, 0.0065)                       # slot walls: [-6.5, -5.5] and [5.5, 6.5] mm
    walls = [_box_mesh((-0.03, -0.02, 0.0), (0.03, 0.02, floor)),
             _box_mesh((-0.03, -0.02, floor), (xs[0], 0.02, top)),
             _box_mesh((xs[1], -0.02, floor), (xs[2], 0.02, top)),
             _box_mesh((xs[3], -0.02, floor), (0.03, 0.02, top)),
             _box_mesh((xs[0], -0.02, floor), (xs[1], -0.003, top)), _box_mesh((xs[0], 0.003, floor), (xs[1], 0.02, top)),
             _box_mesh((xs[2], -0.02, floor), (xs[3], -0.003, top)), _box_mesh((xs[2], 0.003, floor), (xs[3], 0.02, top))]
    strip = _obj("strip", "socket", 0, (0.1, -0.1, 0.765), (0.03, 0.02, 0.01), mesh=_merge(*walls))
    body = _box_mesh((-0.015, -0.01, 0.0), (0.015, 0.01, 0.02))
    blades = [_box_mesh((-0.0064, -0.0025, -0.015), (-0.0056, 0.0025, 0.0)),
              _box_mesh((0.0056, -0.0025, -0.015), (0.0064, 0.0025, 0.0))]
    plug = _obj("plug", "charger", 0, (0, 0, 0), (0.015, 0.01, 0.02), mesh=_merge(body, *blades))
    T0 = T_from((0.1 + 0.0006, -0.1, 0.765 + top + 0.015), np.eye(3))      # tips at the surface, 0.6 mm off in x
    out = refine_insertion(plug, strip, T0, half=0.002, res=0.0001, yaws_deg=(0,))
    assert out is not None
    T, info = out
    assert info["inside"] >= 0.999
    assert abs(T[0, 3] - 0.1) < 0.00025
    assert abs((T[2, 3] - 0.015) - (0.765 + floor)) < 0.0006            # tips on the floor


def test_no_refine_labels_skip_the_hook():
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills
    rows = [{"label": "t", "type": "Rigid", "category": "mahjong", "category_idx": 0, "pos": [0.1, -0.2, 0.8],
             "quat": [1, 0, 0, 0], "half": [0.01, 0.01, 0.01], "center": [0, 0, 0]}]
    env = KinematicEnv(rows, step_lim=100)
    calls = []
    sk = Skills(env, GTProvider(env, None), refine=lambda s, l, a, k: calls.append((l, k)))
    sk._refine("t", "left", "pick")
    sk.no_refine.add("t")
    sk._refine("t", "left", "pick")
    assert calls == [("t", "pick")]


def test_sag_is_identity_when_the_arm_tracks():
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills
    env = KinematicEnv([], step_lim=100)
    sk = Skills(env, GTProvider(env, None))
    C = sk._sag("left", sk.refresh())
    assert np.allclose(C, np.eye(4), atol=1e-6)


def test_planner_still_routes_insert_and_plug():
    """Both precision insertions keep going through insert_into (servo tail, fine fit)."""
    from dojo.skills.planner import plan
    from tests.dojo.test_skills2 import _world
    key = _obj("key", "key", 0, (0.15, -0.2, 0.77), (0.011, 0.005, 0.031))
    slot = _obj("slot", "key_slot", 0, (-0.14, -0.15, 0.765), (0.05, 0.05, 0.0275), "Geometry")
    p = plan("Pick up the key, hand it over to the other hand, insert it into the keyhole, then turn it.",
             _world([key, slot]))
    assert p.skill == "insert_into" and p.args["obj"] == "key" and p.args["turn_deg"] == -60.0


def test_plan_stack_ordered_keeps_the_list_order():
    """ordered=True: objects[0] is the base (it stays), the rest go on in list order."""
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills
    from dojo.skills.stacking import plan_stack
    from dojo.skills.library import _on_top_of
    rows = [{"label": l, "type": "Rigid", "category": "cube", "category_idx": 0, "pos": p, "quat": [1, 0, 0, 0],
             "half": [0.02, 0.02, 0.02], "center": [0, 0, 0]}
            for l, p in (("red", [0.15, -0.2, 0.785]), ("green", [-0.1, -0.15, 0.785]), ("blue", [0.05, -0.25, 0.785]))]
    env = KinematicEnv(rows, step_lim=500)
    sk = Skills(env, GTProvider(env, None))
    jobs = plan_stack(sk, ["green", "blue", "red"], 0.9, _on_top_of(4), yaw_sym=4, ordered=True)
    assert jobs
    placed = [j.label for j in jobs]
    assert placed[-2:] == ["blue", "red"] and "green" not in placed[-2:]


def test_planner_toaster_two_breads_one_per_slot_then_the_lever():
    from dojo.skills.planner import plan
    from tests.dojo.test_skills2 import _world
    breads = [_obj(f"bread_{i}", "bread", 0, (-0.2 + 0.03 * i, -0.1, 0.8), (0.05, 0.05, 0.008)) for i in range(4)]
    toaster = _obj("toaster", "toaster", 0, (0.2, -0.08, 0.84), (0.1, 0.06, 0.08), "Articulation",
                   functional={"toast_slot1": {"frame": [[-0.078, -0.041, -0.051, 1, 0, 0, 0], [0.072, -0.011, -0.051, 1, 0, 0, 0]]},
                               "toast_slot2": {"frame": [[-0.078, 0.014, -0.051, 1, 0, 0, 0], [0.072, 0.039, -0.051, 1, 0, 0, 0]]}})
    toaster.asset.active_functional = {"toast_botton": {"parent_joint": "joint_1", "frame": [[0.002, 0.004, -0.001, 1, 0, 0, 0]]}}
    p = plan("Pick up two slices of bread, place them into the toaster, and press the lever down.",
             _world(breads + [toaster]))
    assert p.skill == "slots_and_press"
    assert len(p.args["objects"]) == 2 and p.args["container"] == "toaster"
    assert sorted(p.args["slots"]) == ["toast_slot1", "toast_slot2"] and p.args["press_tag"] == "toast_botton"


def test_planner_pieces_onto_pegs_by_group_size_and_peg_height():
    from dojo.skills.planner import plan
    from tests.dojo.test_skills2 import _world
    base = _obj("base", "stack_base", 0, (0, -0.2, 0.765), (0.13, 0.03, 0.03), "Geometry")
    base.asset.support = {f"pole/{i}": [(np.array([-0.096 + 0.065 * i, 0, z, 1, 0, 0, 0]), 0.005)]
                          for i, z in enumerate((0.063, 0.053, 0.043, 0.028))}
    pieces = []
    for idx, n in ((0, 4), (1, 3), (2, 2), (3, 1)):
        pieces += [_obj(f"p{idx}{j}", "stackingblocks", idx, (0.1 * j - 0.2, -0.1 - 0.03 * idx, 0.77), (0.03, 0.03, 0.0065))
                   for j in range(n)]
    p = plan("Place all stacking toy pieces onto the correct pegs.", _world(pieces + [base]))
    assert p.skill == "stack_on_pegs" and p.args["base"] == "base"
    assert [len(g) for g in p.args["groups"]] == [4, 3, 2, 1]
    assert p.args["pegs"] == ["pole/0", "pole/1", "pole/2", "pole/3"]


def test_planner_stack_in_colour_order():
    from dojo.skills.planner import plan
    from tests.dojo.test_skills2 import _world
    cols = {"r": (255, 0, 0), "g": (0, 255, 0), "b": (0, 0, 255)}
    objs = []
    for i, (lab, rgb) in enumerate(cols.items()):
        o = _obj(lab, "cube", i + 1, (0.1 * i, -0.2, 0.78), (0.0175, 0.0175, 0.0175))
        o.asset.rgb = np.asarray(rgb, float)
        objs.append(o)
    p = plan("stack the blocks from bottom to top in the order of green, blue, and red, then reset the robot arm.",
             _world(objs))
    assert p.skill == "stack_objects" and p.args["ordered"] is True
    assert p.args["objects"] == ["g", "b", "r"]


def test_rests_stably_needs_a_footprint_wide_for_the_com_height():
    """A tall body on two thin blades (a charger standing on its prongs) does not rest; a low box does."""
    from dojo.skills.library import rests_stably
    body = _box_mesh((-0.021, 0.0, -0.0135), (0.021, 0.048, 0.0135))
    blades = [_box_mesh((-0.0063, -0.015, -0.003), (-0.0055, 0.0, 0.003)),
              _box_mesh((0.0055, -0.015, -0.003), (0.0063, 0.0, 0.003))]
    plug = _obj("plug", "charger", 0, (0, -0.2, 0.8), (0.021, 0.032, 0.0135), mesh=_merge(body, *blades))
    R_blades_down = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])     # body y -> world z
    assert not rests_stably(plug, R_blades_down)
    box = _obj("box", "block", 0, (0, -0.2, 0.8), (0.03, 0.03, 0.01), mesh=_box_mesh((-0.03, -0.03, -0.01), (0.03, 0.03, 0.01)))
    assert rests_stably(box, np.eye(3))


def test_wrist_safe():
    from dojo.skills.skills import wrist_safe
    assert wrist_safe([0, 1.2, 1.0, 0.5, 0.0, 1.0])
    assert not wrist_safe([0, 1.2, 1.0, 1.7, 0.0, 1.0])
    assert not wrist_safe([0, 1.2, 1.0, 0.5, 0.0, 3.14])


def test_push_contact_height_follows_the_trailing_face():
    """A wedge 5 mm tall at its front edge, 22 mm at the back: a push towards +y meets the 5 mm front face."""
    from dojo.skills.library import _face_height
    V = np.array([[-0.15, -0.08, 0.0], [0.15, -0.08, 0.0], [-0.15, 0.08, 0.0], [0.15, 0.08, 0.0],
                  [-0.15, -0.08, 0.005], [0.15, -0.08, 0.005], [-0.15, 0.08, 0.022], [0.15, 0.08, 0.022]])
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    kb = _obj("kb", "keyboard", 0, (0.0, -0.18, 0.765), (0.15, 0.08, 0.011), mesh=(V, F))
    h_front = _face_height(kb, np.array([0.0, 1.0]))
    h_back = _face_height(kb, np.array([0.0, -1.0]))
    assert h_front is not None and h_front < 0.008
    assert h_back is not None and h_back > 0.018
