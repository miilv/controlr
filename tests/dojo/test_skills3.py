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
