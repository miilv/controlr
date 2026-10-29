"""store_laptop's laptop stage (route "store_laptop", off): the checker's frames (M1) and the closed geometry (M2)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills import artgeom, artroot, store
from dojo.skills.world import AssetInfo, ObjectState, WorldState

Z = 0.765
KEY = "Articulation/laptop/00000"


def _laptop(pos=(0.05, 0.02, Z + 0.09)):
    """Laptop/00000 in brief: body 180 x 126 x 6 mm, lid open at -110 deg (authored), root link 77 mm behind the
    asset origin's front, ``bottom`` on the body link."""
    body = np.array([[x, y, z] for x in (-0.09, 0.09) for y in (-0.078, 0.048) for z in (0.0, 0.006)])
    d = np.array([0.0, math.sin(math.radians(20)), math.cos(math.radians(20))])   # the lid leans 20 deg back (110 open)
    hinge = np.array([0.0, 0.04644, 0.00373])
    lid0 = np.array([np.array([x, 0, 0]) + hinge + t * d + n * np.array([0, d[2], -d[1]])
                     for x in (-0.09, 0.09) for t in (0.0, 0.12) for n in (0.0, 0.005)])
    info = AssetInfo(key=KEY, bbox_min=np.array([-0.09, -0.078, 0.0]), bbox_max=np.array([0.09, 0.09, 0.121]),
                     active_functional={"bottom": {"base_link": "E_body_1", "frame": [[0.0, 0.001, -0.001, 1, 0, 0, 0]]}},
                     functional={"laptop_hinge": {"base_link": "E_displayer_5", "parent_joint": "J",
                                                  "frame": [[0.1, 0.0, 0.0, 1, 0, 0, 0]]}},
                     links={"E_body_1": {"bbox_min": body.min(0), "bbox_max": body.max(0), "points": body},
                            "E_displayer_5": {"bbox_min": lid0.min(0), "bbox_max": lid0.max(0), "points": lid0}},
                     joints={"J": {"type": "revolute", "body0": "E_body_1", "body1": "E_displayer_5", "axis": [1.0, 0, 0],
                                   "origin": [-0.069, 0.04644, 0.00373], "lower": -110.0, "upper": 0.0, "position": -110.0}})
    return ObjectState("laptop", "Articulation", "laptop", 0, np.asarray(pos, float), np.array([1.0, 0, 0, 0]), info)


def _rack():
    y = math.radians(101.5)
    info = AssetInfo(key="Geometry/vertical_laptop_storage_rack/00000", bbox_min=np.array([-0.068, -0.026, -0.024]),
                     bbox_max=np.array([0.068, 0.027, 0.024]))
    return ObjectState("rack", "Geometry", "vertical_laptop_storage_rack", 0, np.array([0.33, -0.05, 0.79]),
                       np.array([math.cos(y / 2), 0, 0, math.sin(y / 2)]), info)


@pytest.fixture(autouse=True)
def _root(monkeypatch):
    monkeypatch.setattr(artroot, "table", lambda: {KEY: {"prim": "/root/E_body_1", "pos": [0.0, -0.0771, 0.0035],
                                                         "quat": [1.0, 0.0, 0.0, 0.0]}})


def test_the_checker_reads_bottom_on_its_base_link():
    o = _laptop()
    p = artgeom.checker_point(o, "bottom")[0][:3, 3]
    assert np.allclose(p, [0.0, -0.0761, 0.0025], atol=1e-4)       # the body's front edge, not its middle


def test_upright_targets_meet_the_checkers_rack_and_up_conditions():
    o, rk = _laptop(), _rack()
    w = WorldState("", "", {"laptop": o, "rack": rk}, {}, {})
    tg = store.upright_targets(SimpleNamespace(world=w), "laptop", "rack")
    assert tg and all(store.checker_ok(o, T, rk) for T in tg)
    # the frame's own offset (the body's middle) would sit 77 mm higher, above the 48 mm tall rack box
    T = tg[0]
    naive = (T @ np.r_[0.0, 0.001, -0.001, 1.0])[:3]
    assert naive[2] > rk.pos[2] + rk.asset.bbox_max[2]


def test_posed_asset_shuts_the_lid_over_the_body():
    """M2: the lid link turned about the hinge from its authored -110 deg to 0 lies on the body: a thin slab."""
    o = _laptop()
    shut = artgeom.posed_asset(o.asset, "J", 0.0)
    ext = shut.extents
    assert ext[2] < 0.03 and ext[0] == pytest.approx(0.18, abs=1e-6)          # 12 cm tall open -> under 3 cm shut
    lid = shut.links["E_displayer_5"]
    assert lid["bbox_max"][2] < 0.03 and lid["bbox_min"][1] < -0.06            # over the body, towards its front
    assert shut.mesh is None and o.asset.extents[2] == pytest.approx(0.121)    # the original is unchanged
    assert artgeom.posed_asset(o.asset, "nope", 0.0) is o.asset


def test_measured_joint_poses_the_world_object(monkeypatch):
    """``Skills.artic_geom`` "measured": refresh gives the laptop the measured joint's geometry; "authored" keeps it."""
    from dojo.skills.skills import Skills
    for mode, tall in (("authored", True), ("measured", False)):
        o = _laptop()
        o.joints = np.array([math.radians(-2.0)])                                # measured: shut (radians)
        w = WorldState("", "", {"laptop": o}, {}, {})
        sk = Skills.__new__(Skills)
        sk.provider = SimpleNamespace(refresh=lambda holding=None, w=w: w)
        sk.held = {"left": None, "right": None}
        sk.coll = SimpleNamespace(world=None)
        sk.artic_geom = mode
        out = sk.refresh().obj("laptop")
        assert bool(out.asset.extents[2] > 0.1) is tall


def test_route_off_by_default_and_on():
    from dojo.skills import planner
    instr = "Hang the headphones on the headphone stand, close the laptop, then place it into the vertical laptop stand."
    stand = ObjectState("stand", "Geometry", "headset_stand", 0, np.array([-0.3, 0.04, Z + 0.14]), np.array([1.0, 0, 0, 0]),
                        AssetInfo(key="Geometry/headset_stand/00000", bbox_min=np.array([-0.09, -0.05, -0.14]),
                                  bbox_max=np.array([0.09, 0.05, 0.14]),
                                  functional={"support_cradle": {"frame": [[0.0, 0.0, 0.12, 1, 0, 0, 0]]}}))
    hs = ObjectState("headset", "Rigid", "headset", 0, np.array([0.0, -0.2, Z + 0.04]), np.array([1.0, 0, 0, 0]),
                     AssetInfo(key="Rigid/headset/00000", bbox_min=np.full(3, -0.05), bbox_max=np.full(3, 0.05),
                               active_functional={"headband_bridge": {"frame": [[0.0, 0.0, 0.09, 1, 0, 0, 0]]}}))
    lap_stand = ObjectState("laptop_stand", "Geometry", "laptop_stand", 0, np.array([0.05, 0.02, Z + 0.02]),
                            np.array([1.0, 0, 0, 0]), AssetInfo(key="Geometry/laptop_stand/00001",
                                                                bbox_min=np.array([-0.1, -0.1, -0.02]),
                                                                bbox_max=np.array([0.1, 0.1, 0.02])))
    objs = {o.label: o for o in (_laptop(), _rack(), stand, hs, lap_stand)}
    w_off = WorldState("", instr, objs, {}, {})
    assert planner.plan(instr.lower(), w_off).skill != "store_laptop"
    w_on = WorldState("", instr, objs, {}, {}, notes={"routes": ["store_laptop"]})
    p = planner.plan(instr.lower(), w_on)
    assert p.skill == "store_laptop" and p.args["laptop"] == "laptop" and p.args["rack"] == "rack"
    assert p.args["hang"]["rack"] == "stand" and p.args["hang"]["objects"] == ["headset"]
    assert p.args["hinge_tag"] == "laptop_hinge"


def test_overhang_and_the_overhang_edge_pinch():
    """The laptop's footprint past its stand along a direction; a pinch counts only past the stand's footprint."""
    o = _laptop(pos=(0.05, 0.02, Z + 0.05))
    o.asset = artgeom.posed_asset(o.asset, "J", 0.0)
    st = ObjectState("laptop_stand", "Geometry", "laptop_stand", 0, np.array([0.05, 0.02, Z + 0.02]),
                     np.array([1.0, 0, 0, 0]), AssetInfo(key="Geometry/laptop_stand/00001",
                                                         bbox_min=np.array([-0.15, -0.06, -0.03]),
                                                         bbox_max=np.array([0.15, 0.06, 0.03])))
    oh = store.overhang(o, st, np.array([0.0, -1.0]))
    assert oh == pytest.approx(0.018, abs=1e-3)                    # front edge -0.078 vs stand -0.06
    ok = store.overhang_edge(None, o, st)
    from dojo.skills.geometry import T_from
    inside = SimpleNamespace(T=T_from([0.05, -0.03, Z + 0.05], np.eye(3)))
    outside = SimpleNamespace(T=T_from([0.05, -0.075, Z + 0.05], np.eye(3)))
    T_tcp_obj = np.linalg.inv(outside.T) @ o.T
    assert ok(outside, T_tcp_obj, None, None) and not ok(inside, np.linalg.inv(inside.T) @ o.T, None, None)
