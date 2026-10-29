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
    lid0 = np.array([[x, y, z] for x in (-0.09, 0.09) for y in (0.045, 0.09) for z in (0.001, 0.121)])
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
