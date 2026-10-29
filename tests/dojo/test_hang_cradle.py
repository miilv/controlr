"""``Skills.hang_cradle`` "straddle" (default "strict"): a cradle hang's held object may straddle the rack's top
(``Collision.held_skip``); the arm and the hand still avoid it."""

from __future__ import annotations

import numpy as np

from dojo.skills.kinematics import ARMS, X5Kin
from dojo.skills.motion import Collision, HeldBox
from dojo.skills.world import AssetInfo, ObjectState, WorldState

Z = 0.765


def _world():
    stand = ObjectState("stand", "Geometry", "headset_stand", 0, np.array([-0.2, -0.1, Z]), np.array([1.0, 0, 0, 0]),
                        AssetInfo(key="Geometry/headset_stand/00000", bbox_min=np.array([-0.02, -0.02, 0.0]),
                                  bbox_max=np.array([0.02, 0.02, 0.3])))
    return WorldState("", "", {"stand": stand}, {}, {})


def _coll():
    kin = X5Kin()
    c = Collision(kin)
    c.world = _world()
    c.include.add("stand")
    c.parts["stand"] = (np.array([[0.0, 0.0, 0.15]]), np.array([[0.02, 0.02, 0.15]]))   # the pole, one box
    return c, kin


def test_held_points_skip_the_cradle_region_and_the_hand_does_not():
    c, kin = _coll()
    arm = "left"
    q = np.zeros(6)
    T_tcp = kin.tcp_from_flange(kin.frames(arm, q)[-1])
    # a held "object" whose dense points sit on the pole's top (straddling it)
    pole_top_tcp = np.linalg.inv(T_tcp) @ np.array([-0.2, -0.1, Z + 0.28, 1.0])
    hb = HeldBox(np.eye(4), np.zeros(3), np.full(3, 0.001), points=None, dense=pole_top_tcp[None, :3])
    ok, why = c.check(arm, q, 0.5, held=hb, table=False)
    assert not ok and why == "hits stand"                                  # strict: the held object hits the pole
    c.held_skip["stand"] = (np.array([-0.2, -0.1]), 0.06, Z, Z + 0.32)
    ok2, _ = c.check(arm, q, 0.5, held=hb, table=False)
    assert ok2                                                             # straddle: the held object may be there
    # the hand itself is still checked against the pole: put the pole where the hand is
    c.world.objects["stand"].pos = np.r_[T_tcp[:2, 3], Z]
    c.held_skip["stand"] = (T_tcp[:2, 3].copy(), 0.06, Z - 1.0, Z + 1.0)
    c.parts["stand"] = (np.array([[0.0, 0.0, T_tcp[2, 3] - Z]]), np.array([[0.03, 0.03, 0.05]]))
    ok3, why3 = c.check(arm, q, 0.5, held=hb, table=False)
    assert not ok3 and why3 == "hits stand"


def test_hang_cradle_defaults_strict():
    from dojo.runtime.policy import RuntimeConfig
    from dojo.skills.g3 import SKILL_ATTRS
    from dojo.skills.skills import Skills
    assert RuntimeConfig().hang_cradle == "strict" and SKILL_ATTRS["hang_cradle"][0] == "strict"
    assert getattr(Skills.__new__(Skills), "hang_cradle", "strict") == "strict"
