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


def test_hang_wide_ranks_wide_grasps_first(monkeypatch):
    """``Skills.hang_wide`` (default off): the hang's transfer gets a pair_rank that puts grasps >= HANG_WIDE_W first."""
    from types import SimpleNamespace

    from dojo.skills import library as L
    seen = {}

    def fake_transfer(lab, targets, **kw):
        seen.setdefault("rank", kw.get("pair_rank"))            # the hang plan's own call (relay legs come after)
        from dojo.skills.skills import SkillResult
        return SkillResult(False, "transfer", "ik_fail", 0, None, note="right: 0/1 grasps reachable, 0 grasp-place pairs")
    for flag, expect in ((False, None), ("on", "fn")):
        seen.clear()
        stand = ObjectState("stand", "Geometry", "headset_stand", 0, np.array([-0.2, -0.1, Z]), np.array([1.0, 0, 0, 0]),
                            AssetInfo(key="Geometry/headset_stand/00000", bbox_min=np.array([-0.02, -0.02, 0.0]),
                                      bbox_max=np.array([0.02, 0.02, 0.3]),
                                      functional={"support_cradle": {"frame": [[0.0, 0.0, 0.3, 1, 0, 0, 0]]}}))
        hs = ObjectState("headset", "Rigid", "headset", 0, np.array([0.0, -0.2, Z + 0.05]), np.array([1.0, 0, 0, 0]),
                         AssetInfo(key="Rigid/headset/00000", bbox_min=np.full(3, -0.05), bbox_max=np.full(3, 0.05),
                                   active_functional={"headband_bridge": {"frame": [[0.0, 0.0, 0.05, 1, 0, 0, 0]]}}))
        w = WorldState("", "", {"stand": stand, "headset": hs}, {}, {})
        sk = SimpleNamespace(world=w, refresh=lambda: w, log=lambda m: None, hang_wide=flag, hang_cradle="strict",
                             coll=SimpleNamespace(include=set(), parts={}, held_skip={}), plan_transfer=fake_transfer,
                             bank=lambda: None, held={"left": None, "right": None})
        L.hang_on_hooks(sk, ["headset"], "stand", "support_cradle", "headband_bridge")
        r = seen.get("rank")
        if expect is None:
            assert r is None
        else:
            wide, narrow = SimpleNamespace(width=0.04), SimpleNamespace(width=0.025)
            assert r(wide, None, None, None) == 0.0 and r(narrow, None, None, None) == L.HANG_WIDE_RANK
