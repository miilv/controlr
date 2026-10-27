"""Precision stream (2026-10-27): deposit_coin's handover keeps the taker's hand pointing down and carries a disc that slid
in the giver's pads slower."""

import math

import numpy as np

from dojo.skills import precision as P
from dojo.skills.geometry import T_from, axis_angle_R, inv_T
from dojo.skills.kinematics import TABLE_Z, down_R
from dojo.skills.world import AssetInfo, ObjectState


def test_hand_down_first_puts_the_spin_with_the_hand_straight_down_first():
    info = AssetInfo(key="Rigid/coin/00000", bbox_min=np.array([-0.0142, -0.00095, -0.0142]),
                     bbox_max=np.array([0.0142, 0.00095, 0.0142]), mesh=None)
    coin = ObjectState("coin0", "Rigid", "coin", 0, np.array([0.3, -0.15, TABLE_Z + 0.1]), np.array([1.0, 0, 0, 0]), info)
    assert P.disc_axis(coin) == 1
    T0 = coin.T
    # held by its top rim from above: the TCP 2.6 mm over the rim, approach (TCP x) straight down
    T_tcp = T_from(T0[:3, 3] + np.array([0.0, 0.0, 0.0142 + 0.0026]), down_R(0.0))
    Tto = inv_T(T_tcp) @ T0
    # the slot's target turned 120 deg about the disc's normal, then all 12 spins of it (the first leans the hand)
    Rn = np.eye(4)
    Rn[:3, :3] = axis_angle_R(np.array([0.0, 1.0, 0.0]), math.radians(120))
    tg = P.spin_about_axis(coin, T0 @ Rn, 1)
    lean = lambda T: math.degrees(math.acos(-float((T @ inv_T(Tto))[2, 0])))
    assert lean(tg[0]) > 100
    best = P.hand_down_first(tg, Tto)
    assert lean(best[0]) < 1.0 and len(best) == len(tg)
    assert [round(lean(T)) for T in best] == sorted(round(lean(T)) for T in best)


def _coin():
    info = AssetInfo(key="Rigid/coin/00000", bbox_min=np.array([-0.0142, -0.00095, -0.0142]),
                     bbox_max=np.array([0.0142, 0.00095, 0.0142]), mesh=None)
    return ObjectState("coin0", "Rigid", "coin", 0, np.array([0.3, -0.15, TABLE_Z + 0.1]), np.array([1.0, 0, 0, 0]), info)


def _carry_rate(drift_m: float) -> float:
    """The giver's carry rate run_disc_handover picks for a disc that slid ``drift_m`` in its pads at the pick (a fake
    Skills: the pick succeeds, the carry's motion records its rate and stops the episode)."""
    from types import SimpleNamespace

    from dojo.skills.executor import BudgetExceeded
    from dojo.skills.skills import Held, PickPlan, SkillResult
    coin = _coin()
    Tto = T_from(np.array([0.0, 0.0, 0.03]), np.eye(3))
    pp = PickPlan("right", None, None, Tto, 10, "coin0")
    held = Tto.copy()
    held[0, 3] += drift_m
    rates = []

    def joint(arm, q0, q1, rate, name, **kw):
        rates.append(rate)
        raise BudgetExceeded("stop")
    sk = SimpleNamespace(held={"right": Held("coin0", held, None), "left": None}, last_pick=pp,
                         world=SimpleNamespace(obj=lambda label: coin), motion=SimpleNamespace(joint=joint),
                         ex=SimpleNamespace(q={"right": np.zeros(6), "left": np.zeros(6)}),
                         _record=lambda r: r, run_pick=lambda p: SkillResult(True, "pick", "ok", 10, "right"))
    h = {"giver": "right", "taker": "left", "pick": pp, "q_g": np.zeros(6)}
    r = P.run_disc_handover(sk, h)
    assert r.status == "step_budget" and len(rates) == 1
    return rates[0]


def test_a_disc_that_slid_in_the_givers_pads_is_carried_slower():
    assert _carry_rate(0.0005) == P.DISC_CARRY
    assert _carry_rate(0.005) == P.DISC_CARRY_DRIFTED < P.DISC_CARRY
