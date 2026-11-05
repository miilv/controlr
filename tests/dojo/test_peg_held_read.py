"""Opt-in held-ring measurement: sub-clearance gate, guarded threading, blind fallback."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import library, pegs
from dojo.skills.env import KinematicEnv
from dojo.skills.g3 import OffsetBeliefProvider, SKILL_ATTRS
from dojo.skills.heldsim import HeldBeliefProvider
from dojo.skills.kinematics import TABLE_Z
from dojo.skills.skills import GTProvider, Skills


def _place_kw(monkeypatch, mode="on", remeasure=False):
    seen = {}
    monkeypatch.setattr(pegs, "side_bias", lambda *a: 0.0)
    monkeypatch.setattr(pegs, "piece_target", lambda *a, **kw: (np.eye(4), np.zeros(3)))
    monkeypatch.setattr(library, "place_oriented", lambda *a, **kw: seen.update(kw["place_kw"]))
    obj = SimpleNamespace(yaw=lambda: 0.0, center_w=np.zeros(3))
    sk = SimpleNamespace(peg_held_read=mode, remeasure=remeasure, refresh=lambda: None,
                         world=SimpleNamespace(obj=lambda lab: obj))
    pegs._one_piece(sk, "ring", obj, "base", {"top": 0.85}, [], False, pegs.THREAD, 0.003)
    return seen


def test_request_is_scoped_to_mode_b_and_keeps_guarded_threading(monkeypatch):
    kw = _place_kw(monkeypatch)
    assert kw["held_read"] and kw["guarded"] and kw["probe"]
    assert kw["held_gate"] == kw["held_tol"] == 0.0005
    assert "blind_up" not in kw  # never adds a deeper seating target
    for mode, remeasure in (("off", False), ("on", True)):
        kw = _place_kw(monkeypatch, mode, remeasure)
        assert not {"held_read", "held_gate", "held_tol"} & kw.keys()


@pytest.mark.parametrize("sigma,adopt", [(0.2, True), (0.501, False), (99.0, False)])
def test_requested_read_corrects_the_guarded_tail_or_falls_back(monkeypatch, sigma, adopt):
    kw = _place_kw(monkeypatch)
    row = {"ok": int(sigma < 99), "why": "", "err": [0.] * 6,
           "sig": [sigma] * 4 + [0.1]}
    env = KinematicEnv([{"label": "ring", "category": "block", "pos": [-0.22, -0.15, TABLE_Z + 0.01],
                         "quat": [1, 0, 0, 0], "half": [0.03, 0.012, 0.01]}])
    prov = HeldBeliefProvider(OffsetBeliefProvider(GTProvider(env, None), {"ring": (0.003, 0.)}), env,
                             mode="noise", table={"block": {"hover": [row]}})
    prov.held_pose_on_request = True
    sk = Skills(env, prov, remeasure=False, refine=lambda *a: None)
    sk.held_looks = False
    assert sk.run_pick(sk.plan_pick("ring", "left")).ok
    T = sk.world.obj("ring").T.copy()
    T[:3, 3] = [-0.18, -0.22, TABLE_Z + 0.01]
    kw.update(pre_h=0.04)
    r = sk.run_place(sk.plan_place("left", T, **kw))
    assert len(sk.held_events) == 1 and sk.held_events[0]["adopted"] == adopt
    assert "probing tail" in r.note
    assert ("held read adopted" in r.note) == adopt
    if adopt:
        true = next(x for x in env.gt_objects() if x["label"] == "ring")
        assert np.linalg.norm(np.asarray(true["pos"])[:2] - T[:2, 3]) < 0.0005


def test_config_and_g3_validate_the_switch():
    assert RuntimeConfig().peg_held_read == "off"
    assert SKILL_ATTRS["peg_held_read"] == ("off", "on")
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(peg_held_read="maybe"))
