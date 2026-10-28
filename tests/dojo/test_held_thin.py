"""RuntimeConfig.held_thin: a held_look "not held" resting on the jaw opening alone, for an object thinner than the
opening can resolve. The numbers are the 5.1 deposit_coin camera rows c4b_coin_off 100000 / 03: the 1.9 mm coin, held
149 mm up at the handover pose, read "opening measured 0.0, commanded 0.0" at confidence 1.0, and the grasp was dropped
from the belief (the skill gave up the handover and opened the jaws)."""
import copy

import numpy as np
import pytest
from test_runtime import STACK, FakeAssets, FakeEpisode, _env, objects_from_gt_rows

from dojo.runtime.bridge import Bridge

COIN_EXTENTS = np.array([0.0285, 0.0285, 0.0019])     # Rigid/coin/00000
# step 111 of c4b_coin_off_100000: held_look on the left wrist with the coin in the jaws (GT z 0.93)
COIN_READ = {"held": False, "confidence": 1.0, "source": "b1", "evidence": "opening",
             "opening": {"measured": 0.0, "commanded": 0.0, "score": 0.93}}
HEAD_CONFIRM = {"held": None, "confidence": 0.0, "source": "head", "calls": 0}


def test_opening_resolution_from_the_margin_and_the_finger_geometry():
    from dojo.perception.j1_adapter import HELD_CLOSED_FLOOR, HELD_OPEN_MARGIN, opening_resolution_gap
    from dojo.skills.kinematics import grip_to_gap
    full = opening_resolution_gap(0.0)
    assert full == pytest.approx(grip_to_gap(HELD_CLOSED_FLOOR + HELD_OPEN_MARGIN))
    assert 0.010 < full < 0.012                               # 10.8 mm: the coin (1.9 mm) is far below it
    assert opening_resolution_gap(None) == full and opening_resolution_gap(0.1) == full   # below the floor: closed limit
    assert opening_resolution_gap(0.5) > full                 # a partial close needs a wider object to read "held"


def _prov(label, extents, read, held_thin):
    from dojo.runtime.belief import RuntimeProvider
    env = _env()
    b = Bridge()
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    prov.held_thin = held_thin
    prov.held_checker = lambda obs, arm, lab: dict(read)
    prov.held_confirm = lambda obs, arm, lab: dict(HEAD_CONFIRM)
    bel = prov.belief
    o = bel.objects[label]
    if extents is not None:
        o.asset = copy.copy(o.asset)
        o.asset.bbox_min, o.asset.bbox_max = -np.asarray(extents, float) / 2, np.asarray(extents, float) / 2
    pose0 = np.r_[o.pos, o.quat].copy()
    z0 = float(bel.kin.tcp("left", bel.q["left"])[2, 3])
    bel.attached["left"] = (label, np.eye(4))
    bel.grasp_z["left"] = (z0 - 0.149, 1, pose0)               # lifted 149 mm since the close (GT rise 149.3)
    prov._held_check(b.obs)
    return prov, bel


def test_thin_coin_opening_only_not_held_is_dropped_today():
    prov, bel = _prov("block_1", COIN_EXTENTS, COIN_READ, "off")
    ev = prov.events[-1]
    assert "left" not in bel.attached and ev["dropped"] is True and ev.get("sure") == "jaws closed"


def test_thin_coin_opening_only_not_held_is_unknown_with_the_flag():
    prov, bel = _prov("block_1", COIN_EXTENTS, COIN_READ, "on")
    ev = prov.events[-1]
    assert "left" in bel.attached and ev["unknown"] and ev["kept"] and ev["thin_mm"] == pytest.approx(1.9)
    assert prov.stats["held_look_thin"] == 1 and "dropped" not in ev


def test_thick_object_or_other_evidence_still_drops_with_the_flag():
    # a block (the fake assets' cube) with the same opening reading: wide enough for the opening to see it
    prov, bel = _prov("block_1", None, COIN_READ, "on")
    assert "left" not in bel.attached and prov.events[-1]["dropped"] is True
    # the coin, but the verdict came from the colour test (evidence "colour"), not the opening: not this rule's case
    colour = {**COIN_READ, "evidence": "colour", "opening": None}
    prov, bel = _prov("block_1", COIN_EXTENTS, colour, "on")
    assert "left" not in bel.attached and "thin_mm" not in prov.events[-1]


def test_held_thin_default_off_and_validated():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    assert RuntimeConfig().held_thin == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(held_thin="maybe"))
