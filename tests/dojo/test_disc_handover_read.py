"""Skills.disc_handover_read: the taker's coin insertion after an in-air handover asks for the in-hand read."""
import pytest


class _Sk:
    pass


def test_taker_place_kw_asks_for_the_held_read_only_when_on():
    from dojo.skills.precision import DISC_CARRY, DISC_PLACE_KW, taker_place_kw
    sk = _Sk()
    kw = taker_place_kw(sk, DISC_PLACE_KW)
    assert kw["approach_rad"] == DISC_CARRY and "held_read" not in kw and kw["servo"] is True
    sk.disc_handover_read = "on"
    assert taker_place_kw(sk, DISC_PLACE_KW)["held_read"] is True
    assert "held_read" not in DISC_PLACE_KW                  # the shared defaults stay untouched


def test_disc_handover_read_default_off_and_validated():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    assert RuntimeConfig().disc_handover_read == "off" and SKILL_ATTRS["disc_handover_read"] == ("off", "on")
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(disc_handover_read="maybe"))
