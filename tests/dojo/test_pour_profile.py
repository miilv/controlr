"""CPU contract for opt-in multi-bottle profiles (no simulator or model calls)."""
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.g3 import SKILL_ATTRS
from dojo.skills.planner import pour_bottle_call


def test_profile_default_preserves_legacy_dispatch():
    assert RuntimeConfig().ported_pour_profile == "off"
    assert SKILL_ATTRS["ported_pour_profile"] == ("off", "sparse", "via")
    for aim in ("v1", "b10", "b20"):
        for pace in ("v1", "lean", "overlap"):
            assert pour_bottle_call(aim, pace) == pour_bottle_call(aim, pace, "off")


@pytest.mark.parametrize("profile,via", [("sparse", False), ("via", True)])
def test_new_profile_dispatch_and_aim(profile, via):
    assert pour_bottle_call("b10", "v1", profile) == (
        "pour_bottle_v2", {"via": via, "spout_back": .01})


def test_bad_profile_is_rejected():
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(ported_pour_profile="bad"), assets=object())
