"""Stream H: ``RuntimeConfig.press_repeat`` / ``Skills.press_repeat`` ("off" = today | "cross"). CPU only."""

import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import library as L


def test_default_and_validation():
    assert RuntimeConfig().press_repeat == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", press_repeat="twice"), assets=None)
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["press_repeat"] == ("off", "cross")
    assert 0.002 <= L.PRESS_REPEAT_SHIFT <= 0.006
