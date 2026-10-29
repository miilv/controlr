"""belt_deposit (pick_from_conveyor_by_image, parked): default off, validated, a skill switch for G3."""
import pytest


def test_belt_deposit_and_container_default_off_and_validated():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    cfg = RuntimeConfig()
    assert cfg.belt_container == "off" and cfg.belt_deposit == "off"
    assert SKILL_ATTRS["belt_deposit"] == ("off", "on")
    for bad in ({"belt_container": "maybe"}, {"belt_deposit": "maybe"}):
        with pytest.raises(ValueError):
            RuntimePolicy(RuntimeConfig(**bad))


def test_box_rim_z_ignores_a_handle_above_the_rim():
    import numpy as np
    from dojo.skills.grasps import box_rim_z
    xs, ys = np.meshgrid(np.linspace(-0.18, 0.18, 19), np.linspace(-0.12, 0.12, 13))
    body = np.vstack([np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, z)] for z in (-0.11, 0.0, 0.03)])
    handle = np.c_[np.zeros(31), np.linspace(-0.07, 0.07, 31), np.full(31, 0.11)]
    assert abs(box_rim_z(np.vstack([body, handle])) - 0.03) < 0.003
