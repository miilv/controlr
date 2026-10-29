"""Stream H: ``RuntimeConfig.egg_carry`` / ``Skills.egg_carry`` ("v1" = today | "slow": ``holders._fill_drop`` lifts and
carries the egg at ``EGG_CARRY_RAD``). The effect is screened in G3; here: the default, validation, plumbing. CPU only."""

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.holders import EGG_CARRY_RAD
from dojo.skills.motion import CARRY, MEDIUM
from dojo.skills.world import AssetInfo


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015), bbox_max=np.full(3, 0.015))


def test_slow_is_slower_than_the_defaults():
    assert EGG_CARRY_RAD < MEDIUM < CARRY


def test_default_validation_and_plumbing():
    assert RuntimeConfig().egg_carry == "v1"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", egg_carry="fast"), assets=_Assets())
    from dojo.skills.env import KinematicEnv
    env = KinematicEnv([{"label": "b", "category": "block", "category_idx": 5, "pos": (0.3, -0.1, 0.78),
                         "quat": [1, 0, 0, 0], "half": [0.015] * 3}], step_lim=50, instruction="Stack the blocks.")
    pol = RuntimePolicy(RuntimeConfig(mode="gt", verify="off", egg_carry="slow"), assets=_Assets())
    pol.reset({"task": "x", "instruction": env.instruction, "step_lim": 50, "layout_id": 1})
    obs = {"images": {c: np.zeros((8, 8, 3), np.uint8) for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
           "state": env.proprio(), "instruction": env.instruction, "steps_used": 0, "step_lim": 50,
           "_dev_gt": env.gt_objects()}
    pol.next(obs)
    assert pol._live["sk"].egg_carry == "slow"
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["egg_carry"][0] == RuntimeConfig().egg_carry
