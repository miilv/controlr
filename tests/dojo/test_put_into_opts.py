"""``RuntimeConfig.put_into_opts`` -> ``Skills.put_into_opts`` (``library.put_into``: "spot", "patient", "sequence"; empty =
today) and its helpers. The options' effect is screened in G3 (physics); here: plumbing, validation, geometry. CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.library import PUT_PICK_CLEAR, _pick_width
from dojo.skills.world import AssetInfo


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015), bbox_max=np.full(3, 0.015))


def test_options_are_validated_and_off_by_default():
    assert RuntimeConfig().put_into_opts == [] and RuntimeConfig().drop_verdict == "belief"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", put_into_opts=["hurry"]), assets=_Assets())
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", drop_verdict="maybe"), assets=_Assets())
    RuntimePolicy(RuntimeConfig(mode="perception", put_into_opts=["spot", "patient", "sequence"], drop_verdict="release"),
                  assets=_Assets())


def test_the_skills_instance_gets_the_options():
    from dojo.skills.env import KinematicEnv
    env = KinematicEnv([{"label": "b", "category": "block", "category_idx": 5, "pos": (0.3, -0.1, 0.78),
                         "quat": [1, 0, 0, 0], "half": [0.015] * 3}], step_lim=50, instruction="Stack the blocks.")
    pol = RuntimePolicy(RuntimeConfig(mode="gt", verify="off", put_into_opts=["spot", "sequence"], drop_verdict="release"),
                        assets=_Assets())
    pol.reset({"task": "x", "instruction": env.instruction, "step_lim": 50, "layout_id": 1})
    obs = {"images": {c: np.zeros((8, 8, 3), np.uint8) for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
           "state": env.proprio(), "instruction": env.instruction, "steps_used": 0, "step_lim": 50,
           "_dev_gt": env.gt_objects()}
    pol.next(obs)
    sk = pol._live["sk"]
    assert sk.put_into_opts == {"spot", "sequence"} and sk.drop_verdict == "release"


def test_pick_width_is_the_widest_object_being_picked():
    objs = {"a": SimpleNamespace(asset=SimpleNamespace(extents=np.array([0.055, 0.055, 0.17]))),
            "b": SimpleNamespace(asset=SimpleNamespace(extents=np.array([0.03, 0.03, 0.03])))}
    sk = SimpleNamespace(world=SimpleNamespace(obj=lambda l: objs[l]))
    assert _pick_width(sk, ["a", "b"]) == pytest.approx(0.17) and _pick_width(sk, ["zz"]) == 0.0
    assert PUT_PICK_CLEAR == 0.12                    # Skills.parallel's own arm clearance, not a new tuned number
