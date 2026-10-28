"""Stream H: ``RuntimeConfig.stack_reach`` / ``Skills.stack_reach`` ("centre" = today | "grasp": ``stacking.plan_stack``
measures each pick's reach to its planned grasp's TCP, not the object's centre). The effect is screened in G3; here: the
plan, the default, plumbing. CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import stacking as S
from dojo.skills.kinematics import X5Kin
from dojo.skills.world import AssetInfo


def _bowl(xy):
    ext = np.array([0.16, 0.16, 0.06])
    return SimpleNamespace(center_w=np.array([xy[0], xy[1], 0.795]), asset=SimpleNamespace(extents=ext),
                           bottom_z=0.765, top_z=0.825)


class _Sk:
    """Three bowls whose centres are past MAX_REACH from the arm that can pick them, their rims within it."""

    def __init__(self, mode):
        self.kin = X5Kin()
        self.objs = {"bowl0": _bowl((-0.45, 0.10)), "bowl1": _bowl((0.45, 0.10)), "bowl2": _bowl((0.0, -0.15))}
        self.world = SimpleNamespace(obj=lambda l: self.objs[l])
        if mode:
            self.stack_reach = mode

    def reach_down(self, arm, xyz, tilts=()):
        return True

    def can_pick(self, label, arm, method="auto"):
        x = self.objs[label].center_w[0]
        return x == 0.0 or (x < 0) == (arm == "left")

    def pick_reach(self, label, arm, method="auto"):
        c = self.objs[label].center_w[:2]
        d = float(np.linalg.norm(c - self.kin.base(arm)[:2, 3]))
        return d - 0.08                          # the wall pinch at the near rim


@pytest.mark.parametrize("mode, planned", [(None, False), ("centre", False), ("grasp", True)])
def test_grasp_reach_lets_far_bowls_stack(mode, planned):
    sk = _Sk(mode)
    far = [float(np.linalg.norm(sk.objs[l].center_w[:2] - sk.kin.base(a)[:2, 3]))
           for l, a in (("bowl0", "left"), ("bowl1", "right"))]
    assert min(far) > S.MAX_REACH                       # the setup: past the cap by their centres
    jobs = S.plan_stack(sk, ["bowl0", "bowl1", "bowl2"], 0.92, lambda b, l: (lambda s: None), 8, pick_method="wall")
    assert bool(jobs) is planned
    if planned:
        assert {j.label for j in jobs} >= {"bowl0", "bowl1"}


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015), bbox_max=np.full(3, 0.015))


def test_default_validation_and_plumbing():
    assert RuntimeConfig().stack_reach == "centre"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", stack_reach="tcp"), assets=_Assets())
    from dojo.skills.env import KinematicEnv
    env = KinematicEnv([{"label": "b", "category": "block", "category_idx": 5, "pos": (0.3, -0.1, 0.78),
                         "quat": [1, 0, 0, 0], "half": [0.015] * 3}], step_lim=50, instruction="Stack the blocks.")
    pol = RuntimePolicy(RuntimeConfig(mode="gt", verify="off", stack_reach="grasp"), assets=_Assets())
    pol.reset({"task": "x", "instruction": env.instruction, "step_lim": 50, "layout_id": 1})
    obs = {"images": {c: np.zeros((8, 8, 3), np.uint8) for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
           "state": env.proprio(), "instruction": env.instruction, "steps_used": 0, "step_lim": 50,
           "_dev_gt": env.gt_objects()}
    pol.next(obs)
    sk = pol._live["sk"]
    assert sk.stack_reach == "grasp"
    assert sk.pick_reach("b", "right", "auto") > 0
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["stack_reach"][0] == RuntimeConfig().stack_reach
