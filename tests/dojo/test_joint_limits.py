"""Stream H: ``RuntimeConfig.joint_limits`` / ``Skills.joint_limits`` ("wide" = today | "measured": joint 5 inside the
sim's +-pi/2 stop). The effect is screened in G3; here: the IK honours the range, the default, plumbing. CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.kinematics import J5_STOP, JOINT_LIMIT_SETS, JOINT_LIMITS, X5Kin
from dojo.skills.skills import Skills
from dojo.skills.world import AssetInfo


def test_the_sets_differ_only_in_joint_5():
    wide, meas = JOINT_LIMIT_SETS["wide"], JOINT_LIMIT_SETS["measured"]
    assert np.array_equal(wide, JOINT_LIMITS)
    assert np.array_equal(np.delete(wide, 4, axis=0), np.delete(meas, 4, axis=0))
    assert meas[4, 1] < J5_STOP and meas[4, 0] > -J5_STOP and wide[4, 1] > J5_STOP


def test_measured_ik_never_commands_joint_5_past_the_stop():
    kin = X5Kin()
    q5 = np.array([0.0, 1.2, 0.6, 0.5, 1.8, 0.0])          # a pose the wide range allows: joint 5 at 1.8 rad
    T = kin.tcp("left", q5)
    wide = kin.ik("left", T, q5)
    assert wide.ok and abs(wide.q[4]) > J5_STOP
    sk = SimpleNamespace(kin=kin, _joint_limits="wide")
    Skills.joint_limits.fset(sk, "measured")
    assert "_ik_cache" not in kin.__dict__ and sk._joint_limits == "measured"
    r = kin.ik("left", T, q5)
    assert r.q is None or abs(r.q[4]) <= 1.56 + 1e-9
    with pytest.raises(ValueError):
        Skills.joint_limits.fset(sk, "tight")


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015), bbox_max=np.full(3, 0.015))


def test_defaults_validation_and_plumbing():
    assert RuntimeConfig().joint_limits == "wide"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", joint_limits="tight"), assets=_Assets())
    from dojo.skills.env import KinematicEnv
    env = KinematicEnv([{"label": "b", "category": "block", "category_idx": 5, "pos": (0.3, -0.1, 0.78),
                         "quat": [1, 0, 0, 0], "half": [0.015] * 3}], step_lim=50, instruction="Stack the blocks.")
    pol = RuntimePolicy(RuntimeConfig(mode="gt", verify="off", joint_limits="measured"), assets=_Assets())
    pol.reset({"task": "x", "instruction": env.instruction, "step_lim": 50, "layout_id": 1})
    obs = {"images": {c: np.zeros((8, 8, 3), np.uint8) for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
           "state": env.proprio(), "instruction": env.instruction, "steps_used": 0, "step_lim": 50,
           "_dev_gt": env.gt_objects()}
    pol.next(obs)
    sk = pol._live["sk"]
    assert sk.joint_limits == "measured" and sk.kin.limits[4, 1] == pytest.approx(1.56)


def test_g3_skill_attrs_match_the_runtime_default():
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["joint_limits"][0] == RuntimeConfig().joint_limits
