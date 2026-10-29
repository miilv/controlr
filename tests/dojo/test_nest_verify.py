"""Stream H: ``RuntimeConfig.nest_verify`` / ``Skills.nest_verify`` ("target" = today | "support"): a nested place that
missed its planned pose but rests in its support (centre within 0.03 m, above it, axis within 30 deg; the checker's
is_stacked takes 0.04 m / 45 deg) counts, so the stack chain goes on. G3 stack_bowls v15 draws: 8 "misses" of 31-45 mm
were nested 5-22 mm from the lower bowl's centre. CPU only."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.geometry import rot_z
from dojo.skills.skills import Job, SkillResult, Skills


def _R(tilt_deg):
    a = math.radians(tilt_deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def _sk(mode, upper_xy, upper_z=0.83, tilt=12.0):
    lower = SimpleNamespace(center_w=np.array([0.0, -0.15, 0.80]), R=np.eye(3))
    upper = SimpleNamespace(center_w=np.array([upper_xy[0], upper_xy[1], upper_z]), R=_R(tilt) @ rot_z(0.3))
    w = SimpleNamespace(objects={"bowl0": upper, "bowl1": lower}, obj=lambda l: {"bowl0": upper, "bowl1": lower}[l])
    sk = Skills.__new__(Skills)
    sk.nest_verify = mode
    sk.refresh = lambda: w
    sk._record = lambda r: r
    return sk


MISS = SkillResult(False, "place", "place_miss", 30, "left", note="xy err 33 mm")
JOB = Job("bowl0", "left", None, support="bowl1")


def test_a_nested_miss_counts_only_with_support():
    assert _sk("target", (0.012, -0.15))._nested_on_support(JOB, MISS) is MISS
    r = _sk("support", (0.012, -0.15))._nested_on_support(JOB, MISS)
    assert r.ok and r.status == "ok" and "nested in bowl1" in r.note


@pytest.mark.parametrize("xy, z, tilt", [((0.05, -0.15), 0.83, 12.0),    # 50 mm off: on the rim, not in
                                         ((0.01, -0.15), 0.78, 12.0),    # below its support
                                         ((0.01, -0.15), 0.83, 45.0)])   # tipped
def test_real_misses_stay_misses(xy, z, tilt):
    assert _sk("support", xy, z, tilt)._nested_on_support(JOB, MISS) is MISS


def test_only_misses_of_jobs_with_a_support():
    sk = _sk("support", (0.012, -0.15))
    assert sk._nested_on_support(Job("bowl0", "left", None), MISS) is MISS          # no support: as today
    other = SkillResult(False, "place", "collision", 0, "left")
    assert sk._nested_on_support(JOB, other) is other                               # only a place_miss


def test_default_and_validation():
    assert RuntimeConfig().nest_verify == "target"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", nest_verify="loose"), assets=None)
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["nest_verify"][0] == RuntimeConfig().nest_verify
