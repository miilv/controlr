"""Stream H: ``RuntimeConfig.push_goal`` / ``Skills.push_goal`` ("centre" = today | "cover"): ``library.push_to`` stops
once the pushed object's footprint covers (or lies inside) its support's, aiming at the nearest such centre, instead of
the support's centre within 12 mm. organize_table: the 368 mm keyboard over its 240 mm frame. CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import library


def _obj(c, ext, yaw=0.0):
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    ext = np.asarray(ext, float)
    c = np.asarray(c, float)
    half = np.abs(R) @ (ext / 2)
    return SimpleNamespace(R=R, asset=SimpleNamespace(extents=ext), center_w=c,
                           aabb=lambda: (c - half, c + half))


def test_cover_region_for_a_keyboard_over_its_frame():
    kb = _obj([-0.04, -0.16, 0.77], [0.368, 0.152, 0.013])
    frame = _obj([0.1, 0.0, 0.766], [0.24, 0.10, 0.001])
    lo, hi = library.cover_region(kb, frame)
    m = library.COVER_MARGIN
    assert lo[0] == pytest.approx(0.22 - 0.184 + m) and hi[0] == pytest.approx(-0.02 + 0.184 - m)
    assert lo[1] == pytest.approx(0.05 - 0.076 + m) and hi[1] == pytest.approx(-0.05 + 0.076 - m)


def test_cover_region_inside_for_a_small_object_and_collapse():
    small = _obj([0.0, 0.0, 0.8], [0.04, 0.04, 0.02])
    pad = _obj([0.3, 0.1, 0.77], [0.20, 0.10, 0.002])
    lo, hi = library.cover_region(small, pad)
    assert np.all(lo < hi) and lo[0] == pytest.approx(0.2 + 0.02 + library.COVER_MARGIN)
    tight = _obj([0.0, 0.0, 0.8], [0.099, 0.04, 0.02])               # just fits: the range collapses to the centre
    lo2, hi2 = library.cover_region(tight, pad)
    assert lo2[1] < hi2[1] and lo2[0] <= hi2[0]


def test_default_and_validation():
    assert RuntimeConfig().push_goal == "centre"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", push_goal="edge"), assets=None)
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["push_goal"][0] == RuntimeConfig().push_goal
