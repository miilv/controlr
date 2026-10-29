"""Stream H: ``RuntimeConfig.push_goal`` / ``Skills.push_goal`` ("centre" = today | "cover" | "cover_obb"): ``library.push_to`` stops
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


def _covers(kb_c, yaw, h, frame, m=0.0):
    """Every corner of ``frame``'s AABB inside the keyboard's footprint ``m`` in (the checker's footprint test)."""
    mn, mx = frame.aabb()
    A = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    for x in (mn[0], mx[0]):
        for y in (mn[1], mx[1]):
            q = A.T @ (np.array([x, y]) - np.asarray(kb_c)[:2])
            if np.any(np.abs(q) > np.asarray(h) - m + 1e-9):
                return False
    return True


def test_cover_obb_matches_cover_at_zero_yaw():
    frame = _obj([0.1, 0.0, 0.766], [0.24, 0.10, 0.001])
    for c in ([-0.04, -0.16, 0.77], [0.08, 0.01, 0.77], [0.3, 0.05, 0.77]):
        kb = _obj(c, [0.368, 0.152, 0.013])
        lo, hi = library.cover_region(kb, frame)
        inside, tgt = library.cover_goal_obb(kb, frame)
        assert inside == bool(np.all(kb.center_w[:2] >= lo) and np.all(kb.center_w[:2] <= hi))
        assert tgt == pytest.approx(np.clip(kb.center_w[:2], lo, hi))


def test_cover_obb_on_the_rotated_footprint():
    frame = _obj([0.1, 0.0, 0.766], [0.24, 0.10, 0.001])
    h = (0.184, 0.076)
    rng = np.random.default_rng(0)
    n_in = 0
    for _ in range(400):
        yaw = float(rng.uniform(-0.7, 0.7))
        c = [float(rng.uniform(-0.1, 0.3)), float(rng.uniform(-0.08, 0.08)), 0.77]
        kb = _obj(c, [0.368, 0.152, 0.013], yaw)
        inside, tgt = library.cover_goal_obb(kb, frame)
        if inside:                                   # in range = the footprint covers the frame, margin in
            n_in += 1
            assert _covers(c, yaw, h, frame, library.COVER_MARGIN - 1e-6)
        if abs(yaw) < np.radians(6):                 # feasible at this yaw: the target covers
            assert _covers(tgt, yaw, h, frame)
    assert n_in > 0
    # 20 deg: across, the frame's corners span more than the keyboard's width; that axis aims at the frame's centre
    kb = _obj([0.06, 0.03, 0.77], [0.368, 0.152, 0.013], np.radians(20))
    inside, tgt = library.cover_goal_obb(kb, frame)
    lo, hi = library.cover_region(kb, frame)
    assert not inside and bool(np.all(kb.center_w[:2] >= lo) and np.all(kb.center_w[:2] <= hi))   # AABB says "covers"
    across = np.array([-np.sin(np.radians(20)), np.cos(np.radians(20))])
    assert float(across @ (tgt - np.array([0.1, 0.0]))) == pytest.approx(0.0, abs=1e-9)


def test_cover_obb_falls_back_on_a_tipped_object():
    frame = _obj([0.1, 0.0, 0.766], [0.24, 0.10, 0.001])
    kb = _obj([0.08, 0.0, 0.77], [0.368, 0.152, 0.013])
    t = np.radians(60)
    kb.R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]) @ kb.R
    lo, hi = library.cover_region(kb, frame)
    inside, tgt = library.cover_goal_obb(kb, frame)
    assert tgt == pytest.approx(np.clip(kb.center_w[:2], lo, hi))


def test_default_and_validation():
    assert RuntimeConfig().push_goal == "centre"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", push_goal="edge"), assets=None)
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["push_goal"][0] == RuntimeConfig().push_goal
    assert set(SKILL_ATTRS["push_goal"]) == {"centre", "cover", "cover_obb"}
    RuntimePolicy(RuntimeConfig(mode="perception", push_goal="cover_obb"), assets=None)
