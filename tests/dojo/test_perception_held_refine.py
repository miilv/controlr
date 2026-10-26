"""held_look verdict rules (jaw opening, absence cap) and refine's support test (pure python + numpy)."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from dojo.perception import j1_adapter as J  # noqa: E402
from dojo.perception.refine import _under  # noqa: E402


def test_opening_wider_than_command_is_held_even_when_the_colour_test_sees_pads():
    # 5.1 stack 100004: a dark block lifted 77 mm, colour test "pads" (not held at 1.0 before)
    held, conf = J.held_from_opening(0.55, 0.298, held_px=False, dark_like=True)
    assert held and conf >= 0.7
    held, conf = J.held_from_opening(0.65, 0.344, held_px=False, dark_like=True)  # 100008
    assert held and conf == 1.0


def test_empty_closed_jaw_is_not_held_and_commands_below_the_floor_clip():
    held, conf = J.held_from_opening(0.15, 0.0, held_px=False, dark_like=True)
    assert not held and conf == pytest.approx(0.675)
    assert J.held_from_opening(0.15, 0.1, False, True) == J.held_from_opening(0.15, 0.0, False, True)


def test_disagreement_caps_the_confidence():
    held, conf = J.held_from_opening(J.HELD_CLOSED_FLOOR + 0.15, 0.0, held_px=False, dark_like=False)
    assert held and conf <= 0.6  # opening says held (gap 0.15), colour says pads, near the threshold
    held, conf = J.held_from_opening(J.HELD_CLOSED_FLOOR, 0.0, held_px=True, dark_like=True)
    assert not held and conf <= 0.4  # jaws at the command, but an object-coloured dark region


def test_colour_fallback_absence_is_capped():
    held, conf, ev = J.held_from_colour(0.05, dark_like=True)
    assert not held and conf <= J.HELD_ABSENCE_CAP and ev == "absent"
    held, conf, ev = J.held_from_colour(0.95, dark_like=False)
    assert held and conf == 1.0 and ev == "colour"


def _box(cx, cy, cz, s=0.03):
    c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * s / 2
    return SimpleNamespace(Vs=c), [cx, cy, cz, 1.0, 0.0, 0.0, 0.0]


def test_under_is_a_support_below_with_overlapping_footprint():
    a, p_low = _box(0, 0, 0.78)
    _, p_top = _box(0.005, 0, 0.81)
    assert _under(a, p_low, a, p_top)          # the block of a stack below the target
    assert not _under(a, p_top, a, p_low)      # the one above is not a support
    _, p_side = _box(0.05, 0, 0.78)
    assert not _under(a, p_side, a, p_top)     # no footprint overlap
    _, p_same = _box(0.02, 0, 0.78)
    assert not _under(a, p_same, a, p_low)     # same level


def test_rests_on_needs_contact_and_overlap():
    from dojo.perception.refine import _rests_on

    a, p_low = _box(0, 0, 0.78)
    _, p_top = _box(0.002, 0, 0.81)
    assert _rests_on(a, p_top, a, p_low)          # the next block of a stack covers the lower one
    _, p_hover = _box(0, 0, 0.86)
    assert not _rests_on(a, p_hover, a, p_low)    # a block hovering in the jaws above does not
    _, p_edge = _box(0.027, 0, 0.81)
    assert not _rests_on(a, p_edge, a, p_low)     # 10 % footprint overlap: not covering
    assert not _rests_on(a, p_low, a, p_top)


def test_support_z_is_the_highest_known_top_under_the_footprint(monkeypatch):
    from dojo.perception import render, refine

    a, p_low = _box(0, 0, 0.78)          # table-resting block: top 0.795
    _, p_mid = _box(0.001, 0, 0.81)      # on it: top 0.825
    _, p_far = _box(0.2, 0, 0.78)
    monkeypatch.setattr(render, "load_asset", lambda key: a)
    _, p_float = _box(0.002, 0, 0.896)   # a carried-height belief after a release (5.1 stack 100007 s142)
    assert refine.support_z(a, p_float, [("b", p_low), ("b", p_mid), ("b", p_far)]) == pytest.approx(0.825)
    assert refine.support_z(a, p_float, [("b", p_far)]) == pytest.approx(refine.TABLE_Z)
    _, p_under = _box(0.002, 0, 0.80)    # an "other" above the target's bottom is not its support
    assert refine.support_z(a, p_under, [("b", p_mid)]) == pytest.approx(refine.TABLE_Z)
