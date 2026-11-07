"""The development settler removes total belt travel and preserves vertical rest."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def remove_phase():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/skills/byimg/settle_native_phase.py"
    )
    spec = importlib.util.spec_from_file_location("byimg_settle_native_phase", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.remove_phase


def test_phase_is_relative_to_authored_pose_and_does_not_change_height(remove_phase):
    rows = [
        dict(
            type="Rigid",
            clutter=False,
            pos=[x + 0.306, 0.1, 0.81],
            layout_pos=[x, 0.1, 0.85],
        )
        for x in (-1.0, -0.5, 0.1)
    ]
    rows.append(
        dict(
            type="Dynamic",
            clutter=False,
            pos=[2.0, 0.0, 0.0],
            layout_pos=[2.0, 0.0, 0.0],
        )
    )
    fixed, shift = remove_phase(rows)
    np.testing.assert_allclose(shift, [0.306, 0.0, 0.0])
    np.testing.assert_allclose([o["pos"][0] for o in fixed[:3]], [-1.0, -0.5, 0.1])
    assert all(o["pos"][2] == 0.81 for o in fixed[:3])
    assert fixed[3] == rows[3]
    assert rows[0]["pos"][0] == pytest.approx(-0.694)  # source snapshot stays immutable


def test_no_rider_refuses_a_snapshot(remove_phase):
    with pytest.raises(ValueError, match="no riders"):
        remove_phase([])
