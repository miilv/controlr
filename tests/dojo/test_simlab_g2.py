"""G2 trajectory comparison (pure)."""

import pytest

from dojo.simlab.g2 import PERTURBATIONS, compare


def run(offsets, score=0.15):
    frames = [{"step": k, "objects": {"block_1": [0.1 + dx, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0],
                                      "block_0": [0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0]}}
              for k, dx in enumerate(offsets)]
    return {"frames": frames, "reward": {"success": False}, "score": score}


def test_identical_runs_are_bitwise_equal():
    c = compare(run([0, 0, 0]), run([0, 0, 0]))
    assert c["bitwise_equal"] and c["max_mm"] == 0 and c["first_diff_step"] is None


def test_first_difference_max_and_final():
    c = compare(run([0, 0, 0.001, 0.002]), run([0, 0, 0, 0]))
    assert not c["bitwise_equal"]
    assert c["first_diff_step"] == 2
    assert c["max_mm"] == pytest.approx(2.0)
    assert c["final"]["block_1"]["mm"] == pytest.approx(2.0)
    assert c["final"]["block_0"]["mm"] == 0
    assert c["per_step_max_mm"] == pytest.approx([0, 0, 1.0, 2.0])


def test_unequal_lengths_compare_the_common_prefix():
    c = compare(run([0, 0, 0, 0, 0]), run([0, 0, 0]))
    assert c["steps_compared"] == 3 and c["len_a"] == 5


def test_perturbation_grid():
    assert ("dx", 0.5) in PERTURBATIONS and ("dy", 5.0) in PERTURBATIONS and ("yaw", 2.0) in PERTURBATIONS
    assert len(PERTURBATIONS) == 10
