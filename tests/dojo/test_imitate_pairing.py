"""imitate_sorting_sequence: the demonstrator's objects paired one-to-one with ours by appearance (B1 ``twin_pairing``,
default off; journal 2026-10-28-dojo-imitate-pairing). Legal under D-C17 (lead, 28 Oct): the correspondence is the
instruction's, the candidates are our OBSERVED assets, the pixels decide; no config list / quota / copy relation is
read. Rule: identical observations with a changed hidden config relation -> the identical pairing."""

import inspect
import re

import numpy as np
import pytest


def _IL():
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as IL
    return IL


def test_twin_pairing_is_off_by_default_and_reached_only_behind_its_flag():
    IL = _IL()
    assert IL.EstimatorConfig().twin_pairing is False
    src = inspect.getsource(IL)
    calls = [ln.strip() for ln in src.splitlines() if re.search(r"\b_twin_pairing\(", ln) and "def " not in ln]
    assert calls == ["chosen = _twin_pairing(chosen, views, cfg, groups, pending)"], calls
    lines = src.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip() == calls[0])
    assert lines[i - 1].strip() == "if cfg.twin_pairing and task == TWIN_TASK:"


def test_twin_pairing_reads_no_config_relation():
    """The pairing's code reads no select relation / list / quota / copy structure of the task config: what it sees are
    the chosen hypotheses (their observed assets and positions), the images (views) and each label group's region."""
    IL = _IL()
    for fn in (IL._twin_pairing, IL.twin_sides, IL.twin_assign):
        src = inspect.getsource(fn)
        for pat in (r"select_mode", r"[\"']same[\"']", r"same_label", r"same_as_label", r"same_index", r"hierarchical",
                    r"quota", r"task_map", r"copy_of", r"\blists\b", r"\blabels\b"):
            assert not re.search(pat, src), (fn.__name__, pat)


def test_same_observations_under_a_changed_hidden_relation_give_the_same_pairing(monkeypatch):
    """The pairing's inputs are positions and fit costs; flipping the config's copy relation (same_as_label <-> unique)
    changes what ``quota_lists`` would hand out, and nothing the pairing computes."""
    IL = _IL()
    xy = [(0.30, -0.07), (-0.25, -0.10), (0.05, -0.12), (0.12, 0.20), (-0.20, 0.25), (0.35, 0.22)]
    C = np.array([[0.20, 0.31, 0.29], [0.40, 0.18, 0.33], [0.27, 0.26, 0.15]])
    ref = (IL.twin_sides(xy), IL.twin_assign(C, 0.02))
    for mode in ("same_as_label", "unique"):
        monkeypatch.setattr(IL, "quota_lists", lambda *a, _m=mode, **k: [{"copy_of": 0 if _m == "same_as_label" else None}])
        assert (IL.twin_sides(xy), IL.twin_assign(C, 0.02)) == ref


def test_sides_split_by_distance_from_our_arms():
    IL = _IL()
    xy = [(0.30, -0.07), (0.12, 0.20), (-0.25, -0.10), (-0.20, 0.25)]
    ours, demo = IL.twin_sides(xy)
    assert sorted(ours) == [0, 2] and sorted(demo) == [1, 3]
    ours, demo = IL.twin_sides(xy + [(0.0, 0.05)])                  # odd: the middle one is in neither
    assert len(ours) == len(demo) == 2 and 4 not in ours + demo


def test_twin_assign_is_one_to_one_and_marks_near_ties_uncertain():
    IL = _IL()
    C = np.array([[0.20, 0.31, 0.29],
                  [0.40, 0.18, 0.33],
                  [0.27, 0.26, 0.255]])                            # row 2: 0.255 vs 0.26 is a near-tie
    a = IL.twin_assign(C, 0.02)
    assert sorted(c for c, _m, _u in a.values()) == [0, 1, 2]     # each of our assets once
    assert a[0][0] == 0 and not a[0][2] and a[1][0] == 1 and not a[1][2]
    assert a[2][0] == 2 and a[2][2] and a[2][1] == pytest.approx(0.005)
    C2 = np.array([[0.2, 9.0], [9.0, 9.0]])                        # nothing fitted for row 1: left out
    assert set(IL.twin_assign(C2, 0.02)) == {0}
