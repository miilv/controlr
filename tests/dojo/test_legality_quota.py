"""Legality triage 2026-10-28 (an independent perception review, after the audit's P1-2): B1 decided identities from the
task config's select-list structure. ``_category_quota`` forced "each category once" on imitate's lists (re-fitting
repeated categories as the missing ones, proposing missing categories on unexplained foreground) and
``_quota_copy_assets`` gave the demonstrator's ``same_as_label`` copy list our list's assets. Both are config identity
relations, not observations: off by default. ``_swap_pairs`` uses the labels' generator REGIONS (a public position
prior) and image fit only: kept. Rule (the audit's): identical observations with a changed hidden config relation ->
identical identities."""

import inspect
import re

import pytest


def test_the_config_identity_relations_are_off_by_default():
    pytest.importorskip("cv2")
    from dojo.perception.inverse_layout import EstimatorConfig
    cfg = EstimatorConfig()
    assert cfg.category_quota is False and cfg.quota_copy_assets is False
    assert cfg.swap_pairs is True                       # a position prior + image fit: not a relation


def test_estimate_reaches_the_quota_only_behind_its_flags():
    """The pipeline calls the relation code only when the flag is on: with the defaults, a changed select relation
    cannot change an identity through it."""
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as IL
    src = inspect.getsource(IL)
    calls = [ln.strip() for ln in src.splitlines() if re.search(r"\b_category_quota\(|\b_quota_copy_assets\(", ln)
             and "def " not in ln]
    assert sorted(calls) == ["chosen = _category_quota(chosen, views, cfg, groups, pending, task)",
                             "out = _quota_copy_assets(out, views, cfg, groups, gi, g, prior, lists)"], calls
    lines = src.splitlines()
    for c in calls:
        i = next(k for k, ln in enumerate(lines) if ln.strip() == c)
        assert lines[i - 1].strip() in ("if cfg.category_quota:", "if cfg.quota_copy_assets:"), lines[i - 1]


def test_swap_pairs_reads_no_select_relation():
    """``_swap_pairs`` trades two objects' labels only where both labels' generator regions admit the other's place and
    both fit better there: no select_mode / same / hierarchical / copy relation is read."""
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as IL
    src = inspect.getsource(IL._swap_pairs)
    for pat in (r"select_mode", r"[\"']same[\"']", r"same_label", r"same_as_label", r"same_index", r"hierarchical",
                r"quota", r"task_map"):
        assert not re.search(pat, src), pat


def test_quota_lists_follow_the_hidden_relation_so_they_stay_off(monkeypatch):
    """What the flags would consume: the lists change with the config's select relation alone (same labels, same
    candidates, same regions). With the defaults off, estimate never uses them."""
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as IL
    cats = [{"name": c, "index": [0]} for c in ("watch", "phone", "toy_car")]
    place = {"xlim": [-0.3, 0.3], "ylim": [-0.2, 0.0]}

    def tm(copy_mode):
        labels = {f"t{i}": {"select_mode": {"mode": "hierarchical", "label": ["t0", "t1", "t2"],
                                            "select_category_nums": [3, 3], "select_instance_nums": [1, 1]},
                            "categories": cats, "placement": place, "type": "Rigid"} for i in range(3)}
        labels.update({f"aim{i}": {"select_mode": {"mode": copy_mode, "same_label": f"t{i}"}, "categories": cats,
                                   "placement": place, "type": "Rigid"} for i in range(3)})
        return {"imitate_like": {"labels": labels}}
    g = {"labels": ["t0", "t1", "t2", "aim0", "aim1", "aim2"]}
    monkeypatch.setattr(IL, "task_map", lambda: tm("same_as_label"))
    with_copy = IL.quota_lists("imitate_like", g)
    monkeypatch.setattr(IL, "task_map", lambda: tm("unique"))
    without = IL.quota_lists("imitate_like", g)
    assert [L["labels"] for L in with_copy] == [["t0", "t1", "t2"], ["aim0", "aim1", "aim2"]]
    assert [L["labels"] for L in without] == [["t0", "t1", "t2"]]
