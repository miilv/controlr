"""inverse_layout twin_union_pool "weak" (default off): the routed set's estimate, the union estimate only when a label
the twin would extend is missing or weak, kept when it finds as many of those labels at no lower mean confidence
(make_toast_random's toaster 00004 under the routed make_toast set)."""
import pytest


def _rec(conf):
    return {"Articulation": {"toaster": [{"label": "toaster", "category_idx": 4, "_perception": {"confidence": conf}}]}
            if conf is not None else {},
            "Rigid": {"bread": [{"label": "bread_0", "category_idx": 0, "_perception": {"confidence": 0.47}}]},
            "_perception": {}}


def _setup(monkeypatch, routed, union):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as IL
    g0 = [{"labels": ["toaster"], "candidates": ["Articulation/toaster/00000", "Articulation/toaster/00001"]},
          {"labels": ["bread_0"], "candidates": ["Rigid/bread/00000"]}]
    g1 = [{"labels": ["toaster"], "candidates": g0[0]["candidates"] + ["Articulation/toaster/00004"]}, g0[1]]
    monkeypatch.setattr(IL, "label_groups", lambda task: g0)
    monkeypatch.setattr(IL, "union_twin_pool", lambda task, groups: g1)
    calls = []

    def fake(images, state, task, instruction="", bgs=None, cfg=None, **kw):
        calls.append(cfg.twin_union_pool)
        return _rec(union if cfg.twin_union_pool is True else routed)
    monkeypatch.setattr(IL, "estimate", fake)
    return IL, calls


def test_weak_routed_fit_takes_the_union_estimate_when_it_is_no_worse(monkeypatch):
    IL, calls = _setup(monkeypatch, routed=0.316, union=0.315)
    rec = IL._estimate_twin_weak({}, {}, "make_toast", "", None, IL.EstimatorConfig(twin_union_pool="weak"), None, None, None)
    assert calls == [False, True]
    tu = rec["_perception"]["twin_union"]
    assert tu["extended"] == ["toaster"] and tu["weak"] == ["toaster"] and tu["kept"] == "union"


def test_confident_routed_fit_never_runs_the_union(monkeypatch):
    IL, calls = _setup(monkeypatch, routed=0.9, union=0.95)
    rec = IL._estimate_twin_weak({}, {}, "make_toast", "", None, IL.EstimatorConfig(twin_union_pool="weak"), None, None, None)
    assert calls == [False] and rec["_perception"]["twin_union"]["kept"] == "routed"


def test_union_that_loses_the_label_or_confidence_is_dropped(monkeypatch):
    for routed, union in ((0.316, None), (0.198, 0.0)):
        IL, calls = _setup(monkeypatch, routed=routed, union=union)
        rec = IL._estimate_twin_weak({}, {}, "make_toast", "", None, IL.EstimatorConfig(twin_union_pool="weak"), None, None,
                                     None)
        assert calls == [False, True] and rec["_perception"]["twin_union"]["kept"] == "routed"
    IL, calls = _setup(monkeypatch, routed=None, union=0.331)      # routed missed it: a found one is kept
    rec = IL._estimate_twin_weak({}, {}, "make_toast", "", None, IL.EstimatorConfig(twin_union_pool="weak"), None, None, None)
    assert rec["_perception"]["twin_union"]["kept"] == "union"
