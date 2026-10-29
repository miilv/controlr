"""B1's slot fallback (a config-pinned label whose local fit jumped > 2 cm is put at the config slot pose, unfitted:
bit-identical in every scene, confidence 0.5) is marked ``slot_prior``; the J1 adapter labels it source "prior" only
with J1_SLOT_PRIOR_SOURCE=1 (the label is read by the adapter's carry logic and the agent's features)."""
import inspect

import pytest


def test_j1_source_default_unchanged_and_prior_on_opt_in(monkeypatch):
    pytest.importorskip("cv2")
    from dojo.perception import j1_adapter as J
    monkeypatch.delenv("J1_SLOT_PRIOR_SOURCE", raising=False)
    assert J.j1_source({"n_views": 2}) == "fused" and J.j1_source({"n_views": 1}) == "head"
    assert J.j1_source({"n_views": 0, "slot_prior": True}) == "head"            # default: as before
    monkeypatch.setenv("J1_SLOT_PRIOR_SOURCE", "1")
    assert J.j1_source({"n_views": 0, "slot_prior": True}) == "prior"
    assert J.j1_source({"n_views": 3}) == "fused"                               # fitted entries keep their source


def test_the_slot_fallback_is_marked_and_reaches_the_record():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as IL
    src = inspect.getsource(IL.estimate)
    i = src.index("slot_prior = bool(np.hypot(xr - x, yr - y) > 0.02)")
    assert src.index('h.per_view["slot_prior"] = True', i) > i
    assert '{"slot_prior": True} if h.per_view.get("slot_prior")' in src
