"""B1 flags from the perception bake-off (perceiver-models): both default off; the on-base refit leaves table
children and static-container children alone."""
import pytest


def test_new_b1_flags_default_off():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as il

    cfg = il.EstimatorConfig()
    assert cfg.identity_margin_swap is False and cfg.on_base_refit is False


def test_on_base_refit_skips_table_children_and_moving_bases(monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as il

    labels = {"a": {"placement": {"relative_plane": "Table", "xlim": [0.0, 0.1]}, "type": "Rigid"},
              "b": {"placement": {"relative_plane": "belt/belt/0", "xlim": [0.0, 0.8]}, "type": "Rigid"},
              "belt": {"placement": {"relative_plane": "Ground"}, "type": "Dynamic"}}
    monkeypatch.setattr(il, "task_map", lambda: {"t": {"labels": labels}})
    groups = [{"labels": ["a"], "placement": labels["a"]["placement"], "candidates": ["Rigid/x/00000"]},
              {"labels": ["b"], "placement": labels["b"]["placement"], "candidates": ["Rigid/x/00000"]},
              {"labels": ["belt"], "placement": labels["belt"]["placement"], "candidates": ["Dynamic/belt/00000"]}]
    chosen = [il.Hypothesis(0, "Rigid/x/00000", "up", 0.0, 0.0, 0.0, 0.3), il.Hypothesis(1, "Rigid/x/00000", "up", 0.2, 0.0, 0.0, 0.3)]
    pending = [(i, g, {}) for i, g in enumerate(groups[:2])]
    out = il._on_base_refit(chosen, {}, il.EstimatorConfig(on_base_refit=True), groups, pending, "t")
    assert out == chosen  # a table child and a rider of a non-Geometry base are not touched
