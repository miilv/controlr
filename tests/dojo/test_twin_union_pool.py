"""twin_union_pool (default off): a std / _random pair shares one instruction, so every label's candidates come from both
public configs."""
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from dojo.perception import inverse_layout as il  # noqa: E402


def test_the_pool_is_the_union_of_both_variants(monkeypatch):
    tm = {"pour": {"labels": {"cup": {"candidates": ["Rigid/mug/00015"], "categories": [{"name": "mug", "index": [15]}]}}},
          "pour_random": {"labels": {"cup": {"candidates": ["Rigid/mug/00017"], "categories": [{"name": "mug", "index": [17]}]}}},
          "solo": {"labels": {"x": {"candidates": ["Rigid/a/00000"], "categories": []}}}}
    monkeypatch.setattr(il, "task_map", lambda: tm)
    g = [{"labels": ["cup"], "candidates": ["Rigid/mug/00015"], "categories": [{"name": "mug", "index": [15]}]}]
    out = il.union_twin_pool("pour", g)
    assert out[0]["candidates"] == ["Rigid/mug/00015", "Rigid/mug/00017"] and len(out[0]["categories"]) == 2
    assert g[0]["candidates"] == ["Rigid/mug/00015"]                                  # the input is not modified
    back = il.union_twin_pool("pour_random", [{"labels": ["cup"], "candidates": ["Rigid/mug/00017"], "categories": []}])
    assert back[0]["candidates"] == ["Rigid/mug/00017", "Rigid/mug/00015"]           # symmetric
    assert il.union_twin_pool("solo", g) is g and il.twin_task("solo") is None
    assert il.EstimatorConfig().twin_union_pool is False

