"""classify_objects(_by_language) frame-0 priors (default off): the instruction's named categories
(instruction_categories) and no free object inside a pinned walled container (container_free_drop)."""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from types import SimpleNamespace  # noqa: E402

from dojo.perception import inverse_layout as il  # noqa: E402


def _group(cats, k=3):
    return {"labels": ["rigid0"], "candidates": [f"Rigid/{c}/{i:05d}" for c in cats for i in range(2)],
            "categories": [{"name": c} for c in cats],
            "select_mode": {"mode": "hierarchical", "select_category_nums": [k, k], "label_prefix": ["cat0", "cat1", "cat2"]}}


def test_the_instruction_names_three_of_five_categories():
    g = _group(["watch", "car", "wooden_toy", "pepper", "chocolate_bar"])
    instr = ("Put watch objects into the left basket, chocolate_bar objects into the middle basket, and pepper objects "
             "into the right basket, then reset the robot arm.")
    out, info = il.restrict_categories_by_instruction([g], instr)
    assert info == {"rigid0": ["chocolate_bar", "pepper", "watch"]}
    assert {k.split("/")[1] for k in out[0]["candidates"]} == {"watch", "chocolate_bar", "pepper"}
    assert len(g["candidates"]) == 10                                                # the input is not modified
    out, info = il.restrict_categories_by_instruction([g], "Sort the objects by category into the three baskets.")
    assert info == {} and out[0]["candidates"] == g["candidates"]                    # names none: unchanged
    out, info = il.restrict_categories_by_instruction([g], "Put car objects left and watch objects right.")
    assert info == {}                                                                # 2 of 3 named: unchanged
    g2 = _group(["toy_car", "action_camera", "pen", "watch", "garage"])
    out, info = il.restrict_categories_by_instruction([g2], "Put toy car objects left, pen objects middle, watch right.")
    assert info["rigid0"] == ["pen", "toy_car", "watch"]                             # "toy car" names toy_car
    assert il.EstimatorConfig().instruction_categories is False


def test_a_free_object_inside_a_pinned_basket_is_dropped():
    sil_b = np.zeros((40, 40), np.uint8)
    sil_b[0:20, 0:30] = 1

    def hyp(x, y, sil, V=None):
        sm = SimpleNamespace(sil=lambda cam, T: sil, asset=SimpleNamespace(V=V))
        T = np.eye(4)
        T[:3, 3] = [x, y, 0.75]
        return SimpleNamespace(x=x, y=y, sm=lambda: sm, T=lambda: T)

    V = np.array([[x, y, z] for x in (-0.126, 0.126) for y in (-0.084, 0.084) for z in (0.0, 0.077)])
    basket = hyp(0.0, 0.05, sil_b, V)
    on_rim = np.zeros_like(sil_b)
    on_rim[12:18, 10:16] = 1
    rim = hyp(0.01, -0.02, on_rim)                     # inside the footprint (y >= -0.034), pixels on the basket
    front = np.zeros_like(sil_b)
    front[16:26, 10:16] = 1
    in_front = hyp(0.0, -0.06, front)                  # in front of it: outside the footprint
    half = hyp(0.0, -0.02, front)                      # inside the footprint but mostly off the basket's pixels
    views = {"cam_head": SimpleNamespace(cam=None)}
    kept, n = il._off_containers([rim, in_front, half], [basket], views)
    assert kept == [in_front, half] and n == 1
    assert il._off_containers([rim], [], views) == ([rim], 0)
    assert "basket" in il.WALLED_CONTAINERS and "plate" not in il.WALLED_CONTAINERS
    assert il.EstimatorConfig().container_free_drop is False


def test_an_unknown_class_mask_can_be_an_item_of_a_clutter_free_series(monkeypatch):
    """seg_add_agnostic: v2b reads classify_by_language's watches as tape_measure / book / card; in a clutter-free task
    a Rigid mask of a class no task label lists (or another asset of the series' category) inside the series' region
    goes to the series; a basket (Geometry), a class another label lists, or a group not marked stays out."""
    mask = np.zeros((20, 20), np.uint8)
    mask[5:9, 5:9] = 1
    head = SimpleNamespace(cam=SimpleNamespace(ray_plane=lambda uv, z: np.array([[0.0, -0.1, z]])))
    monkeypatch.setattr(il, "region_ok", lambda g, x, y, extra=0.0: True)
    g = {"candidates": ["Rigid/watch/00004", "Rigid/pepper/00000"],
         "agnostic": {"max": 9, "known": {"Rigid/watch/00004", "Rigid/pepper/00000", "Geometry/basket/00002",
                                         "Rigid/garage/00014"}}}
    pending = [(0, g, None), (1, {"candidates": ["Rigid/pepper/00000"]}, None)]
    det = lambda key: SimpleNamespace(key=key, mask=mask)  # noqa: E731
    assert il._agnostic_groups(det("Rigid/tape_measure/00000"), pending, head) == [0]
    assert il._agnostic_groups(det("Rigid/watch/00000"), pending, head) == [0]          # another watch asset
    assert il._agnostic_groups(det("Geometry/basket/00002"), pending, head) == []
    assert il._agnostic_groups(det("Rigid/garage/00014"), pending, head) == []           # another label lists it
    monkeypatch.setattr(il, "region_ok", lambda g, x, y, extra=0.0: False)
    assert il._agnostic_groups(det("Rigid/tape_measure/00000"), pending, head) == []     # outside the region
    assert il.EstimatorConfig().seg_add_agnostic is False
