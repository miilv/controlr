"""EstimatorConfig.slot_settle_merged: the agentic read's merged record re-places slot children from their base, by the
task's labels (the agent may name the coin "coin" for coin0: 5.1 deposit_coin 100005)."""
import pytest


def _rec():
    return {"Rigid": {"coin": [{"label": "coin0", "category_idx": 0, "default_pos": [-0.054, -0.069, 0.765],
                                "default_ori": [1.0, 0.0, 0.0, 0.0], "_perception": {"confidence": 0.39}}]},
            "Geometry": {"vertical_coin_stand": [{"label": "vertical_coin_stand", "category_idx": 0,
                                                  "default_pos": [-0.056, -0.07, 0.769], "default_ori": [1.0, 0, 0, 0],
                                                  "_perception": {"confidence": 0.5}}]}}


def test_settle_record_slots_writes_the_settled_child_back_by_label(monkeypatch):
    from dojo.perception.agentic import record, seed_refine
    seen = {}

    def fake(objs, task, radius_m=0.04):
        seen["labels"] = sorted(o["label"] for o in objs)
        for o in objs:
            if o["label"] == "coin0":
                o["pos"], o["quat"], o["slot_from_base"] = [-0.056, -0.071, 0.781], [0.07, -0.04, -0.71, -0.70], "vertical_coin_stand"
        return 1

    monkeypatch.setattr(seed_refine, "settle_slot_children", fake)
    rec = _rec()
    assert record.settle_record_slots(rec, "deposit_coin") == 1
    c = rec["Rigid"]["coin"][0]
    assert seen["labels"] == ["coin0", "vertical_coin_stand"]
    assert c["default_pos"] == [-0.056, -0.071, 0.781] and c["default_ori"] == [0.07, -0.04, -0.71, -0.70]
    assert c["_perception"]["slot_from_base"] == "vertical_coin_stand"
    assert rec["Geometry"]["vertical_coin_stand"][0]["default_pos"] == [-0.056, -0.07, 0.769]   # the base untouched


def test_slot_settle_merged_default_off():
    pytest.importorskip("cv2")
    from dojo.perception.inverse_layout import EstimatorConfig
    assert EstimatorConfig().slot_settle_merged is False
