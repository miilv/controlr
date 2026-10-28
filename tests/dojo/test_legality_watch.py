"""Single-fixed-system audit (b3): the runtime's watch read is generic (``task_state.actor_change``: standing movable
objects at public config slots, any vocabulary), not a make_kong branch; make_kong's behaviour is unchanged (the
knocked push tile, the same score and confidence as the old ``kong_discard``)."""

import inspect

import numpy as np
import pytest

HAND = {f"mahjong{i}_{j}": (-0.325 + 0.046 * (3 * i + j), -0.15, "straight_front") for i in range(5) for j in range(3)
        if not (i == 4 and j == 2)}
PUSH = {"mahjong5_0": (0.0, 0.05, "straight_back"), "mahjong6_0": (0.046, 0.05, "straight_back"),
        "mahjong7_0": (-0.046, 0.05, "straight_back"), "mahjong8_0": (-0.092, 0.05, "straight_back")}
LYING = {"other0": (-0.4, -0.05, "down"), "other1": (-0.4, -0.002, "down")}


def _kong(monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as IL
    slots = {**HAND, **PUSH, **LYING}
    monkeypatch.setattr(IL, "fixed_slot_labels", lambda task: slots if task == "kong_like" else
                        {"piece0": (-0.15, 0.2, None), "piece1": (0.0, 0.2, None)})
    monkeypatch.setattr(IL, "task_map", lambda: {"kong_like": {"labels": {k: {"type": "Rigid"} for k in slots}},
                                                 "ttt_like": {"labels": {"piece0": {"type": "Rigid"},
                                                                         "piece1": {"type": "Rigid"}}}})
    from dojo.perception.cameras import TABLE_Z, head_camera
    cam = head_camera()
    before = np.full((cam.h, cam.w, 3), 120, np.uint8)
    after = before.copy()
    uv, _ = cam.project(np.array([[0.046, 0.05, TABLE_Z + 0.03]]))     # mahjong6_0 falls
    u, v = np.round(uv[0]).astype(int)
    after[v - 6:v + 6, u - 4:u + 4] = (240, 30, 30)
    return before, after


def test_the_generic_read_finds_the_knocked_push_tile_as_kong_discard_did(monkeypatch):
    before, after = _kong(monkeypatch)
    from dojo.perception import task_state
    old = task_state.kong_discard(before, after, task="kong_like")
    new = task_state.actor_change(before, after, "kong_like")
    assert new["supported"] and new["knocked"] == old["knocked"] == "mahjong6_0"
    assert new["confident"] == old["confident"] is True
    assert {k: new["scores"][k] for k in old["scores"]} == old["scores"]          # the push tiles score as before
    assert set(new["scores"]) == set(HAND) | set(PUSH)                             # every standing slot, no lying one
    still = task_state.actor_change(before, before.copy(), "kong_like")
    assert still["supported"] and not still["confident"]                           # nothing changed: nothing reported


def test_a_vocabulary_without_standing_slot_objects_has_nothing_to_watch(monkeypatch):
    before, after = _kong(monkeypatch)
    from dojo.perception import task_state
    assert task_state.actor_change(before, after, "ttt_like")["supported"] is False


def test_the_watch_op_names_no_task():
    from dojo.runtime import perception_service as PS
    src = inspect.getsource(PS._Perception.watch)
    assert "make_kong" not in src and "vocabulary !=" not in src and "actor_change" in src
