"""j1-adapter-3: the strict server's ``perceive`` route (robot-harness J1 / J4) perceives every instruction the runtime can
run, with the scene backgrounds the local perception service uses.

Before this adapter version, an instruction outside the six J1 patterns answered "unknown task". The adapter also
always used the table backgrounds, so the conveyor tasks (belt in the head view) never fit. CPU only: the candidate sets
and the background dirs are stand-ins."""

import json
from pathlib import Path

import numpy as np
import pytest

from dojo.perception import j1_adapter as J

INSTR = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "robodojo_instructions.json").read_text())["instructions"]


def _first(task):
    v = INSTR[task]
    return v if isinstance(v, str) else v[0]


TASKS = {
    "match_and_pick_from_conveyor": {"labels": {"belt": {"type": "Dynamic", "candidates": ["Dynamic/conveyor/00000"]},
                                                "target0": {"type": "Rigid", "candidates": ["Rigid/apple/00001"]}}},
    "imitate_sorting_sequence": {"labels": {"basket0": {"type": "Geometry", "candidates": ["Geometry/basket/00005"]},
                                            "cube0": {"type": "Rigid", "candidates": ["Rigid/cube/00000"]}}},
    "stack_blocks": {"labels": {"block_0": {"type": "Rigid", "candidates": ["Rigid/block/00005"]}}},
}


def test_the_six_j1_tasks_route_as_before_and_every_runtime_task_now_routes():
    for task in ("press_by_number", "push_T", "stack_blocks", "general_pickup"):
        assert J.task_for(_first(task)) == (task, "j1_patterns")
    for task in ("match_and_pick_from_conveyor", "imitate_sorting_sequence", "plug_in_charger",
                 "pick_from_conveyor_by_image", "insert_tubes", "make_kong"):
        assert J.task_for(_first(task)) == (task, "vocabulary"), task
    assert J.task_for("anything", task="swap_blocks") == ("swap_blocks", "caller")


def _png(path: Path, value: int) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((4, 6, 3), value, np.uint8))


@pytest.fixture
def bg_dirs(tmp_path, monkeypatch):
    pytest.importorskip("cv2")                                              # the perception extra (CI: dev only)
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as IL
    from dojo.perception import segment as SEG
    root = tmp_path / "bg"
    for c in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        _png(root / f"{c}.png", 10)                                         # the table
    _png(root / "scene" / "Dynamic_conveyor_00000" / "cam_head.png", 200)   # the belt (dojo.perception.belt_bg)
    monkeypatch.setattr(SEG, "BG_DIRS", (str(root),))
    monkeypatch.setattr(IL, "task_map", lambda: TASKS)
    monkeypatch.setattr(J, "_BG", {})
    return root


def test_backgrounds_follow_the_scene_fixtures(bg_dirs):
    bgs, used = J.backgrounds_for("match_and_pick_from_conveyor")
    assert used == "Dynamic_conveyor_00000" and int(bgs["cam_head"][0, 0, 0]) == 200
    bgs, used = J.backgrounds_for("stack_blocks")                           # no scene fixture: the table
    assert used is None and int(bgs["cam_head"][0, 0, 0]) == 10
    bgs, used = J.backgrounds_for("imitate_sorting_sequence")              # a Geometry fixture without a scene dir yet
    assert used is None and int(bgs["cam_head"][0, 0, 0]) == 10
    _png(bg_dirs / "scene" / "Geometry_basket_00005" / "cam_head.png", 120)   # generic: any Dynamic / Geometry key
    J._BG.clear()
    bgs, used = J.backgrounds_for("imitate_sorting_sequence")
    assert used == "Geometry_basket_00005" and int(bgs["cam_head"][0, 0, 0]) == 120


def test_scene_keys_are_the_ones_the_local_service_uses(bg_dirs):
    from dojo.perception import inverse_layout as IL
    assert IL.scene_keys("match_and_pick_from_conveyor") == ("Dynamic/conveyor/00000",)
    assert IL.scene_keys("stack_blocks") == () and IL.scene_keys("nope") == () and IL.scene_keys(None) == ()
    from dojo.runtime.perception_service import _Perception
    p = _Perception()
    p.set_scene("match_and_pick_from_conveyor")
    assert p._scene == ("Dynamic/conveyor/00000",)


def test_the_note_passes_the_strict_servers_schema():
    from dojo.perception.agentic.vendor.j1_schema import validate_perceive
    out = {"objects": [{"label": "belt", "pos": [0.0, 0.0, 0.78], "confidence": 0.9}], "regions": [], "needs_look": [],
           "task": "match_and_pick_from_conveyor", "t_ms": 1.0, "version": J.VERSION,
           "notes": "candidate set via vocabulary; background scene Dynamic_conveyor_00000"}
    clean, rep = validate_perceive(out)
    assert clean is not None and not rep["rejected"] and "scene Dynamic_conveyor_00000" in clean["notes"], rep
    assert J.VERSION == "j1-adapter-3"
