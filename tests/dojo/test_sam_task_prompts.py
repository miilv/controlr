"""Offline bake-off guards: no GPU, checkpoint or recorded dev data needed."""

import importlib.util
from pathlib import Path

import numpy as np

p = Path(__file__).resolve().parents[2] / "scripts/perception/sam/task_prompt_eval.py"
spec = importlib.util.spec_from_file_location("sam_task_eval", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_camera_is_not_car_and_catalogue_aliases():
    assert m.phrase_categories("classify_objects", "black camera") == ["action_camera"]
    assert m.phrase_categories("imitate_sorting_sequence", "small black camera") == [
        "action_camera"
    ]
    assert m.phrase_categories("classify_objects", "toy car") == ["toy_car"]
    assert m.phrase_categories("organize_table", "black mouse pad") == ["mousemat"]
    assert m.phrase_categories("pour_by_language", "blue bottle") == ["wine_bottle"]
    assert m.phrase_categories("fill_pen_holder", "pen") == ["pen", "oil_pen"]


def test_union_keeps_base_identity_and_adds_disjoint_mask():
    a = np.zeros((8, 8), np.uint8)
    a[:3, :3] = 1
    b = np.zeros((8, 8), np.uint8)
    b[5:, 5:] = 1
    base = m.Det(a, "Rigid/pen/00000", 0.5)
    result = m.dedup([base, m.Det(a, "pen", 0.99), m.Det(b, "coin", 0.8)])
    assert len(result) == 2 and result[0] is base
    assert result[1].key == "coin" and np.array_equal(result[1].mask, b)


def test_ambiguous_category_does_not_become_exact_asset_identity():
    codes = np.ones((8, 8), np.uint16)
    objs = [{"type": "Rigid", "category": "pen", "category_idx": 0, "label": "target0"}]
    rows, ph = m.score([m.Det(codes.astype(np.uint8), "pen|oil_pen", 0.9)], codes, objs)
    assert rows[0]["iou"] == 1 and not rows[0]["cat_ok"] and not rows[0]["class_ok"]
    assert ph == 0


def test_banks_have_no_unexpanded_placeholders():
    assert len(m.BANKS) == 11
    for b in m.BANKS.values():
        assert b["catalogue_labels"]
        assert all("{" not in x and "}" not in x for x in b["phrases"])


def test_collector_uses_episode_layout_and_rejects_nondev_dirs(tmp_path):
    import io
    import json
    import subprocess
    import sys
    import tarfile

    root = tmp_path / "runs"
    root.mkdir()
    run = root / "c4b_salted_100000"
    run.mkdir()
    task = "deposit_coin"
    layout = tmp_path / "salted" / "layouts" / task / "100000.settled.json"
    layout.parent.mkdir(parents=True)
    layout.write_text('{"salted": true}')
    (run / f"{task}_perception_verify-bundle_100000_report.json.gz").touch()
    (run / f"{task}_perception_verify-bundle_episodes.jsonl").write_text(
        json.dumps({"task": task, "layout_id": 100000, "layout": str(layout)}) + "\n"
    )
    look = run / f"{task}_perception_verify-bundle_100000_look_frames"
    look.mkdir()
    (look / "frame0_state.json").write_text("{}")
    for cam in m.CAMS:
        (look / f"frame0_{cam}.png").write_bytes(b"example recorded image")
    # Disallowed directory must be rejected before any contained path is opened.
    (root / "c4b_sealed_100100").mkdir()
    script = p.with_name("collect_task_frames.py")
    result = subprocess.run(
        [sys.executable, str(script), str(root)], capture_output=True, check=True
    )
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:gz") as tar:
        assert (
            tar.extractfile(f"{task}/100000/settled.json").read() == layout.read_bytes()
        )
        source = json.load(tar.extractfile(f"{task}/100000/source.json"))
        assert source["layout_path"] == str(layout)


def test_belt_phase_uses_logged_native_step_not_51_step_fallback():
    spec = importlib.util.spec_from_file_location(
        "sam_collect", p.with_name("collect_task_frames.py")
    )
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    layout = {
        "Rigid": {
            "bottle": [
                {
                    "label": "target0",
                    "category": "bottle",
                    "default_pos": [0.2, 0.04, 0.84],
                }
            ]
        }
    }
    events = [
        {
            "kind": "track_read",
            "step": 10,
            "found": [["rider0", 0.9, "bottle", [0.444, 0.04]]],
        }
    ]
    phase = c.belt_phase(layout, events, "target0")
    assert abs(phase["native_steps"] - 51) < 1e-9
    assert phase["anchor_step"] == 10
    assert c.belt_phase(layout, [], "target0") is None
