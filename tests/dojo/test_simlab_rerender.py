"""rerender_d0 pure helpers."""

from dojo.simlab.rerender_d0 import compare, frame_boundaries


def test_frame_boundaries_follow_the_d0_moves():
    segs = [{"name": "left_look_at_object", "steps": 30}, {"name": "right_look_at_object", "steps": 25},
            {"name": "both_centre", "steps": 40}, {"name": "both_centre_retry", "steps": 5}]
    assert frame_boundaries(segs) == [30, 55, 100]
    assert frame_boundaries([{"name": "both_centre", "steps": 10}]) == [0, 0, 10]      # failed looks: no segment


def test_compare_matches_objects_by_key_not_instance_name():
    def gt(name, x):
        return {"objects": [{"inst_name": name, "type": "Rigid", "category": "block", "category_idx": 5,
                             "label": "block_0", "clutter": False, "pos": [x, 0, 0.78], "quat": [1, 0, 0, 0]}],
                "robot": {"left": {"joint_pos": [0] * 8}}}
    c = compare(gt("block_5_1", 0.1), gt("block_5_17", 0.1005))
    assert abs(c["max_obj_mm"] - 0.5) < 1e-6 and c["task_objects"]["block_0"]["mm"] == 0.5
