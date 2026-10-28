"""dojo.simlab.twin_crossbox: release segments from a gripper command track, and the flip classification."""

from dojo.simlab.twin_crossbox import classify, grip_events, segments, tilt_deg


def _a(lg, rg=1.0):
    return {"left_arm_joint_state": [0.0] * 6, "right_arm_joint_state": [0.0] * 6, "left_ee_joint_state": [lg],
            "right_ee_joint_state": [rg]}


def test_release_segments_stop_at_the_neighbouring_gripper_events():
    acts = [_a(1.0)] * 10 + [_a(0.0)] * 100 + [_a(1.0)] * 30 + [_a(0.0)] * 20 + [_a(1.0)] * 80
    assert grip_events(acts) == [(10, "left", "close"), (110, "left", "open"), (140, "left", "close"),
                                 (160, "left", "open")]
    s = segments(acts)
    assert [(x["t0"], x["t_rel"], x["t1"]) for x in s] == [(50, 110, 140), (141, 160, 200)]


def test_flips_are_qualitative_and_the_divergence_is_reported():
    up, side = [1, 0, 0, 0], [0.7071, 0.7071, 0, 0]
    assert round(tilt_deg(side)) == 90 and tilt_deg(up) == 0
    base = {"reward": {"success_flag": False, "ended": False, "stages_left": 2, "final_left": 1}, "score": 0.2,
            "final_objects": {"cup": [0.1, 0.2, 0.80, *up]}, "moved_mm": {"cup": 30.0}, "held": []}
    near = {**base, "final_objects": {"cup": [0.101, 0.2, 0.80, *up]}}
    c = classify(base, near)
    assert c["flips"] == [] and abs(c["max_mm"] - 1.0) < 1e-6
    toppled = {**base, "final_objects": {"cup": [0.1, 0.2, 0.78, *side]}, "moved_mm": {"cup": 5.0},
               "reward": {**base["reward"], "stages_left": 1}, "score": 0.4}
    assert set(classify(base, toppled)["flips"]) == {"topple:cup", "moved:cup", "checker:stages_left", "checker:score"}
