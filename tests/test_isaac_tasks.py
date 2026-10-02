"""CPU tests for the Isaac task registry (no Isaac needed): seeded sampling
within PHANTOM's documented ranges, success rules on synthetic privileged
snapshots, and reachability of the scripted reference plan / reach markers
with controlr's own UR3 kinematics."""

from __future__ import annotations

import numpy as np
import pytest

from controlr.robot.isaac import tasks
from controlr.robot.kinematics import UR3Kinematics, matrix_to_rotvec

# Scene facts as the server reports them for PHANTOM's
# configs/sim/waffles_w2l_adaptive_parallel_dt1ms_20260909.json (measured rig).
CAM_T_WC = np.array([
    [0.9999720414746357, -0.005669109656577374, 0.00487621418213748, -0.35399058583569004],
    [-0.007215535335001398, -0.9026929426481264, 0.4302248102365732, -0.44643379477503714],
    [0.0019627325028448834, -0.4302479662810364, -0.9027086103456388, 0.9535],
    [0.0, 0.0, 0.0, 1.0]])
SCENE = {
    "object": {"center": [-0.3987114560752966, -0.2733251174313464, 0.0355], "yaw": 0.2595497954083121,
               "size": [0.17, 0.035, 0.09]},
    "table_top_z": -0.0125,
    "mat": {"center": [-0.354, -0.239, -0.011], "size": [0.4, 0.38, 0.003], "top_z": -0.0095},
    "bin": {"lower": [-0.5705, -0.0651, -0.0055], "upper": [-0.2105, 0.1949, 0.1805],
            "center": [-0.3905, 0.0649, -0.0095], "yaw": -0.012089355098796523},
    "camera": {"K": [[609.28173828125, 0, 337.8138732910156], [0, 608.1331176757812, 249.65126037597656], [0, 0, 1]],
               "T_cam_base": np.linalg.inv(CAM_T_WC).tolist()},
}
KIN = UR3Kinematics()


def snapshot(pos, yaw=0.0, pads=(0.0, 0.0), robot_obj=0.0, vel=(0, 0, 0), tcp=(-0.3, -0.2, 0.3)):
    return {"object_pos": np.asarray(pos, float), "object_quat_wxyz": tasks.yaw_quat_wxyz(yaw),
            "object_lin_vel": np.asarray(vel, float), "object_ang_vel": np.zeros(3),
            "object_size": np.asarray(SCENE["object"]["size"]), "pad_object_force_n": np.asarray(pads, float),
            "robot_object_force_n": robot_obj, "tcp_pos": np.asarray(tcp, float), "bin": SCENE["bin"]}


def test_registry_and_aliases():
    assert set(tasks.TASKS) == {"waffle_pick_place", "reach", "push"}
    assert tasks.get_task("pick_place").name == "waffle_pick_place"
    for spec in tasks.TASKS.values():
        assert spec.instruction and spec.scene == tasks.WAFFLE_SCENE
    with pytest.raises(KeyError):
        tasks.get_task("fly")


@pytest.mark.parametrize("name", sorted(tasks.TASKS))
def test_sampling_is_seeded_and_within_phantom_ranges(name):
    spec = tasks.get_task(name)
    a = spec.sample(np.random.default_rng(7), {}, SCENE)
    b = spec.sample(np.random.default_rng(7), {}, SCENE)
    for k in ("object_pos", "object_yaw"):
        np.testing.assert_allclose(a[k], b[k])
    c0 = np.asarray(SCENE["object"]["center"])
    for seed in range(200):
        s = spec.sample(np.random.default_rng(seed), {}, SCENE)
        assert np.all(np.abs(s["object_pos"][:2] - c0[:2]) <= 0.020 + 1e-12)      # +-20 mm
        assert abs(s["object_yaw"] - SCENE["object"]["yaw"]) <= np.radians(10) + 1e-12
        assert s["object_pos"][2] == c0[2]
    nominal = spec.sample(np.random.default_rng(1), {"nominal": True}, SCENE)
    np.testing.assert_allclose(nominal["object_pos"], c0)


def test_reach_markers_visible_clear_and_reachable_from_start():
    q0 = np.asarray(tasks.START_Q)
    rv0 = KIN.fk(q0)[1]
    box = tasks.REACH_DEFAULTS["occluded_px"]
    for seed in range(40):
        s = tasks.sample_reach(np.random.default_rng(seed), {}, SCENE)
        m = s["marker"]
        assert np.linalg.norm(m[:2] - s["object_pos"][:2]) >= 0.08
        u, v = tasks.project(SCENE, m)
        assert 0 <= u < 640 and 0 <= v < 480, "marker must be inside the D435 image"
        assert not (box[0] <= u <= box[2] and box[1] <= v <= box[3]), "marker hidden behind the gripper"
        # reachable while keeping the start orientation (action.rotation=none)
        assert KIN.ik(m, rv0, q0) is not None, f"seed {seed}: marker {m} unreachable"


def test_push_target_on_mat_along_long_axis():
    for seed in range(50):
        s = tasks.sample_push(np.random.default_rng(seed), {}, SCENE)
        d = s["zone"][:2] - s["object_pos"][:2]
        assert 0.06 - 1e-9 <= np.linalg.norm(d) <= 0.10 + 1e-9
        axis = np.array([np.cos(s["object_yaw"]), np.sin(s["object_yaw"])])
        assert abs(abs(np.dot(d / np.linalg.norm(d), axis)) - 1.0) < 1e-9
        mat_c, half = np.asarray(SCENE["mat"]["center"][:2]), np.asarray(SCENE["mat"]["size"][:2]) / 2
        assert np.all(np.abs(s["zone"][:2] - mat_c) <= half)


def test_waffle_success_rules():
    ep = {"object_pos": np.asarray(SCENE["object"]["center"]), "params": {}}
    inside = [-0.39, 0.06, -0.0055 + 0.045]                     # standing on the bin floor
    assert tasks.evaluate_waffle(snapshot(inside), ep)["success"]
    r = tasks.evaluate_waffle(snapshot(inside, pads=(2.0, 3.0)), ep)
    assert not r["success"] and r["metrics"]["gripped"] and "held" in r["message"]
    assert not tasks.evaluate_waffle(snapshot(inside, robot_obj=1.0), ep)["success"]
    assert not tasks.evaluate_waffle(snapshot(inside, vel=(0.1, 0, 0)), ep)["success"]
    # straddling the wall: one end outside the opening
    assert not tasks.evaluate_waffle(snapshot([-0.39, -0.06, 0.04]), ep)["metrics"]["inside_bin"]
    # still on the mat, held and lifted -> partial progress
    lifted = tasks.evaluate_waffle(snapshot([-0.40, -0.27, 0.10], pads=(2, 2)), ep)
    assert not lifted["success"] and lifted["progress"] == 0.5
    assert tasks.evaluate_waffle(snapshot(SCENE["object"]["center"]), ep)["progress"] == 0.0


def test_reach_and_push_rules():
    ep = {"marker": np.array([-0.4, -0.2, 0.1]), "tcp_pos0": np.array([-0.33, -0.3, 0.35]),
          "object_pos": np.asarray(SCENE["object"]["center"]), "zone": np.array([-0.47, -0.29, -0.0125]),
          "params": {}}
    assert tasks.evaluate_reach(snapshot(SCENE["object"]["center"], tcp=(-0.41, -0.2, 0.1)), ep)["success"]
    far = tasks.evaluate_reach(snapshot(SCENE["object"]["center"], tcp=(-0.33, -0.3, 0.35)), ep)
    assert not far["success"] and far["progress"] == 0.0
    assert tasks.evaluate_push(snapshot([-0.465, -0.285, 0.0355]), ep)["success"]
    carried = tasks.evaluate_push(snapshot([-0.465, -0.285, 0.0355], pads=(1, 1)), ep)
    assert not carried["success"] and "pushed" in carried["message"]


@pytest.mark.parametrize("seed", range(10))
def test_scripted_plan_is_reachable_on_one_ik_branch(seed):
    s = tasks.sample_waffle(np.random.default_rng(seed), {}, SCENE)
    plan = tasks.scripted_pick_place_plan(s["object_pos"], s["object_yaw"], SCENE["bin"]["center"])
    q = np.asarray(tasks.START_Q)
    for ph in plan:
        p0 = KIN.fk(q)[0]
        n = max(1, int(np.ceil(np.linalg.norm(ph["pos"] - p0) / 0.02))) if ph["linear"] else 1
        for k in range(1, n + 1):
            tp = p0 + (ph["pos"] - p0) * k / n if ph["linear"] else ph["pos"]
            qn = KIN.ik(tp, matrix_to_rotvec(ph["R"]), q)
            assert qn is not None, f"{ph['name']}: IK failed at {tp}"
            if ph["linear"]:
                assert np.max(np.abs(qn - q)) < 0.5, f"{ph['name']}: IK branch jump"
            q = qn
    # the fingers (tool x) close across the packet's thin axis
    R = plan[0]["R"]
    thin = np.array([-np.sin(s["object_yaw"]), np.cos(s["object_yaw"])])
    x_h = R[:2, 0] / np.linalg.norm(R[:2, 0])
    assert abs(np.dot(x_h, thin)) > 0.99


# ---------------------------------------------------------------------------
# review fixes (docs/FIXLOG.md)
# ---------------------------------------------------------------------------

def test_old_seed0_reach_marker_is_rejected_as_infeasible():
    """Review control-safety #1 (blocker): the seed-0 marker of every live reach run put
    the gripper body through the blue box's near wall."""
    why = tasks.reach_feasibility((-0.496, -0.159, 0.138), SCENE, tasks.START_Q)
    assert "blue box" in why


def test_reach_samples_are_feasible_for_seeds_0_to_49():
    for seed in range(50):
        m = tasks.sample_reach(np.random.default_rng(seed), {}, SCENE)["marker"]
        assert tasks.reach_feasibility(m, SCENE, tasks.START_Q) == "", (seed, m)


def test_explicit_infeasible_marker_raises_and_exhausted_sampler_raises():
    with pytest.raises(ValueError, match="infeasible"):
        tasks.sample_reach(np.random.default_rng(0), {"marker": [-0.496, -0.159, 0.138]}, SCENE)
    with pytest.raises(RuntimeError, match="no visible"):
        tasks.sample_reach(np.random.default_rng(0), {"marker_y": (0.05, 0.06)}, SCENE)   # all inside the box


def test_push_fails_when_the_packet_was_carried():
    """Review control-safety #18: pick up, carry and release on the square is not a push."""
    ep = tasks.sample_push(np.random.default_rng(0), {}, SCENE)
    ep["object_pos"] = np.asarray(ep["object_pos"])
    on_zone = (ep["zone"][0], ep["zone"][1], ep["object_pos"][2])
    assert tasks.evaluate_push(snapshot(on_zone), {**ep, "max_lift_m": 0.0, "ever_held": False})["success"]
    assert not tasks.evaluate_push(snapshot(on_zone), {**ep, "max_lift_m": 0.06})["success"]
    assert not tasks.evaluate_push(snapshot(on_zone), {**ep, "ever_held": True})["success"]
