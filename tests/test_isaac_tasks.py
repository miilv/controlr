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


def test_reach_targets_are_text_relative_to_objects_and_reachable():
    """Ilia: the text instruction must be enough — no marker; the target is named relative to
    a visible object and checked from that object's CURRENT pose."""
    q0 = np.asarray(tasks.START_Q)
    rv0 = KIN.fk(q0)[1]
    seen = set()
    for seed in range(40):
        s = tasks.sample_reach(np.random.default_rng(seed), {}, SCENE)
        t = s["reach_target"]
        seen.add(t["index"])
        assert t["text"] in s["instruction"] and "red ball" not in s["instruction"]
        assert "marker" not in s and "zone" not in s
        pt = tasks.reach_target_point(t, s["object_pos"], tasks.yaw_quat_wxyz(s["object_yaw"]),
                                      SCENE["object"]["size"], SCENE["bin"])
        np.testing.assert_allclose(pt, t["point0"])
        u, v = tasks.project(SCENE, pt)
        assert 0 <= u < 640 and 0 <= v < 480, "the target must be inside the D435 image"
        # reachable while keeping the start orientation (action.rotation=none)
        assert KIN.ik(pt, rv0, q0) is not None, f"seed {seed}: target {pt} unreachable"
    assert len(seen) >= 3, seen
    # packet top: 60 mm above the top face of the packet, wherever the packet is now
    tgt = {"ref": "packet_top", "offset": [0, 0, 0.06]}
    pt = tasks.reach_target_point(tgt, [-0.4, -0.27, 0.0355], tasks.yaw_quat_wxyz(0.3), [0.17, 0.035, 0.09], None)
    np.testing.assert_allclose(pt, [-0.4, -0.27, 0.0355 + 0.045 + 0.06])
    # box corner moves with the box
    moved = tasks.bin_info_at(SCENE["bin"], (np.asarray(SCENE["bin"]["center"]), tasks.yaw_quat_wxyz(SCENE["bin"]["yaw"])),
                              np.asarray(SCENE["bin"]["center"]) + [0.03, 0.0, 0.0],
                              tasks.yaw_quat_wxyz(SCENE["bin"]["yaw"]))
    a = tasks.reach_reference("box_near_right", None, None, None, SCENE["bin"])
    b = tasks.reach_reference("box_near_right", None, None, None, moved)
    np.testing.assert_allclose(b - a, [0.03, 0.0, 0.0], atol=1e-9)


def test_push_target_is_text_along_the_long_axis():
    for seed in range(50):
        s = tasks.sample_push(np.random.default_rng(seed), {}, SCENE)
        tgt = s["push_target"]
        d = tgt["point"][:2] - s["object_pos"][:2]
        assert 0.06 - 1e-9 <= np.linalg.norm(d) <= 0.10 + 1e-9
        axis = np.array([np.cos(s["object_yaw"]), np.sin(s["object_yaw"])])
        assert abs(abs(np.dot(d / np.linalg.norm(d), axis)) - 1.0) < 1e-9 and d[0] < 0
        mat_c, half = np.asarray(SCENE["mat"]["center"][:2]), np.asarray(SCENE["mat"]["size"][:2]) / 2
        assert np.all(np.abs(tgt["point"][:2] - mat_c) <= half)
        assert tgt["text"] in s["instruction"] and "green square" not in s["instruction"]
        assert f"{int(round(np.linalg.norm(d) * 100)) * 10} mm" in s["instruction"]


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
    c = np.asarray(SCENE["object"]["center"])
    top = c + [0, 0, 0.045 + 0.06]                                  # 60 mm above the packet top (yaw 0)
    ep = {"reach_target": {"ref": "packet_top", "offset": [0, 0, 0.06], "point0": top},
          "tcp_pos0": top + [0.07, -0.03, 0.25], "object_pos": c,
          "push_target": {"point": np.array([-0.47, -0.29, -0.0125]), "distance": 0.08}, "params": {}}
    assert tasks.evaluate_reach(snapshot(c, tcp=top + [0.01, 0, 0]), ep)["success"]
    far = tasks.evaluate_reach(snapshot(c, tcp=top + [0.07, -0.03, 0.25]), ep)
    assert not far["success"] and far["progress"] == 0.0
    # the target follows the packet's CURRENT pose: the packet was knocked 50 mm
    moved = tasks.evaluate_reach(snapshot(c + [0.05, 0, 0], tcp=top + [0.05, 0, 0]), ep)
    assert moved["success"]
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
# review fixes (docs/reviews/2026-10-02-fixlog.md)
# ---------------------------------------------------------------------------

def test_old_seed0_reach_marker_is_rejected_as_infeasible():
    """Review control-safety #1 (blocker): the seed-0 marker of every live reach run put
    the gripper body through the blue box's near wall."""
    why = tasks.reach_feasibility((-0.496, -0.159, 0.138), SCENE, tasks.START_Q)
    assert "blue box" in why


def test_reach_samples_are_feasible_for_seeds_0_to_49():
    for seed in range(50):
        t = tasks.sample_reach(np.random.default_rng(seed), {}, SCENE)["reach_target"]
        assert tasks.reach_feasibility(t["point0"], SCENE, tasks.START_Q) == "", (seed, t)


def test_explicit_target_choice_and_no_feasible_target_raises():
    s = tasks.sample_reach(np.random.default_rng(0), {"target_index": 2}, SCENE)
    assert s["reach_target"]["ref"] == "box_opening"
    with pytest.raises(ValueError, match="none of the targets"):
        tasks.sample_reach(np.random.default_rng(0), {"targets": [0], "start_q": [0, -1.57, 0, -1.57, 0, 0]}, SCENE)


def test_push_fails_when_the_packet_was_carried():
    """Review control-safety #18: pick up, carry and release on the square is not a push."""
    ep = tasks.sample_push(np.random.default_rng(0), {}, SCENE)
    ep["object_pos"] = np.asarray(ep["object_pos"])
    on_zone = (ep["push_target"]["point"][0], ep["push_target"]["point"][1], ep["object_pos"][2])
    assert tasks.evaluate_push(snapshot(on_zone), {**ep, "max_lift_m": 0.0, "ever_held": False})["success"]
    assert not tasks.evaluate_push(snapshot(on_zone), {**ep, "max_lift_m": 0.06})["success"]
    assert not tasks.evaluate_push(snapshot(on_zone), {**ep, "ever_held": True})["success"]


# ---------------------------------------------------------------------------
# rotation experiment (docs/experiments/2026-10-02-rotation-yaw.md): wider packet yaw, start yaw, scene record
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_default_sampling_is_unchanged_by_the_new_params(seed):
    """Defaults stay backward compatible: same RNG draws, recorded start joints."""
    rng = np.random.default_rng(seed)
    xy = np.clip(rng.normal(0.0, 0.010, 2), -0.020, 0.020)
    yaw = float(np.clip(rng.normal(0.0, np.radians(5.0)), -np.radians(10), np.radians(10)))
    s = tasks.sample_waffle(np.random.default_rng(seed), {}, SCENE)
    np.testing.assert_allclose(s["object_pos"][:2], np.asarray(SCENE["object"]["center"][:2]) + xy)
    assert s["object_yaw"] == pytest.approx(SCENE["object"]["yaw"] + yaw)
    np.testing.assert_allclose(s["start_q"], tasks.START_Q)
    assert s["start_yaw_offset"] == 0.0


def test_uniform_packet_yaw_covers_the_range():
    offs = [tasks.sample_waffle(np.random.default_rng(k), {"yaw_dist": "uniform", "yaw_max_deg": 40}, SCENE)
            ["object_yaw_offset"] for k in range(200)]
    offs = np.degrees(offs)
    assert offs.min() >= -40 - 1e-9 and offs.max() <= 40 + 1e-9
    assert offs.min() < -30 and offs.max() > 30
    with pytest.raises(ValueError):
        tasks.sample_waffle(np.random.default_rng(0), {"yaw_dist": "cauchy"}, SCENE)


@pytest.mark.parametrize("seed", range(8))
def test_start_yaw_keeps_the_tcp_and_tilt_and_turns_the_heading(seed):
    from controlr.robot.kinematics import matrix_to_rpy
    q_sim = (0.1796, -1.4011, 0.8725, 1.176, 1.2852, -2.9406)        # configs/sim_waffle.yaml start_q
    s = tasks.sample_waffle(np.random.default_rng(seed), {"start_yaw_deg": 20, "start_q": q_sim}, SCENE)
    ref = tasks.sample_waffle(np.random.default_rng(seed), {"start_q": q_sim}, SCENE)
    np.testing.assert_allclose(s["object_pos"], ref["object_pos"])       # drawn after the packet
    assert abs(np.degrees(s["start_yaw_offset"])) <= 20
    T0, T1 = KIN.fk_matrix(q_sim), KIN.fk_matrix(s["start_q"])
    assert np.linalg.norm(T1[:3, 3] - T0[:3, 3]) < 1e-3
    r0, r1 = matrix_to_rpy(T0[:3, :3]), matrix_to_rpy(T1[:3, :3])
    np.testing.assert_allclose(r1[:2], r0[:2], atol=np.radians(0.5))
    dyaw = (r1[2] - r0[2] + np.pi) % (2 * np.pi) - np.pi
    assert dyaw == pytest.approx(s["start_yaw_offset"], abs=np.radians(0.5))
    assert np.max(np.abs(np.asarray(s["start_q"]) - q_sim)) < np.radians(60)    # same IK branch


def test_start_yaw_range_pair_and_unreachable_start_raises():
    s = tasks.sample_waffle(np.random.default_rng(3), {"start_yaw_deg": [10, 10]}, SCENE)
    assert np.degrees(s["start_yaw_offset"]) == pytest.approx(10)
    with pytest.raises(ValueError, match="reach limit"):
        tasks.rotate_start_q((0.1796, -1.4011, 0.8725, 1.176, 1.2852, -2.9406), np.radians(-45))


def test_scene_record_has_packet_box_start_and_targets():
    from controlr.robot.kinematics import matrix_to_rpy
    s = tasks.sample_waffle(np.random.default_rng(2), {"yaw_dist": "uniform", "yaw_max_deg": 40,
                                                       "start_yaw_deg": 15}, SCENE)
    ep = {"task": "waffle_pick_place", "seed": 2, "object_pos": s["object_pos"],
          "object_quat_wxyz": tasks.yaw_quat_wxyz(s["object_yaw"]), "object_pos_sampled": s["object_pos"],
          "object_yaw_sampled": s["object_yaw"], "start_q": s["start_q"],
          "start_yaw_offset": s["start_yaw_offset"], "start_gripper": "open", "settle_drift_m": 0.0004,
          "box_dynamic": True}
    rec = tasks.scene_record(ep, SCENE)
    import json
    json.dumps(rec)                                               # plain JSON
    assert rec["packet"]["yaw_offset_deg"] == pytest.approx(np.degrees(s["object_yaw_offset"]), abs=1e-3)
    assert rec["packet"]["yaw_deg"] == pytest.approx(np.degrees(s["object_yaw"]), abs=1e-3)
    assert rec["packet"]["size_mm"] == [170.0, 35.0, 90.0] and rec["packet"]["tilt_deg"] == pytest.approx(0, abs=1e-6)
    assert rec["box"]["center_mm"][0] == pytest.approx(-390.5)
    assert rec["start"]["yaw_offset_deg"] == pytest.approx(np.degrees(s["start_yaw_offset"]), abs=1e-3)
    yaw0 = np.degrees(matrix_to_rpy(KIN.fk_matrix(s["start_q"])[:3, :3])[2])
    assert rec["start"]["tcp_yaw_deg"] == pytest.approx(yaw0, abs=1e-3)
    assert rec["reach_target"] is None and rec["push_target"] is None and rec["box"]["dynamic"] is True
    r = tasks.sample_reach(np.random.default_rng(1), {}, SCENE)
    rec = tasks.scene_record({**ep, "reach_target": r["reach_target"]}, SCENE)
    json.dumps(rec)
    assert rec["reach_target"]["text"] == r["reach_target"]["text"] and len(rec["reach_target"]["point_mm"]) == 3


def test_explicit_packet_yaw_offset():
    for nominal in (False, True):
        s = tasks.sample_waffle(np.random.default_rng(4), {"yaw_offset_deg": -40, "nominal": nominal}, SCENE)
        assert np.degrees(s["object_yaw_offset"]) == pytest.approx(-40)
    a = tasks.sample_waffle(np.random.default_rng(4), {"yaw_offset_deg": 30}, SCENE)
    b = tasks.sample_waffle(np.random.default_rng(4), {}, SCENE)
    np.testing.assert_allclose(a["object_pos"], b["object_pos"])


SIM_START_Q = (0.1796, -1.4011, 0.8725, 1.176, 1.2852, -2.9406)      # configs/sim_waffle.yaml


@pytest.mark.parametrize("start_yaw", [-20, 0, 20])
@pytest.mark.parametrize("offset,dy", [(-40, 0.0), (-15, 0.0), (15, 0.0), (25, 0.0), (30, 0.0), (35, -0.02)])
def test_scripted_yaw_expert_is_feasible_through_the_envelope(offset, dy, start_yaw):
    """Kinematic twin of tests/test_isaac_sim.py::test_scripted_yaw_expert_succeeds: every
    action of the closed-loop yaw expert passes the rotation=yaw envelope unclamped."""
    from controlr.config import SafetyConfig
    from controlr.robot.safety import SafetyEnvelope
    from controlr.robot.spec import ur3_cb3_spec
    from controlr.types import Action, ActionMode, RobotState

    spec = ur3_cb3_spec(table_z=-0.0095)
    q0 = tasks.rotate_start_q(SIM_START_Q, np.radians(start_yaw))
    env = SafetyEnvelope(spec, SafetyConfig(), KIN, rotation="yaw")

    def state(q, g):
        p, rv = KIN.fk(q)
        return RobotState(0.0, np.asarray(q), p, rv, g * 1000, False, False)

    env.reset(state(q0, 0.0913))
    cur = {"q": np.asarray(q0), "g": 0.0913}

    def act(vals, grip, phase=""):
        out, ev = env.filter([Action(ActionMode.EE_DELTA, vals, grip)], state(cur["q"], cur["g"]))
        bad = [e.message for e in ev if e.kind in ("reach", "ik_fail", "step_limit", "workspace", "table")]
        assert out and not bad, bad
        if phase not in ("lower", "release", "retreat"):       # those are meant to be in the box
            for qq in list(out[0].q_path or ()) + [out[0].q_target]:
                clear[phase] = min(clear.get(phase, 1.0), tasks.body_box_clearance(qq, SCENE))
        cur["q"] = np.asarray(out[0].q_target)
        cur["g"] = grip if grip is not None else cur["g"]
        return KIN.fk_matrix(cur["q"])

    clear: dict[str, float] = {}
    yaw = SCENE["object"]["yaw"] + np.radians(offset)
    obj = np.asarray(SCENE["object"]["center"]) + [0.0, dy, 0.0]
    log = tasks.run_scripted_yaw_pick_place(obj, yaw, SCENE["bin"]["center"], act, KIN.fk_matrix(q0))
    # the wrist / housing keep > 55 mm from the box (47 mm meant contact in Isaac)
    assert min(clear.values()) > 0.055, {k: round(v * 1000) for k, v in clear.items()}
    R = [r for r in log if r["phase"] == "close"][0]["T_after"][:3, :3]
    thin = np.array([-np.sin(yaw), np.cos(yaw)])
    assert abs(np.dot(R[:2, 0] / np.linalg.norm(R[:2, 0]), thin)) > 0.999      # jaws across the thin axis


@pytest.mark.parametrize("offset,dy", [(40, 0.0), (40, -0.02), (25, 0.02)])
def test_large_positive_packet_yaw_is_boxed_in(offset, dy):
    """The limit of the graspable range (docs/experiments/2026-10-02-rotation-yaw.md): with the demo tilt, jaws
    turned +40 deg put the gripper housing within ~50 mm of the box's near wall, and moving the
    grasp away from the box runs out of reach; +25 deg fails once the packet sits 20 mm closer
    to the box. Rotation alone cannot solve these (a different tilt would be needed)."""
    with pytest.raises(AssertionError):
        test_scripted_yaw_expert_is_feasible_through_the_envelope(offset, dy, 0)


def test_goal_check_follows_a_moved_and_tilted_box():
    """The box is a dynamic body (contacts-and-speed): resting partly on the 3 mm mat it tips
    ~0.7 deg, and a push moves it. A packet standing on the (tilted, moved) floor is inside;
    a packet where the box USED to be is not."""
    b0 = SCENE["bin"]
    c0 = np.asarray(b0["center"])
    pose0 = (c0, tasks.yaw_quat_wxyz(b0["yaw"]))
    tilt = np.radians(0.7)
    q_tilt = np.array([np.cos(tilt / 2), np.sin(tilt / 2), 0.0, 0.0])          # about +x (box far side down)
    def qmul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    shift = np.array([0.06, 0.02, 0.0015])
    b = tasks.bin_info_at(b0, pose0, c0 + shift, qmul(q_tilt, pose0[1]))
    assert b["tilt_deg"] == pytest.approx(0.7, abs=0.01) and b["shift_m"] == pytest.approx(np.linalg.norm(shift))
    R = tasks.quat_wxyz_to_matrix(q_tilt)
    floor_pt = c0 + shift + R @ np.array([0.0, -0.03, 0.004])                     # on the floor, 30 mm toward -y
    pos = floor_pt + R @ np.array([0.0, 0.0, 0.045])                               # packet centre, standing on it
    snap = snapshot(pos)
    snap["object_quat_wxyz"] = qmul(q_tilt, tasks.yaw_quat_wxyz(0.0))
    snap["bin"] = b
    ep = {"object_pos": np.asarray(SCENE["object"]["center"]), "params": {}}
    assert tasks.evaluate_waffle(snap, ep)["metrics"]["inside_bin"]
    # a packet inside where the box was before the 60 mm push now hangs over its -x wall
    before = snapshot(np.asarray([-0.47, 0.065, 0.04]))
    assert tasks.evaluate_waffle({**before, "bin": b0}, ep)["metrics"]["inside_bin"]
    assert not tasks.evaluate_waffle({**before, "bin": b}, ep)["metrics"]["inside_bin"]
