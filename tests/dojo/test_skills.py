"""dojo.skills: kinematics, motion, grasps, executor, a whole program on the kinematic env. CPU only."""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from dojo.skills import grasps as G
from dojo.skills.env import KinematicEnv
from dojo.skills.executor import ArmPlan, Executor
from dojo.skills.geometry import R_to_quat, T_from, quat_to_R, rot_z
from dojo.skills.kinematics import TCP_OFFSET, X5Kin, down_R, gap_to_grip, grip_to_gap
from dojo.skills.motion import Collision, Motion, Segment, merge_steps, segment_steps, segment_targets
from dojo.skills.skills import GTProvider, Skills
from dojo.skills.world import AssetInfo, ObjectState

KIN = X5Kin()


def test_home_flange_matches_robodojo_frame():
    # all joints 0: flange 0.157 m in front of the base, 0.157 m up, pointing +y (forward)
    for arm, x in (("left", -0.3), ("right", 0.3)):
        F = KIN.flange(arm, np.zeros(6))
        assert F[:3, 3] == pytest.approx([x + 0.0005, -0.3523, 0.9215], abs=2e-3)
        assert F[:3, 0] == pytest.approx([0, 1, 0], abs=1e-3)
        T = KIN.tcp(arm, np.zeros(6))
        assert np.linalg.norm(T[:3, 3] - F[:3, 3]) == pytest.approx(TCP_OFFSET)


@pytest.mark.skipif(not any(Path(p, "X5A.urdf").exists() for p in (os.environ.get("DOJO_X5_DIR", "/nonexistent"),
                                                                      "/tmp/dojo-skills-dev/x5")),
                    reason="X5A.urdf not available")
def test_chain_constants_match_urdf():
    d = os.environ.get("DOJO_X5_DIR") or "/tmp/dojo-skills-dev/x5"
    k2 = X5Kin.from_urdf(Path(d, "X5A.urdf"))
    q = np.random.default_rng(3).uniform(-1, 1, 6)
    assert np.abs(KIN.flange("left", q) - k2.flange("left", q)).max() < 1e-12


@pytest.mark.parametrize("xyz,yaw", [((-0.25, -0.15, 0.80), 0.3), ((0.2, -0.2, 0.85), 1.2), ((0.0, -0.25, 0.92), -0.7)])
def test_ik_roundtrip(xyz, yaw):
    arm = "left" if xyz[0] <= 0 else "right"
    T = T_from(xyz, down_R(yaw))
    r = KIN.ik(arm, T, np.zeros(6))
    assert r.ok
    Tr = KIN.tcp(arm, r.q)
    assert np.linalg.norm(Tr[:3, 3] - T[:3, 3]) < 1e-3
    assert np.degrees(np.arccos(np.clip((np.trace(Tr[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1))) < 0.5


def test_ik_out_of_reach_fails_fast():
    r = KIN.ik("left", T_from((0.6, 0.3, 0.8), down_R(0)), np.zeros(6))
    assert not r.ok and r.note == "out of reach"


def test_gripper_mapping():
    assert grip_to_gap(1.0) == pytest.approx(0.088)
    assert grip_to_gap(0.0) == 0.0
    # the scorer's 0.8 banking threshold is a 66 mm gap
    assert grip_to_gap(0.8) == pytest.approx(0.0664, abs=1e-4)
    assert gap_to_grip(grip_to_gap(0.6)) == pytest.approx(0.6)


def test_segment_targets_respect_step():
    q = np.array([[0, 0, 0, 0, 0, 0], [0.5, -0.2, 0, 0, 0, 0], [0.5, 0.3, 0.1, 0, 0, 0]], float)
    t = segment_targets(q, 0.1)
    assert len(t) == segment_steps(q, 0.1) == 10
    prev = q[0]
    for x in t:
        assert np.abs(x - prev).max() <= 0.1 + 1e-9
        prev = x
    assert np.allclose(t[-1], q[-1])


def test_line_is_straight_and_checks_table():
    mot = Motion(KIN)
    q0 = KIN.ik("left", T_from((-0.25, -0.15, 0.90), down_R(0.0)), np.zeros(6)).q
    T1 = T_from((-0.25, -0.15, 0.80), down_R(0.0))
    mv = mot.line("left", q0, T1, 0.05)
    assert mv.ok
    for q in mv.segments[0].q:
        p = KIN.tcp("left", q)[:3, 3]
        assert abs(p[0] + 0.25) < 1e-3 and abs(p[1] + 0.15) < 1e-3
    # through the table: refused (fingertips 12.6 mm below the TCP)
    bad = mot.line("left", q0, T_from((-0.25, -0.15, 0.770), down_R(0.0)), 0.05)
    assert not bad.ok and ("table" in bad.note)


def _box(label, xyz, half, yaw=0.0, mesh=None, cat="block"):
    info = AssetInfo(key=f"Rigid/{cat}/00000", bbox_min=-np.asarray(half, float), bbox_max=np.asarray(half, float),
                     mesh=mesh)
    return ObjectState(label, "Rigid", cat, 0, np.asarray(xyz, float), R_to_quat(rot_z(yaw)), info)


def test_top_grasps_close_on_narrow_axes():
    o = _box("board", (-0.2, -0.15, 0.775), (0.19, 0.033, 0.01))
    gs = G.top_grasps(o, KIN, "left")
    assert gs and all(abs(g.width - 0.066) < 1e-6 for g in gs)        # only across the 65.5 mm width
    for g in gs:
        assert abs(g.T[:3, 1] @ np.array([0, 1.0, 0])) > 0.99            # closing axis = board y


def test_prism_grasps_only_on_end_faces():
    # triangular prism: triangle in x-y, extruded along z (like build_tower's block4 asset)
    V = np.array([[-0.03, -0.0125, -0.023], [0.03, -0.0125, -0.023], [0.0, 0.025, -0.023],
                  [-0.03, -0.0125, 0.023], [0.03, -0.0125, 0.023], [0.0, 0.025, 0.023]])
    F = np.array([[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]])
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)             # local y up
    info = AssetInfo(key="Rigid/block/00003", bbox_min=V.min(0), bbox_max=V.max(0), mesh=(V, F))
    o = ObjectState("prism", "Rigid", "block", 3, np.array([-0.2, -0.15, 0.7775]), R_to_quat(R), info)
    assert G.antipodal_axes(o) == [2]
    gs = G.top_grasps(o, KIN, "left")
    assert gs and all(abs(g.width - 0.046) < 1e-6 for g in gs)


def test_arm_plans_run_concurrently():
    env = KinematicEnv([], step_lim=200)
    ex = Executor(env)
    qa = KIN.ik("left", T_from((-0.25, -0.15, 0.85), down_R(0)), np.zeros(6)).q
    qb = KIN.ik("right", T_from((0.25, -0.15, 0.85), down_R(0)), np.zeros(6)).q
    pl = ArmPlan("left", ex.q["left"], 1.0).move([Segment(np.stack([np.zeros(6), qa]), 0.15)]).grip(0.0, 3)
    pr = ArmPlan("right", ex.q["right"], 1.0).move([Segment(np.stack([np.zeros(6), qb]), 0.15)])
    n = ex.run({"left": pl, "right": pr})["steps"]
    assert n == max(pl.steps, pr.steps) == env.steps_used         # both arms in the same env steps
    assert np.allclose(env.q["left"], qa, atol=1e-6) and np.allclose(env.q["right"], qb, atol=1e-6)
    assert env.g["left"] == 0.0 and env.g["right"] == 1.0
    a = ex.log.actions[0]
    assert set(a) == {"left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state"}


def test_merge_steps_holds_finished_arm():
    out = merge_steps({"left": [np.ones(6)], "right": [np.full(6, 2.0), np.full(6, 3.0)]},
                      {"left": np.zeros(6), "right": np.zeros(6)})
    assert len(out) == 2 and np.allclose(out[1]["left"], 1.0) and np.allclose(out[1]["right"], 3.0)


def _blocks_env(positions, half=0.015, step_lim=550):
    rows = [{"label": f"block_{i}", "category": "block", "category_idx": 5, "pos": p, "quat": [1, 0, 0, 0],
             "half": [half] * 3} for i, p in enumerate(positions)]
    return KinematicEnv(rows, step_lim=step_lim)


@pytest.mark.parametrize("positions", [
    [(0.412, -0.09, 0.78), (-0.26, -0.189, 0.78), (-0.344, -0.1, 0.78)],   # far apart: shared site
    [(-0.358, -0.206, 0.78), (-0.449, -0.092, 0.78), (-0.3, -0.165, 0.78)],  # all on the left
])
def test_stack_blocks_program_on_kinematic_env(positions):
    from dojo.skills.programs import stack_blocks
    env = _blocks_env(positions)
    sk = Skills(env, GTProvider(env, None, "stack_blocks"))
    res = stack_blocks.run(sk)
    assert res["ok"], res
    p = sorted((np.asarray(o["pos"]) for o in env.objects), key=lambda v: v[2])
    for a, b in zip(p[:-1], p[1:]):
        assert np.linalg.norm(a[:2] - b[:2]) < 0.005 and b[2] - a[2] == pytest.approx(0.03, abs=0.003)
    # home at the end, grippers open, well within the step limit
    assert all(np.abs(env.q[a]).max() < 0.02 for a in ("left", "right"))
    assert env.g["left"] == env.g["right"] == 1.0
    assert env.steps_used < 300


def test_collision_sees_neighbour_object():
    o = _box("wall", (-0.25, -0.15, 0.80), (0.05, 0.05, 0.035))
    from dojo.skills.world import WorldState
    w = WorldState("t", "", {"wall": o}, {}, {})
    coll = Collision(KIN, w)
    q = KIN.ik("left", T_from((-0.25, -0.15, 0.82), down_R(0)), np.zeros(6)).q
    ok, why = coll.check("left", q, 1.0)
    assert not ok and "wall" in why
    ok, _ = coll.check("left", q, 1.0, ignore=("wall",))
    assert ok


def test_quat_roundtrip():
    R = quat_to_R(R_to_quat(rot_z(1.0) @ down_R(0.3)))
    assert np.allclose(R, rot_z(1.0) @ down_R(0.3), atol=1e-9)
    assert math.isclose(np.linalg.det(R), 1.0, rel_tol=1e-9)


def test_verify_mode_picks_a_passing_candidate():
    from dojo.skills.env import KinematicVerifier
    from dojo.skills.programs import stack_blocks
    env = _blocks_env([(-0.25, -0.15, 0.78), (-0.33, -0.05, 0.78), (0.25, -0.2, 0.78)])
    sk = Skills(env, GTProvider(env, None, "stack_blocks"), verifier=KinematicVerifier(env), verify_k=1, n_cands=3)
    res = stack_blocks.run(sk)
    assert res["ok"], res
    assert sk.vstats["segments"] >= 3 and sk.vstats["none_ok"] == 0
    p = sorted((np.asarray(o["pos"]) for o in env.objects), key=lambda v: v[2])
    for a, b in zip(p[:-1], p[1:]):
        assert np.linalg.norm(a[:2] - b[:2]) < 0.005
