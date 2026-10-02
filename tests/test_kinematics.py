"""UR3 CB3 kinematics: FK vs PHANTOM, Jacobian vs finite differences, IK
round trips, rotation helpers."""

from __future__ import annotations

import time

import numpy as np
import pytest

from controlr.robot import kinematics as K
from controlr.robot.kinematics import (
    IKOptions, UR3Kinematics, matrix_to_rotvec, matrix_to_rpy, rotation_angle, rotvec_to_matrix,
    rpy_to_matrix, rpy_to_rotvec, rotvec_to_rpy,
)
from controlr.robot.spec import HOME_Q_PHANTOM, HOME_Q_TOPDOWN

# Reference poses computed with PHANTOM phantom/sim/kinematics.py::forward_pose
# (scipy rotations) on compute3: (q, TCP pose with +0.18 m offset, tool0 pose).
PHANTOM_REF = [
    ([0, 0, 0, 0, 0, 0],
     [-0.4569, -0.37425, 0.06655, 1.570796327, 0.0, 0.0],
     [-0.4569, -0.19425, 0.06655, 1.570796327, 0.0, 0.0]),
    ([0.1804, -1.53, 1.2218, 0.5795, 1.2155, -3.1032],
     [-0.383396828, -0.276737828, 0.312005661, -1.170324488, -1.790921768, 1.494537164],
     [-0.234684845, -0.185966145, 0.357230076, -1.170324488, -1.790921768, 1.494537164]),
    ([0.001427, -1.5621, 1.588, 0.00954, 1.5737, -3.0957],
     [-0.473847862, -0.112265824, 0.295442069, -1.270231599, -1.211565629, 1.224263785],
     [-0.293961085, -0.112531786, 0.301819907, -1.270231599, -1.211565629, 1.224263785]),
    ([0.3, -1.0, 1.0, -1.2, -1.5, 0.4],
     [-0.276374751, -0.222487468, 0.082507611, 2.154561285, 1.923335083, 0.447991494],
     [-0.342292674, -0.229550301, 0.249854388, 2.154561285, 1.923335083, 0.447991494]),
    ([-0.7, -2.0, -1.3, 2.0, 0.6, -1.0],
     [-0.066173705, -0.373770081, 0.459470877, 1.159211319, 0.942515627, -2.749338867],
     [0.050325622, -0.277659426, 0.361539019, 1.159211319, 0.942515627, -2.749338867]),
]


def _rot_close(rv1, rv2, tol=1e-6):
    return rotation_angle(rotvec_to_matrix(rv1) @ rotvec_to_matrix(rv2).T) < tol


@pytest.mark.parametrize("q,tcp,tool0", PHANTOM_REF)
def test_fk_matches_phantom(q, tcp, tool0):
    pos, rv = UR3Kinematics().fk(q)
    assert np.allclose(pos, tcp[:3], atol=1e-8)
    assert _rot_close(rv, tcp[3:])
    pos0, rv0 = UR3Kinematics(tcp_offset=(0, 0, 0, 0, 0, 0)).fk(q)
    assert np.allclose(pos0, tool0[:3], atol=1e-8)
    assert _rot_close(rv0, tool0[3:])


def test_fk_zero_pose_by_hand():
    # q = 0: arm stretched along -x: x = a2 + a3, y = -(d4 + d6 + tcp), z = d1 - d5
    pos, _ = K.fk(np.zeros(6))
    assert np.allclose(pos, [-0.24365 - 0.21325, -(0.11235 + 0.0819 + 0.18), 0.1519 - 0.08535])


def test_tcp_offset_is_along_tool_z():
    k0, k = UR3Kinematics(tcp_offset=(0,) * 6), UR3Kinematics()
    q = np.array(HOME_Q_PHANTOM)
    T0 = k0.fk_matrix(q)
    assert np.allclose(k.fk_matrix(q)[:3, 3], T0[:3, 3] + 0.18 * T0[:3, 2])


def test_topdown_home_points_down():
    T = UR3Kinematics().fk_matrix(HOME_Q_TOPDOWN)
    assert T[2, 2] < -0.999                         # tool z = -base z
    assert np.allclose(T[:3, 3], [-0.30, -0.15, 0.15], atol=1e-3)


def test_jacobian_matches_finite_differences():
    k = UR3Kinematics()
    q = np.array([0.3, -1.0, 1.0, -1.2, -1.5, 0.4])
    J = k.jacobian(q)
    eps = 1e-6
    T0 = k.fk_matrix(q)
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        T1 = k.fk_matrix(q + dq)
        v = (T1[:3, 3] - T0[:3, 3]) / eps
        w = matrix_to_rotvec(T1[:3, :3] @ T0[:3, :3].T) / eps
        assert np.allclose(J[:3, i], v, atol=1e-5)
        assert np.allclose(J[3:, i], w, atol=1e-5)


def test_rotation_helpers_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        rv = rng.normal(size=3)
        rv *= rng.uniform(0, np.pi) / np.linalg.norm(rv)
        assert _rot_close(matrix_to_rotvec(rotvec_to_matrix(rv)), rv, 1e-9)
        rpy = rotvec_to_rpy(rv)
        assert _rot_close(rpy_to_rotvec(rpy), rv, 1e-9)
    # near pi and identity
    for rv in ([np.pi, 0, 0], [0, 0, np.pi - 1e-7], [1e-9, 0, 0], [0, 0, 0]):
        assert _rot_close(matrix_to_rotvec(rotvec_to_matrix(rv)), rv, 1e-6)
    # extrinsic xyz convention: yaw alone is a rotation about base z
    assert np.allclose(rpy_to_matrix([0, 0, np.pi / 2]) @ [1, 0, 0], [0, 1, 0])
    # roll then yaw (extrinsic): R = Rz @ Rx
    R = rpy_to_matrix([0.3, 0, 0.5])
    assert np.allclose(R, rpy_to_matrix([0, 0, 0.5]) @ rpy_to_matrix([0.3, 0, 0]))
    assert np.allclose(matrix_to_rpy(R), [0.3, 0, 0.5])
    # gimbal lock still reproduces the matrix
    R = rpy_to_matrix([0.2, np.pi / 2, 0.7])
    assert np.allclose(rpy_to_matrix(matrix_to_rpy(R)), R, atol=1e-9)


def test_ik_roundtrip_200_random_reachable_poses():
    """Targets from FK of random joint configs; seeds perturbed by up to 0.3 rad
    (the safety envelope's situation: small steps from the current pose)."""
    k = UR3Kinematics()
    rng = np.random.default_rng(42)
    home = np.array(HOME_Q_TOPDOWN)
    ok, perr, rerr = 0, [], []
    t0 = time.perf_counter()
    for _ in range(200):
        q = home + rng.uniform(-1.0, 1.0, 6)
        pos, rv = k.fk(q)
        seed = q + rng.uniform(-0.3, 0.3, 6)
        sol = k.ik(pos, rv, seed)
        if sol is None:
            continue
        ok += 1
        p2, r2 = k.fk(sol)
        perr.append(np.linalg.norm(p2 - pos))
        rerr.append(rotation_angle(rotvec_to_matrix(r2) @ rotvec_to_matrix(rv).T))
    dt = time.perf_counter() - t0
    assert ok >= 198, f"IK success {ok}/200"
    assert max(perr) <= 1e-3 and max(rerr) <= np.deg2rad(0.5)
    assert np.median(perr) < 1e-4                    # iterates well past the acceptance tolerance
    assert dt < 30.0


def test_ik_from_far_seed_uses_restarts():
    """From the home seed to a random reachable pose (often needs restarts)."""
    k = UR3Kinematics()
    rng = np.random.default_rng(7)
    home = np.array(HOME_Q_TOPDOWN)
    ok = 0
    for _ in range(40):
        q = home + rng.uniform(-1.0, 1.0, 6)
        pos, rv = k.fk(q)
        sol = k.ik(pos, rv, home)
        if sol is not None:
            ok += 1
            p2, r2 = k.fk(sol)
            assert np.linalg.norm(p2 - pos) <= 1e-3
    assert ok >= 36, f"{ok}/40"


def test_ik_unreachable_returns_none():
    k = UR3Kinematics()
    assert k.ik([1.5, 0, 0.2], [np.pi, 0, 0], HOME_Q_TOPDOWN) is None
    # tool-down at 0.4 m height above the mat is out of reach for a UR3 + 18 cm TCP
    assert k.ik([-0.35, -0.15, 0.40], rpy_to_rotvec([np.pi, 0, 0]), HOME_Q_TOPDOWN) is None


def test_ik_respects_joint_limits():
    k = UR3Kinematics()
    q = np.array(HOME_Q_TOPDOWN)
    pos, rv = k.fk(q)
    lim = np.array([[-np.pi, np.pi]] * 6)
    lim[5] = [-4.0, 0.0]                           # wrist_3 home is -4.59: excluded
    sol = k.ik(pos, rv, np.clip(q, lim[:, 0], lim[:, 1]), lim)
    if sol is not None:
        assert np.all(sol >= lim[:, 0] - 1e-12) and np.all(sol <= lim[:, 1] + 1e-12)
        assert np.linalg.norm(k.fk(sol)[0] - pos) <= 1e-3
    # a range that makes the pose impossible
    lim2 = UR3Kinematics().joint_limits.copy()
    lim2[1] = [0.0, 0.5]                           # shoulder lift forced upward-ish
    assert k.ik(pos, rv, np.clip(q, lim2[:, 0], lim2[:, 1]), lim2) is None


def test_ik_lock_and_free_orientation():
    k = UR3Kinematics()
    q = np.array(HOME_Q_TOPDOWN)
    p0, rv0 = k.fk(q)
    target = p0 + [0.03, -0.04, -0.05]
    sol = k.ik(target, None, q, orientation="lock")
    assert sol is not None
    p, rv = k.fk(sol)
    assert np.linalg.norm(p - target) < 1e-3 and _rot_close(rv, rv0, np.deg2rad(0.5))
    sol = k.ik(target, None, q, orientation="free")
    assert sol is not None and np.linalg.norm(k.fk(sol)[0] - target) < 1e-3
    with pytest.raises(ValueError):
        k.ik(target, None, q, orientation="full")


def test_ik_branch_guard():
    k = UR3Kinematics()
    q = np.array(HOME_Q_TOPDOWN)
    pos, rv = k.fk(q + [0.6, 0, 0, 0, 0, 0])
    assert k.ik(pos, rv, q, options=IKOptions(max_joint_delta=0.3)) is None
    assert k.ik(pos, rv, q, options=IKOptions(max_joint_delta=1.0)) is not None


def test_module_level_functions():
    q = PHANTOM_REF[1][0]
    pos, rv = K.fk(q)
    sol = K.ik(pos, rv, np.array(q) + 0.05)
    assert sol is not None and np.linalg.norm(K.fk(sol)[0] - pos) < 1e-3
    assert K.jacobian(q).shape == (6, 6)
