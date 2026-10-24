"""dojo.perception: camera model, FK, silhouettes, task candidates, metrics. No GPU, no assets.

The perception extra (scipy, opencv) is needed; without it the module is skipped (CI installs only dev).
"""
import textwrap

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from dojo.perception import cameras  # noqa: E402
from dojo.perception.eval_g1 import rot_err_deg, yaw_err_deg  # noqa: E402
from dojo.perception.render import cluster_decimate, silhouette  # noqa: E402
from dojo.perception.robot_model import X5Model  # noqa: E402
from dojo.perception.tasks import task_candidates  # noqa: E402


def test_head_camera_constants():
    cam = cameras.head_camera()
    assert cam.K[0, 0] == pytest.approx(640 * 10.0 / 22.212)  # 288.13 px
    assert cam.K[0, 0] == cam.K[1, 1] and cam.K[0, 2] == 320 and cam.K[1, 2] == 240
    # optical axis (OpenCV +z) points forward (+y) and 60 deg down: euler XYZ (30, 0, 0) on a -z USD camera
    z = cam.T_wc[:3, 2]
    assert z == pytest.approx([0.0, 0.5, -np.sqrt(3) / 2], abs=1e-9)
    # image right is world +x
    assert cam.T_wc[:3, 0] == pytest.approx([1, 0, 0], abs=1e-9)


def test_project_ray_plane_roundtrip():
    cam = cameras.head_camera()
    rng = np.random.default_rng(0)
    X = np.c_[rng.uniform(-0.4, 0.4, 50), rng.uniform(-0.3, 0.2, 50), np.full(50, 0.78)]
    uv, z = cam.project(X)
    assert (z > 0).all()
    X2 = cam.ray_plane(uv, 0.78)
    assert np.abs(X2 - X).max() < 1e-9
    # the principal point maps onto the optical axis
    d = cam.rays(np.array([[320.0, 240.0]]))[0]
    assert d == pytest.approx(cam.T_wc[:3, 2])


def test_wrist_mount_is_a_rotation():
    T = cameras.wrist_mount_in_link()
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3))
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)


URDF = textwrap.dedent("""
<robot name="t">
  <link name="base_link"/><link name="a"/><link name="b"/>
  <joint name="joint1" type="revolute"><origin xyz="0 0 0.1" rpy="0 0 0"/><parent link="base_link"/><child link="a"/><axis xyz="0 0 1"/></joint>
  <joint name="joint2" type="prismatic"><origin xyz="0.2 0 0" rpy="0 0 1.5707963267948966"/><parent link="a"/><child link="b"/><axis xyz="1 0 0"/></joint>
</robot>
""")


def test_urdf_fk(tmp_path):
    p = tmp_path / "t.urdf"
    p.write_text(URDF)
    m = X5Model.load(p)
    m.arm_joints = ("joint1", "joint2")
    poses = m.fk([np.pi / 2, 0.05])
    # joint1 turns 90 deg about z: link a's x axis -> world y; b sits 0.2 along a's x, then 0.05 along b's x
    assert poses["a"][:3, 3] == pytest.approx([0, 0, 0.1])
    bx = poses["b"][:3, 0]
    assert bx == pytest.approx([-1, 0, 0], abs=1e-9)  # a.x = +y, b = a rotated 90 deg more
    assert poses["b"][:3, 3] == pytest.approx([0 - 0.05, 0.2, 0.1], abs=1e-9)


def test_silhouette_area_of_a_square():
    cam = cameras.Camera("t", cameras.intrinsics(10, 20, 640, 480), np.eye(4))
    # a 0.2 x 0.2 square at z = 1 m in front of a camera at the origin looking down +z (OpenCV)
    V = np.array([[-0.1, -0.1, 1], [0.1, -0.1, 1], [0.1, 0.1, 1], [-0.1, 0.1, 1.0]])
    F = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 2]])  # an overlapping triangle must not cancel
    m = silhouette(cam, V, F)
    side = 0.2 * cam.K[0, 0]
    # fillConvexPoly includes boundary pixels: about one extra pixel row/column
    assert (side - 1) ** 2 <= m.sum() <= (side + 2) ** 2


def test_cluster_decimate_keeps_extent():
    import itertools

    g = np.linspace(0, 0.05, 30)
    V = np.array(list(itertools.product(g, g, [0.0])))
    F = []
    n = len(g)
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            F += [[a, b, d], [a, d, c]]
    V2, F2 = cluster_decimate(V, np.array(F), 0.01)
    assert len(F2) < len(F)
    assert np.abs(V2.max(0)[:2] - V.max(0)[:2]).max() < 0.01


def test_task_candidates_from_config():
    cfg = {
        "Rigid": [
            {"common": {"xlim": [-0.4, 0.4], "rotate_rand": True},
             "category": [{"name": "block", "index": [5, 8]}, {"name": "block", "index": [6]}],
             "select_mode": {"label": ["b0", "b1"], "mode": "unique"}},
        ],
        "Geometry": [{"common": {"xlim": [0, 0]}, "category": [{"name": "pad", "index": [0]}],
                      "select_mode": {"label": ["target"]}}],
    }
    out = task_candidates(cfg, None)
    assert out["b0"]["candidates"] == ["Rigid/block/00005", "Rigid/block/00006", "Rigid/block/00008"]
    assert out["b1"]["candidates"] == out["b0"]["candidates"]
    assert out["target"]["type"] == "Geometry" and out["target"]["candidates"] == ["Geometry/pad/00000"]


def test_rotation_errors_respect_symmetry():
    q0 = [1, 0, 0, 0]
    q90 = cameras.R_to_quat_wxyz(cameras.euler_xyz_deg([0, 0, 90]))
    assert rot_err_deg(q90, q0, sym=1) == pytest.approx(90, abs=1e-6)
    assert rot_err_deg(q90, q0, sym=4) == pytest.approx(0, abs=1e-6)
    assert rot_err_deg(q90, q0, sym=36) == pytest.approx(0, abs=1e-6)
    q5 = cameras.R_to_quat_wxyz(cameras.euler_xyz_deg([0, 0, 95]))
    assert yaw_err_deg(q5, q0, sym=4) == pytest.approx(5, abs=1e-6)
    tilt = cameras.R_to_quat_wxyz(cameras.euler_xyz_deg([10, 0, 0]))
    assert rot_err_deg(tilt, q0, sym=36) == pytest.approx(10, abs=1e-6)


def test_world_z_symmetry_of_placed_shapes():
    from dojo.perception.inverse_layout import world_z_symmetry

    # a 4 x 2 x 1 box centred at the origin: 2-fold about z; a cube: 4-fold; a prism with an
    # equilateral cross-section lying on a side face: 2-fold about the vertical
    import itertools

    def box(a, b, c):
        V = np.array(list(itertools.product([-a, a], [-b, b], [-c, c])), dtype=float)
        F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        return V, F

    assert world_z_symmetry(*box(0.04, 0.02, 0.01)) == 2
    assert world_z_symmetry(*box(0.02, 0.02, 0.02)) == 4
    tri = np.array([[0.0, 0.025], [0.0216506, -0.0125], [-0.0216506, -0.0125]])  # equilateral, centroid at 0
    F = np.array([[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 3], [2, 3, 0]])
    standing = np.array([[t[0], t[1], z] for z in (0.0, 0.046) for t in tri])
    assert world_z_symmetry(standing, F) == 3
    iso = np.array([[t[0] * 1.4, t[1], z] for z in (0.0, 0.046) for t in tri])  # isosceles (RoboDojo block 3)
    assert world_z_symmetry(iso, F) == 1
    lying = np.array([[x, t[0], t[1] + 0.0125] for x in (-0.023, 0.023) for t in tri])  # apex up
    assert world_z_symmetry(lying, F) == 2
    # a T (bar + stem) has no rotational symmetry
    bar, stem = box(0.04, 0.01, 0.0075), box(0.01, 0.02, 0.0075)
    V = np.concatenate([bar[0] + [0, 0.02, 0], stem[0] + [0, -0.01, 0]])
    assert world_z_symmetry(V, np.concatenate([bar[1], stem[1] + 8])) == 1


def test_flange_for_camera_inverts_the_wrist_mount():
    try:
        model = X5Model.default()
    except FileNotFoundError:
        pytest.skip("X5A.urdf not available")
    from dojo.perception.cameras import pose_to_T
    from dojo.perception.look import flange_for_camera

    rng = np.random.default_rng(1)
    for arm in ("left", "right"):
        q = rng.uniform(-1, 1, 6)
        cam = cameras.wrist_camera(arm, q, model)
        T6 = model.fk_world(arm, q)["link6"]
        Tf = pose_to_T(*np.split(flange_for_camera(cam.T_wc), [3]))
        assert np.abs(Tf - T6).max() < 1e-9
