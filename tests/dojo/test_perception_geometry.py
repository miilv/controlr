"""dojo.perception: camera model, FK, silhouettes, task candidates, metrics. No GPU, no assets.

The perception extra (scipy, opencv) is needed; without it the module is skipped (CI installs only dev).
"""
import json
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


def test_colour_order_from_instruction():
    from dojo.perception.language import colour_order

    s = "stack the blocks from bottom to top in the order of cyan, red, and orange, then reset the robot arm."
    assert colour_order(s) == ["cyan", "red", "orange"]


def test_region_seeds_only_for_compact_fixed_yaw_regions():
    from dojo.perception.inverse_layout import _region, _seed_region, region_ok

    prior = {"fixed_xy": None}
    # organize_table's drawer: x range, y pinned, no random yaw -> a 5-point seed line
    drawer = {"regions": [_region({"xlim": [-0.45, -0.25], "ylim": [0.15, 0.15]})]}
    xs, ys, yaw0 = _seed_region(drawer, prior)
    assert len(xs) == 5 and np.allclose(ys, [0.15]) and yaw0 == 0.0
    assert region_ok(drawer, -0.36, 0.153) and not region_ok(drawer, -0.36, 0.099 - 0.05)
    # a wide region (the whole table half) or a free yaw gets no seeds
    wide = {"regions": [_region({"xlim": [-0.45, 0.0], "ylim": [-0.3, 0.05]})]}
    assert _seed_region(wide, prior) is None
    spun = {"regions": [_region({"xlim": [-0.05, 0.05], "ylim": [-0.2, -0.15], "rotate_rand": True, "rotate_deg": 45})]}
    assert _seed_region(spun, prior) is None


def test_j1_adapter_task_inference_and_wire_format():
    from dojo.perception import j1_adapter as j

    assert j.infer_task("Press the two red buttons the required number of times according to the number cards, "
                        "then press the blue button to confirm.") == "press_by_number"
    assert j.infer_task("Push the T-shaped block to align it precisely with the gray T-shaped pad.") == "push_T"
    assert j.infer_task("Pick up the two T-shaped blocks, swap their positions, and place them back with the "
                        "correct orientations.") == "swap_T"
    assert j.infer_task("Stack the three blocks with different textures.") == "stack_blocks"
    assert j.infer_task("Pick up the bell by 10 cm.") == "general_pickup"
    a = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    back = j._dec(j._enc({"img": a, "x": [np.float32(1.5), "s"]}))
    assert back["img"].dtype == np.uint8 and (back["img"] == a).all() and back["x"] == [1.5, "s"]


def test_target_phrase_from_general_pickup_instruction():
    from dojo.perception.inverse_layout import _tokens, target_phrase

    assert target_phrase("Pick up the yellow ice axe by 10 cm.") == "yellow ice axe"
    assert target_phrase("Stack the three blocks.") is None
    assert _tokens("Pick up the carved wooden elephant by 10 cm") == {"carved", "wooden", "elephant"}


def test_j1_labels_follow_the_prior_by_position():
    from dojo.perception.j1_adapter import _carry_static, stabilize_labels  # noqa: F401

    prior = [{"label": "block_0", "pos": [-0.14, -0.03, 0.78]}, {"label": "block_1", "pos": [0.26, -0.14, 0.78]},
             {"label": "block_2", "pos": [0.37, -0.15, 0.78]},
             {"label": "target_t", "asset": "Geometry/t_cushion/00000", "pos": [-0.31, -0.11, 0.765], "confidence": 0.9}]
    # a later call: same blocks, labels permuted by the estimator, one block moved 3 cm
    cur = [{"label": "block_0", "pos": [0.37, -0.15, 0.78]}, {"label": "block_1", "pos": [-0.14, -0.03, 0.78]},
           {"label": "block_2", "pos": [0.29, -0.13, 0.78]}]
    changed = stabilize_labels(cur, prior, label_sets=[["block_0", "block_1", "block_2"]])
    assert [o["label"] for o in cur] == ["block_2", "block_0", "block_1"]
    assert changed == {"block_0": "block_2", "block_1": "block_0", "block_2": "block_1"}
    out = _carry_static(cur, prior)  # the pad hidden under the T keeps its prior pose
    assert out[-1]["label"] == "target_t" and out[-1]["source"] == "prior"


def test_j1_refine_result_goes_to_the_object_at_that_place():
    from dojo.perception.j1_adapter import match_refined

    a = {"label": "block_0", "category": "block", "pos": [0.26, -0.14, 0.78]}
    b = {"label": "block_1", "category": "block", "pos": [-0.14, -0.03, 0.78]}
    prior_a = {"label": "block_1", "category": "block", "pos": [0.261, -0.143, 0.78]}  # prior label differs
    o, why = match_refined([a, b], prior_a, [0.2605, -0.1425, 0.78])
    assert o is a and why is None
    c = {"label": "block_2", "category": "block", "pos": [0.27, -0.14, 0.78]}
    o, why = match_refined([a, c], prior_a, [0.268, -0.140, 0.78])  # lands on block_2 -> rejected
    assert o is None and why == "lands on another object"


def test_support_under_an_object_is_not_its_occluder():
    from types import SimpleNamespace

    from dojo.perception import inverse_layout as il

    def shape(key, ext):
        V = np.array([[x, y, z] for x in (-ext[0] / 2, ext[0] / 2) for y in (-ext[1] / 2, ext[1] / 2) for z in (0.0, ext[2])])
        a = SimpleNamespace(V=V, F=np.zeros((0, 3), int), extents=np.asarray(ext), info={})
        sm = il.ShapeModel(key, "test", a, np.eye(3), 0.0, ext[2], 1, True, None, centre_local=np.zeros(3))
        il._FIXED_POSE_SHAPES[(key, "test")] = sm
        return sm

    shape("Geometry/pad/t", (0.084, 0.068, 0.0001))
    shape("Rigid/t/t", (0.08, 0.06, 0.015))
    pad = il.Hypothesis(0, "Geometry/pad/t", "test", -0.31, -0.11, 0.0, 0.1)
    t_on_pad = il.Hypothesis(1, "Rigid/t/t", "test", -0.309, -0.111, 0.0, 0.1, {"z_off": 0.0001})
    assert il._supports(pad, t_on_pad) and not il._supports(t_on_pad, pad)
    far = il.Hypothesis(1, "Rigid/t/t", "test", 0.2, -0.111, 0.0, 0.1)
    assert not il._supports(pad, far)


def test_j1_merge_prior_rules():
    from dojo.perception.j1_adapter import merge_prior

    prior = [
        {"label": "target_t", "category": "t_cushion", "asset": "Geometry/t_cushion/00000", "pos": [-0.107, -0.224, 0.765],
         "quat": [1, 0, 0, 0], "confidence": 0.71},
        {"label": "num0", "category": "number_card", "asset": "Geometry/number_card/00005", "pos": [-0.15, -0.05, 0.805],
         "quat": [0, 0, 0.42, -0.91], "confidence": 0.86, "attributes": {"digit": 5},
         "alternative": {"asset": "Geometry/number_card/00006", "margin": 0.053}},
        {"label": "block_1", "category": "block", "asset": "Rigid/block/00006", "pos": [0.216, -0.078, 0.78],
         "quat": [1, 0, 0, 0], "confidence": 0.72},
        {"label": "block_2", "category": "block", "asset": "Rigid/block/00009", "pos": [0.319, -0.058, 0.78],
         "quat": [1, 0, 0, 0], "confidence": 0.95},
    ]
    cur = [
        # J1 push L3 call 2: the pad 54 mm off at 0.20 after the look
        {"label": "target_t", "category": "t_cushion", "asset": "Geometry/t_cushion/00000", "pos": [-0.138, -0.268, 0.765],
         "quat": [1, 0, 0, 0], "confidence": 0.20},
        # press L4 call 2: the card read 1 at margin 0.003
        {"label": "num0", "category": "number_card", "asset": "Geometry/number_card/00001", "pos": [-0.149, -0.056, 0.805],
         "quat": [0, 0, 0.42, -0.91], "confidence": 0.60, "attributes": {"digit": 1},
         "alternative": {"asset": "Geometry/number_card/00004", "margin": 0.003}},
        # the base re-read under a hovering arm: degraded, 19 mm off
        {"label": "block_1", "category": "block", "asset": "Rigid/block/00007", "pos": [0.219, -0.059, 0.78],
         "quat": [1, 0, 0, 0], "confidence": 0.20},
    ]
    out, last_seen = merge_prior(cur, prior)
    by = {o["label"]: o for o in out}
    assert by["target_t"]["pos"] == prior[0]["pos"] and by["target_t"]["source"] == "prior"
    assert by["num0"]["attributes"]["digit"] == 5
    assert by["block_1"]["pos"] == prior[2]["pos"] and by["block_1"]["source"] == "prior"
    assert [x["label"] for x in last_seen] == ["block_2"]  # picked or moved: listed, not carried
    assert "block_2" not in by


def test_tower_shape_stacks_copies_straight_up():
    from types import SimpleNamespace

    from dojo.perception import inverse_layout as il

    h = 0.035
    V = np.array([[x, y, z] for x in (-h / 2, h / 2) for y in (-h / 2, h / 2) for z in (-h / 2, h / 2)])
    F = np.array([[0, 1, 2]])
    a = SimpleNamespace(V=V, F=F, Vg=V, Fg=F, info={}, dir=None, extents=np.full(3, h))
    sm = il.ShapeModel("Rigid/block/test", "up", a, np.eye(3), -h / 2, h, 4, True, None, centre_local=np.zeros(3))
    t = il.tower_shape(sm, 3)
    T = t.pose(0.1, -0.1, 0.0)
    z = (t.asset.V @ T[:3, :3].T + T[:3, 3])[:, 2]
    assert abs(z.min() - il.TABLE_Z) < 1e-9 and abs(z.max() - (il.TABLE_Z + 3 * h)) < 1e-9
    assert len(t.asset.Vg) == 3 * len(V) and t.height == 3 * h


def _conveyor_labels():
    cands = ["Rigid/bottle/00047", "Rigid/puppet/00000"]
    cats = [{"name": "bottle", "index": [47]}, {"name": "puppet", "index": [0]}]
    rp = "conveyor/conveyor/0"
    slots = [[-0.3, -0.3], [-0.14, -0.14], [0.02, 0.02], [0.2, 0.2]]
    labels = {"target0": {"type": "Rigid", "candidates": cands, "categories": cats,
                          "select_mode": {"nums": 1, "mode": "same_as_label", "same_label": "target1", "label": ["target0"]},
                          "placement": {"xlim": [1.88, 1.88], "ylim": [-0.13, -0.13], "rotate_deg": 35, "rotate_rand": True,
                                        "relative_plane": rp}}}
    for lab in ("target1", "other0", "other1"):
        labels[lab] = {"type": "Rigid", "candidates": cands, "categories": cats,
                       "select_mode": {"nums": 1, "mode": "unique", "label": [lab]},
                       "placement": {"xlim": slots, "ylim": [-0.3, -0.2], "rotate_deg": 35, "rotate_rand": True,
                                     "relative_plane": rp}}
    labels["conveyor"] = {"type": "Dynamic", "candidates": ["Dynamic/conveyor/00000"], "categories": [{"name": "conveyor"}],
                          "select_mode": {"nums": 1, "mode": "unique", "label": ["conveyor"]},
                          "placement": {"xlim": [-2.08, -2.08], "ylim": [0.175, 0.175], "zlim": [0.0, 0.0], "rotate_deg": 0,
                                        "rotate_rand": True, "relative_plane": "Ground", "qpos": [0, 0, 0, 1]}}
    # the public catalogue entry of the belt (info.json): its support frame and bbox
    info = {"support": {"conveyor/0": {"center": [[-0.445, 0.003, 0.718, 0.0, 0.0, 0.0, 1.0]], "radius": [0.44]}},
            "bbox_min": [-3.962948, -0.601345, 0.0], "bbox_max": [1e-06, 0.575493, 0.92438]}
    return labels, info


def test_conveyor_regions_are_mapped_through_the_belt_and_span_it(monkeypatch):
    from dojo.perception import inverse_layout as IL

    labels, info = _conveyor_labels()
    monkeypatch.setattr(IL, "task_map", lambda: {"belt": {"labels": labels}})
    monkeypatch.setattr(IL, "asset_info", lambda key: info if key.startswith("Dynamic/conveyor") else {})
    groups = IL.label_groups("belt")
    g = next(g for g in groups if "target0" in g["labels"])
    assert set(g["labels"]) == {"target0", "target1", "other0", "other1"}
    # frame 0 of our dev layouts: target0 rides at x 0.23-0.31, y ~0.04; the others spawn at x -1.94..-1.43
    for x, y in ((0.245, 0.042), (0.6, 0.03), (-0.5, -0.08), (-1.935, -0.124)):
        assert IL.region_ok(g, x, y, yaw=2.5)                       # anywhere along the belt, any yaw
    assert not IL.region_ok(g, 0.2, -0.35) and not IL.region_ok(g, 0.2, 0.2)   # off the belt's band
    assert not IL.region_ok(g, 2.3, 0.0)                           # past the belt's end
    prior = IL._placement_prior(g)
    assert prior["fixed_xy"] is None                               # target0's 1.88 is not a world pin
    assert IL._seed_region(g, prior) is None and IL.yaw_range(g, 2 * np.pi) is None
    conv = next(g for g in groups if "conveyor" in g["labels"])
    assert IL._placement_prior(conv)["fixed_xy"] == (-2.08, 0.175)  # the ground fixture keeps its config pose


def test_regions_relative_to_a_pinned_or_an_estimated_base(monkeypatch):
    from dojo.perception import inverse_layout as IL

    labels = {
        "tile": {"type": "Rigid", "candidates": ["Rigid/mahjong/00001"], "categories": [{"name": "mahjong"}],
                 "select_mode": {"mode": "unique", "label": ["tile"]},
                 "placement": {"xlim": [-0.4, -0.4], "ylim": [-0.002, -0.002], "relative_plane": "Table"}},
        "top": {"type": "Rigid", "candidates": ["Rigid/mahjong/00002"], "categories": [{"name": "mahjong"}],
                "select_mode": {"mode": "unique", "label": ["top"]},
                "placement": {"xlim": [0.0, 0.0], "ylim": [0.0, 0.0], "relative_plane": "tile/mahjong_0"}},
        "basket": {"type": "Rigid", "candidates": ["Rigid/egg_basket/00000"], "categories": [{"name": "egg_basket"}],
                   "select_mode": {"mode": "unique", "label": ["basket"]},
                   "placement": {"xlim": [-0.4, 0.4], "ylim": [-0.2, -0.1], "rotate_rand": True, "rotate_deg": 20,
                                 "relative_plane": "Table"}},
        "egg": {"type": "Rigid", "candidates": ["Rigid/egg/00000"], "categories": [{"name": "egg"}],
                "select_mode": {"mode": "unique", "label": ["egg"]},
                "placement": {"xlim": [-0.1, 0.1], "ylim": [-0.1, 0.1], "relative_plane": "basket/egg_basket/0"}},
        "table_obj": {"type": "Rigid", "candidates": ["Rigid/cup/00000"], "categories": [{"name": "cup"}],
                      "select_mode": {"mode": "unique", "label": ["table_obj"]},
                      "placement": {"xlim": [-0.3, 0.3], "ylim": [-0.2, 0.0], "rotate_rand": True, "rotate_deg": 180}},
    }
    monkeypatch.setattr(IL, "task_map", lambda: {"t": {"labels": labels}})
    monkeypatch.setattr(IL, "asset_info", lambda key: {})
    gs = {g["labels"][0]: g for g in IL.label_groups("t")}
    # a pinned zero offset on a pinned base: the base's point (it used to be world (0, 0))
    assert IL._placement_prior(gs["top"])["fixed_xy"] == (-0.4, -0.002)
    assert IL.region_ok(gs["top"], -0.4, 0.0) and not IL.region_ok(gs["top"], 0.0, 0.0)
    # an estimated base: anywhere the basket can be, widened by the offsets; yaw free
    assert IL.region_ok(gs["egg"], 0.45, -0.25, yaw=3.0) and IL.region_ok(gs["egg"], -0.5, 0.0)
    assert not IL.region_ok(gs["egg"], 0.0, 0.2)
    # table placements are read exactly as before
    assert gs["table_obj"]["regions"] == [IL._region(labels["table_obj"]["placement"])]


def test_scene_background_replaces_the_table_for_its_fixture(tmp_path):
    import cv2

    from dojo.perception.segment import load_backgrounds, scene_dir_name, with_scene

    fold = tmp_path / "fold0"
    fold.mkdir()
    for c, v in (("cam_head", 10), ("cam_left_wrist", 20), ("cam_right_wrist", 30), ("cam_head_clean", 40)):
        cv2.imwrite(str(fold / f"{c}.png"), np.full((4, 6, 3), v, np.uint8))
    cv2.imwrite(str(fold / "cam_head_clean_unseen.png"), np.zeros((4, 6), np.uint8))
    sc = fold / "scene" / scene_dir_name("Dynamic/conveyor/00000")
    sc.mkdir(parents=True)
    cv2.imwrite(str(sc / "cam_head.png"), np.full((4, 6, 3), 99, np.uint8))
    base = load_backgrounds(fold)
    assert with_scene(base, fold, ["Geometry/camera_stand/00000"]) is base          # no scene background: the table
    b = with_scene(base, fold, ["Geometry/camera_stand/00000", "Dynamic/conveyor/00000"])
    assert int(b["cam_head"][0, 0, 0]) == 99
    assert "cam_left_wrist" not in b and "cam_head_clean" not in b                     # no table image survives
def _box_shape(il, key, ext):
    from types import SimpleNamespace

    V = np.array([[x, y, z] for x in (-ext[0] / 2, ext[0] / 2) for y in (-ext[1] / 2, ext[1] / 2) for z in (0.0, ext[2])])
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    a = SimpleNamespace(V=V, F=F, Vg=V, Fg=F, Vs=V, Fs=F, info={}, dir=None, extents=np.asarray(ext))
    sm = il.ShapeModel(key, "test", a, np.eye(3), 0.0, ext[2], 4, True, V, centre_local=np.zeros(3))
    il._FIXED_POSE_SHAPES[(key, "test")] = sm
    return sm


def test_merged_blob_of_several_pieces_adds_pieces_and_a_single_piece_does_not():
    from dojo.perception import inverse_layout as il
    from dojo.perception.segment import ViewMasks

    sm = _box_shape(il, "Rigid/block/split_test", (0.038, 0.038, 0.062))
    cam = cameras.head_camera()
    fg = np.zeros((cam.h, cam.w), np.uint8)
    for x, y in ((-0.30, -0.10), (-0.255, -0.10), (-0.30, -0.055), (-0.255, -0.055), (0.25, -0.10)):  # 4 touching + 1
        sm.sil(cam, sm.pose(x, y, 0.0), out=fg)
    v = ViewMasks(cam, np.zeros((cam.h, cam.w, 3), np.uint8), fg, np.ones_like(fg), np.zeros(fg.shape, np.float32), np.zeros_like(fg))
    bl = il.blobs(v, "cam_head")
    assert len(bl) == 2
    pending = [(0, {"labels": ["a", "b", "c", "d", "e"]}, {})]
    out = il._split_merged(bl, pending, {0: {"g": [sm]}}, {"cam_head": v})
    merged = max(bl, key=lambda b: b.area)
    assert out[:2] == bl and len(out) >= 4                # both blobs stay; the merged cluster adds >= 2 pieces
    assert sum(b.area for b in out[2:]) == merged.area    # the pieces partition it; the lone block adds none
    assert all(b.area < 0.6 * merged.area for b in out[2:])



def test_polish_paired_texture_cannot_override_a_clearly_worse_colour():
    from dojo.perception import inverse_layout as il

    cfg = il.EstimatorConfig()
    white = il.Hypothesis(0, "Rigid/mug/00004", "up", 0.0, 0.0, 0.0, 0.2, {"sil_loss": 0.05})
    white.appearance = 0.10
    dark = il.Hypothesis(0, "Rigid/mug/00002", "up", 0.0, 0.0, 0.0, 0.3, {"sil_loss": 0.06})
    dark.appearance = 0.46
    pt = (0.47, 0.30)                                  # paired texture prefers the dark mug by 0.17
    assert not il._polish_prefers(white, dark, pt, cfg)                    # the colour vetoes it
    import dataclasses
    assert il._polish_prefers(white, dark, pt, dataclasses.replace(cfg, polish_color_veto=None))   # as before
    close = il.Hypothesis(0, "Rigid/mug/00003", "up", 0.0, 0.0, 0.0, 0.25, {"sil_loss": 0.06})
    close.appearance = 0.13                            # colour-compatible: paired texture decides
    assert il._polish_prefers(white, close, pt, cfg)
    assert not il._polish_prefers(white, close, None, cfg)                 # no texture: the combined cost (0.25 > 0.2)


def test_same_mode_binds_one_select_list_not_the_merged_group():
    from types import SimpleNamespace

    from dojo.perception import inverse_layout as il

    geom = {i: [SimpleNamespace(key=f"Rigid/stackingblocks/0000{i}")] for i in range(4)}
    sel = {"mode": "same", "nums": 4, "label": ["block0", "block1", "block2", "block3"]}
    # play_stacking_toy: 10 pieces with identical candidates merge into one group of 4 select lists (4/3/2/1)
    wide = {"select_mode": sel, "labels": [f"block{i}" for i in range(10)]}
    assert il._options(wide, geom) is None
    one_list = {"select_mode": sel, "labels": ["block0", "block1", "block2", "block3"]}
    assert il._options(one_list, geom) == [{f"Rigid/stackingblocks/0000{i}"} for i in range(4)]


def test_task_map_skips_candidates_without_geometry(tmp_path, monkeypatch):
    from dojo.perception import inverse_layout as il

    (tmp_path / "index.json").write_text("{}")
    tasks = {"sweep": {"labels": {
        "broom_shovel": {"type": "Rigid", "candidates": ["Rigid/bs/00000", "Rigid/bs/00001"]},
        "ghost": {"type": "Rigid", "candidates": ["Rigid/ghost/00000"]}}}}
    (tmp_path / "tasks.json").write_text(json.dumps(tasks))
    d = tmp_path / "Rigid" / "bs" / "00000"
    d.mkdir(parents=True)
    np.savez(d / "geom.npz", vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
    monkeypatch.setenv("DOJO_CATALOG", str(tmp_path))
    il.task_map.cache_clear()
    try:
        with pytest.warns(UserWarning, match="no geometry"):
            tm = il.task_map()
        lab = tm["sweep"]["labels"]
        assert lab["broom_shovel"]["candidates"] == ["Rigid/bs/00000"]          # the unshipped one is skipped
        assert lab["broom_shovel"]["candidates_missing"] == ["Rigid/bs/00001"]
        assert "ghost" not in lab                                              # nothing left to fit
    finally:
        il.task_map.cache_clear()


def test_table_generator_turns_the_asset_about_its_origin():
    """RoboDojo's table generator puts the place centre at the sampled point, then turns the orientation about the
    asset ORIGIN: an off-centre asset (a phone, origin at one end) lies with its centre anywhere on a circle around
    the origin, and the constrained point is origin + R0 @ centre whatever the yaw. On another object (gen_layouts'
    place_on_object) the turn is about the placement point: the yawed centre, as before."""
    from types import SimpleNamespace

    from dojo.perception import inverse_layout as il

    V = np.array([[x, y, z] for x in (-0.024, 0.024) for y in (0.0, 0.115) for z in (0.0, 0.012)])
    a = SimpleNamespace(V=V, F=np.zeros((0, 3), int), extents=np.ptp(V, axis=0), info={})
    sm = il.ShapeModel("Rigid/phone/t", "t", a, np.eye(3), 0.0, 0.012, 1, True, None,
                       centre_local=np.array([0.0, 0.0575, 0.006]))
    il._FIXED_POSE_SHAPES[("Rigid/phone/t", "t")] = sm
    table = {"placement": {"relative_plane": "Table"}, "regions": [il._region({"xlim": [-0.3, 0.4], "ylim": [-0.2, -0.05],
                                                                               "rotate_rand": True, "rotate_deg": 180})]}
    on_obj = {"placement": {"relative_plane": "stand/stand/0"}, "regions": table["regions"]}
    # the generator sampled (0.1, -0.19): origin = point - R0 @ centre; then a half turn about the origin
    h = il.Hypothesis(0, "Rigid/phone/t", "t", 0.1, -0.19 - 0.0575, np.pi, 0.1)
    assert np.allclose(il._centre_xy(h, table), (0.1, -0.19), atol=1e-6)
    assert np.allclose(il._centre_xy(h), (0.1, -0.19 - 0.115), atol=1e-6)   # the yawed centre: 11.5 cm off
    assert il.region_ok(table, *il._centre_xy(h, table)) and not il.region_ok(table, *il._centre_xy(h))
    assert np.allclose(il._centre_xy(h, on_obj), il._centre_xy(h))          # on an object: unchanged
    assert il._blob_margin(table, sm) == pytest.approx(0.115) and il._blob_margin(on_obj, sm) == 0.0
    old = il.GEN_YAW_ABOUT_ORIGIN
    try:
        il.GEN_YAW_ABOUT_ORIGIN = False
        assert np.allclose(il._centre_xy(h, table), il._centre_xy(h))
    finally:
        il.GEN_YAW_ABOUT_ORIGIN = old
