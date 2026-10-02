"""Observation renderers: projection, resize, overlays, derived images, determinism."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from controlr.config import ActionConfig, ObservationConfig
from controlr.observation.renderers import (
    ObservationRenderer,
    _project_segment,
    describe_axes,
    project_points,
    resize_long_edge,
)
from controlr.types import CameraInfo, Observation, RobotState

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def cam_identity() -> CameraInfo:
    """Camera frame == base frame (x right, y down, z forward)."""
    return CameraInfo("scene", 640, 480, K, np.eye(4))


def cam_top_down(height: float = 1.0, centre=(0.3, 0.0)) -> CameraInfo:
    """Looking straight down from above: image right = base +x, image up = base +y."""
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    eye = np.array([centre[0], centre[1], height])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ eye
    return CameraInfo("scene", 640, 480, K, T)


def make_obs(seed: int = 0, tcp=(0.3, 0.0, 0.2), grip=60.0, cam=None, shift: int = 0) -> Observation:
    rng = np.random.default_rng(seed)
    img = (rng.integers(0, 60, size=(480, 640, 3))).astype(np.uint8)
    img[200:260, 300 + shift:360 + shift] = (200, 40, 40)   # a "block"
    st = RobotState(t=0.0, q=np.zeros(6), tcp_pos=np.asarray(tcp, float),
                    tcp_rotvec=np.array([math.pi, 0, 0]), gripper_mm=grip, gripper_closed=False)
    return Observation(t=0.0, images={"scene": img}, cameras={"scene": cam or cam_top_down()}, state=st)


def fake_encoder(rgb, quality, label):
    """Stand-in for controlr.llm.transcript.encode_image (keeps these tests
    independent of the llm module)."""
    class Part:
        pass
    p = Part()
    p.sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    p.width, p.height, p.label, p.quality = rgb.shape[1], rgb.shape[0], label, quality
    return p


def test_projection_known_point_and_scaling():
    info = cam_identity()
    uv, ok = project_points(info, np.array([[0.1, 0.05, 1.0]]))
    assert ok[0] and uv[0] == pytest.approx((370.0, 265.0))
    uv, ok = project_points(info, np.array([[0.1, 0.05, 1.0]]), 320, 240)   # resized image
    assert uv[0] == pytest.approx((185.0, 132.5))
    uv, ok = project_points(info, np.array([[0.0, 0.0, -1.0], [0, 0, 0.0]]))
    assert not ok.any() and np.isnan(uv).all()
    # top-down camera: point under the camera projects to the principal point
    uv, ok = project_points(cam_top_down(), np.array([[0.3, 0.0, 0.0], [0.4, 0.1, 0.0]]))
    assert uv[0] == pytest.approx((320, 240)) and uv[1] == pytest.approx((370, 190))


def test_segment_clipping_behind_camera():
    info = cam_identity()
    seg = _project_segment(info, np.array([0.0, 0.0, -1.0]), np.array([0.1, 0.0, 1.0]), 640, 480)
    assert seg is not None and all(np.isfinite(c) for p in seg for c in p)
    assert seg[1] == pytest.approx((370.0, 240.0))
    assert _project_segment(info, np.array([0, 0, -1.0]), np.array([0, 0, -2.0]), 640, 480) is None


def test_describe_axes_top_down():
    s = describe_axes(cam_top_down(), np.array([0.3, 0.0, 0.0]))
    assert s == ("+x points right (50 px per 100 mm), +y points up (50 px per 100 mm), "
                 "+z points toward the camera in the 640-px image")
    s2 = describe_axes(cam_top_down(), np.array([0.3, 0.0, 0.0]), long_edge=320, per="10.0 cm")
    assert "(25 px per 10.0 cm)" in s2 and "320-px image" in s2


def test_resize_never_upscales():
    img = np.zeros((480, 640, 3), np.uint8)
    assert resize_long_edge(img, 448).shape == (336, 448, 3)
    assert resize_long_edge(img, 1000).shape == (480, 640, 3)


def test_raw_render_deterministic_and_sized():
    r = ObservationRenderer(ObservationConfig(size=320), encoder=fake_encoder)
    a1 = r.render_arrays(make_obs(), None, 1)
    a2 = r.render_arrays(make_obs(), None, 1)
    assert [l for l, _ in a1] == ["scene"] and a1[0][1].shape == (240, 320, 3)
    assert np.array_equal(a1[0][1], a2[0][1])
    out = r.render(make_obs(), None, 1)
    assert len(out.images) == 1 and out.images[0].label == "scene" and out.images[0].quality == 90
    assert out.text == "IMAGES: scene 320x240"


def test_first_turn_size():
    r = ObservationRenderer(ObservationConfig(size=224, first_turn_size=448), encoder=fake_encoder)
    assert r.render_arrays(make_obs(), None, 0)[0][1].shape[1] == 448
    assert r.render_arrays(make_obs(), None, 1)[0][1].shape[1] == 224


def test_overlays_draw_and_are_deterministic():
    cfg = ObservationConfig(size=640, renderers=["grid", "axes", "ee_marker"])
    r = ObservationRenderer(cfg, ActionConfig(), encoder=fake_encoder)
    raw = ObservationRenderer(ObservationConfig(size=640), encoder=fake_encoder)
    o = make_obs()
    a = r.render_arrays(o, None, 1)[0][1]
    assert not np.array_equal(a, raw.render_arrays(o, None, 1)[0][1])
    assert np.array_equal(a, r.render_arrays(make_obs(), None, 1)[0][1])
    # the TCP (0.3, 0, 0.2) seen from 1 m above projects to the principal point; the
    # marker's crosshair arm 7 px to the right is magenta
    px = a[240, 327].astype(int)
    assert px[0] > 200 and px[2] > 200 and px[1] < 80
    text = r.render(o, None, 1).text
    assert "grid every 50 mm" in text and "TCP (magenta)" in text


def test_overlays_tolerate_points_behind_camera():
    # camera frame == base frame: TCP at z=0.2 is in front, the grid plane z=0 lies
    # exactly at the camera -> must be clipped, not crash
    cfg = ObservationConfig(size=320, renderers=["grid", "axes", "ee_marker"])
    r = ObservationRenderer(cfg, encoder=fake_encoder)
    r.render_arrays(make_obs(cam=cam_identity(), tcp=(0.0, 0.0, -0.5)), None, 1)
    r.render_arrays(make_obs(cam=cam_identity()), None, 1)


def test_diff_and_heatmap_need_prev():
    cfg = ObservationConfig(size=320, renderers=["raw", "diff", "heatmap"])
    r = ObservationRenderer(cfg, encoder=fake_encoder)
    assert [l for l, _ in r.render_arrays(make_obs(), None, 0)] == ["scene"]
    same = r.render_arrays(make_obs(), make_obs(), 1)
    assert [l for l, _ in same] == ["scene", "scene diff", "scene heatmap"]
    assert same[1][1].max() == 0                                  # no change -> black diff
    assert np.array_equal(same[2][1], (same[0][1].astype(np.float32) * 0.8).astype(np.uint8))
    moved = r.render_arrays(make_obs(shift=40), make_obs(), 1)
    assert moved[1][1].max() > 200                                 # the block moved
    assert not np.array_equal(moved[2][1], same[2][1])
    text = r.render(make_obs(shift=40), make_obs(), 1).text
    assert "scene diff 320x240" in text and "scene heatmap" in text


def test_tile_combines_into_one_image():
    cfg = ObservationConfig(size=320, renderers=["raw", "diff"], tile=True)
    r = ObservationRenderer(cfg, encoder=fake_encoder)
    out = r.render_arrays(make_obs(shift=10), make_obs(), 1)
    assert len(out) == 1 and out[0][0] == "tile"
    h, w = out[0][1].shape[:2]
    assert w == 640 and h > 240
    assert np.array_equal(out[0][1], r.render_arrays(make_obs(shift=10), make_obs(), 1)[0][1])


def test_config_errors():
    with pytest.raises(ValueError):
        ObservationRenderer(ObservationConfig(renderers=["raw", "sparkles"]))
    r = ObservationRenderer(ObservationConfig(cameras=["wrist"]), encoder=fake_encoder)
    with pytest.raises(KeyError):
        r.render_arrays(make_obs(), None, 0)


def test_default_encoder_is_transcript_encode_image():
    pytest.importorskip("controlr.llm.transcript")
    r = ObservationRenderer(ObservationConfig(size=160))
    a = r.render(make_obs(), None, 1)
    b = r.render(make_obs(), None, 1)
    assert a.images[0].jpeg == b.images[0].jpeg and a.images[0].label == "scene"


def test_grid_z_defaults_to_table_and_jaw_is_tool_x():
    """Integration fixes: the grid plane follows RobotSpec.table_z unless set,
    and the fingertip markers lie along tool x (PHANTOM pads close along x)."""
    from controlr.config import ObservationConfig
    from controlr.observation.renderers import JAW_AXIS, resolve_grid_z
    from controlr.robot.spec import ur3_cb3_spec

    spec = ur3_cb3_spec()
    assert resolve_grid_z(ObservationConfig(), spec) == spec.table_z
    assert resolve_grid_z(ObservationConfig(grid_z=0.1), spec) == 0.1
    assert resolve_grid_z(ObservationConfig(), None) == 0.0
    assert JAW_AXIS == 0


def test_lookback_guard_warns_on_too_many_blocks_per_turn():
    """Review caching #12: > ~18 blocks per turn defeat Anthropic's 20-block lookback."""
    from controlr.config import ObservationConfig
    from controlr.observation.renderers import blocks_per_turn, lookback_warning

    many = ObservationConfig(cameras=[f"c{i}" for i in range(6)], renderers=["raw", "diff", "heatmap"])
    assert blocks_per_turn(many) == 21 and "tile" in lookback_warning(many)
    many.tile = True
    assert lookback_warning(many) is None
    assert lookback_warning(ObservationConfig()) is None
