"""Virtual Isaac cameras (numpy only): poses, intrinsics, how the base axes read in each view."""

from __future__ import annotations

import numpy as np
import pytest

from controlr.observation.renderers import axis_directions, project_points
from controlr.robot.isaac.cameras import intrinsics, virtual_camera
from controlr.robot.isaac.client import server_args_for
from controlr.types import CameraInfo

CENTRE = (-0.31, -0.02)
TABLE = 0.0


def _info(name: str) -> CameraInfo:
    vc = virtual_camera(name, CENTRE, TABLE)
    return CameraInfo(name=name, width=640, height=480, K=intrinsics(vc),
                      T_cam_base=np.linalg.inv(np.asarray(vc["world_from_cv"], float)))


def test_both_views_look_at_the_centre():
    for name, pt in (("top", [CENTRE[0], CENTRE[1], TABLE]), ("side", [CENTRE[0], CENTRE[1], TABLE + 0.12])):
        uv, ok = project_points(_info(name), np.array([pt]))
        assert ok[0] and uv[0] == pytest.approx([320, 240], abs=1e-6), name


def test_axes_read_unambiguously():
    top = axis_directions(_info("top"), np.array([CENTRE[0], CENTRE[1], 0.05]))
    assert top["x"] == "right" and top["y"] == "up" and top["z"] is None     # z along the ray
    side = axis_directions(_info("side"), np.array([CENTRE[0], CENTRE[1], 0.05]))
    assert side["y"] == "left" and side["z"] == "up"     # x (along the ray) drifts "up" by perspective
    from controlr.llm.decisions import axis_views
    v = axis_views({"top": _info("top"), "side": _info("side")}, ["top", "side"],
                   np.array([CENTRE[0], CENTRE[1], 0.05]))
    assert v["x"]["image"] == "top" and v["z"]["image"] == "side"   # side never chosen for x


def test_rotations_are_proper_and_config_is_phantom_shaped():
    for name in ("top", "side"):
        vc = virtual_camera(name, CENTRE, TABLE, (448, 336))
        R = np.asarray(vc["world_from_cv"])[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
        assert vc["resolution"] == [448, 336] and vc["fx"] == vc["fy"] == pytest.approx(0.9 * 448)
    with pytest.raises(ValueError):
        virtual_camera("wrist", CENTRE, TABLE)


def test_server_args_carry_the_extra_cameras():
    assert "--extra-cameras" not in server_args_for({})
    args = server_args_for({"extra_cameras": ["top", "side"]})
    assert args[args.index("--extra-cameras") + 1] == "top,side"
