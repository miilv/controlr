"""regrasp: the rest handover's lengthwise giver grasps and the taker's grasps chosen at the target (CPU only)."""

from __future__ import annotations

import math

import numpy as np

from tests.dojo.test_skills2 import _box_mesh, _merge, _obj, _world


def _key_mesh():
    """A key in its own frame like RoboDojo's: bow (20 x 10 x 20 mm) around the origin, shaft (5 x 5 mm) down to
    z -52.8 mm, a bit along +x near the tip."""
    return _merge(_box_mesh((-0.010, -0.005, -0.010), (0.010, 0.005, 0.010)),
                  _box_mesh((-0.0025, -0.0025, -0.0528), (0.0025, 0.0025, -0.010)),
                  _box_mesh((0.0025, -0.0023, -0.0491), (0.0127, 0.0023, -0.0345)))


class _SK:
    def __init__(self, objs):
        self.world = _world(objs)


def _lying_key(yaw: float = 0.3):
    # lying on its face: +y_key up (a turn of +90 deg about x), then turned about the vertical
    qx = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qx
    q = (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    return _obj("key", "key", 900, (0.16, -0.2, 0.770), (0.0114, 0.005, 0.0314), quat=q, mesh=_key_mesh())


def test_lengthwise_grasps_pinch_the_key_end_to_end_straight_down():
    from dojo.skills.kinematics import TIP_BEYOND_TCP
    from dojo.skills.regrasp import LENGTHWISE_TIP, lengthwise_grasps
    k = _lying_key()
    gs = lengthwise_grasps(_SK([k]), "key", "right")
    assert len(gs) == 2 * len(LENGTHWISE_TIP)
    z_axis = k.R[:, 2]                                       # the key's long axis, horizontal when it lies
    V = k.asset.mesh[0] @ k.R.T + k.pos
    for g in gs:
        assert np.allclose(g.T[:3, 0], [0, 0, -1])            # straight down
        assert abs(abs(float(g.T[:3, 1] @ z_axis)) - 1.0) < 1e-6   # closing along the key's length
        assert abs(g.width - 0.0628) < 1e-4
        tip_z = g.T[2, 3] - TIP_BEYOND_TCP
        assert any(abs(tip_z - (V[:, 2].min() + dz)) < 1e-6 for dz in LENGTHWISE_TIP)
    # centred between the two ends along the length
    s = V @ z_axis
    assert abs(float(gs[0].T[:3, 3] @ z_axis) - (s.min() + s.max()) / 2) < 1e-6


def test_lengthwise_grasps_skip_objects_longer_than_the_gripper_opens():
    from dojo.skills.regrasp import lengthwise_grasps
    long = _obj("bar", "bar", 901, (0.1, -0.2, 0.775), (0.06, 0.01, 0.01),
                mesh=_box_mesh((-0.06, -0.01, -0.01), (0.06, 0.01, 0.01)))
    assert lengthwise_grasps(_SK([long]), "bar", "left") == []


def test_top_grasps_at_the_target_close_across_the_bow_and_point_down():
    from dojo.skills.geometry import T_from, rot_z
    from dojo.skills.kinematics import TIP_BEYOND_TCP
    from dojo.skills.regrasp import TILTS, top_grasps_at
    k = _lying_key()
    T_tg = T_from(np.array([-0.14, -0.155, 0.843]), rot_z(0.4))      # standing in its lock
    out = top_grasps_at(k, T_tg)
    assert len(out) == 2 * len(TILTS) * 2 * 2
    for note, T_ko, width in out:
        assert abs(width - 0.010) < 1e-6                     # the bow's thickness, not its 20 mm width
        assert abs(abs(T_ko[1, 1]) - 1.0) < 1e-9             # closing along y_key
        Tw = T_tg @ T_ko
        a = Tw[:3, 0]
        assert a[2] < -0.2                                   # pointing down at the target (tilted >= 15 deg)
        tip = T_ko[:3, 3] + T_ko[:3, 0] * TIP_BEYOND_TCP
        assert abs(tip[1]) < 1e-9 and -0.003 < tip[2] < 0.0 and abs(tip[0]) <= 0.0041   # at the bow's centre
