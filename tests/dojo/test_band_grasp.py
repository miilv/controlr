"""pick_from_conveyor_by_image: a basket's narrow top (its handle arch) is grasped there (``top_band_grasps``), not on
a rim circle under the bbox top (which closed on air, 10 / 10 in G3 on v13)."""
import math

import numpy as np


def _basket(yaw_deg=90.0):
    from dojo.skills.world import AssetInfo, ObjectState
    xs, ys = np.meshgrid(np.linspace(-0.18, 0.18, 19), np.linspace(-0.12, 0.12, 13))
    body = np.c_[xs.ravel(), ys.ravel(), np.zeros(xs.size)]
    V = np.vstack([body + [0, 0, z] for z in (-0.11, -0.05, 0.03)])            # the body: 36 x 24 cm, rim at 3 cm
    hy = np.linspace(-0.075, 0.075, 31)
    handle = np.vstack([np.c_[np.full_like(hy, x), hy, np.full_like(hy, z)]
                        for x in (-0.006, 0.006) for z in (0.095, 0.11)])      # a 12 mm arch along y, 11 cm up
    V = np.vstack([V, handle])
    a = AssetInfo(key="Rigid/basket_grasp/00002", bbox_min=V.min(0), bbox_max=V.max(0), mass=0.2, mesh=(V, np.zeros((0, 3), int)))
    h = math.radians(yaw_deg) / 2
    return ObjectState(label="basket", obj_type="Rigid", category="basket_grasp", idx=2,
                       pos=np.array([-0.2, 0.03, 0.874]), quat=np.array([math.cos(h), 0, 0, math.sin(h)]), asset=a)


def test_band_grasp_takes_the_handle():
    from dojo.skills import grasps as G
    from dojo.skills.kinematics import X5Kin
    o = _basket()
    gs = G.top_band_grasps(o, X5Kin(), "left")
    assert gs and all(g.method == "band" for g in gs)
    g = min(gs, key=lambda g: g.score)
    p = g.T[:3, 3]
    assert np.linalg.norm(p[:2] - o.pos[:2]) < 0.005                  # over the handle's middle
    assert abs(p[2] - (o.pos[2] + 0.11 - 0.012 + G.TIP_BEYOND_TCP)) < 1e-6   # just under the handle's top
    closing = g.T[:3, 1]
    assert abs(float(closing @ (o.R @ np.array([1.0, 0, 0])))) > 0.99  # across the 12 mm bar (object x)
    assert g.width < 0.02 and g.T[2, 0] < -0.8                        # narrow, from above


def test_band_grasp_skips_full_width_tops_and_upside_down():
    from dojo.skills import grasps as G
    from dojo.skills.kinematics import X5Kin
    from dojo.skills.world import AssetInfo, ObjectState
    xs, ys = np.meshgrid(np.linspace(-0.03, 0.03, 7), np.linspace(-0.03, 0.03, 7))
    cube = np.vstack([np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, z)] for z in (-0.03, 0.03)])
    a = AssetInfo(key="Rigid/cube/00000", bbox_min=cube.min(0), bbox_max=cube.max(0), mesh=(cube, np.zeros((0, 3), int)))
    o = ObjectState(label="c", obj_type="Rigid", category="cube", idx=0, pos=np.zeros(3), quat=np.array([1.0, 0, 0, 0]),
                    asset=a)
    assert G.top_band_grasps(o, X5Kin(), "left") == []                # its top is its whole footprint
    b = _basket()
    b.quat = np.array([0.0, 1.0, 0.0, 0.0])                           # upside down
    assert G.top_band_grasps(b, X5Kin(), "left") == []


def _walled_basket():
    from dojo.skills.world import AssetInfo, ObjectState
    pts = []
    for z in np.linspace(-0.11, 0.02, 14):                      # 2 cm walls, rim at 2 cm
        for x in np.linspace(-0.18, 0.18, 37):
            for y in (-0.12, -0.11, 0.11, 0.12):
                pts.append((x, y, z))
        for y in np.linspace(-0.12, 0.12, 25):
            for x in (-0.18, -0.165, 0.165, 0.18):
                pts.append((x, y, z))
    hy = np.linspace(-0.075, 0.075, 31)
    pts += [(x, y, z) for x in (-0.006, 0.006) for z in (0.095, 0.11) for y in hy]   # the handle arch above
    V = np.asarray(pts, float)
    a = AssetInfo(key="Rigid/basket_grasp/00002", bbox_min=V.min(0), bbox_max=V.max(0), mesh=(V, np.zeros((0, 3), int)))
    h = math.radians(90.0) / 2
    return ObjectState(label="basket", obj_type="Rigid", category="basket_grasp", idx=2,
                       pos=np.array([-0.2, 0.03, 0.874]), quat=np.array([math.cos(h), 0, 0, math.sin(h)]), asset=a)


def test_box_rim_grasp_takes_the_near_wall_at_the_real_rim():
    from dojo.skills import grasps as G
    from dojo.skills.kinematics import X5Kin
    o = _walled_basket()
    gs = sorted(G.box_rim_grasps(o, X5Kin(), "left"), key=lambda g: g.score)
    assert gs
    g = gs[0]
    loc = o.R.T @ (g.T[:3, 3] - o.pos)
    assert abs(loc[2] - (0.02 - 0.012 + G.TIP_BEYOND_TCP)) < 0.003       # under the rim, not under the handle's top
    assert abs(abs(loc[0]) - 0.1725) < 0.005 and loc[0] < 0               # on the wall nearer the arm (object -x)
    assert 0.01 < g.width < 0.02
