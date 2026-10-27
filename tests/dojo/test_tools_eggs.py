"""Tools-articulated stream: eggs into an egg holder without Isaac (lying drop targets, dense finger samples)."""

import math

import numpy as np

from dojo.skills import holders as HO
from dojo.skills.kinematics import X5Kin
from dojo.skills.world import AssetInfo, ObjectState


def _egg(pos=(0.0, -0.2, 0.786), quat=(0.7071, 0.7071, 0.0, 0.0)):
    info = AssetInfo(key="Rigid/egg/00000", bbox_min=np.array([-0.0206, -0.0211, -0.03]),
                     bbox_max=np.array([0.0206, 0.0211, 0.0276]))
    return ObjectState("e", "Rigid", "egg", 0, np.asarray(pos, float), np.asarray(quat, float), info)


def test_drop_targets_put_the_egg_off_the_hole_centre_along_its_axis_over_the_rim():
    o = _egg()                                         # lying: its long axis (local z) along world -y
    hole = {"center": np.array([0.03, -0.15]), "radius": 0.0245, "top": 0.814, "floor": 0.768}
    tg = HO.egg_drop_targets(o, hole, 0.814, o.R)
    assert tg and all(len(t) == 4 for t in tg)
    offs = sorted({t[0] for t in tg})
    assert offs == sorted(HO.EGG_DROP_OFFSETS)
    for off, yd, side, T in tg:
        R = T[:3, :3]
        c = R @ o.asset.center + T[:3, 3]
        u = R[:, 2].copy()
        u[2] = 0
        u /= np.linalg.norm(u)
        d = c[:2] - hole["center"]
        assert abs(float(np.linalg.norm(d)) - off) < 1e-9                # off the centre by the offset ...
        assert abs(abs(float(d @ u[:2])) - off) < 1e-9                   # ... along the long axis
        # the underside clears the far rim; the near end has nothing under it
        a, b = 0.0288, 0.0211
        x = (hole["radius"] - off) / a
        assert c[2] - 0.814 >= b * math.sqrt(1 - x * x) + HO.EGG_DROP_CLEAR - 1e-9
        assert hole["radius"] + off > a
    # the larger offset sits higher (its far rim is nearer its middle)
    z7 = min(T[2, 3] for off, _, _, T in tg if off == 0.007)
    z12 = min(T[2, 3] for off, _, _, T in tg if off == 0.012)
    assert z12 > z7


def test_dense_finger_samples_cover_the_root_block_between_its_corners():
    kin = X5Kin()
    T = np.eye(4)
    P = HO.finger_samples(kin, T, 1.0, n=4)
    Pc = kin.finger_points(T, 1.0)
    assert len(P) > 4 * len(Pc)
    lo, hi = Pc.min(0), Pc.max(0)
    assert np.all(P >= lo - 1e-9) and np.all(P <= hi + 1e-9)            # inside the corners' hull


def test_vectorised_wall_clearance_matches_the_per_point_one():
    prof = {"center": np.zeros(2), "rim": 0.04, "r_out": 0.1, "bin": 0.004,
            "surf": np.r_[np.full(15, -0.038), np.linspace(-0.038, 0.04, 11)], "floor": -0.038}

    class C:
        pos = np.array([0.1, -0.2, 0.82])
        R = np.eye(3)
    pts = np.array([[0.1, -0.2, 0.79], [0.17, -0.2, 0.81], [0.15, -0.15, 0.8], [0.5, 0.5, 0.8]])
    assert abs(HO.wall_clearance_v(C, prof, pts) - HO.wall_clearance(C, prof, pts)) < 1e-12
    assert HO.wall_clearance_v(C, prof, pts[3:]) == float("inf")


# ---------------------------------------------------------------------------------------------- G3 yaw belief offset
from dojo.skills.geometry import inv_T  # noqa: E402
from dojo.skills.kinematics import TABLE_Z  # noqa: E402
def _cyl_mesh(r, y0, y1, n=24):
    """A closed cylinder along y (vertices only at its two end rings)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(a), np.zeros(n), r * np.sin(a)], axis=1)
    V = np.vstack([ring + [0, y0, 0], ring + [0, y1, 0], [[0, y0, 0], [0, y1, 0]]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return V, np.asarray(F)


def _obj(label, typ, cat, pos, quat=(1, 0, 0, 0), lo=(-0.02, -0.02, 0), hi=(0.02, 0.02, 0.04), mesh=None, **asset):
    info = AssetInfo(key=f"{typ}/{cat}/00000", bbox_min=np.asarray(lo, float), bbox_max=np.asarray(hi, float),
                     mesh=mesh, **asset)
    return ObjectState(label, typ, cat, 0, np.asarray(pos, float), np.asarray(quat, float), info)


def _mallet(pos=(-0.25, -0.15, TABLE_Z + 0.03)):
    V, F = _cyl_mesh(0.004, 0.014, 0.2)
    return _obj("m", "Rigid", "mallet", pos, lo=(-0.013, -0.013, -0.013), hi=(0.013, 0.2, 0.013), mesh=(V, F))


def test_g3_yaw_belief_offset_turns_about_the_bbox_centre():
    from dojo.skills.g3 import OffsetBeliefProvider

    o = _mallet()
    c0 = o.R @ o.asset.center + o.pos

    class _W:
        objects = {"m": o}

    class _Base:
        def refresh(self, holding=None):
            return _W()

    prov = OffsetBeliefProvider(_Base(), {"m": [0.002, 0.0, 10.0]})
    w = prov.refresh()
    m = w.objects["m"]
    c1 = m.R @ m.asset.center + m.pos
    assert np.allclose(c1 - c0, [0.002, 0.0, 0.0], atol=1e-9)
    yaw = math.degrees(math.atan2(m.R[1, 0], m.R[0, 0]))
    assert abs(yaw - 10.0) < 1e-6
    assert np.allclose(inv_T(m.T)[:3, :3] @ m.R, np.eye(3), atol=1e-9)
