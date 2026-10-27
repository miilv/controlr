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


def test_tipped_in_pose_stands_the_egg_in_the_hole_far_end_up():
    o = _egg(quat=(0.7071, 0.0, 0.7071, 0.0))             # lying: its long axis (local z) along world +x
    hole = {"center": np.array([0.03, -0.15]), "radius": 0.0245, "top": 0.814, "floor": 0.769}
    T_rel = np.eye(4)
    T_rel[:3, :3] = o.R
    T_rel[:3, 3] = np.r_[0.03 + 0.012, -0.15, 0.835] - o.R @ o.asset.center     # released 12 mm towards +x
    T = HO.tipped_in_pose(o, hole, T_rel)
    ax = T[:3, 2]
    assert abs(math.degrees(math.acos(abs(ax[2]))) - HO.EGG_TIP_DEG) < 1e-6        # 30 deg from vertical
    up_end = ax if ax[2] > 0 else -ax
    assert up_end[0] > 0                                                             # the far (+x) end stays up
    c = T[:3, :3] @ o.asset.center + T[:3, 3]
    assert np.linalg.norm(c[:2] - hole["center"]) < 0.006
    assert 0.769 + 0.02 < c[2] < 0.814                                               # in the hole, under the plate


def test_heldsim_confirm_ignores_the_spin_of_a_round_object():
    """G3's confirm_pose stand-in compares a round object's long axis, not its spin about it (the head view cannot
    tell an egg turned about its axis); a box is still compared in full."""
    from dojo.skills.env import KinematicEnv
    from dojo.skills.geometry import R_to_quat, quat_to_R
    from dojo.skills.heldsim import HeldBeliefProvider
    from dojo.skills.skills import GTProvider
    for half, adopted in (([0.0205, 0.021, 0.029], True), ([0.03, 0.012, 0.01], False)):
        env = KinematicEnv([{"label": "e", "category": "egg", "pos": [-0.2, -0.15, 0.80], "quat": [1, 0, 0, 0],
                             "half": half}])
        prov = HeldBeliefProvider(GTProvider(env, None), env, mode=None, drop="fixtures")
        w = prov.refresh()
        o = w.obj("e")
        k = int(np.argmax(o.asset.extents))
        prov._frozen["e"] = (o.pos + np.array([0.1, 0.0, 0.0]), o.quat.copy())     # the belief: 10 cm off
        R_spun = o.R @ HO._rot(np.eye(3)[k], math.radians(90.0))                    # the same object turned about it
        r = prov.confirm_pose("e", o.pos.copy(), R_to_quat(R_spun))
        assert r["adopted"] is adopted, half
        assert np.allclose(quat_to_R(R_to_quat(R_spun)), R_spun, atol=1e-9)


def test_slide_path_pushes_a_lying_egg_along_its_axis_towards_the_side_it_was_let_go_on():
    o = _egg()                                         # lying: its long axis (local z) along world -y
    hole = {"center": np.array([0.0, -0.2]), "radius": 0.0245, "top": 0.814, "floor": 0.768}
    half = 0.5 * float(o.asset.extents[2])
    for dy in (0.006, -0.006):                          # let go 6 mm off the hole's centre, on either side
        T = o.T.copy()
        T[:3, 3] = np.array([0.0, -0.2 + dy, 0.79]) - o.R @ o.asset.center      # its middle dy off the hole's centre
        p0, p1, u = HO.slide_path(o, hole, T)
        assert np.allclose(np.abs(u), [0.0, 1.0, 0.0]) and math.copysign(1.0, u[1]) == math.copysign(1.0, dy)
        # it starts beyond the egg's other end and pushes towards the side it was let go on, just over the plate
        assert math.isclose(p0[1], -0.2 + dy - u[1] * (half + HO.EGG_SLIDE_START), abs_tol=1e-9)
        assert np.allclose(p1 - p0, u * HO.EGG_SLIDE)
        assert math.isclose(p0[2], hole["top"] + HO.EGG_SLIDE_Z) and math.isclose(p1[2], p0[2])
    up = _egg(quat=(1.0, 0.0, 0.0, 0.0))               # standing: nothing to push
    assert HO.slide_path(up, hole, up.T) is None
