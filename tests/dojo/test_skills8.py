"""skills8: pens dropped into a holder's opening (fill_container's drop poses, the tip-down check) (CPU only)."""

from __future__ import annotations

import math

import numpy as np

from dojo.skills.env import KinematicEnv
from dojo.skills.skills import GTProvider, Skills

from tests.dojo.test_skills2 import _box_mesh, _merge, _obj

TABLE = 0.765


def _cup(label="h", xy=(0.0, -0.25), inner=0.03, wall=0.004, depth=0.11):
    """A square cup (an open box): 2 mm floor, ``inner`` half-width opening, ``depth`` deep."""
    a, b = inner, inner + wall
    parts = [_box_mesh((-b, -b, 0.0), (b, b, 0.002)),
             _box_mesh((-b, -b, 0.0), (-a, b, depth)), _box_mesh((a, -b, 0.0), (b, b, depth)),
             _box_mesh((-a, -b, 0.0), (a, -a, depth)), _box_mesh((-a, a, 0.0), (a, b, depth))]
    return _obj(label, "pen_holder", 0, (xy[0], xy[1], TABLE), (b, b, depth / 2), mesh=_merge(*parts))


def _pen(label="p0", xy=(-0.2, -0.15), length=0.15, r=0.006):
    """A lying pen along body x, its narrow (leading) end at -x."""
    mesh = _merge(_box_mesh((-length / 2 + 0.01, -r, -r), (length / 2, r, r)),
                  _box_mesh((-length / 2, -r / 3, -r / 3), (-length / 2 + 0.01, r / 3, r / 3)))
    return _obj(label, "pen", 0, (xy[0], xy[1], TABLE + r), (length / 2, r, r), mesh=mesh)


def _sk(objs):
    from dojo.skills.world import AssetDB
    rows = [{"label": o.label, "inst_name": o.label, "type": o.obj_type, "category": o.category, "category_idx": o.idx,
             "clutter": False, "pos": o.pos, "quat": o.quat, "half": o.asset.extents / 2, "center": o.asset.center}
            for o in objs]
    env = KinematicEnv(rows, step_lim=500, instruction="")
    sk = Skills(env, GTProvider(env, None), reserve=30, log=lambda m: None)
    for o in objs:                                   # the synthetic assets (meshes) instead of a catalogue's
        sk.world.objects[o.label].asset = o.asset
    sk.provider.refresh = lambda holding=None, _w=sk.world: _w
    return sk


def test_drop_target_holds_the_leading_end_just_over_the_rim():
    from dojo.skills.library import FILL_DROP_H, _fill_drop_target, _leading_end
    cup, pen = _cup(), _pen()
    sk = _sk([cup, pen])
    T, fit = _fill_drop_target(sk, "p0", "h", [])
    assert T is not None
    V = pen.asset.mesh[0] @ T[:3, :3].T + T[:3, 3]
    assert abs(float(V[:, 2].min()) - (TABLE + 0.11 + FILL_DROP_H)) < 0.002     # lowest point 4 mm over the rim
    assert float(np.degrees(np.arccos(T[:3, :3] @ [-1.0, 0, 0] @ [0, 0, -1.0]))) < 1.0   # leading end straight down
    # the first one in the middle of the opening
    tip_xy = V[np.argmin(V[:, 2])][:2] - cup.pos[:2]
    assert float(np.max(np.abs(tip_xy))) < 0.006
    pen.pos, pen.quat = T[:3, 3].copy(), _quat(T[:3, :3])
    tip, up = _leading_end(pen)
    assert abs(float(tip[2]) - float(V[:, 2].min())) < 1e-6 and float(up[2]) > 0.999


def test_leaning_drop_poses_keep_the_tip_and_go_inside():
    from dojo.skills.library import FILL_DROP_H, FILL_LEAN_DEG, FILL_LEAN_IN, _fill_drop_target, _leaning
    cup, pen = _cup(), _pen()
    sk = _sk([cup, pen])
    T, _ = _fill_drop_target(sk, "p0", "h", [])
    poses = _leaning(sk, "p0", T)
    assert len(poses) == 1 + 2 * len(FILL_LEAN_DEG)
    V0 = pen.asset.mesh[0] @ T[:3, :3].T + T[:3, 3]
    tip0 = V0[np.argmin(V0[:, 2])]
    for k, Tl in enumerate(poses[1:]):
        deg = FILL_LEAN_DEG[k // 2]
        axis = Tl[:3, :3] @ [1.0, 0, 0]                    # body +x: from the leading end to the other end
        assert abs(math.degrees(math.acos(float(axis[2]))) - deg) < 0.5
        V = pen.asset.mesh[0] @ Tl[:3, :3].T + Tl[:3, 3]
        low = V[np.argmin(V[:, 2])]
        assert np.linalg.norm(low[:2] - tip0[:2]) < 0.005          # another corner of the narrow end
        assert abs(float(low[2]) - (TABLE + 0.11 - FILL_LEAN_IN)) < 0.002
        arm = ("left", "right")[k % 2]                     # the top leans towards that arm's base
        to_base = sk.kin.base(arm)[:2, 3] - low[:2]
        assert float(axis[:2] @ to_base) > 0
    assert FILL_DROP_H > 0


def test_in_tip_down():
    from dojo.skills.library import in_tip_down
    cup, pen = _cup(), _pen()
    sk = _sk([cup, pen])
    p = sk.world.obj("p0")
    assert not in_tip_down(sk, "p0", "h")                 # lying on the table
    # upright, leading end on the cup's floor
    up = np.array([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]])                # body +x -> world +z
    p.quat = _quat(up)
    p.pos = np.array([cup.pos[0] + 0.01, cup.pos[1], TABLE + 0.002 + 0.075])
    assert in_tip_down(sk, "p0", "h")
    # upside down (leading end up): no
    p.quat = _quat(np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]))
    assert not in_tip_down(sk, "p0", "h")
    # leaning 50 deg: no
    a = math.radians(50)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1.0, 0], [-math.sin(a), 0, math.cos(a)]]) @ up
    p.quat = _quat(R)
    assert not in_tip_down(sk, "p0", "h")


def _quat(R):
    from dojo.skills.geometry import R_to_quat
    return R_to_quat(np.asarray(R, float))


def test_next_drop_is_packed_and_failed_grasps_are_skipped():
    from dojo.skills.library import FILL_PACK_CLEAR, _fill_drop_target, _fingers_above
    cup, pen, other = _cup(), _pen(), _pen("p1", xy=(-0.2, -0.1))
    sk = _sk([cup, pen, other])
    o1 = sk.world.obj("p1")                               # one already standing in the middle of the opening
    o1.quat = _quat(np.array([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]]))
    o1.pos = np.array([cup.pos[0], cup.pos[1], TABLE + 0.002 + 0.075])
    T, fit = _fill_drop_target(sk, "p0", "h", ["p1"])
    assert T is not None and fit.clearance >= FILL_PACK_CLEAR - 1e-9
    V = pen.asset.mesh[0] @ T[:3, :3].T + T[:3, 3]
    tip_xy = V[np.argmin(V[:, 2])][:2] - cup.pos[:2]
    assert float(np.max(np.abs(tip_xy))) > 0.012              # against the wall, away from the one in the middle
    src = _fingers_above(sk, "p0", T, TABLE + 0.13)
    gs = src("left")
    assert gs
    skip = gs[0].note
    assert all(g.note != skip for g in _fingers_above(sk, "p0", T, TABLE + 0.13, skip=(skip,))("left"))
