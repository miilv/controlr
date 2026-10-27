"""Holders with holes and a hinged lid: put objects upright into the holes, then swing the lid shut.

* ``fill_holes(objects, holder, lid_tag)``  each object stood (long axis near vertical) over a free hole of the holder's
                                            body and let go with the fingertips above it, so it drops in; then the
                                            lid (the revolute part ``lid_tag`` rides on) is swung closed.

Geometry comes from the catalogue only: the holes are openings in a depth map of the body link's mesh, the lid is the
moving link of the joint the lid's functional point names (``links.json``: per-link points, joint axis / origin /
limits). Objects lying inside a round container (a basket) are grasped only where the fingers stay inside its wall.
"""

from __future__ import annotations

import math

import numpy as np

from dojo.skills import grasps as G
from dojo.skills.executor import ArmPlan, BudgetExceeded, EpisodeEnded
from dojo.skills.geometry import T_from
from dojo.skills.kinematics import ARMS, TIP_BEYOND_TCP, gap_to_grip
from dojo.skills.motion import FAST, MEDIUM
from dojo.skills.skills import CLOSED, OPEN, SkillResult, Skills


def _rot(axis, ang: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


# ============================================================================ geometry


def lid_joint(o, tag: str | None = None):
    """(joint name, joint dict, moving link, body link) of an articulated object's revolute joint: the one ``tag``'s
    functional point names, else the first revolute joint."""
    joints = o.asset.joints or {}
    jn = None
    if tag:
        raw = (o.asset.functional or {}).get(tag) or (o.asset.active_functional or {}).get(tag) or {}
        jn = raw.get("parent_joint")
        jn = jn[0] if isinstance(jn, list) and jn else jn
    if jn not in joints:
        jn = next((k for k, v in joints.items() if v.get("type") == "revolute"), None)
    if jn is None:
        return None
    J = joints[jn]
    return jn, J, J.get("body1"), J.get("body0")


def joint_angle(o, J: dict) -> float:
    """The joint's position in the asset's units (degrees for USD revolute joints): measured when the state has it
    (radians from physics are converted), else the authored rest position."""
    lo, hi = float(J.get("lower", 0.0)), float(J.get("upper", 0.0))
    if o.joints is not None and len(np.ravel(o.joints)):
        v = float(np.ravel(o.joints)[0])
        if max(abs(lo), abs(hi)) > 2 * math.pi + 0.1 and abs(v) <= 2 * math.pi + 0.1:
            v = math.degrees(v)
        return v
    return float(J.get("position") or lo)


def joint_ratio(o, J: dict) -> float:
    lo, hi = float(J.get("lower", 0.0)), float(J.get("upper", 0.0))
    return (joint_angle(o, J) - lo) / (hi - lo) if hi != lo else 0.0


def body_holes(o, body_link: str | None, min_depth: float = 0.02) -> list[dict]:
    """The holes of the holder's body seen from above, WORLD frame: {"center" (x, y), "radius", "floor" z, "top" z}.
    From the catalogue mesh restricted to the body link's box (the open lid stands above it)."""
    from dojo.skills.insertion import openings
    if o.asset.mesh is None:
        return []
    V, F = o.asset.mesh
    V = np.asarray(V, float)
    if body_link and o.asset.links and body_link in o.asset.links:
        mx = np.asarray(o.asset.links[body_link]["bbox_max"], float)
        keep = np.all(V[F][:, :, 2] <= mx[2] + 0.0005, axis=1)
        F = np.asarray(F)[keep]
        used, inv = np.unique(F, return_inverse=True)
        V, F = V[used], inv.reshape(F.shape)
    ops = openings(V, F, min_depth=min_depth, res=0.002, min_cells=60)
    out = []
    for op in ops:
        c = np.r_[op["center"], op["floor"]]
        cw = o.R @ c + o.pos
        top = float((o.R @ np.r_[op["center"], op["top"]] + o.pos)[2])
        out.append({"center": cw[:2], "radius": 0.5 * min(op["width"], op["length"]), "floor": float(cw[2]), "top": top})
    return out


def standing_targets(o, xy, z_bottom: float, tilts=(0.0, 20.0, 35.0), spins=(0.0, 90.0),
                     ups=(1.0, -1.0)) -> list[np.ndarray]:
    """Poses of ``o`` with its long axis (largest bbox extent) tilted ``tilt`` deg from vertical (either end up), spun
    about that axis, its bbox centre over ``xy`` and its lowest mesh point at ``z_bottom``. Tilts toward world +x only:
    the caller's yaw symmetry turns them about the vertical."""
    k = int(np.argmax(o.asset.extents))
    V = np.asarray(o.asset.mesh[0], float) if o.asset.mesh is not None else \
        np.array([[x, y, z] for x in (o.asset.bbox_min[0], o.asset.bbox_max[0])
                  for y in (o.asset.bbox_min[1], o.asset.bbox_max[1]) for z in (o.asset.bbox_min[2], o.asset.bbox_max[2])])
    e = np.zeros(3)
    e[k] = 1.0
    out = []
    for tilt in tilts:
        u = np.array([math.sin(math.radians(tilt)), 0.0, math.cos(math.radians(tilt))])
        for sgn in ups:
            # the rotation taking the body long axis (sgn * e) to u
            a = sgn * e
            v = np.cross(a, u)
            s, c = float(np.linalg.norm(v)), float(a @ u)
            if s > 1e-9:
                R0 = _rot(v, math.atan2(s, c))
            elif c > 0:
                R0 = np.eye(3)
            else:                                              # anti-parallel: half a turn about a perpendicular
                R0 = _rot(np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0]), math.pi)
            for spin in spins:
                R = R0 @ _rot(e, math.radians(spin))
                W = V @ R.T
                c3 = (W.min(0) + W.max(0)) / 2
                p = np.array([xy[0] - c3[0], xy[1] - c3[1], z_bottom - W[:, 2].min()])
                out.append(T_from(p, R))
    return out


# ---- objects inside a round container: grasps whose fingers stay inside its wall


def wall_profile(c, bin_r: float = 0.004) -> dict | None:
    """A round open container's inner surface as a radial height field (object frame): axis centre (x, y), and per
    radius bin the highest mesh point (the floor near the axis, the wall further out, the rim at the edge). None for a
    mesh that is not bowl-like (no floor well below the rim)."""
    if c.asset.mesh is None:
        return None
    V = np.asarray(c.asset.mesh[0], float)
    ctr = (V[:, :2].min(0) + V[:, :2].max(0)) / 2
    r = np.linalg.norm(V[:, :2] - ctr, axis=1)
    r_out = float(r.max())
    rim = float(V[:, 2].max())
    nb = int(math.ceil(r_out / bin_r)) + 1
    surf = np.full(nb, -np.inf)
    idx = np.minimum((r / bin_r).astype(int), nb - 1)
    np.maximum.at(surf, idx, V[:, 2])
    for i in range(1, nb):                              # empty bins: the inner neighbour's height
        if not np.isfinite(surf[i]):
            surf[i] = surf[i - 1]
    if not np.isfinite(surf[0]) or rim - float(surf[0]) < 0.02:
        return None
    return {"center": ctr, "rim": rim, "r_out": r_out, "bin": bin_r, "surf": surf, "floor": float(surf[0])}


def wall_clearance(c, prof: dict, pts_w: np.ndarray) -> float:
    """Smallest height (m) of world points above the container's inner surface (negative: inside its material)."""
    P = (np.asarray(pts_w, float) - c.pos) @ c.R
    r = np.linalg.norm(P[:, :2] - prof["center"], axis=1)
    out = np.inf
    for p, rp in zip(P, r):
        if rp > prof["r_out"] + 0.004:
            continue
        i = min(int(rp / prof["bin"]), len(prof["surf"]) - 1)
        lo = max(0, i - 1)
        s = float(prof["surf"][lo:i + 2].max())          # the bin and its neighbours (finger width)
        out = min(out, float(p[2]) - s)
    return out


def inside_wall(c, prof: dict, pts_w: np.ndarray, margin: float = 0.003) -> bool:
    """Do world points stay above the container's inner surface (inside its wall, over its floor) by ``margin``?"""
    return wall_clearance(c, prof, pts_w) >= margin


def round_grasps(o, kin, arm, rolls=(0.0, 12.0, -12.0, 25.0, -25.0, 40.0, -40.0),
                 tilts=(0.0, 25.0, -25.0, 40.0, -40.0), along: float = 0.0, ends: bool = True) -> list:
    """Grasps of an object with a round cross-section about its long axis (an egg, a can lying down): the pads close
    through the long axis at its middle, the closing direction rolled about the long axis by ``rolls`` (deg from
    horizontal: the fingers clear a wall next to it), the approach tilted about the closing axis by ``tilts`` (it turns
    the long axis up when the object is stood in a place). ``along``: the pads that far along the body long axis from
    the middle (stood up with that end down, the wider middle rests on the pads instead of sliding out between them)."""
    ext = o.asset.extents
    k = int(np.argmax(ext))
    short = [i for i in range(3) if i != k]
    if abs(float(ext[short[0]] - ext[short[1]])) > 0.1 * float(max(ext[short])):
        return []
    width = float(max(ext[short]))
    if width > G.MAX_WIDTH:
        return []
    ax = o.R[:, k]
    ax = ax / np.linalg.norm(ax)
    c = o.center_w + ax * along
    h = np.cross(ax, [0.0, 0.0, 1.0])                   # horizontal, across the long axis
    if np.linalg.norm(h) < 0.3:
        return []                                       # standing on its end: not this generator
    h = h / np.linalg.norm(h)
    out = []
    for roll in rolls:
        cl = _rot(ax, math.radians(roll)) @ h
        a0 = np.cross(cl, ax)
        a0 = a0 if a0[2] < 0 else -a0                   # approach from above
        for tilt in tilts:
            appr = _rot(cl, math.radians(tilt)) @ a0
            if appr[2] > -0.35:
                continue
            for flip in (1.0, -1.0):
                Rg = G._frame(appr, flip * cl)
                out.append(G.Grasp("round", T_from(c, Rg), width,
                                   score=abs(tilt) / 40.0 + abs(roll) / 60.0 + 0.2 * width / 0.07,
                                   note=f"roll {roll:.0f} tilt {tilt:.0f}"))
    # end to end (pads on the two tips): free when neighbours touch its sides; it cannot stand the object up, but it
    # gets it out of a crowded container
    L = float(ext[k])
    if ends and L <= G.MAX_WIDTH and abs(float(ax[2])) < 0.5:
        a_dn = np.array([0.0, 0.0, -1.0]) - ax * float(-ax[2])
        a_dn = a_dn / np.linalg.norm(a_dn)
        for tilt in (0.0, 20.0, -20.0, 35.0, -35.0, 50.0, -50.0):     # leaning in keeps the palm off a wall
            appr = _rot(ax, math.radians(tilt)) @ a_dn
            for flip in (1.0, -1.0):
                Rg = G._frame(appr, flip * ax)
                out.append(G.Grasp("ends", T_from(c, Rg), L, score=0.6 + abs(tilt) / 40.0, note=f"ends tilt {tilt:.0f}"))
    return out


def container_of(sk: Skills, label: str):
    """The Geometry container (bowl-like, larger) whose footprint the object lies in, or None."""
    w = sk.world
    o = w.obj(label)
    best = None
    for c in w.task_objects():
        if c.obj_type != "Geometry" or c.label == label:
            continue
        mn, mx = c.aabb()
        if mn[0] <= o.pos[0] <= mx[0] and mn[1] <= o.pos[1] <= mx[1] and o.pos[2] < mx[2] and \
                float(np.prod((mx - mn)[:2])) > 0.01:
            best = c
    return best


WALL_MARGIN = -0.008   # m; finger samples may touch a container's wall / rim by this much (the samples are the finger's
                      # bounding corners: the 38 x 56 mm root block meets a basket's rim before the finger does)


def grasps_inside(sk: Skills, label: str, arm: str, tilts=(0.0, 25.0, 40.0, -25.0, -40.0),
                  margin: float = WALL_MARGIN, along: float | None = None) -> list:
    """Box + usual grasps of ``label``; when it lies inside a round container, only those whose open fingers stay
    inside the wall at the grasp and on the way in."""
    o = sk.world.obj(label)
    if along is not None:                 # a stand-up place: round grasps off the middle toward the end that goes down,
        # pads level (twin: of the eggs turned up from grasps rolled 12 deg about the long axis none stayed in the pads,
        # 0/6; level pads 1/1)
        cands = round_grasps(o, sk.kin, arm, rolls=(0.0,), along=along, ends=False)
    else:
        cands = round_grasps(o, sk.kin, arm) + G.box_grasps(o, sk.kin, arm, tilts=tuple(tilts)) + \
            G.candidates(o, sk.kin, arm)
    c = container_of(sk, label)
    prof = wall_profile(c) if c is not None else None
    if prof is None:
        return cands
    # in a container: also opened only 4 mm per side (the finger roots stay 7 mm further in)
    cands = cands + [G.narrowed(g) for g in cands if g.method in ("round", "ends")]
    import dataclasses
    out = []
    for g in cands:
        go = gap_to_grip(g.open_gap())
        # raise a grasp whose fingers dip into the floor by up to 6 mm (the pads still cover the widest section)
        cl = wall_clearance(c, prof, sk.kin.finger_points(sk.kin.flange_from_tcp(g.T), go))
        if cl < margin:
            if margin - cl > 0.006 or along is not None:      # a raised grasp does not hold an egg turned up
                continue
            T = g.T.copy()
            T[2, 3] += margin - cl + 0.0005
            g = dataclasses.replace(g, T=T, note=g.note + " raised")
        ok = True
        for d in (0.0, 0.03, 0.06):
            T = g.T.copy()
            T[:3, 3] -= g.approach * d
            if not inside_wall(c, prof, sk.kin.finger_points(sk.kin.flange_from_tcp(T), go), margin=margin):
                ok = False
                break
        if ok:
            out.append(g)
    return out


# ---- dense finger samples against a container's wall (the corner samples straddle a basket's rim)


def finger_samples(kin, T_flange, gripper: float, n: int = 4) -> np.ndarray:
    """Points over both fingers' zone boxes (``X5Kin.finger_points``' zones, ``n`` per edge, both faces and the
    middle): world frame. The root block is 38 x 56 mm; its corners alone let a basket's rim pass between them
    (egg_v6: hands stopped 15-65 mm short of eggs whose planned grasps cleared the wall at every corner)."""
    from dojo.skills.kinematics import FINGER_MAX, GRIP_SCALE
    qf = min(max(GRIP_SCALE[0] + gripper * (GRIP_SCALE[1] - GRIP_SCALE[0]), 0.0), FINGER_MAX)
    zones = ((0.075, 0.105, 0.0378, 0.028), (0.107, 0.126, 0.0134, 0.015), (0.128, 0.1576, 0.0034, 0.009))
    loc = []
    for s in (1, -1):
        for x0, x1, yo, zh in zones:
            for x in np.linspace(x0, x1, n):
                for y in (0.0004, 0.5 * yo, yo):
                    for z in np.linspace(-zh, zh, n):
                        loc.append((x, s * (qf + y), z))
    loc = np.asarray(loc)
    T = np.asarray(T_flange)
    return (T[:3, :3] @ loc.T).T + T[:3, 3]


def wall_clearance_v(c, prof: dict, pts_w: np.ndarray) -> float:
    """``wall_clearance`` for many points at once: the smallest height of the points above the container's inner
    surface (each radius bin and its two neighbours: the finger's width), +inf when none is over the container."""
    P = (np.asarray(pts_w, float) - c.pos) @ c.R
    r = np.linalg.norm(P[:, :2] - prof["center"], axis=1)
    m = r <= prof["r_out"] + 0.004
    if not m.any():
        return float("inf")
    surf = np.asarray(prof["surf"], float)
    s3 = np.maximum(np.maximum(surf, np.r_[surf[1:], surf[-1]]), np.r_[surf[0], surf[:-1]])
    i = np.minimum((r[m] / prof["bin"]).astype(int), len(surf) - 1)
    return float((P[m, 2] - s3[i]).min())


EGG_WALL_CLEAR = -0.010       # m: dense finger samples at most this far into a container's inner surface profile (grasp
                              # + descent). egg_v6 traces: hands that went down had >= -12.6 mm, blocked ones <= -17 mm
                              # (the profile is the max over 4 mm radius bins of a woven mesh; the collider is leaner)
EGG_ENDS = False              # pick eggs by their tips too (last)
EGG_PUSH_OK = -0.003         # m: the fingers may push a neighbouring egg this far (it rolls aside)
EGG_LIFT_OVER = 0.045         # m: an object picked inside a container is lifted until its centre is this far over the rim
EGG_DROP_OFFSETS = (0.012, 0.007)   # m: a lying egg is let go with its centre this far along its long axis off the hole
                              # centre: its far end rests on the rim, the near end has nothing under it and it tips in
                              # (centred, a lying egg spans the hole and can stay). 12 mm first: 17 eggs in vs 13 with
                              # 7 mm first, same 11 layouts (G3 mode A; 4 layouts up, 1 down)
EGG_DROP_CLEAR = 0.002        # m: its underside over the far rim at the release
EGG_STAND_FALLBACK = True     # no lying drop plans (the back holes, under the open lid): stand the held egg up instead
STAND_PLACE_KW = {"servo": True, "fingers_out": True, "probe": False, "hover_clear": 0.008, "descend_rad": 0.03,
                  "pre_h": 0.05, "approach_rad": 0.05, "ease": 6}


def egg_grasps(sk: Skills, label: str, arm: str, wall_clear: float = EGG_WALL_CLEAR) -> list:
    """Top-down grasps of a round-section object lying (an egg on its side) for a lying carry: level pads across its
    middle, pads on its two tips, then rolled / tilted variants; in a round container only those whose dense finger
    samples stay ``wall_clear`` above its inner surface at the grasp and along the descent from above its rim (the
    pre-grasp is moved there)."""
    import dataclasses
    o = sk.world.obj(label)
    cands = round_grasps(o, sk.kin, arm, rolls=(0.0,), tilts=(0.0,), ends=False)
    cands += [g for g in round_grasps(o, sk.kin, arm, rolls=(12.0, -12.0, 25.0, -25.0), tilts=(0.0, 20.0, -20.0),
                                      ends=False)]
    if EGG_ENDS:                                    # pads on its two tips: they slipped (egg_drop1 100000: rose 1 mm)
        cands += [g for g in round_grasps(o, sk.kin, arm, rolls=(0.0,), tilts=(0.0,), ends=True) if g.method == "ends"]
    if not cands:                                   # standing on its end, or not round: the usual generators
        cands = G.box_grasps(o, sk.kin, arm, tilts=(0.0, 25.0, -25.0)) + G.candidates(o, sk.kin, arm)
    c = container_of(sk, label)
    prof = wall_profile(c) if c is not None else None
    if prof is None:
        return cands
    rim_w = float((c.R @ np.r_[prof["center"], prof["rim"]] + c.pos)[2])
    out = []
    for g in cands + [G.narrowed(x) for x in cands if x.method in ("round", "ends")]:
        go = gap_to_grip(g.open_gap())
        a = g.approach
        if a[2] > -0.5:
            continue
        # the pre-grasp above the rim (fingertips 2 cm over it), the descent a straight line from there
        d_pre = max(g.pre_dist, (rim_w + 0.02 + TIP_BEYOND_TCP - float(g.T[2, 3])) / max(-float(a[2]), 0.3))
        ok = True
        for d in np.linspace(0.0, d_pre, 6):
            T = g.T.copy()
            T[:3, 3] -= a * d
            P = finger_samples(sk.kin, sk.kin.flange_from_tcp(T), go)
            if wall_clearance_v(c, prof, P) < wall_clear:
                ok = False
                break
        if ok:
            out.append(dataclasses.replace(g, pre_dist=float(d_pre)))
    return out


def egg_drop_targets(o, hole: dict, plate_top: float, R_now, offsets=EGG_DROP_OFFSETS, clear: float = EGG_DROP_CLEAR,
                     yaws=range(0, 360, 30)) -> list[tuple[float, float, float, np.ndarray]]:
    """Poses of a lying object carried as it is held (``R_now`` turned about the vertical by ``yaws``): its bbox centre
    ``offset`` along its horizontal long axis (either way) from the hole centre, as low as its underside clears the far
    rim by ``clear`` (an ellipsoid of the bbox's half extents). (offset, yaw deg, side, T), smaller offsets first."""
    k = int(np.argmax(o.asset.extents))
    a_ = 0.5 * float(o.asset.extents[k])
    b_ = 0.5 * float(max(np.delete(o.asset.extents, k)))
    r_h = float(hole["radius"])
    out = []
    for off in offsets:
        x = (r_h - off) / a_
        z_c = plate_top + b_ * math.sqrt(max(0.0, 1.0 - x * x)) + clear
        for yd in yaws:
            c_, s_ = math.cos(math.radians(yd)), math.sin(math.radians(yd))
            R = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]]) @ np.asarray(R_now, float)
            u = R[:, k].copy()
            u[2] = 0.0
            if np.linalg.norm(u) < 0.3:
                continue
            u /= np.linalg.norm(u)
            for side in (1.0, -1.0):
                cw = np.r_[np.asarray(hole["center"], float)[:2] + side * off * u[:2], z_c]
                out.append((float(off), float(yd), side, T_from(cw - R @ o.asset.center, R)))
    return out


EGG_NUDGE = 0.03              # m: an egg resting against a container's wall is pushed this far towards its middle
EGG_NUDGE_GAP = 0.0045        # m: the closed fingertips go down this far outside the egg's side facing the wall


def nudge_from_wall(sk: Skills, label: str, wall_clear: float = EGG_WALL_CLEAR) -> SkillResult | None:
    """Push a round object lying against a round container's wall ``EGG_NUDGE`` towards the container's middle: the
    closed fingertips go down into the gap between its side and the sloped wall at its widest height (the closing axis
    along the push, 7 mm thick) and move inward (egg_drop1: eggs resting on the wall's foot had no grasp whose fingers
    stay out of the wall, 3 of 4 per layout). None when it is not against a wall or no push plans."""
    o = sk.world.obj(label)
    c = container_of(sk, label)
    prof = wall_profile(c) if c is not None else None
    if prof is None:
        return None
    ctr = c.R @ np.r_[prof["center"], 0.0] + c.pos
    d = np.r_[o.center_w[:2] - ctr[:2], 0.0]
    if float(np.linalg.norm(d)) < 0.02:
        return None
    u = d / np.linalg.norm(d)
    V = np.asarray(o.asset.mesh[0], float) @ o.R.T + o.pos if o.asset.mesh is not None else o.world_bbox_corners()
    ext = float(((V[:, :2] - o.center_w[:2]) @ u[:2]).max())         # its side facing the wall
    z_mid = float(o.center_w[2])
    from dojo.skills.kinematics import tilted_R
    out_note = []
    arms = sorted(ARMS, key=lambda a: float(np.linalg.norm(sk.kin.base(a)[:2, 3] - o.pos[:2])))
    # at its widest height, or higher on its side where the sloped wall leaves the fingertips room (a push above its
    # middle rolls it the same way)
    combos = [(a, lean, dz) for dz in (0.0, 0.006, 0.010, 0.014) for a in arms for lean in (0.0, 15.0)]
    for a, lean, dz in combos:
        tip0 = np.r_[o.center_w[:2] + u[:2] * (ext + EGG_NUDGE_GAP), z_mid + dz]
        if True:
            Rg = tilted_R(math.atan2(u[1], u[0]), 90.0 - lean, math.atan2(u[1], u[0]))     # wrist over the middle
            x = Rg[:, 0]

            def tcp_at(tip):
                return T_from(np.asarray(tip, float) - x * TIP_BEYOND_TCP, Rg)
            path = [tip0 + np.r_[0, 0, 0.06], tip0, tip0 - u * EGG_NUDGE]
            ok = True
            for k in range(len(path) - 1):
                for t in np.linspace(0.0, 1.0, 6):
                    P = finger_samples(sk.kin, sk.kin.flange_from_tcp(tcp_at(path[k] + t * (path[k + 1] - path[k]))),
                                       CLOSED)
                    if wall_clearance_v(c, prof, P) < wall_clear:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                out_note.append(f"{a} lean {lean:.0f} +{dz * 1000:.0f}mm: wall")
                continue
            T_hi = tcp_at(path[0])
            to = sk.motion.to_pose(a, sk.ex.q[a], T_hi, FAST, "nudge_approach", gripper=CLOSED, ignore=(label,))
            if not to.ok:
                out_note.append(f"{a} lean {lean:.0f}: approach {to.note[:40]}")
                continue
            dn = sk.motion.line(a, to.q_end, tcp_at(path[1]), 0.04, "nudge_down", gripper=CLOSED, ignore=(label,))
            if not dn.ok:
                out_note.append(f"{a} lean {lean:.0f}: down {dn.note[:40]}")
                continue
            ps = sk.motion.line(a, dn.q_end, tcp_at(path[2]), 0.03, "nudge_push", gripper=CLOSED, check=False)
            if not ps.ok:
                out_note.append(f"{a} lean {lean:.0f}: push {ps.note[:40]}")
                continue
            T_up = tcp_at(path[2])
            T_up[2, 3] += 0.06
            up = sk.motion.line(a, ps.q_end, T_up, MEDIUM, "nudge_up", gripper=CLOSED, check=False)
            pl = ArmPlan(a, sk.ex.q[a].copy(), sk.ex.g[a]).grip(CLOSED, 2, "close_for_push")
            pl.move(to, "approach").move(dn, "down").move(ps, "push").hold(1, "settle")
            if up.segments:
                pl.move(up, "up")
            p0 = o.center_w.copy()
            try:
                sk.ex.run({a: pl}, name=f"nudge {label}")
            except BudgetExceeded as e:
                return sk._record(SkillResult(False, "nudge", "step_budget", 0, a, note=str(e)))
            except EpisodeEnded as e:
                return sk._record(SkillResult(False, "nudge", "ended", 0, a, note=str(e)))
            # mode B: a push moves the egg without a grasp (no release, no re-measure): the provider confirms the
            # predicted outcome (rolled inward by the push past the contact), as a topple's is (``confirm_moved``)
            T_exp = o.T.copy()
            T_exp[:3, 3] -= u * max(0.0, EGG_NUDGE - EGG_NUDGE_GAP)
            sk.confirm_moved(label, T_exp, "nudge")
            o2 = sk.refresh().obj(label)
            moved = float(np.dot((o2.center_w - p0)[:2], -u[:2]))
            return sk._record(SkillResult(moved > 0.008, "nudge", "ok" if moved > 0.008 else "place_miss", pl.steps, a,
                                          note=f"{label} {moved * 1000:.0f} mm off the wall ({a}, lean {lean:.0f}, "
                                               f"+{dz * 1000:.0f} mm over its middle)"))
    sk.log(f"[holes] {label}: no nudge ({'; '.join(out_note)[:200]})")
    return None


def HO_align(a, b) -> np.ndarray:
    """Rotation taking unit vector ``a`` onto ``b`` (shortest arc)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    s_, c_ = float(np.linalg.norm(v)), float(a @ b)
    if s_ < 1e-9:
        return np.eye(3) if c_ > 0 else _rot(np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0]), math.pi)
    return _rot(v / s_, math.atan2(s_, c_))


EGG_TIP_DEG = 30.0            # deg from vertical: how an egg let go lying off a hole's centre ends up in it (G3: 16-51)


def tipped_in_pose(o, hole: dict, T_release, tilt_deg: float = EGG_TIP_DEG) -> np.ndarray:
    """The expected pose of a lying object let go at ``T_release`` off ``hole``'s centre after it tipped in: its near end
    (towards the hole's centre) went down, its long axis keeps its heading and stands ``tilt_deg`` from vertical, its
    lower end on the hole's floor (mode B: nothing re-measures where it went; ``confirm_pose`` decides)."""
    k = int(np.argmax(o.asset.extents))
    R0 = np.asarray(T_release, float)[:3, :3]
    c0 = R0 @ o.asset.center + np.asarray(T_release, float)[:3, 3]
    ax = R0[:, k].copy()
    hc = np.asarray(hole["center"], float)[:2]
    far = c0[:2] - hc                                  # the release offset: from the hole centre to the far rim
    sgn = 1.0 if float(ax[:2] @ far) >= 0 else -1.0     # the axis' end that points at the far rim stays up
    u = np.r_[sgn * ax[:2], 0.0]
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    t = math.radians(90.0 - tilt_deg)                   # elevation of the far end
    a_new = math.cos(t) * u + np.array([0.0, 0.0, math.sin(t)])
    a_old = sgn * ax
    v = np.cross(a_old, a_new)
    sv, cv = float(np.linalg.norm(v)), float(a_old @ a_new)
    Rt = _rot(v / sv, math.atan2(sv, cv)) if sv > 1e-9 else np.eye(3)
    R = Rt @ R0
    half = 0.5 * float(o.asset.extents[k])
    w = 0.5 * float(max(np.delete(o.asset.extents, k)))
    zc = float(hole["floor"]) + math.hypot(half * math.sin(t), w * math.cos(t))      # an ellipsoid on its lower end
    c = np.r_[hc + u[:2] * 0.003, zc]
    return T_from(c - R @ o.asset.center, R)


EGG_SLIDE_IN = True           # an egg left lying across its hole is pushed along its axis first (mode B: every drop)
EGG_PRESS = True              # an egg left lying across its hole is pressed in from above
EGG_SUNK = 0.003              # m: its centre this far under the plate's top counts as in (lying across: 4-11 mm over it;
                              # the checker wants the centre < 40 mm over the holder's bottom frame, 5 mm over its base)
EGG_PRESS_DEPTH = 0.014       # m: the closed fingertips go this far below the egg's top


EGG_SLIDE = 0.045             # m: the fingertips' push along a lying egg's axis (it starts this far beyond its end)
EGG_SLIDE_START = 0.022       # m: the push starts this far out from the egg's believed end (in-hand error up to 15 mm)
EGG_SLIDE_Z = 0.007           # m: the fingertips' height over the plate during the push


def slide_in(sk: Skills, label: str, hole: dict, T_release, ignore=(), arm: str | None = None) -> str:
    """Push a lying object, let go across ``hole``, along its long axis towards the side it was let go on: once one end
    is over the hole with nothing under it, it tips in. Closed fingertips straight down beside its end on the other side
    (``EGG_SLIDE_START`` out from where it should be: the in-hand position along the axis is known to +-15 mm only in
    mode B), at its middle's height, then ``EGG_SLIDE`` along the axis. Uses the release pose, not the belief after it.
    (G3 mode B traces: let go 6 mm off the hole's centre instead of 12, the egg wedged across the hole.)"""
    o = sk.world.obj(label)
    k = int(np.argmax(o.asset.extents))
    T = np.asarray(T_release, float)
    R0 = T[:3, :3]
    c0 = R0 @ o.asset.center + T[:3, 3]
    ax = R0[:, k].copy()
    ax[2] = 0.0
    if float(np.linalg.norm(ax)) < 0.3:
        return "not lying"
    ax /= np.linalg.norm(ax)
    hc = np.asarray(hole["center"], float)[:2]
    u = ax if float(ax[:2] @ (c0[:2] - hc)) >= 0 else -ax          # towards the side it was let go on
    half = 0.5 * float(o.asset.extents[k])
    # just over the plate: the egg's lower end whatever its height in the hand (mode B: 10-14 mm lower than believed;
    # at its believed middle the fingertips passed over it, egg_mbD 100001)
    z_tip = float(hole["top"]) + EGG_SLIDE_Z
    p0 = np.r_[c0[:2] - u[:2] * (half + EGG_SLIDE_START), z_tip]
    p1 = p0 + u * EGG_SLIDE
    from dojo.skills.kinematics import down_R
    R = down_R(math.atan2(u[1], u[0]) + math.pi / 2)                 # closing across the push, the finger's width along it

    def tcp_at(p):
        return T_from(np.asarray(p, float) + np.array([0.0, 0.0, TIP_BEYOND_TCP]), R)
    # the arm that let it go first (it is right there; the other one's path crossed its hand: egg_mbE 100001)
    arms = sorted(ARMS, key=lambda a: (a != arm, float(np.linalg.norm(sk.kin.base(a)[:2, 3] - p0[:2]))))
    for a in arms:
        other = "right" if a == "left" else "left"
        if a != arm and arm is not None:
            sk.ex.run({other: sk.plan_home(other)}, name=f"clear {other}")
        to = sk.motion.to_pose(a, sk.ex.q[a], tcp_at(p0 + np.array([0.0, 0.0, 0.05])), FAST, "slide_approach",
                               gripper=CLOSED, ignore=(label,) + tuple(ignore))
        if not to.ok:
            continue
        dn = sk.motion.line(a, to.q_end, tcp_at(p0), 0.04, "slide_down", gripper=CLOSED, check=False)
        if not dn.ok:
            continue
        ps = sk.motion.line(a, dn.q_end, tcp_at(p1), 0.03, "slide_push", gripper=CLOSED, check=False)
        if not ps.ok:
            continue
        up = sk.motion.line(a, ps.q_end, tcp_at(p1 + np.array([0.0, 0.0, 0.06])), MEDIUM, "slide_up", gripper=CLOSED,
                            check=False)
        pl = ArmPlan(a, sk.ex.q[a].copy(), sk.ex.g[a]).grip(CLOSED, 2, "close_for_slide")
        pl.move(to, "approach").move(dn, "down").move(ps, "push").hold(2, "settle")
        if up.segments:
            pl.move(up, "up")
        try:
            sk.ex.run({a: pl}, name=f"slide {label}")
        except (BudgetExceeded, EpisodeEnded) as e:
            return f"stopped: {e}"
        sk.confirm_moved(label, tipped_in_pose(o, hole, T_release), "slid into a hole")
        sk.bank()
        return f"{a}, {pl.steps} steps"
    return "no slide plan"


def press_in(sk: Skills, label: str, hole: dict, ignore=(), at_end: bool = False) -> str:
    """Closed fingertips straight down onto the top of an object lying across ``hole``, at its point over the hole
    nearest the hole's centre, ``EGG_PRESS_DEPTH`` below its top, and back up (egg_nudge1: 2 of 9 eggs let go off a
    hole's centre stayed wedged across it, 4 mm over the plate, tilted 14 deg from level)."""
    o = sk.world.obj(label)
    k = int(np.argmax(o.asset.extents))
    ax = o.R[:, k]
    hc = np.asarray(hole["center"], float)[:2]
    V = np.asarray(o.asset.mesh[0], float) @ o.R.T + o.pos if o.asset.mesh is not None else o.world_bbox_corners()
    # the point of its long axis over the hole nearest the hole centre, on its upper surface
    s_ = float(np.clip((hc - o.center_w[:2]) @ ax[:2] / max(float(ax[:2] @ ax[:2]), 1e-9), -0.012, 0.012))
    if at_end:
        # its higher end (the one resting on the rim), 12 mm in from its tip: that end slides down into the hole
        half = 0.5 * float(o.asset.extents[k])
        s_ = (half - 0.012) * (1.0 if float(ax[2]) > 0 else -1.0)
    p = o.center_w + ax * s_
    near = V[np.linalg.norm(V[:, :2] - p[:2], axis=1) < 0.006]
    top = float(near[:, 2].max()) if len(near) else float(o.top_z)
    from dojo.skills.kinematics import down_R
    arms = sorted(ARMS, key=lambda a: float(np.linalg.norm(sk.kin.base(a)[:2, 3] - p[:2])))
    for a in arms:
        for yaw in (math.atan2(ax[1], ax[0]) + math.pi / 2, math.atan2(ax[1], ax[0])):
            R = down_R(yaw)

            def tcp_at(z):
                return T_from(np.r_[p[:2], z + TIP_BEYOND_TCP], R)
            to = sk.motion.to_pose(a, sk.ex.q[a], tcp_at(top + 0.04), FAST, "press_approach", gripper=CLOSED,
                                   ignore=(label,) + tuple(ignore))
            if not to.ok:
                continue
            dn = sk.motion.line(a, to.q_end, tcp_at(top - EGG_PRESS_DEPTH), 0.04, "press_down", gripper=CLOSED,
                                check=False)
            if not dn.ok:
                continue
            up = sk.motion.line(a, dn.q_end, tcp_at(top + 0.05), MEDIUM, "press_up", gripper=CLOSED, check=False)
            pl = ArmPlan(a, sk.ex.q[a].copy(), sk.ex.g[a]).grip(CLOSED, 2, "close_for_press")
            pl.move(to, "approach").move(dn, "down").hold(2, "pressed")
            if up.segments:
                pl.move(up, "up")
            try:
                sk.ex.run({a: pl}, name=f"press {label}")
            except (BudgetExceeded, EpisodeEnded) as e:
                return f"stopped: {e}"
            # mode B: no grasp, no re-measure; the expected outcome (in the hole, long axis near vertical, its lower end
            # on the hole's floor) goes to the provider's confirm_pose
            e = np.zeros(3)
            e[k] = 1.0
            R_up = HO_align(e, np.array([0.0, 0.0, 1.0]))
            c_exp = np.r_[hc, float(hole["floor"]) + 0.5 * float(o.asset.extents[k]) + 0.002]
            sk.confirm_moved(label, T_from(c_exp - R_up @ o.asset.center, R_up), "press")
            sk.bank()
            return f"{a}, {pl.steps} steps"
    return "no press plan"


# ============================================================================ skill


def _hole_free(w, hole: dict, others: list[str], skip: str) -> bool:
    for l in others:
        if l == skip or l not in w.objects:
            continue
        o = w.obj(l)
        if float(np.linalg.norm(o.pos[:2] - hole["center"])) < hole["radius"] and o.bottom_z < hole["top"] + 0.02:
            return False
    return True


def _to_table(sk: Skills, lab: str, H) -> bool:
    """Relay: move ``lab`` (as it lies, any grasp that fits where it is) to a free table spot near the holder, where
    every grasp is free for the stand-up place."""
    from dojo.skills.kinematics import TABLE_Z
    from dojo.skills.library import free_spots
    from dojo.skills.targets import rest_pose
    o = sk.world.obj(lab)
    side = 1.0 if float(o.pos[0]) > float(H.pos[0]) else -1.0
    spots = free_spots(sk, lab, near_xy=(float(H.pos[0]) + side * 0.14, -0.27))
    if not spots:
        return False
    targets = [rest_pose(sk, lab, p, TABLE_Z) for p in spots[:6]]
    plans = sk.plan_transfer(lab, targets, yaw_sym=2, grasps=lambda a: grasps_inside(sk, lab, a))
    if isinstance(plans, SkillResult):
        sk.log(f"[holes] {lab}: no move to the table either ({plans.note[:160]})")
        return False
    r = sk.run_transfer(*plans, tol_xy=0.03, tol_tilt=30.0)
    sk.bank()
    sk.log(f"[holes] {lab} moved to the table ({r.status})")
    return r.status not in ("ended", "step_budget")


def in_hole(o, hole: dict) -> bool:
    return bool(float(np.linalg.norm(o.pos[:2] - hole["center"])) < hole["radius"] - 0.004
                and o.bottom_z < hole["top"] - 0.01)


FILL_MODE = "drop"            # "drop": carried lying as picked, let go low over a hole off its centre (it tips in);
                              # "stand": turned upright in the pads first (skills-6: the egg left the pads on most carries)


def _container_obstacle(sk: Skills, c, on: bool) -> None:
    """A round container (a basket) as a collision obstacle shaped by its mesh's top surface (``Collision.heightmaps``):
    transits and places go over its rim, not through it (egg_v6: arms moving at rim height stopped on it, 7 picks).
    Off while a pick inside it is planned: the object resting on its floor would hit it."""
    if c is None:
        return
    if on:
        from dojo.skills.precision import height_map
        hm = height_map(c)
        if hm is None:
            return
        sk.coll.heightmaps[c.label] = hm
        sk.coll.margin_of[c.label] = 0.004
        sk.coll.include.add(c.label)
    else:
        sk.coll.include.discard(c.label)


def _fill_drop(sk: Skills, objects: list[str], holder: str, H, holes: list, ignore: tuple, st: dict,
               tries: int = 3) -> list[str]:
    """``FILL_MODE`` "drop": each object picked top-down (``egg_grasps``), lifted over its container's rim, carried as
    held and let go lying over a free hole, its centre ``EGG_DROP_OFFSETS`` off the hole's centre along its long axis
    (``egg_drop_targets``); a grasp that failed is not tried again. Returns the labels that did not go in."""
    failed = []
    plate_top = float(holes[0]["top"])
    for lab in sorted(objects, key=lambda l: float(np.linalg.norm(sk.world.obj(l).pos[:2] - H.pos[:2]))):
        w = sk.refresh()
        if any(in_hole(w.obj(lab), h) for h in holes):
            continue
        ok, bad, nudged = False, [], set()
        for attempt in range(tries + 1):
            w = sk.refresh()
            o = w.obj(lab)
            free = [h for h in holes if _hole_free(w, h, objects, lab)]
            if not free:
                return failed + [lab]
            c = container_of(sk, lab)
            _container_obstacle(sk, c, False)
            lift_z = None                 # the TCP (about the object's centre for these grasps) over the rim
            if c is not None and (prof := wall_profile(c)) is not None:
                lift_z = float((c.R @ np.r_[prof["center"], prof["rim"]] + c.pos)[2]) + EGG_LIFT_OVER
            arms = sorted(ARMS, key=lambda a: float(np.linalg.norm(sk.kin.base(a)[:2, 3] - o.pos[:2])))
            pp = None
            for a in arms:
                gs = [g for g in egg_grasps(sk, lab, a) if g.note not in bad]
                if not gs:
                    continue
                r = sk.plan_pick(lab, a, grasps=gs, lift=0.06, lift_to_z=lift_z, max_eval=60)
                if not isinstance(r, SkillResult):
                    pp = r
                    break
                sk.log(f"[holes] {lab}: {a}: no pick ({r.note[:160]})")
            if pp is None:
                if lab not in nudged and nudge_from_wall(sk, lab) is not None:
                    nudged.add(lab)
                    continue                  # re-plan the pick from where it rolled to
                sk.log(f"[holes] {lab}: no pick")
                break
            sk.log(f"[holes] {lab}: {pp.arm} picks ({pp.grasp.method} {pp.grasp.note}, pre {pp.grasp.pre_dist * 1000:.0f} mm)")
            r = sk._record(sk.run_pick(pp))
            if r.status in ("ended", "step_budget"):
                return failed + [lab]
            if not r.ok:
                bad.append(pp.grasp.note)
                _container_obstacle(sk, c, True)
                sk.bank()
                continue
            _container_obstacle(sk, c, True)
            a = pp.arm
            w = sk.refresh()
            o = w.obj(lab)
            pl, last = None, ""
            for h in free:                 # front holes first (``holes`` order)
                for off, yd, side, T in egg_drop_targets(o, h, plate_top, o.R):
                    q = sk.plan_place(a, T, yaw_sym=1, release_gap=0.0, pre_h=0.05, ignore=ignore, descend_rad=0.03,
                                      settle=2, retreat=0.04)
                    if not isinstance(q, SkillResult):
                        pl = (h, yd, side, q, off)
                        break
                    last = q.note
                if pl:
                    break
            if pl is None and EGG_STAND_FALLBACK:
                # a back hole under the open lid has no room for a lying egg: stood up in the pads (a level pinch turns
                # it about the pads' axis), let go with the fingertips over the body
                for h in free:
                    for T in standing_targets(o, h["center"], h["top"] - 0.010, tilts=(0.0, 25.0)):
                        q = sk.plan_place(a, T, yaw_sym=8, ignore=ignore, ref_top=plate_top + 0.006, servo_ref=holder,
                                          **STAND_PLACE_KW)
                        if not isinstance(q, SkillResult):
                            pl = (h, float("nan"), 0.0, q, float("nan"))      # stood up (no offset / yaw)
                            break
                        last = q.note
                    if pl:
                        break
            if pl is None:
                sk.log(f"[holes] {lab}: no drop plan ({last[:200]})")
                sk.home(gripper=OPEN)
                sk.held = {x: None for x in sk.held}
                break
            h, yd, side, q, off = pl
            rr = sk.run_place(q, tol_xy=0.05, tol_tilt=90.0, tol_z=0.08)
            if rr.status == "ended":
                return failed + [lab]
            sk.ex.hold(4)
            sk.bank()
            if off == off:
                # mode B: the belief's release rule does not know the holder's holes (an Articulation, not a fixture)
                # and puts the egg on the table; the provider's confirm_pose (the head view) decides on the expected one
                sk.confirm_moved(lab, tipped_in_pose(o, h, q.T_obj_target), "dropped into a hole")
            w = sk.refresh()
            o = w.obj(lab)
            if EGG_SLIDE_IN and off == off and (not sk.remeasure or float(o.center_w[2]) > plate_top - EGG_SUNK):
                # lying across the hole (mode A: measured; mode B: the release cannot be checked, so always): slid along
                # its axis until one end is over the hole
                r_s = slide_in(sk, lab, h, q.T_obj_target, ignore, arm=a)
                sk.log(f"[holes] {lab}: slid along its axis ({r_s})")
                w = sk.refresh()
                o = w.obj(lab)
            if EGG_PRESS and off == off and float(o.center_w[2]) > plate_top - EGG_SUNK:
                # still lying across the hole (wedged between the rim and the hole's wall): pressed in from above
                for at_end in (False, True):
                    r_p = press_in(sk, lab, h, ignore, at_end=at_end)
                    sk.log(f"[holes] {lab} lay across the hole (centre {1000 * (float(o.center_w[2]) - plate_top):+.0f} mm "
                           f"over the plate): pressed {'its high end' if at_end else 'its middle'} ({r_p})")
                    w = sk.refresh()
                    o = w.obj(lab)
                    if float(o.center_w[2]) <= plate_top - EGG_SUNK:
                        break
            hit = next((x for x in holes if in_hole(o, x)), None)
            if hit is not None and off == off and float(o.center_w[2]) > plate_top - EGG_SUNK:
                hit = None                    # lying across it, not in (the checker wants its centre lower)
            how = "stood up" if off != off else f"offset {off * 1000:.0f} mm, yaw {yd:.0f}, side {side:+.0f}"
            sk.log(f"[holes] {lab} dropped over the hole at {np.round(h['center'], 3).tolist()} ({how}; {rr.status}): "
                   f"{'in a hole' if hit else 'NOT in a hole'}, pos {np.round(o.pos, 3).tolist()} tilt {o.tilt((0, 0, 1)):.0f}")
            if hit is not None:
                ok = True
                st["done"].append(lab)
                break
            if any(sk.held.values()):
                sk.home(gripper=OPEN)
                sk.held = {x: None for x in sk.held}
        if not ok:
            failed.append(lab)
    return failed


def fill_holes(sk: Skills, objects: list[str], holder: str, lid_tag: str | None = None, close: bool = True,
               tries: int = 3, hold_below: float = 0.010, carry_rad: float = 0.05, mode: str | None = None) -> dict:
    """Put each object into a free hole of ``holder``'s body, back holes first (nearest the lid hinge); bank after
    each. Then, with ``close``, swing the lid shut (``swing_closed``). Objects lying in a round container are grasped
    inside its wall. ``mode`` (default ``FILL_MODE``): "drop" carries each object lying as picked and lets it go low over
    a hole, off its centre, so it tips in (``_fill_drop``); "stand" stands it upright in the pads first and lets it
    drop in with the fingertips above the body."""
    w = sk.refresh()
    H = w.obj(holder)
    lj = lid_joint(H, lid_tag)
    body = lj[3] if lj else None
    if lj:
        sk.coll.slab_links.add(holder)
    for l in objects:                     # identical objects touch in their container: their boxes without a margin
        sk.coll.margin_of.setdefault(l, 0.0)
    holes = body_holes(H, body)
    if not holes:
        return {"ok": False, "stage": "holes", "note": f"no holes found in {holder}"}
    hinge_w = H.R @ np.asarray(lj[1].get("origin", H.asset.center), float) + H.pos if lj else H.center_w
    # the holes farthest from the lid's hinge first: the open lid stands behind the back holes and blocks most carries
    # to them (dry runs: front holes plan, back ones rarely)
    holes.sort(key=lambda h: -float(np.linalg.norm(h["center"] - hinge_w[:2])))
    st = w.notes.setdefault("fill_holes", {"done": []})
    mode = mode or FILL_MODE
    sk.log(f"[holes] {holder}: {len(holes)} holes r {[round(h['radius'] * 1000) for h in holes]} mm, "
           f"top {holes[0]['top']:.3f} floor {holes[0]['floor']:.3f}; lid {lj[0] if lj else None}; mode {mode}")
    ignore = (f"{holder}#{body}",) if body else (holder,)
    if mode == "drop":
        from dojo.skills.motion import mesh_parts
        for l in objects:
            # neighbours as small boxes from their mesh (an egg's box corners stand 8 mm off its shell and closed every
            # finger corridor between two touching eggs), pushed aside up to EGG_PUSH_OK
            o = w.obj(l)
            if o.asset.mesh is not None:
                sk.coll.parts[l] = mesh_parts(*o.asset.mesh, cell=0.012)
                sk.coll.margin_of[l] = EGG_PUSH_OK
        failed = _fill_drop(sk, objects, holder, H, holes, ignore, st, tries=tries)
        res = {"ok": not failed, "stage": "holes", "note": f"in: {st['done']}, failed: {failed}"}
        if close and lj is not None and not failed:
            r = swing_closed(sk, holder, lid_tag)
            res["lid"] = r.note
            if not r.ok:
                res["ok"] = False
                res["stage"] = "lid"
            sk.bank()
        return res
    failed = []
    for lab in sorted(objects, key=lambda l: float(np.linalg.norm(w.obj(l).pos[:2] - H.pos[:2]))):
        w = sk.refresh()
        if any(in_hole(w.obj(lab), h) for h in holes):
            continue
        ok = False
        for attempt in range(tries):
            w = sk.refresh()
            free = [h for h in holes if _hole_free(w, h, objects, lab)]
            if not free:
                return {"ok": False, "stage": "holes", "note": "no free hole left"}
            o = w.obj(lab)
            plans = SkillResult(False, "transfer", "ik_fail", 0, None, note="no hole")
            arms = sorted(ARMS, key=lambda a: float(np.linalg.norm(sk.kin.base(a)[:2, 3] - o.pos[:2])))
            # stood up with one end down, the pads hold it below its widest section (twin: held at the middle it slid
            # out between the pads as the hand turned it up, 2/2); one hole, one end and one arm per plan
            combos = [(h, up, a) for h in free[:2] for up in (1.0, -1.0) for a in arms]
            for h, up, a in combos:
                targets = standing_targets(o, h["center"], h["top"] - 0.010, tilts=(0.0, 25.0), ups=(up,))
                plans = sk.plan_transfer(lab, targets, arms=[a], ignore=ignore, yaw_sym=8,
                                         grasps=lambda a, lab=lab, up=up: grasps_inside(sk, lab, a,
                                                                                        along=-up * hold_below),
                                         place_kw={"servo": True, "servo_ref": holder,
                                                   "ref_top": holes[0]["top"] + 0.006, "fingers_out": True,
                                                   "probe": False, "hover_clear": 0.008, "descend_rad": 0.03,
                                                   "pre_h": 0.05, "approach_rad": carry_rad, "ease": 6})
                if not isinstance(plans, SkillResult):
                    break
            if isinstance(plans, SkillResult):
                sk.log(f"[holes] {lab}: no transfer ({plans.note[:200]})")
                if attempt == tries - 1 or not _to_table(sk, lab, H):
                    break
                continue
            r = sk.run_transfer(*plans, tol_xy=0.02, tol_tilt=40.0, tol_z=0.06)
            if r.status == "ended":
                return {"ok": False, "stage": "ended"}
            sk.bank()
            w = sk.refresh()
            o = w.obj(lab)
            hit = next((h for h in holes if in_hole(o, h)), None)
            sk.log(f"[holes] {lab} ({r.status}): {'in a hole' if hit else 'NOT in a hole'}, "
                   f"pos {np.round(o.pos, 3).tolist()} tilt {o.tilt((0, 0, 1)):.0f}")
            if hit is not None:
                ok = True
                st["done"].append(lab)
                break
            if any(sk.held.values()):
                sk.home(gripper=OPEN)
                sk.held = {a: None for a in sk.held}
        if not ok:
            failed.append(lab)
    res = {"ok": not failed, "stage": "holes", "note": f"in: {st['done']}, failed: {failed}"}
    # the lid only counts with every object in (the checker's last stage); a half-filled holder keeps its open lid
    if close and lj is not None and not failed:
        r = swing_closed(sk, holder, lid_tag)
        res["lid"] = r.note
        if not r.ok:
            res["ok"] = False
            res["stage"] = "lid"
        sk.bank()
    return res


# ---- the lid


def swing_closed(sk: Skills, label: str, tag: str | None = None, ratio: float = 0.95, push_deg: float = 75.0,
                 heights=(0.75, 0.6, 0.9), tries: int = 2) -> SkillResult:
    """Swing an articulated object's lid (the revolute joint ``tag`` names) to ``ratio`` of its range: closed fingertips
    against the lid's outer face at a fraction of its height, moved along the lid's arc about the hinge for
    ``push_deg`` (gravity takes it the rest of the way once it is past the hinge), back out; checked on the joint, and
    a second, steeper push if it stopped short."""
    w = sk.refresh()
    o = w.obj(label)
    lj = lid_joint(o, tag)
    if lj is None or not o.asset.links or lj[2] not in o.asset.links:
        return sk._record(SkillResult(False, "swing_closed", "logic", 0, None, note=f"{label}: no lid joint"))
    jn, J, lid, body = lj
    lo, hi = float(J["lower"]), float(J["upper"])
    deg = max(abs(lo), abs(hi)) > 2 * math.pi + 0.1
    notes = []
    for attempt in range(tries):
        w = sk.refresh()
        o = w.obj(label)
        cur = joint_angle(o, J)
        rr = (cur - lo) / (hi - lo)
        sk.log(f"[lid] {label} {jn}: measured {None if o.joints is None else np.round(np.ravel(o.joints), 4).tolist()}, "
               f"range {lo}..{hi}, angle {cur:.1f}, ratio {rr:.2f}")
        if rr >= ratio:
            return sk._record(SkillResult(True, "swing_closed", "ok", 0, None, note=f"lid at {rr:.2f}"))
        to_go = (lo + ratio * (hi - lo) - cur)
        sgn = 1.0 if to_go > 0 else -1.0
        cur_rad = math.radians(cur) if deg else cur
        rest_rad = math.radians(float(J.get("position") or lo)) if deg else float(J.get("position") or lo)
        axis = np.asarray(J["axis"], float)
        origin = np.asarray(J["origin"], float)
        # the lid's points as authored, turned to the measured angle (object frame)
        P = np.asarray(o.asset.links[lid]["points"], float)
        R_now = _rot(axis, cur_rad - rest_rad)
        P = (P - origin) @ R_now.T + origin
        # the face that leads the motion: the lid points whose arc velocity points against the swing direction most
        vel = np.cross(sgn * axis, P - origin)
        rad = np.linalg.norm(np.cross(axis, P - origin), axis=1)
        r_max = float(rad.max())
        done = False
        for frac in heights:
            band = np.abs(rad - frac * r_max) < 0.01
            if not band.any():
                continue
            Q, Vq = P[band], vel[band]
            # the trailing face: the band's points furthest against the swing direction
            u = Vq.mean(0)
            u = u / max(float(np.linalg.norm(u)), 1e-9)
            back = Q[int(np.argmin(Q @ u))]
            mid = float(np.mean(P @ axis))                    # the middle of the lid's width (along the axis)
            back = back + axis * (mid - float(back @ axis))
            n_out = -u
            p0 = back + n_out * 0.008                         # fingertip 8 mm behind the face
            steps = [0.0] + [math.radians(d) for d in np.arange(10.0, push_deg + 1e-6, 10.0)]
            path = [(_rot(axis, sgn * d) @ (p0 - origin) + origin) for d in steps]
            path_w = [o.R @ p + o.pos for p in path]
            plan = _arc_plan(sk, label, path_w, n_out_w=o.R @ n_out)
            if plan is None:
                notes.append(f"h {frac:.2f}: no arc plan")
                continue
            arm, pl = plan
            other = "right" if arm == "left" else "left"
            try:
                if not sk.clear_of(pl, ArmPlan(other, sk.ex.q[other].copy(), sk.ex.g[other]), min_dist=0.12):
                    sk.ex.run({other: sk.plan_home(other)}, name=f"clear {other}")
                sk.ex.run({arm: pl}, name=f"swing {label}")
                sk.ex.hold(8)
            except BudgetExceeded as e:
                return sk._record(SkillResult(False, "swing_closed", "step_budget", 0, arm, note=str(e)))
            except EpisodeEnded as e:
                return sk._record(SkillResult(True, "swing_closed", "ended", 0, arm, note=str(e)))
            o2 = sk.refresh().obj(label)
            notes.append(f"{arm} h {frac:.2f}: lid {joint_ratio(o2, J):.2f}")
            done = True
            break
        if not done:
            break
        push_deg = 90.0
    o2 = sk.refresh().obj(label)
    rr = joint_ratio(o2, J)
    return sk._record(SkillResult(rr >= ratio, "swing_closed", "ok" if rr >= ratio else "place_miss", 0, None,
                                  note=f"lid at {rr:.2f}; " + "; ".join(notes[-4:])))


def _arc_plan(sk: Skills, label: str, path_w: list, n_out_w: np.ndarray):
    """Closed-gripper plan that puts the fingertip on ``path_w[0]`` (coming down from above the lid) and moves it
    along the path. The hand points down, tilted up to 50 deg toward the path's motion."""
    from dojo.skills.kinematics import tilted_R
    p0 = np.asarray(path_w[0])
    o = sk.world.obj(label)
    top = float(o.top_z)
    arms = sorted(ARMS, key=lambda a: float(np.linalg.norm(sk.kin.base(a)[:2, 3] - p0[:2])))
    for a in arms:
        base = sk.kin.base(a)[:3, 3]
        for tilt in (40.0, 25.0, 55.0, 10.0):
            for yaw in (0.0, math.pi / 2):
                # approach tilted so the hand leans over the lid from the arm's side: heading from the base
                h_arm = math.atan2(p0[1] - base[1], p0[0] - base[0])
                Rg = tilted_R(h_arm + math.pi / 2 + yaw, 90.0 - tilt, h_arm)
                x = Rg[:, 0]

                def tcp_at(p):
                    return T_from(np.asarray(p) - x * TIP_BEYOND_TCP, Rg)
                T0 = tcp_at(path_w[0])
                T_hi = T0.copy()
                T_hi[2, 3] = max(T0[2, 3], top + 0.05) + 0.02
                to_hi = sk.motion.to_pose(a, sk.ex.q[a], T_hi, FAST, "swing_approach", gripper=CLOSED)
                if not to_hi.ok:
                    continue
                dn = sk.motion.line(a, to_hi.q_end, T0, MEDIUM, "swing_lower", gripper=CLOSED, check=False)
                if not dn.ok:
                    continue
                pl = ArmPlan(a, sk.ex.q[a].copy(), sk.ex.g[a]).grip(CLOSED, 0, "close_for_swing")
                pl.move(to_hi, "swing_approach").move(dn, "swing_lower").hold(2, "settle")
                q = dn.q_end
                ok = True
                for p in path_w[1:]:
                    mv = sk.motion.line(a, q, tcp_at(p), 0.04, "swing_arc", gripper=CLOSED, check=False)
                    if not mv.ok:
                        ok = False
                        break
                    pl.move(mv, "swing_arc")
                    q = mv.q_end
                if not ok:
                    continue
                Tq = sk.kin.tcp(a, q)
                Tu = Tq.copy()
                Tu[2, 3] += 0.06
                up = sk.motion.line(a, q, Tu, MEDIUM, "swing_retreat", gripper=CLOSED, check=False)
                if up.segments:
                    pl.move(up, "swing_retreat")
                return a, pl
    return None


__all__ = ["fill_holes", "swing_closed", "body_holes", "standing_targets", "wall_profile", "grasps_inside",
           "lid_joint", "joint_ratio"]
