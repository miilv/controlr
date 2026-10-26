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


def fill_holes(sk: Skills, objects: list[str], holder: str, lid_tag: str | None = None, close: bool = True,
               tries: int = 3, hold_below: float = 0.010, carry_rad: float = 0.05) -> dict:
    """Stand each object upright over a free hole of ``holder``'s body and let it drop in (the fingertips stay above
    the body), back holes first (nearest the lid hinge); bank after each. Then, with ``close``, swing the lid shut
    (``swing_closed``). Objects lying in a round container are grasped inside its wall."""
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
    sk.log(f"[holes] {holder}: {len(holes)} holes r {[round(h['radius'] * 1000) for h in holes]} mm, "
           f"top {holes[0]['top']:.3f} floor {holes[0]['floor']:.3f}; lid {lj[0] if lj else None}")
    ignore = (f"{holder}#{body}",) if body else (holder,)
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
