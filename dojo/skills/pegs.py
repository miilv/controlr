"""Pieces with a hole threaded onto a base's pegs (``stack_on_pegs``).

The geometry comes from the meshes: a piece's through hole (``through_hole``: the enclosed empty region of a vertical
depth map of the piece in its "up" pose), a peg's axis and top from the base's support frame plus the base mesh (the
rounded top above the frame, the plate around the peg). A round hole on a round peg turns freely, so every turn of the
piece about its hole is a target: the yaw is left to the grasp search (``plan_transfer(yaw_sym, yaw_about)``).
"""

from __future__ import annotations

import math

import numpy as np

from dojo.skills.geometry import T_from, quat_to_R, rot_z
from dojo.skills.kinematics import ARMS
from dojo.skills.skills import SkillResult, Skills
from dojo.skills.targets import level_R

PEG_YAWS = 12                 # turns of a piece about its hole tried per place (every 30 deg)
THREAD = 0.004                # m: the piece is lowered this far onto the peg (below its top) before it is let go
SEAT_TOL = 0.0025             # m: a piece held higher than this above its threaded target rests on the peg's top
GUARD_SEATED_PEG = 0.0015     # m: the same for the guarded tail (mode B: no measured object, the arm's lag is the sensor)
GUARD_LAG_PEG = 0.0012        # m: its contact threshold (free descent lags 0.3-0.5 mm; a piece stopped by the top: 1.7)
PEG_CARRY = 0.10              # rad/step: the carry to the peg (at 0.14 the arm lagged 7 mm and a piece tilted 6 deg in the pads)
PEG_RETREAT = 0.02            # m: up after the release (the far peg is at the edge of reach: 3 and 5 cm failed the IK)
RELAY_DY = 0.12               # m: relay spots this far in front of the base (towards the robot; both arms reach there)
BIAS_DIR = (0.0, -1.0)        # world xy: the side of the peg every piece is let go against (``side_bias``)
PLACED_MARGIN = 0.002         # m: collision margin of pieces already on a peg (pegs 65 mm apart, pieces 58 mm wide)

_HOLES: dict = {}


def through_hole(o, res: float = 0.0005) -> tuple[np.ndarray, float] | None:
    """(point on the hole's axis in the object frame, hole radius) of a piece in its "up" place pose: the empty cells of
    a vertical depth map of the mesh that the outside cannot reach (flood fill from the border), the component nearest
    the footprint's centre. None without a mesh or a hole."""
    key = getattr(o.asset, "key", None)
    if key in _HOLES:
        return _HOLES[key]
    out = None
    if o.asset.mesh is not None:
        from dojo.skills.insertion import _ray_down_grid
        V, F = o.asset.mesh
        q = o.asset.place.get("up")
        R_up = quat_to_R(np.asarray(q, float)) if q is not None else np.eye(3)
        W = V @ R_up.T
        lo, hi = W.min(0), W.max(0)
        xs = np.arange(lo[0] - 2 * res, hi[0] + 2 * res, res)
        ys = np.arange(lo[1] - 2 * res, hi[1] + 2 * res, res)
        Z = _ray_down_grid(W, F, xs, ys, float(hi[2]) + 0.01)
        empty = ~np.isfinite(Z)
        outside = np.zeros_like(empty)
        stack = [(i, j) for i in range(empty.shape[0]) for j in (0, empty.shape[1] - 1) if empty[i, j]]
        stack += [(i, j) for j in range(empty.shape[1]) for i in (0, empty.shape[0] - 1) if empty[i, j]]
        while stack:
            i, j = stack.pop()
            if outside[i, j] or not empty[i, j]:
                continue
            outside[i, j] = True
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < empty.shape[0] and 0 <= b < empty.shape[1] and empty[a, b] and not outside[a, b]:
                    stack.append((a, b))
        inner = empty & ~outside
        if inner.sum() >= 12:
            # connected components of the enclosed cells; the one nearest the footprint centre
            lab = np.zeros(inner.shape, int)
            n = 0
            comps = []
            for i0, j0 in zip(*np.nonzero(inner)):
                if lab[i0, j0]:
                    continue
                n += 1
                cells, st = [], [(i0, j0)]
                lab[i0, j0] = n
                while st:
                    i, j = st.pop()
                    cells.append((i, j))
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        a, b = i + di, j + dj
                        if 0 <= a < inner.shape[0] and 0 <= b < inner.shape[1] and inner[a, b] and not lab[a, b]:
                            lab[a, b] = n
                            st.append((a, b))
                comps.append(np.array(cells))
            ctr = (lo[:2] + hi[:2]) / 2
            best = None
            for c in comps:
                if len(c) < 12:
                    continue
                p = np.array([xs[c[:, 1]].mean(), ys[c[:, 0]].mean()])
                d = float(np.linalg.norm(p - ctr))
                if best is None or d < best[0]:
                    best = (d, p, math.sqrt(len(c) * res * res / math.pi))
            if best is not None:
                p_up = np.array([best[1][0], best[1][1], float(lo[2])])
                out = (R_up.T @ p_up, float(best[2]))
    _HOLES[key] = out
    return out


def peg_geometry(b, tag: str) -> dict:
    """A peg of base ``b`` (support frame ``tag``): its axis point and direction (world), the frame's height, the top of
    its rounded end from the mesh (the frame sits up to 2 mm under it) and the plate's height around it."""
    from dojo.skills.library import _support_raw
    raw = np.asarray(_support_raw(b, tag)[0], float)
    F = T_from(raw[:3], quat=raw[3:7] / np.linalg.norm(raw[3:7]))
    Fw = b.T @ F
    p, ax = Fw[:3, 3], Fw[:3, 2]
    top, plate, radius = float(p[2]), None, None
    if b.asset.mesh is not None:
        V = b.asset.mesh[0]
        d = np.hypot(V[:, 0] - raw[0], V[:, 1] - raw[1])
        shaft = V[d < 0.0065]
        if len(shaft):
            top = float(b.T[2, 2] * shaft[:, 2].max() + b.T[2, 3])
        ring = V[(d > 0.008) & (d < 0.02)]
        z_plate = float(np.percentile(ring[:, 2], 90)) if len(ring) else float(V[:, 2].min())
        if len(ring):
            plate = float(b.T[2, 2] * z_plate + b.T[2, 3])
        band = (d < 0.0075) & (V[:, 2] > z_plate + 0.004) & (V[:, 2] < float(raw[2]) - 0.003)
        if band.sum() >= 8:
            radius = float(np.median(d[band]))
    return {"tag": tag, "p": p, "axis": ax, "frame_z": float(p[2]), "top": top, "plate": plate, "radius": radius}


def side_bias(o, peg: dict, keep: float = 0.00015) -> float:
    """How far (m) the piece's hole is put off the peg's axis, all pieces of a peg towards the same side: the hole's
    clearance on the peg less ``keep``. Let go centred, a piece ends anywhere within the clearance (0.3-0.65 mm here),
    and two pieces on one peg differed by up to 1.14 mm against the 1 mm the stack is judged by (twin); against one
    side, they line up."""
    h = through_hole(o)
    if h is None or peg.get("radius") is None:
        return 0.0
    return max(0.0, float(h[1]) - float(peg["radius"]) - keep)


def piece_target(sk: Skills, label: str, peg: dict, base_yaw: float, bottom_z: float,
                 bias=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """(object pose, hole point in the object frame): the piece flat (its "up" pose, turned with the base), its hole on
    the peg's axis (shifted by ``bias``, world xy), its lowest point at ``bottom_z``."""
    o = sk.world.obj(label)
    q = o.asset.place.get("up")
    R = rot_z(base_yaw) @ (quat_to_R(np.asarray(q, float)) if q is not None else level_R(o.R))
    h = through_hole(o)
    hole = h[0] if h is not None else np.zeros(3)
    if o.asset.mesh is not None:
        low = float((o.asset.mesh[0] @ R.T)[:, 2].min())
    else:
        ext = np.abs(R) @ o.asset.extents
        low = float((R @ o.asset.center)[2] - ext[2] / 2)
    p_hole_w = R @ hole
    pos = np.array([peg["p"][0] + bias[0] - p_hole_w[0], peg["p"][1] + bias[1] - p_hole_w[1], bottom_z - low])
    return T_from(pos, R), hole


def on_peg(sk: Skills, label: str, peg: dict, tol: float = 0.003) -> bool:
    """Is the piece's hole on the peg's axis, below the peg's top (threaded)?"""
    o = sk.world.obj(label)
    h = through_hole(o)
    hole = h[0] if h is not None else np.zeros(3)
    hw = o.R @ hole + o.pos
    if float(np.linalg.norm(hw[:2] - peg["p"][:2])) > tol:
        return False
    low = (o.asset.mesh[0] @ o.R.T + o.pos)[:, 2].min() if o.asset.mesh is not None else o.bottom_z
    return bool(float(low) < peg["top"] - 0.002)


def centred_grasps(sk: Skills, label: str, arm: str, tilts=(0.0, 25.0, -25.0, 40.0, -40.0), off: float = 0.006) -> list:
    """``plan_transfer``'s usual grasps of a piece (box + the category's), only those centred on it: a round piece
    pinched 16 mm off its centre turned out of the pads (twin: 14-21 mm in-hand drift, one dropped on the carry)."""
    from dojo.skills import grasps as G
    o = sk.world.obj(label)
    c = o.center_w
    cands = G.box_grasps(o, sk.kin, arm, tilts=tuple(tilts)) + G.candidates(o, sk.kin, arm)
    return [g for g in cands if float(np.linalg.norm(g.T[:2, 3] - c[:2])) < off]


def stack_on_pegs(sk: Skills, groups: list[list[str]], base: str, pegs: list[str], drop: float = 0.003,
                  thread: float = THREAD) -> dict:
    """Thread pieces onto a base's pegs: group i (identical pieces) onto peg ``pegs[i]`` (a support frame of ``base``),
    one at a time, flat (the piece's "up" pose), its hole (``through_hole``, from the mesh) on the peg's axis, any turn
    about the hole (``PEG_YAWS``). The shortest peg is filled first: every later piece then goes in above the finished
    neighbours (tallest first, the next peg's pieces were let go level with the stack beside them and found no
    collision-free grasp: 0 grasp-place pairs for 3-7 of 10 pieces, twin). Each piece is lowered ``thread`` onto the
    peg's top through the closed-loop tail (``servo``: measured, sag-corrected, probing around the axis when it rests
    on the top) and let go there; it slides down the peg. ``thread=0``: let go ``drop`` above the top instead. Grippers
    open after each piece (per-peg credit); no in-air handovers (a ring handed over in the air fell 2/2)."""
    from dojo.skills.library import place_oriented
    w = sk.refresh()
    b = w.obj(base)
    sk.coll.include.add(base)
    geo = {tag: peg_geometry(b, tag) for tag in pegs}
    order = sorted(range(len(groups)), key=lambda i: geo[pegs[i]]["top"])
    done, failed = [], []
    for i in order:
        grp, tag = groups[i], pegs[i]
        peg = geo[tag]
        for lab in grp:
            if lab not in sk.world.objects:
                continue
            sk.refresh()
            if on_peg(sk, lab, peg):
                done.append(lab)
                continue
            z_bot = peg["top"] - thread if thread > 0 else peg["top"] + drop
            bias = side_bias(sk.world.obj(lab), peg) * np.asarray(BIAS_DIR)
            T, hole = piece_target(sk, lab, peg, b.yaw(), z_bot, bias=bias)
            pkw = {"pre_h": 0.02, "descend_rad": 0.03, "release_gap": 0.0, "settle": 2, "servo": True, "retreat": PEG_RETREAT,
                   "servo_ref": base, "ref_top": peg["top"], "seat_tol": SEAT_TOL if thread > 0 else 0.004,
                   "probe": thread > 0, "guard_seated": GUARD_SEATED_PEG, "guard_lag": GUARD_LAG_PEG,
                   "approach_rad": PEG_CARRY}
            r = place_oriented(sk, lab, [T], yaw_sym=PEG_YAWS, tol_xy=0.01, tol_tilt=15.0, tol_z=0.08, ignore=(base,),
                               place_kw=pkw, air_handover=False, spot_avoid=(base,),
                               relay_near=(float(b.center_w[0]), float(b.center_w[1]) - RELAY_DY),
                               transfer_kw={"lift_to_pre": True, "yaw_about": hole,
                                            "grasps": (lambda a, lab=lab: centred_grasps(sk, lab, a))})
            for a in ARMS:
                if sk.held[a] is not None:
                    sk._record(sk.regrasp(a))
            sk.bank()
            ok = on_peg(sk, lab, peg)
            sk._record(SkillResult(ok, "peg", "ok" if ok else "place_miss", 0, None,
                                   note=f"{lab} {'on' if ok else 'not on'} {tag}"))
            (done if ok else failed).append(lab)
            if ok:
                sk.coll.margin_of[lab] = PLACED_MARGIN
            if r.status in ("step_budget", "ended"):
                return {"ok": False, "stage": f"peg {tag}", "result": r}
    if failed:
        return {"ok": False, "stage": "pegs", "note": f"done {done}, failed {failed}"}
    return {"ok": True, "stage": "pegs"}
