"""Planned flat-object pushes (stream ORG2; all switches default off, ``Skills`` attributes mirrored by ``RuntimeConfig``).

Three independent capabilities of ``library.push_to``:

* ``Skills.push_contact`` "pair": the stroke pushes with the gripper OPEN and its jaw line across the stroke, so the two
  fingertips (``OPEN_GAP`` apart) meet the trailing face at two points. A two-point push squares the face up against the
  jaw line instead of turning the object about a single contact (G3 dev: single closed-finger strokes of a 37 cm flat slab
  along its long face turned it 10-14 deg per 80 mm stroke, always the same way). With a support given, the strokes run
  along the support's horizontal axes, so the squared-up object ends aligned with the support.
* ``Skills.push_plan`` "sequence": before each stroke, the whole remaining stroke sequence is planned kinematically (the
  object moved virtually by each planned stroke) for both axis orders, and the first stroke of the first order whose
  every stroke plans is executed. The legacy rule pushes the axis with most to go first; one axis's strokes can leave the
  object where the other axis's start pose is out of reach or in a fixture (dev: one axis first left the slab's end
  beside a tall fixture where no stroke along the other axis fits).
* ``Skills.push_goal`` "clear": the goal is the point of the cover region (the centres at which the object, aligned with
  the support, covers or lies inside it) with the most clearance from other objects' footprints and from ``keep_clear``
  discs (the caller's pending place sites), at least ``GOAL_EDGE`` inside the region where it can be.

Nothing here reads a task name, a layout id or a checker: geometry from the world and the asset catalogue only.
"""
from __future__ import annotations

import math

import numpy as np

from dojo.skills.geometry import T_from
from dojo.skills.kinematics import ARMS, TIP_BEYOND_TCP, down_R, tilted_R
from dojo.skills.motion import FAST, MEDIUM

OPEN_GAP = 0.088          # m: the fingertips' spread with the gripper open (grasps.MAX_GAP)
STANDOFF = 0.02           # m: the stroke starts this far behind the trailing face
GOAL_EDGE = 0.015         # m: the "clear" goal stays this far inside the cover region where the region allows
GOAL_CLEAR = 0.06         # m: clearance beyond which more does not count (then: nearest the region's middle)
KEEP_MARGIN = 0.01        # m: added to every keep-clear disc
OVERSHOOT = 0.85          # strokes longer than 30 mm are cut to this fraction of what is left (dev: 80 mm planned
                          # strokes moved a flat slab 90-95 mm); the re-read after each stroke takes the rest
PAIR_GRIP = 1.0           # the gripper command of a pair stroke (open)


def support_axes(sup) -> list[np.ndarray]:
    """The support's two horizontal body axes in the world (unit 2-vectors)."""
    out = []
    for k in range(3):
        a = np.asarray(sup.R[:, k], float).copy()
        a[2] = 0.0
        if np.linalg.norm(a) > 0.5:
            out.append(a[:2] / np.linalg.norm(a[:2]))
    return out[:2]


def body_axes(o) -> list[np.ndarray]:
    return support_axes(o)


def footprint(o, centre=None, yaw: float | None = None) -> np.ndarray:
    """The object's bbox footprint (4 x 2 world corners) at ``centre`` (bbox centre xy, default where it is), turned
    to world yaw ``yaw`` about the vertical (default its own)."""
    R = np.asarray(o.R, float)
    if yaw is not None:
        cy, sy = math.cos(yaw - _yaw(o)), math.sin(yaw - _yaw(o))
        R = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]) @ R
    ext = np.asarray(o.asset.extents, float)
    up = int(np.argmax(np.abs(R[2, :])))
    i, j = [k for k in range(3) if k != up]
    c = np.asarray(o.center_w[:2] if centre is None else centre, float)
    out = []
    for si, sj in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        p = R[:2, i] * si * ext[i] / 2 + R[:2, j] * sj * ext[j] / 2
        out.append(c + p)
    return np.asarray(out)


def _yaw(o) -> float:
    R = np.asarray(o.R, float)
    up = int(np.argmax(np.abs(R[2, :])))
    k = [m for m in range(3) if m != up][0]
    return math.atan2(R[1, k], R[0, k])


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _inside(p, poly) -> bool:
    s = None
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(c) < 1e-12:
            continue
        if s is None:
            s = c > 0
        elif s != (c > 0):
            return False
    return True


def poly_dist(P, Q) -> float:
    """Distance between two convex polygons (0 when they overlap)."""
    if any(_inside(p, Q) for p in P) or any(_inside(q, P) for q in Q):
        return 0.0
    d = min(_seg_dist(p, Q[k], Q[(k + 1) % len(Q)]) for p in P for k in range(len(Q)))
    return min(d, min(_seg_dist(q, P[k], P[(k + 1) % len(P)]) for q in Q for k in range(len(P))))


def aligned_yaw(o, sup) -> float:
    """The world yaw of ``o`` turned to the support's axis nearest its own (the yaw a pair push squares it to)."""
    y0, ys = _yaw(o), _yaw(sup)
    best = min((ys + k * math.pi / 2 for k in range(-4, 5)), key=lambda y: abs(math.remainder(y - y0, 2 * math.pi)))
    return math.remainder(best, 2 * math.pi)


def cover_box(o, sup, yaw: float) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """The cover region of ``o``'s centre in the support's frame, ``o`` turned to ``yaw`` (aligned with the support):
    per support axis, the range where ``o``'s extent covers the support's (``o`` the larger) or lies inside it, 5 mm in.
    Returns (lo, hi, axes) with ``axes`` the support's world axes (2-vectors) and lo / hi offsets from its centre."""
    from dojo.skills.library import COVER_MARGIN
    axes = support_axes(sup)
    fp = footprint(o, centre=np.zeros(2), yaw=yaw)
    half_o = np.array([float(np.max(fp @ a)) for a in axes])
    sfp = footprint(sup, centre=np.zeros(2))
    half_s = np.array([float(np.max(sfp @ a)) for a in axes])
    lo, hi = np.empty(2), np.empty(2)
    for k in range(2):
        if half_o[k] >= half_s[k]:
            lo[k], hi[k] = half_s[k] - half_o[k] + COVER_MARGIN, -half_s[k] + half_o[k] - COVER_MARGIN
        else:
            lo[k], hi[k] = -half_s[k] + half_o[k] + COVER_MARGIN, half_s[k] - half_o[k] - COVER_MARGIN
        if lo[k] > hi[k]:
            lo[k] = hi[k] = 0.0
    return lo, hi, axes


def clear_goal(sk, label: str, support: str, keep_clear=()) -> tuple[np.ndarray, dict]:
    """``push_goal`` "clear": the world xy goal for ``label``'s centre and a diagnostic dict."""
    w = sk.world
    o, sup = w.obj(label), w.obj(support)
    yaw = aligned_yaw(o, sup)
    lo, hi, axes = cover_box(o, sup, yaw)
    sc = np.asarray(sup.center_w[:2], float)
    A = np.stack(axes, axis=1)                          # 2 x 2: support offsets -> world
    others = []
    for lab, ob in w.objects.items():
        if lab in (label, support) or lab in sk.held.values():
            continue
        if float(ob.top_z - ob.bottom_z) < 0.003:       # flat markers (a stand plate) are no obstacle to a slide
            continue
        others.append(footprint(ob))
    discs = [(np.asarray(c, float)[:2], float(r) + KEEP_MARGIN) for c, r in keep_clear]
    lo_e = np.where(hi - lo > 2 * GOAL_EDGE, lo + GOAL_EDGE, (lo + hi) / 2)
    hi_e = np.where(hi - lo > 2 * GOAL_EDGE, hi - GOAL_EDGE, (lo + hi) / 2)
    mid = (lo_e + hi_e) / 2
    best, best_s, best_clear = None, None, None
    n = 13
    for s0 in np.linspace(lo_e[0], hi_e[0], n):
        for s1 in np.linspace(lo_e[1], hi_e[1], n):
            off = np.array([s0, s1])
            p = sc + A @ off
            fp = footprint(o, centre=p, yaw=yaw)
            clr = GOAL_CLEAR
            for Q in others:
                clr = min(clr, poly_dist(fp, Q))
            for c, r in discs:
                inside = _inside(c, fp)
                d = 0.0 if inside else min(_seg_dist(c, fp[k], fp[(k + 1) % 4]) for k in range(4))
                clr = min(clr, d - r)
            score = (round(clr, 3), -float(np.linalg.norm(off - mid)))
            if best_s is None or score > best_s:
                best, best_s, best_clear = p, score, clr
    diag = {"yaw_deg": round(math.degrees(yaw), 1), "clearance_mm": round(1000 * float(best_clear), 1),
            "region_lo": np.round(lo, 3).tolist(), "region_hi": np.round(hi, 3).tolist(),
            "goal": np.round(best, 3).tolist(), "keep_clear": len(discs), "others": len(others)}
    return best, diag


def covers(o, sup, centre=None) -> bool:
    """Is ``o`` (as it is, or moved to ``centre``) covering / inside ``sup`` (the library's rotated-footprint test)?"""
    from dojo.skills.library import cover_goal_obb
    if centre is None:
        return cover_goal_obb(o, sup)[0]
    saved = o.pos.copy()
    try:
        o.pos = o.pos + np.r_[np.asarray(centre, float) - o.center_w[:2], 0.0]
        return cover_goal_obb(o, sup)[0]
    finally:
        o.pos = saved


def trail_dist(o, u, centre=None) -> float:
    """How far the object's footprint extends behind ``centre`` against unit ``u`` (2-D)."""
    fp = footprint(o, centre=np.zeros(2))
    return float(np.max(fp @ (-np.asarray(u, float))))


def pair_stroke(sk, o, label: str, u, step: float, centre=None):
    """``push_contact`` "pair": one stroke of ``o`` along world unit ``u`` by ``step`` with the gripper open and its
    jaw line across ``u`` (two fingertips on the trailing face), at the face's mid height (from the mesh). The centre of
    the face's contact line lies on the line through the object's centre along ``u``. An ``ArmPlan`` or None."""
    from dojo.skills.executor import ArmPlan
    from dojo.skills.library import _face_height
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    c = o.center_w[:2] if centre is None else np.asarray(centre, float)
    d_tr = trail_dist(o, u)
    fh = _face_height(o, u)
    hgt = float(o.top_z - o.bottom_z) if fh is None else fh
    h = max(0.0035, min(0.012, 0.5 * hgt))
    z = o.bottom_z + h + TIP_BEYOND_TCP
    yaw = math.atan2(u[1], u[0]) + math.pi / 2               # the closing axis (jaw line) across the stroke
    start = c - u * (d_tr + STANDOFF)
    end = c - u * (d_tr - step)
    arms = sorted(ARMS, key=lambda x: float(np.linalg.norm(sk.kin.base(x)[:2, 3] - start)))
    for arm in arms:
        base = sk.kin.base(arm)[:2, 3]
        heading = math.atan2(start[1] - base[1], start[0] - base[0])
        poses = [down_R(yw) for yw in (yaw, yaw + math.pi)]
        poses += [tilted_R(yw, pitch, heading) for pitch in (65.0, 50.0) for yw in (yaw, yaw + math.pi)]
        for R in poses:
            offset = -R[:, 0] * TIP_BEYOND_TCP - np.array([0.0, 0.0, TIP_BEYOND_TCP])
            T_s = T_from(np.array([start[0], start[1], z]) + offset, R)
            T_ab = T_s.copy()
            T_ab[2, 3] = o.top_z + 0.03 + TIP_BEYOND_TCP + offset[2]
            T_e = T_from(np.array([end[0], end[1], z]) + offset, R)
            T_up = T_e.copy()
            T_up[2, 3] = o.top_z + 0.03 + TIP_BEYOND_TCP + offset[2]
            pre = None
            m1 = sk.motion.to_pose(arm, sk.ex.q[arm], T_ab, FAST, "push_above", gripper=PAIR_GRIP, ignore=(label,))
            if not m1.ok:
                # by way of home: an arm left out by a fixture (the place before) can neither move across nor rise
                # from there ("hits <fixture>; via failed: lift unreachable")
                pre = escape_home(sk, arm)
                if pre is not None:
                    m1 = sk.motion.to_pose(arm, pre.q_end, T_ab, FAST, "push_above", gripper=PAIR_GRIP,
                                           ignore=(label,))
            if not m1.ok:
                continue
            m2 = sk.motion.line(arm, m1.q_end, T_s, MEDIUM, "push_down", gripper=PAIR_GRIP, ignore=(label,))
            if not m2.ok:
                continue
            m3 = sk.motion.line(arm, m2.q_end, T_e, 0.04, "push", gripper=PAIR_GRIP, ignore=(label,),
                                max_step_m=0.01)
            if not m3.ok:
                continue
            m4 = sk.motion.line(arm, m3.q_end, T_up, MEDIUM, "push_up", gripper=PAIR_GRIP, check=False)
            plan = pre if pre is not None else ArmPlan(arm, sk.ex.q[arm].copy(), sk.ex.g[arm])
            plan.grip(PAIR_GRIP, 0).move(m1).move(m2).move(m3)
            plan.move(m4)
            return plan
    return None


ESCAPE_RISES = (0.0, 0.05, 0.10)     # m: straight rises tried before the joint move home (``escape_home``)


def escape_home(sk, arm: str):
    """A collision-checked way home for an arm that cannot move across from where it is: a straight rise (none, 5 or
    10 cm), then a checked joint move home, the gripper as it is. ``ArmPlan`` or None (never an unchecked move: an
    unchecked home swing from a place beside a tall fixture swept the open fingers through the just-placed object, G3 dev)."""
    from dojo.skills.executor import ArmPlan
    q0 = np.asarray(sk.ex.q[arm], float)
    g = float(sk.ex.g[arm])
    home = sk.world.home_q[arm]
    for rise in ESCAPE_RISES:
        p = ArmPlan(arm, q0.copy(), g)
        q = q0
        if rise > 0:
            T = sk.kin.tcp(arm, q0).copy()
            T[2, 3] += rise
            up = sk.motion.line(arm, q0, T, MEDIUM, "escape_rise", gripper=g)
            if not up.ok:
                continue
            p.move(up, "escape_rise")
            q = up.q_end
        mv = sk.motion.joint(arm, q, home, FAST, "escape_home", gripper=g)
        if mv.ok:
            return p.move(mv, "escape_home")
    return None


GAIN_PRIOR = 1.0 / OVERSHOOT     # object travel / hand travel before any stroke was measured
GAIN_RANGE = (0.6, 1.6)
PATH_MARGIN = 0.005               # m: the pushed object's footprint keeps this from other objects along a planned path


def stroke_len(left: float, seg: float, gain: float | None = None) -> float:
    """The next stroke's hand travel for ``left`` metres still to go along its axis. ``gain`` (object travel per hand
    travel, measured on the strokes so far; ``update_gain``) scales long strokes; without one the prior ``OVERSHOOT``."""
    s = min(seg, left + 0.003)
    if s <= 0.03:
        return s
    g = GAIN_PRIOR if gain is None else float(np.clip(gain, *GAIN_RANGE))
    return min(seg, s / g)


def update_gain(gain: float | None, moved_along: float, hand: float) -> float | None:
    """Running estimate of object travel per hand travel (strokes of at least 20 mm; a 50/50 blend with the previous)."""
    if hand < 0.02 or moved_along <= 0.0:
        return gain
    g = float(np.clip(moved_along / hand, *GAIN_RANGE))
    return g if gain is None else 0.5 * (gain + g)


def path_clear(sk, label: str, support: str | None, yaw: float | None = None):
    """A ``plan_sequence`` path test: the object's footprint at a virtual centre stays ``PATH_MARGIN`` from every other
    object's footprint (the support, held objects and flat markers under 3 mm excepted). Objects it already touches at
    the start are not held against it (it cannot get away from them in one step anyway)."""
    w = sk.world
    o = w.obj(label)
    held = set(getattr(sk, "held", {}).values()) if isinstance(getattr(sk, "held", None), dict) else set()
    others = []
    for lab, ob in w.objects.items():
        if lab in (label, support) or lab in held or float(ob.top_z - ob.bottom_z) < 0.003:
            continue
        others.append((lab, footprint(ob)))
    fp0 = footprint(o, yaw=yaw)
    others = [(lab, Q) for lab, Q in others if poly_dist(fp0, Q) > PATH_MARGIN]

    def ok(centre) -> bool:
        fp = footprint(o, centre=centre, yaw=yaw)
        return all(poly_dist(fp, Q) > PATH_MARGIN for _, Q in others)
    return ok


SPLITS = (0.4, 0.7)       # fractions of the first axis done before the other axis, then the rest (``plan_sequence``)


def plan_sequence(sk, label: str, goal_xy, axes, stroke_fn, seg: float = 0.08, tol: float = 0.012,
                  max_strokes: int = 10, gain: float | None = None, path_ok=None):
    """``push_plan`` "sequence": the stroke sequence to ``goal_xy`` along ``axes`` (two world unit 2-vectors), the
    object moved virtually (planning world only) by each planned stroke. Candidate phase orders, in turn: the axis with
    most to go first, then the other; the other first; then each order with its first axis split (``SPLITS`` of it, the
    other axis, the rest of the first: a stroke whose start is blocked by a neighbour at the start pose and by a fixture
    at the end pose can fit in between). Returns {"order", "strokes": [(u, step, arm)], "first": plan, "why"} for the
    first candidate whose every stroke plans, else None (``sk.log`` says what blocked). ``gain``: object travel per hand
    travel (``stroke_len``); ``path_ok(centre_xy) -> bool``: every intermediate object position must pass it
    (``path_clear``)."""
    o = sk.world.obj(label)
    c0 = np.asarray(o.center_w[:2], float)
    goal = np.asarray(goal_xy, float)
    d = goal - c0
    comps = [(float(a @ d), k, np.asarray(a, float)) for k, a in enumerate(axes)]
    order0 = sorted(range(len(comps)), key=lambda k: -abs(comps[k][0]))
    cands = [[(k, 1.0) for k in order0]]
    if len(order0) == 2:
        k0, k1 = order0
        cands.append([(k1, 1.0), (k0, 1.0)])
        for first, second in ((k0, k1), (k1, k0)):
            for f in SPLITS:
                cands.append([(first, f), (second, 1.0), (first, 1.0)])
    saved = o.pos.copy()
    why = []
    try:
        for cand in cands:
            pos = c0.copy()
            strokes, first_plan, ok = [], None, True
            for k, frac in cand:
                a = comps[k][2]
                aim = float(a @ (c0 + frac * d - c0)) if frac < 1.0 else None
                for _ in range(max_strokes):
                    left = float(a @ (goal - pos)) if aim is None else aim - float(a @ (pos - c0))
                    if abs(left) <= tol / 2:
                        break
                    u = a * math.copysign(1.0, left)
                    step = stroke_len(abs(left), seg, gain)
                    moved = step * (GAIN_PRIOR if gain is None else float(np.clip(gain, *GAIN_RANGE)))
                    moved = min(moved, abs(left)) if step > 0.03 else step
                    if path_ok is not None and not path_ok(pos + u * moved):
                        why.append(f"{[(k_, f_) for k_, f_ in cand]}: path along axis {k} at {np.round(pos, 3).tolist()}")
                        ok = False
                        break
                    o.pos = saved + np.r_[pos - c0, 0.0]
                    p = stroke_fn(sk, o, label, u, step)
                    if p is None:
                        why.append(f"{[(k_, f_) for k_, f_ in cand]}: axis {k} at {np.round(pos, 3).tolist()}")
                        ok = False
                        break
                    if first_plan is None:
                        first_plan = p
                    strokes.append((u, step, p.arm))
                    pos = pos + u * moved
                if not ok:
                    break
            if ok and strokes:
                return {"order": [k for k, _ in cand], "strokes": strokes, "first": first_plan, "why": why,
                        "end": pos}
    finally:
        o.pos = saved
    sk.log(f"[push_plan] {label}: no complete stroke sequence ({'; '.join(why)[:240]})")
    return None
