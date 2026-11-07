"""Torso grasps and stand targets for tall irregular objects (stream ORG2; default off).

``Skills.grasp_torso`` "on": ``plan_transfer`` adds ``torso_grasps`` to an irregular object's candidates: near-
horizontal approaches that close across the body at mid / upper height, the closing direction 90 or 45 degrees off the
object's long horizontal axis (the line of its outstretched appendages). Top-down slice pinches of a 19 cm figure with spread arms
closed 57-78 mm wide at mid height with a 40 deg tilt and slid out at the lift (G3 dev: rose -41 to +16 mm of 80, 4 of
9 layouts). Across the arm line the body is 40-60 mm deep, and a side approach keeps the TCP between the arm's base and
the object, which reaches 0.55 m out where a top pinch does not.

``Skills.stand_offset`` "base": a stand smaller than the object (``library._on_targets``' small-support case) gets
extra place positions on rings of 0.25 and 0.5 x the object's base radius around its centre, ordered by clearance from
the other objects (a stand 4 cm from a tall fixture: an object 14 cm across its base does not fit at the stand's centre
without its footprint entering the fixture). Half the base radius keeps the stand under the object's base.

``Skills.retreat_axis`` "on": a place whose hand approached from the side backs out along the fingers first and only
then rises (``Skills._place_tail``: today's retreat rises 2 cm while backing out, which drags an appendage that hangs
over the fingers).

Geometry from the asset catalogue's mesh only; no task names, layout ids or checker values.
"""
from __future__ import annotations

import math

import numpy as np

from dojo.skills.geometry import T_from
from dojo.skills.kinematics import TABLE_Z, TCP_OFFSET
from dojo.skills.world import ObjectState

TORSO_HEIGHTS = (0.5, 0.6, 0.7, 0.8)     # fractions of the object's height (from its bottom)
TORSO_OFFSETS = (90.0, 45.0, -45.0)      # deg: closing direction relative to the long horizontal axis
TORSO_PITCHES = (0.0, 20.0, 35.0)        # deg below horizontal
BAND = 0.006                              # m: half-height of a slice
CORRIDOR = 0.010                          # m: half-width (across the closing axis) of the body section taken
FINGER_HALF = 0.012                       # m: a finger's half-width across the closing axis
FINGER_REACH = 0.014                      # m: how far a finger extends past the TCP along the approach
PRE = 0.07                                # m: the pre-grasp retreat (``Grasp.pre_dist``)
TORSO_FIRST = -0.6                        # score offset: screened and planned before the slice pinches (G3 dev: all 5
                                          # executed slice picks of a figure slid out, dev baseline)
PALM_DEPTH, PALM_HALF_W, PALM_HALF_H = 0.084, 0.08, 0.03   # m: ``X5Kin.palm_points``' box (flange frame)


def long_axis(V: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The long horizontal direction (unit 2-vector, world) of the points between heights ``lo`` and ``hi``."""
    P = V[(V[:, 2] >= lo) & (V[:, 2] <= hi), :2]
    if len(P) < 10:
        P = V[:, :2]
    P = P - P.mean(0)
    w, U = np.linalg.eigh(P.T @ P)
    a = U[:, int(np.argmax(w))]
    return a / np.linalg.norm(a)


def base_radius(o: ObjectState, band: float = 0.008) -> float | None:
    """The radius (m) of the object's lowest band of mesh points about its origin's vertical (its base)."""
    if o.asset.mesh is None:
        return None
    V = o.asset.mesh[0] @ o.R.T + o.pos
    z0 = float(V[:, 2].min())
    P = V[V[:, 2] < z0 + band, :2]
    if len(P) < 3:
        return None
    return float(np.max(np.linalg.norm(P - o.pos[:2], axis=1)))


def torso_grasps(obj: ObjectState, kin, arm: str, heights=TORSO_HEIGHTS, offsets=TORSO_OFFSETS,
                 pitches=TORSO_PITCHES):
    """Side-ish grasps across the body of an upright irregular object (see the module doc). Each candidate's pads close
    on the body section within ``CORRIDOR`` of the closing line; the open fingers' paths from the pre-grasp to the
    grasp must not meet the object's own points."""
    from dojo.skills.grasps import CLEAR, MAX_GAP, MAX_WIDTH, Grasp, _frame
    if obj.asset.mesh is None:
        return []
    V = obj.asset.mesh[0] @ obj.R.T + obj.pos
    zmin, zmax = float(V[:, 2].min()), float(V[:, 2].max())
    H = zmax - zmin
    if H < 0.08:
        return []
    a = long_axis(V, zmin + 0.3 * H, zmin + 0.9 * H)
    base = kin.base(arm)[:2, 3]
    out = []
    for frac in heights:
        z = zmin + frac * H
        slab = V[np.abs(V[:, 2] - z) < BAND]
        if len(slab) < 8:
            continue
        cxy = (slab[:, :2].min(0) + slab[:, :2].max(0)) / 2
        for off in offsets:
            th = math.atan2(a[1], a[0]) + math.radians(off)
            c = np.array([math.cos(th), math.sin(th)])          # closing direction
            n = np.array([-c[1], c[0]])                         # across it (the approach's horizontal direction)
            qc, qn = (slab[:, :2] - cxy) @ c, (slab[:, :2] - cxy) @ n
            sec = np.abs(qn) < CORRIDOR
            if sec.sum() < 4:
                continue
            w = float(qc[sec].max() - qc[sec].min())
            if w > MAX_WIDTH - 0.004 or w < 0.012:
                continue
            p_xy = cxy + c * float(qc[sec].max() + qc[sec].min()) / 2
            gap = min(MAX_GAP, w + 2 * CLEAR + 0.006)
            for sgn in (1.0, -1.0):
                hd = sgn * n                                    # the approach's horizontal direction
                if float(hd @ (p_xy - base)) <= 0:              # from the arm's side only
                    continue
                for pitch in pitches:
                    pr = math.radians(pitch)
                    appr = np.array([hd[0] * math.cos(pr), hd[1] * math.cos(pr), -math.sin(pr)])
                    T = T_from(np.array([p_xy[0], p_xy[1], z]), _frame(appr, np.array([c[0], c[1], 0.0])))
                    if T[2, 3] - (FINGER_HALF + 0.004) < TABLE_Z:
                        continue
                    # the finger corridors (outside the pads along c, a finger wide) from the pre-grasp to the grasp
                    D = V - T[:3, 3]
                    q_a, q_c, q_z = D @ T[:3, 0], D @ T[:3, 1], D @ T[:3, 2]
                    in_f = ((np.abs(q_c) > gap / 2 - 0.001) & (np.abs(q_c) < gap / 2 + 0.014)
                            & (np.abs(q_z) < FINGER_HALF) & (q_a > -PRE) & (q_a < FINGER_REACH))
                    if in_f.any():
                        continue
                    # the palm / finger-rail box at the grasp (``X5Kin.palm_points``: flange x 0..84 mm, y +-80, z +-30;
                    # the TCP is ``TCP_OFFSET`` past the flange): nothing of the object may sit there (an arm that
                    # reaches back over the hand at 45 deg), and the straight approach only sweeps it from behind
                    in_p = ((q_a > -TCP_OFFSET - 0.005) & (q_a < -TCP_OFFSET + PALM_DEPTH + 0.005)
                            & (np.abs(q_c) < PALM_HALF_W + 0.005) & (np.abs(q_z) < PALM_HALF_H + 0.005))
                    if in_p.any():
                        continue
                    out.append(Grasp("torso", T, w, pre_dist=PRE,
                                     score=TORSO_FIRST + 0.2 * abs(frac - 0.65) / 0.15 + pitch / 100.0 + 0.3 * w / 0.08,
                                     note=f"torso h {frac:.2f} off {off:+.0f} w {w * 1000:.0f}mm pitch {pitch:.0f}"))
    return out


CLEAR_CAP = 0.02         # m: clearance beyond this does not rank a stand position higher (then: nearest the centre)
STAND_KEEP = 10          # the clearest stand positions kept (the transfer screen's IK budget: positions x headings x grasps)
BEARING_SPREAD = (0.0, math.radians(30.0), -math.radians(30.0))


def stand_yaws(sk, label: str, support: str) -> list[float]:
    """Headings (world yaw of the object's ``up`` pose about the vertical) that keep a grasp's bearing from each arm:
    the object's own heading turned by how much the bearing from the arm's base changes from where it stands to the
    support (a side grasp fixed in the object frame then still points away from that arm), and +-30 deg about it."""
    from dojo.skills.kinematics import ARMS
    from dojo.skills.library import _up_yaw
    w = sk.world
    o, b = w.obj(label), w.obj(support)
    yo = _up_yaw(o)
    out = []
    for a in ARMS:
        base = sk.kin.base(a)[:2, 3]
        d0 = o.center_w[:2] - base
        d1 = b.center_w[:2] - base
        turn = math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])
        out += [yo + turn + s for s in BEARING_SPREAD]
    return out


def stand_points(sk, label: str, support: str, pts=()) -> list[np.ndarray]:
    """``Skills.stand_offset`` "base": ``pts`` (the caller's positions) plus rings around a small support's centre
    (module doc), all ordered by clearance from other objects' footprints (capped at ``CLEAR_CAP``), then nearest the
    centre: the transfer screen's IK budget goes to positions that can be clear first (a wide figure's centre
    positions beside a tall fixture used up 6000 place IKs with no pair, CPU dev screen)."""
    from dojo.skills.push_plan import _inside, _seg_dist, footprint
    w = sk.world
    o, b = w.obj(label), w.obj(support)
    r = base_radius(o)
    if r is None:
        return []
    c = np.asarray(b.center_w[:2], float)
    pts = [np.asarray(p, float) for p in pts] + [c + f * r * np.array([math.cos(k * math.pi / 4),
                                                                         math.sin(k * math.pi / 4)])
                                                  for f in (0.25, 0.5) for k in range(8)]
    others = [footprint(ob) for lab, ob in w.objects.items()
              if lab not in (label, support) and float(ob.top_z - ob.bottom_z) >= 0.003]
    disc = 0.5 * float(np.max(o.asset.extents[:2]))

    def clearance(p) -> float:
        # signed: the object's disc (its widest half extent) against each footprint; inside one counts as -disc
        out = 1.0
        for Q in others:
            d = 0.0 if _inside(p, Q) else min(_seg_dist(p, Q[k], Q[(k + 1) % len(Q)]) for k in range(len(Q)))
            out = min(out, d - disc)
        return out
    return sorted(pts, key=lambda p: (-min(clearance(p), CLEAR_CAP), float(np.linalg.norm(p - c))))


VIA_NEAR = 0.20          # m: fixtures whose footprint lies within this of the support's centre shape the vias


def clear_vias(sk, label: str, support: str) -> list[np.ndarray]:
    """``Skills.place_via`` "clear": xy offsets (from the place target) for ``Motion.via_xy``: out from each fixture
    next to the support (along the normal from its footprint's nearest point to the support's centre), and towards the
    arms' bases, so the held object comes in from its clear side (a stand 4 cm beside a tall fixture, the carry's joint move
    swept a 19 cm figure through the fixture's corner, G3 dev 4 of 5 plans)."""
    from dojo.skills.kinematics import ARMS
    from dojo.skills.push_plan import _seg_dist, footprint
    w = sk.world
    c = np.asarray(w.obj(support).center_w[:2], float)
    bases = np.mean([sk.kin.base(a)[:2, 3] for a in ARMS], axis=0)
    b = bases - c
    b = b / max(float(np.linalg.norm(b)), 1e-9)
    normals = []
    for lab, ob in w.objects.items():
        if lab in (label, support) or float(ob.top_z - ob.bottom_z) < 0.03:
            continue
        fp = footprint(ob)
        d, p_best = 9.0, None
        for k in range(4):
            a_, b_ = fp[k], fp[(k + 1) % 4]
            ab = b_ - a_
            t = float(np.clip((c - a_) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
            q = a_ + t * ab
            if _seg_dist(c, a_, b_) < d:
                d, p_best = _seg_dist(c, a_, b_), q
        if p_best is None or d > VIA_NEAR or d < 1e-6:
            continue
        n = c - p_best
        normals.append((d, n / np.linalg.norm(n)))
    out = []
    for _, n in sorted(normals, key=lambda x: x[0]):
        nb = n + b
        nb = nb / max(float(np.linalg.norm(nb)), 1e-9)
        out += [n * 0.08, nb * 0.10, n * 0.14, nb * 0.16]
    out.append(b * 0.12)
    return out
