"""Torso grasps and stand targets for tall irregular objects (stream ORG2; default off).

``Skills.grasp_torso`` "on": ``plan_transfer`` adds ``torso_grasps`` to an irregular object's candidates: near-
horizontal approaches that close across the body at mid / upper height, the closing direction 90 or 45 degrees off the
object's long horizontal axis (a figurine's arm line). Top-down slice pinches of a 19 cm figurine with spread arms
closed 57-78 mm wide at mid height with a 40 deg tilt and slid out at the lift (G3 dev: rose -41 to +16 mm of 80, 4 of
9 layouts). Across the arm line the body is 40-60 mm deep, and a side approach keeps the TCP between the arm's base and
the object, which reaches 0.55 m out where a top pinch does not.

``Skills.stand_offset`` "base": a stand smaller than the object (``library._on_targets``' small-support case) gets
extra place positions on rings of 0.25 and 0.5 x the object's base radius around its centre, ordered by clearance from
the other objects (a stand 4 cm from a drawer: an object 14 cm across its base does not fit at the stand's centre
without its footprint entering the drawer). Half the base radius keeps the stand under the object's base.

``Skills.retreat_axis`` "on": a place whose hand approached from the side backs out along the fingers first and only
then rises (``Skills._place_tail``: today's retreat rises 2 cm while backing out, which drags an appendage that hangs
over the fingers).

Geometry from the asset catalogue's mesh only; no task names, layout ids or checker values.
"""
from __future__ import annotations

import math

import numpy as np

from dojo.skills.geometry import T_from
from dojo.skills.kinematics import TABLE_Z
from dojo.skills.world import ObjectState

TORSO_HEIGHTS = (0.5, 0.6, 0.7, 0.8)     # fractions of the object's height (from its bottom)
TORSO_OFFSETS = (90.0, 45.0, -45.0)      # deg: closing direction relative to the long horizontal axis
TORSO_PITCHES = (0.0, 20.0, 35.0)        # deg below horizontal
BAND = 0.006                              # m: half-height of a slice
CORRIDOR = 0.010                          # m: half-width (across the closing axis) of the body section taken
FINGER_HALF = 0.012                       # m: a finger's half-width across the closing axis
FINGER_REACH = 0.014                      # m: how far a finger extends past the TCP along the approach
PRE = 0.07                                # m: the pre-grasp retreat (``Grasp.pre_dist``)


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
                    out.append(Grasp("torso", T, w, pre_dist=PRE,
                                     score=0.2 * abs(frac - 0.65) / 0.15 + pitch / 100.0 + 0.3 * w / 0.08,
                                     note=f"torso h {frac:.2f} off {off:+.0f} w {w * 1000:.0f}mm pitch {pitch:.0f}"))
    return out


def stand_points(sk, label: str, support: str) -> list[np.ndarray]:
    """``Skills.stand_offset`` "base": extra xy positions around a small support's centre (module doc), the most
    clearance from other objects' footprints first."""
    from dojo.skills.push_plan import footprint, poly_dist
    w = sk.world
    o, b = w.obj(label), w.obj(support)
    r = base_radius(o)
    if r is None:
        return []
    c = np.asarray(b.center_w[:2], float)
    pts = [c + f * r * np.array([math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)])
           for f in (0.25, 0.5) for k in range(8)]
    others = [footprint(ob) for lab, ob in w.objects.items()
              if lab not in (label, support) and float(ob.top_z - ob.bottom_z) >= 0.003]
    disc = 0.5 * float(np.max(o.asset.extents[:2]))

    def clearance(p) -> float:
        sq = np.array([[p[0] - disc, p[1] - disc], [p[0] + disc, p[1] - disc], [p[0] + disc, p[1] + disc],
                       [p[0] - disc, p[1] + disc]])
        return min([poly_dist(sq, Q) for Q in others] or [1.0])
    return sorted(pts, key=lambda p: -clearance(p))
