"""Grasp-time perception metric (stream M, PERCEPTION_95.md): is the belief right where the manipulator acts?

**Event** = every grasp (jaws close on a target) and every place (jaws open on a held object) of an episode.
**Required objects**: the grasp target (for a place: the held object) and, for a place, its reference (the support /
container / slot / stacked base under it). **Neighbours**: every GT object whose footprint lies within ``R`` (xy) of
the approach corridor (the grasp / place point plus the last ``CORRIDOR_M`` of the TCP's path before it). **Phantoms**:
believed objects within ``R`` of the corridor that match no GT object (``PHANTOM_MM`` from every one, and not
matched to a neighbour).
**Clean** = required objects within the task's tolerance (``eval_g1.TOLERANCE``; 15 mm where none) with the right
identity; neighbours found within ``NEIGHBOUR_MM``; no phantom. Score = clean events / events.

Two layers:

* ``score_event(event, belief, gt, task, ...)``: pure, for any belief source (the runtime's, stream C's per-event
  belief, a frame-0 record). ``belief`` = {label: {"key": "Rigid/cat/00003", "pos": xyz, "quat": wxyz}} (asset frame),
  ``gt`` = list of GT rows (``objects_gt``: label, type, category, category_idx, pos, quat, clutter; root frames are
  converted with ``artroot.row_to_asset``), ``event`` = {"kind": "grasp" | "place", "target", "reference",
  "point", "corridor"} (``Event``).
* ``episode_events(report, actions)``: the runtime's OWN belief at each grasp / release of a saved integ episode:
  the report's ``dev_belief_probe`` snapshot where the run has one (``dojo.runtime.eval.BeliefProbe``), else rebuilt
  from its report (the planning record, then every logged refine / held / drop / post-place update) and its
  executed actions (jaws from the gripper commands, the TCP from the commanded joints: what was attached where).
  GT comes from a physics-only replay of the same actions (``scripts/perception/m95_replay.py``).

DEV-ONLY: GT is read by this scorer, never by a policy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

ARMS = ("left", "right")
CLOSE_BELOW = 0.5           # a closing command below half open closes the jaws
RELEASE_OPEN = 0.15         # a command this far above the tightest since the close releases (dojo.runtime.belief's rule)
RELEASE_ABOVE = 0.6         # ... or any command above this
ATTACH_MARGIN = 0.015       # m: the runtime attaches the object whose bbox + this holds the TCP at a close (KinematicBelief)
CORRIDOR_M = 0.10           # m: the last approach segment, as TCP path length before the event
NEIGHBOUR_MM = 15.0         # mm: a neighbour is found when a believed object is this close (collision-safe)
DEFAULT_TOL_MM = 15.0       # mm: tasks with no positional tolerance in eval_g1.TOLERANCE
ATTRIB_M = 0.06             # m: a close / release with no logged refine is attributed to the believed object this near
RADII = (0.05, 0.10, 0.15)  # m: the R values reported
IGNORE_LABELS = {"camera_stand"}   # GT fixtures perception never estimates (dojo.runtime.eval.perception_eval)
PHANTOM_MM = 50.0           # mm: a believed object this far from every GT object is a phantom (nearer: a pose error)
NOT_HELD_MM = 30.0          # mm: the TCP this far outside the held object's GT bbox at a release: the jaws are not on it


# ---------------------------------------------------------------------------------------------------------- events
def grip_events(actions: list[dict]) -> list[dict]:
    """[{"kind": "grasp" | "release", "step": k, "arm"}] from the commanded grippers; ``step`` is the 1-based index of
    the action that closes / opens (the state before it is the replay's frame ``k - 1``). A grasp is the LAST closing
    command (below ``CLOSE_BELOW``) before the jaws open again: a pre-shape on the approach (hang_mugs' wall grasp
    1.0 -> 0.43, then 0.0 at the real close) is not the grasp (stream C, 2026-10-28; ``GTProbe``'s first-close rule
    takes the pre-shape). A release is the runtime belief's (``KinematicBelief.step``): a command above
    ``RELEASE_ABOVE`` or ``RELEASE_OPEN`` above the tightest since the close."""
    g = {a: 1.0 for a in ARMS}
    pending: dict = {a: None for a in ARMS}         # the latest closing command of the current closing run
    tight: dict = {a: 1.0 for a in ARMS}            # the tightest command since the run began
    out = []
    for k, a in enumerate(actions, 1):
        for arm in ARMS:
            key = f"{arm}_ee_joint_state"
            if key not in a:
                continue
            v = float(np.asarray(a[key], float).reshape(-1)[0])
            if v < CLOSE_BELOW and v < g[arm] - 1e-6:
                pending[arm] = k
                tight[arm] = min(tight[arm], v)
            elif pending[arm] is not None and v > min(RELEASE_ABOVE, tight[arm] + RELEASE_OPEN):
                out.append({"kind": "grasp", "step": pending[arm], "arm": arm})
                out.append({"kind": "release", "step": k, "arm": arm})
                pending[arm], tight[arm] = None, 1.0
            g[arm] = v
    out += [{"kind": "grasp", "step": k, "arm": arm} for arm, k in pending.items() if k is not None]
    order = {"release": 0, "grasp": 1}              # same step: a hand opens before another closes
    return sorted(out, key=lambda e: (e["step"], order[e["kind"]], ARMS.index(e["arm"])))


def tcp_track(actions: list[dict], kin=None) -> dict[str, list]:
    """Per arm, the commanded TCP pose (4x4) after each action (index k = after action k; index 0 = action 1's)."""
    if kin is None:
        from dojo.skills.kinematics import X5Kin
        kin = X5Kin()
    out = {}
    for arm in ARMS:
        key = f"{arm}_arm_joint_state"
        Ts, last = [], None
        for a in actions:
            if key in a:
                last = kin.tcp(arm, np.asarray(a[key], float).reshape(-1)[:6])
            Ts.append(last)
        first = next((T for T in Ts if T is not None), np.eye(4))
        Ts = [first if T is None else T for T in Ts]
        out[arm] = [first] + Ts
    return out


def corridor(track: list, step: int, length: float = CORRIDOR_M) -> np.ndarray:
    """TCP positions from the event (``track[step]``) back along the path until ``length`` of travel: (N, 3)."""
    pts = [np.asarray(track[step], float)[:3, 3]]
    s, k = 0.0, step - 1
    while k >= 0 and s < length:
        p = np.asarray(track[k], float)[:3, 3]
        s += float(np.linalg.norm(p - pts[-1]))
        pts.append(p)
        k -= 1
    return np.asarray(pts)


# ---------------------------------------------------------------------------------------------------------- geometry
def _quat_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q, float) / np.linalg.norm(q))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _R_quat(R) -> np.ndarray:
    """Rotation matrix -> quaternion wxyz (numpy only: the scorer runs where scipy is not installed)."""
    m = np.asarray(R, float)
    t = float(np.trace(m))
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def pose_T(pos, quat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _quat_R(quat)
    T[:3, 3] = np.asarray(pos, float)[:3]
    return T


def _seg_point_dist(a, b, p) -> float:
    ab = b - a
    t = 0.0 if float(ab @ ab) < 1e-12 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - p))


def _seg_seg_dist(a, b, c, d) -> float:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return 0.0
    return min(_seg_point_dist(a, b, c), _seg_point_dist(a, b, d), _seg_point_dist(c, d, a), _seg_point_dist(c, d, b))


def _inside(poly: np.ndarray, p) -> bool:
    """Point in a convex polygon (counter-clockwise or clockwise)."""
    if len(poly) < 3:
        return False
    s = None
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(c) < 1e-15:
            continue
        if s is None:
            s = c > 0
        elif (c > 0) != s:
            return False
    return True


def hull2d(P: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points (monotone chain), counter-clockwise."""
    P = np.unique(np.round(np.asarray(P, float)[:, :2], 5), axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in P[::-1]:
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.asarray(lo[:-1] + up[:-1])


def footprint_dist(poly: np.ndarray, path: np.ndarray) -> float:
    """xy distance (m) between a footprint polygon (M, 2) and a polyline (N, >= 2): 0 when they touch."""
    path = np.asarray(path, float)[:, :2]
    poly = np.asarray(poly, float)[:, :2]
    if any(_inside(poly, p) for p in path):
        return 0.0
    segs = [(path[i], path[i + 1]) for i in range(len(path) - 1)] or [(path[0], path[0])]
    edges = [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))] or [(poly[0], poly[0])]
    return min(_seg_seg_dist(a, b, c, d) for a, b in segs for c, d in edges)


class Footprints:
    """World-xy footprint polygon of an asset at a pose: the convex hull of its catalogue mesh vertices
    (``dojo.perception.render.load_asset``), else a disc of ``default_r``. Cached per asset key."""

    def __init__(self, loader=None, default_r: float = 0.03):
        self._loader = loader
        self._V: dict = {}
        self.default_r = default_r
        self.missing: set = set()

    def verts(self, key: str):
        if key not in self._V:
            V = None
            try:
                if self._loader is not None:
                    V = self._loader(key)
                else:
                    from .render import load_asset
                    V = np.asarray(load_asset(key).V, float)
            except Exception:  # noqa: BLE001 (no catalogue here, an unknown key: the disc)
                V = None
            if V is not None and len(V) > 400:          # the hull of a subsample is close enough for a footprint
                V = V[np.random.default_rng(0).choice(len(V), 400, replace=False)]
            if V is None:
                self.missing.add(key)
            self._V[key] = V
        return self._V[key]

    def centre(self, key: str | None, pos, quat) -> np.ndarray:
        """World position of the asset's bbox centre (its body, not its origin: a 180-degree flip of an off-centre
        origin is a rotation error, not a 50 mm one); the origin without a mesh."""
        V = self.verts(key) if key else None
        p = np.asarray(pos, float)[:3]
        if V is None or quat is None:
            return p
        c = (V.min(0) + V.max(0)) / 2
        return p + _quat_R(quat) @ c

    def polygon(self, key: str | None, pos, quat) -> np.ndarray:
        V = self.verts(key) if key else None
        if V is None or quat is None:
            a = np.linspace(0, 2 * np.pi, 13)[:-1]
            return np.asarray(pos, float)[:2] + self.default_r * np.c_[np.cos(a), np.sin(a)]
        W = V @ _quat_R(quat).T + np.asarray(pos, float)[:3]
        return hull2d(W[:, :2])


# ---------------------------------------------------------------------------------------------------------- scoring
def gt_key(row: dict) -> str | None:
    if row.get("category") is None or row.get("category_idx") is None:
        return row.get("key")
    return f"{row.get('type', 'Rigid')}/{row['category']}/{int(row['category_idx']):05d}"


def gt_rows_asset(rows: list[dict]) -> list[dict]:
    """GT rows with their key, poses in the asset frame (an Articulation's root prim -> its mesh frame)."""
    try:
        from dojo.skills.artroot import row_to_asset
    except Exception:  # noqa: BLE001
        def row_to_asset(r):
            return r
    out = []
    for r in rows:
        if r.get("pos") is None or not r.get("label"):
            continue
        try:
            r2 = row_to_asset(r)
        except Exception:  # noqa: BLE001 (no articulation-root table entry)
            r2 = r
        out.append({**r2, "key": gt_key(r2)})
    return out


def tolerance(task: str) -> tuple[float, float | None]:
    """(translation mm, rotation deg or None) of a task: eval_g1.TOLERANCE, 15 mm where it has none."""
    try:
        from .eval_g1 import TOLERANCE
    except Exception:  # noqa: BLE001 (eval_g1 needs scipy: pass ``tol`` to score_event where it is missing)
        TOLERANCE = {}
    t = TOLERANCE.get(task, (None, None, ""))
    return float(t[0] or DEFAULT_TOL_MM), (float(t[1]) if t[1] else None)


class Groups:
    """Interchangeable labels of a task (``inverse_layout.label_groups``; series objects under their config label,
    ``eval_g1.gt_label``). Without the catalogue: each label its own group."""

    def __init__(self, task: str):
        self.lab2g: dict = {}
        self.series: dict = {}
        try:
            from .eval_g1 import series_labels
            from .inverse_layout import label_groups
            for gi, g in enumerate(label_groups(task)):
                for lab in g["labels"]:
                    self.lab2g[lab] = gi
            self.series = series_labels(task)
        except Exception:  # noqa: BLE001
            pass

    def of(self, label: str):
        if label in self.lab2g:
            return self.lab2g[label]
        base = str(label).rsplit("_", 1)[0] if "_" in str(label) else None
        if base in self.series and self.series[base] in self.lab2g:
            return self.lab2g[self.series[base]]
        return f"label:{label}"


def _rot_err(key: str | None, q_pred, q_gt) -> float | None:
    """Rotation error (deg) modulo the GT placement's symmetry (eval_g1); None without the catalogue."""
    if q_pred is None or q_gt is None:
        return None
    try:
        from .eval_g1 import gt_world_symmetry, rot_err_deg
        sym = gt_world_symmetry(key, q_gt) if key else 1
        return float(rot_err_deg(q_pred, q_gt, sym))
    except Exception:  # noqa: BLE001
        return None


@dataclass
class Event:
    kind: str                       # "grasp" | "place"
    target: str | None              # the grasped / held object's label (the belief's)
    point: list                     # xyz of the TCP at the event
    corridor: list                  # (N, 3) TCP path: the event point, then back along the last approach segment
    reference: str | None = None    # a place's support / container / base (the belief's label)
    step: int = 0
    arm: str = ""
    meta: dict = field(default_factory=dict)


def score_required(label: str | None, belief: dict, gts: list[dict], groups: Groups, tol, role: str) -> dict:
    """One required object: the believed pose vs the nearest (3-D) GT object of its group (interchangeable labels: what
    the jaws meet; stacked objects share xy). Class: ok | identity (right place, wrong asset) | wrong_object (gross for its own object and within
    tolerance of another one, not of its group) | overshoot (tol < xy <= max(15, 2 tol)) | gross | rot | not_believed (no such object in the belief)."""
    t_tol, r_tol = tol
    out = {"role": role, "label": label}
    b = belief.get(label) if label else None
    if b is None or b.get("pos") is None:
        out.update(cls="not_believed", ok=False)
        return out
    bp = np.asarray(b.get("c", b["pos"]), float)
    g_lab = groups.of(label)
    same = [g for g in gts if groups.of(g["label"]) == g_lab]
    gc = lambda g: np.asarray(g.get("c", g["pos"]), float)   # noqa: E731
    # nearest in 3-D: stacked objects share xy (the held block over its base, a tower's blocks)
    near_any = min(gts, key=lambda g: float(np.linalg.norm(gc(g) - bp)), default=None)
    if not same:
        out.update(cls="no_gt", ok=False)
        return out
    g = min(same, key=lambda g: float(np.linalg.norm(gc(g) - bp)))
    d = (bp - gc(g)) * 1000
    xy = float(np.linalg.norm(d[:2]))
    rot = _rot_err(g.get("key"), b.get("quat"), g.get("quat")) if r_tol else None
    out.update(gt_label=g["label"], xy_mm=round(xy, 1), dz_mm=round(float(d[2]), 1),
               rot_deg=None if rot is None else round(rot, 1), key=b.get("key"), gt_key=g.get("key"),
               source=b.get("source"), perc=b.get("perc"))
    if near_any is not None and groups.of(near_any["label"]) != g_lab:
        dn = float(np.linalg.norm(gc(near_any)[:2] - bp[:2])) * 1000
        if dn <= t_tol and xy > max(15.0, 2 * t_tol):   # gross for its own object, on another one: a mix-up
            out.update(cls="wrong_object", ok=False, on=near_any["label"], on_mm=round(dn, 1))
            return out
    if xy > max(15.0, 2 * t_tol):
        cls = "gross"
    elif xy > t_tol:
        cls = "overshoot"
    elif r_tol and rot is not None and rot > r_tol:
        cls = "rot"
    elif b.get("key") and g.get("key") and b["key"] != g["key"]:
        cls = "identity"
    else:
        cls = "ok"
    out.update(cls=cls, ok=cls == "ok")
    return out


def score_event(event: Event | dict, belief: dict, gt: list[dict], task: str, radii=RADII,
                footprints: Footprints | None = None, groups: Groups | None = None,
                neighbour_mm: float = NEIGHBOUR_MM, include_clutter: bool = True, gt_asset_frame: bool = False,
                tol: tuple | None = None) -> dict:
    """Score one event. ``belief``: {label: {"key", "pos", "quat"}} at the event (asset frame); ``gt``: GT rows at the
    event (``objects_gt``; ``gt_asset_frame`` when already converted). Returns the required rows, every neighbour and
    phantom candidate with its corridor distance, and ``clean`` / ``clean_required`` / ``clean_target`` (bools; clean
    per R in ``clean_r``: {"5": .., "10": .., "15": ..} cm) plus ``reasons`` per R. ``tol`` overrides the task's
    (mm, deg or None). A place whose ``event.meta["held_gap_mm"]`` (how far the TCP is outside the GT held object's
    oriented bbox at the release, from the caller's GT; else ``held_drift_mm``, its motion in the TCP frame since the
    grasp) exceeds ``NOT_HELD_MM`` scores the held object ``not_held``: the jaws are not on it."""
    ev = event if isinstance(event, Event) else Event(**{k: v for k, v in event.items() if k in Event.__dataclass_fields__})
    footprints = footprints or Footprints()
    groups = groups or Groups(task)
    tol = tuple(tol) if tol is not None else tolerance(task)
    gts = [g for g in (gt if gt_asset_frame else gt_rows_asset(gt)) if g.get("label") not in IGNORE_LABELS]
    if not include_clutter:
        gts = [g for g in gts if not g.get("clutter")]
    gts = [{**g, "c": footprints.centre(g.get("key"), g["pos"], g.get("quat"))} for g in gts]
    belief = {lab: ({**b, "c": footprints.centre(b.get("key"), b["pos"], b.get("quat"))} if b.get("pos") is not None else b)
              for lab, b in belief.items()}
    path = np.asarray(ev.corridor if len(ev.corridor) else [ev.point], float).reshape(-1, 3)
    req = [score_required(ev.target, belief, [g for g in gts if not g.get("clutter")], groups, tol,
                          "target" if ev.kind == "grasp" else "held")]
    gap = (ev.meta or {}).get("held_gap_mm", (ev.meta or {}).get("held_drift_mm"))
    if ev.kind == "place" and gap is not None and float(gap) > NOT_HELD_MM and req[0].get("cls") != "not_believed":
        req[0].update(cls="not_held", ok=False, gap_mm=round(float(gap), 1))
    if ev.kind == "place" and ev.reference:
        req.append(score_required(ev.reference, belief, [g for g in gts if not g.get("clutter")], groups, tol, "reference"))
    used_gt = {r.get("gt_label") for r in req if r.get("gt_label")}
    used_b = {r["label"] for r in req if r.get("label")}
    # neighbours: GT objects (not the required ones) by footprint distance to the corridor
    neigh = []
    for g in gts:
        if g["label"] in used_gt:
            continue
        poly = footprints.polygon(g.get("key"), g["pos"], g.get("quat"))
        neigh.append({"label": g["label"], "key": g.get("key"), "clutter": bool(g.get("clutter")),
                      "dist_m": round(footprint_dist(poly, path), 4), "pos": [round(float(v), 4) for v in g["c"]]})
    cand_b = {lab: b for lab, b in belief.items() if lab not in used_b and b.get("pos") is not None}
    # match neighbours to believed objects (any label: collision safety is geometric), nearest first, 3-D distance
    pairs = sorted(((float(np.linalg.norm(np.asarray(b["c"], float) - np.asarray(n["pos"], float))), i, lab)
                    for i, n in enumerate(neigh) for lab, b in cand_b.items()))
    taken_n, taken_b = set(), set()
    for d, i, lab in pairs:
        if i in taken_n or lab in taken_b:
            continue
        xy = float(np.linalg.norm(np.asarray(cand_b[lab]["c"], float)[:2] - np.asarray(neigh[i]["pos"])[:2])) * 1000
        if xy > max(3 * neighbour_mm, 60.0):
            continue
        taken_n.add(i)
        taken_b.add(lab)
        neigh[i].update(belief_label=lab, xy_mm=round(xy, 1), belief_source=cand_b[lab].get("source"),
                        dz_mm=round((float(cand_b[lab]["c"][2]) - neigh[i]["pos"][2]) * 1000, 1))
    for n in neigh:
        n["found"] = n.get("xy_mm") is not None and n["xy_mm"] <= neighbour_mm
    # phantoms: believed objects near the corridor that sit on no GT object
    phant = []
    for lab, b in cand_b.items():
        if lab in taken_b:                          # matched to a GT neighbour: its error counts there, once
            continue
        bp = np.asarray(b["c"], float)
        dmin = min((float(np.linalg.norm(np.asarray(g["c"]) - bp)) * 1000 for g in gts), default=1e9)
        if dmin <= max(PHANTOM_MM, tol[0]):
            continue
        poly = footprints.polygon(b.get("key"), b["pos"], b.get("quat"))
        phant.append({"label": lab, "key": b.get("key"), "source": b.get("source"), "dist_m": round(footprint_dist(poly, path), 4),
                      "nearest_gt_mm": round(dmin, 1), "pos": [round(float(v), 4) for v in bp]})
    out = {"kind": ev.kind, "step": ev.step, "arm": ev.arm, "target": ev.target, "reference": ev.reference,
           "point": [round(float(v), 4) for v in np.asarray(ev.point, float)[:3]], "required": req,
           "neighbours": [n for n in neigh if n["dist_m"] <= max(radii)],
           "phantoms": [p for p in phant if p["dist_m"] <= max(radii)], "meta": ev.meta}
    out["clean_target"] = bool(req[0]["ok"])
    out["clean_required"] = all(r["ok"] for r in req)
    out["clean_r"], out["reasons"], out["clean_r_noclutter"] = {}, {}, {}
    for R in radii:
        tag = str(int(round(R * 100)))
        bad = [f"{r['role']} {r['label']} {r['cls']}" + (f" {r['xy_mm']:.0f}mm" if r.get("xy_mm") is not None else "")
               for r in req if not r["ok"]]
        bad += [f"neighbour {n['label']} " + ("not found" if n.get("xy_mm") is None else f"{n['xy_mm']:.0f}mm")
                for n in out["neighbours"] if n["dist_m"] <= R and not n["found"]]
        bad += [f"phantom {p['label']} ({p['nearest_gt_mm']:.0f}mm from any GT)" for p in out["phantoms"] if p["dist_m"] <= R]
        out["clean_r"][tag] = not bad
        out["reasons"][tag] = bad
        clut = {n["label"] for n in out["neighbours"] if n.get("clutter")}
        out["clean_r_noclutter"][tag] = not [b for b in bad if not any(b.startswith(f"neighbour {c} ") for c in clut)]
    out["clean"] = out["clean_r"].get("10", next(iter(out["clean_r"].values())))
    return out


# ---------------------------------------------------------------------------------------------------------- the belief
def flatten_record(rec: dict | None) -> dict:
    """A layout record (Rigid / Geometry / Articulation / Dynamic -> category -> [placements]) as a belief dict."""
    out = {}
    for typ in ("Rigid", "Geometry", "Articulation", "Dynamic"):
        for cat, lst in ((rec or {}).get(typ) or {}).items():
            for r in lst or []:
                if not r.get("label") or r.get("default_pos") is None:
                    continue
                pp = r.get("_perception") or {}
                out[r["label"]] = {"key": f"{typ}/{cat}/{int(r.get('category_idx', 0)):05d}",
                                   "pos": [float(v) for v in r["default_pos"]][:3],
                                   "quat": [float(v) for v in (r.get("default_ori") or [1, 0, 0, 0])][:4],
                                   "type": typ, "source": "record",
                                   "perc": {k: pp[k] for k in ("confidence", "occluded", "hidden_frac", "ambiguous", "source")
                                            if pp.get(k) is not None}}
    return out


PLANNING_ORDER = ("agentic2", "searched", "fused", "agentic0", "scan", "initial")   # = dojo.runtime.estimate's


def _applied(e: dict) -> bool:
    """A refine event whose pose the runtime wrote into the belief (policy._refine_hook / _held_refine)."""
    if not e.get("ok") or e.get("pose7") is None or e.get("applied") is False:
        return False
    w = str(e.get("why") or "")
    return not any(s in w for s in ("prior pose kept", "prior kept", "reverted", "not confirmed", "plan kept",
                                    "on a neighbour"))


class BeliefTracker:
    """The runtime's belief over an episode, rebuilt from its report and executed actions. Starts from the planning
    record; applies, at their steps, accepted refines (pick / place: the object's pose; held: the in-hand transform),
    the jaws (a close attaches the target to the TCP, a release puts it where the TCP carried it), ``belief_drop``
    (the drop rule's z), ``post_place`` re-measures and held_look drops. What it cannot see (a divergence's
    re-estimate, the twin's nominal outcome under verify) is flagged per event (``after_divergence``)."""

    def __init__(self, report: dict, actions: list[dict], kin=None):
        recs = report.get("records") or {}
        name = next((k for k in PLANNING_ORDER if recs.get(k)), None)
        self.record = name
        self.belief = flatten_record(recs.get(name) if name else None)
        self.not_found = [x.get("label") for x in ((recs.get(name) or {}).get("_not_found") or []) if isinstance(x, dict)]
        self.track = tcp_track(actions, kin)
        self.attached: dict = {}            # arm -> (label, T_tcp_obj, pose before the grasp)
        self.revs = sorted((e for e in report.get("events") or [] if isinstance(e.get("step"), (int, float))),
                           key=lambda e: e["step"])
        self.diverged_at = None
        self._i = 0
        self.last_refine: dict = {}         # (arm, for) -> (step, label)

    def T(self, arm: str, step: int) -> np.ndarray:
        tr = self.track[arm]
        return np.asarray(tr[min(max(step, 0), len(tr) - 1)], float)

    def _apply(self, e: dict) -> None:
        k, lab = e.get("kind"), e.get("label")
        if k == "refine" and lab:
            if e.get("arm"):
                self.last_refine[(e["arm"], e.get("for"))] = (int(e["step"]), lab)
            if e.get("for") == "held":
                if _applied(e) and e.get("arm") in self.attached and self.attached[e["arm"]][0] == lab:
                    T_obj = pose_T(e["pose7"][:3], e["pose7"][3:7])
                    a = self.attached[e["arm"]]
                    self.attached[e["arm"]] = (lab, np.linalg.inv(self.T(e["arm"], int(e["step"]))) @ T_obj, a[2])
            elif _applied(e) and lab in self.belief:
                p7 = e["pose7"]
                q = p7[3:7]
                if "orientation kept" in str(e.get("why") or ""):
                    q = self.belief[lab]["quat"]
                self.belief[lab].update(pos=list(p7[:3]), quat=list(q), source=f"refine_{e.get('for')}")
            elif e.get("rehover") in ("confirm", "recentre") and e.get("pose7") is not None and lab in self.belief:
                pass                        # pending: the next refine of the label confirms or reverts it
        elif k == "post_place" and lab in self.belief and e.get("ok") and e.get("pose7") is not None:
            self.belief[lab].update(pos=list(e["pose7"][:3]), quat=list(e["pose7"][3:7]), source="post_place")
        elif k == "post_place" and lab in self.belief and e.get("ok") and e.get("to") is not None:
            # RuntimeProvider logs a post-place move as from / to positions (its new orientation is not logged; W,
            # 2026-10-28: P3r3 hang_mugs 100000 mug1 moved 18 mm, the rebuild kept the released pose)
            self.belief[lab].update(pos=[float(v) for v in e["to"]][:3], source="post_place")
        elif k == "belief_drop" and lab in self.belief and e.get("z") is not None:
            self.belief[lab]["pos"][2] = float(e["z"])
        elif k in ("held_look", "grip_check") and e.get("dropped") and e.get("arm") in self.attached:
            lab0, _, pre = self.attached.pop(e["arm"])
            if lab0 in self.belief and pre is not None:
                self.belief[lab0].update(pos=pre[0], quat=pre[1], source="held_dropped")
        elif k in ("divergence_replan", "lost_search") or (k == "resync" and e.get("diverged")):
            if self.diverged_at is None:
                self.diverged_at = int(e["step"])

    def advance(self, upto: int) -> None:
        """Apply every report event logged at a step <= ``upto`` (after that many actions)."""
        while self._i < len(self.revs) and self.revs[self._i]["step"] <= upto:
            self._apply(self.revs[self._i])
            self._i += 1

    def held_pose(self, arm: str, step: int):
        lab, T_rel, _ = self.attached[arm]
        T = self.T(arm, step) @ T_rel
        return lab, T[:3, 3].tolist(), _R_quat(T[:3, :3]).tolist()

    def snapshot(self, step: int) -> dict:
        """The belief after ``step`` actions: held objects where the TCP carries them."""
        b = copy.deepcopy(self.belief)
        for arm in list(self.attached):
            lab, p, q = self.held_pose(arm, step)
            if lab in b:
                b[lab].update(pos=p, quat=q, source="held")
        return b

    def nearest(self, xyz, exclude=(), below_of=None) -> str | None:
        best = None
        for lab, o in self.belief.items():
            if lab in exclude or o.get("pos") is None:
                continue
            d = float(np.linalg.norm(np.asarray(o["pos"], float) - np.asarray(xyz, float)))
            if below_of is not None:
                d = float(np.linalg.norm(np.asarray(o["pos"], float)[:2] - np.asarray(below_of, float)[:2]))
                if float(o["pos"][2]) > float(below_of[2]) + 0.01:
                    continue
            if best is None or d < best[0]:
                best = (d, lab)
        return best[1] if best and best[0] <= ATTRIB_M * (1.5 if below_of is not None else 1.0) else None


def probe_belief(snap: dict, track: dict, step: int, kin) -> dict:
    """A ``dev_belief_probe`` snapshot (``dojo.runtime.eval.BeliefProbe``) as a belief after ``step`` actions: the
    provider's objects, an attached one carried from the joints the belief last stepped to the commanded TCP now."""
    b = {lab: {**o, "source": f"probe_{o.get('source')}"} for lab, o in (snap.get("objects") or {}).items()}
    for arm, lab in (snap.get("attached") or {}).items():
        if lab not in b or arm not in (snap.get("q") or {}):
            continue
        T0 = kin.tcp(arm, np.asarray(snap["q"][arm], float))
        T1 = np.asarray(track[arm][min(step, len(track[arm]) - 1)], float)
        T = T1 @ np.linalg.inv(T0) @ pose_T(b[lab]["pos"], b[lab]["quat"])
        b[lab] = {**b[lab], "pos": T[:3, 3].tolist(), "quat": _R_quat(T[:3, :3]).tolist(), "source": "probe_held"}
    return b


def attach_label(belief: dict, p, footprints: "Footprints", exclude=()) -> str | None:
    """The object the runtime's ``KinematicBelief`` attaches at a close: a believed movable (not Geometry, not Dynamic)
    object whose bbox + ``ATTACH_MARGIN`` holds the TCP ``p``, the nearest to its centre; one another hand holds only
    when no free one does (a handover). None: the runtime believed the jaws closed on nothing."""
    def pick(excl):
        best = None
        for lab, o in belief.items():
            key = str(o.get("key") or "")
            if lab in excl or o.get("pos") is None or key.split("/")[0] in ("Geometry", "Dynamic"):
                continue
            V = footprints.verts(key)
            if V is None:
                continue
            lo, hi = V.min(0), V.max(0)
            R = _quat_R(o["quat"])
            loc = R.T @ (np.asarray(p, float) - (np.asarray(o["pos"], float) + R @ ((lo + hi) / 2)))
            if np.all(np.abs(loc) <= (hi - lo) / 2 + ATTACH_MARGIN):
                d = float(np.linalg.norm(loc))
                if best is None or d < best[0]:
                    best = (d, lab)
        return best[1] if best else None
    return pick(set(exclude)) or pick(set())


def episode_events(report: dict, actions: list[dict], kin=None,
                   footprints: "Footprints | None" = None) -> list[tuple[Event, dict]]:
    """(event, belief) at every grasp / place of an episode, from the runtime's own logs. A close with no logged pick
    refine of that arm since its last grip event is attributed to the believed object nearest the TCP (within
    ``ATTRIB_M``), else it is not a grasp (a press with closed jaws: kind ``close_other``). With ``footprints`` the
    jaws attach what the runtime's own rule attaches (``attach_label``; nothing when the TCP is off every believed
    object, the runtime knew it held nothing); without, the target. A release of an arm with nothing attached is
    ``release_other``."""
    if kin is None:
        from dojo.skills.kinematics import X5Kin
        kin = X5Kin()
    bt = BeliefTracker(report, actions, kin)
    snaps = {int(x["step"]): x for x in ((report.get("dev_belief_probe") or {}).get("snaps") or [])}
    out = []
    last_grip = {a: 0 for a in ARMS}

    def belief_at(j: int) -> tuple[dict, str]:
        sn = snaps.get(j)
        return (probe_belief(sn, bt.track, j, kin), "probe") if sn is not None else (bt.snapshot(j), "rebuilt")

    for ge in grip_events(actions):
        k, arm = ge["step"], ge["arm"]
        bt.advance(k - 1)
        Tk = bt.T(arm, k)
        pt = Tk[:3, 3]
        cor = corridor(bt.track[arm], k)
        belief, src = belief_at(k - 1)
        sn = snaps.get(k - 1)
        meta = {"record": bt.record, "belief": src,
                "after_divergence": bt.diverged_at is not None and bt.diverged_at <= k}
        if ge["kind"] == "grasp":
            lr = bt.last_refine.get((arm, "pick"))
            held_now = {v[0] for v in bt.attached.values()}
            if sn is not None:
                held_now = set((sn.get("attached") or {}).values())
            if lr and lr[0] >= last_grip[arm] and lr[1] not in held_now:
                lab, how = lr[1], "refine"
            else:
                cand = None
                for lab_, o in belief.items():
                    if lab_ in held_now or o.get("pos") is None:
                        continue
                    d = float(np.linalg.norm(np.asarray(o["pos"], float) - pt))
                    if cand is None or d < cand[0]:
                        cand = (d, lab_)
                lab, how = (cand[1], "nearest") if cand and cand[0] <= ATTRIB_M else (None, None)
            if lab is None:
                out.append((Event("close_other", None, pt.tolist(), cor.tolist(), step=k, arm=arm, meta=meta), belief))
            else:
                meta["attributed"] = how
                out.append((Event("grasp", lab, pt.tolist(), cor.tolist(), step=k, arm=arm, meta=meta), belief))
                att = lab
                if footprints is not None:      # what the runtime's own attach rule takes (H, 2026-10-28: 18 of 98
                    att = attach_label(belief, pt, footprints, exclude=held_now)    # not_held were never attached)
                    meta["attached"] = att
                o = (bt.snapshot(k - 1).get(att) or belief.get(att)) if att else None
                if o is not None:
                    lab = att
                    for a2 in list(bt.attached):    # a handover: the giver's object is the taker's now
                        if bt.attached[a2][0] == lab:
                            bt.attached.pop(a2)
                    pre = (list(o["pos"]), list(o["quat"]))
                    bt.attached[arm] = (lab, np.linalg.inv(Tk) @ pose_T(o["pos"], o["quat"]), pre)
                    bt.belief.setdefault(lab, {**o})
        else:
            held = (sn.get("attached") or {}).get(arm) if sn is not None else None
            if held is None and arm in bt.attached:
                held = bt.attached[arm][0]
            if held is None:
                out.append((Event("release_other", None, pt.tolist(), cor.tolist(), step=k, arm=arm, meta=meta), belief))
            else:
                lab = held
                if src == "rebuilt" or lab not in belief:
                    _, p, q = bt.held_pose(arm, k) if arm in bt.attached else (lab, belief[lab]["pos"], belief[lab]["quat"])
                else:
                    p, q = belief[lab]["pos"], belief[lab]["quat"]
                lr = bt.last_refine.get((arm, "place"))
                if lr and lr[0] >= last_grip[arm] and lr[1] != lab:
                    ref, how = lr[1], "refine"
                else:
                    ref, how = bt.nearest(p, exclude={lab}, below_of=p), "nearest"
                meta.update(attributed_ref=how if ref else None)
                out.append((Event("place", lab, pt.tolist(), cor.tolist(), reference=ref, step=k, arm=arm, meta=meta),
                            belief))
                bt.attached.pop(arm, None)
                if lab in bt.belief:
                    bt.belief[lab].update(pos=list(p), quat=list(q), source="released")
        last_grip[arm] = k
    return out


# ---------------------------------------------------------------------------------------------------------- summary
def summarize(rows: list[dict], radii=RADII) -> dict:
    """Clean rates over scored events (grasp + place): per R, required-only and target-only."""
    ev = [r for r in rows if r.get("kind") in ("grasp", "place")]
    n = len(ev)
    out = {"events": n, "grasp": sum(r["kind"] == "grasp" for r in ev), "place": sum(r["kind"] == "place" for r in ev)}
    for R in radii:
        tag = str(int(round(R * 100)))
        c = sum(bool(r["clean_r"].get(tag)) for r in ev)
        out[f"clean_R{tag}"] = round(c / n, 3) if n else None
        out[f"n_clean_R{tag}"] = c
    out["clean_required"] = round(sum(r["clean_required"] for r in ev) / n, 3) if n else None
    out["clean_target"] = round(sum(r["clean_target"] for r in ev) / n, 3) if n else None
    return out


__all__ = ["Event", "BeliefTracker", "Footprints", "Groups", "episode_events", "grip_events", "score_event",
           "score_required", "summarize", "tcp_track", "corridor", "flatten_record", "footprint_dist", "hull2d"]
