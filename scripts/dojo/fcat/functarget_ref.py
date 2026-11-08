"""PROTOTYPE (night/GLIB-SHAPE, 7-8 Nov 2026): functional targets, phase 4.0 of the F strawman v2.

Input to the joint planning of 8 Nov, where the two teams decide who owns the adapter and who owns the catalogue
extraction. Nothing in glib calls this module; it may be thrown away or moved whole. It holds no perception code, no
catalogue extraction, no simulator read and no gripper observation read.

Contents:
  * ``FuncTarget``: one functional target exactly as strawman v2 section 1 defines it, validated on construction.
  * ``from_asset(asset_frames, root_to_bbox, ...)`` / ``as_frames(targets)``: the adapter both ways between targets and
    the asset frame layout today's frame verbs read (``AssetInfo.functional`` / ``active_functional`` / ``support``,
    pose7 = xyz + wxyz in the owner's root frame; a lever's joint beside it). Round trip within 1e-9.
  * ``decide(target, tol, ...)``: the verb-side check (strawman v2 section 3, Ilia's side's COIN2 markup 1.5): each
    toleranced component's k-sigma against the verb's tolerance -> go / measure_first / refuse, with the measurement
    that covers what is out of tolerance. Tolerances: ``slot_insert``, ``hole_insert``, ``peg_thread``,
    ``stand_place``, ``support_release``.

Frame conventions (``CONV``): which frame column is a target's ``axis`` and which its ``across``, per type. They are
NOT pinned against real catalogue frames here (that needs the catalogue-wide round trip, an agenda item); the adapter
is exact for any convention in the table, so pinning one changes a table row, not the code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TYPES = ("hole", "slot", "socket", "keyhole", "peg", "hook", "handle", "rim", "strike_point", "lever", "hinge",
         "support", "base")
SOURCES = ("perception", "catalogue", "tactile")
SECTIONS = ("functional", "active_functional", "support")
EXTENT_KEYS = {"radius", "width", "length", "depth", "height"}
SIGMA_POS = ("across", "along", "axis")              # 1-sigma position (m), the target's own frame
SIGMA_ANG = ("about_axis", "tilt")                    # 1-sigma angle (deg)
REQUIRED = {"perception": ("query", "build_hash"), "catalogue": ("key", "tag", "catalogue_hash"),
            "tactile": ("method",)}
JOINT_KINDS = ("revolute", "prismatic")
TOL = 1e-6                                            # unit length / orthogonality tolerance of axis and across

# type -> (axis column, its sign, across column) of the frame: axis = sign * R[:, a], across = R[:, c]
CONV = {t: (2, 1.0, 0) for t in TYPES}
CONV["hole"] = (2, -1.0, 0)                           # a hole mouth's frame z out of the hole (believed): axis = into it
CONV["base"] = (2, -1.0, 0)                           # a base points down
SECTION_OF = {t: "functional" for t in TYPES}         # where a target without catalogue provenance is written
SECTION_OF["hole"] = "support"                        # insert_each reads hole mouths from the support section
CORNER_TYPES = ("slot",)                              # a tag of several frames is one target with corners
NO_ORIENTATION = ("lever", "hinge")                   # axis = the joint's; the frame's own orientation is not kept


# ==================================================================================================== the target
@dataclass
class FuncTarget:
    name: str
    owner: str
    type: str
    point: np.ndarray                                  # owner root frame (m)
    axis: np.ndarray                                   # unit, owner root frame
    provenance: list                                   # [{source, step, ...}], at least one
    across: np.ndarray | None = None                   # unit, perpendicular to axis
    extent: dict = field(default_factory=dict)         # radius | width, length, depth | height (m, > 0)
    lip: list | None = None                            # [[depth_m, half_width_m], ...], depth ascending
    surface: dict | None = None                        # support only: z_min, z_max, flat_tol, rim_height
    corners: list | None = None                        # pose7 per corner, owner root frame
    joint: dict | None = None                          # {kind, range: [lo, hi]} (+ name)
    owner_fixed: bool | None = None                    # True | False | None (unknown)
    sigma: dict = field(default_factory=lambda: {"pos": [0.0, 0.0, 0.0], "ang": [0.0, 0.0]})
    conf: float = 1.0
    active: bool = False                               # a feature of a moved object that goes onto a passive one

    def __post_init__(self):
        self.point = np.asarray(self.point, float).reshape(-1)
        self.axis = np.asarray(self.axis, float).reshape(-1)
        if self.across is not None:
            self.across = np.asarray(self.across, float).reshape(-1)
        why = validate(self)
        if why:
            raise ValueError(f"FuncTarget {self.owner}/{self.name}: {why}")

    def frame(self) -> np.ndarray:
        """Its 4x4 in the owner's root frame under ``CONV`` (across chosen perpendicular to axis when absent)."""
        a, sgn, c = CONV[self.type]
        x = self.across if self.across is not None else _any_perp(self.axis)
        R = _R_from(a, sgn * self.axis, c, x)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, self.point
        return T

    def sources(self) -> set:
        return {p["source"] for p in self.provenance}


def _finite(v, n=None) -> bool:
    a = np.asarray(v, float).reshape(-1)
    return bool(np.all(np.isfinite(a))) and (n is None or a.size == n)


def validate(t: FuncTarget) -> str | None:
    """Why ``t`` is not a valid target, or None."""
    if t.type not in TYPES:
        return f"type {t.type!r} not in {TYPES}"
    if not (isinstance(t.name, str) and t.name and isinstance(t.owner, str) and t.owner):
        return "name and owner must be non-empty strings"
    if not _finite(t.point, 3):
        return "point must be 3 finite numbers"
    if not _finite(t.axis, 3) or abs(float(np.linalg.norm(t.axis)) - 1.0) > TOL:
        return "axis must be a finite unit vector"
    if t.across is not None:
        if not _finite(t.across, 3) or abs(float(np.linalg.norm(t.across)) - 1.0) > TOL:
            return "across must be a finite unit vector"
        if abs(float(t.across @ t.axis)) > TOL:
            return "across must be perpendicular to axis"
    if t.type == "slot" and t.across is None:
        return "a slot needs across (its long direction)"
    if not isinstance(t.extent, dict) or set(t.extent) - EXTENT_KEYS:
        return f"extent keys must be among {sorted(EXTENT_KEYS)}"
    if any(not _finite(v, 1) or float(v) <= 0 for v in t.extent.values()):
        return "extent values must be finite and > 0"
    if t.lip is not None:
        L = np.asarray(t.lip, float)
        if L.ndim != 2 or L.shape[1] != 2 or not len(L) or not np.all(np.isfinite(L)):
            return "lip must be [[depth_m, half_width_m], ...]"
        if np.any(L[:, 0] < 0) or np.any(np.diff(L[:, 0]) <= 0) or np.any(L[:, 1] <= 0):
            return "lip depths must ascend from >= 0 and half widths be > 0"
    if t.surface is not None:
        if t.type != "support":
            return "surface is for a support only"
        s = t.surface
        if set(s) - {"z_min", "z_max", "flat_tol", "rim_height"} or not all(_finite(v, 1) for v in s.values()):
            return "surface: z_min, z_max, flat_tol, rim_height, finite"
        if "z_min" in s and "z_max" in s and s["z_min"] > s["z_max"]:
            return "surface z_min > z_max"
        if s.get("flat_tol", 0) < 0 or s.get("rim_height", 0) < 0:
            return "surface flat_tol and rim_height must be >= 0"
    if t.corners is not None:
        if len(t.corners) < 1 or any(not _finite(c, 7) or np.linalg.norm(np.asarray(c, float)[3:]) < 1e-9
                                     for c in t.corners):
            return "corners must be pose7 (xyz + a nonzero wxyz)"
    if t.type in NO_ORIENTATION and t.joint is None:
        return f"a {t.type} needs its joint"
    if t.joint is not None:
        j = t.joint
        r = j.get("range")
        if j.get("kind") not in JOINT_KINDS or r is None or not _finite(r, 2) or not float(r[0]) < float(r[1]):
            return f"joint needs kind in {JOINT_KINDS} and range [lo < hi]"
    if t.owner_fixed not in (True, False, None):
        return "owner_fixed must be True, False or None (unknown)"
    s = t.sigma or {}
    if set(s) != {"pos", "ang"} or not _finite(s["pos"], 3) or not _finite(s["ang"], 2) \
            or min(list(s["pos"]) + list(s["ang"])) < 0:
        return "sigma must be {pos: [across, along, axis] m, ang: [about_axis, tilt] deg}, all >= 0"
    if not (_finite(t.conf, 1) and 0.0 <= float(t.conf) <= 1.0):
        return "conf must be in [0, 1]"
    if not isinstance(t.provenance, list) or not t.provenance:
        return "provenance must be a non-empty list"
    for p in t.provenance:
        if not isinstance(p, dict) or p.get("source") not in SOURCES:
            return f"each provenance entry needs a source in {SOURCES}"
        if not isinstance(p.get("step"), int) or isinstance(p.get("step"), bool) or p["step"] < 0:
            return "each provenance entry needs an integer step >= 0"
        missing = [k for k in REQUIRED[p["source"]] if not p.get(k)]
        if missing:
            return f"a {p['source']} provenance entry needs {missing}"
    return None


# ==================================================================================================== geometry
def _q2R(q) -> np.ndarray:
    from dojo.skills.geometry import quat_to_R
    return quat_to_R(np.asarray(q, float))


def _R2q(R) -> np.ndarray:
    from dojo.skills.geometry import R_to_quat
    q = np.asarray(R_to_quat(R), float)
    return q if q[0] >= 0 else -q


def _pose7_T(c) -> np.ndarray:
    c = np.asarray(c, float).reshape(-1)
    T = np.eye(4)
    T[:3, 3] = c[:3]
    T[:3, :3] = _q2R(c[3:7]) if c.size >= 7 else np.eye(3)
    return T


def _T_pose7(T) -> list:
    return [float(v) for v in T[:3, 3]] + [float(v) for v in _R2q(T[:3, :3])]


def _any_perp(a: np.ndarray) -> np.ndarray:
    e = np.eye(3)[int(np.argmin(np.abs(a)))]
    v = e - a * float(e @ a)
    return v / np.linalg.norm(v)


def _R_from(a: int, col_a: np.ndarray, c: int, col_c: np.ndarray) -> np.ndarray:
    """The rotation whose column a is col_a and column c is col_c (made perpendicular), right-handed."""
    za = np.asarray(col_a, float) / np.linalg.norm(col_a)
    xc = np.asarray(col_c, float) - za * float(np.asarray(col_c, float) @ za)
    xc /= np.linalg.norm(xc)
    b = 3 - a - c
    eps = float(np.eye(3)[b] @ np.cross(np.eye(3)[a], np.eye(3)[c]))
    R = np.zeros((3, 3))
    R[:, a], R[:, c], R[:, b] = za, xc, eps * np.cross(za, xc)
    return R


def _from_frame(typ: str, T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, sgn, c = CONV[typ]
    return T[:3, 3].copy(), sgn * T[:3, a], T[:3, c].copy()


# ==================================================================================================== adapter
def default_type(tag: str, section: str) -> str | None:
    """A target type from a frame tag's words (``peg_0`` -> peg, ``point_1`` -> strike_point), else None; a support
    entry with no type word is a hole mouth (what insert_each reads there today)."""
    words = [w for w in str(tag).lower().replace("/", "_").replace("-", "_").split("_") if w]
    for w in words:
        if w in TYPES:
            return w
        if w in ("point", "points"):
            return "strike_point"
        if w in ("opening", "insert"):
            return "slot"
        if w in ("outlet", "plug"):
            return "socket"
    return "hole" if section == "support" else None


def from_asset(asset_frames: dict, root_to_bbox, owner: str, step: int = 0, key: str = "", catalogue_hash: str = "",
               type_of=None, joints_by_tag: dict | None = None, conf: float = 1.0) -> list[FuncTarget]:
    """Targets from asset frames (``{"functional": {tag: {"frame": [pose7], ...}}, "active_functional": {...},
    "support": {tag: [(pose7 | xyz, radius), ...]}}``, owner root frame) with catalogue provenance. ``root_to_bbox``:
    the owner's bbox centre in its root frame (catalogue metadata), carried in every target's provenance so a
    consumer holding a bbox-centred pose can place it. ``type_of(tag, section) -> type`` (default ``default_type``);
    a tag it cannot type is skipped (the catalogue extraction decides types; an agenda item). A slot tag with several
    frames is one target with corners; any other tag gives one target per frame (``tag`` or ``tag/k``); a lever /
    hinge tag needs its joint in ``joints_by_tag``, else it is skipped."""
    type_of = type_of or default_type
    r2b = [float(v) for v in np.asarray(root_to_bbox, float).reshape(3)]
    out = []

    def prov(section, tag, k, extras):
        return [{"source": "catalogue", "step": int(step), "key": key or "unkeyed", "tag": tag,
                 "catalogue_hash": catalogue_hash or "unhashed", "section": section, "index": k,
                 "root_to_bbox": r2b, "extras": extras}]
    for section in ("functional", "active_functional"):
        for tag, raw in sorted((asset_frames.get(section) or {}).items()):
            typ = type_of(tag, section)
            if typ is None:
                continue
            frames = [np.asarray(f, float) for f in (raw or {}).get("frame") or []]
            extras = {k: v for k, v in (raw or {}).items() if k != "frame"}
            if typ in CORNER_TYPES and len(frames) > 1:
                Ts = [_pose7_T(f) for f in frames]
                p, ax, ac = _from_frame(typ, Ts[0])
                P = np.array([T[:3, 3] for T in Ts])
                w_dir = np.cross(ax, ac)
                ext = {}
                ln, wd = float(np.ptp(P @ ac)), float(np.ptp(P @ w_dir))
                if ln > 0:
                    ext["length"] = ln
                if wd > 0:
                    ext["width"] = wd
                out.append(FuncTarget(tag, owner, typ, P.mean(axis=0), ax, prov(section, tag, -1, extras),
                                      across=ac, extent=ext, corners=[_T_pose7(T) for T in Ts], conf=conf,
                                      active=section == "active_functional"))
                continue
            for k, f in enumerate(frames):
                T = _pose7_T(f)
                p, ax, ac = _from_frame(typ, T)
                jt = None
                if typ in NO_ORIENTATION:
                    j = (joints_by_tag or {}).get(tag) or {}
                    if not j:
                        continue
                    ax = np.asarray(j["axis"], float) / np.linalg.norm(j["axis"])
                    ac = None
                    jt = {"kind": j["type"], "range": [float(j["lower"]), float(j["upper"])], "name": j.get("name")}
                out.append(FuncTarget(tag if len(frames) == 1 else f"{tag}/{k}", owner, typ, p, ax,
                                      prov(section, tag, k, extras), across=ac, joint=jt, conf=conf,
                                      active=section == "active_functional"))
    for tag, lst in sorted((asset_frames.get("support") or {}).items()):
        typ = type_of(tag, "support")
        if typ is None:
            continue
        for k, (c, r) in enumerate(lst):
            T = _pose7_T(c)
            p, ax, ac = _from_frame(typ, T)
            ext = {"radius": float(r)} if float(r) > 0 else {}
            out.append(FuncTarget(tag if len(lst) == 1 else f"{tag}/{k}", owner, typ, p, ax,
                                  prov("support", tag, k, {}), across=ac, extent=ext, conf=conf))
    return out


def _cat(t: FuncTarget) -> dict | None:
    return next((p for p in t.provenance if p["source"] == "catalogue" and "section" in p), None)


def slot_corners(t: FuncTarget) -> list:
    """A slot's footprint corners as pose7: its own when it has them, else the rectangle of its width x length about
    its point in the plane across x (axis x across), at its frame's orientation."""
    if t.corners is not None:
        return [list(map(float, c)) for c in t.corners]
    T = t.frame()
    w_dir = np.cross(t.axis, t.across)
    hl, hw = t.extent.get("length", 0.0) / 2, t.extent.get("width", 0.0) / 2
    q = list(map(float, _R2q(T[:3, :3])))
    return [[float(v) for v in t.point + sl * hl * t.across + sw * hw * w_dir] + q
            for sl, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def as_frames(targets: list[FuncTarget]) -> dict:
    """Targets written into the layout the frame verbs read: ``{"functional": {tag: {"frame": [pose7], ...extras}},
    "active_functional": {...}, "support": {tag: [(pose7, radius)]}, "joints_by_tag": {tag: {name, type, axis,
    lower, upper}}}``. A catalogue target goes back to its section / tag / index; any other target to
    ``SECTION_OF[type]`` (active ones to active_functional) under its name. A lever / hinge frame carries the target's
    point at the identity orientation (its axis is the joint's)."""
    out = {"functional": {}, "active_functional": {}, "support": {}, "joints_by_tag": {}}
    slots = {}
    for t in targets:
        c = _cat(t)
        section = c["section"] if c else ("active_functional" if t.active else SECTION_OF[t.type])
        tag = c["tag"] if c else t.name
        k = c["index"] if c else 0
        if t.type in NO_ORIENTATION:
            pose = [float(v) for v in t.point] + [1.0, 0.0, 0.0, 0.0]
            j = t.joint
            out["joints_by_tag"][tag] = {"name": j.get("name"), "type": j["kind"], "axis": [float(v) for v in t.axis],
                                         "lower": float(j["range"][0]), "upper": float(j["range"][1])}
        else:
            pose = _T_pose7(t.frame())
        if section == "support":
            out["support"].setdefault(tag, {})[k] = (pose, float(t.extent.get("radius", 0.0)))
            continue
        entry = slots.setdefault((section, tag), {"frames": {}, "extras": dict((c or {}).get("extras") or {})})
        if t.type in CORNER_TYPES and (t.corners is not None or c is None):
            for i, cp in enumerate(slot_corners(t)):
                entry["frames"][(k, i)] = cp
        else:
            entry["frames"][(k, 0)] = pose
    for (section, tag), e in slots.items():
        out[section][tag] = {"frame": [e["frames"][kk] for kk in sorted(e["frames"])], **e["extras"]}
    out["support"] = {tag: [d[k] for k in sorted(d)] for tag, d in out["support"].items()}
    return out


def to_asset_info(base, frames: dict):
    """A copy of ``AssetInfo`` ``base`` whose functional / active_functional / support are ``frames``' (so today's
    accessors, ``library.functional_frames``, ``precision.hole_frames``, ``precision.frame_points_w``, read targets)."""
    import copy
    a = copy.copy(base)
    a.functional = dict(frames.get("functional") or {})
    a.active_functional = dict(frames.get("active_functional") or {})
    a.support = {k: [(np.asarray(c, float), float(r)) for c, r in v] for k, v in (frames.get("support") or {}).items()}
    return a


# ==================================================================================================== verb-side check
K_SIGMA = 2.0              # a component is within tolerance when K_SIGMA x its 1-sigma is at most the tolerance
NARROW = 0.002             # m: clearance below which a slot or hole is narrow (half perception's 4 mm hull voxel): its
#                            width must come from an identity, and without one the verb measures first, whatever sigma
MEASURES = {               # what each measurement observes (components), and whether it needs a fixed owner
    "wall_squeeze": ({"ang.about_axis", "pos.across"}, True),       # tactile, COIN2's wall probe
    "wrist_look": ({"pos.across", "pos.along", "ang.about_axis", "ang.tilt"}, False),
    "guarded_descent": ({"pos.axis"}, False),
}


def slot_insert(width: float, length: float, thickness: float) -> dict:
    """Tolerances of a thin object (``thickness``) standing in a slot: across = half the clearance, about the slot's
    axis = atan(clearance / length) (COIN2 markup 1.5: 2.6 mm, ~28 mm, a 1.94 mm disc -> about 1.4 deg)."""
    cl = float(width) - float(thickness)
    return {"pos.across": cl / 2.0, "ang.about_axis": math.degrees(math.atan2(cl, float(length))) if cl > 0 else 0.0,
            "_clearance": cl}


def hole_insert(radius: float, obj_radius: float, depth: float | None = None) -> dict:
    """Tolerances of a round object in a round hole: across and along = the radial clearance; tilt = atan(2 x
    clearance / depth) when the depth is known."""
    cl = float(radius) - float(obj_radius)
    tol = {"pos.across": cl, "pos.along": cl, "_clearance": 2 * cl}
    if depth:
        tol["ang.tilt"] = math.degrees(math.atan2(2 * cl, float(depth))) if cl > 0 else 0.0
    return tol


def peg_thread(peg_radius: float, hole_radius: float) -> dict:
    cl = float(hole_radius) - float(peg_radius)
    return {"pos.across": cl, "pos.along": cl, "_clearance": 2 * cl}


def stand_place(base_radius: float) -> dict:
    """ORG2 markup 2: a stand place holds when the base lands within 0.5 x its radius of the stand's centre."""
    return {"pos.across": 0.5 * float(base_radius), "pos.along": 0.5 * float(base_radius)}


def support_release(gap: float = 0.003) -> dict:
    """ORG2 markup 2: an open-loop release above a resting surface needs its height within the release gap."""
    return {"pos.axis": float(gap)}


@dataclass
class Decision:
    action: str                       # go | measure_first | refuse
    why: str
    over: list                        # the components out of tolerance (or all of them, when measuring is mandatory)
    measure: str | None = None        # the measurement to take first


def _sigma_of(t: FuncTarget, comp: str, owner_sigma: dict | None) -> float:
    kind, name = comp.split(".")
    i = (SIGMA_POS if kind == "pos" else SIGMA_ANG).index(name)
    s = float(t.sigma[kind][i])
    o = float((owner_sigma or {}).get(comp, 0.0))
    return math.hypot(s, o)


def measured(t: FuncTarget) -> bool:
    """Has the target been measured up close: a tactile entry, or a perception entry from a wrist look
    (``"look": "wrist"``)?"""
    return any(p["source"] == "tactile" or (p["source"] == "perception" and p.get("look") == "wrist")
               for p in t.provenance)


def decide(t: FuncTarget, tol: dict, available=("wrist_look",), steps_left: int | None = None,
           reserve: dict | None = None, owner_sigma: dict | None = None, k: float = K_SIGMA) -> Decision:
    """go when every toleranced component's k-sigma (target and owner pose combined) is within its tolerance.
    Otherwise measure first with the cheapest available measurement that observes every component out of tolerance
    (``wall_squeeze`` only on an owner known fixed), if the steps left cover its reserve. A narrow opening
    (clearance < NARROW) with no catalogue provenance (no accepted identity) measures first until it has been
    measured up close (``measured``), whatever its sigma says; after that its sigma decides. Refuse when the
    object does not fit, or nothing available covers what is out, or the steps left are below the reserve."""
    clearance = tol.get("_clearance")
    comps = [c for c in tol if not c.startswith("_")]
    if clearance is not None and clearance <= 0:
        return Decision("refuse", f"it does not fit: clearance {clearance * 1000:.2f} mm", comps)
    if any(tol[c] <= 0 for c in comps):
        return Decision("refuse", "a tolerance is zero", comps)
    over = [c for c in comps if k * _sigma_of(t, c, owner_sigma) > tol[c]]
    narrow = clearance is not None and clearance < NARROW and "catalogue" not in t.sources() and not measured(t)
    if narrow:
        over = comps
    if not over:
        return Decision("go", "within tolerance: " + ", ".join(
            f"{c} {k:.0f}x{_sigma_of(t, c, owner_sigma):.4g} <= {tol[c]:.4g}" for c in comps), [])
    why = ("narrow opening without an identity: measure first" if narrow else
           "out of tolerance: " + ", ".join(f"{c} {k:.0f}x{_sigma_of(t, c, owner_sigma):.4g} > {tol[c]:.4g}"
                                             for c in over))
    usable = []
    for m in available:
        if m not in MEASURES:
            continue
        sees, needs_fixed = MEASURES[m]
        if needs_fixed and t.owner_fixed is not True:
            continue
        if set(over) <= sees:
            usable.append(m)
    if not usable:
        return Decision("refuse", why + "; no available measurement observes " + ", ".join(over), over)
    for m in usable:
        need = int((reserve or {}).get(m, 0))
        if steps_left is None or steps_left >= need:
            return Decision("measure_first", why, over, m)
    return Decision("refuse", why + f"; {steps_left} steps left, below the probe reserve", over)


__all__ = ["FuncTarget", "validate", "from_asset", "as_frames", "to_asset_info", "slot_corners", "default_type",
           "decide", "Decision", "measured", "slot_insert", "hole_insert", "peg_thread", "stand_place", "support_release", "CONV",
           "TYPES", "MEASURES", "NARROW", "K_SIGMA"]
