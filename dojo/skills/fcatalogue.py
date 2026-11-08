"""f-catalogue/1: the catalogue-wide extraction of functional frames, joints and mesh-derived contact geometry.

One JSON file per catalogue build, the input contract of the partner team's F adapter (their ``fadapter.py``; spec
``f-adapter-spec.md`` section 1, agreed split of 7 Nov 23:12Z). Our side owns this file: the extraction, the frame
convention per type (``CONV``), the tag -> type map (``tag_type``) and the round trip against the raw metadata.

What it reads (``reader`` / ``read_*``; nothing else is ever opened, and every root is checked before the walk):
  * RoboDojo's ``Assets/Object/RoboDojo/<Type>/<category>/<idx>/``: ``metadata.json`` for every key, and for
    Articulations ``object.usd[z]`` (the articulation-root prim and the joints, through ``pxr``);
  * perception's catalogue: ``index.json`` (the build hash, ``catalogue_hash``) and ``<key>/geom.npz`` (the mesh,
    converted from ``object.usd`` catalogue-wide by ``dojo.perception.catalog``).
Nothing else is opened: no other file of the catalogue (its annotation files included), and nothing outside the two
roots; the denied path components are ``dojo.legality.roots``'. Every key under the asset root is listed; a key without
usable metadata is reported in ``build.sources.skipped``, never silently dropped.

Frames: everything in the file is in the asset's ROOT frame, the prim RoboDojo's official pose reads (the
articulation-root prim for an Articulation, ``dojo.skills.artroot``; the default prim otherwise). The raw metadata,
meshes and USD joints are in the default-prim frame; ``root_in_asset`` (pose7, Articulations only) maps one to the
other. Units: m, rad (revolute), m (prismatic).

Types: ``tag_type`` reads only a tag's words and its frame structure (section, number of frames, a joint), plus one
catalogue-wide relation for an active frame with no feature word (the passive tags that name its owner's category). No
task name or task file is consulted; tags whose words name a result check or a goal role are typed null and listed
(``CODEX_WORDS``) for a ruling.

Derived contact geometry (``derived``; every entry names its method and parameters): the usable resting surface on
top (``@top``, support), the lowest band (``@base``, base), and per opening tag the mesh-fit width / length / depth
and the holder's outer wall profile (``lip``). Alternatives are kept in ``SURFACE_METHODS`` / ``compare_*`` for the
method comparison (docs/experiments/2026-11-08-f-catalogue.md); the file carries the chosen ones only.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import time
from collections import deque
from pathlib import Path

import numpy as np

SCHEMA = "f-catalogue/1"
RULES_VERSION = "fcat-rules-1"     # bump with any change to the type map, CONV or a derived method
TYPES = ("hole", "slot", "socket", "keyhole", "peg", "hook", "handle", "rim", "strike_point", "lever", "hinge",
         "support", "base")
SECTIONS = ("functional", "active_functional", "support")
UNITS = {"length": "m", "angle_revolute": "rad", "travel_prismatic": "m"}
ASSET_TYPES = ("Rigid", "Geometry", "Articulation", "Clutter", "Dynamic", "Garment", "Fluid")

# ============================================================================================ legality: what is opened
ASSET_FILES = ("metadata.json", "object.usd", "object.usdz")
CATALOG_FILES = ("geom.npz", "index.json")
from dojo.legality.roots import forbidden_part  # noqa: E402  (the denied path components live with the legality tools)


def check_root(path) -> Path:
    """``path`` resolved, after checking the root itself (not only what lies under it) against the forbidden list."""
    p = Path(os.path.expanduser(str(path)))
    for q in (p, p.resolve()):
        bad = forbidden_part(q)
        if bad is not None:
            raise PermissionError(f"f-catalogue: root {q} has a forbidden component {bad!r}")
    p = p.resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"f-catalogue: root {p} is not a directory")
    return p


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reader(assets_root, catalog_root=None) -> dict:
    """The extraction's reading state: every file it opens goes through ``read_*`` below, which allow only
    ``ASSET_FILES`` under the asset root and ``CATALOG_FILES`` under the catalogue root, each hashed into ``opened``
    (path relative to its root, prefixed ``assets/`` or ``catalog/``). Plain functions, no class: the shared audit
    engine resolves an untyped ``x.m()`` call to every method named ``m``, which would pull this offline tool into the
    inference call graph."""
    return {"assets": check_root(assets_root), "catalog": check_root(catalog_root) if catalog_root else None,
            "opened": {}}


def _rel(rd: dict, p) -> str:
    p = Path(p).resolve()
    if forbidden_part(p) is not None:
        raise PermissionError(f"f-catalogue: refusing {p}")
    if p.name in ASSET_FILES and p.is_relative_to(rd["assets"]):
        return "assets/" + str(p.relative_to(rd["assets"]))
    if rd["catalog"] is not None and p.name in CATALOG_FILES and p.is_relative_to(rd["catalog"]):
        return "catalog/" + str(p.relative_to(rd["catalog"]))
    raise PermissionError(f"f-catalogue: {p} is not an allowed asset / catalogue file")


def read_bytes(rd: dict, p) -> bytes:
    rel = _rel(rd, p)
    b = Path(p).read_bytes()
    rd["opened"][rel] = sha256_bytes(b)
    return b


def read_json(rd: dict, p):
    return json.loads(read_bytes(rd, p))


def read_mesh(rd: dict, p) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(io.BytesIO(read_bytes(rd, p)))
    return np.asarray(z["vertices"], float), np.asarray(z["faces"], np.int64)


def read_usd(rd: dict, p, usd_reader=None) -> dict:
    rel = _rel(rd, p)
    rd["opened"][rel] = sha256_file(p)
    return (usd_reader or usd_articulation)(Path(p))


def listing_sha256(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\0{files[rel]}\n".encode())
    return h.hexdigest()


def iter_keys(assets_root: Path):
    """(key, dir) for every ``<Type>/<category>/<idx>`` dir under the asset root (sorted; no filter but the layout)."""
    for t in ASSET_TYPES:
        td = assets_root / t
        if not td.is_dir():
            continue
        for cat in sorted(os.listdir(td)):
            cd = td / cat
            if not cd.is_dir():
                continue
            for idx in sorted(os.listdir(cd)):
                if idx.isdigit() and (cd / idx).is_dir():
                    yield f"{t}/{cat}/{int(idx):05d}", cd / idx


# ============================================================================================ geometry helpers
def quat_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_quat(R) -> list:
    from dojo.skills.geometry import R_to_quat
    q = np.asarray(R_to_quat(np.asarray(R, float)), float)
    return [float(v) for v in (q if q[0] >= 0 else -q)]


def pose7_T(c) -> np.ndarray:
    c = np.asarray(c, float).reshape(-1)
    T = np.eye(4)
    T[:3, 3] = c[:3]
    if c.size >= 7:
        T[:3, :3] = quat_R(c[3:7])
    return T


def T_pose7(T) -> list:
    return [float(v) for v in T[:3, 3]] + R_quat(T[:3, :3])


def root_map(T_root_in_asset: np.ndarray | None = None) -> dict:
    """Asset (default-prim) frame -> root frame: ``x_root = inv(T_root_in_asset) @ x_asset``. Exact pass-through when
    the root is the default prim; a pure translation keeps a raw quaternion's values bit-identical."""
    T = np.eye(4) if T_root_in_asset is None else np.asarray(T_root_in_asset, float)
    rot_identity = bool(np.array_equal(T[:3, :3], np.eye(3)))
    Ti = np.linalg.inv(T)
    if rot_identity:
        Ti[:3, :3] = np.eye(3)
        Ti[:3, 3] = -T[:3, 3]
    return {"T": T, "Ti": Ti, "identity": bool(np.array_equal(T, np.eye(4))), "rot_identity": rot_identity}


def rm_point(rm: dict, p) -> list:
    p = np.asarray(p, float)
    if rm["identity"]:
        return [float(v) for v in p]
    return [float(v) for v in rm["Ti"][:3, :3] @ p + rm["Ti"][:3, 3]]


def rm_vector(rm: dict, v) -> list:
    v = np.asarray(v, float)
    return [float(x) for x in (v if rm["rot_identity"] else rm["Ti"][:3, :3] @ v)]


def rm_pose(rm: dict, c) -> list:
    """A raw pose7 (or xyz) into the root frame, keeping the raw quaternion's values when the map has no rotation."""
    c = [float(v) for v in c]
    if rm["identity"]:
        return c
    if rm["rot_identity"] or len(c) < 7:
        return rm_point(rm, c[:3]) + c[3:]
    return T_pose7(rm["Ti"] @ pose7_T(c))


def rm_back(rm: dict, c) -> list:
    """Root -> asset (for the round trip)."""
    c = [float(v) for v in c]
    if rm["identity"]:
        return c
    if rm["rot_identity"] or len(c) < 7:
        return [float(v) for v in np.asarray(c[:3]) + rm["T"][:3, 3]] + c[3:]
    return T_pose7(rm["T"] @ pose7_T(c))


# ============================================================================================ USD (Articulations)
def usd_articulation(usd_path: Path) -> dict:
    """The articulation-root prim's transform in the default-prim frame (Isaac's ``get_articulation_root_api_prim_
    path``: the first prim with ``PhysicsArticulationRootAPI`` breadth-first from the default prim) and the prismatic /
    revolute joints (axis and origin in the default-prim frame, m; revolute limits in rad, prismatic in m). The same
    reading as ``scripts/skills/articulation_roots.py`` and ``scripts/skills/articulation_links.py``."""
    from pxr import Usd, UsdGeom, UsdPhysics

    st = Usd.Stage.Open(str(usd_path))
    dp = st.GetDefaultPrim()
    if not dp:
        return {"error": "no default prim"}
    mpu = float(UsdGeom.GetStageMetersPerUnit(st) or 1.0)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())

    def M(prim) -> np.ndarray:
        return np.array(xc.GetLocalToWorldTransform(prim), float).T

    W_inv = np.linalg.inv(M(dp))
    out: dict = {"root_prim": None, "root_in_asset": None, "joints": {}, "meters_per_unit": mpu}
    q = deque([dp])
    root = None
    while q:
        p = q.popleft()
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            root = p
            break
        q.extend(p.GetChildren())
    if root is not None:
        out["root_prim"] = str(root.GetPath())
        if root != dp:
            if root.IsA(UsdGeom.Xformable):
                T = W_inv @ M(root)
                R = T[:3, :3] / np.linalg.norm(T[:3, :3], axis=0)
                Tr = np.eye(4)
                Tr[:3, :3], Tr[:3, 3] = R, T[:3, 3] * mpu
                if np.linalg.norm(Tr[:3, 3]) > 1e-5 or np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))) > 0.01:
                    out["root_in_asset"] = T_pose7(Tr)
            else:
                out["root_note"] = f"root API on a non-Xformable {root.GetTypeName()}: the default prim stands"
    for prim in Usd.PrimRange(dp):
        t = prim.GetTypeName()
        if t not in ("PhysicsPrismaticJoint", "PhysicsRevoluteJoint"):
            continue
        b0 = prim.GetRelationship("physics:body0").GetTargets()
        b1 = prim.GetRelationship("physics:body1").GetTargets()
        e = {"X": [1.0, 0, 0], "Y": [0, 1.0, 0], "Z": [0, 0, 1.0]}[str(prim.GetAttribute("physics:axis").Get() or "X")]
        r0 = prim.GetAttribute("physics:localRot0").Get()
        p0 = prim.GetAttribute("physics:localPos0").Get()
        R0 = quat_R([r0.GetReal(), *r0.GetImaginary()]) if r0 is not None else np.eye(3)
        Tb0 = W_inv @ M(st.GetPrimAtPath(b0[0])) if b0 else np.eye(4)
        axis = Tb0[:3, :3] @ R0 @ np.asarray(e)
        origin = Tb0[:3, :3] @ (np.asarray(p0, float) * mpu if p0 is not None else np.zeros(3)) + Tb0[:3, 3] * mpu
        kind = "prismatic" if "Prismatic" in t else "revolute"
        lo, hi = prim.GetAttribute("physics:lowerLimit").Get(), prim.GetAttribute("physics:upperLimit").Get()
        scale = mpu if kind == "prismatic" else math.pi / 180.0
        out["joints"][prim.GetName()] = {
            "type": kind, "body0": b0[0].name if b0 else None, "body1": b1[0].name if b1 else None,
            "axis": [float(v) for v in axis / (np.linalg.norm(axis) or 1.0)], "origin": [float(v) for v in origin],
            "lower": None if lo is None or not np.isfinite(lo) else float(lo) * scale,
            "upper": None if hi is None or not np.isfinite(hi) else float(hi) * scale}
    return out


# ============================================================================================ tag -> type
# general feature words (no task, object or category names): the word's type
LEXICON = {
    "hole": "hole", "holes": "hole",
    "slot": "slot", "slots": "slot", "opening": "slot", "insert": "slot",
    "socket": "socket", "outlet": "socket", "plug": "socket",
    "keyhole": "keyhole",
    "peg": "peg", "pole": "peg", "pin": "peg", "post": "peg",
    "hook": "hook",
    "handle": "handle", "grip": "handle",
    "rim": "rim", "mouth": "rim",
    "hit": "strike_point", "strike": "strike_point", "beat": "strike_point",
    "lever": "lever", "button": "lever", "botton": "lever", "press": "lever", "switch": "lever", "trigger": "lever",
    "hinge": "hinge",
    "support": "support", "cradle": "support",
    "base": "base", "bottom": "base", "foot": "base",
}
# word pairs read together: "<handle>_support" holds a handle by it, i.e. a hook
COMPOUNDS = {("handle", "support"): "hook"}
# words that name a result check or a goal role: the tag is typed null and listed for a Codex ruling
CODEX_WORDS = {"check": "names a point where a result is checked", "checkpoint": "names a point where a result is checked",
               "be_placed": "names a goal role ('to be placed')",
               "same_side": "qualifier relative to an unstated reference (a goal convention?)",
               "diff_side": "qualifier relative to an unstated reference (a goal convention?)"}
ROUND_FREE_TYPES = ("slot",)       # a round support entry (centre + radius) cannot be one of these: "slot" -> hole


def tag_words(tag: str) -> list[str]:
    """``test_tube_slot/3`` -> [test, tube, slot]; ``toast_slot1`` -> [toast, slot]; ``hit_0`` -> [hit]."""
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(tag))
    return [w for w in re.split(r"[_/\-\s]+|(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])", s.lower()) if w and not w.isdigit()]


def codex_flag(tag: str) -> str | None:
    w = tag_words(tag)
    joined = "_".join(w)
    for k, why in CODEX_WORDS.items():
        if k in w or f"_{k}_" in f"_{joined}_":
            return why
    return None


def lexicon_type(words: list[str]) -> tuple[str | None, str]:
    for i in range(len(words) - 1):
        t = COMPOUNDS.get((words[i], words[i + 1]))
        if t:
            return t, f"compound '{words[i]}_{words[i + 1]}'"
    for w in reversed(words):            # the head word is the last feature word: test_tube_slot -> slot
        if w in LEXICON:
            return LEXICON[w], f"word '{w}'"
    return None, "no feature word"


def mates_index(assets_tags: dict[str, dict]) -> dict[str, set]:
    """Catalogue-wide: per word sequence, the types of passive tags (and the categories of their owners) that name it,
    e.g. "hammer" -> {slot} from a ``hammer_slot`` tag. Input: {key: {section: [tags]}}; reads names only."""
    idx: dict[str, set] = {}
    for key, secs in assets_tags.items():
        cat_words = tag_words(key.split("/")[1])
        for section in ("functional", "support"):
            for tag in secs.get(section) or []:
                t, _ = lexicon_type(tag_words(tag))
                if section == "support" and t is None:
                    t = "support"
                if t is None or codex_flag(tag):
                    continue
                words = tag_words(tag)
                for seq in (words, cat_words):
                    for i in range(len(seq)):
                        for j in range(i + 1, len(seq) + 1):
                            idx.setdefault("_".join(seq[i:j]), set()).add(t)
    return idx


def tag_type(tag: str, section: str, raw, owner_key: str, mates: dict | None = None,
             joint: dict | None = None) -> tuple[str | None, str]:
    """(type | None, the rule that decided it) for one tag of one asset. Grammar first (``LEXICON`` / ``COMPOUNDS``
    on the tag's words, head word last), then frame structure: no frame -> null; a support entry is a circle, so a
    round entry is never a slot (-> hole) and a support group without a feature word is a support; a ``bottom`` that
    outlines a region (several frames) is a support, not a base; a lever / hinge needs a joint with limits. An active
    frame with no feature word takes the passive type that the catalogue's tags give its owner's category
    (``mates_index``: ``hammer`` -> ``hammer_slot`` -> slot). Result-check / goal-role words -> null (``CODEX_WORDS``)."""
    why = codex_flag(tag)
    if why:
        return None, f"null for Codex: {why}"
    if section == "support":
        n = len((raw or {}).get("center") or []) if isinstance(raw, dict) else len(raw or [])
        if not n:
            return None, "null: no support entry"
        t, rule = lexicon_type(tag_words(tag))
        if t is None:
            return "support", "support section, no feature word"
        if t in ROUND_FREE_TYPES:
            return "hole", f"{rule} in a round support entry (centre + radius): hole"
        return t, f"{rule} (support section)"
    frames = (raw or {}).get("frame") if isinstance(raw, dict) else None
    if not frames:
        return None, "null: no frame (not a target)"
    words = tag_words(tag)
    t, rule = lexicon_type(words)
    if t is None and section == "active_functional" and mates:
        cat = "_".join(tag_words(owner_key.split("/")[1]))
        got = sorted(mates.get(cat) or ())
        if len(got) == 1:
            t, rule = got[0], f"active frame, no feature word: the catalogue's passive '{cat}' tags are {got[0]}"
        elif got:
            return None, f"null: active frame, owner category '{cat}' mates with several types {got}"
    if t is None:
        return None, f"null: {rule}"
    if t == "base" and len(frames) > 1:
        return "support", f"{rule} outlining a region ({len(frames)} frames): support"
    if t in ("lever", "hinge"):
        if not joint:
            return None, f"null: {rule} -> {t}, but its parent_joint has no USD joint"
        if joint.get("lower") is None or joint.get("upper") is None or not joint["lower"] < joint["upper"]:
            return None, f"null: {rule} -> {t}, but its joint has no limits"
    return t, rule


# ============================================================================================ CONV
# type -> (axis column, sign, across column): axis = sign * R[:, a], across = R[:, c]. Pinned by measurement over the
# catalogue (docs/experiments/2026-11-08-f-catalogue.md): the column of each typed frame that points along its
# semantic direction on an upright owner (into an opening = down; up a peg / support / strike face; base = down).
# Frames that follow another column get a per-(asset, tag) override (``tag_conv``), measured the same way.
CONV = {
    "hole": (2, -1, 0), "slot": (2, -1, 0), "socket": (2, -1, 0), "keyhole": (2, -1, 0), "rim": (2, -1, 0),
    "peg": (2, 1, 0), "support": (2, 1, 0), "strike_point": (2, 1, 0),
    "base": (2, -1, 0),
    "hook": (2, 1, 0), "handle": (2, 1, 0), "lever": (2, 1, 0), "hinge": (2, 1, 0),
}
SEMANTIC = {"hole": -1, "slot": -1, "socket": -1, "keyhole": -1, "rim": -1, "base": -1,
            "peg": 1, "support": 1, "strike_point": 1}     # sign of the axis along the owner's up; others: none
CONV_TOL_DEG = 25.0


def upright(place: dict | None) -> bool:
    """Does the asset rest with its own +z up in one of its ``active.place`` orientations (within 10 deg)? Without
    place frames the authored frame is taken as upright."""
    if not place:
        return True
    for v in place.values():
        q = (v or {}).get("orientation")
        if q is None:
            continue
        try:
            up = quat_R(q).T @ np.array([0.0, 0, 1])
        except ValueError:
            continue
        if up[2] > math.cos(math.radians(10.0)):
            return True
    return False


def measure_axis(R: np.ndarray, typ: str) -> tuple[int, int, float] | None:
    """(column, sign, angle deg) of the frame column closest to the type's semantic direction (root +z times
    ``SEMANTIC``), or None for a type without one."""
    s = SEMANTIC.get(typ)
    if s is None:
        return None
    d = np.array([0.0, 0, float(s)])
    best = max(((c, sg, float(sg * R[:, c] @ d)) for c in range(3) for sg in (1, -1)), key=lambda x: x[2])
    return best[0], best[1], math.degrees(math.acos(max(-1.0, min(1.0, best[2]))))


def conv_valid(conv: dict) -> str | None:
    """Why a conv table is unusable (the adapter's refusal rule), or None."""
    for t in TYPES:
        r = conv.get(t)
        if r is None or len(r) != 3:
            return f"conv misses {t}"
        a, s, c = r
        if a not in (0, 1, 2) or c not in (0, 1, 2) or a == c or s not in (1, -1):
            return f"conv {t} {r} is not a pair of distinct columns with a sign"
    return None


# ============================================================================================ mesh: derived geometry
def _sample_chunks(V, F, step: float, max_seg: int = 400, chunk: int = 2_000_000):
    """Points on every triangle about ``step`` apart (plus the vertices), in chunks of at most ~``chunk`` points."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    yield V
    if not len(F):
        return
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    L = np.maximum(np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1)), np.linalg.norm(c - b, axis=1))
    nseg = np.clip(np.ceil(L / step), 1, max_seg).astype(int)
    for n in np.unique(nseg):
        idx = np.flatnonzero(nseg == n)
        u, v = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
        k = (u + v) <= 1.0
        u, v = u[k], v[k]
        per = max(1, chunk // len(u))
        for i in range(0, len(idx), per):
            sl = idx[i:i + per]
            P = a[sl][:, None, :] + u[None, :, None] * (b[sl] - a[sl])[:, None, :] + v[None, :, None] * (c[sl] - a[sl])[:, None, :]
            yield P.reshape(-1, 3)


def surface_samples(V, F, step: float, max_seg: int = 400) -> np.ndarray:
    """Points on every triangle about ``step`` apart (plus the vertices)."""
    return np.vstack(list(_sample_chunks(V, F, step, max_seg)))


def top_height_map(V, F, cell: float) -> tuple[np.ndarray, float, float]:
    """Max z per ``cell`` over dense surface samples (root frame, z up); -inf where nothing projects. No dilation."""
    V = np.asarray(V, float)
    x0, y0 = float(V[:, 0].min()), float(V[:, 1].min())
    nx = int(math.floor((float(V[:, 0].max()) - x0) / cell)) + 1
    ny = int(math.floor((float(V[:, 1].max()) - y0) / cell)) + 1
    Z = np.full((ny, nx), -np.inf)
    for P in _sample_chunks(V, F, 0.7 * cell):
        ix = np.clip(np.floor((P[:, 0] - x0) / cell).astype(int), 0, nx - 1)
        iy = np.clip(np.floor((P[:, 1] - y0) / cell).astype(int), 0, ny - 1)
        np.maximum.at(Z, (iy, ix), P[:, 2])
    return Z, x0, y0


def label4(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected components of a boolean grid."""
    lab = np.zeros(mask.shape, int)
    n = 0
    H, W = mask.shape
    for i, j in zip(*np.nonzero(mask)):
        if lab[i, j]:
            continue
        n += 1
        lab[i, j] = n
        st = [(i, j)]
        while st:
            a, b = st.pop()
            for u, v in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if 0 <= u < H and 0 <= v < W and mask[u, v] and not lab[u, v]:
                    lab[u, v] = n
                    st.append((u, v))
    return lab, n


def _dilate(m: np.ndarray, k: int) -> np.ndarray:
    out = m.copy()
    for _ in range(k):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


ISOTROPIC = 0.98     # singular-value ratio above which a region's long direction is taken as the root x axis


def _region_summary(P_xy: np.ndarray, z: np.ndarray, area: float, cell: float = 0.0) -> dict:
    c = P_xy.mean(0)
    along = np.array([1.0, 0.0])
    if len(P_xy) >= 2:
        _, sv, Vt = np.linalg.svd(P_xy - c, full_matrices=False)
        # a near-isotropic region has no long direction (and an SVD free to pick any): the root x axis then
        if len(sv) > 1 and sv[0] > 0 and sv[1] / sv[0] < ISOTROPIC:
            along = Vt[0]
    if along[0] < 0 or (along[0] == 0 and along[1] < 0):
        along = -along
    thin = np.array([-along[1], along[0]])
    pa, pt = (P_xy - c) @ along, (P_xy - c) @ thin
    c = c + along * (pa.max() + pa.min()) / 2 + thin * (pt.max() + pt.min()) / 2
    return {"xy": c, "along": along, "length": float(np.ptp(pa) + cell), "width": float(np.ptp(pt) + cell),
            "z_min": float(z.min()), "z_max": float(z.max()), "z": float(np.median(z)), "area": float(area)}


def resting_surface_heightmap(V, F, cell: float = 0.003, flat_tol: float = 0.004, min_area: float = 4e-4,
                              n_levels: int = 8, hmap=None) -> dict:
    """The usable resting surface seen from above: the largest 4-connected region of the top height map whose heights
    lie within ``flat_tol`` (ORG2's "flat within 4 mm" on ``precision.height_map``'s 3 mm cells), searched over the
    ``n_levels`` most populated height bands. ``rim_height``: how far the ring just outside it rises above it."""
    Z, x0, y0 = hmap if hmap is not None else top_height_map(V, F, cell)
    fin = np.isfinite(Z)
    if not fin.any():
        return {"refused": "empty height map"}
    zs = Z[fin]
    edges = np.arange(zs.min(), zs.max() + flat_tol, flat_tol / 2)
    if len(edges) < 2:
        edges = np.array([zs.min(), zs.min() + flat_tol])
    h, _ = np.histogram(zs, bins=edges)
    best = None
    for b in np.argsort(-h)[:n_levels]:
        if h[b] == 0:
            continue
        lv = float(np.median(zs[(zs >= edges[b]) & (zs <= edges[b + 1])]))
        m = fin & (np.abs(Z - lv) <= flat_tol / 2)
        lab, n = label4(m)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        k = int(np.argmax(sizes)) + 1
        key = (int(sizes[k - 1]), lv)
        if best is None or key > best[0]:
            best = (key, lab == k)
    if best is None:
        return {"refused": "no flat region"}
    reg = best[1]
    iy, ix = np.nonzero(reg)
    area = float(reg.sum()) * cell * cell
    if area < min_area:
        return {"refused": f"largest flat region {area * 1e4:.1f} cm2 < {min_area * 1e4:.1f} cm2"}
    s = _region_summary(np.c_[x0 + (ix + 0.5) * cell, y0 + (iy + 0.5) * cell], Z[reg], area, cell)
    ring = _dilate(reg, 3) & ~reg & fin
    s["rim_height"] = float(max(0.0, Z[ring].max() - s["z_max"])) if ring.any() else 0.0
    s["top"] = float(zs.max())
    return s


def resting_surface_slice(V, F, flat_tol: float = 0.004, min_area: float = 4e-4, max_tilt_deg: float = 10.0) -> dict:
    """Mesh slicing: the horizontal band (``flat_tol`` thick) holding the most up-facing triangle area, visible or
    not (an internal shelf counts); its triangles' extent."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    A = np.linalg.norm(n, axis=1) / 2
    up = (A > 0) & (n[:, 2] / np.maximum(2 * A, 1e-30) > math.cos(math.radians(max_tilt_deg)))
    if not up.any():
        return {"refused": "no up-facing triangles"}
    zc = ((a + b + c) / 3)[up, 2]
    edges = np.arange(zc.min(), zc.max() + flat_tol, flat_tol / 2)
    if len(edges) < 2:
        edges = np.array([zc.min(), zc.min() + flat_tol])
    h, _ = np.histogram(zc, bins=edges, weights=A[up])
    k = int(np.argmax(h))
    lv = (edges[k] + edges[k + 1]) / 2
    sel = np.flatnonzero(up)[np.abs(zc - lv) <= flat_tol / 2]
    area = float(A[sel].sum())
    if area < min_area:
        return {"refused": f"largest up-facing band {area * 1e4:.1f} cm2 < {min_area * 1e4:.1f} cm2"}
    P = np.vstack([a[sel], b[sel], c[sel]])
    s = _region_summary(P[:, :2], P[:, 2], area)
    s["rim_height"] = float(max(0.0, V[:, 2].max() - s["z_max"]))
    s["top"] = float(V[:, 2].max())
    return s


def _cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _hull2(P: np.ndarray) -> np.ndarray:
    P = np.unique(np.round(P, 9), axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and _cross2(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h
    lo, hi = half(P), half(P[::-1])
    return np.array(lo[:-1] + hi[:-1])


def resting_surface_hull(V, F, flat_tol: float = 0.004, min_area: float = 4e-4) -> dict:
    """Convex hull: the hull's top facet, the hull of the vertices within ``flat_tol`` of the highest point (what a
    camera's silhouette top or the bbox gives)."""
    V = np.asarray(V, float)
    zt = float(V[:, 2].max())
    P = V[V[:, 2] >= zt - flat_tol]
    H = _hull2(P[:, :2])
    area = 0.5 * abs(float(np.sum(H[:, 0] * np.roll(H[:, 1], -1) - np.roll(H[:, 0], -1) * H[:, 1]))) if len(H) >= 3 else 0.0
    if area < min_area:
        return {"refused": f"hull top facet {area * 1e4:.1f} cm2 < {min_area * 1e4:.1f} cm2"}
    s = _region_summary(P[:, :2], P[:, 2], area)
    s["rim_height"] = 0.0
    s["top"] = zt
    return s


SURFACE_METHODS = {"heightmap": resting_surface_heightmap, "slice": resting_surface_slice, "hull": resting_surface_hull}
SURFACE_METHOD, SURFACE_PARAMS = "heightmap", {"cell": 0.003, "flat_tol": 0.004, "min_area": 4e-4}


def base_band(V, F, band: float = 0.008, step: float = 0.002) -> dict:
    """The lowest ``band`` of the mesh (root z up): its points' xy box centre, the lowest z, and the radius about that
    centre (and about the root origin, ORG2's ``torso.base_radius``)."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    z0 = float(V[:, 2].min())
    Fb = F[V[F, 2].min(axis=1) < z0 + band]
    P = surface_samples(V, Fb, step, max_seg=60) if len(Fb) else V
    P = P[P[:, 2] < z0 + band]
    if len(P) < 3:
        return {"refused": "fewer than 3 points in the band"}
    c = (P[:, :2].min(0) + P[:, :2].max(0)) / 2
    return {"xy": c, "z": z0, "radius": float(np.max(np.linalg.norm(P[:, :2] - c, axis=1))),
            "radius_about_origin": float(np.max(np.linalg.norm(P[:, :2], axis=1))), "n_points": int(len(P))}


BASE_BAND = 0.008               # m: ORG2's stand-placement band (torso.base_radius); the radius at every band in
BASE_BANDS = (0.004, 0.008, 0.012)   # BASE_BANDS rides along (band-sensitive bases: docs/experiments/2026-11-08-f-catalogue.md)


def section_points(V, F, n, s: float) -> np.ndarray:
    """Where the triangles' edges cross the plane ``x . n = s``."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    d = V @ np.asarray(n, float) - s
    out = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = F[:, i], F[:, j]
        m = (d[a] * d[b]) < 0
        t = d[a[m]] / (d[a[m]] - d[b[m]])
        out.append(V[a[m]] + t[:, None] * (V[b[m]] - V[a[m]]))
    return np.vstack(out) if out else np.zeros((0, 3))


def ray_hits(V, F, o, d) -> np.ndarray:
    """Distances along ray ``o + t d`` (t > 0) at which it meets the mesh (Moller-Trumbore, vectorised)."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = b - a, c - a
    p = np.cross(np.broadcast_to(d, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-15
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o - a
    u = np.einsum("ij,ij->i", tv, p) * inv
    q = np.cross(tv, e1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    m = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9)
    return np.sort(t[m])


def _local(V, T: np.ndarray, into, long) -> tuple[np.ndarray, np.ndarray]:
    """Mesh vertices in the opening's local frame: origin at the frame point, z out of the opening, x along it."""
    z = -np.asarray(into, float)
    z /= np.linalg.norm(z)
    x = np.asarray(long, float) - z * float(np.asarray(long, float) @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    B = np.stack([x, y, z], axis=1)
    return (np.asarray(V, float) - T[:3, 3]) @ B, B


MOUTH_DEPTHS = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012)


def opening_fit(V, F, T, into, long, half_window=(0.03, 0.015), pad_x: float = 0.009, lip_depths=(0.001, 0.002,
                0.003, 0.004, 0.005, 0.006, 0.008, 0.01), lip_slab: float = 0.0, width_at: float = 0.5) -> dict:
    """The receptacle's opening at frame ``T`` from its own mesh, in the opening's local frame (z out = -``into``, x
    along ``long``, y across):

    * ``top``: the local top above the frame point (the highest of the vertices and of a 7 x 5 grid of vertical rays
      within ``half_window``); ``depth``: from it down to the floor (the first hit of the ray down the frame's axis);
    * ``width``: the gap across (y) between the inner walls of the horizontal section at ``width_at`` x the depth,
      within |x| <= ``pad_x``; ``width_profile``: the same gap at 10 / 25 / 50 / 75 / 90 % of the depth;
    * ``length``: the overlap along x of that section's inner-wall points on the two sides; ``open_ended`` when it
      reaches the window's edge;
    * ``lip``: the holder's outer half-width across (max |y| of the section within |x| <= ``pad_x``) at
      ``lip_depths`` below the top (``lip_slab`` > 0: the max over vertices within that slab instead, COIN2's
      ``coin_contact.wall_intercept`` reading)."""
    VL, B = _local(V, T, into, long)
    Fa = np.asarray(F, np.int64)
    wx, wy = half_window
    fx = VL[Fa][:, :, :2]
    inwin = (fx[:, :, 0].min(1) <= wx) & (fx[:, :, 0].max(1) >= -wx) & (fx[:, :, 1].min(1) <= wy) & (fx[:, :, 1].max(1) >= -wy)
    Fn = Fa[inwin]
    if not len(Fn):
        return {"refused": "no mesh within the window around the frame"}
    nb = (np.abs(VL[:, 0]) <= wx) & (np.abs(VL[:, 1]) <= wy)
    zs = [float(VL[nb, 2].max())] if nb.any() else []
    z_hi = float(VL[np.unique(Fn), 2].max()) + 0.01
    for gx in np.linspace(-wx, wx, 7):
        for gy in np.linspace(-wy, wy, 5):
            h = ray_hits(VL, Fn, np.array([gx, gy, z_hi]), np.array([0.0, 0.0, -1.0]))
            if len(h):
                zs.append(z_hi - float(h[0]))
    top = max(zs)
    hits = ray_hits(VL, Fn, np.array([0.0, 0.0, z_hi]), np.array([0.0, 0.0, -1.0]))
    if not len(hits):
        return {"refused": "the axis ray meets no mesh (a through opening, or the frame is off the mesh)"}
    floor = z_hi - float(hits[0])
    depth = top - floor
    if depth < 0.001:
        return {"refused": f"closed at the frame point: the floor is {depth * 1000:.2f} mm below the local top"}

    def section(z):
        P = section_points(VL, Fn, [0, 0, 1], z)
        return P[(np.abs(P[:, 0]) <= wx) & (np.abs(P[:, 1]) <= wy)]

    def gap(P, x_lo, x_hi):
        P = P[(P[:, 0] >= x_lo) & (P[:, 0] <= x_hi)]
        pos, neg = P[P[:, 1] > 0, 1], P[P[:, 1] < 0, 1]
        return float(pos.min() - neg.max()) if len(pos) and len(neg) else None

    prof = [[round(f * depth, 6), gap(section(top - f * depth), -pad_x, pad_x)] for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
    zw = width_at * depth
    Pm = section(top - zw)
    w = gap(Pm, -pad_x, pad_x)
    width_from = "mid-depth"
    if w is None:          # a mouth into a hollow body (a bank's slot, a rack's top plate): the deepest walled section
        for d in MOUTH_DEPTHS[::-1]:
            if d >= zw:
                continue
            Pm = section(top - d)
            w = gap(Pm, -pad_x, pad_x)
            if w is not None:
                zw, width_from = d, f"mouth ({d * 1000:.1f} mm below the top; no walls at mid-depth)"
                break
    if w is None:
        return {"refused": "no wall on one side across the opening at any depth", "top": top, "depth": depth}
    # the length: the overlap along x of the inner-wall points on the two sides (within 1 mm of the width's walls)
    Q = Pm[np.abs(Pm[:, 1]) <= w / 2 + 0.001]
    qp, qn = Q[Q[:, 1] > 0, 0], Q[Q[:, 1] < 0, 0]
    length, open_ended = None, False
    if len(qp) and len(qn):
        lo_x, hi_x = max(qp.min(), qn.min()), min(qp.max(), qn.max())
        if hi_x > lo_x:
            length = float(hi_x - lo_x)
            open_ended = bool(lo_x <= -wx + 0.001 or hi_x >= wx - 0.001)
    lip = []
    for d in lip_depths:
        z = top - d
        if lip_slab > 0:
            P = VL[(np.abs(VL[:, 2] - z) <= lip_slab) & (np.abs(VL[:, 0]) <= pad_x) & (np.abs(VL[:, 1]) <= wy)]
        else:
            P = section(z)
            P = P[np.abs(P[:, 0]) <= pad_x]
        if len(P) and (P[:, 1] > 0).any() and (P[:, 1] < 0).any():
            lip.append([float(d), float(max(P[:, 1].max(), -P[:, 1].min()))])
    return {"top": top, "floor": floor, "depth": depth, "width": w, "width_depth": zw, "width_from": width_from,
            "length": length, "open_ended": open_ended,
            "width_profile": prof, "lip": lip}


def corners_extent(frames: list, into, long) -> dict:
    """A slot's width / length from its corner frames (the footprint they outline), in the plane across ``into``."""
    P = np.array([np.asarray(f, float)[:3] for f in frames])
    z = -np.asarray(into, float)
    x = np.asarray(long, float) - z * float(np.asarray(long, float) @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return {"length": float(np.ptp(P @ x)), "width": float(np.ptp(P @ y)), "n": int(len(P))}


# ============================================================================================ one asset
OPENING_TYPES = ("slot", "hole", "socket", "keyhole", "support")


def _frames_of(raw_section: dict, section: str) -> dict:
    """{tag: [pose7 | xyz, ...]} of one raw section."""
    out = {}
    for tag, v in (raw_section or {}).items():
        if section == "support":
            out[tag] = [list(map(float, c)) for c in ((v or {}).get("center") or [])]
        else:
            out[tag] = [list(map(float, f)) for f in ((v or {}).get("frame") or [])]
    return out


def asset_entry(key: str, md: dict, mesh=None, usd: dict | None = None, mates: dict | None = None,
                files: dict | None = None, derive: bool = True) -> dict:
    """One ``assets[key]`` entry (spec section 1) from the raw metadata, the catalogue mesh (asset frame) and, for an
    Articulation, ``usd_articulation``'s reading."""
    g = (md.get("geometry") or {}).get("aligned_bbox") or {}
    verts = np.asarray(g.get("vertices") or [], float).reshape(-1, 3)
    if not len(verts) or not np.all(np.isfinite(verts)) or bool(np.all(np.ptp(verts, axis=0) <= 0)):
        raise ValueError("no aligned bbox in metadata.json")
    usd = usd or {}
    T_ra = pose7_T(usd["root_in_asset"]) if usd.get("root_in_asset") else None
    rm = root_map(T_ra)
    C = verts if rm["identity"] else np.array([rm_point(rm, v) for v in verts])
    bmin, bmax = C.min(0), C.max(0)
    act, pas = md.get("active") or {}, md.get("passive") or {}
    raw = {"functional": pas.get("functional") or {}, "active_functional": act.get("functional") or {},
           "support": pas.get("support") or {}}
    frames = {"functional": {}, "active_functional": {}, "support": {}}
    for section in ("functional", "active_functional"):
        for tag, v in raw[section].items():
            v = dict(v or {})
            e = {k: x for k, x in v.items() if k != "frame"}
            e["frame"] = [rm_pose(rm, f) for f in (v.get("frame") or [])]
            frames[section][tag] = e
    for tag, v in raw["support"].items():
        cs, rs = (v or {}).get("center") or [], (v or {}).get("radius") or []
        frames["support"][tag] = [[rm_pose(rm, c), float(r)] for c, r in zip(cs, rs)]
    joints = (usd.get("joints") or {})
    joints_by_tag, tag_types, tag_rules = {}, {}, {}
    for section in SECTIONS:
        for tag, v in raw[section].items():
            jn = (v or {}).get("parent_joint") if section != "support" else None
            J = joints.get(jn) if jn else None
            if J is not None and J["type"] in ("prismatic", "revolute") and tag not in joints_by_tag \
                    and J.get("lower") is not None and J.get("upper") is not None:
                joints_by_tag[tag] = {"name": jn, "type": J["type"], "axis": rm_vector(rm, J["axis"]),
                                      "lower": float(J["lower"]), "upper": float(J["upper"]),
                                      "origin": rm_point(rm, J["origin"])}
            t, rule = tag_type(tag, section, v, key, mates, J)
            if tag in tag_types and tag_types[tag] != t:        # one tag in two sections must agree
                t, rule = None, f"null: typed {tag_types[tag]} in one section and {t} in another"
            tag_types[tag], tag_rules[tag] = t, rule
    tag_conv, conv_why = {}, {}
    place = act.get("place") or {}
    up = upright(place)
    for section in ("functional", "support"):
        for tag, fl in _frames_of(raw[section], section).items():
            t = tag_types.get(tag)
            if t is None or not fl or len(fl[0]) < 7 or not up:
                continue
            m = measure_axis(quat_R(fl[0][3:7]), t)
            if m is None or m[2] > CONV_TOL_DEG:
                continue
            if (m[0], m[1]) != CONV[t][:2]:
                c = CONV[t][2] if CONV[t][2] != m[0] else next(i for i in range(3) if i != m[0])
                tag_conv[tag] = [m[0], m[1], c]
                conv_why[tag] = (f"axis: frame column {m[0]} x {m[1]:+d} is {m[2]:.1f} deg from the upright owner's "
                                 f"{'up' if SEMANTIC[t] > 0 else 'down'}")
    entry = {"root_to_bbox": [float(v) for v in (bmin + bmax) / 2], "bbox": {"min": [float(v) for v in bmin],
             "max": [float(v) for v in bmax]}, "frames": frames, "joints_by_tag": joints_by_tag,
             "tag_types": tag_types, "tag_rules": tag_rules, "tag_conv": tag_conv, "derived": {},
             "source": {"files": dict(sorted((files or {}).items()))}}
    if usd.get("root_in_asset"):
        entry["root_in_asset"] = list(usd["root_in_asset"])
    if not up:
        entry["derived_note"] = "no upright place orientation: no semantic axis measured, no @top / @base"
    if derive:
        entry["derived"], entry["derived_refused"] = derive_entry(entry, mesh, rm, up)
        for tag, d in entry["derived"].items():
            if tag.startswith("@"):                       # derived-only targets: typed, framed in ``derived``
                tag_types[tag], tag_rules[tag] = d["type"], f"derived from the mesh ({d['method']})"
            elif "long_col" in d and d["type"] in ("slot", "support"):      # round openings have no long direction
                a, sg, c = _conv_of(entry, tag)
                if d["long_col"] != c:
                    tag_conv[tag] = [a, sg, d["long_col"]]
                    conv_why[tag] = (conv_why.get(tag, "axis: the type's") + f"; across: frame column {d['long_col']} "
                                     "lies along the mesh opening")
    if conv_why:
        entry["tag_conv_why"] = conv_why
    return entry


def _conv_of(entry: dict, tag: str) -> tuple[int, int, int]:
    c = entry["tag_conv"].get(tag)
    return tuple(c) if c else CONV[entry["tag_types"][tag]]


WIDE_WINDOW = (0.08, 0.05)     # m: half-windows for an opening wider than the default (a tool-shaped recess)
LOCAL_SUPPORT_RING = 0.01      # m: the ring outside a support entry's radius whose rise is its rim height


def derive_entry(entry: dict, mesh, rm: dict, up: bool) -> tuple[dict, dict]:
    """Mesh-derived contact geometry of one asset, root frame: ``@top`` (the resting surface, support), ``@base``
    (the lowest band, base), and per opening-type tag (slot, hole, socket, keyhole, support) the opening fit; a
    support entry without an opening gets the local surface under its circle instead. Returns (derived, refused)."""
    der, refused = {}, {}
    if mesh is None:
        return der, {"*": "no mesh in the catalogue"}
    V, F = np.asarray(mesh[0], float), np.asarray(mesh[1], np.int64)
    if not rm["identity"]:
        V = (V - rm["T"][:3, 3]) @ rm["T"][:3, :3]               # asset -> root: R^T (v - t)
    hm = None
    if up:
        cell = SURFACE_PARAMS["cell"]
        hm = top_height_map(V, F, cell)
        s = resting_surface_heightmap(V, F, hmap=hm, **SURFACE_PARAMS)
        if "refused" in s:
            refused["@top"] = s["refused"]
        else:
            Rr = np.eye(3)
            Rr[:2, 0], Rr[:2, 1] = s["along"], [-s["along"][1], s["along"][0]]
            der["@top"] = {"type": "support", "frame": [float(s["xy"][0]), float(s["xy"][1]), s["z"]] + R_quat(Rr),
                           "extent": {"width": s["width"], "length": s["length"]},
                           "surface": {"z_min": s["z_min"], "z_max": s["z_max"], "flat_tol": SURFACE_PARAMS["flat_tol"],
                                       "rim_height": s["rim_height"]},
                           "below_top": s["top"] - s["z_max"], "area": s["area"],
                           "method": SURFACE_METHOD, "params": dict(SURFACE_PARAMS)}
        b = base_band(V, F, BASE_BAND)
        if "refused" in b:
            refused["@base"] = b["refused"]
        else:
            by_band = {f"{int(round(x * 1000))}mm": (lambda r: r.get("radius"))(base_band(V, F, x)) for x in BASE_BANDS}
            der["@base"] = {"type": "base", "frame": [float(b["xy"][0]), float(b["xy"][1]), b["z"], 1.0, 0.0, 0.0, 0.0],
                            "extent": {"radius": b["radius"]}, "radius_about_origin": b["radius_about_origin"],
                            "radius_by_band": by_band,
                            "method": "lowest_band", "params": {"band": BASE_BAND, "step": 0.002,
                                                                "centre": "xy box centre of the band"}}
    else:
        refused["@top"] = refused["@base"] = "no upright place orientation"
    for section in ("functional", "support"):
        fr = entry["frames"][section]
        for tag in fr:
            t = entry["tag_types"].get(tag)
            if t not in OPENING_TYPES:
                continue
            fl = [c[0] for c in fr[tag]] if section == "support" else fr[tag]["frame"]
            if not fl or len(fl[0]) < 7:
                continue
            a, sg, c = _conv_of(entry, tag)
            R = quat_R(fl[0][3:7])
            into = sg * R[:, a] * (-1.0 if t == "support" else 1.0)    # a support's axis is up; an opening goes down
            P = np.mean([np.asarray(f[:3], float) for f in fl], axis=0) if t == "slot" and len(fl) > 1 \
                else np.asarray(fl[0][:3], float)
            T = np.eye(4)
            T[:3, 3] = P
            r = float(fr[tag][0][1]) if section == "support" else 0.0
            win = (max(0.01, 2.0 * r), max(0.01, 2.0 * r)) if t in ("hole", "socket") and r > 0.001 else (0.03, 0.015)
            best = None                          # the long direction: the non-axis column with the longer, narrower run
            for w_ in ([win] if t in ("hole", "socket") and r > 0.001 else [win, WIDE_WINDOW]):
                for ci in (i for i in range(3) if i != a):
                    o = opening_fit(V, F, T, into, R[:, ci], half_window=w_)
                    if "refused" in o:
                        continue
                    score = (o["length"] or 0.0) - o["width"]
                    if best is None or score > best[0]:
                        best = (score, ci, o)
                if best is not None:
                    win = w_
                    break
            if best is None:
                why = opening_fit(V, F, T, into, R[:, c], half_window=win).get("refused", "no opening")
                if t == "support" and hm is not None:
                    loc = local_surface(hm, SURFACE_PARAMS["cell"], P, r)
                    if loc is not None:
                        der[tag] = {"type": t, "surface": loc, "method": "heightmap_local",
                                    "params": {"cell": SURFACE_PARAMS["cell"], "ring": LOCAL_SUPPORT_RING,
                                               "radius": "the support entry's radius (min one cell)"}}
                        continue
                refused[tag] = why
                continue
            _, ci, o = best
            d = {"type": t, "extent": {"width": o["width"], "depth": o["depth"]}, "width_profile": o["width_profile"],
                 "width_from": o["width_from"], "width_depth": o["width_depth"],
                 "open_ended": o["open_ended"], "long_col": ci, "top_above_frame": o["top"],
                 "floor_below_frame": -o["floor"], "method": "opening_fit",
                 "params": {"window_m": list(win), "pad_x_m": 0.009, "width_at": 0.5,
                            "width": "inner-wall gap of the horizontal section at mid-depth, else at the deepest "
                                     "walled mouth depth",
                            "depth": "local top to the first hit of the axis ray",
                            "lip": "outer half-width, exact horizontal section, |x| <= 9 mm (COIN2's wall_intercept "
                                   "reads a +-1 mm vertex slab: its max lies one slab deeper)"}}
            if o["length"]:
                d["extent"]["length"] = o["length"]
            if o["lip"]:
                d["lip"] = o["lip"]
            if t == "slot" and len(fl) > 1:
                d["corners"] = corners_extent(fl, into, R[:, ci])
            der[tag] = d
    return der, refused


def local_surface(hm, cell: float, p, r: float) -> dict | None:
    """The height map's surface under a support entry's circle (radius ``r``, at least one cell) and the rise of the
    ring just outside it."""
    Z, x0, y0 = hm
    H, W = Z.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = x0 + (xx + 0.5) * cell, y0 + (yy + 0.5) * cell
    dist = np.hypot(cx - p[0], cy - p[1])
    inside = (dist <= max(r, cell)) & np.isfinite(Z)
    if not inside.any():
        return None
    ring = (dist > max(r, cell)) & (dist <= max(r, cell) + LOCAL_SUPPORT_RING) & np.isfinite(Z)
    zmax = float(Z[inside].max())
    return {"z_min": float(Z[inside].min()), "z_max": zmax, "flat_tol": float(Z[inside].max() - Z[inside].min()),
            "rim_height": float(max(0.0, Z[ring].max() - zmax)) if ring.any() else 0.0}


# ============================================================================================ the build
def catalogue_hash(rd: dict) -> tuple[str, list]:
    """sha256 of the catalogue's ``index.json`` (the hash B1 reports as ``data.catalog.sha256``; ``catalog_md5``
    hashes the same bytes in md5) and its keys."""
    b = read_bytes(rd, rd["catalog"] / "index.json")
    return sha256_bytes(b), sorted(json.loads(b))


def _entry_job(args):
    key, md, mesh, usd, mates, files, derive = args
    try:
        e = asset_entry(key, md, mesh, usd if usd and "error" not in usd else None, mates, files, derive)
    except Exception as ex:   # noqa: BLE001 (reported per key, the build goes on)
        return key, None, f"{type(ex).__name__}: {ex}"[:200]
    if usd and "error" in usd:
        e["usd_error"] = usd["error"]
    if usd and usd.get("root_note"):
        e["root_note"] = usd["root_note"]
    return key, e, None


def build(assets_root, catalog_root, usd_reader=None, derive: bool = True, script_path: str | None = None,
          only: list[str] | None = None, log=None, workers: int = 1, allow_index_mismatch: bool = False) -> dict:
    """The whole ``f-catalogue/1`` document: every key under ``assets_root`` (``Assets/Object/RoboDojo``), meshes
    from the ONE catalogue root ``catalog_root``. Refuses (ValueError) when the catalogue's ``index.json`` keys differ
    from the keys it has a ``geom.npz`` for (``catalogue_hash`` would then not name the meshes read), unless
    ``allow_index_mismatch`` (the file then lists the difference). ``only``: keys to build (a test / debug aid; the
    file says so in ``sources.only``). ``workers`` > 1: the per-key work in that many processes."""
    t0 = time.time()
    rd = reader(assets_root, catalog_root)
    chash, index_keys = catalogue_hash(rd) if rd["catalog"] is not None else (None, [])
    keys = list(iter_keys(rd["assets"]))
    if only:
        keys = [(k, d) for k, d in keys if k in set(only)]
    mds, skipped = {}, {}
    for key, d in keys:
        p = d / "metadata.json"
        if not p.is_file():
            skipped[key] = "no metadata.json"
            continue
        mds[key] = read_json(rd, p)
    mesh_keys = sorted(k for k in mds if rd["catalog"] is not None and (rd["catalog"] / k / "geom.npz").is_file())
    mismatch = {}
    if rd["catalog"] is not None and not only:
        mismatch = {"in_index_without_mesh": sorted(set(index_keys) - set(mesh_keys)),
                    "mesh_without_index": sorted(set(mesh_keys) - set(index_keys))}
        if (mismatch["in_index_without_mesh"] or mismatch["mesh_without_index"]) and not allow_index_mismatch:
            raise ValueError(f"f-catalogue: index.json keys differ from the meshes: {json.dumps(mismatch)[:600]}")
    tags = {k: {"functional": list(((m.get("passive") or {}).get("functional") or {})),
                "support": list(((m.get("passive") or {}).get("support") or {}))} for k, m in mds.items()}
    mates = mates_index(tags)

    def jobs():
        for key, md in sorted(mds.items()):
            mesh, usd = None, None
            if key in set(mesh_keys):
                mesh = read_mesh(rd, rd["catalog"] / key / "geom.npz")
            d = rd["assets"] / key
            if key.startswith("Articulation/"):
                u = next((d / n for n in ("object.usd", "object.usdz") if (d / n).is_file()), None)
                if u is not None:
                    try:
                        usd = read_usd(rd, u, usd_reader)
                    except Exception as e:   # noqa: BLE001
                        usd = {"error": f"{type(e).__name__}: {e}"[:200]}
            files = {rel: h for rel, h in rd["opened"].items()
                     if rel.startswith(f"assets/{key}/") or rel.startswith(f"catalog/{key}/")}
            yield key, md, mesh, usd, mates, files, derive

    assets = {}
    if workers > 1:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        done_n = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            pending = set()
            it = jobs()

            def collect(fs):
                nonlocal done_n
                for f in fs:
                    key, e, err = f.result()
                    if e is None:
                        skipped[key] = err
                    else:
                        assets[key] = e
                    done_n += 1
                    if log and done_n % 50 == 0:
                        log(f"[fcat] {done_n}/{len(mds)} {key} ({time.time() - t0:.0f} s)")
            for job in it:                   # at most 2 x workers meshes in memory
                pending.add(ex.submit(_entry_job, job))
                if len(pending) >= 2 * workers:
                    fin, pending = wait(pending, return_when=FIRST_COMPLETED)
                    collect(fin)
            collect(pending)
    else:
        for i, job in enumerate(jobs()):
            key, e, err = _entry_job(job)
            if e is None:
                skipped[key] = err
            else:
                assets[key] = e
            if log and i % 50 == 0:
                log(f"[fcat] {i}/{len(mds)} {key} ({time.time() - t0:.0f} s)")
    assets = dict(sorted(assets.items()))
    lib = Path(__file__)
    script = {"path": script_path or str(lib), "sha256": sha256_file(script_path) if script_path else sha256_file(lib),
              "library": {"path": "dojo/skills/fcatalogue.py", "sha256": sha256_file(lib)}, "rules": RULES_VERSION}
    script["code_sha256"] = sha256_bytes(f"{script['sha256']}\0{script['library']['sha256']}".encode())
    doc = {"schema": SCHEMA,
           "build": {"catalogue_hash": chash,
                     "script": script,
                     "sources": {"roots": [str(rd["assets"])] + ([str(rd["catalog"])] if rd["catalog"] else []),
                                 "n_assets": len(keys), "n_entries": len(assets), "n_index_keys": len(index_keys),
                                 "n_meshes": len(mesh_keys), "listing_sha256": listing_sha256(rd["opened"]),
                                 "n_files": len(rd["opened"]), "skipped": skipped,
                                 **({"index_mismatch": mismatch} if any(mismatch.values()) else {}),
                                 **({"only": sorted(only)} if only else {})},
                     "conv": {t: list(v) for t, v in CONV.items()},
                     "units": dict(UNITS),
                     "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "wall_s": round(time.time() - t0, 1)},
           "assets": assets}
    return doc


# ============================================================================================ checks
def validate(doc: dict) -> list[str]:
    """What the adapter would refuse the file for (spec section 1, field rules), as a list of problems."""
    bad = []
    if doc.get("schema") != SCHEMA:
        return [f"schema {doc.get('schema')!r}"]
    b = doc.get("build") or {}
    for k in ("catalogue_hash", "script", "sources", "conv", "units", "created_utc"):
        if not b.get(k):
            bad.append(f"build.{k} missing")
    if b.get("units") != UNITS:
        bad.append(f"units {b.get('units')}")
    why = conv_valid({t: tuple(v) for t, v in (b.get("conv") or {}).items()})
    if why:
        bad.append(why)
    for key, e in (doc.get("assets") or {}).items():
        if not re.fullmatch(r"[A-Za-z]+/[A-Za-z0-9_]+/\d{5}", key):
            bad.append(f"{key}: not a canonical key")
        for k in ("root_to_bbox", "bbox", "frames", "joints_by_tag", "tag_types", "source"):
            if k not in e:
                bad.append(f"{key}: {k} missing")
        tags = set()
        for s in SECTIONS:
            tags |= set((e.get("frames") or {}).get(s) or {})
        missing = tags - set(e.get("tag_types") or {})
        if missing:
            bad.append(f"{key}: untyped tags {sorted(missing)}")
        for tag, t in (e.get("tag_types") or {}).items():
            if t is not None and t not in TYPES:
                bad.append(f"{key}: {tag} type {t}")
        for tag, d in (e.get("derived") or {}).items():
            if "method" not in d or "params" not in d:
                bad.append(f"{key}: derived {tag} without method / params")
            if tag.startswith("@") and (d.get("type") not in TYPES or len(d.get("frame") or []) != 7):
                bad.append(f"{key}: derived-only {tag} needs a type and a pose7 frame")
    return bad


__all__ = ["SCHEMA", "TYPES", "CONV", "LEXICON", "CODEX_WORDS", "reader", "read_bytes", "read_json", "read_mesh",
           "read_usd", "build", "validate", "asset_entry",
           "tag_type", "tag_words", "mates_index", "measure_axis", "check_root", "forbidden_part",
           "resting_surface_heightmap", "resting_surface_slice", "resting_surface_hull", "SURFACE_METHODS",
           "base_band", "opening_fit", "corners_extent", "usd_articulation", "root_map", "rm_pose", "rm_back", "quat_R", "pose7_T",
           "listing_sha256", "conv_valid"]
