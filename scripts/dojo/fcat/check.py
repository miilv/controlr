"""Checks of an ``f-catalogue/1`` file against the raw metadata: the catalogue-wide round trip and the verb readers.

* ``round_trip``: every frame tag of the raw ``metadata.json`` -> the file's root-frame frames -> targets (the partner's
  prototype ``from_asset``, vendored verbatim as ``scripts/dojo/fcat/functarget_ref.py``) -> ``as_frames`` -> back to the asset
  frame, compared as 4x4 transforms (and radii, joints, raw extra keys) with the raw values. ``mode="typed"`` is the
  adapter's path (null tags skipped); ``mode="all"`` types every null frame tag structurally so that every frame of
  every asset goes through the adapter. ``p1=True`` applies the two changes the prototype needs: a lever / hinge keeps
  its frame's orientation (``as_frames`` writes the identity today), and the tags keep the file's order
  (``from_asset`` sorts them, so a verb taking the first matching frame would try another one first).
* ``reader_check``: today's frame readers (``library.functional_frames``, ``precision.frame_points_w``,
  ``precision.hole_frames``, ``library.slot_frame``, ``library._support_raw``, ``library.slot_targets`` and
  ``insert_into``'s frame composition) fed the raw asset and the round-tripped one, at a seeded random owner pose,
  must give identical answers.
* ``conv_check``: per typed frame, whether the target's axis under the pinned ``CONV`` / ``tag_conv`` points along
  the type's semantic direction on an upright owner, and whether a slot target's ``frame()`` is the frame
  ``coin_contact.slot_targets_from_pose`` reads (x along, y across, z out of the opening).

CPU only; reads nothing but the file and the raw metadata it is given. It lives under scripts/ with the vendored
prototype: both are check harnesses, not inference code (nothing in the runtime imports them).
"""

from __future__ import annotations

import contextlib
import copy
import math
from types import SimpleNamespace

import numpy as np

from dojo.skills import fcatalogue as fc
from scripts.dojo.fcat import functarget_ref as FT

TOL = 1e-9


# ============================================================================================ adapter calls
@contextlib.contextmanager
def conv_for(rows: dict):
    """The prototype's module-level ``CONV`` temporarily set to ``rows`` (type -> (a, s, c))."""
    saved = dict(FT.CONV)
    FT.CONV.update({t: (int(r[0]), float(r[1]), int(r[2])) for t, r in rows.items()})
    try:
        yield
    finally:
        FT.CONV.clear()
        FT.CONV.update(saved)


def structural_type(tag: str, section: str, entry: dict) -> str:
    """A type for a null tag in ``mode="all"``: one that keeps every frame and its orientation (handle / support)."""
    return "support" if section == "support" else "handle"


def targets_of(key: str, entry: dict, mode: str = "typed", p1: bool = False, chash: str = "h") -> list:
    """The adapter's targets for one asset (per tag, so a tag's ``tag_conv`` row applies to it alone)."""
    conv = {t: tuple(v) for t, v in fc.CONV.items()}
    out = []
    frames = entry["frames"]
    for section in fc.SECTIONS:
        # the prototype sorts a section's tags (``from_asset``); with ``p1`` the file's (= the raw metadata's) order is
        # kept, so verbs that take the first matching frame see the catalogue's order
        for tag in (list(frames.get(section) or {}) if p1 else sorted(frames.get(section) or {})):
            t = entry["tag_types"].get(tag)
            if t is None and mode == "all":
                t = structural_type(tag, section, entry)
            if t is None:
                continue
            one = {s: ({tag: frames[s][tag]} if s == section else {}) for s in fc.SECTIONS}
            rows = dict(conv)
            if tag in (entry.get("tag_conv") or {}):
                rows[t] = tuple(entry["tag_conv"][tag])
            with conv_for(rows):
                tg = FT.from_asset(one, entry["root_to_bbox"], owner=key, step=0, key=key, catalogue_hash=chash,
                                   type_of=lambda _tag, _sec, _t=t: _t, joints_by_tag=entry.get("joints_by_tag"))
                for x in tg:
                    x.provenance[0]["conv"] = rows[t]
                    if p1 and x.type in FT.NO_ORIENTATION:
                        c = x.provenance[0]
                        raw = frames[section][tag]["frame"][c["index"]]
                        c["frame_quat"] = list(raw[3:7]) if len(raw) >= 7 else [1.0, 0.0, 0.0, 0.0]
            out += tg
    return out


def frames_of(targets: list, p1: bool = False) -> dict:
    """``as_frames`` with each target's own conv row (and, with ``p1``, a lever / hinge's frame orientation back)."""
    out = {"functional": {}, "active_functional": {}, "support": {}, "joints_by_tag": {}}
    for t in targets:
        with conv_for({t.type: t.provenance[0]["conv"]}):
            f = FT.as_frames([t])
        c = t.provenance[0]
        if p1 and t.type in FT.NO_ORIENTATION and c.get("frame_quat") is not None:
            for s in ("functional", "active_functional"):
                for tag, e in f[s].items():
                    e["frame"] = [list(p[:3]) + list(c["frame_quat"]) for p in e["frame"]]
        for s in ("functional", "active_functional"):
            for tag, e in f[s].items():
                d = out[s].setdefault(tag, {"frame": {}, **{k: v for k, v in e.items() if k != "frame"}})
                for i, p in enumerate(e["frame"]):
                    d["frame"][(c["index"], i)] = p
        for tag, lst in f["support"].items():
            out["support"].setdefault(tag, {})[c["index"]] = lst[0]
        out["joints_by_tag"].update(f["joints_by_tag"])
    for s in ("functional", "active_functional"):
        for tag, d in out[s].items():
            d["frame"] = [d["frame"][k] for k in sorted(d["frame"])]
    out["support"] = {tag: [d[k] for k in sorted(d)] for tag, d in out["support"].items()}
    return out


def _T(c) -> np.ndarray:
    return fc.pose7_T(c)


def _err(a, b) -> float:
    return float(np.max(np.abs(_T(a) - _T(b))))


# ============================================================================================ round trip
def round_trip(key: str, entry: dict, md: dict, mode: str = "typed", p1: bool = False) -> dict:
    """Raw metadata -> file -> targets -> ``as_frames`` -> asset frame, per tag: the max |T_raw - T_back| over its
    frames; also the file's own frames against the raw (the root-frame map) and the raw extra keys / radii / joints.
    Returns {"tags": {section/tag: err}, "fail": [...], "skipped": [null tags], "n_frames": int}."""
    rm = fc.root_map(fc.pose7_T(entry["root_in_asset"]) if entry.get("root_in_asset") else None)
    act, pas = md.get("active") or {}, md.get("passive") or {}
    raw = {"functional": pas.get("functional") or {}, "active_functional": act.get("functional") or {},
           "support": pas.get("support") or {}}
    back = frames_of(targets_of(key, entry, mode, p1), p1)
    res = {"tags": {}, "fail": [], "skipped": [], "n_frames": 0, "file_err": 0.0}
    for section in fc.SECTIONS:
        for tag, v in raw[section].items():
            name = f"{section}/{tag}"
            if section == "support":
                cs, rs = (v or {}).get("center") or [], (v or {}).get("radius") or []
                raw_fr = [list(map(float, c)) for c in cs]
                file_fr = [c for c, _ in entry["frames"]["support"][tag]]
                got = back["support"].get(tag)
                got_fr = [fc.rm_back(rm, c) for c, _ in got] if got else None
                extra_ok = got is None or [float(r) for _, r in got] == [float(r) for r in rs[:len(cs)]]
            else:
                raw_fr = [list(map(float, f)) for f in ((v or {}).get("frame") or [])]
                file_fr = entry["frames"][section][tag]["frame"]
                got = back[section].get(tag)
                got_fr = [fc.rm_back(rm, c) for c in got["frame"]] if got else None
                extra_ok = got is None or all(got.get(k) == x for k, x in (v or {}).items() if k != "frame")
            if not raw_fr:
                continue
            res["n_frames"] += len(raw_fr)
            fe = max(_err(fc.rm_back(rm, f), r) for f, r in zip(file_fr, raw_fr)) if len(file_fr) == len(raw_fr) else math.inf
            res["file_err"] = max(res["file_err"], fe)
            if got_fr is None:
                res["skipped"].append(name)
                continue
            if len(got_fr) != len(raw_fr):
                res["tags"][name] = math.inf
                res["fail"].append(f"{name}: {len(got_fr)} frames back for {len(raw_fr)}")
                continue
            e = max(_err(g, r) for g, r in zip(got_fr, raw_fr))
            res["tags"][name] = e
            if e > TOL or fe > TOL or not extra_ok:
                res["fail"].append(f"{name}: err {e:.3g}, file {fe:.3g}" + ("" if extra_ok else ", extra keys / radii differ"))
    for tag, j in (entry.get("joints_by_tag") or {}).items():
        g = back["joints_by_tag"].get(tag)
        if g is None:
            if entry["tag_types"].get(tag) in FT.NO_ORIENTATION:
                res["fail"].append(f"joint {tag}: not written back")
            continue
        if not (np.allclose(g["axis"], j["axis"], atol=TOL, rtol=0) and abs(g["lower"] - j["lower"]) <= TOL
                and abs(g["upper"] - j["upper"]) <= TOL and g["type"] == j["type"]):
            res["fail"].append(f"joint {tag}: {g} != {j}")
    return res


# ============================================================================================ verb readers
def _state(key: str, asset, seed: int):
    from dojo.skills.world import ObjectState
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    t, cat, idx = key.split("/")
    return ObjectState(label=key.replace("/", "_"), obj_type=t, category=cat, idx=int(idx),
                       pos=rng.uniform(-0.3, 0.3, 3), quat=q, asset=asset)


def assets_pair(key: str, entry: dict, md: dict, mode: str = "typed", p1: bool = True):
    """(raw AssetInfo restricted to the tags the adapter keeps, the round-tripped AssetInfo), both asset frame."""
    from dojo.skills.world import asset_from_metadata
    rm = fc.root_map(fc.pose7_T(entry["root_in_asset"]) if entry.get("root_in_asset") else None)
    base = asset_from_metadata(key, md)
    back = frames_of(targets_of(key, entry, mode, p1), p1)
    for s in ("functional", "active_functional"):
        for tag, e in back[s].items():
            e["frame"] = [fc.rm_back(rm, c) for c in e["frame"]]
    back["support"] = {tag: [(fc.rm_back(rm, c), r) for c, r in lst] for tag, lst in back["support"].items()}
    rt = FT.to_asset_info(base, back)
    keep = {s: set(back[s]) for s in ("functional", "active_functional", "support")}
    ref = copy.copy(base)
    ref.functional = {k: v for k, v in base.functional.items() if k in keep["functional"]}
    ref.active_functional = {k: v for k, v in base.active_functional.items() if k in keep["active_functional"]}
    ref.support = {k: v for k, v in base.support.items() if k in keep["support"]}
    return ref, rt


def _same(a, b) -> float:
    """Max abs difference of two reader outputs (nested lists / tuples / arrays / dicts / scalars); inf on a shape
    or key mismatch."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            return math.inf
        return max([_same(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, str) or isinstance(b, str):
        return 0.0 if a == b else math.inf
    if a is None or b is None:
        return 0.0 if a is b else math.inf
    if isinstance(a, (list, tuple)) and (not a or not isinstance(a[0], (int, float, np.floating))):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return math.inf
        return max([_same(x, y) for x, y in zip(a, b)] or [0.0])
    A, B = np.asarray(a, float), np.asarray(b, float)
    if A.shape != B.shape:
        return math.inf
    return float(np.max(np.abs(A - B))) if A.size else 0.0


def reader_check(key: str, entry: dict, md: dict, seed: int = 0, mode: str = "typed", p1: bool = True) -> dict:
    """Today's readers on the raw asset vs the round-tripped one at a seeded random owner pose. {reader: max diff}."""
    from dojo.skills import library as L
    from dojo.skills import precision as PR
    ref, rt = assets_pair(key, entry, md, mode, p1)
    o_ref, o_rt = _state(key, ref, seed), _state(key, rt, seed)
    out = {}

    def put(name, a, b):
        out[name] = max(out.get(name, 0.0), _same(a, b))
    for active in (False, True):
        src = ref.active_functional if active else ref.functional
        for tag in src:
            put("functional_frames", L.functional_frames(o_ref, tag, active), L.functional_frames(o_rt, tag, active))
            put("frame_points_w", PR.frame_points_w(o_ref, tag, active), PR.frame_points_w(o_rt, tag, active))
            if not active and len(src[tag].get("frame") or []) >= 3:
                put("slot_frame", L.slot_frame(o_ref, tag), L.slot_frame(o_rt, tag))
    put("hole_frames", sorted(PR.hole_frames(o_ref), key=lambda x: x[0]), sorted(PR.hole_frames(o_rt), key=lambda x: x[0]))
    for tag in ref.support:          # raw pose7s, quaternions as authored: compared as transforms
        put("_support_raw", [fc.pose7_T(c) for c in L._support_raw(o_ref, tag)],
            [fc.pose7_T(c) for c in L._support_raw(o_rt, tag)])
    return out


def pair_check(occupant: tuple, container: tuple, seed: int = 0) -> dict:
    """Readers that combine two assets: ``library.slot_targets`` (the occupant's active frames on the container's
    ``<category>_slot`` frames) and ``insert_into``'s composition (the occupant's active ``insert`` frames on the
    container's support and ``slot|opening|insert`` frames), raw vs round-tripped. Arguments: (key, entry, md)."""
    import re

    from dojo.skills import library as L
    from dojo.skills.geometry import inv_T
    (ko, eo, mo), (kc, ec, mc) = occupant, container
    ro, to = assets_pair(ko, eo, mo)
    rc, tc = assets_pair(kc, ec, mc)
    out = {}
    for name, (A, C) in (("ref", (ro, rc)), ("rt", (to, tc))):
        o, c = _state(ko, A, seed), _state(kc, C, seed + 1)
        sk = SimpleNamespace(world=SimpleNamespace(obj=lambda lab, _o=o, _c=c: _o if lab == "o" else _c))
        st = L.slot_targets(sk, "o", "c")
        ins = [F for tag in o.asset.active_functional if "insert" in tag.lower()
               for F in L.functional_frames(o, tag, active=True)]
        frames = [fc.pose7_T(f[0]) for tag in c.asset.support for f in [[x] for x in L._support_raw(c, tag)]]
        frames += [F for tag in c.asset.functional if re.search(r"slot|opening|insert", tag.lower())
                   for F in L.functional_frames(c, tag)]
        comp = [c.T @ Ts @ inv_T(Ti) for Ts in frames for Ti in ins]
        out[name] = {"slot_targets": st, "insert_compose": comp}
    return {k: _same(out["ref"][k], out["rt"][k]) for k in out["ref"]} | \
        {f"n_{k}": len(out["ref"][k]) for k in out["ref"]}


# ============================================================================================ CONV
def conv_check(key: str, entry: dict, md: dict) -> list[dict]:
    """Per typed passive frame of an upright owner: the angle between the target's axis (pinned conv / tag_conv)
    and the type's semantic direction, the angle under the prototype's own CONV row, and, for a slot, whether its
    ``frame()`` is the frame ``slot_targets_from_pose`` reads (z out of the opening, x along it)."""
    rows = []
    if not fc.upright((md.get("active") or {}).get("place")):
        return rows
    tg = targets_of(key, entry, "typed")
    for t in tg:
        c = t.provenance[0]
        if c["section"] == "active_functional" or fc.SEMANTIC.get(t.type) is None:
            continue
        d = np.array([0.0, 0, float(fc.SEMANTIC[t.type])])
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(t.axis @ d)))))
        raw = (entry["frames"]["support"][c["tag"]][max(c["index"], 0)][0] if c["section"] == "support"
               else entry["frames"][c["section"]][c["tag"]]["frame"][max(c["index"], 0)])
        R = fc.quat_R(raw[3:7]) if len(raw) >= 7 else np.eye(3)
        a, s, _ = FT_PROTO_CONV[t.type]
        ang_proto = math.degrees(math.acos(max(-1.0, min(1.0, float(s * R[:, a] @ d)))))
        row = {"key": key, "tag": t.name, "type": t.type, "axis_deg": ang, "proto_axis_deg": ang_proto,
               "override": c["tag"] in (entry.get("tag_conv") or {})}
        if t.type == "slot":
            # the frame a perception-side reader gets from (point, axis, across) under the TYPE row, and the raw
            # catalogue frame (what ``frame()`` rebuilds under a tag_conv override, for the exact round trip)
            for name, row_ in (("reader", fc.CONV[t.type]), ("raw", c["conv"])):
                with conv_for({t.type: row_}):
                    T = t.frame()
                row[f"{name}_z_out_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, float(-t.axis @ T[:3, 2])))))
                row[f"{name}_x_along_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(t.across @ T[:3, 0]))))))
        rows.append(row)
    return rows


FT_PROTO_CONV = dict(FT.CONV)        # the prototype's own table, as vendored (for the comparison only)


def slot_reader_check(thin_key: str, thin_md: dict, axis_into, across, point) -> float:
    """``coin_contact.slot_targets_from_pose`` on a slot target's ``frame()`` (pinned slot conv) against the pose
    its docstring defines (x along the slot, y across it, z out of the opening): max abs difference of the poses."""
    from dojo.skills.coin_contact import slot_targets_from_pose
    from dojo.skills.world import asset_from_metadata
    o = _state(thin_key, asset_from_metadata(thin_key, thin_md), 0)
    t = FT.FuncTarget("s", "c", "slot", np.asarray(point, float), np.asarray(axis_into, float),
                      [{"source": "perception", "step": 0, "query": "q", "build_hash": "b"}],
                      across=np.asarray(across, float))
    with conv_for({"slot": fc.CONV["slot"]}):
        T = t.frame()
    z = -t.axis
    x = t.across
    R = np.stack([x, np.cross(z, x), z], axis=1)
    want = list(map(float, point)) + fc.R_quat(R)
    got = fc.T_pose7(T)
    a = slot_targets_from_pose(o, got)
    b = slot_targets_from_pose(o, want)
    return _same(a, b)


__all__ = ["round_trip", "reader_check", "pair_check", "conv_check", "slot_reader_check", "targets_of", "frames_of"]
