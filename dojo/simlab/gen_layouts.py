"""Dev-layout generator: NEW RoboDojo layout records from the public task configs.

RoboDojo's own sampler is not public: eval layouts are pre-generated JSON. What is public is
(a) every task config (``task/RoboDojo/config/<task>.yml``: per-group category/index pools,
xlim/ylim/zlim, rotate_deg, place_tag, select_mode, labels, Clutter blocks, ProhibitedArea) and
(b) RoboDojo's placement engine ``utils/cluttered_generator.ClutteredGenerator`` (footprint
polygons from the asset's oriented bbox, rest pose from ``metadata.json`` ``active.place``,
rejection sampling against already-placed footprints). This module drives (b) with (a) and
writes records in the eval schema (``dojo/simlab/layout.py``).

Seeds: our namespace starts at ``paths.SEED_BASE`` (100000). Layout ``id`` == its seed; the same
id always gives the same record (numpy + random seeded per layout).

Run in RoboDojo's conda python (needs shapely, rtree, transforms3d from that env; no Isaac):

    $RD_PYTHON -m dojo.simlab.gen_layouts --task stack_blocks --n 25 [--start 100000] [--out DIR]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dojo.simlab import paths
from dojo.simlab.layout import OBJECT_TYPES, fixtures_from_scene, validate

if str(paths.RD_ROOT) not in sys.path:
    sys.path.insert(0, str(paths.RD_ROOT))

GENERATOR = "simlab/v15"     # v15: solve_equation task semantics (a legal equation, one slot missing; other tasks as v14)
#                              v14: a child on an object without a place_tag takes a random place frame (v13: the first)
# Home-pose gripper footprint of both ARX arms (link6 .. fingertips, y up to ~-0.20, from GT) + 3 cm. At home the
# fingers hang ~15 cm above the table, so only objects taller than ROBOT_KEEPOUT_TOP collide with them;
# RoboDojo's placement engine does not know the robot (v1 put bottles on the grippers: 9/34 layouts then
# failed RoboDojo's stability check).
ROBOT_KEEPOUT = [(-0.42, -0.40, -0.18, -0.17), (0.18, -0.40, 0.42, -0.17)]     # (x0, y0, x1, y1), env frame
ROBOT_KEEPOUT_TOP = 0.08

SAMPLING_DEFAULTS = {"group": None, "xlim": None, "ylim": None, "zlim": None, "qpos": [1, 0, 0, 0],
                     "rotate_deg": 0, "rotate_rand": False, "relative_plane": "Table", "place_tag": None,
                     "margin": 0.01, "check_mode": "bbox", "need_check_stable": True}


class LayoutError(RuntimeError):
    pass


_KEEPOUT_ON = [True]          # generate() turns it off for its last attempts (fixed poses under a gripper)


def hits_robot(meta: dict, pose, polygon, plane: str) -> bool:
    """A placed object is taller than ROBOT_KEEPOUT_TOP and its footprint overlaps a home gripper."""
    if not _KEEPOUT_ON[0] or plane != "table" or polygon is None:
        return False
    from shapely.geometry import box
    if not any(polygon.intersects(box(*b)) for b in ROBOT_KEEPOUT):
        return False
    from dojo.simlab.motion import quat_to_mat
    v = np.asarray(meta["geometry"]["oriented_bbox"]["vertices"], float).reshape(-1, 3)
    top = float((v @ quat_to_mat(pose[3:] / np.linalg.norm(pose[3:])).T)[:, 2].max() + pose[2])
    from dojo.simlab.layout import TABLE_TOP_Z
    return top - TABLE_TOP_Z > ROBOT_KEEPOUT_TOP


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def task_config(task: str) -> dict:
    return load_yaml(paths.TASK_CONFIGS / f"{task}.yml")


def scene_config(task: str) -> dict:
    idx = load_yaml(paths.TASK_CONFIGS / "_task.yml")
    name = (idx.get("tasks", {}).get(task) or {}).get("scene_config") or idx.get("common", {}).get("scene_config",
                                                                                                    "default")
    return load_yaml(paths.ENV_CFG / "scene" / f"{name}.yml")


def merged_object_groups(task: str) -> dict[str, list]:
    """Task object groups per type in the task config's own key order (organize_table places its
    Geometry drawer before the Rigid objects), with the scene's groups appended (camera_stand), as
    ``LayoutManager.process_config`` merges them."""
    tcfg, scfg = task_config(task), scene_config(task)
    order = [k for k in tcfg if k in OBJECT_TYPES] + [t for t in OBJECT_TYPES if t not in tcfg]
    out: dict[str, list] = {}
    for t in order:
        groups = list(tcfg.get(t) or [])
        groups += list(scfg.get(t) or [])
        if groups:
            out[t] = groups
    return out


def required_labels(task: str) -> set[str]:
    """Labels every layout of ``task`` carries, i.e. what RoboDojo's reward may look up. Per object group:
    the first (minimum count) entries of ``select_mode.label``. Hierarchical groups give min(category nums)
    x min(instance nums); groups labelled only through ``label_prefix`` / linked counts are skipped. This is
    a lower bound, used to refuse incomplete worlds (``Twin.load_from_estimate``). Task-config groups only:
    the scene's own fixtures (camera_stand) are not looked up by rewards, and estimates may leave them out."""
    def lo(n):
        return int(n[0]) if isinstance(n, (list, tuple)) else int(n)
    out: set[str] = set()
    tcfg = task_config(task)
    for groups in (tcfg.get(t) or [] for t in OBJECT_TYPES):
        for g in groups:
            sm = g.get("select_mode") or {}
            labels = list(sm.get("label") or [])
            if not labels:
                continue
            if sm.get("mode") == "hierarchical":
                n = lo(sm.get("select_category_nums", [1])) * lo(sm.get("select_instance_nums", [1]))
            elif sm.get("mode") == "linked_count":
                continue
            else:
                n = lo(sm.get("nums", 1))
            out |= set(labels[:n])
    return out


def available_indices(obj_type: str, category: str, clutter: bool = False) -> list[int]:
    d = paths.OBJECTS / ("Clutter" if clutter else obj_type) / category
    if not d.is_dir():
        return []
    return sorted(int(p.name) for p in d.iterdir() if p.name.isdigit())


def metadata(obj_type: str, category: str, idx: int, clutter: bool = False) -> dict | None:
    p = paths.OBJECTS / ("Clutter" if clutter else obj_type) / category / f"{idx:05d}" / "metadata.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def usd_exists(obj_type: str, category: str, idx: int, clutter: bool = False) -> bool:
    d = paths.OBJECTS / ("Clutter" if clutter else obj_type) / category / f"{idx:05d}"
    return (d / "object.usdz").exists() or (d / "object.usd").exists()


# --------------------------------------------------------------------------- regions
def _ranges(lim) -> list[tuple[float, float]]:
    """xlim as [a, b] or [[a, b], [c, d], ...] -> list of (lo, hi)."""
    if lim is None:
        return []
    if isinstance(lim[0], (list, tuple)):
        return [(float(min(r)), float(max(r))) for r in lim]
    return [(float(min(lim)), float(max(lim)))]


def region(xlim, ylim):
    from shapely.geometry import LineString, MultiPolygon, Point, box
    from shapely.ops import unary_union
    parts = []
    for x0, x1 in _ranges(xlim):
        for y0, y1 in _ranges(ylim):
            if x0 == x1 and y0 == y1:
                parts.append(Point(x0, y0))
            elif x0 == x1 or y0 == y1:
                parts.append(LineString([(x0, y0), (x1, y1)]))
            else:
                parts.append(box(x0, y0, x1, y1))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    if all(p.geom_type == "Polygon" for p in parts):
        return MultiPolygon(parts)
    return unary_union(parts)


def make_generators(scene_cfg: dict, env_spacing: float = 7.0) -> dict:
    """One ClutteredGenerator per support plane, set up as ``LayoutManager.cluttered_generator_init``."""
    from shapely.geometry import box
    from utils.cluttered_generator import ClutteredGenerator
    g_table = None
    if "Table" in scene_cfg:                 # the conveyor scene has no table
        t = scene_cfg["Table"]
        pos, sc = t["default_pos"], t["scale"]
        size = [-sc[0] / 2, -sc[1] / 2, sc[0] / 2, sc[1] / 2]
        table_h = pos[2] + sc[2] / 2
        g_table = ClutteredGenerator()
        g_table.reset(box(size[0] + 0.05 + pos[0], size[1] + 0.05 + pos[1], size[2] - 0.05 + pos[0],
                          size[3] - 0.05 + pos[1]), frame=np.array([0.0, 0.0, table_h, 1.0, 0.0, 0.0, 0.0]))
    gr = scene_cfg["Ground"]
    gpos = gr.get("default_pos", [0, 0, 0])
    gsize = [-env_spacing / 2, -env_spacing / 2, env_spacing / 2, env_spacing / 2]
    g_ground = ClutteredGenerator()
    g_ground.reset(box(gsize[0] + 0.05 + gpos[0], gsize[1] + 0.05 + gpos[1], gsize[2] - 0.05 + gpos[0],
                       gsize[3] - 0.05 + gpos[1]),
                   frame=np.array([0.0, 0.0, gr.get("thickness", 0.1) / 2, 1.0, 0.0, 0.0, 0.0]))
    return {"table": g_table if g_table is not None else g_ground, "ground": g_ground}


# --------------------------------------------------------------------------- selection
def _count(n, rng: random.Random) -> int:
    if isinstance(n, (list, tuple)):
        return int(n[0]) if len(n) == 1 else rng.randint(int(n[0]), int(n[1]))
    return int(n)


UNSUPPORTED_SELECT: tuple = ()          # every select_mode feature of the 54 configs is implemented (v8)


def _layout_assets(layout: dict, obj_type: str) -> set:
    """(category, index) of the ``obj_type`` objects placed so far. Per type: a Fluid ``wuliangye/0`` is the
    liquid inside the Rigid ``wuliangye/0`` bottle (pour_liquid_into_cup), not a second copy of it."""
    from dojo.simlab.layout import iter_objects
    return {(cat, int(inst["category_idx"])) for t, cat, inst in iter_objects(layout or {}) if t == obj_type}


def select_instances(obj_type: str, group: dict, rng: random.Random, layout: dict | None = None,
                     state: dict | None = None) -> list[tuple[dict, int]]:
    """(category entry, asset index) per instance, following ``select_mode`` (identity rules need the
    layout placed so far):

    * ``unique``: distinct instances, also distinct from every (category, index) of the same type already in
      the layout
      (make_kong: the discards must not match a hand triple);
    * ``same`` (+ ``allow_global_same: False``: an index no other such group used);
    * ``same_index_as_label`` / ``same_as_label`` + ``same_label``: the labelled object's index (and category);
    * ``hierarchical``, ``allow_duplicate``; ``label_map`` is handled by the caller.
    Modes / keys not implemented raise instead of being silently ignored."""
    sm = group.get("select_mode") or {}
    dl = group.get("dynamic_layout")
    if dl and dl.get("type") != "x_symmetric_line":
        raise NotImplementedError(f"dynamic_layout {dl.get('type')!r} not implemented")
    if sm.get("mode") == "linked_count":
        raise LayoutError("linked_count groups are placed by place_task_objects")
    state = state if state is not None else {}
    if sm.get("mode") in ("same_index_as_label", "same_as_label"):
        from dojo.simlab.layout import find
        ref = find(layout, sm["same_label"])
        names = [e["name"] for e in group["category"]]
        if sm["mode"] == "same_as_label":
            entry = next((e for e in group["category"] if e["name"] == ref["category"]), {"name": ref["category"]})
        else:
            entry = group["category"][0] if ref["category"] not in names else \
                next(e for e in group["category"] if e["name"] == ref["category"])
        return [(entry, int(ref["category_idx"]))] * _count(sm.get("nums", 1), rng)
    entries = []
    for e in group["category"]:
        idx = list(e.get("index") if e.get("index") is not None else available_indices(obj_type, e["name"]))
        entries.append((e, [int(i) for i in idx]))
    mode = sm.get("mode", "allow_duplicate")
    if mode == "hierarchical":
        n_cat = _count(sm.get("select_category_nums", [1, 1]), rng)
        n_inst = _count(sm.get("select_instance_nums", [1, 1]), rng)
        groups: dict[Any, list] = {}
        for j, (e, idx) in enumerate(entries):          # no "group" key: each category entry is a group
            groups.setdefault(e.get("group", f"_entry{j}"), []).append((e, idx))
        prefix = sm.get("label_prefix")
        per_group = isinstance(prefix, list) and len(prefix) > 1
        out, k = [], 0
        for gi, g in enumerate(rng.sample(sorted(groups, key=str), n_cat)):
            pool = [(e, i) for e, idx in groups[g] for i in idx]
            picks = rng.sample(pool, n_inst) if sm.get("instance_sample_mode", "unique") == "unique" \
                else [rng.choice(pool) for _ in range(n_inst)]
            for ii, (e, i) in enumerate(picks):
                if prefix is not None:                      # labels the task code looks up by prefix
                    lab = f"{prefix[gi]}_{ii}" if per_group else \
                        f"{prefix[0] if isinstance(prefix, list) else prefix}_{k}"
                    e = dict(e, _label=lab)
                out.append((e, i))
                k += 1
        return out
    n = _count(sm.get("nums", 1), rng)
    pool = [(e, i) for e, idx in entries for i in idx]
    if mode == "same":
        if sm.get("allow_global_same", True) is False:
            used = state.setdefault("same_used", {}).setdefault(obj_type, set()) | _layout_assets(layout, obj_type)
            pool = [(e, i) for e, i in pool if (e["name"], i) not in used]          # also tiles placed earlier
            if not pool:
                raise LayoutError(f"allow_global_same: no free index for {sm.get('label')}")
            pick = rng.choice(pool)
            state["same_used"][obj_type].add((pick[0]["name"], pick[1]))
            return [pick] * n
        return [rng.choice(pool)] * n
    if mode == "unique":
        out, seen = [], set(_layout_assets(layout, obj_type))
        for e, i in rng.sample(pool, len(pool)):
            if (e["name"], i) in seen:
                continue
            seen.add((e["name"], i))
            out.append((e, i))
            if len(out) == n:
                return out
        raise LayoutError(f"not enough unique instances for {[e['name'] for e, _ in entries]}")
    return [rng.choice(pool) for _ in range(n)]


# --------------------------------------------------------------------------- placement
FIXTURE_PLANES = ("table", "ground")


def on_object(plane) -> bool:
    """``relative_plane`` names other objects (labels, e.g. swap_blocks' ["mat0", "mat1", "mat2"])."""
    if isinstance(plane, (list, tuple)):
        return True
    return str(plane).split("/")[0].lower() not in FIXTURE_PLANES


def _qmul(a, b):
    from dojo.simlab.layout import quat_mul
    return quat_mul(a, b)


def _qrot(q, v):
    from dojo.simlab.motion import quat_to_mat
    return quat_to_mat(np.asarray(q, float) / np.linalg.norm(q)) @ np.asarray(v, float)


def place_on_object(prm: dict, meta: dict, layout: dict, used: set, rng: random.Random, gens: dict,
                    siblings: dict):
    """Pose of an object placed relative to another object (``relative_plane`` = label, list of labels,
    or "label/support_key"). Returns (pose[7], parent_label).

    * list of labels (swap_blocks' mats): pick a parent not used yet;
    * the parent's ``passive.support`` point (first centre of the given / first key) is the reference;
      without one, the parent's origin;
    * x/y offsets from xlim/ylim (and z from ``zlim``) are in the support frame (v12; world axes around the
      reference for parents without supports); with ``zlim`` the orientation is the parent's x ``qpos``
      (e.g. the conveyor's photo, a fluid inside its bottle), otherwise the child rests on the support point
      (its ``active.place`` frame);
    * a ``zlim`` child of a parent without supports (a fluid in its bottle) is placed from the parent's
      local bbox top centre;
    * ``rotate_rand`` turns about world z; children of one support (``relative_plane`` string) keep their
      footprints apart unless ``check_mode: enforce`` (RoboDojo's semantics); make_toast's four bread slots
      are four supports.
    """
    from dojo.simlab.layout import find, quat_z
    multi = isinstance(prm["relative_plane"], (list, tuple))
    cands = list(prm["relative_plane"]) if multi else [prm["relative_plane"]]
    free = [c for c in cands if not (multi and str(c).split("/")[0] in used)]
    if not free:
        raise LayoutError(f"no free support among {cands}")
    pick = str(rng.choice(free))
    plabel, _, skey = pick.partition("/")
    parent = find(layout, plabel)
    ptype = next(t for t in OBJECT_TYPES for c, insts in (layout.get(t) or {}).items() if parent in insts)
    pmeta = metadata(ptype, parent["category"], int(parent["category_idx"])) or {}
    supports = (pmeta.get("passive") or {}).get("support") or {}
    ppos, pq = np.asarray(parent["default_pos"], float), np.asarray(parent["default_ori"], float)
    max_off = None
    fixture = bool(supports) and not skey and prm["zlim"] is not None and \
        bool((pmeta.get("geometry") or {}).get("oriented_bbox"))
    if fixture:
        # a fixture placed by the bare label of a parent with supports, at a fixed z (pick_from_conveyor_by_image's
        # photo board and basket on the conveyor): world xy from the parent's root, z from the parent's world bbox top,
        # orientation qpos in the world (v13; the public PhysicalRSI sampler's place_fixture,
        # "world_axes_parent_origin_xy_world_bbox_top_z"). v12 offset them from the first belt support, which put
        # the basket's 22 cm body inside the conveyor (ejected at load).
        v = np.asarray(pmeta["geometry"]["oriented_bbox"]["vertices"], float)
        ztop = max(float(_qrot(pq, x)[2]) for x in v)
        ref_pos, ref_q = np.array([ppos[0], ppos[1], ppos[2] + ztop]), np.array([1.0, 0.0, 0.0, 0.0])
        supports = {}
    elif supports:
        sup = supports[skey if skey in supports else next(iter(supports))]
        centre = np.asarray(sup["center"][0], float)
        ref_pos, ref_q = ppos + _qrot(pq, centre[:3]), _qmul(pq, centre[3:])
        if len(sup["center"]) == 1 and sup.get("radius"):
            max_off = float(sup["radius"][0])     # one support disk (egg_basket): stay on it
    elif prm["zlim"] is not None and (pmeta.get("geometry") or {}).get("oriented_bbox"):
        # no support (a fluid in its bottle): zlim is from the parent's local bbox top centre, as in the
        # public PhysicalRSI sampler ("parent_local_bbox_top_center"); from the origin it lands under the table
        v = np.asarray(pmeta["geometry"]["oriented_bbox"]["vertices"], float)
        top = np.array([(v[:, 0].min() + v[:, 0].max()) / 2, (v[:, 1].min() + v[:, 1].max()) / 2, v[:, 2].max()])
        ref_pos, ref_q = ppos + _qrot(pq, top), pq
    else:
        ref_pos, ref_q = ppos, pq
    places = (meta.get("active") or {}).get("place") or {}
    req = prm["place_tag"]
    if isinstance(req, str) and req:
        tags = [req]
    elif isinstance(req, (list, tuple)) and req:
        tags = list(req)
    else:
        # no tag in the config: any of the asset's frames with a projection circle, drawn per attempt, as RoboDojo's
        # ClutteredGenerator.add_model_from_config and DataGen's conveyor children (PhysicalRSI layout_sampling
        # _make_conveyor_child_instance) do: a bottle rides the belt standing or lying (v14; v13 took the first frame).
        tags = [t for t, v in places.items() if ((v or {}).get("projection_circle") or {}).get("center") is not None]
        tags = tags or list(places)[:1]
    ext = ((meta.get("geometry") or {}).get("oriented_bbox") or {}).get("extents")

    def radius_of(tag):
        r_ = float(places[tag]["projection_circle"]["radius"]) if tag else 0.03
        return max(r_, 0.5 * float(max(ext))) if ext else r_    # an egg on its side is longer than its circle
    xrs, yrs = _ranges(prm["xlim"] or [0, 0]), _ranges(prm["ylim"] or [0, 0])   # several ranges = several slots
    # the support radius bounds the placement point (footprints are kept apart separately below)
    for _ in range(400):
        # a single frame draws nothing: every child with one frame keeps the random stream it had
        tag = (tags[0] if tags else None) if len(tags) <= 1 else rng.choice(tags)
        radius = radius_of(tag)
        xr, yr = rng.choice(xrs), rng.choice(yrs)
        if xr[0] == xr[1] and yr[0] == yr[1]:   # a pinned offset (make_kong: a tile stacked on a tile)
            off = np.array([xr[0], yr[0], 0.0])
        elif max_off is not None and (xr[0] == xr[1] or yr[0] == yr[1]):   # one axis pinned (laptop on its stand)
            off = np.array([rng.uniform(*xr), rng.uniform(*yr), 0.0])
            pin = abs(xr[0]) if xr[0] == xr[1] else abs(yr[0])
            if np.hypot(off[0], off[1]) > max_off and pin <= max_off:
                continue                        # off the disk (a pin outside it is the config's call)
        elif max_off is not None:               # uniform on the support disk (inside the config box)
            r, th = max_off * math.sqrt(rng.random()), rng.uniform(0.0, 2 * math.pi)
            off = np.array([r * math.cos(th), r * math.sin(th), 0.0])
            if not (xr[0] - 1e-9 <= off[0] <= xr[1] + 1e-9 and yr[0] - 1e-9 <= off[1] <= yr[1] + 1e-9):
                continue
        else:
            off = np.array([rng.uniform(*xr), rng.uniform(*yr), 0.0])
        q = ref_q
        if prm["rotate_rand"]:
            rd = prm["rotate_deg"] or 0
            lo, hi = (-abs(rd), abs(rd)) if np.isscalar(rd) else (min(rd), max(rd))
            q = _qmul(quat_z(rng.uniform(lo, hi)), q)
        zoff = rng.uniform(*_ranges(prm["zlim"])[0]) if prm["zlim"] is not None else 0.0
        # offsets live in the support frame (v12; RoboDojo / PhysicalRSI convention): make_toast's slot frames
        # stand the slices up (Ry 90), so its xlim -0.02 lifts a slice in its slot. Parents without supports:
        # world axes around the reference.
        off_w = _qrot(ref_q, off + [0.0, 0.0, zoff]) if supports else off + [0.0, 0.0, zoff]
        if prm["zlim"] is not None:
            qq = _qmul(q, prm["qpos"]) if prm["qpos"] is not None else q
            pose = np.r_[ref_pos + off_w, qq]
        else:
            c_frame = np.asarray(places[tag]["projection_circle"]["center"], float)
            pose = gens["ground"]._placement_to_origin(np.r_[ref_pos + off_w, q], c_frame)
        sib = siblings.setdefault(pick, [])     # per support, like RoboDojo's per-plane generators
        foot = (ref_pos + off_w)[:2] if prm["zlim"] is None else pose[:2]   # footprint centre = placement point
        if prm["check_mode"] == "enforce" or all(np.linalg.norm(foot - xy) > radius + r + 0.002 for xy, r in sib):
            sib.append((np.asarray(foot, float).copy(), radius))
            used.add(plabel)
            return pose, plabel
    raise LayoutError(f"no room for a child of {plabel}")


def _is_fixed(group: dict) -> bool:
    c = group.get("common") or {}
    xr, yr = _ranges(c.get("xlim")), _ranges(c.get("ylim"))
    return len(xr) == 1 and len(yr) == 1 and xr[0][0] == xr[0][1] and yr[0][0] == yr[0][1]


def _deps(group: dict) -> set:
    rp = (group.get("common") or {}).get("relative_plane", "Table")
    deps = set()
    if on_object(rp):
        deps |= {str(c).split("/")[0] for c in (rp if isinstance(rp, (list, tuple)) else [rp])}
    sm = group.get("select_mode") or {}
    if sm.get("mode") == "label_map":
        deps.add(sm["key_label"])
    if sm.get("mode") in ("same_index_as_label", "same_as_label"):
        deps.add(sm["same_label"])
    return deps


def linked_instances(obj_type: str, group: dict, rng: random.Random, linked_n: dict) -> list[tuple[dict, int]]:
    """``select_mode: linked_count``: one instance per object of the linked group, labelled
    ``<label_prefix>_<i>``, at ``dynamic_layout`` positions (``x_symmetric_line``: x = center + (i - (n-1)/2) *
    spacing, one y drawn from ylim for the whole row; i = 0 is the leftmost, world -x)."""
    sm, dl = group["select_mode"], group.get("dynamic_layout") or {}
    n = linked_n[id(group)]
    prefix = sm.get("label_prefix")
    prefix = prefix[0] if isinstance(prefix, list) else (prefix or "linked")
    entry = group["category"][0]
    idx_pool = entry.get("index") if entry.get("index") is not None else available_indices(obj_type, entry["name"])
    yr = _ranges((group.get("common") or {}).get("ylim") or [0, 0])[0]
    y = rng.uniform(*yr)
    c, sp = float(dl.get("center", 0.0)), float(dl.get("spacing", 0.1))
    out = []
    for i in range(n):
        e = dict(entry, _label=f"{prefix}_{i}", _fixed_xy=(c + (i - (n - 1) / 2) * sp, y))
        out.append((e, int(idx_pool[0])))
    return out


def label_map_instances(obj_type: str, group: dict, layout: dict) -> list[tuple[dict, int]]:
    """``select_mode: label_map``: the instance listed for the key object in ``<category>/map.json``
    (e.g. the photo that shows pick_from_conveyor_by_image's target)."""
    from dojo.simlab.layout import find
    sm = group["select_mode"]
    key_obj = find(layout, sm["key_label"])
    entry = group["category"][0]
    with open(paths.OBJECTS / obj_type / entry["name"] / "map.json", encoding="utf-8") as f:
        mp = json.load(f)
    idx = int(mp[f"{key_obj['category']}/{int(key_obj['category_idx']):05d}"])
    return [(entry, idx)] * _count(sm.get("nums", 1), random.Random(0))


def place_task_objects(groups: dict[str, list], gens: dict, rng: random.Random, layout: dict) -> None:
    # groups go in config order, each once everything it refers to (parent objects, label_map keys)
    # has been placed
    pending = [(t, g) for t, gl in groups.items() for g in gl]
    ordered, placed = [], set()
    while pending:
        ready = [(t, g) for t, g in pending if _deps(g) <= placed]
        if not ready:
            raise LayoutError(f"unresolvable placement dependencies: {[sorted(_deps(g)) for _, g in pending]}")
        # fixed-pose groups first (zero-area region): a random object placed earlier must not take the
        # spot the config pins (organize_table: drawer vs the pinned cube_cushion)
        fixed = [(t, g) for t, g in ready if _is_fixed(g)]
        t, g = (fixed or ready)[0]
        pending.remove((t, g))
        ordered.append((t, g))
        placed |= set((g.get("select_mode") or {}).get("label") or [])
    used_supports: set = set()
    siblings: dict = {}
    sel_state: dict = {}
    # linked_count (arrange_largest_number's mats, one per digit): select the linked group's instances first,
    # then place the mats before the digits, so the digits avoid them
    presel, linked_n = {}, {}
    for t, g in list(ordered):
        sm = g.get("select_mode") or {}
        if sm.get("mode") != "linked_count":
            continue
        src = next(((t2, g2) for t2, g2 in ordered if (g2.get("select_mode") or {}).get("label_prefix") in
                    (sm["linked_label_prefix"], [sm["linked_label_prefix"]])), None)
        if src is None:
            raise LayoutError(f"linked_count: no group with label_prefix {sm['linked_label_prefix']!r}")
        presel[id(src[1])] = select_instances(src[0], src[1], rng, layout, sel_state)
        linked_n[id(g)] = len(presel[id(src[1])])
        ordered.remove((t, g))
        ordered.insert(0, (t, g))
    for obj_type, group in ordered:
        common = dict(group.get("common") or {})
        labels = list((group.get("select_mode") or {}).get("label") or [])
        sm_mode = (group.get("select_mode") or {}).get("mode")
        if id(group) in presel:
            insts = presel.pop(id(group))
        elif sm_mode == "linked_count":
            insts = linked_instances(obj_type, group, rng, linked_n)
        elif sm_mode == "label_map":
            insts = label_map_instances(obj_type, group, layout)
        else:
            insts = select_instances(obj_type, group, rng, layout, sel_state)
        for k, (entry, idx) in enumerate(insts):
            prm = dict(SAMPLING_DEFAULTS)
            prm.update(common)
            prm.update({key: v for key, v in entry.items() if key not in ("name", "index", "_label", "_fixed_xy")})
            cat = entry["name"]
            meta = metadata(obj_type, cat, idx)
            if meta is None or not usd_exists(obj_type, cat, idx):
                raise LayoutError(f"missing asset {obj_type}/{cat}/{idx:05d}")
            label = entry.get("_label") or (labels[k] if k < len(labels) else f"{cat}{k}")
            if "_fixed_xy" in entry:                       # linked_count / dynamic_layout position
                prm["xlim"], prm["ylim"] = [entry["_fixed_xy"][0]] * 2, [entry["_fixed_xy"][1]] * 2
            if on_object(prm["relative_plane"]):
                pose, parent_label = place_on_object(prm, meta, layout, used_supports, rng, gens, siblings)
                phys = dict(meta.get("physics") or {})
                phys["type"] = obj_type.lower()
                inst = {"category": cat, "category_idx": int(idx)}
                for f in ("group", "xlim", "ylim", "zlim", "qpos", "rotate_deg", "rotate_rand", "relative_plane",
                          "place_tag", "margin", "check_mode", "need_check_stable"):
                    inst[f] = copy.deepcopy(prm[f])
                inst["relative_plane"] = parent_label     # RoboDojo rewards read it back as a label
                inst.update({"label": label, "default_pos": [float(v) for v in pose[:3]],
                             "default_ori": [float(v) for v in pose[3:] / np.linalg.norm(pose[3:])],
                             "scale": [1.0, 1.0, 1.0], "physics": phys, "visual": {}})
                layout.setdefault(obj_type, {}).setdefault(cat, []).append(inst)
                continue
            plane = str(prm["relative_plane"]).split("/")[0].lower()
            gen = gens["ground" if plane == "ground" else "table"]
            ok, pose, poly = gen.add_model_from_config(
                config=meta, place_tag=prm["place_tag"], rotate_rand=bool(prm["rotate_rand"]),
                rotate_deg=prm["rotate_deg"], margin=float(prm["margin"] or 0.0), max_attempts=200,
                name=label, allowed_region=region(prm["xlim"], prm["ylim"]),
                zlim=prm["zlim"], qpos=prm["qpos"] if prm["zlim"] is not None else None,
                cluttered=False, check_mode=prm["check_mode"])
            if not ok:
                raise LayoutError(f"could not place {label} ({cat}/{idx})")
            if hits_robot(meta, pose, poly, plane):
                raise LayoutError(f"{label} ({cat}/{idx}) under a home-pose gripper")
            phys = dict(meta.get("physics") or {})          # e.g. SpringButton's reset_drive
            phys["type"] = obj_type.lower()
            inst = {"category": cat, "category_idx": int(idx)}
            for f in ("group", "xlim", "ylim", "zlim", "qpos", "rotate_deg", "rotate_rand", "relative_plane",
                      "place_tag", "margin", "check_mode", "need_check_stable"):
                inst[f] = copy.deepcopy(prm[f])
            inst.update({"label": label, "default_pos": [float(v) for v in pose[:3]],
                         "default_ori": [float(v) for v in pose[3:] / np.linalg.norm(pose[3:])],
                         "scale": [1.0, 1.0, 1.0], "physics": phys, "visual": {}})
            layout.setdefault(obj_type, {}).setdefault(cat, []).append(inst)


def place_clutter(clutter_cfg: list, gens: dict, rng: random.Random, layout: dict, exclude: set[str]) -> int:
    placed = 0
    used: set[str] = set()
    for block in clutter_cfg:
        yml = load_yaml(paths.OBJECTS / block.get("yaml_path", "Clutter/clutter.yml"))["Clutter"]
        cats = sorted(c for c in yml if c not in exclude)
        mode = block.get("mode", "allow_duplicate")
        plane = str(block.get("relative_plane", "Table")).lower()
        gen = gens["ground" if plane == "ground" else "table"]
        want, tries = int(block.get("nums", 0)), 0
        got = 0
        while got < want and tries < 20 * max(want, 1):
            tries += 1
            pool = [c for c in cats if not (mode in ("category_unique", "unique") and c in used)]
            if not pool:
                break
            cat = rng.choice(pool)
            idx = int(rng.choice(list(yml[cat])))
            meta = metadata("Rigid", cat, idx, clutter=True)
            if meta is None or not usd_exists("Rigid", cat, idx, clutter=True):
                continue
            ok, pose, poly = gen.add_model_from_config(
                config=meta, rotate_rand=bool(block.get("rotate_rand", True)), rotate_deg=block.get("rotate_deg", 30),
                margin=float(block.get("margin", 0.015)), max_attempts=50, name=f"clutter_{cat}",
                allowed_region=region(block["xlim"], block["ylim"]), cluttered=True, check_mode="bbox")
            if not ok or hits_robot(meta, pose, poly, plane):
                continue
            used.add(cat)
            phys = {k: v for k, v in (meta.get("physics") or {}).items() if k in ("mass", "friction")}
            phys["type"] = "rigid"
            layout.setdefault("Rigid", {}).setdefault(cat, []).append({
                "category_idx": idx, "physics": phys, "scale": [1.0, 1.0, 1.0], "type": "cluttered",
                "clutter_idx": placed, "yaml_path": block.get("yaml_path", "Clutter/clutter.yml"),
                "relative_plane": block.get("relative_plane", "Table"),
                "default_pos": [float(v) for v in pose[:3]],
                "default_ori": [float(v) for v in pose[3:] / np.linalg.norm(pose[3:])]})
            placed += 1
            got += 1
    return placed


VARIANT_BASE = 100   # variant draws start at sub-seed seed * 1000 + VARIANT_BASE (attempts of the plain draw: 0-39)


def variant_subseed(seed: int, variant: int, attempt: int, max_tries: int = 40) -> int:
    """The RNG seed of one generation attempt. Variant 0 is the plain draw (``seed * 1000 + attempt``, the stored
    records). Variant v >= 1 is a NEW random draw of the same id (another layout from the same config): sub-seeds
    ``seed * 1000 + VARIANT_BASE + max_tries * (v - 1) + attempt``, which stay inside the id's own 1000-block, so they
    never coincide with another id's draws or with the plain draw."""
    if variant < 0 or attempt < 0 or attempt >= max_tries:
        raise ValueError(f"variant {variant} / attempt {attempt}")
    off = 0 if variant == 0 else VARIANT_BASE + max_tries * (variant - 1)
    if off + attempt >= 1000:
        raise ValueError(f"variant {variant} leaves the id's sub-seed block (max {(1000 - VARIANT_BASE) // max_tries})")
    return seed * 1000 + off + attempt


# solve_equation task semantics: copied from feat/dojo-simlab 53022ed ("simlab: generator v13: solve_equation task
# semantics"), never merged to main; main's records for solve_equation had all four tiles and no missing slot, on
# which RoboDojo's run_reward crashes (find_missing_list returns a list)
EQ_OPS = ("plus", "minus", "multiplication", "division")


def equation_groups(groups: dict[str, list], rng: random.Random) -> tuple[dict[str, list], dict]:
    """solve_equation's task semantics, which the config alone does not carry. RoboDojo's reward needs a legal
    ``a op b = answer`` on the mats, with exactly one of t0/t1/t2/t4(/t5) missing (the policy fills it). Without
    that, ``find_missing_list`` returns a list and ``run_reward`` crashes. Same convention as the public
    PhysicalRSI sampler:
    - a uniform legal single-digit equation;
    - mat5 + t5 for a two-digit answer;
    - a uniform missing slot;
    - digit tiles drawn among the ``number`` assets with index % 10 == digit."""
    eqs = []
    for a in range(10):
        for b in range(10):
            vals = {"plus": a + b, "minus": a - b, "multiplication": a * b}
            if b and a % b == 0:
                vals["division"] = a // b
            eqs += [(a, op, b, v) for op, v in vals.items() if v >= 0]
    a, op, b, ans = rng.choice(eqs)
    values: dict = {0: a, 1: op, 2: b, 4: ans}
    out = {t: [copy.deepcopy(g) for g in gl] for t, gl in groups.items()}
    by_label = {lab: (t, g) for t, gl in out.items() for g in gl for lab in ((g.get("select_mode") or {}).get("label") or [])}
    if ans >= 10:
        values[4], values[5] = divmod(ans, 10)
        tm, m4 = by_label["mat4"]
        m3 = by_label["mat3"][1]
        step = m4["common"]["xlim"][0] - m3["common"]["xlim"][0]
        m5 = copy.deepcopy(m4)
        m5["select_mode"]["label"] = ["mat5"]
        m5["common"]["xlim"] = [v + step for v in m4["common"]["xlim"]]
        out[tm].append(m5)
        tt, t4 = by_label["t4"]
        t5 = copy.deepcopy(t4)
        t5["select_mode"]["label"] = ["t5"]
        t5["common"]["relative_plane"] = "mat5"
        out[tt].append(t5)
    missing = rng.choice(sorted(values))
    for t, gl in out.items():
        keep = []
        for g in gl:
            labs = (g.get("select_mode") or {}).get("label") or []
            lab = labs[0] if len(labs) == 1 else None
            if lab is not None and lab[0] == "t" and lab[1:].isdigit():
                k = int(lab[1:])
                if k == missing:
                    continue
                v = values[k]
                if isinstance(v, int):
                    idx = [i for i in available_indices(t, "number") if i % 10 == v]
                    g["category"] = [{"name": "number", "index": [rng.choice(idx)]}]
                else:
                    g["category"] = [{"name": v}]
            keep.append(g)
        out[t] = keep
    return out, {"operands": [a, b], "operator": op, "result": ans, "missing_slot": missing}


def extra_groups(groups: dict, assets: list[str], label_prefix: str = "seg_extra") -> dict:
    """``groups`` + one group per asset key ("Rigid/<category>/<index>") placed like a task object (collision-checked,
    random yaw) in the region of the task's first table-level Rigid group. Segmenter TRAINING draws only
    (``generate(..., extra=...)``): rare assets shown more often than the config draws them."""
    region = ([-0.4, 0.4], [-0.25, 0.05])
    for g in groups.get("Rigid") or []:
        c = g.get("common") or {}
        if c.get("relative_plane", "Table") == "Table" and c.get("xlim") and c["xlim"][0] < c["xlim"][1] \
                and c.get("ylim") and c["ylim"][0] < c["ylim"][1]:
            region = (list(c["xlim"]), list(c["ylim"]))
            break
    out = {t: list(gl) for t, gl in groups.items()}
    for i, key in enumerate(assets):
        typ, cat, idx = key.split("/")
        out.setdefault(typ, []).append({
            "common": {"xlim": region[0], "ylim": region[1], "rotate_rand": True, "rotate_deg": 180,
                       "relative_plane": "Table", "margin": 0.01},
            "category": [{"name": cat, "index": [int(idx)]}],
            "select_mode": {"nums": 1, "mode": "allow_duplicate", "label": [f"{label_prefix}{i}"]}})
    return out


def generate(task: str, seed: int, max_tries: int = 40, variant: int = 0, extra: list[str] | None = None) -> dict:
    """One layout record for ``task`` (seed >= SEED_BASE). Deterministic in ``seed`` (and ``variant``: v >= 1 is a new
    random draw for the same id, ``variant_subseed``; the twin-trained segmenter's training layouts). ``extra``
    (default none): asset keys added as extra objects (``extra_groups``); only for variant draws (v >= 1), never for
    a stored / measured record (v0)."""
    if seed < paths.SEED_BASE:
        raise ValueError(f"our layout seeds start at {paths.SEED_BASE}")
    if extra and variant < 1:
        raise ValueError("extra objects only in training draws (variant >= 1), never in a measured record (v0)")
    tcfg, scfg = task_config(task), scene_config(task)
    groups = merged_object_groups(task)
    if extra:
        groups = extra_groups(groups, list(extra))
    last = None
    for attempt in range(max_tries):
        # the keep-out is soft: some configs fix objects under the home-pose grippers on purpose
        # (fill_egg_holder's egg_basket, organize_table's alarm); then the second half drops it
        _KEEPOUT_ON[0] = attempt < max_tries // 2
        sub = variant_subseed(seed, variant, attempt, max_tries)
        try:
            layout, n_clutter, want, task_meta = _draw(tcfg, scfg, groups, sub, semantics=True)
        except LayoutError as e:
            last = e
            continue
        errs = validate(layout)
        if errs:
            raise LayoutError(f"generated an invalid record: {errs[:5]}")
        layout["_simlab"] = {"task": task, "id": seed, "seed": seed, "attempt": attempt, "generator": GENERATOR,
                             "n_clutter": n_clutter, "clutter_wanted": want,
                             "robot_keepout": _KEEPOUT_ON[0]}
        if task_meta:
            layout["_simlab"]["task_semantics"] = task_meta
        if variant:
            layout["_simlab"]["variant"] = variant
        if extra:
            layout["_simlab"]["seg_extra"] = list(extra)
        return layout
    raise LayoutError(f"{task} seed {seed}: no valid layout in {max_tries} tries ({last})")


def _draw(tcfg: dict, scfg: dict, groups: dict, sub: int, semantics: bool = False) -> tuple:
    """One placement attempt of the config with the random streams seeded by ``sub``: (record, clutter placed,
    clutter wanted), + the task semantics drawn (``equation_groups``; None for other tasks) when ``semantics``;
    LayoutError when it does not fit."""
    task_cats = {e["name"] for gl in groups.values() for g in gl for e in g["category"]}
    np.random.seed(sub % (2**32))
    rng = random.Random(sub)
    gens = make_generators(scfg)
    for a in (tcfg.get("ProhibitedArea") or []) + (tcfg.get("Prohibited_Area") or []):   # one config has the typo
        x0, y0, x1, y1 = a
        gens["table"].add_box_prohibited_area(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    layout: dict = {}
    task_meta = None
    if semantics and _is_equation(groups):
        groups, task_meta = equation_groups(groups, rng)
    place_task_objects(groups, gens, rng, layout)
    n_clutter = place_clutter(tcfg.get("Clutter") or [], gens, rng, layout, exclude=task_cats)
    # RoboDojo's own records hold fewer clutter objects than `nums` when the strips are tight
    # (push_T_random eval record: ~13 of 20), so a partial fill is kept, not retried.
    want = sum(int(b.get("nums", 0)) for b in tcfg.get("Clutter") or [])
    layout.update(fixtures_from_scene(scfg))
    return (layout, n_clutter, want, task_meta) if semantics else (layout, n_clutter, want)


def _is_equation(groups: dict) -> bool:
    """solve_equation's config: the mats mat0-mat4 and the answer tile t4 (keyed by the config's labels, as
    ``equation_groups`` itself)."""
    labs = {lab for gl in groups.values() for g in gl for lab in ((g.get("select_mode") or {}).get("label") or [])}
    return {"mat0", "mat1", "mat2", "mat3", "mat4", "t0", "t1", "t2", "t4"} <= labs


# template draws: sub-seeds 1, 2, ... Every layout record's sub-seed is seed * 1000 + ... with seed >= SEED_BASE, so
# >= 100_000_000: a template draw is never a layout of ours (nor any evaluated one)
TEMPLATE_SUBSEED_MAX = 10_000


def template_draws(task: str, n: int = 4) -> list[dict]:
    """``n`` draws of the task's public config that are no layout of ours or anyone's (sub-seeds 1, 2, ...): what all
    of them agree on is what the config fixes. For the twin's template of public constants
    (``dojo.runtime.estimate.public_constants``); never a record to run or to evaluate."""
    tcfg, scfg = task_config(task), scene_config(task)
    groups = merged_object_groups(task)
    out, sub = [], 0
    while len(out) < n and sub < TEMPLATE_SUBSEED_MAX:
        sub += 1
        lay = None
        for keepout in (True, False):           # as generate: some configs fix objects under the home grippers
            _KEEPOUT_ON[0] = keepout
            try:
                lay = _draw(tcfg, scfg, groups, sub)[0]
                break
            except LayoutError:
                continue
        _KEEPOUT_ON[0] = True
        if lay is None:
            continue
        errs = validate(lay)
        if errs:
            raise LayoutError(f"generated an invalid record: {errs[:5]}")
        out.append(strip_meta(lay))
    if len(out) < n:
        raise LayoutError(f"{task}: {len(out)} of {n} template draws fit")
    return out


def strip_meta(layout: dict) -> dict:
    """The record RoboDojo loads (our ``_simlab`` metadata key removed)."""
    return {k: v for k, v in layout.items() if not k.startswith("_")}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--start", type=int, default=paths.SEED_BASE)
    ap.add_argument("--out", default=None, help="default /root/dojo-data/simlab/layouts/<task>")
    ap.add_argument("--reserved-slice", action="store_true",
                    help="allow ids >= paths.RESERVED_MIN_ID (D-C10: only the agreed joint-run generation)")
    a = ap.parse_args(argv)
    if paths.is_reserved_id(a.start + a.n - 1) and not a.reserved_slice:
        raise SystemExit(f"ids {a.start}-{a.start + a.n - 1} reach the reserved slice (>= {paths.RESERVED_MIN_ID}, D-C10): "
                         "only the agreed joint-run generation may create them (--reserved-slice)")
    out = Path(a.out) if a.out else paths.LAYOUTS / a.task
    paths.assert_not_eval(out)
    out.mkdir(parents=True, exist_ok=True)
    for s in range(a.start, a.start + a.n):
        lay = generate(a.task, s)
        (out / f"{s}.json").write_text(json.dumps(lay, indent=1))
        n_obj = sum(len(v) for t in OBJECT_TYPES for v in (lay.get(t) or {}).values())
        print(f"{a.task} {s}: {n_obj} objects ({lay['_simlab']['n_clutter']} clutter, attempt "
              f"{lay['_simlab']['attempt']}) -> {out / f'{s}.json'}")


if __name__ == "__main__":
    main()
