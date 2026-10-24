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

GENERATOR = "simlab/v2"
# Home-pose gripper footprint of both ARX arms (link6..fingers xy at reset, from GT) + 3 cm. At home the
# fingers hang ~15 cm above the table, so only objects taller than ROBOT_KEEPOUT_TOP collide with them;
# RoboDojo's placement engine does not know the robot (v1 put bottles on the grippers: 9/34 layouts then
# failed RoboDojo's stability check).
ROBOT_KEEPOUT = [(-0.42, -0.40, -0.18, -0.13), (0.18, -0.40, 0.42, -0.13)]     # (x0, y0, x1, y1), env frame
ROBOT_KEEPOUT_TOP = 0.08

SAMPLING_DEFAULTS = {"group": None, "xlim": None, "ylim": None, "zlim": None, "qpos": [1, 0, 0, 0],
                     "rotate_deg": 0, "rotate_rand": False, "relative_plane": "Table", "place_tag": None,
                     "margin": 0.01, "check_mode": "bbox", "need_check_stable": True}


class LayoutError(RuntimeError):
    pass


def hits_robot(meta: dict, pose, polygon, plane: str) -> bool:
    """A placed object is taller than ROBOT_KEEPOUT_TOP and its footprint overlaps a home gripper."""
    if plane != "table" or polygon is None:
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
    """Task object groups per type, with the scene's own groups appended (camera_stand), as
    ``LayoutManager.process_config`` merges them."""
    tcfg, scfg = task_config(task), scene_config(task)
    out: dict[str, list] = {}
    for t in OBJECT_TYPES:
        groups = list(tcfg.get(t) or [])
        groups += list(scfg.get(t) or [])
        if groups:
            out[t] = groups
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
    return {"table": g_table, "ground": g_ground}


# --------------------------------------------------------------------------- selection
def _count(n, rng: random.Random) -> int:
    if isinstance(n, (list, tuple)):
        return int(n[0]) if len(n) == 1 else rng.randint(int(n[0]), int(n[1]))
    return int(n)


def select_instances(obj_type: str, group: dict, rng: random.Random) -> list[tuple[dict, int]]:
    """(category entry, asset index) per instance, following ``select_mode``."""
    entries = []
    for e in group["category"]:
        idx = list(e.get("index") if e.get("index") is not None else available_indices(obj_type, e["name"]))
        entries.append((e, [int(i) for i in idx]))
    sm = group.get("select_mode") or {}
    mode = sm.get("mode", "allow_duplicate")
    if mode == "hierarchical":
        n_cat = _count(sm.get("select_category_nums", [1, 1]), rng)
        n_inst = _count(sm.get("select_instance_nums", [1, 1]), rng)
        groups: dict[Any, list] = {}
        for e, idx in entries:
            groups.setdefault(e.get("group"), []).append((e, idx))
        out = []
        for g in rng.sample(sorted(groups, key=str), n_cat):
            pool = [(e, i) for e, idx in groups[g] for i in idx]
            if sm.get("instance_sample_mode", "unique") == "unique":
                out += rng.sample(pool, n_inst)
            else:
                out += [rng.choice(pool) for _ in range(n_inst)]
        return out
    n = _count(sm.get("nums", 1), rng)
    pool = [(e, i) for e, idx in entries for i in idx]
    if mode == "same":
        return [rng.choice(pool)] * n
    if mode == "unique":
        out, seen = [], set()
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
def place_task_objects(groups: dict[str, list], gens: dict, rng: random.Random, layout: dict) -> None:
    for obj_type, glist in groups.items():
        for group in glist:
            common = dict(group.get("common") or {})
            labels = list((group.get("select_mode") or {}).get("label") or [])
            for k, (entry, idx) in enumerate(select_instances(obj_type, group, rng)):
                prm = dict(SAMPLING_DEFAULTS)
                prm.update(common)
                prm.update({key: v for key, v in entry.items() if key not in ("name", "index")})
                cat = entry["name"]
                meta = metadata(obj_type, cat, idx)
                if meta is None or not usd_exists(obj_type, cat, idx):
                    raise LayoutError(f"missing asset {obj_type}/{cat}/{idx:05d}")
                plane = str(prm["relative_plane"]).split("/")[0].lower()
                gen = gens["ground" if plane == "ground" else "table"]
                label = labels[k] if k < len(labels) else f"{cat}{k}"
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
                phys = {k2: v for k2, v in (meta.get("physics") or {}).items() if k2 in ("mass", "friction")}
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


def generate(task: str, seed: int, max_tries: int = 20) -> dict:
    """One layout record for ``task`` (seed >= SEED_BASE). Deterministic in ``seed``."""
    if seed < paths.SEED_BASE:
        raise ValueError(f"our layout seeds start at {paths.SEED_BASE}")
    tcfg, scfg = task_config(task), scene_config(task)
    groups = merged_object_groups(task)
    task_cats = {e["name"] for gl in groups.values() for g in gl for e in g["category"]}
    last = None
    for attempt in range(max_tries):
        sub = seed * 1000 + attempt
        np.random.seed(sub % (2**32))
        rng = random.Random(sub)
        gens = make_generators(scfg)
        for a in tcfg.get("ProhibitedArea") or []:
            x0, y0, x1, y1 = a
            gens["table"].add_box_prohibited_area(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        layout: dict = {}
        try:
            place_task_objects(groups, gens, rng, layout)
            n_clutter = place_clutter(tcfg.get("Clutter") or [], gens, rng, layout, exclude=task_cats)
        except LayoutError as e:
            last = e
            continue
        # RoboDojo's own records hold fewer clutter objects than `nums` when the strips are tight
        # (push_T_random eval record: ~13 of 20), so a partial fill is kept, not retried.
        want = sum(int(b.get("nums", 0)) for b in tcfg.get("Clutter") or [])
        layout.update(fixtures_from_scene(scfg))
        errs = validate(layout)
        if errs:
            raise LayoutError(f"generated an invalid record: {errs[:5]}")
        layout["_simlab"] = {"task": task, "id": seed, "seed": seed, "attempt": attempt, "generator": GENERATOR,
                             "n_clutter": n_clutter, "clutter_wanted": want}
        return layout
    raise LayoutError(f"{task} seed {seed}: no valid layout in {max_tries} tries ({last})")


def strip_meta(layout: dict) -> dict:
    """The record RoboDojo loads (our ``_simlab`` metadata key removed)."""
    return {k: v for k, v in layout.items() if not k.startswith("_")}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--start", type=int, default=paths.SEED_BASE)
    ap.add_argument("--out", default=None, help="default /root/dojo-data/simlab/layouts/<task>")
    a = ap.parse_args(argv)
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
