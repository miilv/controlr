"""WorldState -> the scene the agent reads.

The agent never sees WorldState labels: in a dev layout they are the CHECKER's labels ("target", "cat0_1",
"red", "board0"), which would leak the answer. Objects get neutral ids ``<category>_<k>`` (k = left-to-right
order within the category, world x ascending) and the agent's choices are mapped back to WorldState labels
through ``SceneIds``. What an object is described by comes only from what perception reports (catalogue
category + asset identity + pose) and the asset's PUBLIC files (``metadata.json`` bbox, ``description.json``
caption: name / colour / material), which perception's identity makes available at eval time too.

``scene_text(world, ...)`` is the one presentation used by the grounding benchmark (Phase 1) and by the agent
loop (Phase 2), so a grounding result transfers.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from dojo.skills.world import ObjectState, WorldState

# glyph identity (perception reads it by textured render-and-compare): the digit an asset shows
DIGIT_CATEGORIES = {"number": lambda idx: idx % 10, "number_card": lambda idx: idx}


def shown_digit(o: ObjectState) -> int | None:
    """The digit an object shows: perception's read (``o.digit``), None when perception could not read it
    (``digit_source`` "unread": its index is a placeholder), else its asset's (GT, objects not from perception)."""
    if getattr(o, "digit", None) is not None:
        return int(o.digit)
    if getattr(o, "digit_source", None) == "unread" or o.category not in DIGIT_CATEGORIES:
        return None
    return DIGIT_CATEGORIES[o.category](int(o.idx))


SKIP_CATEGORIES = {"camera_stand"}           # fixed rig furniture, never part of a task
KIND = {"Rigid": "movable", "Geometry": "fixed", "Articulation": "articulated", "Dynamic": "dynamic",
        "Fluid": "fluid", "Garment": "cloth"}


@lru_cache(maxsize=4096)
def _caption(objects_root: str, key: str) -> dict:
    p = Path(objects_root) / key / "description.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    cap = d.get("caption") or {}
    out = {k: (v[0] if isinstance(v, list) and v else v) for k, v in cap.items() if k in ("name", "color", "material")}
    names = cap.get("name")
    out["names"] = [str(n) for n in names] if isinstance(names, list) else ([str(names)] if names else [])
    if not out["names"]:            # no caption: a short description is the asset's name ("yellow ice axe")
        desc = d.get("description")
        desc = [desc] if isinstance(desc, str) else (desc if isinstance(desc, list) else [])
        if 0 < len(desc) <= 4 and all(isinstance(x, str) and len(x) <= 40 for x in desc):
            out["names"] = list(desc)
    return out


def catalog_roots() -> list[Path]:
    env = [c for c in os.environ.get("DOJO_CATALOG", "").split(os.pathsep) if c]
    roots = env + ["~/dojo-perception-data/catalog"]
    return [Path(os.path.expanduser(r)) for r in roots if Path(os.path.expanduser(r)).is_dir()]


@lru_cache(maxsize=4096)
def _catalog_colour(key: str) -> str | None:
    """Colour word nearest the asset's rendered mean colour in perception's catalogue (the colour the cameras see;
    the asset captions are sometimes wrong, e.g. a red cube captioned "metallic gray")."""
    from dojo.perception.language import COLOUR_RGB
    for root in catalog_roots():
        p = root / key / "info.json"
        if p.is_file():
            try:
                c = json.loads(p.read_text()).get("mean_color")
            except (OSError, ValueError):
                return None
            if c is None:
                return None
            c = np.asarray(c, float)[:3]
            names = [n for n in COLOUR_RGB if n != "grey"]
            return min(names, key=lambda n: float(np.linalg.norm(np.asarray(COLOUR_RGB[n]) - c)))
    return None


def caption(o: ObjectState, objects_root: str | Path | None) -> dict:
    if objects_root is None:
        return {}
    key = o.asset.key if not o.clutter else f"Clutter/{o.category}/{int(o.idx):05d}"
    cap = dict(_caption(str(objects_root), key))
    col = _catalog_colour(key)
    if col is not None:
        cap["color"] = col
    return cap


@dataclass
class SceneIds:
    """Neutral id <-> WorldState label (one episode)."""
    to_label: dict[str, str] = field(default_factory=dict)
    to_id: dict[str, str] = field(default_factory=dict)

    def label(self, oid: str) -> str | None:
        return self.to_label.get(str(oid))

    def oid(self, label: str) -> str | None:
        return self.to_id.get(label)

    def map_args(self, args, to: str = "label"):
        """Replace ids by labels (``to='label'``) or labels by ids (``to='id'``) anywhere in a JSON-like value."""
        table = self.to_label if to == "label" else self.to_id
        if isinstance(args, str):
            return table.get(args, args)
        if isinstance(args, dict):
            return {k: self.map_args(v, to) for k, v in args.items()}
        if isinstance(args, (list, tuple)):
            return [self.map_args(v, to) for v in args]
        return args


def visible_objects(world: WorldState) -> list[ObjectState]:
    return [o for o in world.objects.values() if o.category not in SKIP_CATEGORIES]


def assign_ids(world: WorldState) -> SceneIds:
    ids = SceneIds()
    by_cat: dict[str, list[ObjectState]] = {}
    for o in visible_objects(world):
        by_cat.setdefault(o.category, []).append(o)
    for cat in sorted(by_cat):
        for k, o in enumerate(sorted(by_cat[cat], key=lambda o: (round(float(o.pos[0]), 3), float(o.pos[1])))):
            oid = f"{cat}_{k}"
            ids.to_label[oid] = o.label
            ids.to_id[o.label] = oid
    return ids


def _fmt(v: float, nd: int = 2) -> str:
    return f"{v:+.{nd}f}"


def frame_tags(o: ObjectState, limit: int = 8) -> list[str]:
    """The asset's named frames a skill can target (public metadata): support points (pegs, poles: the
    ``pegs`` of ``stack_on_pegs``) and functional points (slots, levers, buttons: ``slots`` / ``press_tag`` of
    ``slots_and_press``)."""
    tags = [t for t in (o.asset.support or {}) if t not in ("up", "down", "top", "bottom")]
    tags += [t for t in list(o.asset.functional or {}) + list(o.asset.active_functional or {}) if t not in tags]
    return tags[:limit]


SERIES_RE = re.compile(r"^(.*?)(\d+)$")


def _frame_point_w(o: ObjectState, tag: str) -> np.ndarray | None:
    """World position of a named frame (the mean of its points): support centres, else functional frames."""
    P = [np.asarray(c, float)[:3] for c, _ in (o.asset.support or {}).get(tag) or []]
    if not P:
        raw = (o.asset.functional or {}).get(tag) or (o.asset.active_functional or {}).get(tag) or {}
        P = [np.asarray(f, float)[:3] for f in (raw.get("frame") or []) if len(f) >= 3]
    if not P:
        return None
    return o.R @ np.mean(P, axis=0) + np.asarray(o.pos, float)


FRAME_ENTRIES_V4 = 16          # entries per object (a numbered series is one entry, never cut)


def frames_v4(o: ObjectState) -> str:
    """Scene v4 frames: every named frame (v3 cut the list at 8: a xylophone's ``hit_*`` keys, a board's ninth cell
    and a rack's ninth and tenth holes were missing), each name once; a numbered series (``pole/0`` .. ``pole/3``,
    ``hit_0`` .. ``hit_7``) as one entry with each member's world position, so its order in space ("left to right")
    and its heights (pegs) can be read. At most ``FRAME_ENTRIES_V4`` entries."""
    tags = [t for t in (o.asset.support or {}) if t not in ("up", "down", "top", "bottom")]
    tags += list(o.asset.functional or {}) + list(o.asset.active_functional or {})
    tags = list(dict.fromkeys(tags))
    series: dict[str, list[tuple[int, str]]] = {}
    for t in tags:
        m = SERIES_RE.match(t)
        if m:
            series.setdefault(m.group(1), []).append((int(m.group(2)), t))
    out, done = [], set()
    for t in tags:
        if t in done:
            continue
        m = SERIES_RE.match(t)
        members = sorted(series.get(m.group(1), [])) if m else []
        if len(members) < 2:
            out.append(t)
            done.add(t)
            continue
        names = [n for _, n in members]
        done.update(names)
        pos = []
        for n in names:
            p = _frame_point_w(o, n)
            pos.append("?" if p is None else f"({_fmt(p[0])}, {_fmt(p[1])}, {p[2]:.2f})")
        out.append(f"{names[0]} .. {names[-1]} at " + ", ".join(pos))
    return "; ".join(out[:FRAME_ENTRIES_V4])


def object_line(o: ObjectState, oid: str, objects_root=None, all_names: bool = True, frames: bool = False,
                frames_version: int = 3) -> str:
    """``all_names``: every caption name of the asset (``"hoe" / "ice axe" / ...``; instructions name objects by
    any of their descriptions), else only the first one when it differs from the category (scene v1).
    ``frames_version`` 3: the first 8 frame names (``frame_tags``); 4: ``frames_v4``."""
    ext = np.asarray(o.asset.extents, float) * 100
    cap = caption(o, objects_root)
    desc = []
    if all_names and cap.get("names"):
        desc.append("names " + " / ".join(f'"{n}"' for n in cap["names"][:4]))
    elif cap.get("name") and cap["name"] != o.category.replace("_", " "):
        desc.append(f'"{cap["name"]}"')
    if cap.get("color"):
        desc.append(f"colour {cap['color']}")
    if o.category in DIGIT_CATEGORIES or getattr(o, "digit", None) is not None:
        d = shown_digit(o)
        desc.append(f"shows {d}" if d is not None else "digit unread")
    if frames and frames_version >= 4:
        fr4 = frames_v4(o)
        if fr4:
            desc.append(f"frames [{fr4}]")
    elif frames:
        fr = frame_tags(o)
        if fr:
            desc.append("frames " + ", ".join(fr))
    c = o.center_w
    tilt = o.tilt()
    pose = f"at ({_fmt(c[0])}, {_fmt(c[1])}, {c[2]:.2f})"
    rot = f"yaw {math.degrees(o.yaw()):.0f}" + (f", tilted {tilt:.0f}" if tilt > 10 else "")
    size = f"size {ext[0]:.1f}x{ext[1]:.1f}x{ext[2]:.1f} cm"
    kind = KIND.get(o.obj_type, o.obj_type.lower())
    return (f"- {oid}: {o.category.replace('_', ' ')} ({kind}, asset {int(o.idx)})"
            + (f", {', '.join(desc)}" if desc else "") + f"; {pose}, {rot}, {size}")


def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def scene_text(world: WorldState, ids: SceneIds | None = None, objects_root=None, step_lim: int | None = None,
               max_objects: int = 60, version: int = 2) -> tuple[str, SceneIds]:
    """``version`` 1: the first caption name only (Phase 1 run 1); 2: all caption names; 3: + the asset's named
    support / functional frames (J2: ``stack_on_pegs`` / ``slots_and_press`` take them as arguments), the first 8;
    4: every frame, numbered series with their members' positions (``frames_v4``), and the header says what frames
    are (2026-10-27 routing: with v2 every frame-taking skill was answered with code, v3 hid the xylophone's keys)."""
    ids = ids or assign_ids(world)
    head = ("SCENE (world frame, metres: x to the robot's right, y away from the robot, z up; table top z 0.765;"
            " arm bases left (-0.30, -0.45), right (+0.30, -0.45); positions are object centres, yaw/tilt in deg")
    if version >= 4:
        head += ("; frames = the named points of an object a skill argument can name (slots, pegs, hooks, keys, "
                 "levers, cells), a numbered series with each member's world position")
    lines = [head + ")"]
    objs = sorted(visible_objects(world), key=lambda o: _natural(ids.oid(o.label) or o.label))
    for o in objs[:max_objects]:
        lines.append(object_line(o, ids.oid(o.label), objects_root, all_names=version >= 2, frames=version >= 3,
                                 frames_version=version))
    if len(objs) > max_objects:
        lines.append(f"- ... {len(objs) - max_objects} more objects not listed")
    arms = []
    for a, s in sorted(world.arms.items()):
        arms.append(f"{a} gripper {'open' if s.gripper > 0.5 else 'closed'}"
                    + (f", holding {ids.oid(s.holding) or s.holding}" if s.holding else ""))
    lines.append("ARMS: at their start pose; " + "; ".join(arms) if arms else "ARMS: at their start pose")
    lim = step_lim if step_lim is not None else world.step_lim
    if lim:
        lines.append(f"STEP BUDGET: {lim - world.step} of {lim} env steps left")
    return "\n".join(lines), ids


__all__ = ["SceneIds", "assign_ids", "scene_text", "object_line", "visible_objects", "caption"]
