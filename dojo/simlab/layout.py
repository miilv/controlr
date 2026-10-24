"""RoboDojo layout records: schema, fixtures, validation, perturbation (pure Python + numpy).

A RoboDojo episode is fully described by one JSON layout record. RoboDojo's eval client reads it
from ``Assets/Eval_Layout/RoboDojo/<cfg>/<seed>/<task>_<n>.json`` (``SeedManager``) and hands it to
``LayoutManager.load_saved_layout``; nothing is sampled at eval time (the sampler that wrote
the eval files is not public). The record (all poses in the env frame, metres, quaternions
``[w, x, y, z]``):

```
{
  "Rigid":    {<category>: [<instance>, ...], ...},   # task objects AND clutter
  "Geometry": {<category>: [<instance>, ...]},         # static, no rigid body (pads, dustbin, camera_stand)
  "Articulation" / "Dynamic" / "Garment" / "Fluid": same shape, per task
  "Room":       {"default", "default_pos", "default_rot", "scale"},
  "Table":      {"default", "scale", "default_pos", "default_ori"},
  "Ground":     {"geometry", "thickness", "axis", "default_pos", "materials", "physics_material"},
  "Background": {"intensity", "category_name"},
}
```

Task-object instance (from the task config + the asset's ``metadata.json``):
``category, category_idx, group, xlim, ylim, zlim, qpos, rotate_deg, rotate_rand, relative_plane,
place_tag, margin, check_mode, need_check_stable, label, default_pos, default_ori, scale,
physics {mass, friction, type}, visual {}``. RoboDojo uses ``label`` (reward predicates),
``category``/``category_idx`` (USD path ``Objects/<Type>/<category>/<idx:05d>/object.usd[z]``),
``default_pos/ori``, ``scale``, ``physics.type`` (which class spawns it), ``physics.mass``
(capped at 0.5 kg; ``friction`` is not read: rigid bodies get static 0.6 / dynamic 1.5),
``need_check_stable`` and ``relative_plane`` (layout stability check), ``visual`` (colour or
material override). The sampling fields (xlim ... check_mode) are carried along, not read.

Clutter instance (``_random`` configs), stored under ``Rigid`` with ``type: "cluttered"``:
``category_idx, physics {type, [mass, friction]}, scale, type, clutter_idx, yaml_path,
relative_plane, default_pos, default_ori`` (USD under ``Objects/Clutter/<category>/``).

There is no ``Light`` key: lighting is the Background dome + the room USD. Fixtures equal
``env_cfg/scene/default.yml`` (randomisation is off in ``arx_x5.yml``).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterator

import numpy as np

OBJECT_TYPES = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")
FIXTURE_KEYS = ("Room", "Table", "Ground", "Background")
TABLE_TOP_Z = 0.765            # scene default: Table pos z 0.74 + thickness 0.05 / 2
GROUND_TOP_Z = 0.05            # Ground thickness 0.1 / 2

TASK_FIELDS = ("category", "category_idx", "group", "xlim", "ylim", "zlim", "qpos", "rotate_deg",
               "rotate_rand", "relative_plane", "place_tag", "margin", "check_mode", "need_check_stable",
               "label", "default_pos", "default_ori", "scale", "physics", "visual")
CLUTTER_FIELDS = ("category_idx", "physics", "scale", "type", "clutter_idx", "yaml_path", "relative_plane",
                  "default_pos", "default_ori")


# --------------------------------------------------------------------------- quaternions (wxyz)
def quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_z(deg: float) -> np.ndarray:
    h = math.radians(deg) / 2
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


def quat_angle_deg(a, b) -> float:
    """Rotation angle between two orientations (degrees, 0..180)."""
    d = abs(float(np.dot(np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b))))
    return math.degrees(2 * math.acos(min(1.0, d)))


def yaw_deg(q) -> float:
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


# --------------------------------------------------------------------------- fixtures
def fixtures_from_scene(scene_cfg: dict) -> dict:
    """Room/Table/Ground/Background exactly as the eval records carry them (scene default.yml,
    randomisation off)."""
    room = scene_cfg["Room"]
    ground = copy.deepcopy(scene_cfg["Ground"])
    ground.setdefault("materials", {})["random"] = False
    bg = scene_cfg["Background"]
    out = {
        "Room": {"default": room["default"], "default_pos": list(room["default_pos"]),
                 "default_rot": list(room.get("default_ori", [1.0, 0.0, 0.0, 0.0])), "scale": list(room["scale"])},
        "Ground": ground,
        "Background": {"intensity": bg["intensity"], "category_name": bg["default"]},
    }
    if "Table" in scene_cfg:                  # the conveyor scene has none
        table = scene_cfg["Table"]
        out["Table"] = {"default": table["default"], "scale": list(table["scale"]),
                        "default_pos": list(table["default_pos"]), "default_ori": list(table["default_ori"])}
    return out


# --------------------------------------------------------------------------- access
def iter_objects(layout: dict) -> Iterator[tuple[str, str, dict]]:
    """(object_type, category, instance) for every object instance in the record."""
    for t in OBJECT_TYPES:
        for cat, insts in (layout.get(t) or {}).items():
            for inst in insts:
                yield t, cat, inst


def is_clutter(inst: dict) -> bool:
    return inst.get("type") == "cluttered"


def object_table(layout: dict) -> list[dict]:
    """Flat summary: one row per object (no sampling fields)."""
    rows = []
    for t, cat, inst in iter_objects(layout):
        rows.append({"type": t, "category": cat, "category_idx": int(inst["category_idx"]),
                     "label": inst.get("label"), "clutter": is_clutter(inst),
                     "pos": [float(v) for v in inst["default_pos"]],
                     "quat": [float(v) for v in inst["default_ori"]]})
    return rows


def find(layout: dict, label: str) -> dict:
    for _, _, inst in iter_objects(layout):
        if inst.get("label") == label:
            return inst
    raise KeyError(label)


def validate(layout: dict) -> list[str]:
    """Schema problems (empty list = loadable by RoboDojo's LayoutManager)."""
    errs = []
    for k in ("Room", "Ground", "Background"):    # Table: only in table scenes
        if k not in layout:
            errs.append(f"missing fixture {k}")
    labels = set()
    for t, cat, inst in iter_objects(layout):
        need = CLUTTER_FIELDS if is_clutter(inst) else TASK_FIELDS
        for f in need:
            if f not in inst:
                errs.append(f"{t}/{cat}: missing {f}")
        if len(inst.get("default_pos", [])) != 3 or len(inst.get("default_ori", [])) != 4:
            errs.append(f"{t}/{cat}: bad pose")
        else:
            n = float(np.linalg.norm(inst["default_ori"]))
            if abs(n - 1) > 1e-3:
                errs.append(f"{t}/{cat}: quaternion norm {n:.4f}")
        ptype = (inst.get("physics") or {}).get("type")
        if ptype != t.lower():
            errs.append(f"{t}/{cat}: physics.type {ptype!r} != {t.lower()!r}")
        lab = inst.get("label")
        if lab is not None:
            if lab in labels:
                errs.append(f"duplicate label {lab}")
            labels.add(lab)
    return errs


def perturb(layout: dict, label: str, dxyz=(0.0, 0.0, 0.0), dyaw_deg: float = 0.0) -> dict:
    """Copy of ``layout`` with one object shifted (env frame) and turned about world z."""
    out = copy.deepcopy(layout)
    inst = find(out, label)
    inst["default_pos"] = [float(a + b) for a, b in zip(inst["default_pos"], dxyz)]
    if dyaw_deg:
        q = quat_mul(quat_z(dyaw_deg), inst["default_ori"])
        inst["default_ori"] = [float(v) for v in q / np.linalg.norm(q)]
    return out
