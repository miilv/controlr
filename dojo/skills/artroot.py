"""RoboDojo poses an Articulation on its articulation-root prim, not on the asset's default prim.

RoboDojo's ``ArticulationObject`` is an Isaac ``SingleArticulation``; Isaac's ``Articulation`` view resolves the
object's prim path to the prim carrying ``PhysicsArticulationRootAPI`` (``get_articulation_root_api_prim_path``).
The record's ``default_pos/default_ori`` lands on that prim at spawn (``translation=``) and on every reset
(``set_local_pose``), and ``get_local_pose`` (the layout manager's ``get_instance_pose``, our ``objects_gt``) reads
its physics pose back. Our catalogue meshes, the metadata bbox / place / support / functional frames and
``links.json`` are in the asset's default-prim frame. Where the root-API prim is a child with its own transform
(laptop/00000-00003: the base link ``/root/E_body_1`` at (0, -77.1, +3.5) mm), the mesh sits at
``record ∘ inv(T)``, not at the record pose. Probe on genuine RoboDojo (2026-10-27, store_laptop 100000 / 100003,
physics-only): the asset frame implied by both links is the record + (0, +78, -11) mm in the object frame (the
laptop also settles ~8 mm down after spawn); toaster, egg_holder and SpringButton agree with their records
within 0.5 mm.

Convention: our side (ObjectState, perception's records, the scorers) is in the asset frame; RoboDojo's side (the
layout records the sim loads, ``objects_gt`` rows) is in the root frame. ``to_asset`` / ``to_root`` convert where
they meet: ``world.objects_from_gt`` (GT rows -> skills, runtime mode gt, ``estimate.pose_errors``),
``estimate.estimate_to_layout`` (belief -> the twin's record), ``twin.Twin._task_labels`` (verify outcomes),
``eval_g1.score_record`` (GT rows vs perception's record).

``data/articulation_roots.json`` (``scripts/skills/articulation_roots.py``, from the asset USDs): asset key ->
{"prim", "pos", "quat"}: the root prim's transform in the asset frame (m, wxyz), only where it is not the identity.
The record's ``scale`` is not applied (every record so far has scale 1).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from dojo.skills.geometry import R_to_quat, quat_to_R

DATA = Path(__file__).with_name("data") / "articulation_roots.json"


def asset_key(obj_type: str, category: str, idx) -> str:
    return f"{obj_type}/{category}/{int(idx):05d}"


@lru_cache(maxsize=1)
def table() -> dict:
    try:
        return dict(json.loads(DATA.read_text()).get("assets") or {})
    except FileNotFoundError:
        return {}


def root_in_asset(key: str) -> np.ndarray | None:
    """4x4 transform of ``key``'s root prim in its asset frame, or None (identity: nothing to convert)."""
    e = table().get(key)
    if not e:
        return None
    T = np.eye(4)
    T[:3, :3] = quat_to_R(e["quat"])
    T[:3, 3] = e["pos"]
    return T


def _T(pos, quat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_R(quat)
    T[:3, 3] = np.asarray(pos, float)[:3]
    return T


def to_asset(pos, quat, key: str) -> tuple[np.ndarray, np.ndarray]:
    """RoboDojo's pose of ``key`` (record / ``objects_gt``) -> the pose of its asset (catalogue mesh) frame."""
    T = root_in_asset(key)
    if T is None:
        return np.asarray(pos, float), np.asarray(quat, float)
    M = _T(pos, quat) @ np.linalg.inv(T)
    return M[:3, 3].copy(), R_to_quat(M[:3, :3])


def to_root(pos, quat, key: str) -> tuple[np.ndarray, np.ndarray]:
    """The pose of ``key``'s asset frame -> the pose RoboDojo takes (record ``default_pos/ori``)."""
    T = root_in_asset(key)
    if T is None:
        return np.asarray(pos, float), np.asarray(quat, float)
    M = _T(pos, quat) @ T
    return M[:3, 3].copy(), R_to_quat(M[:3, :3])


def row_to_asset(row: dict) -> dict:
    """A GT row (``objects_gt``: type, category, category_idx, pos, quat) with pos / quat in the asset frame; the
    same dict when nothing changes (not an Articulation with a root offset, or no pose)."""
    if row.get("type") != "Articulation" or row.get("clutter") or row.get("pos") is None or row.get("quat") is None:
        return row
    key = asset_key(row["type"], row["category"], row["category_idx"])
    if root_in_asset(key) is None:
        return row
    p, q = to_asset(row["pos"], row["quat"], key)
    return {**row, "pos": p.tolist(), "quat": q.tolist(), "root_pos": list(row["pos"]), "root_quat": list(row["quat"])}


__all__ = ["asset_key", "root_in_asset", "to_asset", "to_root", "row_to_asset", "table"]
