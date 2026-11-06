"""Serializable cuRobo-v2 stage evidence; no simulator or GPU imports."""
from __future__ import annotations

import numpy as np


def values(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def result_flags(result):
    if result is None:
        return {"returned": False}
    out = {"returned": True}
    status = getattr(result, "status", None)
    if status is not None:
        out["native_status"] = str(status)
    for name in ("success", "feasible", "position_error", "rotation_error"):
        data = values(getattr(result, name, None))
        if data is not None:
            out[name] = data.reshape(-1).tolist()
    for name in ("position_tolerance", "orientation_tolerance"):
        data = values(getattr(result, name, None))
        if data is not None:
            out[name] = float(np.max(data))
    metrics = getattr(result, "metrics", None)
    constraints = getattr(metrics, "costs_and_constraints", None)
    if constraints is not None:
        out["constraints"] = constraint_flags(constraints)
        out["start_constraints"] = constraint_flags(constraints, 0)
        out["goal_constraints"] = constraint_flags(constraints, -1)
    return out


def constraint_flags(collection, horizon=None):
    out = {}
    for group in ("constraints", "hybrid_costs_constraints"):
        items = getattr(collection, group, None)
        if items is None:
            continue
        for name, value in zip(items.names, items.values):
            data = values(value)
            if horizon is not None and data.ndim >= 2:
                data = data[:, horizon]
            out[name] = float(np.max(data)) if data.size else 0.
    return out


def failure_category(diagnostics):
    """Stage-derived labels: installed v2 has no MotionGenStatus enum."""
    start = diagnostics.get("start", {})
    if not start.get("constraints"):
        start = {"constraints": diagnostics.get("trajopt", {}).get("start_constraints", {})}
    collision = lambda data: any(v > 0 for k, v in data.get("constraints", {}).items()
                                 if "collision" in k)
    if collision(start):
        return "start_state_collision"
    ik = diagnostics.get("ik", {})
    if ik and not any(ik.get("success", [])):
        pos, rot = ik.get("position_error", []), ik.get("rotation_error", [])
        if collision(ik) and any(p < ik.get("configured_position_tolerance", ik.get("position_tolerance") or .001) and r < ik.get("configured_orientation_tolerance", ik.get("orientation_tolerance") or .02) for p, r in zip(pos, rot)):
            return "goal_collision"
        return "ik_fail"
    if diagnostics.get("graph") and not any(diagnostics["graph"].get("success", [])):
        return "graph_fail"
    if diagnostics.get("trajopt"):
        return "trajopt_fail"
    return "unclassified"
