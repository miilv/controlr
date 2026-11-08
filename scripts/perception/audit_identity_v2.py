"""Offline audit v2; deliberately outside the inference-indexed tree."""
from __future__ import annotations

import numpy as np

from dojo.perception.audit_identity import AuditIdentity
from dojo.perception.grasp_metric import flatten_record, gt_rows_asset


class AuditIdentityV2(AuditIdentity):
    """Scorer-only, label-agnostic frame-zero assignment, finalized by first pick.

    Seeds farther than 50 mm are rejected. A competing object within 20 mm of
    the assigned distance makes the seed uncertain until physical pick evidence.
    """
    VERSION = "spatial-assignment-v2"

    def __init__(self, report, frames, task, footprints, **kwargs):
        # V2 never constructs task-conditioned label groups.
        from types import SimpleNamespace
        from scipy.optimize import linear_sum_assignment

        kwargs["group"] = SimpleNamespace(of=lambda label: None)
        super().__init__(report, frames, task, footprints, **kwargs)
        self.evidence = [e for e in self.evidence if e[5] != "initial_tight"]
        objects = list(flatten_record((report.get("records") or {}).get("initial") or {}).items())
        rows = gt_rows_asset(frames.get("0", {}).get("objects", []))
        if not objects or not rows:
            return
        costs = np.array([[np.linalg.norm(
            self.fp.centre(o["key"], o["pos"], o["quat"]) -
            self.fp.centre(r["key"], r["pos"], r.get("quat"))) for r in rows] for _, o in objects])
        # Dummy columns allow unmatched beliefs; maximize valid matches before distance.
        valid = costs <= .050
        penalized = np.where(valid, costs, 1e6)
        penalized = np.concatenate([penalized, np.full((len(objects), len(objects)), 1000.)], axis=1)
        ii, jj = linear_sum_assignment(penalized)
        for i, j in zip(ii, jj):
            if j >= len(rows) or not valid[i, j]:
                continue
            other = np.delete(costs[i], j)
            ambiguous = bool(len(other) and np.min(other) <= costs[i, j] + .020)
            self.bindings[objects[i][0]] = {
                "gt_label": rows[j]["label"], "identity_source": "assignment", "identity_step": 0,
                "identity_distance_mm": round(float(costs[i, j]) * 1000, 3),
                "identity_uncertain": ambiguous}

    def candidates(self, label, frame):
        return gt_rows_asset(frame["objects"])

    def _tight(self, step, label, pos, quat, key, source):
        # Snapshot/write proximity never silently replaces a global assignment.
        if source not in ("consumed_pick", "grasp_close"):
            return
        super()._tight(step, label, pos, quat, key, source)

    def resolve(self, write, frame, before_centre):
        result = super().resolve(write, frame, before_centre)
        if result["identity_fallback"]:
            row = next(r for r in self.candidates(None, frame) if r["label"] == result["gt_label"])
            result["identity_distance_mm"] = round(float(np.linalg.norm(
                self.fp.centre(row["key"], row["pos"], row.get("quat")) - before_centre)) * 1000, 3)
        result["instrument_version"] = self.VERSION
        result["identity_rule"] = ("fallback" if result["identity_fallback"] else
                                   "assignment" if result["identity_source"] == "assignment" else "pick")
        result.setdefault("identity_uncertain", False)
        return result
