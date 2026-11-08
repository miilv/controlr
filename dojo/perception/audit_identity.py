"""Offline write-audit identity anchors. No GT association is exposed to a controller."""
from __future__ import annotations

import bisect

import numpy as np

from dojo.perception.grasp_metric import Groups, flatten_record, grip_events, gt_rows_asset

ASSOC_RADIUS_M = 0.010  # tight initial/pick association; distant nearest matches never become persistent identities


class AuditIdentity:
    def __init__(self, report, frames, task, footprints, back=4, group=None, actions=()):
        self.fp, self.frames, self.back = footprints, frames, back
        self.steps = sorted(int(s) for s in frames)
        self.group = group if group is not None else Groups(task)
        self.bindings = {}
        self.keys = {}
        self.evidence = []
        records = report.get("records") or {}
        planning = records.get("initial") or {}
        for lab, o in flatten_record(planning).items():
            self.evidence.append((0, lab, o["pos"], o["quat"], o["key"], "initial_tight"))
            self.keys[lab] = o["key"]
        for e in report.get("events") or []:
            if e.get("kind") == "consume" and e.get("skill") == "pick" and e.get("pose7_used"):
                p = e["pose7_used"]
                self.evidence.append((int(e["step"]), e["label"], p[:3], p[3:7], e.get("key"), "consumed_pick"))
        closes = {}
        for g in grip_events(actions):
            if g["kind"] == "grasp":
                closes.setdefault(g["step"] - 1, set()).add(g["arm"])
        for sn in (report.get("dev_belief_probe") or {}).get("snaps", []):
            arms = closes.get(int(sn["step"]), set())
            picks = [e for e in report.get("events", []) if e.get("arm") in arms and e.get("label")
                     and int(e.get("step", -1)) <= int(sn["step"])
                     and ((e.get("kind") == "consume" and e.get("skill") == "pick")
                          or (e.get("kind") == "refine" and e.get("for") == "pick"))] if arms else []
            targets = {max((e for e in picks if e["arm"] == arm), key=lambda e: e["step"])["label"]
                       for arm in arms if any(e["arm"] == arm for e in picks)}
            for lab, o in sn.get("objects", {}).items():
                if o.get("pos") is not None:
                    self.evidence.append((int(sn["step"]), lab, o["pos"], o.get("quat"), o.get("key"),
                                          "grasp_close" if lab in targets else "grasp_probe"))
        self.evidence.sort(key=lambda x: x[0])
        self.index = 0

    def frame(self, step):
        i = bisect.bisect_right(self.steps, step) - 1
        if i < 0 or step - self.steps[i] > self.back:
            return None
        return self.frames[str(self.steps[i])]

    def candidates(self, label, frame):
        rows = gt_rows_asset(frame["objects"])
        return [r for r in rows if self.group.of(r["label"]) == self.group.of(label)] or rows

    def _tight(self, step, label, pos, quat, key, source):
        previous = self.bindings.get(label)
        strong = source in ("consumed_pick", "grasp_close")
        # Initial/snapshot proximity is provisional until the first physical pick consumes the label.
        # Perception may rename interchangeable instances before that; later retries never rebind it.
        if previous and (previous["identity_source"] in ("consumed_pick", "grasp_close") or not strong):
            return
        frame = self.frame(step)
        if frame is None:
            return
        p = self.fp.centre(key, pos, quat)
        ranked = sorted((float(np.linalg.norm(self.fp.centre(r["key"], r["pos"], r.get("quat")) - p)), r["label"])
                        for r in self.candidates(label, frame))
        # Do not fix an identity from an ambiguous tie, or bind two labels to one initial physical instance.
        claimed = {b["gt_label"] for b in self.bindings.values()}
        if (ranked and ranked[0][0] <= ASSOC_RADIUS_M and (strong or ranked[0][1] not in claimed)
                and (len(ranked) == 1 or ranked[1][0] - ranked[0][0] > 0.002)):
            d, gt = ranked[0]
            self.bindings[label] = {"gt_label": gt, "identity_source": source, "identity_step": step,
                                    "identity_distance_mm": round(d * 1000, 3)}
            if previous is not None and previous["gt_label"] != gt:
                self.bindings[label]["identity_prior_gt_label"] = previous["gt_label"]

    def resolve(self, write, frame, before_centre):
        step, label = write["step"], write["label"]
        while self.index < len(self.evidence) and self.evidence[self.index][0] <= step:
            evidence = self.evidence[self.index]
            if evidence[4] is not None:
                self.keys[evidence[1]] = evidence[4]
            self._tight(*evidence)
            self.index += 1
        self._tight(step, label, write["before"], write["q_before"], self.keys.get(label), "write_tight")
        if label in self.bindings:
            b = self.bindings[label]
            if not any(r["label"] == b["gt_label"] for r in gt_rows_asset(frame["objects"])):
                raise ValueError(f"anchored GT {b['gt_label']} missing at write {step}; refusing nearest substitution")
            return {**b, "identity_fallback": False}
        # Deliberately not saved: an untrustworthy nearest guess must not turn into an identity anchor.
        rows = self.candidates(label, frame)
        if not rows:
            raise ValueError(f"no GT at write {step}")
        r = min(rows, key=lambda r: float(np.linalg.norm(
            self.fp.centre(r["key"], r["pos"], r.get("quat")) - before_centre)))
        return {"gt_label": r["label"], "identity_source": "nearest_fallback", "identity_step": step,
                "identity_distance_mm": None, "identity_fallback": True}


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
