"""Persistent per-instance evidence: what perception has seen of every instance of an episode, kept for its length.

General for every episode and vocabulary (no task names, no config labels as meaning). Two parts:

* **instances**: one record per believed object label, with the identity evidence every read gave it (``identity`` votes:
  the asset key a read proposed and its confidence) kept apart from the pose writes. A read whose key differs from the
  current one is a vote, not a write; the key changes only when the votes for another key outweigh the current one by
  ``switch_margin`` (``Ledger.identity``);
* **candidates**: foreground regions of the head view that no believed object explains (the estimate's
  ``_perception.unexplained``: table-plane xy + pixel area), object-sized, not over a fixture's footprint and not on a
  believed object. A candidate is kept, never dropped, until a look confirms it (it becomes an instance) or rejects it;
  each later read that finds a region there again adds a sighting.

The ledger never writes the world itself: consumers (a look that confirms a candidate, a skill that reads identity
votes) decide. Measured offline on recorded camera frames (journal 2026-11-07-dojo-classify3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CAND_MIN_PX = 150        # a head-view region this small is a fragment or an artefact, not an unseen object
CAND_MAX_PX = 2000       # larger: several objects or a shadow merged into one region; kept apart as "merged"
CAND_NEAR_M = 0.04       # a region this close (xy) to a believed movable object is that object's unexplained part
CAND_SAME_M = 0.03       # a later sighting this close to a candidate is the same candidate
FIXTURE_PAD_M = 0.01     # a region inside a fixture's footprint grown by this is the fixture's (a basket's inside)


@dataclass
class Candidate:
    xy: tuple
    area_px: int
    first_step: int
    sightings: int = 1
    status: str = "open"           # open | confirmed | rejected
    merged: bool = False           # an over-sized region (several objects or a shadow)
    resolution: dict = field(default_factory=dict)


def _rows(record: dict, typ: str):
    for cat, lst in (record.get(typ) or {}).items():
        for r in lst:
            yield cat, r


def candidate_regions(record: dict, half_extent, min_px: int = CAND_MIN_PX, max_px: int = CAND_MAX_PX,
                      near_m: float = CAND_NEAR_M) -> list[dict]:
    """The estimate record's unexplained head regions that may be unseen objects: area >= ``min_px``, centre not over a
    Geometry object's footprint (``half_extent(key)`` -> (hx, hy) of its box, axis-aligned at its yaw; None: skipped),
    farther than ``near_m`` from every believed Rigid object. Regions above ``max_px`` come back with ``merged``."""
    fixtures = []
    for cat, r in _rows(record, "Geometry"):
        he = half_extent(f"Geometry/{cat}/{int(r['category_idx']):05d}")
        if he is None:
            continue
        q = np.asarray(r.get("default_ori") or [1, 0, 0, 0], float)
        yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        fixtures.append((np.asarray(r["default_pos"][:2], float), yaw, np.asarray(he, float) + FIXTURE_PAD_M))
    movables = [np.asarray(r["default_pos"][:2], float) for _, r in _rows(record, "Rigid")]
    out = []
    for u in (record.get("_perception") or {}).get("unexplained") or []:
        a, xy = int(u.get("area_px", 0)), np.asarray(u["xy"][:2], float)
        if a < min_px:
            continue
        on_fixture = False
        for c, yaw, he in fixtures:
            d = xy - c
            loc = np.array([np.cos(yaw) * d[0] + np.sin(yaw) * d[1], -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]])
            if np.all(np.abs(loc) <= he):
                on_fixture = True
                break
        if on_fixture or any(float(np.linalg.norm(xy - m)) <= near_m for m in movables):
            continue
        out.append({"xy": [round(float(v), 4) for v in xy], "area_px": a, "merged": a > max_px})
    return out


class Ledger:
    """The episode's evidence. ``update(record, step, half_extent)`` after every estimate / re-estimate; ``vote`` for
    every identity a read proposes; ``resolve`` when a look confirms or rejects a candidate."""

    def __init__(self, switch_margin: float = 0.5):
        self.candidates: list[Candidate] = []
        self.votes: dict[str, dict[str, float]] = {}
        self.current: dict[str, str] = {}
        self.switch_margin = float(switch_margin)
        self.events: list[dict] = []

    # ---------------------------------------------------------------- candidates
    def update(self, record: dict, step: int, half_extent) -> list[Candidate]:
        """New / re-sighted candidates from one estimate record; returns the open ones."""
        for cat, r in _rows(record, "Rigid"):
            lab = r.get("label") or cat
            key = f"Rigid/{cat}/{int(r['category_idx']):05d}"
            conf = float((r.get("_perception") or {}).get("confidence") or 0.0)
            self.current.setdefault(lab, key)
            self.vote(lab, key, conf, step, source="estimate")
        for c in candidate_regions(record, half_extent):
            xy = np.asarray(c["xy"], float)
            hit = next((k for k in self.candidates if k.status == "open"
                        and float(np.linalg.norm(np.asarray(k.xy) - xy)) <= CAND_SAME_M), None)
            if hit is not None:
                hit.sightings += 1
                hit.area_px = max(hit.area_px, c["area_px"])
            else:
                self.candidates.append(Candidate(tuple(c["xy"]), c["area_px"], int(step), merged=c["merged"]))
                self.events.append({"kind": "candidate", "step": int(step), "xy": c["xy"], "area_px": c["area_px"],
                                    "merged": c["merged"]})
        return [k for k in self.candidates if k.status == "open"]

    def resolve(self, cand: Candidate, confirmed: bool, step: int, **detail) -> None:
        cand.status = "confirmed" if confirmed else "rejected"
        cand.resolution = {"step": int(step), **detail}
        self.events.append({"kind": "candidate_" + cand.status, "step": int(step), "xy": list(cand.xy), **detail})

    # ---------------------------------------------------------------- identity
    def vote(self, label: str, key: str, conf: float, step: int, source: str) -> None:
        """One read's identity for ``label`` (asset key, confidence in [0, 1]); kept apart from pose writes."""
        v = self.votes.setdefault(label, {})
        v[key] = v.get(key, 0.0) + max(0.0, float(conf))
        if key != self.current.get(label):
            self.events.append({"kind": "identity_vote", "label": label, "step": int(step), "key": key,
                                "conf": round(float(conf), 3), "source": source})

    def identity(self, label: str) -> str | None:
        """The key the evidence supports: the current one unless another key's votes exceed it by ``switch_margin``."""
        cur, v = self.current.get(label), self.votes.get(label) or {}
        if not v:
            return cur
        best = max(v, key=v.get)
        if cur is None or (best != cur and v[best] >= v.get(cur, 0.0) + self.switch_margin):
            if best != cur:
                self.events.append({"kind": "identity_switch", "label": label, "from": cur, "to": best})
            self.current[label] = best
        return self.current[label]

    def open_candidates(self) -> list[Candidate]:
        return [k for k in self.candidates if k.status == "open"]

    def summary(self) -> dict:
        return {"candidates": [{"xy": list(k.xy), "area_px": k.area_px, "first_step": k.first_step,
                                "sightings": k.sightings, "status": k.status, "merged": k.merged}
                               for k in self.candidates],
                "identity": dict(self.current),
                "contested": sorted(lab for lab, v in self.votes.items() if len(v) > 1)}
