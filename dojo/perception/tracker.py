"""Episode object-state filter: the scene the policy believes, carried from perceive call to perceive call.

Rules (on J1-style objects: label, category, asset, pos, quat, confidence, source):
1. Pairing: each tracked object takes the current reading of its category nearest to it (labels break ties), within
   ``pair_m``.
2. Jitter: a reading within ``jitter_m`` of the tracked pose keeps the tracked pose (no drift from re-reads).
3. Only touched objects move: an object no gripper came near since its last good reading (``touch_m`` around any TCP
   seen at a call, or a gripper closed next to it) keeps its pose unless the new reading is confident
   (>= ``move_conf``) AND at least as confident as the tracked one. Else the reading is logged
   ``implausible_move``. This was J1's failure class: the untouched base re-read 19 mm off under a hovering arm.
4. Unseen untouched objects stay at their last good pose with ``age`` + 1 (``source: "prior"``). Unseen touched
   objects go to ``last_seen`` (they may be in the jaws or moved).
5. New readings with no tracked object are added (``new_reading``).
6. Support: a non-held object reading more than ``float_m`` above the table with nothing tracked under it gets its
   confidence halved (``floating``).
State lives in the ``Tracker`` instance (one per episode); ``update`` returns (objects, last_seen, events).
"""
from __future__ import annotations

import math

import numpy as np

TABLE_Z = 0.765
TCP_OFFSET = 0.1501  # flange -> grasp point along flange +x (robot config)


def tcp_from_flange(flange) -> np.ndarray:
    from .cameras import quat_wxyz_to_R

    f = np.asarray(flange, float)
    return f[:3] + quat_wxyz_to_R(f[3:7])[:, 0] * TCP_OFFSET


def _xy(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cat(o: dict) -> str | None:
    a = str(o.get("asset") or "").split("/")
    return a[1] if len(a) == 3 else o.get("category")


class Tracker:
    def __init__(self, pair_m: float = 0.08, jitter_m: float = 0.004, touch_m: float = 0.08, move_conf: float = 0.8,
                 float_m: float = 0.015, relocate_conf: float = 0.5):
        self.pair_m, self.jitter_m, self.touch_m, self.move_conf = pair_m, jitter_m, touch_m, move_conf
        self.float_m, self.relocate_conf = float_m, relocate_conf
        self.objects: list[dict] = []   # each carries "_tid" (stable id) and "age"
        self.touched: set[int] = set()  # _tids touched since their last confident reading
        self.calls = 0
        self._next = 0

    def _tid(self) -> int:
        self._next += 1
        return self._next

    def _mark_touched(self, tcps: list[np.ndarray], closed: list[np.ndarray]) -> None:
        for o in self.objects:
            p = np.asarray(o["pos"], float)
            if any(_xy(p, t) < self.touch_m and abs(p[2] - t[2]) < 0.15 for t in tcps) or \
               any(_xy(p, t) < 0.05 for t in closed):
                self.touched.add(o["_tid"])

    @staticmethod
    def _public(o: dict) -> dict:
        return {k: v for k, v in o.items() if k != "_tid"}

    def update(self, readings: list[dict], flange: dict | None = None, grip_cmd: dict | None = None) -> tuple[list, list, list]:
        self.calls += 1
        tcps, closed = [], []
        for arm, f in (flange or {}).items():
            t = tcp_from_flange(f)
            tcps.append(t)
            g = (grip_cmd or {}).get(arm, 1.0)
            if (g if not isinstance(g, str) else (0.0 if g.startswith("clos") else 1.0)) < 0.5:
                closed.append(t)
        events = []
        if not self.objects:  # first call: the reading is the state
            self.objects = [dict(o, age=0, _tid=self._tid()) for o in readings]
            self.touched = set()
            return [self._public(o) for o in self.objects], [], [{"event": "init", "n": len(readings)}]
        self._mark_touched(tcps, closed)
        used, done = set(), {}
        for t in self.objects:  # 1. pairing within pair_m
            cands = [(j, r) for j, r in enumerate(readings) if j not in used and _cat(r) == _cat(t) and _xy(r["pos"], t["pos"]) <= self.pair_m]
            if not cands:
                continue
            cands.sort(key=lambda jr: (jr[1].get("label") != t.get("label"), _xy(jr[1]["pos"], t["pos"])))
            j, r = cands[0]
            used.add(j)
            d = _xy(r["pos"], t["pos"])
            rc, tc = float(r.get("confidence", 0.5)), float(t.get("confidence", 0.5))
            touched = t["_tid"] in self.touched
            if d < self.jitter_m:  # 2. jitter
                k = dict(r, pos=list(t["pos"]), quat=list(t.get("quat") or r.get("quat") or [1, 0, 0, 0]), age=0,
                         confidence=max(rc, tc), _tid=t["_tid"])
            elif not touched and (rc < self.move_conf or rc < tc):  # 3. only touched objects move
                k = dict(t, source="prior", age=t.get("age", 0) + 1)
                events.append({"event": "implausible_move", "label": t.get("label"), "moved_mm": round(d * 1000, 1),
                               "reading_conf": rc, "tracked_conf": tc})
            else:
                k = dict(r, age=0, _tid=t["_tid"])
                if touched:
                    events.append({"event": "moved", "label": t.get("label"), "moved_mm": round(d * 1000, 1)})
            if k.get("source") != "prior" and float(k.get("confidence", 0)) >= self.move_conf:
                self.touched.discard(t["_tid"])
            done[t["_tid"]] = k
        left_t = [t for t in self.objects if t["_tid"] not in done]
        left_r = [(j, r) for j, r in enumerate(readings) if j not in used]
        for t in left_t:  # 4. relocated (conservation per category) / carried / last_seen
            rel = [(j, r) for j, r in left_r if _cat(r) == _cat(t) and float(r.get("confidence", 0.5)) >= self.relocate_conf]
            if rel:
                j, r = min(rel, key=lambda jr: _xy(jr[1]["pos"], t["pos"]))
                left_r.remove((j, r))
                done[t["_tid"]] = dict(r, age=0, _tid=t["_tid"])
                events.append({"event": "relocated", "label": t.get("label"), "moved_mm": round(_xy(r["pos"], t["pos"]) * 1000, 1)})
            elif t["_tid"] in self.touched:
                events.append({"event": "unseen_touched", "label": t.get("label")})
            else:
                done[t["_tid"]] = dict(t, source="prior", age=t.get("age", 0) + 1)
                events.append({"event": "carried", "label": t.get("label"), "age": done[t["_tid"]]["age"]})
        last_seen = [dict(self._public(t), source="prior") for t in left_t if t["_tid"] not in done]
        out = [done[t["_tid"]] for t in self.objects if t["_tid"] in done]
        for j, r in left_r:  # 5. new readings
            out.append(dict(r, age=0, new_reading=True, _tid=self._tid()))
            events.append({"event": "new", "label": r.get("label")})
        for k in out:  # 6. support plausibility
            if k.get("source") == "prior" or any(_xy(k["pos"], t) < 0.05 for t in closed) or not k.get("size"):
                continue
            bottom = k["pos"][2] - float(k["size"][2]) / 2
            if bottom > TABLE_Z + self.float_m:
                under = [o for o in out if o is not k and _xy(o["pos"], k["pos"]) < 0.04 and o["pos"][2] < k["pos"][2]]
                if not under:
                    k["confidence"] = round(0.5 * float(k.get("confidence", 0.5)), 3)
                    events.append({"event": "floating", "label": k.get("label")})
        self.objects = [dict(o) for o in out]
        return [self._public(o) for o in out], last_seen, events
