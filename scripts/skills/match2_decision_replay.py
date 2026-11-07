"""Dev-only: match_and_pick's DECISION (first object, its identity, the belt speed, the match it would pick) replayed on
recorded camera episodes, under two switch settings. CPU only, no model calls, no simulator, no perception service.

    python scripts/skills/match2_decision_replay.py INITIAL.json READS.json [--set belt_ref=tracked ...] > rows.jsonl

``INITIAL.json``: {episode: the report's ``records.initial`` (the frame-0 belief record)}; ``READS.json``: {episode:
{"tr": [[step, found]], "gt": [{label, cat, x, y}]}} with ``found`` the report's logged ``track_read`` entries
([label, confidence, category, [x, y]]: the first 60 reads of an episode). The robot stays at home and the pick is a stub
that records what it was asked to pick: grasp outcomes are not modelled. GT (the layout's authored poses moved along the
belt from the reference's first read) is used for SCORING only, never by the replayed policy. Episodes whose decision
needs reads after the 60th end undecided ("replay ends").
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np

TASK = "match_and_pick_from_conveyor"
IDX = {"puppet": 0, "game_machine": 0, "phone": 0, "camera": 2, "ice_cream": 0, "garage": 16, "bottle": 47, "bag": 0,
       "plate": 2, "headset": 7}
V_BELT = 0.004          # scoring only: the authored belt speed of the GT model


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        from dojo.skills.world import AssetInfo
        if typ == "Dynamic":
            return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.array([-3.96, -0.6, 0.0]),
                             bbox_max=np.array([0.0, 0.58, 0.92]))
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.04), bbox_max=np.full(3, 0.04))


class _Ex:
    def __init__(self, sk):
        self.sk = sk
        self.steps_left = None

    def hold(self, n):
        from dojo.skills.executor import BudgetExceeded
        self.sk.step += int(n)
        if self.sk.step > self.sk.last_read:
            raise BudgetExceeded(f"replay ends at step {self.sk.last_read}")


class _Skills:
    def __init__(self, prov, b, obs, reads, attrs):
        self.provider, self.b, self.obs, self.reads = prov, b, obs, reads
        prov.track_reader = lambda o: reads[o["steps_used"]]
        self.step, self.read_upto, self.last_read = 0, -1, max(reads)
        self.ex, self.held, self.logs = _Ex(self), {}, []
        for k, v in attrs.items():
            setattr(self, k, v)

    def refresh(self):
        from dojo.skills.world import WorldState
        for s in sorted(r for r in self.reads if self.read_upto < r <= self.step):
            self.b.obs, self.b.fresh = dict(self.obs, steps_used=s), True
            self.provider._track(self.b.obs)
        self.read_upto = self.step
        self.provider._predict(self.step)
        return WorldState(TASK, "", copy.deepcopy(self.provider.belief.objects), {}, {}, step=self.step, step_lim=700,
                          notes=self.provider.notes)

    def log(self, *a, **k):
        self.logs.append(str(a[0]) if a else "")

    def home(self, **k):
        pass


def _records(tr):
    out = {}
    for step, found in tr:
        rec: dict = {"Rigid": {}}
        for lab, conf, cat, (x, y) in found:
            rec["Rigid"].setdefault(cat, []).append({"category": cat, "category_idx": IDX.get(cat, 0), "label": lab,
                                                     "default_pos": [x, y, 0.8], "default_ori": [1.0, 0.0, 0.0, 0.0],
                                                     "_perception": {"confidence": conf}})
        out[int(step)] = rec
    return out


def _gt_offset(e) -> float:
    t0 = [g for g in e["gt"] if g["label"] == "target0"][0]
    for s, f in e["tr"][:6]:
        for _lab, _c, _cat, xy in f:
            if abs(xy[1] - t0["y"]) < 0.05 and xy[0] > t0["x"]:
                return (xy[0] - t0["x"]) / V_BELT - s
    return 51.0


def replay(initial: dict, e: dict, attrs: dict) -> dict:
    import dojo.skills.interactive as I
    from dojo.runtime.belief import RuntimeProvider
    from dojo.runtime.bridge import Bridge
    from dojo.runtime.estimate import objects_from_record
    from dojo.skills.env import KinematicEnv
    env = KinematicEnv([], step_lim=700, instruction="")
    b = Bridge()
    obs = {"images": {}, "state": env.proprio(), "instruction": "", "steps_used": 0, "step_lim": 700}
    b.obs, b.fresh = obs, True
    prov = RuntimeProvider(b, TASK, "", "perception", _Assets())
    prov.belt_speed = attrs.get("belt_speed", "off")
    start = {k: v for k, v in initial.items() if k in ("Rigid", "Dynamic")}
    prov.init_objects(objects_from_record(start, _Assets(), 0))
    reads = _records(e["tr"])
    sk = _Skills(prov, b, obs, reads, dict(attrs))
    picked = []

    def pick(sk_, label, v, **k):
        w = sk_.refresh()
        picked.append((sk_.step, label, w.obj(label).pos[:2].copy(), np.asarray(v, float).copy()))
        return SimpleNamespace(ok=True, status="ok", note="", arm="left")
    orig = I.pick_from_belt
    I.pick_from_belt = pick
    try:
        labs = [lab for typ in ("Rigid",) for lst in start.get(typ, {}).values() for lab in [x["label"] for x in lst]]
        r = I.match_and_pick(sk, labs or ["target0"], conveyor="conveyor")
    finally:
        I.pick_from_belt = orig
    st = sk.refresh().notes.get("belt", {})
    gt = {g["label"]: g for g in e["gt"]}
    off = _gt_offset(e)
    out = {"result": r, "first_key": st.get("first_key"), "ref_n": st.get("ref_n"),
           "gt_target": f"Rigid/{gt['target0']['cat']}/{IDX.get(gt['target0']['cat'], 0):05d}"}
    out["ref_ok"] = out["first_key"] == out["gt_target"]
    if picked:
        s, label, xy, v = picked[0]
        near = min(gt.values(), key=lambda g: float(np.hypot(g["x"] + V_BELT * (s + off) - xy[0], g["y"] - xy[1])))
        d = float(np.hypot(near["x"] + V_BELT * (s + off) - xy[0], near["y"] - xy[1]))
        out.update(pick_step=int(s), pick_label=label, pick_near=near["label"], pick_d_mm=round(d * 1000, 1),
                   v_mm=[round(float(x) * 1000, 2) for x in v])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("initial")
    ap.add_argument("reads")
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE skill / provider switch")
    a = ap.parse_args(argv)
    attrs = dict(kv.split("=", 1) for kv in a.set)
    init = json.load(open(a.initial))
    reads = json.load(open(a.reads))
    for name in sorted(reads):
        if name not in init:
            continue
        try:
            row = replay(init[name], reads[name], attrs)
        except Exception as ex:  # noqa: BLE001
            row = {"error": f"{type(ex).__name__}: {ex}"}
        print(json.dumps({"name": name, "attrs": attrs, **row}, default=str), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
