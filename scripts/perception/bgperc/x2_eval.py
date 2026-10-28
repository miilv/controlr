"""Stream C, X2: the official offline harness (lead, 2026-10-28): rt vs the A1 arms on genuine-5.1 per-step replays,
scored with stream M's grasp-time metric. 0 LLM calls.

    PYTHONPATH=<the recorded runs' code + A1> PYOPENGL_PLATFORM=egl <venv python> \\
        scripts/perception/bgperc/x2_eval.py SEQ_ROOT --arm null|oracle|a1 --out OUT.jsonl [--valid VALID.json]
        [--names a,b] [--floor FIDELITY.json]

Protocol (reviewer's design, adopted by the lead; journal 2026-10-28-dojo-bgperc §7):
- rt = the RECORDED belief: M's ``episode_events`` (the runtime's ``dev_belief_probe`` at every grasp / place).
- an arm = the recorded belief + that arm's writes, computed on the stored frames. The labels an arm wrote follow
  their own causal chain (``ArmWorld``):
  - an arm write sets the label's pose;
  - every later recorded write of that label (pick / place / held refine, post-place) is re-computed on the stored
    frame of its step from the arm's pose (``write_rerun``), and adopted by the runtime's main gates (``accept``;
    no re-hovers offline: the actions are fixed);
  - a close attaches what the runtime's attach rule takes in the arm's belief (M's ``attach_label``): it rides the
    TCP until the release leaves it where the jaws let it go. An attach that differs from rt's makes the other label
    diverge too;
  - untouched labels replay as recorded.
- every arm reads the same stored frames; agentic records replay as recorded (0 LLM).
- per scored event: rt and arm clean at R 5 / 10 / 15, the arm's belief change of the event's target / reference
  against rt (mm) and the writer that last set it; ``--floor`` (``write_rerun`` fidelity output) adds that writer's
  p95, and "fixed" / "broken" count only above it (else "within noise").

Arms: ``null`` (writes nothing: must reproduce rt exactly), ``oracle`` (dev sanity: at every release, the released
label's GT pose; never a result), ``a1`` (W's tick, imported unchanged: the adapter lands with W's interface).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WR = _load("write_rerun")


# ------------------------------------------------------------------------------------------- the runtime's gates
def accept(w: str, r: dict, prior7, cfg: dict, others: dict) -> tuple[list | None, str]:
    """The pose the runtime would have adopted from a re-computed write ``r`` (``_Perception.refine``'s result) of
    writer ``w``, None when it keeps the prior; with the reason. The main gates of ``policy._refine_hook`` /
    ``_held_refine`` / ``KinematicBelief.post_place``; re-hovers, second views and the post-place head-view confirm
    cannot happen offline (the actions are fixed): a write that needs one is not adopted."""
    if not r.get("ok") or r.get("pos") is None:
        return None, "not ok"
    pos, quat = [float(v) for v in r["pos"]], [float(v) for v in r["quat"]]
    shift, iou, turned = float(r.get("shift_mm") or 0), float(r.get("iou") or 0), float(r.get("turned_deg") or 0)
    read_yaw = bool(r.get("yaw_source"))
    if w == "post_place":
        if shift > float(cfg.get("post_place_confirm_mm", 20.0)):
            return None, "post-place shift needs the head-view confirm (not replayed)"
        dz = abs(pos[2] - float(prior7[2])) * 1000
        if shift >= 2.0 or dz >= 2.0 or r.get("z_dropped_mm") is not None:
            return pos + quat, "post-place moved"
        return None, "post-place below 2 mm"
    if w == "refine_held":
        dc = float(np.linalg.norm(np.subtract(pos, prior7[:3]))) * 1000
        if dc > float(cfg.get("held_refine_max_mm", 10.0)) and iou < float(cfg.get("held_refine_min_iou", 0.85)):
            return None, "in-hand shift not clear enough"
        if cfg.get("held_flip_confirm") and turned > float(cfg.get("refine_max_turn_deg", 90.0)):
            return None, "in-hand flip pending a second view"
        return pos + quat, "held refine"
    if turned > float(cfg.get("refine_turn_check_deg", 45.0)) and iou < float(cfg.get("refine_turn_min_iou", 0.9)) \
            and not read_yaw:
        return None, "turn at low iou: prior kept"
    if shift > float(cfg.get("big_shift_mm", 30.0)):
        nb = float(cfg.get("neighbour_mm", 25.0)) / 1000
        if any(float(np.linalg.norm(np.subtract(pos[:2], o["pos"][:2]))) < nb for o in others.values()):
            return None, "on a neighbour"
        if iou < float(cfg.get("big_shift_min_iou", 0.7)):
            return None, "large correction needs a confirming re-hover (not replayed)"
    if w == "refine_place" and cfg.get("place_confirm"):
        if shift > float(cfg.get("place_cap_mm", 12.0)) or float(r.get("visible_frac") or 0) < float(
                cfg.get("place_min_vis", 0.9)):
            return None, "place correction needs a confirming view (not replayed)"
    if turned > float(cfg.get("refine_max_turn_deg", 90.0)) and not read_yaw:
        return pos + list(prior7[3:7]), "orientation kept"
    return pos + quat, "refined"


# ------------------------------------------------------------------------------------------- the arm's world
class ArmWorld:
    """The arm's belief: the recorded one (``RecordedBelief`` between events, M's probe beliefs at scored events) with
    the diverged labels' own chain. ``chain[label]`` = {"free": pose7} | {"arm": a, "T": T_tcp_obj}, ``hist[label]``
    = [(step, writer, pose7)]."""

    def __init__(self, rb, track, cfg: dict, report: dict):
        self.rb, self.track, self.cfg, self.report = rb, track, cfg, report
        self.chain: dict = {}
        self.hist: dict = {}

    def T(self, arm, step):
        tr = self.track[arm]
        return np.asarray(tr[min(max(int(step), 0), len(tr) - 1)], float)

    def pose(self, lab, step) -> list | None:
        c = self.chain.get(lab)
        if c is None:
            return None
        if "free" in c:
            return list(c["free"])
        from dojo.perception.grasp_metric import _R_quat
        Tw = self.T(c["arm"], step) @ c["T"]
        return Tw[:3, 3].tolist() + _R_quat(Tw[:3, :3]).tolist()

    def set_free(self, lab, p7, step, writer):
        self.chain[lab] = {"free": [float(v) for v in p7]}
        self.hist.setdefault(lab, []).append((int(step), writer, [round(float(v), 5) for v in p7]))

    def attach(self, lab, arm, p7, step, writer="attach"):
        from dojo.perception.grasp_metric import pose_T
        self.chain[lab] = {"arm": arm, "T": np.linalg.inv(self.T(arm, step)) @ pose_T(p7[:3], p7[3:7])}
        self.hist.setdefault(lab, []).append((int(step), writer, [round(float(v), 5) for v in p7]))

    def overlay(self, belief: dict, step: int) -> dict:
        """``belief`` (label -> {key, pos, quat, ...}) with the diverged labels at their chain pose after ``step``
        actions."""
        out = dict(belief)
        for lab in self.chain:
            p7 = self.pose(lab, step)
            if lab in out and p7 is not None:
                out[lab] = {**out[lab], "pos": p7[:3], "quat": p7[3:7], "source": "arm"}
        return out

    def attached(self, rt_att: dict) -> dict:
        """The arm world's jaws: chain attachments, else rt's unless rt's label diverged."""
        out = {a: lab for a, lab in rt_att.items() if lab not in self.chain}
        for lab, c in self.chain.items():
            if "arm" in c:
                out[c["arm"]] = lab
        return out


def rerun_write(P, seq, ev, i, world: ArmWorld, report: dict) -> tuple[list | None, str, dict]:
    """A recorded write of a diverged label re-computed on the stored frame of its step from the arm's pose."""
    w, lab, s = WR.writer(ev), ev["label"], int(ev["step"])
    if s not in seq.by_step:
        return None, "no stored frame", {}
    snap, att, _src = world.rb.before(i)
    snap = world.overlay(snap, s)
    att = world.attached(att)
    prior = world.pose(lab, s)
    if w == "post_place" and ev.get("from") is not None:
        pass                                            # the arm's released pose, not the logged one
    req = WR.inputs(ev, snap, att, world.cfg, report, prior7=prior)
    r = WR.rerun(P, seq.frame(s), req)
    others = {x["label"]: x for x in req["objects"][1:]}
    p7, why = accept(w, r, prior, world.cfg, others)
    return p7, why, {"ok": r.get("ok"), "iou": r.get("iou"), "shift_mm": r.get("shift_mm")}


# ------------------------------------------------------------------------------------------- arms
class NullArm:
    name = "null"

    def ticks(self, report, actions, revs) -> list[int]:
        return []

    def tick(self, step, frame, belief, att, ctx) -> dict:
        return {}


class OracleArm:
    """DEV SANITY ONLY (reads GT): at every release, the released label's GT pose. Exercises the causal chain."""
    name = "oracle"

    def __init__(self, seq):
        self.seq = seq

    def ticks(self, report, actions, revs) -> list[int]:
        from dojo.perception.grasp_metric import grip_events
        steps = sorted(self.seq.by_step)
        out = []
        for ge in grip_events(actions):
            if ge["kind"] == "release":
                nxt = next((s for s in steps if s >= ge["step"] + 8), None)
                if nxt is not None:
                    out.append(nxt)
        return sorted(set(out))

    def tick(self, step, frame, belief, att, ctx) -> dict:
        from dojo.perception.grasp_metric import gt_key
        gts = self.seq.gt(step)
        out = {}
        for lab in ctx.get("released") or []:
            if lab not in belief:
                continue
            same = [g for g in gts if gt_key(g) == belief[lab].get("key") and g.get("pos") is not None]
            if same:
                g = min(same, key=lambda g: float(np.linalg.norm(np.subtract(g["pos"][:3], belief[lab]["pos"][:3]))))
                out[lab] = list(g["pos"])[:3] + list(g["quat"])[:4]
        return out


# ------------------------------------------------------------------------------------------- one episode
def rt_attached(rb, step: int) -> dict:
    """rt's jaws after ``step`` actions: the last probe snapshot at or before it (the jaws change only at snapshots)."""
    sn = next((x for x in reversed(rb.snaps) if x["step"] <= step), None)
    return dict((sn or {}).get("attached") or {})


def gt_at(seq, k: int, back: int = 3):
    for j in range(k, max(-1, k - back - 1), -1):
        if j in seq.by_step:
            return seq.gt(j)
    return None


def run_episode(P, seq, arm, floor: dict | None = None, log=print) -> list[dict]:
    from dojo.perception.grasp_metric import (Footprints, Groups, attach_label, episode_events, grip_events,
                                              score_event)
    report, actions = WR.load_src(seq)
    cfg = report.get("config") or {}
    revs = sorted((e for e in report.get("events") or [] if isinstance(e.get("step"), (int, float))),
                  key=lambda e: e["step"])
    rb = WR.RecordedBelief(report, actions, revs)
    kin, fp, task = rb.kin, Footprints(), seq.task
    groups = Groups(task)
    world = ArmWorld(rb, rb.track, cfg, report)
    scored = [(ev, b) for ev, b in episode_events(report, actions, kin, footprints=fp) if ev.kind in ("grasp", "place")]
    # the timeline: recorded writes (index i), gripper changes, the arm's ticks, the scored events (at k - 1)
    tl = []
    for i, e in enumerate(revs):
        if WR.writer(e) is not None:
            tl.append((int(e["step"]), 1, "write", i))
    for ge in grip_events(actions):
        tl.append((int(ge["step"]), 0, "grip", ge))
    for s in arm.ticks(report, actions, revs):
        tl.append((int(s), 2, "tick", s))
    for j, (ev, _b) in enumerate(scored):
        tl.append((int(ev.step) - 1, 3, "score", j))
    tl.sort(key=lambda x: (x[0], x[1]))
    released_since: list = []
    rows = []
    for step, _o, kind, x in tl:
        if kind == "write":
            ev = revs[x]
            if ev["label"] not in world.chain:
                continue
            t0 = time.time()
            p7, why, raw = rerun_write(P, seq, ev, x, world, report)
            log({"episode": seq.dir.name, "recompute": WR.writer(ev), "step": step, "label": ev["label"], "why": why,
                 "adopted": p7 is not None, **raw, "time_s": round(time.time() - t0, 2)})
            if p7 is not None:
                if WR.writer(ev) == "refine_held" and "arm" in world.chain[ev["label"]]:
                    world.attach(ev["label"], world.chain[ev["label"]]["arm"], p7, step, WR.writer(ev))
                else:
                    world.set_free(ev["label"], p7, step, WR.writer(ev))
        elif kind == "grip":
            ge = x
            a, k = ge["arm"], ge["step"]
            i_next = next((j for j, e in enumerate(revs) if int(e["step"]) >= k), len(revs))
            snap, _att, _ = rb.before(i_next, step=k - 1)
            rt_att = rt_attached(rb, k - 1)                # the jaws before this change
            if ge["kind"] == "release":
                lab = next((lab for lab, c in world.chain.items() if c.get("arm") == a), None)
                if lab is not None:
                    world.set_free(lab, world.pose(lab, k), k, "release")
                held = world.attached(rt_att).get(a)
                if held is not None:
                    released_since.append(held)
            elif world.chain:                            # a close: does the arm's belief attach what rt's did?
                b = world.overlay(snap, k - 1)
                excl = {lab for arm_, lab in world.attached(rt_att).items() if arm_ != a}
                att_arm = attach_label(b, world.T(a, k)[:3, 3], fp, exclude=excl)
                rt_lab = next((m.meta.get("attached") for m, _b in scored if m.kind == "grasp" and m.step == k
                               and m.arm == a), None)
                if att_arm is not None and (att_arm in world.chain or att_arm != rt_lab):
                    world.attach(att_arm, a, [*b[att_arm]["pos"], *b[att_arm]["quat"]], k, "attach")
                if rt_lab is not None and rt_lab != att_arm and rt_lab not in world.chain and rt_lab in snap:
                    world.set_free(rt_lab, [*snap[rt_lab]["pos"], *snap[rt_lab]["quat"]], k, "not attached")
        elif kind == "tick":
            i_next = next((j for j, e in enumerate(revs) if int(e["step"]) > step), len(revs))
            snap, _att, _ = rb.before(i_next, step=step)
            att = world.attached(rt_attached(rb, step))
            b = world.overlay(snap, step)
            writes = arm.tick(step, seq.frame(step) if step in seq.by_step else None, b, att,
                              {"released": list(released_since)})
            released_since = []
            for lab, p7 in sorted(writes.items()):
                if lab in att.values():
                    continue
                world.set_free(lab, p7, step, "a1")
        else:
            ev, b = scored[x]
            gts = gt_at(seq, ev.step - 1)
            if gts is None:
                continue
            b_arm = world.overlay(b, ev.step - 1)
            s_rt = score_event(ev, b, gts, task, footprints=fp, groups=groups)
            s_arm = score_event(ev, b_arm, gts, task, footprints=fp, groups=groups) if b_arm != b else s_rt
            row = {"episode": seq.dir.name, "task": task, "kind": ev.kind, "step": ev.step, "arm_hand": ev.arm,
                   "target": ev.target, "reference": ev.reference,
                   "rt": {R: bool(v) for R, v in s_rt["clean_r"].items()},
                   "arm": {R: bool(v) for R, v in s_arm["clean_r"].items()}, "changed": {}}
            for lab in (ev.target, ev.reference):
                if lab and lab in world.chain and lab in b:
                    d = float(np.linalg.norm(np.subtract(b_arm[lab]["pos"], b[lab]["pos"]))) * 1000
                    wr = world.hist[lab][-1][1] if world.hist.get(lab) else None
                    row["changed"][lab] = {"mm": round(d, 2), "writer": wr,
                                           "floor_p95": ((floor or {}).get(wr) or {}).get("p95")}
            rows.append(row)
    return rows


def verdict(row: dict, R: str = "10") -> str:
    a, b = row["rt"][R], row["arm"][R]
    if a == b:
        return "same"
    big = [c for c in row["changed"].values() if c["floor_p95"] is None or c["mm"] > c["floor_p95"]]
    if not big:
        return "within noise"
    return "fixed" if b and not a else "broken"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--arm", default="null", choices=("null", "oracle"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--valid", default=None)
    ap.add_argument("--names", default="")
    ap.add_argument("--floor", default=None, help="write_rerun fidelity output: per-writer p95 (mm)")
    a = ap.parse_args()
    sd = WR._seq_data()
    floor = None
    if a.floor:
        summ = json.load(open(a.floor))["summary"]
        floor = {w: {"p95": (v.get("mm") or {}).get("p95")} for w, v in summ.items()}
        floor["a1"] = floor.get("refine_pick")              # an arm's own fit: the closest measured writer
        floor["attach"] = floor["release"] = floor["not attached"] = floor.get("refine_pick")
    names = sorted(os.listdir(os.path.join(a.root, "seq")))
    if a.valid:
        ok = {r["name"] for r in json.load(open(a.valid)) if r.get("valid")}
        names = [n for n in names if n in ok]
    if a.names:
        names = [n for n in names if n in set(a.names.split(","))]
    P = None
    if a.arm != "null":
        from dojo.perception import render
        from dojo.runtime.perception_service import _Perception
        if render.gl() is None:
            raise SystemExit("x2_eval: no GL renderer (EGL): never the CPU fallback")
        P = _Perception()
    with open(a.out, "w") as fh:
        def log(r):
            print(json.dumps(r), flush=True)
        for n in names:
            seq = sd.Sequence(os.path.join(a.root, "seq", n))
            arm = OracleArm(seq) if a.arm == "oracle" else NullArm()
            for row in run_episode(P, seq, arm, floor, log):
                row["verdict"] = verdict(row)
                fh.write(json.dumps(row) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
