"""Stream C, X2: the official offline harness (lead, 2026-10-28): rt vs the A1 arms on genuine-5.1 per-step replays,
scored with stream M's grasp-time metric. 0 LLM calls.

    PYTHONPATH=<the recorded runs' code + A1> PYOPENGL_PLATFORM=egl <venv python> \\
        scripts/perception/bgperc/x2_eval.py SEQ_ROOT --arm null|oracle|a1 --out-dir OUT [--valid VALID.json]
        [--names a,b] [--floor FIDELITY.json]

Output in stream M's format (``m95_score.cmd_score``'s ``events.jsonl``: score_event + held metas + frame-0 + base +
attempt + root), one dir per arm: ``OUT/rt/events.jsonl``, ``OUT/<arm>/events.jsonl`` (+ "x2_arm", "changed",
"verdict"; "arm" stays the hand), ``OUT/<arm>/recomputes.jsonl``, ``OUT/<arm>/summary.json``.

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
    """DEV SANITY ONLY (reads GT): one step before every recorded pick / place refine and post-place re-measure, the
    written label's GT pose (the nearest GT object of its key within 15 cm). Exercises the causal chain: that write is
    then re-computed from the GT prior, and the grasp / place scored on the arm's belief."""
    name = "oracle"

    def __init__(self, seq):
        self.seq = seq
        self.labels: dict = {}

    def ticks(self, report, actions, revs) -> list[int]:
        for e in revs:
            w = WR.writer(e)
            if w in ("refine_pick", "refine_place", "post_place"):
                self.labels.setdefault(int(e["step"]) - 1, set()).add(e["label"])
        return sorted(self.labels)

    def tick(self, step, frame, belief, att, ctx) -> dict:
        from dojo.perception.grasp_metric import gt_key
        j = next((j for j in range(step, step - 4, -1) if j in self.seq.by_step), None)
        if j is None:
            return {}
        gts = self.seq.gt(j)
        out = {}
        for lab in sorted(self.labels.get(step, ())):
            if lab not in belief or lab in att.values():
                continue
            same = [g for g in gts if gt_key(g) == belief[lab].get("key") and g.get("pos") is not None]
            if same:
                g = min(same, key=lambda g: float(np.linalg.norm(np.subtract(g["pos"][:3], belief[lab]["pos"][:3]))))
                if float(np.linalg.norm(np.subtract(g["pos"][:3], belief[lab]["pos"][:3]))) <= 0.15:
                    out[lab] = list(g["pos"])[:3] + list(g["quat"])[:4]
        return out


def tick_points(report: dict, actions: list, evs: list, seq, lift_m: float = 0.03, delay: int = 8) -> dict:
    """{step: [Trigger]} from the recorded run: stream W's wiring (``scripts/perception/a1/x2.py`` ``tick_points``):
    grip_check / held_look by the online rule (``tick.triggers_from_events``), a hover at every pick / place refine,
    a release at the place's post-place step (else release + ``delay``), a lift at the first step the TCP rose
    ``lift_m`` after a grasp."""
    from collections import defaultdict

    from dojo.perception import grasp_metric as G
    from dojo.runtime.tick import Trigger, triggers_from_events
    track = G.tcp_track(actions)
    pp, out = defaultdict(list), defaultdict(list)
    for e in report.get("events") or []:
        k = e.get("kind")
        if not isinstance(e.get("step"), (int, float)):
            continue
        s = int(e["step"])
        if k == "post_place" and e.get("label"):
            pp[e["label"]].append(s)
        elif k in ("held_look", "grip_check"):
            out[s].extend(t for t in triggers_from_events([e], [], [], set(), s) if t.label)
        elif k == "refine" and e.get("for") in ("pick", "place") and e.get("label"):
            out[s].append(Trigger("hover", s, e.get("arm"), e["label"], {"for": e.get("for")}))
    last = max(seq.steps)
    for e, _ in evs:
        k = int(e.step)
        if e.kind == "place" and e.target:
            s = next((x for x in sorted(pp.get(e.target, [])) if k <= x <= k + 40), None)
            s = min(s if s is not None else k + delay, last)
            out[s].append(Trigger("release", s, e.arm, e.target, {"release_step": k}))
        elif e.kind == "grasp" and e.target:
            z0 = float(np.asarray(track[e.arm][k])[2, 3])
            up = next((j for j in range(k + 1, min(k + 80, len(track[e.arm]))) if
                       float(np.asarray(track[e.arm][j])[2, 3]) >= z0 + lift_m), None)
            if up is not None:
                out[up].append(Trigger("lift", up, e.arm, e.target, {"grasp_step": k}))
    return dict(out)


def at_or_before(seq, s: int, back: int = 3):
    return next((j for j in range(int(s), int(s) - back - 1, -1) if j in seq.by_step), None)


class StrictDets:
    """W's ``DetSource`` over the frozen cache only: a frame the cache lacks is an error, never an empty detection."""

    def __init__(self, src):
        self.src = src

    def detect(self, img):
        n = self.src.misses
        dets = self.src.detect(img)
        if self.src.misses != n:
            raise KeyError("x2_eval: a tick frame missing from the frozen detection cache (run --prepare-dets)")
        return dets


class A1Arm:
    """Stream W's A1 tick, imported unchanged (``dojo.runtime.tick.run`` + ``dojo.perception.redetect``), on the arm's
    own belief: the inputs are built the way ``RuntimeProvider.a1_tick`` builds them online (W's interface, 28 Oct)."""

    def __init__(self, name: str, seq, dets_pkl: str | None, rd_cfg=None, tk_cfg=None, schedule: str = "event"):
        from dojo.perception.redetect import DetSource, RDConfig
        from dojo.runtime.tick import TickConfig, TickState
        self.name, self.seq = name, seq
        self.src = (DetSource.load(dets_pkl) if dets_pkl and os.path.exists(dets_pkl)
                    else DetSource(None, "seg_v2b", 960, {}))
        self.dets = StrictDets(self.src)
        self.rd_cfg = rd_cfg or RDConfig()
        self.tk_cfg = tk_cfg or TickConfig(mode="apply")
        self.st = TickState()
        self.last = 0
        self.log: list = []
        self.schedule = schedule                        # event (J): W's triggers only; always (S): + every stored frame
        self.jaw_log: list = []                         # [(action idx, arm, label | None, T_tcp_obj | None)]
        self.write_log: list = []

    def setup(self, report, actions, revs, world, scored):
        self.report, self.actions, self.world = report, actions, world
        self.events = [e for e in report.get("events") or [] if isinstance(e.get("step"), (int, float))]
        self.tps = tick_points(report, actions, scored, self.seq)
        self.grasps = [(int(e.step), e.arm, e.target) for e, _b in scored if e.kind == "grasp" and e.target]
        self.grasp7 = {}                                # (label, grasp step) -> the arm's belief pose at the close

    def ticks(self, report, actions, revs) -> list[int]:
        ev = {s for s in self.tps if at_or_before(self.seq, s) is not None}
        return sorted(ev | set(self.seq.steps) if self.schedule == "always" else ev)

    def on_grasp(self, label, step, p7):
        self.grasp7[(label, int(step))] = p7

    def on_jaws(self, step, arm, label, T):
        self.jaw_log.append((int(step), arm, label, None if T is None else np.asarray(T, float).tolist()))

    def tick(self, step, frame, belief, att, ctx) -> dict:
        from dojo.perception.grasp_metric import pose_T
        from dojo.perception.redetect import Scene, redetect
        from dojo.perception.release_read import small_object
        from dojo.runtime.tick import fixture_labels, held_path_from_log, rose_evidence, run
        s, w = int(step), self.world
        trig = self.tps.get(s, [])
        j = at_or_before(self.seq, s)
        if j is None or (not trig and self.schedule != "always"):
            return {}
        if not trig:                                    # S's background tick: activates and seeds nothing (W)
            from dojo.runtime.tick import Trigger
            trig = [Trigger("periodic", s)]
        grasped = {lab for k, _a, lab in self.grasps if k <= s}
        fixture = fixture_labels([e for e in self.events if e["step"] <= s], held_now=set(att.values()),
                                 grasps=[(lab, k) for k, _a, lab in self.grasps if k <= s])
        objs, meta = [], {}
        for lab in sorted(belief):
            o = belief[lab]
            if o.get("pos") is None or not o.get("key"):
                continue
            try:
                small = bool(small_object(o["key"]))
            except Exception:  # noqa: BLE001 (W's x2: an unknown asset counts as small)
                small = True
            objs.append({"label": lab, "key": o["key"], "pose7": [float(v) for v in [*o["pos"][:3], *o["quat"][:4]]],
                         "small": small, "fixture": lab in fixture, "touched": lab in grasped})
        for t in trig:
            if t.kind in ("release", "slip") and t.label in belief:
                rel = int(t.data.get("release_step", s))
                g = max((x for x in self.grasps if x[2] == t.label and x[0] <= rel), default=None)
                g7, lift, rose = None, 0.0, False
                if g is not None:
                    k, arm_g, _ = g
                    g7 = self.grasp7.get((t.label, k))
                    z_close = float(w.T(arm_g, k)[2, 3])
                    lift = max(float(w.T(arm_g, x)[2, 3]) for x in range(k, rel + 1)) - z_close
                    rose = rose_evidence(self.events, t.label, k, rel)
                o = belief[t.label]
                meta[t.label] = {"drop7": [*o["pos"][:3], *o["quat"][:4]], "grasp7": g7, "lift_m": lift, "rose": rose}
        hov = [t for t in trig if t.kind == "hover" and t.label in belief]
        cor = None
        if hov:
            c = np.asarray(belief[hov[0].label]["pos"], float)
            cor = [c.tolist(), (c + [0, 0, 0.12]).tolist()]
        qp, sampled = {}, {}
        for arm in ("left", "right"):
            key = f"{arm}_arm_joint_state"
            idx = [i for i in range(self.last, s, 3) if i < len(self.actions) and key in self.actions[i]]
            if idx:
                qp[arm] = [np.asarray(self.actions[i][key], float).reshape(-1)[:6].tolist() for i in idx]
                sampled[arm] = idx
        self.last = s
        log = list(self.jaw_log)
        for arm, lab in att.items():                   # (the current carry, should the log have missed its close)
            if lab in belief and not any(e[1] == arm and e[2] == lab for e in log):
                T = np.linalg.inv(w.T(arm, s)) @ pose_T(belief[lab]["pos"], belief[lab]["quat"])
                log.append((0, arm, lab, T.tolist()))
        log.sort(key=lambda e: e[0])
        hp = held_path_from_log(log, sampled, {lab: o.get("key") for lab, o in belief.items()})
        close_tcp = {t.label: w.T(t.arm, s)[:3, 3].tolist() for t in trig if t.kind == "close_empty" and t.arm}
        fr = self.seq.frame(j)
        obs = {"images": fr["images"], "state": {k: np.asarray(v, float) for k, v in fr["state"].items()}}
        pend0 = {lab: int(v[0]) for lab, v in self.st.pending.items()}   # hypotheses awaiting a second tick
        coll0 = set(self.st.collapsed)
        t0 = time.perf_counter()
        try:
            writes, ds, resp = run(obs, objs, dict(att), trig, meta, self.st, self.tk_cfg,
                                   lambda f, r: redetect(f, r, self.dets, self.rd_cfg, scene=Scene(f)), s,
                                   touched=grasped & set(belief), q_path=qp, held_path=hp, corridor=cor,
                                   close_tcp=close_tcp)
        except Exception as ex:  # noqa: BLE001 (logged and counted; the episode goes on, as online)
            self.log.append({"step": s, "error": f"{type(ex).__name__}: {ex}"[:400]})
            return {}
        self.log.append({"step": s, "frame": j, "triggers": [(t.kind, t.label, t.arm) for t in trig],
                         "active": resp.get("active"), "dets_hash": resp.get("dets_hash"),
                         "writes": sorted(writes), "discrepancies": [d.to_dict() for d in ds],
                         # a write whose label had a hypothesis pending from an earlier tick: the gap (W: in S, two
                         # ticks < 10 steps apart see nearly the same images)
                         "pending_gap": {lab: s - pend0[lab] for lab in writes if lab in pend0},
                         # W's exact rule: the write needed the two-tick path (small / fixture-held / collapsed before
                         # this tick, and fewer than 2 views at IoU >= 0.5 showing >= 0.9 of it)
                         "two_tick": sorted(lab for lab in writes if any(
                             d.label == lab and int(((d.evidence or {}).get("cand") or {}).get("n50", 0)) < 2
                             for d in ds) and (lab in coll0 or any(o["label"] == lab and (o["small"] or o["fixture"])
                                                                   for o in objs))),
                         "wall_s": round(time.perf_counter() - t0, 3)})
        return writes


# ------------------------------------------------------------------------------------------- one episode
def gt_err(seq, step: int, lab: str, belief: dict, p7, fp, groups) -> float | None:
    """DEV-ONLY (the scorer's side, never a tick input): mm from ``lab``'s pose ``p7`` (its centre) to the nearest GT
    object of its group at ``step`` (W's x2 ``_gt_err``): judges a write helpful / harmful."""
    from dojo.perception.grasp_metric import gt_key, gt_rows_asset
    j = next((j for j in range(int(step), int(step) - 4, -1) if j in seq.by_step), None)
    if j is None or p7 is None:
        return None
    same = [g for g in gt_rows_asset(seq.gt(j)) if groups.of(g["label"]) == groups.of(lab)]
    if not same:
        return None
    c = fp.centre(belief[lab].get("key"), p7[:3], p7[3:7])
    return round(1000 * min(float(np.linalg.norm(fp.centre(gt_key(g), g["pos"], g["quat"]) - c)) for g in same), 1)


def rt_attached(rb, step: int) -> dict:
    """rt's jaws after ``step`` actions: the last probe snapshot at or before it (the jaws change only at snapshots)."""
    sn = next((x for x in reversed(rb.snaps) if x["step"] <= step), None)
    return dict((sn or {}).get("attached") or {})


def run_episode(P, seq, arm, floor: dict | None = None, log=print, allow_no_probe: bool = False) -> tuple[list, list, list]:
    from dojo.perception.grasp_metric import Footprints, Groups, attach_label, episode_events, grip_events
    report, actions = WR.load_src(seq)
    cfg = report.get("config") or {}
    revs = sorted((e for e in report.get("events") or [] if isinstance(e.get("step"), (int, float))),
                  key=lambda e: e["step"])
    rb = WR.RecordedBelief(report, actions, revs)
    if not rb.snaps and not allow_no_probe:
        raise ValueError(f"{seq.dir.name}: no dev_belief_probe (rt = the recorded belief needs it)")
    kin, fp = rb.kin, Footprints()
    world = ArmWorld(rb, rb.track, cfg, report)
    scored = [(ev, b) for ev, b in episode_events(report, actions, kin, footprints=fp) if ev.kind in ("grasp", "place")]
    if hasattr(arm, "setup"):
        arm.setup(report, actions, revs, world, scored)
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
    pairs, recomputes, tick_writes, write_log = [], [], [], []
    groups = Groups(seq.task)
    for step, _o, kind, x in tl:
        if kind == "write":
            ev = revs[x]
            if ev["label"] not in world.chain:
                continue
            t0 = time.time()
            p7, why, raw = rerun_write(P, seq, ev, x, world, report)
            rc = {"episode": seq.dir.name, "recompute": WR.writer(ev), "step": step, "label": ev["label"], "why": why,
                  "adopted": p7 is not None, "offline_blind": "not replayed" in why, **raw,
                  "time_s": round(time.time() - t0, 2)}
            recomputes.append(rc)
            log(rc)
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
            if ge["kind"] == "release" and hasattr(arm, "on_jaws"):
                arm.on_jaws(k, a, None, None)
            if ge["kind"] == "grasp":
                b0 = world.overlay(snap, k - 1)
                rt_lab = next((m.meta.get("attached") for m, _b in scored if m.kind == "grasp" and m.step == k
                               and m.arm == a), None)
                for m, _b in scored:
                    if m.kind == "grasp" and m.step == k and m.arm == a and m.target in b0 and hasattr(arm, "on_grasp"):
                        arm.on_grasp(m.target, k, [*b0[m.target]["pos"][:3], *b0[m.target]["quat"][:4]])
                att_arm = rt_lab
                if world.chain:                          # does the arm's belief attach what rt's did?
                    excl = {lab for arm_, lab in world.attached(rt_att).items() if arm_ != a}
                    att_arm = attach_label(b0, world.T(a, k)[:3, 3], fp, exclude=excl)
                    if att_arm is not None and (att_arm in world.chain or att_arm != rt_lab):
                        world.attach(att_arm, a, [*b0[att_arm]["pos"], *b0[att_arm]["quat"]], k, "attach")
                    if rt_lab is not None and rt_lab != att_arm and rt_lab not in world.chain and rt_lab in snap:
                        world.set_free(rt_lab, [*snap[rt_lab]["pos"], *snap[rt_lab]["quat"]], k, "not attached")
                if hasattr(arm, "on_jaws"):              # what this hand carries from action k on (W's held_path log)
                    from dojo.perception.grasp_metric import pose_T
                    T = (np.linalg.inv(world.T(a, k)) @ pose_T(b0[att_arm]["pos"], b0[att_arm]["quat"])
                         if att_arm is not None and att_arm in b0 else None)
                    arm.on_jaws(k, a, att_arm if T is not None else None, T)
        elif kind == "tick":
            i_next = next((j for j, e in enumerate(revs) if int(e["step"]) > step), len(revs))
            snap, _att, _ = rb.before(i_next, step=step)
            att = world.attached(rt_attached(rb, step))
            b = world.overlay(snap, step)
            writes = arm.tick(step, None, b, att, {"released": list(released_since)})
            released_since = []
            for lab, p7 in sorted(writes.items()):
                if lab in att.values():
                    continue
                d = float(np.linalg.norm(np.subtract(p7[:3], b[lab]["pos"][:3]))) * 1000 if lab in b else None
                e0 = gt_err(seq, step, lab, b, b[lab]["pos"][:3] + list(b[lab]["quat"][:4]), fp, groups) if lab in b else None
                e1 = gt_err(seq, step, lab, b, p7, fp, groups) if lab in b else None
                wl = {"episode": seq.dir.name, "tick": step, "write": lab, "moved_mm": None if d is None else round(d, 1),
                      "err_before_mm": e0, "err_after_mm": e1,
                      "by_gt": (None if e0 is None or e1 is None else "helpful" if e1 < e0 - 2.0 else
                                "harmful" if e1 > e0 + 2.0 else "neutral")}
                log(wl)
                write_log.append(wl)
                tick_writes.append(d)
                world.set_free(lab, p7, step, "a1")
        else:
            ev, b = scored[x]
            b_arm = world.overlay(b, ev.step - 1)
            changed = {}                                 # every object the arm's chain moved off rt's belief here
            for lab in sorted(world.chain):
                if lab not in b:
                    continue
                d = float(np.linalg.norm(np.subtract(b_arm[lab]["pos"], b[lab]["pos"]))) * 1000
                if d < 0.1:
                    continue
                wr = world.hist[lab][-1][1] if world.hist.get(lab) else None
                changed[lab] = {"mm": round(d, 2), "writer": wr, "role": ("target" if lab == ev.target else
                                "reference" if lab == ev.reference else "other"),
                                "floor_p95": ((floor or {}).get(wr) or {}).get("p95")}
            pairs.append((ev, b, b_arm, changed))
    rt_rows = m95_rows(seq, report, actions, [(ev, b) for ev, b, _ba, _c in pairs], track=rb.track, fp=fp)
    arm_rows = m95_rows(seq, report, actions, [(ev, ba) for ev, _b, ba, _c in pairs], track=rb.track, fp=fp)
    if hasattr(arm, "write_log"):
        arm.write_log = write_log
    for r_rt, r_arm, (_ev, _b, _ba, ch) in zip(rt_rows, arm_rows, pairs):
        r_rt.update(x2_arm="rt", changed={}, verdict="rt")
        r_arm.update(x2_arm=arm.name, changed=ch)
        r_arm["verdict"] = verdict({"rt": r_rt["clean_r"], "arm": r_arm["clean_r"], "changed": ch})
    return rt_rows, arm_rows, recomputes


_M95 = None


def m95():
    """Stream M's scorer (``scripts/perception/m95_score.py``) from this checkout: its row format and root causes."""
    global _M95
    if _M95 is None:
        spec = importlib.util.spec_from_file_location("m95_score", Path(__file__).resolve().parents[1] / "m95_score.py")
        _M95 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_M95)
    return _M95


def m95_rows(seq, report: dict, actions: list, pairs: list, track: dict, fp) -> list[dict]:
    """M's ``events.jsonl`` rows (``m95_score.cmd_score``) for (event, belief) pairs: score_event's output, the held
    metas, the frame-0 record's score, base fields, attempt and root cause. GT = the sequence's ``gt.jsonl``."""
    from collections import Counter, defaultdict

    from dojo.perception.grasp_metric import Groups, flatten_record, score_event
    M = m95()
    frames = {str(s): {"objects": seq.gt(s)} for s in seq.steps}
    task = seq.task
    g = Groups(task)
    src = os.path.dirname(seq.meta["actions"])
    run = seq.dir.name
    row = next((r for f in sorted(Path(src).glob("*_episodes.jsonl")) for r in (json.loads(x) for x in open(f) if x.strip())
                if r.get("layout_id") == seq.layout_id), {})
    base = {"run": run, "task": task, "layout_id": seq.layout_id, "success": row.get("success"), "score": row.get("score"),
            "category": row.get("category"), "preset": row.get("preset"), "renderer": row.get("renderer", "isaac60"),
            "head": (row.get("code") or {}).get("head"), "agent_unavailable": False, "family": M.family(run)}
    moving = any(e.get("kind") == "track_moving" for e in report.get("events") or [])
    recs = report.get("records") or {}
    rec0 = flatten_record(recs.get("initial")) if recs.get("initial") else None
    out, last_grasp = [], {}
    for ev, belief in pairs:
        f = M.frame_at(frames, ev.step - 1)
        if f is None:
            continue
        if ev.kind == "place" and ev.arm in last_grasp:
            d = M.held_drift(frames, track, ev.arm, *last_grasp[ev.arm], ev.step)
            if d is not None:
                ev.meta["held_drift_mm"] = round(d, 1)
            gap = M.held_gap(f, track, ev.arm, last_grasp[ev.arm][1], ev.step, fp)
            if gap is not None:
                ev.meta["held_gap_mm"] = round(gap, 1)
        s = score_event(ev, belief, f["objects"], task, footprints=fp, groups=g)
        if rec0 is not None:
            b0 = {**rec0, **{lab: o for lab, o in belief.items() if str(o.get("source", "")).endswith("held")}}
            s0 = score_event(ev, b0, f["objects"], task, footprints=fp, groups=g)
            s["frame0"] = {"clean_r": s0["clean_r"], "clean_required": s0["clean_required"],
                           "clean_target": s0["clean_target"], "target_cls": s0["required"][0]["cls"]}
        if ev.kind == "grasp" and s["required"][0].get("gt_label"):
            last_grasp[ev.arm] = (ev.step, s["required"][0]["gt_label"])
        s.update(base)
        s["moving"] = moving
        out.append(s)
    intended = M.INTENDED_PUSH.get(task)
    intervals = M.carried_intervals(frames, actions, track)
    held_at = {(s_["arm"], s_["step"]): (s_["meta"].get("attached") or s_["target"]) for s_ in out if s_["kind"] == "grasp"}
    seen, hist = Counter(), defaultdict(list)
    for s_ in sorted(out, key=lambda e: (e["step"], e["kind"] != "grasp")):
        seen[(s_["kind"], s_["target"])] += 1
        s_["attempt"] = seen[(s_["kind"], s_["target"])]
        if not s_["clean_r"]["10"]:
            x = M.failure_rows([s_])[0]
            s_["root"] = M.root_cause(s_, x, hist, frames, intervals, held_at, intended)
        hist[s_["target"]].append(s_)
    return out


def verdict(row: dict, R: str = "10") -> str:
    """rt vs arm at ``R`` cm: same | fixed | broken | within noise (every changed object below its writer's p95)."""
    a, b = bool(row["rt"][R]), bool(row["arm"][R])
    if a == b:
        return "same"
    big = [c for c in row["changed"].values() if c["floor_p95"] is None or c["mm"] > c["floor_p95"]]
    if not big:
        return "within noise"
    return "fixed" if b and not a else "broken"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--arm", default="null", choices=("null", "oracle", "a1"))
    ap.add_argument("--arm-name", default=None, help="the arm's name in the output (J / S)")
    ap.add_argument("--schedule", default="event", choices=("event", "always"),
                    help="a1: event = W's triggers only (J); always = + a tick on every stored frame (S)")
    ap.add_argument("--dets-dir", default=None, help="a1: the frozen detection cache, <dir>/<episode>.pkl")
    ap.add_argument("--prepare-dets", action="store_true",
                    help="build --dets-dir: v2b (SEG_WEIGHTS / SEG_CLASSES) on the 3 cameras of the frames, CPU")
    ap.add_argument("--dets-steps", default="all", choices=("all", "ticks"),
                    help="all: every stored frame (S's ticks, a superset of J's); ticks: J's tick frames only")
    ap.add_argument("--worker", default="0/1", help="i/n: this worker's share of the episodes (prepare-dets)")
    ap.add_argument("--out-dir", required=True, help="<out>/rt/events.jsonl, <out>/<arm>/events.jsonl (M's format)")
    ap.add_argument("--valid", default=None)
    ap.add_argument("--names", default="")
    ap.add_argument("--floor", default=None, help="write_rerun fidelity output: per-writer p95 (mm)")
    ap.add_argument("--allow-no-probe", action="store_true", help="dev only: runs without dev_belief_probe (no jaws)")
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
    if a.prepare_dets:
        import zlib
        i, k = (int(x) for x in a.worker.split("/"))
        names = [n for n in names if zlib.crc32(n.encode()) % k == i]      # stable as the valid set grows
        return prepare_dets(a.root, names, a.dets_dir, a.dets_steps)
    if a.arm == "a1" and a.dets_dir:                  # only episodes whose frozen detections are complete
        names = [n for n in names if os.path.exists(os.path.join(a.dets_dir, f"{n}.json"))]
    on = [n for n in names if "_gcoff_" not in n]       # the ON set (the primary baseline) first, tasks in turn
    by: dict = {}
    for n in on:
        by.setdefault(n.split("_")[1] if n.startswith("c4b_") else n, []).append(n)
    lists = [by[t] for t in sorted(by)]
    names = [x[i] for i in range(max(map(len, lists), default=0)) for x in lists if i < len(x)] + \
        [n for n in names if "_gcoff_" in n]
    P = None
    if a.arm != "null":
        from dojo.perception import render
        from dojo.runtime.perception_service import _Perception
        if render.gl() is None:
            raise SystemExit("x2_eval: no GL renderer (EGL): never the CPU fallback")
        P = _Perception()
    od = Path(a.out_dir)
    an = a.arm_name or a.arm
    (od / an).mkdir(parents=True, exist_ok=True)
    done_p = od / an / "done.txt"                     # resumable: episodes already scored for this arm
    done = set(done_p.read_text().split()) if done_p.exists() else set()
    meta = {"arm": an, "kind": a.arm, "schedule": a.schedule if a.arm == "a1" else None, "floor": a.floor,
            "code": os.environ.get("PYTHONPATH", "")}
    json.dump(meta, open(od / an / "meta.json", "w"), indent=1)

    def log(r):
        print(json.dumps(r), flush=True)
    for n in names:
        if n in done:
            continue
        seq = sd.Sequence(os.path.join(a.root, "seq", n))
        if a.arm == "a1":
            arm = A1Arm(an, seq, os.path.join(a.dets_dir, f"{n}.pkl") if a.dets_dir else None, schedule=a.schedule)
        else:
            arm = OracleArm(seq) if a.arm == "oracle" else NullArm()
        rt_rows, arm_rows, rcs = run_episode(P, seq, arm, floor, log, a.allow_no_probe)
        for fn, rows in (("rt_events.jsonl", rt_rows), ("events.jsonl", arm_rows), ("recomputes.jsonl", rcs),
                         ("ticks.jsonl", [{"episode": n, **t} for t in getattr(arm, "log", [])]),
                         ("writes.jsonl", getattr(arm, "write_log", []))):
            with open(od / an / fn, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        with open(done_p, "a") as fh:
            fh.write(n + "\n")
    summ = summary_from_dir(od, an)
    print(json.dumps({k: summ[k] for k in ("arm", "episodes", "online_cost", "writes_by_gt")}, indent=1))


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in open(p) if x.strip()] if p.exists() else []


def summary_from_dir(od: Path, an: str) -> dict:
    """``OUT/<arm>/summary.json`` from the arm's files: rt vs arm per R (all, without put_bottles, per task), the
    re-computed writes, the A1 writes judged by GT, the online cost (ticks, writes, two-tick confirmations from
    frames < 10 steps apart, tick wall time per episode); ``OUT/rt/events.jsonl`` = the rt rows of the scored
    episodes (one copy, for M's ``m95_score.py x2``)."""
    from collections import Counter
    d = od / an
    meta = json.load(open(d / "meta.json")) if (d / "meta.json").exists() else {}
    all_rt, all_arm = _jsonl(d / "rt_events.jsonl"), _jsonl(d / "events.jsonl")
    all_rc, all_wr, tk = _jsonl(d / "recomputes.jsonl"), _jsonl(d / "writes.jsonl"), _jsonl(d / "ticks.jsonl")
    eps = sorted({r["run"] for r in all_rt} | {t["episode"] for t in tk})
    cost = None
    if tk or meta.get("kind") == "a1":
        n = max(1, len(eps))
        cost = {"episodes": len(eps), "ticks": len(tk), "ticks_per_episode": round(len(tk) / n, 1),
                "errors": sum("error" in t for t in tk),
                "dets_cache_misses": sum("frozen detection cache" in str(t.get("error")) for t in tk),
                "writes": sum(len(t.get("writes") or []) for t in tk),
                "periodic_ticks": sum(any(x[0] == "periodic" for x in t.get("triggers") or []) for t in tk),
                "writes_with_pending_lt10_steps": sum(1 for t in tk for g in (t.get("pending_gap") or {}).values()
                                                      if g < 10),
                "two_tick_writes": sum(len(t.get("two_tick") or []) for t in tk),
                "two_tick_writes_lt10_steps": sum(1 for t in tk for lab in t.get("two_tick") or []
                                                  if (t.get("pending_gap") or {}).get(lab, 99) < 10),
                "tick_wall_s_per_episode": round(sum(float(t.get("wall_s") or 0) for t in tk) / n, 1)}

    def wsum(ws):
        return {"n": len(ws), **Counter(w.get("by_gt") for w in ws)}
    nb = [i for i, r in enumerate(all_rt) if r["task"] != "put_bottles_into_dustbin"]
    summ = {**meta, "episodes": len(eps), "online_cost": cost, "writes_by_gt": wsum(all_wr),
            "all": summarize(all_rt, all_arm, all_rc),
            "without_put_bottles": summarize([all_rt[i] for i in nb], [all_arm[i] for i in nb],
                                             [c for c in all_rc if "_bottles_" not in c["episode"]]),
            "by_task": {}}
    for t in sorted({r["task"] for r in all_rt}):
        ii = [i for i, r in enumerate(all_rt) if r["task"] == t]
        runs = {all_rt[i]["run"] for i in ii}
        sm = summarize([all_rt[i] for i in ii], [all_arm[i] for i in ii], [c for c in all_rc if c["episode"] in runs])
        sm.pop("recomputed", None)
        sm["writes_by_gt"] = wsum([w for w in all_wr if w["episode"] in runs])
        summ["by_task"][t] = sm
    json.dump(summ, open(d / "summary.json", "w"), indent=1)
    (od / "rt").mkdir(exist_ok=True)
    rt_all = {}
    for f in sorted(od.glob("*/rt_events.jsonl")):
        for r in _jsonl(f):
            rt_all.setdefault((r["run"], r["step"], r["kind"], r["arm"]), r)
    with open(od / "rt" / "events.jsonl", "w") as fh:
        for r in rt_all.values():
            fh.write(json.dumps(r) + "\n")
    return summ


def prepare_dets(root: str, names: list[str], out: str, steps: str = "all") -> None:
    """The frozen detection cache of the A1 arms: v2b on all three cameras of every stored frame (``steps`` "all": S's
    ticks, a superset of J's) or of the tick frames only ("ticks": W's ``tick_points``), in W's ``DetSource`` (model
    id "seg_v2b", imgsz 960, conf 0.25), one pickle per episode and a sidecar ``<episode>.json`` with its sha1 (the
    cache hash ``valid.json`` records; ``index.json`` gathers them). Run where ultralytics is (CPU); workers on
    disjoint ``--names`` never share a file."""
    import hashlib

    from dojo.perception.grasp_metric import Footprints, episode_events
    from dojo.perception.redetect import DetSource
    from dojo.perception.seg_twin import Det, _SegModel
    from dojo.skills.kinematics import X5Kin

    class _Seg:
        def __init__(self):
            cl = json.load(open(os.environ["SEG_CLASSES"]))
            self.m = _SegModel(os.environ["SEG_WEIGHTS"], cl if isinstance(cl, list) else cl.get("names"))

        def detect(self, img, conf: float = 0.25):
            return [Det(np.asarray(m, np.uint8), k, sc, tuple(b)) for m, k, sc, b in self.m.detect(img, conf)]
    seg, sd, kin, fp = _Seg(), WR._seq_data(), X5Kin(), Footprints()
    os.makedirs(out, exist_ok=True)
    w_sha = hashlib.sha1(open(os.environ["SEG_WEIGHTS"], "rb").read()).hexdigest()
    for n in names:
        dst, side = os.path.join(out, f"{n}.pkl"), os.path.join(out, f"{n}.json")
        if os.path.exists(side):
            continue
        seq = sd.Sequence(os.path.join(root, "seq", n))
        if steps == "all":
            js = list(seq.steps)
        else:
            report, actions = WR.load_src(seq)
            scored = [(e, b) for e, b in episode_events(report, actions, kin, footprints=fp)
                      if e.kind in ("grasp", "place")]
            js = sorted({j for s in tick_points(report, actions, scored, seq) for j in [at_or_before(seq, s)]
                         if j is not None})
        src = DetSource(seg, "seg_v2b", 960)
        t0 = time.time()
        for j in js:
            for _cam, img in seq.frame(j)["images"].items():
                src.detect(img)
        src.save(dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        json.dump({"file": os.path.basename(dst), "sha1": hashlib.sha1(open(dst, "rb").read()).hexdigest(),
                   "frames": len(js), "steps": steps, "model_id": "seg_v2b", "imgsz": 960, "conf": 0.25,
                   "weights_sha1": w_sha}, open(side, "w"), indent=1)
        print(f"[dets] {n}: {len(js)} frames, {time.time() - t0:.0f} s", flush=True)
    idx = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(Path(out).glob("*.json"))
           if f.name != "index.json"}
    json.dump(idx, open(os.path.join(out, "index.json") + f".{os.getpid()}", "w"), indent=1)
    os.replace(os.path.join(out, "index.json") + f".{os.getpid()}", os.path.join(out, "index.json"))


def summarize(rt_rows: list[dict], arm_rows: list[dict], rcs: list[dict]) -> dict:
    """Per R: rt / arm clean counts and the verdicts; the re-computed writes: adopted, and the ones X2 cannot see
    (they need a re-hover / second view / head-view confirm online), per writer."""
    from collections import Counter
    out = {"events": len(rt_rows), "episodes": len({r["run"] for r in rt_rows})}
    for R in ("5", "10", "15"):
        out[f"R{R}"] = {"rt_clean": sum(bool(r["clean_r"][R]) for r in rt_rows),
                        "arm_clean": sum(bool(r["clean_r"][R]) for r in arm_rows),
                        **Counter(verdict({"rt": x["clean_r"], "arm": y["clean_r"], "changed": y["changed"]}, R)
                                  for x, y in zip(rt_rows, arm_rows))}
    out["recomputed"] = {"n": len(rcs), "adopted": sum(c["adopted"] for c in rcs),
                         "offline_blind": sum(c["offline_blind"] for c in rcs),
                         "by_writer": {w: {"n": sum(c["recompute"] == w for c in rcs),
                                           "adopted": sum(c["adopted"] for c in rcs if c["recompute"] == w),
                                           "offline_blind": sum(c["offline_blind"] for c in rcs if c["recompute"] == w)}
                                       for w in sorted({c["recompute"] for c in rcs})},
                         "why": dict(Counter(c["why"] for c in rcs))}
    return out


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    os._exit(0)                                     # skip interpreter teardown: the EGL context segfaults there
