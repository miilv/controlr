"""Stream C: offline replay of a per-step frame sequence (dojo.perception.seq_capture) through belief arms, scored at
every grasp / place with stream M's grasp-time metric (dojo.perception.grasp_metric: events, required objects,
neighbours within R of the approach corridor, phantoms; GT from the sequence's own dev-only rows at step k - 1).

    python scripts/perception/bgperc/replay_eval.py SEQ_DIR [--arms f0,rt,bg_rt,bg_f0] [--lag 15] [--bg JSON] [--out F]

Arms (all share the runtime's own event attribution and in-hand propagation, M's ``BeliefTracker``):
  rt     the recorded runtime's own belief at each event (M's reconstruction: planning record = frame 0 + scan +
         looks, then its refines / drops / post-place re-measures, held objects carried by the TCP);
  f0     rt, but every object the runtime has not moved yet (source record / pick or place refine) at B1's frame-0
         pose (PNG frame 0): what the runtime would believe with frame-0 perception only;
  bg_rt  rt + background perception: the background's belief moves (made from frames <= k - 1 - lag, still in the
         object's current epoch) override the base belief of objects nobody moved since;
  bg_f0  f0 + background perception (priors from f0);
  bgr_*  bg_* + periodic B1 re-reads of later frames (``reread_every``, default 40 steps; head view);
  bgc_*  bg_* + contact suspects only;
  bgd_*  bgc_* + head checks of every untouched object every 10 steps (perception's ``check``: an empty believed place
         is absence evidence) + a head re-read as soon as an object is in doubt (objects moved by an actor);
  bgh_*  bg_* + held checks while carrying (perception's B1 held_look on the holding wrist every 6 steps from 8 steps
         after the close; two confident "not held" in a row = the runtime's held_look drop, lag steps later) + contact
         suspects (a jaw inside an untouched object's footprint below its top: searched at its next view).
The background never sees GT. GT rows are the scorer's.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import gzip
import json
import os
import sys
import time
import zlib

import numpy as np

UNTOUCHED = ("record", "refine_pick", "refine_place")


def load_report(seq):
    d = os.path.dirname(seq.meta["actions"])
    hits = glob.glob(os.path.join(d, f"{seq.task}_*_{seq.layout_id}_report.json.gz"))
    return json.load(gzip.open(hits[0])) if hits else None


def apply_event(bt, ev, k):
    """The belief mutation ``grasp_metric.episode_events`` makes after scoring an event (attach / release)."""
    from dojo.perception.grasp_metric import pose_T
    if ev.kind == "grasp" and ev.target:
        att = (ev.meta or {}).get("attached", ev.target)       # grasp_metric >= #205: the runtime's own attach rule
        o = bt.snapshot(k - 1).get(att) if att else None
        if o is not None:
            for a2 in list(bt.attached):
                if bt.attached[a2][0] == att:
                    bt.attached.pop(a2)
            bt.attached[ev.arm] = (att, np.linalg.inv(bt.T(ev.arm, k)) @ pose_T(o["pos"], o["quat"]),
                                   (list(o["pos"]), list(o["quat"])))
    elif ev.kind == "place" and ev.arm in bt.attached:
        lab, p, q = bt.held_pose(ev.arm, k)
        bt.attached.pop(ev.arm)
        if lab in bt.belief:
            bt.belief[lab].update(pos=p, quat=q, source="released")


def base_belief(bt, step, arm_base, f0):
    b = bt.snapshot(step)
    if arm_base == "f0":
        held = {v[0] for v in bt.attached.values()}
        for lab, o in b.items():
            if lab in held or o.get("source", "record") not in UNTOUCHED:
                continue
            if lab in f0:
                o.update(pos=list(f0[lab]["pos"]), quat=list(f0[lab]["quat"]), key=f0[lab]["key"], source="f0")
            else:
                o["pos"] = None                      # B1 did not find it at frame 0
    return b


def relabel(f0: dict, ref: dict, groups) -> dict:
    """``f0``'s objects under ``ref``'s labels: Hungarian on 3-D distance within each interchangeable label group (two
    estimates of the same frame-0 scene may name a group's members differently)."""
    from scipy.optimize import linear_sum_assignment
    out, used = {}, set()
    gids = {groups.of(lab) for lab in list(f0) + list(ref)}
    for gi in gids:
        A = [lab for lab in f0 if groups.of(lab) == gi]
        B = [lab for lab in ref if groups.of(lab) == gi and ref[lab].get("pos") is not None]
        if not A:
            continue
        if not B:
            for lab in A:
                out[lab] = f0[lab]
            continue
        C = np.array([[float(np.linalg.norm(np.asarray(f0[x]["pos"]) - np.asarray(ref[y]["pos"]))) for y in B] for x in A])
        r, c = linear_sum_assignment(C)
        for i, j in zip(r, c):
            out[B[j]] = f0[A[i]]
            used.add(A[i])
        for lab in A:                                # more f0 objects than the reference has: keep their own label
            if lab not in used and lab not in out:
                out[lab] = f0[lab]
    return out


def make_reader(task: str, fold: int, views: str = "head"):
    """A B1 re-read of one later frame (``views``: head | all): label -> {"key", "pos", "quat", "conf"}."""
    from dojo.perception.bgperc import record_readings, reread_estimate
    from dojo.runtime.perception_service import _Perception
    P = _Perception()
    P.set_scene(task)

    def reader(fr):
        return record_readings(reread_estimate(fr["images"], fr["state"], task, fr.get("instruction") or "", P.bgs(fold),
                                               P.cfg(fold), views))
    return reader


def record_conf(rec: dict | None) -> dict:
    """label -> the estimate's confidence, lowered by how hidden the object was (1 - hidden_frac): a mostly hidden
    object ranks first for the background when a later view shows it."""
    from dojo.runtime.estimate import iter_records
    out = {}
    for _t, _c, r in iter_records(rec or {}):
        if r.get("label"):
            p = r.get("_perception") or {}
            out[r["label"]] = min(float(p.get("confidence", 0.5)), 1.0 - float(p.get("hidden_frac") or 0.0))
    return out


def m95_held_meta():
    """Stream M's not-held evidence for a place (``m95_score.held_drift`` / ``held_gap``: GT motion of the grasped object
    in the TCP frame, TCP distance outside its GT box), so the numbers match M's tables. None when M's driver is not
    importable (``BGPERC_M95`` = its scripts/perception dir)."""
    d = os.environ.get("BGPERC_M95", os.path.expanduser("~/controlr-m95/scripts/perception"))
    if not os.path.isdir(d):
        return None
    sys.path.insert(0, d)
    try:
        import m95_score
    except Exception:  # noqa: BLE001
        return None
    finally:
        sys.path.remove(d)
    return m95_score


def make_head_check(fold: int):
    """Perception's head check of the believed objects (GL z-buffer; the service op ``check``)."""
    from dojo.runtime.perception_service import _Perception, encode_frame
    P = _Perception()
    return lambda fr, objs: P.check(encode_frame(fr), objs, fold)


def make_held_fn():
    """Perception's held check on one frame, B1 only (``mode="b1"`` never calls an LLM; the LLM path is disabled here
    on top of that)."""
    from dojo.perception.agentic import j1
    from dojo.runtime.perception_service import _Perception, encode_frame

    def no_llm(*a, **k):
        raise RuntimeError("stream C replays make no LLM calls")
    j1._chat = no_llm
    P = _Perception()
    return lambda fr, arm: P.held_look(encode_frame(fr), arm, mode="b1")


def run(seq, actions, report, f0, arm, bg_cfg, lag, fp, groups, task, confs=None, reader=None, extra_lags=(),
        held_fn=None, head_fn=None):
    import bisect

    from dojo.perception.bgperc import BackgroundPerceiver
    from dojo.perception.grasp_metric import BeliefTracker, episode_events, score_event
    from dojo.runtime.bg import tcp_from_flange
    import inspect
    kw = {"footprints": fp} if "footprints" in inspect.signature(episode_events).parameters else {}
    evs = [(e, b) for e, b in episode_events(report, actions, **kw)]
    bt = BeliefTracker(report, actions)
    m95 = m95_held_meta()
    track = bt.track if m95 is not None else None
    frames = {str(r["step"]): {"objects": r["gt"]} for r in seq.rows} if m95 is not None else None
    last_grasp: dict = {}
    drops: list = []
    pending_suspect: dict = {}
    base = arm.split("_")[-1]
    use_bg = arm.startswith("bg")
    bp = BackgroundPerceiver(bg_cfg) if use_bg else None
    moves: list = []                                # (step, label, epoch, pose7)
    ev_at: dict = {}
    for e, _ in evs:
        ev_at.setdefault(e.step - 1, []).append(e)
    rows = []
    t0 = time.perf_counter()
    last_s = max(seq.steps)

    def sync(step):
        b = base_belief(bt, step, base, f0)
        if not bp.tracks:
            for lab, o in b.items():
                o["conf"] = (confs or {}).get(lab, 0.5)
        bp.sync(b, held={v[0] for v in bt.attached.values()})

    for s in range(0, last_s + 1):
        bt.advance(s)
        if use_bg and s in seq.by_step:
            sync(s)
            for lab, s0 in list(pending_suspect.items()):     # a dropped object: search for it where it was
                if s >= s0 and lab in bp.tracks and not bp.tracks[lab].held:
                    bp.tracks[lab].suspect = s
                    pending_suspect.pop(lab)
            st = seq.state(s)
            bp.contacts(s, [tcp_from_flange(st[f"{a}_ee_pose"]) for a in ("left", "right") if f"{a}_ee_pose" in st])
            nxt = next((e for e, _ in evs if e.step - 1 >= s and e.kind in ("grasp", "place")), None)
            req = []
            if nxt is not None:
                req = bp.required(nxt.point[:2], target=nxt.target if nxt.kind == "grasp" else None,
                                  reference=nxt.reference)
            fr = seq.frame(s)
            if held_fn is not None:
                for arm_, lab in bp.held_due(s, {a: v[0] for a, v in bt.attached.items()}):
                    t_h = time.perf_counter()
                    try:
                        hr = held_fn(fr, arm_)
                    except Exception as ex:  # noqa: BLE001
                        hr = {"held": None, "error": str(ex)[:120]}
                    bp.stats["held_s"] = bp.stats.get("held_s", 0.0) + time.perf_counter() - t_h
                    d = bp.held_result(s, arm_, lab, hr)
                    if d is not None:                          # the runtime's held_look drop rule, lag steps later
                        e_d = {"kind": "held_look", "step": s + lag, "arm": arm_, "label": lab, "dropped": True,
                               "bg": True}
                        j = bisect.bisect_right([x["step"] for x in bt.revs], s + lag, lo=bt._i)
                        bt.revs.insert(j, e_d)
                        drops.append(d)
                        pending_suspect[lab] = s + lag
            if head_fn is not None and bp.head_due(s):
                t_c = time.perf_counter()
                objs = [{"label": lab, "key": t.key, "pos": t.pose[:3].tolist(), "quat": t.pose[3:].tolist()}
                        for lab, t in bp.tracks.items() if not t.held]
                try:
                    bp.head_result(s, head_fn(fr, objs))
                except Exception as ex:  # noqa: BLE001
                    bp.events.append({"step": s, "head_error": str(ex)[:120]})
                bp.stats["head_s"] = bp.stats.get("head_s", 0.0) + time.perf_counter() - t_c
            chs = bp.step(fr, s, required=req)
            if reader is not None and bp.reread_due(s, fr):
                t_r = time.perf_counter()
                chs += bp.reread(s, reader(fr))
                bp.stats["reread_s"] = bp.stats.get("reread_s", 0.0) + time.perf_counter() - t_r
            for ch in chs:
                moves.append((s, ch["label"], bp.tracks[ch["label"]].epoch, np.asarray(ch["to"], float),
                              bp.tracks[ch["label"]].key))
        for e in ev_at.get(s, []):
            k = e.step
            if e.kind in ("grasp", "place"):
                def belief_at(cut):
                    b = base_belief(bt, k - 1, base, f0)
                    n = 0
                    if use_bg:
                        held = {v[0] for v in bt.attached.values()}
                        for lab, t in bp.tracks.items():
                            if lab in held or t.held or lab not in b:
                                continue
                            m = [mv for mv in moves if mv[1] == lab and mv[0] <= cut and mv[2] == t.epoch]
                            if m:
                                p7 = m[-1][3]
                                b[lab] = dict(b[lab], pos=p7[:3].tolist(), quat=p7[3:].tolist(), key=m[-1][4], source="bg")
                                n += 1
                    return {lab: o for lab, o in b.items() if o.get("pos") is not None}, n

                if m95 is not None and e.kind == "place" and e.arm in last_grasp:
                    dd = m95.held_drift(frames, track, e.arm, *last_grasp[e.arm], e.step)
                    if dd is not None:
                        e.meta["held_drift_mm"] = round(dd, 1)
                    f_ = m95.frame_at(frames, k - 1)
                    gap = m95.held_gap(f_, track, e.arm, last_grasp[e.arm][1], e.step, fp) if f_ else None
                    if gap is not None:
                        e.meta["held_gap_mm"] = round(gap, 1)
                bel, n_over = belief_at(k - 1 - lag)
                sc = score_event(e, bel, seq.gt(k - 1), task, footprints=fp, groups=groups)
                if e.kind == "grasp" and sc["required"][0].get("gt_label"):
                    last_grasp[e.arm] = (e.step, sc["required"][0]["gt_label"])
                sc["bg_overrides"] = n_over
                if use_bg:
                    req = bp.required(e.point[:2], target=e.target if e.kind == "grasp" else None, reference=e.reference)
                    sc["look_requests"] = bp.needs_look(req)
                    sc["by_lag"] = {}
                    for lg in extra_lags:                      # the same event with fresher / older evidence
                        bel2, n2 = belief_at(k - 1 - lg)
                        s2 = score_event(e, bel2, seq.gt(k - 1), task, footprints=fp, groups=groups)
                        sc["by_lag"][str(lg)] = {"clean_r": s2["clean_r"], "reasons": s2["reasons"], "overrides": n2,
                                                 "clean_target": s2["clean_target"]}
                rows.append(sc)
            apply_event(bt, e, k)
    out = {"events": rows, "wall_s": round(time.perf_counter() - t0, 1)}
    if use_bg:
        out["stats"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in bp.stats.items()}
        out["bg_events"] = bp.events[-600:]
        out["moves"] = [(m[0], m[1], m[2], [round(float(x), 4) for x in m[3]], m[4]) for m in moves]
        out["drops"] = drops
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--arms", default="f0,rt,bg_rt,bg_f0")
    ap.add_argument("--lag", type=int, default=15, help="background evidence must be this many steps older than the event")
    ap.add_argument("--extra-lags", default="0", help="background arms: also score each event at these lags")
    ap.add_argument("--bg", default="{}", help="BgConfig overrides (JSON)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from dojo.perception.bgperc import BgConfig
    from dojo.perception.grasp_metric import Footprints, Groups, flatten_record, summarize
    from dojo.perception.seq_capture import load_actions
    from dojo.perception.seq_data import Sequence
    from dojo.runtime.perception_service import _Perception, encode_frame
    seq = Sequence(a.seq)
    task = seq.task
    fold = zlib.crc32(str(seq.layout_id).encode()) % 2
    actions = load_actions(seq.meta["actions"])
    report = load_report(seq)
    if report is None:
        raise SystemExit(f"no recorded report for {seq.dir}")
    arms = a.arms.split(",")
    out = {"task": task, "layout_id": seq.layout_id, "fold": fold, "n_steps": len(seq.steps), "lag": a.lag, "arms": {}}
    f0 = {}
    fp, groups = Footprints(), Groups(task)
    recs = report.get("records") or {}
    from dojo.runtime.estimate import planning_record
    plan = flatten_record(planning_record(recs)[1])
    out["rt_record"] = planning_record(recs)[0]
    if any(x.endswith("f0") for x in arms) and recs.get("initial"):
        f0 = relabel(flatten_record(recs["initial"]), plan, groups)      # the run's own frame-0 B1 (lossless frame)
        out["f0_source"] = "report:initial"
        out["f0_not_found"] = [x.get("label") for x in recs["initial"].get("_not_found") or [] if isinstance(x, dict)]
    elif any(x.endswith("f0") for x in arms):
        P = _Perception()
        P.set_scene(task)
        t = time.perf_counter()
        rec0 = P.estimate([encode_frame(seq.frame(0, fmt="png"))], task, fold)
        out["f0_s"] = round(time.perf_counter() - t, 2)
        f0 = relabel(flatten_record(rec0), plan, groups)
        out["f0_source"] = "b1:frame0.png"
        out["f0_not_found"] = [x.get("label") for x in rec0.get("_not_found") or []]
    cfg = BgConfig(**json.loads(a.bg))
    out["bg_cfg"] = cfg.__dict__
    for arm in arms:
        conf = record_conf(planning_record(recs)[1] if arm.endswith("rt") else recs.get("initial"))
        # "bgr_*": background refines + periodic re-reads; "bgh_*": refines + held checks during carries + contact
        # suspects; "bg_*": refines only
        cfg_a = dataclasses.replace(cfg, reread_every=(cfg.reread_every or 40) if arm.startswith("bgr") else 0)
        if arm.startswith("bgh"):
            cfg_a = dataclasses.replace(cfg_a, held_every=cfg.held_every or 6, contact=True)
        if arm.startswith("bgc"):                    # contact suspects only (the held state is stream H's)
            cfg_a = dataclasses.replace(cfg_a, contact=True)
        if arm.startswith("bgd"):                    # + head checks every 10 steps, a head re-read when in doubt
            cfg_a = dataclasses.replace(cfg_a, contact=True, head_check_every=cfg.head_check_every or 10,
                                        reread_on_doubt=True)
        reader = make_reader(task, fold, cfg_a.reread_views) if (cfg_a.reread_every or cfg_a.reread_on_doubt) else None
        r = run(seq, actions, report, f0, arm, cfg_a, a.lag, fp, groups, task, confs=conf, reader=reader,
                extra_lags=[int(x) for x in a.extra_lags.split(",") if x != ""],
                held_fn=make_held_fn() if cfg_a.held_every else None,
                head_fn=make_head_check(fold) if cfg_a.head_check_every else None)
        r["summary"] = summarize(r["events"])
        out["arms"][arm] = r
        print(f"[bgperc] {task} {seq.layout_id} {arm}: {r['summary']} stats {r.get('stats')} wall {r['wall_s']}s",
              file=sys.stderr, flush=True)
    s = json.dumps(out, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    if a.out:
        open(a.out, "w").write(s)
    else:
        print(s)


if __name__ == "__main__":
    main()
