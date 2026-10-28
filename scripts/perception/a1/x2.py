"""A1 offline (stream W): the tick on stream C's per-step replays of recorded episodes, scored with stream M's grasp-time
metric, rt (the runtime's own belief) vs a1 (the same belief with the tick's writes), fixed / broken per the reviewer's
root-cause classes (/tmp/rev/a9.py: A not_held, B in-hand, C1-C4 touched, D misread, E clutter, F actor, G untouched but
moved). 0 LLM calls.

    python scripts/perception/a1/x2.py --seq-root SEQ --out OUT.json [--episodes a,b] [--require-done RUNS]
        [--weights W --classes C.json] [--det-cache F.pkl] [--mode apply|log]

Tick points, from the recorded run (its report + actions):
- close_empty: a held_look "not held" the runtime acted on, a grip_check "air" / "slip" (step, arm, label);
- release: every place (M's ``episode_events``), at the runtime's post_place step, else release + 8;
- lift: the first step after a grasp where the TCP rose ``lift_m`` (log-only identity check);
- hover: every pick / place refine the runtime logged (label = the refined object; corridor = above it);
- contact: the joints between ticks (sampled), swept against every believed box.
The belief at a tick is M's ``BeliefTracker`` (report events applied up to that step); the tick's writes go into it and
ride until the report's next update of that label. A DEV TOOL (smoke / debugging): the official X2 scorer is stream C's
harness, which calls the same ``dojo.runtime.tick.run`` (as this does). Never run it on the X2 test set.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import glob
import gzip
import json
import os
import time

import numpy as np


# ---------------------------------------------------------------------------------------------------- helpers
def load_report(seq):
    d = os.path.dirname(seq.meta["actions"])
    hits = glob.glob(os.path.join(d, f"{seq.task}_*_{seq.layout_id}_report.json.gz"))
    return json.load(gzip.open(hits[0])) if hits else None


def load_actions(path):
    d = json.load(gzip.open(path, "rt"))
    return d["actions"] if isinstance(d, dict) else d


def at_or_before(seq, s: int, back: int = 3):
    for j in range(int(s), int(s) - back - 1, -1):
        if j in seq.by_step:
            return j
    return None


class LocalSeg:
    """The twin-trained segmenter in this process (offline): ``seg_twin._SegModel`` with the client's interface."""

    def __init__(self, weights: str, classes: str | None = None, model_id: str = "seg_v2b"):
        from dojo.perception.seg_twin import _SegModel
        names = json.load(open(classes))["names"] if classes else None
        self.m = _SegModel(weights, names)
        self.model_id = model_id                         # the detection cache key: the same string online and offline

    def detect(self, img, conf: float = 0.25):
        from dojo.perception.seg_twin import Det
        return [Det(np.asarray(m, np.uint8), k, s, tuple(b)) for m, k, s, b in self.m.detect(img, conf)]


@contextlib.contextmanager
def patched_tracker(G, before_tick, ticks: list[int], hook):
    """M's ``episode_events`` with a BeliefTracker that, while advancing, stops at every tick step: report events up
    to it first, then ``hook(bt, step)``."""
    Base = G.BeliefTracker

    class AttLog(dict):
        """The tracker's ``attached``, logging every change as (action idx, arm, label | None, T_tcp_obj | None): the
        input of ``tick.held_path_from_log``, like ``RuntimeProvider``'s attach log online."""

        def __init__(self, bt):
            super().__init__()
            self.bt = bt

        def __setitem__(self, arm, v):
            super().__setitem__(arm, v)
            self.bt.att_log.append((self.bt.at, arm, v[0], np.asarray(v[1], float)))

        def pop(self, arm, *d):
            had = arm in self
            r = super().pop(arm, *d)
            if had:
                self.bt.att_log.append((self.bt.at, arm, None, None))
            return r

    class P(Base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._ticks = sorted(set(ticks))
            self.at, self.att_log = 0, []
            self.attached = AttLog(self)

        def advance(self, upto: int) -> None:
            while self._ticks and self._ticks[0] <= upto:
                s = self._ticks.pop(0)
                Base.advance(self, s)
                self.at = s + 1
                hook(self, s)
            Base.advance(self, upto)
            self.at = upto + 1                       # episode_events attaches / releases at step upto + 1

    G.BeliefTracker = P
    try:
        yield
    finally:
        G.BeliefTracker = Base


def tick_points(report, actions, evs, seq, lift_m: float = 0.03, delay: int = 8):
    """{step: [Trigger]} from the recorded run."""
    from dojo.runtime.tick import Trigger, triggers_from_events
    from dojo.perception import grasp_metric as G
    track = G.tcp_track(actions)
    pp = collections.defaultdict(list)
    out = collections.defaultdict(list)
    for e in report.get("events") or []:
        k = e.get("kind")
        if not isinstance(e.get("step"), (int, float)):
            continue
        s = int(e["step"])
        if k == "post_place" and e.get("label"):
            pp[e["label"]].append(s)
        elif k in ("held_look", "grip_check"):
            out[s].extend(t for t in triggers_from_events([e], [], [], set(), s) if t.label)   # the online rule
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


def q_path(actions, s0: int, s1: int, every: int = 3) -> dict:
    """The executed actions since the last tick, every 3rd: ``RuntimeProvider.a1_tick``'s ``executed[n_last::3]``."""
    out = {}
    for arm in ("left", "right"):
        key = f"{arm}_arm_joint_state"
        qs = [np.asarray(x[key], float).reshape(-1)[:6].tolist() for x in actions[s0:s1:every] if key in x]
        if qs:
            out[arm] = qs
    return out


# ---------------------------------------------------------------------------------------------------- one episode
def run_episode(ep: str, seq, report, actions, src, rd_cfg, tk_cfg, use_contact: bool = True) -> dict:
    from dojo.perception import grasp_metric as G
    from dojo.perception.redetect import Scene, redetect
    from dojo.perception.release_read import small_object
    from dojo.runtime.tick import TickState, fixture_labels, held_path_from_log, run

    task = seq.task
    fp, groups = G.Footprints(), G.Groups(task)
    evs0 = G.episode_events(report, actions)
    tps = tick_points(report, actions, evs0, seq)
    st = TickState()
    writes_log, disc_log, ticks_log = [], [], []
    track = G.tcp_track(actions)
    # grasps of the recorded run: (step, arm, label, the belief's pose at the close = the pre-grasp pose)
    grasps = [(int(e.step), e.arm, e.target, list(b[e.target]["pos"])[:3] + list(b[e.target]["quat"])[:4])
              for e, b in evs0 if e.kind == "grasp" and e.target and b.get(e.target, {}).get("pos") is not None]
    rep_events = [e for e in report.get("events") or [] if isinstance(e.get("step"), (int, float))]
    held_ev = [(int(e["step"]), e.get("label")) for e in report.get("events") or []
               if (e.get("kind") == "held_look" and e.get("held") is True)
               or (e.get("kind") == "grip_check" and e.get("verdict") == "held")]
    last_tick = [0]

    def hook(bt, s):
        trig = tps.get(s, [])
        j = at_or_before(seq, s)
        if j is None or not trig:
            return
        grasped = {lab for k, _, lab, _ in grasps if k <= s}
        snap = bt.snapshot(s)
        held = {arm: v[0] for arm, v in bt.attached.items()}
        fixture = fixture_labels([e for e in rep_events if e["step"] <= s], held_now=set(held.values()),
                                 grasps=[(lab, k) for k, _, lab, _ in grasps if k <= s])
        objs, meta = [], {}
        for lab in sorted(snap):
            o = snap[lab]
            if o.get("pos") is None or not o.get("key"):
                continue
            p7 = [float(v) for v in list(o["pos"])[:3] + list(o["quat"])[:4]]
            try:
                small = small_object(o["key"])
            except Exception:  # noqa: BLE001
                small = True
            objs.append({"label": lab, "key": o["key"], "pose7": p7, "small": small,
                         "fixture": lab in fixture,
                         "touched": lab in grasped})
        for t in trig:
            if t.kind in ("release", "slip") and t.label in snap and snap[t.label].get("pos") is not None:
                rel = int(t.data.get("release_step", s))
                g = max((x for x in grasps if x[2] == t.label and x[0] <= rel), default=None)
                g7, lift, r = None, 0.0, False
                if g is not None:
                    k, arm_g, _, g7 = g
                    z_close = float(np.asarray(track[arm_g][k])[2, 3])
                    lift = max(float(np.asarray(track[arm_g][x])[2, 3]) for x in range(k, rel + 1)) - z_close
                    r = any(lab == t.label and k <= st_ <= rel for st_, lab in held_ev)
                o = snap[t.label]
                meta[t.label] = {"drop7": list(o["pos"])[:3] + list(o["quat"])[:4], "grasp7": g7, "lift_m": lift,
                                 "rose": r}
        cor = None
        hov = [t for t in trig if t.kind == "hover" and t.label in snap and snap[t.label].get("pos") is not None]
        if hov:
            c = np.asarray(snap[hov[0].label]["pos"], float)
            cor = [c.tolist(), (c + [0, 0, 0.12]).tolist()]
        qp = q_path(actions, last_tick[0], s) if use_contact else {}
        sampled = {arm: [k for k in range(last_tick[0], min(s, len(actions)), 3)
                         if f"{arm}_arm_joint_state" in actions[k]] for arm in qp}
        last_tick[0] = s
        hp = held_path_from_log(bt.att_log, sampled, {lab: o.get("key") for lab, o in snap.items()}) \
            if use_contact else {}
        close_tcp = {t.label: np.asarray(track[t.arm][min(s, len(track[t.arm]) - 1)])[:3, 3].tolist()
                     for t in trig if t.kind == "close_empty" and t.arm in track}
        fr = seq.frame(j)
        frame = {"images": fr["images"], "state": {k: np.asarray(v, float) for k, v in fr["state"].items()}}
        t0 = time.perf_counter()
        try:                                             # the online lines (tick.run), the perception op in-process
            w, ds, resp = run(frame, objs, held, trig, meta, st, tk_cfg,
                              lambda f, r: redetect(f, r, src, rd_cfg, scene=Scene(f)), s,
                              touched=grasped & set(snap), q_path=qp, held_path=hp, corridor=cor, close_tcp=close_tcp)
        except Exception as ex:  # noqa: BLE001
            import traceback
            ticks_log.append({"step": s, "error": f"{type(ex).__name__}: {ex} {traceback.format_exc()[-400:]}"})
            return
        dt = time.perf_counter() - t0
        gts = G.gt_rows_asset(seq.gt(j))
        for lab, p7 in sorted(w.items()):
            if lab not in bt.belief:
                continue
            before = list(bt.belief[lab]["pos"])[:3] + list(bt.belief[lab]["quat"])[:4]
            bt.belief[lab].update(pos=list(p7[:3]), quat=list(p7[3:7]), source="a1")
            writes_log.append({"step": s, "label": lab, "before7": before, "after7": p7,
                               "err_before_mm": _gt_err(G, fp, groups, gts, lab, bt.belief[lab]["key"], before),
                               "err_after_mm": _gt_err(G, fp, groups, gts, lab, bt.belief[lab]["key"], p7)})
        disc_log.extend({**d.to_dict(), "step": s} for d in ds)
        if os.environ.get("X2_VERBOSE"):
            print(f"   [tick] {ep} @{s} {[(t.kind, t.label) for t in trig]} active={resp.get('active')} "
                  f"fits={resp.get('n_fits')} t={resp.get('time_s')} wall={dt:.2f}s writes={sorted(w)}", flush=True)
        ticks_log.append({"step": s, "frame": j, "triggers": [(t.kind, t.label, t.arm) for t in trig],
                          "active": resp.get("active"), "n_dets": resp.get("n_dets"), "n_fits": resp.get("n_fits"),
                          "time_s": resp.get("time_s"), "wall_s": round(dt, 3), "dets_hash": resp.get("dets_hash"),
                          "lift": resp.get("lift")})

    # the a1 pass (ticks applied), then the rt pass (none), scored the same way
    rows = {}
    for arm_name in ("a1", "rt"):
        with patched_tracker(G, None, sorted(tps) if arm_name == "a1" else [], hook):
            evs = G.episode_events(report, actions)
        out = {}
        for e, b in evs:
            jg = at_or_before(seq, int(e.step) - 1)
            if e.kind not in ("grasp", "place") or jg is None:
                continue
            s_ = G.score_event(e, b, seq.gt(jg), task, footprints=fp, groups=groups)
            out[(int(e.step), e.arm)] = s_
        rows[arm_name] = out
    return {"rows": rows, "writes": writes_log, "discrepancies": disc_log, "ticks": ticks_log}


def _gt_err(G, fp, groups, gts, lab, key, p7):
    same = [g for g in gts if groups.of(g["label"]) == groups.of(lab)]
    if not same or p7 is None:
        return None
    c = fp.centre(key, p7[:3], p7[3:7])
    return round(1000 * min(float(np.linalg.norm(fp.centre(G.gt_key(g), g["pos"], g["quat"]) - c)) for g in same), 1)


# ---------------------------------------------------------------------------------------------------- root causes
def first_fail(s: dict):
    bad = [q for q in s["required"] if not q["ok"]]
    if bad:
        q = bad[0]
        return ("req", q["role"], q["cls"], q["label"])
    nb = [n for n in s["neighbours"] if n["dist_m"] <= 0.10 and not n["found"]]
    if nb:
        return ("nb", "nf" if nb[0].get("xy_mm") is None else "off", bool(nb[0].get("clutter")), nb[0]["label"])
    p = [p for p in s["phantoms"] if p["dist_m"] <= 0.10]
    return ("ph", "", "", p[0]["label"] if p else None)


def classify(rt_rows: dict, seq, task: str) -> dict:
    """The reviewer's a9 classes for every unclean rt event (keyed like ``rt_rows``)."""
    out = {}
    touched: set = set()
    hist = collections.defaultdict(list)

    def gt_pos(lab, step):
        j = at_or_before(seq, step, back=10)
        if j is None:
            return None
        g = next((x for x in seq.gt(j) if x["label"] == lab), None)
        return None if g is None else np.asarray(g["pos"], float)

    for key in sorted(rt_rows):
        s = rt_rows[key]
        if not s["clean_r"]["10"]:
            f = first_fail(s)
            lab = f[3]
            if f[0] == "req" and f[2] == "not_held":
                c = "A not_held"
            elif f[0] == "req" and f[1] == "held":
                c = "B in-hand"
            elif task == "play_tic_tac_toe" and "opponent" in str(lab):
                c = "F actor (ttt)"
            elif f[0] == "nb" and f[2]:
                c = "E clutter"
            elif lab in touched:
                p = hist[lab][-1]
                if p["kind"] == "grasp":
                    c = "C1 retry after a missed close"
                else:
                    hq = [q for q in p["required"] if q["role"] == "held"]
                    hc = hq[0]["cls"] if hq else "?"
                    c = ("C2 never-held then placed" if hc == "not_held" else
                         "C3 in-hand error carried through release" if hc != "ok" else "C4 released ok, wrong later")
            else:
                g0, gk = gt_pos(lab, 0), gt_pos(lab, key[0] - 1)
                if g0 is not None and gk is not None and (np.linalg.norm((gk - g0)[:2]) > 0.015 or abs(gk[2] - g0[2]) > 0.015):
                    c = "G untouched but moved"
                elif g0 is None:
                    c = "D ? (no GT)"
                else:
                    c = "D frame-0/refine misread"
            out[key] = c
        if s["kind"] == "grasp":
            touched.add(s["target"])
        hist[s["target"]].append(s)
    return out


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="")
    ap.add_argument("--require-done", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--classes", default="")
    ap.add_argument("--det-cache", default="")
    ap.add_argument("--mode", default="apply")
    ap.add_argument("--no-contact", action="store_true")
    ap.add_argument("--model-id", default="seg_v2b")
    a = ap.parse_args()
    from dojo.perception import render
    assert render.gl() is not None, "A1 needs GL (z-buffered silhouettes); no EGL here"
    if os.environ.get("X2_DUMP_S"):                      # dev: where a slow tick spends its time
        import faulthandler
        import sys
        faulthandler.dump_traceback_later(int(os.environ["X2_DUMP_S"]), repeat=True, file=sys.stderr)

    from dojo.perception.redetect import DetSource, RDConfig
    from dojo.perception.seq_data import Sequence
    from dojo.runtime.tick import TickConfig

    seg = LocalSeg(a.weights, a.classes or None, a.model_id) if a.weights else None
    if a.det_cache and os.path.exists(a.det_cache):
        src = DetSource.load(a.det_cache, seg)
    else:
        src = DetSource(seg, seg.model_id if seg else "", seg.m.imgsz if seg else 0)
    rd_cfg, tk_cfg = RDConfig(), TickConfig(mode=a.mode)
    root = os.path.expanduser(a.seq_root)
    eps = sorted(x for x in os.listdir(root) if x != "smoke" and os.path.exists(os.path.join(root, x, "meta.json")))
    if a.episodes:
        eps = [x for x in eps if x in a.episodes.split(",")]
    if a.require_done:
        eps = [x for x in eps if os.path.exists(os.path.join(os.path.expanduser(a.require_done), x, "_done"))]
    res = {"rd_cfg": rd_cfg.to_dict(), "tk_cfg": tk_cfg.to_dict(), "episodes": {}}
    t_all = time.perf_counter()
    for ep in eps:
        seq = Sequence(os.path.join(root, ep))
        report, actions = load_report(seq), load_actions(seq.meta["actions"])
        if report is None:
            continue
        r = run_episode(ep, seq, report, actions, src, rd_cfg, tk_cfg, use_contact=not a.no_contact)
        cls = classify(r["rows"]["rt"], seq, seq.task)
        evr = []
        for key in sorted(set(r["rows"]["rt"]) | set(r["rows"]["a1"])):
            s0, s1 = r["rows"]["rt"].get(key), r["rows"]["a1"].get(key)
            evr.append({"step": key[0], "arm": key[1], "kind": (s0 or s1)["kind"], "target": (s0 or s1)["target"],
                        "rt": None if s0 is None else {"clean_r": s0["clean_r"], "reasons": s0["reasons"]["10"]},
                        "a1": None if s1 is None else {"clean_r": s1["clean_r"], "reasons": s1["reasons"]["10"]},
                        "cls": cls.get(key)})
        res["episodes"][ep] = {"task": seq.task, "events": evr, "writes": r["writes"],
                               "discrepancies": r["discrepancies"], "ticks": r["ticks"]}
        n_rt = sum(bool(x["rt"] and x["rt"]["clean_r"]["10"]) for x in evr)
        n_a1 = sum(bool(x["a1"] and x["a1"]["clean_r"]["10"]) for x in evr)
        print(f"[x2] {ep}: ticks {len(r['ticks'])} writes {len(r['writes'])} disc {len(r['discrepancies'])} "
              f"clean@10 rt {n_rt} a1 {n_a1} / {len(evr)}", flush=True)
        if a.det_cache:
            src.save(a.det_cache)
    res["wall_s"] = round(time.perf_counter() - t_all, 1)
    res["det_cache"] = {"hits": src.hits, "misses": src.misses}
    json.dump(res, open(a.out, "w"))
    print(summary(res))


def summary(res: dict) -> str:
    tot = collections.Counter()
    by_cls = collections.defaultdict(collections.Counter)
    kinds = collections.Counter()
    worse = better = 0
    for ep, e in res["episodes"].items():
        for x in e["events"]:
            if x["rt"] is None or x["a1"] is None:
                tot["attr_changed"] += 1
                continue
            a0, a1 = x["rt"]["clean_r"]["10"], x["a1"]["clean_r"]["10"]
            tot["n"] += 1
            tot["rt"] += a0
            tot["a1"] += a1
            if a1 and not a0:
                by_cls[x["cls"]]["fixed"] += 1
            if a0 and not a1:
                by_cls["(was clean)"]["broken"] += 1
            if not a0:
                by_cls[x["cls"]]["unclean"] += 1
        for d in e["discrepancies"]:
            kinds[(d["kind"], d["applied"])] += 1
        for w in e["writes"]:
            if w["err_before_mm"] is not None and w["err_after_mm"] is not None:
                worse += w["err_after_mm"] > w["err_before_mm"] + 3
                better += w["err_after_mm"] < w["err_before_mm"] - 3
    lines = [f"events scored in both: {tot['n']} (attribution changed {tot['attr_changed']}); clean@R10 rt {tot['rt']} "
             f"a1 {tot['a1']}",
             "per class (unclean in rt / fixed):"]
    for c in sorted(by_cls):
        lines.append(f"  {c}: unclean {by_cls[c]['unclean']} fixed {by_cls[c]['fixed']} broken {by_cls[c]['broken']}")
    lines.append(f"discrepancies (kind, applied): {dict(kinds)}")
    lines.append(f"writes: GT error better (>3 mm) {better}, worse (>3 mm) {worse}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
