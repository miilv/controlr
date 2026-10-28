"""E2 (stream W): the post-release pose of placed objects, offline on stream C's replayed per-step sequences, scored with
stream M's grasp-time metric at every grasp / place event. 0 LLM calls.

    python scripts/perception/wrist/e2_release.py --seq-root SEQ_ROOT --out OUT.json [--episodes a,b]
        [--arms rt,oracle,read] [--delay 8] [--set k=v ...]

After every release (M's ``episode_events``: the first opening command after a grasp, step k) the released object's
belief is patched at step k + ``delay`` (the frame the read uses; the runtime's own ``post_place`` step when the report
logs one) and the patch rides in the belief until the report's next update of that label. Arms:
  rt      the runtime's own belief (M's ``BeliefTracker``: the released object stays where the TCP let it go, unless a
          logged update moves it);
  oracle  the released object's GT pose at the read step (dev only: the ceiling of a perfect release read);
  read    ``dojo.perception.release_read.release_read`` on the read step's frame (GT-free).
Every later grasp / place event is scored; the event table counts all events of the episodes.
Needs ``dojo.perception.seq_data`` (stream C's sequence reader) importable.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import gzip
import json
import os
import time

import numpy as np


def at_or_before(seq, s: int, back: int = 3):
    """The sequence's observation step nearest at or before ``s`` (within ``back``: strided sequences), else None."""
    for j in range(int(s), int(s) - back - 1, -1):
        if j in seq.by_step:
            return j
    return None


def at_or_after(seq, s: int, ahead: int = 3):
    for j in range(int(s), int(s) + ahead + 1):
        if j in seq.by_step:
            return j
    return at_or_before(seq, s)


def load_report(seq):
    d = os.path.dirname(seq.meta["actions"])
    hits = glob.glob(os.path.join(d, f"{seq.task}_*_{seq.layout_id}_report.json.gz"))
    return json.load(gzip.open(hits[0])) if hits else None


def load_actions(path):
    d = json.load(gzip.open(path, "rt"))
    return d["actions"] if isinstance(d, dict) else d


@contextlib.contextmanager
def patched_tracker(G, patcher):
    """M's ``episode_events`` with a BeliefTracker whose ``advance`` lets ``patcher(bt, upto)`` patch the belief after
    the report's own updates (a release read lands in the belief at its frame)."""
    Base = G.BeliefTracker

    class Patched(Base):
        def advance(self, upto: int) -> None:
            super().advance(upto)
            patcher(self, upto)

    G.BeliefTracker = Patched
    try:
        yield
    finally:
        G.BeliefTracker = Base


def releases(G, report, actions):
    """[(step k, arm, label)] of the episode's place events (the runtime's own attribution)."""
    return [(int(e.step), e.arm, e.target) for e, _ in G.episode_events(report, actions) if e.kind == "place" and e.target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="")
    ap.add_argument("--arms", default="rt,oracle")
    ap.add_argument("--require-done", default="", help="stream C's runs root: only episodes with <root>/<ep>/_done")
    ap.add_argument("--delay", type=int, default=8, help="read frame = release + delay when the report has no post_place")
    ap.add_argument("--set", action="append", default=[], help="release_read cfg field=value (JSON value)")
    a = ap.parse_args()

    from dojo.perception import grasp_metric as G
    from dojo.perception.seq_data import Sequence

    cfg = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    root = os.path.expanduser(a.seq_root)
    eps = sorted(x for x in os.listdir(root) if x != "smoke" and os.path.exists(os.path.join(root, x, "meta.json")))
    if a.episodes:
        eps = [x for x in eps if x in a.episodes.split(",")]
    if a.require_done:
        eps = [x for x in eps if os.path.exists(os.path.join(os.path.expanduser(a.require_done), x, "_done"))]
    arms = a.arms.split(",")
    rows, reads = [], []
    t_all = time.perf_counter()
    for ep in eps:
        seq = Sequence(os.path.join(root, ep))
        task = seq.task
        report, actions = load_report(seq), load_actions(seq.meta["actions"])
        if report is None:
            continue
        fp, groups = G.Footprints(), G.Groups(task)
        pp = {}                                          # (label, arm) -> post_place steps logged by the runtime
        for e in report.get("events") or []:
            if e.get("kind") == "post_place" and e.get("label"):
                pp.setdefault(e["label"], []).append(int(e["step"]))
        rel = releases(G, report, actions)
        head = None
        if "search" in arms:
            import zlib

            from dojo.runtime.perception_service import _Perception
            P = _Perception()
            P.set_scene(task)
            fold = zlib.crc32(str(seq.layout_id).encode()) % 2      # = stream C's replay_eval fold
            head = (P.bgs(fold), float(P.cfg(fold).fg_thr))
        last = max(seq.steps)
        read_at = {}
        for k, arm, lab in rel:
            s = next((s for s in sorted(pp.get(lab, [])) if k <= s <= k + 40), None)
            read_at[(k, lab)] = at_or_after(seq, min(s if s is not None else k + a.delay, last))
        per_arm = {}
        for arm_name in arms:
            patches = []                                  # (step, label, pose7 | None, info)
            for k, arm, lab in rel:
                j = read_at[(k, lab)]
                if arm_name == "oracle":                  # resolved at the read step: the GT object of the label's
                    patches.append((j, lab, None, {"src": "gt"}))   # group nearest to the released belief
                elif arm_name in ("read", "search"):
                    patches.append((j, lab, None, {"src": arm_name, "k": k, "arm": arm}))
            done = set()

            pre: dict = {}                               # label -> its pose before the last grasp (pre-grasp evidence)

            def patcher(bt, upto, patches=patches, done=done, seq=seq, ep=ep, groups=groups, fp=fp, pre=pre):
                for _, (lab_, _, p_) in bt.attached.items():
                    if p_ is not None:
                        pre[lab_] = list(p_[0]) + list(p_[1])
                for i, (j, lab, p7, info) in enumerate(patches):
                    if i in done or j > upto or lab not in bt.belief:
                        continue
                    done.add(i)
                    if info["src"] in ("read", "search"):
                        p7 = _read(bt, seq, j, lab, info, cfg, reads, ep, pre=pre.get(lab), head=head)
                    elif info["src"] == "gt":
                        p7 = _gt_pose(G, bt, seq, j, lab, groups, fp)
                    if p7 is not None:
                        bt.belief[lab].update(pos=list(p7[:3]), quat=list(p7[3:7]), source=f"release_{info['src']}")

            with patched_tracker(G, patcher):
                evs = G.episode_events(report, actions)
            out = {}
            for e, b in evs:
                jg = at_or_before(seq, int(e.step) - 1)
                if e.kind not in ("grasp", "place") or jg is None:
                    continue
                s = G.score_event(e, b, seq.gt(jg), task, footprints=fp, groups=groups)
                out[(int(e.step), e.arm)] = {"kind": e.kind, "step": int(e.step), "target": e.target,
                                             "reference": e.reference, "clean_r": s["clean_r"], "reasons": s["reasons"],
                                             "clean_required": s["clean_required"], "clean_target": s["clean_target"]}
            per_arm[arm_name] = out
        # a changed belief can change an event's attribution (a grasp's target, a close that becomes "close_other"):
        # events are matched by (step, arm); an event an arm does not score counts as not clean there
        miss = {"clean_r": {"5": False, "10": False, "15": False}, "reasons": {"5": ["not scored"], "10": ["not scored"],
                "15": ["not scored"]}, "clean_required": False, "clean_target": False}
        keys = sorted(set().union(*(set(v) for v in per_arm.values())))
        for key in keys:
            ev = next(per_arm[an][key] for an in arms if key in per_arm[an])
            rows.append({"episode": ep, "task": task, "arm": key[1],
                         **{k: ev[k] for k in ("kind", "step", "target", "reference")},
                         "scores": {an: {k: per_arm[an].get(key, miss)[k] for k in ("clean_r", "reasons", "clean_required",
                                                                                   "clean_target")} for an in arms}})
        tags = " ".join(f"{an}:{sum(x['clean_r']['10'] for x in per_arm[an].values())}/{len(per_arm[an])}" for an in arms)
        print(f"[e2] {ep} releases {len(rel)} {tags}", flush=True)
    res = {"arms": arms, "delay": a.delay, "cfg": cfg, "rows": rows, "reads": reads,
           "wall_s": round(time.perf_counter() - t_all, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(summary(rows, arms))


def _gt_pose(G, bt, seq, j, lab, groups, fp):
    """Dev only (the oracle arm): the GT pose of the object of ``lab``'s group nearest to the belief at step ``j``."""
    o = bt.snapshot(j).get(lab)
    if o is None or o.get("pos") is None:
        return None
    c = fp.centre(o.get("key"), o["pos"], o["quat"])
    same = [g for g in G.gt_rows_asset(seq.gt(j)) if groups.of(g["label"]) == groups.of(lab)]
    if not same:
        return None
    g = min(same, key=lambda g: float(np.linalg.norm(fp.centre(G.gt_key(g), g["pos"], g["quat"])[:2] - c[:2])))
    return list(g["pos"]) + list(g["quat"])


def _read(bt, seq, j, lab, info, cfg, reads, ep, pre=None, head=None):
    """The GT-free release read of ``lab`` on frame ``j`` (the belief there as its prior; arm "search": the wide search
    when it fails, with ``pre`` = the pre-grasp pose and ``head`` = (backgrounds, threshold)); logs it with its dev
    error."""
    from dojo.perception import grasp_metric as G
    from dojo.perception.release_read import release_read, release_search

    b = bt.snapshot(j)
    o = b.get(lab)
    if o is None or o.get("pos") is None:
        return None
    others = [(x["key"], list(x["pos"]) + list(x["quat"])) for l2, x in b.items()
              if l2 != lab and x.get("pos") is not None and x.get("key")]
    t0 = time.perf_counter()
    fr = seq.frame(j)
    try:
        r = release_read(fr, o["key"], list(o["pos"]) + list(o["quat"]), info["arm"], others=others, cfg=cfg)
        r["via"] = r.get("via") or "local"
        if (info["src"] == "search" and not r.get("ok") and head is not None
                and r.get("why") in ("fit_poor", "no_models", "no_views", "out_of_range", "one_far_view")):
            r2 = release_search(fr, o["key"], list(o["pos"]) + list(o["quat"]), info["arm"], others=others, pre_pose7=pre,
                                head_bgs=head[0], head_thr=head[1], cfg=cfg)
            r = {**r2, "local_why": r.get("why")}
    except Exception as ex:  # noqa: BLE001
        import traceback
        r = {"ok": False, "why": f"error {type(ex).__name__}: {ex} {traceback.format_exc()[-300:]}"[:600]}
    dt = time.perf_counter() - t0
    # dev only: the error before / after against the GT object of the label's group nearest to the prior
    fp = G.Footprints()
    grp = G.Groups(seq.task)
    c0 = fp.centre(o["key"], o["pos"], o["quat"])
    same = [x for x in G.gt_rows_asset(seq.gt(j)) if grp.of(x["label"]) == grp.of(lab)]
    g = min(same, key=lambda x: float(np.linalg.norm(fp.centre(G.gt_key(x), x["pos"], x["quat"])[:2] - c0[:2])),
            default=None)

    def err(p7):
        if g is None or p7 is None:
            return None
        c = fp.centre(o["key"], p7[:3], p7[3:7])
        gc = fp.centre(G.gt_key(g), g["pos"], g["quat"])
        return round(1000 * float(np.linalg.norm(np.asarray(c) - np.asarray(gc))), 1)

    rec = {"episode": ep, "label": lab, "step": j, "release": info["k"], "arm": info["arm"], "ok": r.get("ok"),
           "src": info["src"], "via": r.get("via"), "tried": r.get("tried"), "local_why": r.get("local_why"),
           "why": r.get("why"), "ious": r.get("ious"), "views": r.get("views"), "moved_mm": r.get("moved_mm"),
           "turned_deg": r.get("turned_deg"), "wall_s": round(dt, 3), "err_before_mm": err(list(o["pos"]) + list(o["quat"])),
           "err_after_mm": err(r.get("pose7")) if r.get("ok") else None,
           "err_fit_mm": err(r.get("fit_pose7") or r.get("pose7"))}
    reads.append(rec)
    print(f"   [{info['src']}] {ep} {lab} @{j} ok={r.get('ok')} via={r.get('via')} why={r.get('why')} err {rec['err_before_mm']} -> "
          f"{rec['err_fit_mm']} mm ious {r.get('ious')} {dt:.2f}s", flush=True)
    return r.get("pose7") if r.get("ok") else None


def summary(rows, arms) -> str:
    """Clean counts over the events every arm scored (a changed belief can change an event's attribution: those are
    counted apart)."""
    allr = rows
    rows = [r for r in rows if not any(r["scores"][an]["reasons"]["10"] == ["not scored"] for an in arms)]
    n = len(rows)
    lines = [f"events {n} scored in every arm (attribution changed: {len(allr) - n})"]
    for R in ("5", "10", "15"):
        cells = []
        for an in arms:
            c = sum(bool(r["scores"][an]["clean_r"][R]) for r in rows)
            fx = sum(bool(r["scores"][an]["clean_r"][R]) and not r["scores"][arms[0]]["clean_r"][R] for r in rows)
            br = sum(not r["scores"][an]["clean_r"][R] and r["scores"][arms[0]]["clean_r"][R] for r in rows)
            cells.append(f"{an} {c}/{n} (+{fx} -{br})")
        lines.append(f"R={R}cm: " + " | ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
