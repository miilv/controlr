"""E1 (stream W, research R1): the grasp-time wrist check offline on stream C's replayed per-step sequences, scored with
stream M's grasp-time metric. 0 LLM calls.

    python scripts/perception/wrist/e1.py --events EVENTS.json --seq-root SEQ_ROOT --out OUT.json
        [--episodes a,b] [--sam] [--embed dinov2_b] [--hover runtime|track] [--set k=v ...]

For every grasp / place event (M's ``episode_events``: the runtime's own belief at frame k - 1, rebuilt from the
recorded report + actions; GT = the sequence's dev-only rows at k - 1):
- the hover frame: the runtime's own pick / place refine step of that arm before the event (C's ``hover_frame``), else
  the top of the final descent on the commanded TCP track (``--hover track`` uses the track for every event);
- the wrist check (``dojo.perception.wrist_check``) on that frame with the belief at the event (held objects carried to
  the hover by the TCP), then the event is scored with the belief before and after, per ablation arm:
  rt (before) | exist (phantoms dropped) | pose (+ neighbour / reference fits) | reloc (+ relocations) | id (+ identity /
  flip, with --embed).
Needs ``dojo.perception.seq_data`` (stream C's sequence reader; copy it next to ``grasp_metric`` in the deploy until
C's PR lands). The 28 Oct numbers (journal 2026-10-28-dojo-wrist-check) used C's copy of ``grasp_metric`` at 9446e4e.
"""
from __future__ import annotations

import argparse
import copy
import glob
import gzip
import json
import os
import time

import numpy as np

ARMS = ("rt", "exist", "pose", "reloc", "id")


def load_report(seq):
    d = os.path.dirname(seq.meta["actions"])
    hits = glob.glob(os.path.join(d, f"{seq.task}_*_{seq.layout_id}_report.json.gz"))
    return json.load(gzip.open(hits[0])) if hits else None


def load_actions(path):
    d = json.load(gzip.open(path, "rt"))
    return d["actions"] if isinstance(d, dict) else d


def track_hover(track, k: int, point, max_xy: float = 0.03) -> int:
    """The top of the final descent before action ``k``: walking back from k - 1 while the TCP stays within ``max_xy``
    of the event point (xy) and does not go down (going back), the step with the highest TCP."""
    j, best = k - 1, k - 1
    zmax = -1e9
    p = np.asarray(point, float)
    while j >= 0:
        t = np.asarray(track[j], float)[:3, 3]
        if np.hypot(t[0] - p[0], t[1] - p[1]) > max_xy or t[2] < zmax - 0.005:
            break
        if t[2] >= zmax:
            zmax, best = t[2], j
        j -= 1
    return best


def arm_belief(before: dict, out: dict, arm: str) -> dict:
    """The belief after the check under one ablation arm, from the check's per-object record."""
    b = copy.deepcopy(before)
    if arm == "rt":
        return b
    objs, found = out["objects"], out["found"]
    reloc = {f["label"]: f for f in found if f.get("applied")} if arm in ("reloc", "id") else {}
    if arm in ("pose", "reloc", "id"):
        for lab, r in objs.items():
            if r.get("applied") and r.get("pose7") and lab in b:
                b[lab].update(pos=r["pose7"][:3], quat=r["pose7"][3:7], source="wrist_check")
    for lab, f in reloc.items():
        if lab in b:
            b[lab].update(pos=f["pose7"][:3], quat=f["pose7"][3:7], source="wrist_relocate")
    for lab, r in objs.items():
        if r.get("dropped") and lab not in reloc:
            b.pop(lab, None)
    if arm == "id":
        for lab, r in objs.items():
            idr = r.get("identity") or {}
            if lab in b and idr.get("change_key"):
                b[lab]["key"] = idr["change_key"]
            if lab in b and idr.get("flip"):
                b[lab].update(pos=idr["flip_pose"][:3], quat=idr["flip_pose"][3:7])
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--seq-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="")
    ap.add_argument("--sam", action="store_true", help="SAM 2.1-S in process (GPU)")
    ap.add_argument("--embed", default="", help="DINOv2 model for identity / flip (GPU), e.g. dinov2_b")
    ap.add_argument("--hover", default="runtime", choices=("runtime", "track"))
    ap.add_argument("--head", action="store_true", help="objects the wrist does not see: head-view existence")
    ap.add_argument("--set", action="append", default=[], help="CheckConfig field=value (JSON value)")
    ap.add_argument("--keep-frames", default="", help="dir: save each hover's wrist frame + masks (debug)")
    a = ap.parse_args()

    from dojo.perception.grasp_metric import BeliefTracker, Footprints, Groups, episode_events, score_event
    from dojo.perception.seq_data import Sequence
    from dojo.perception.wrist_check import CheckConfig, WristCheck

    cfg = CheckConfig()
    for kv in a.set:
        k, v = kv.split("=", 1)
        setattr(cfg, k, json.loads(v))
    sam = None
    if a.sam:
        from dojo.perception.sam_masks import LocalSam
        sam = LocalSam([cfg.sam_model])
    emb = None
    if a.embed:
        from dojo.perception.embed_id import Embedder
        emb = Embedder(a.embed)
        cfg.embed_model = a.embed
    events = json.load(open(os.path.expanduser(a.events)))
    eps = sorted({e["episode"] for e in events})
    if a.episodes:
        eps = [x for x in eps if x in a.episodes.split(",")]
    hov = {(e["episode"], e["kind"], int(e["step"]), e["arm"]): e.get("hover_frame") for e in events}
    rows, t_all = [], time.perf_counter()
    for ep in eps:
        seq = Sequence(os.path.join(os.path.expanduser(a.seq_root), ep))
        task = seq.task
        report, actions = load_report(seq), load_actions(seq.meta["actions"])
        fp, groups = Footprints(), Groups(task)
        from dojo.perception.inverse_layout import label_groups
        lab_cands = {}
        for g in label_groups(task):
            for lab in g["labels"]:
                lab_cands[lab] = list(g["candidates"])
        wc = WristCheck(cfg, sam=sam, embed=emb, candidates=lambda label, key: lab_cands.get(label, [key]))
        head_bgs, head_thr = None, 18.0
        if a.head:
            import zlib

            from dojo.runtime.perception_service import _Perception
            P = _Perception()
            P.set_scene(task)
            fold = zlib.crc32(str(seq.layout_id).encode()) % 2      # = stream C's replay_eval fold
            head_bgs, head_thr = P.bgs(fold), float(P.cfg(fold).fg_thr)
        bt = BeliefTracker(report, actions)
        for e, b in episode_events(report, actions):
            if e.kind not in ("grasp", "place"):
                continue
            k = int(e.step)
            gt = seq.gt(k - 1)
            h = hov.get((ep, e.kind, k, e.arm)) if a.hover == "runtime" else None
            src = "runtime"
            if h is None:
                h, src = track_hover(bt.track[e.arm], k, e.point), "track"
            h = int(h)
            if h not in seq.by_step:
                h = max(s for s in seq.steps if s <= h)
            fr = seq.frame(h)
            held = None
            if e.kind == "place" and e.target in b:
                from dojo.perception.grasp_metric import _R_quat, pose_T
                T = bt.T(e.arm, h) @ np.linalg.inv(bt.T(e.arm, k - 1)) @ pose_T(b[e.target]["pos"], b[e.target]["quat"])
                held = (b[e.target]["key"], T[:3, 3].tolist() + _R_quat(T[:3, :3]).tolist())
            bel_in = {lab: o for lab, o in b.items() if o.get("pos") is not None}
            t0 = time.perf_counter()
            try:
                out = wc.check(fr, bel_in, e.arm, "pick" if e.kind == "grasp" else "place",
                               target=e.target if e.kind == "grasp" else None,
                               reference=e.reference if e.kind == "place" else None,
                               corridor=e.corridor, held=held, held_label=e.target if e.kind == "place" else None,
                               head_bgs=head_bgs, head_thr=head_thr)
                err = None
            except Exception as ex:  # noqa: BLE001
                import traceback
                out, err = {"objects": {}, "found": [], "time_s": {}}, f"{type(ex).__name__}: {ex} | {traceback.format_exc()[-600:]}"
            dt = time.perf_counter() - t0
            row = {"episode": ep, "task": task, "kind": e.kind, "step": k, "arm": e.arm, "target": e.target,
                   "reference": e.reference, "hover": h, "hover_src": src, "check": {k2: v for k2, v in out.items()
                                                                                   if k2 != "belief"},
                   "wall_s": round(dt, 3), "error": err, "scores": {}}
            for arm in ARMS:
                if arm == "id" and emb is None:
                    continue
                bel = arm_belief(bel_in, out, arm)
                s = score_event(e, bel, gt, task, footprints=fp, groups=groups)
                row["scores"][arm] = {"clean_r": s["clean_r"], "reasons": s["reasons"], "clean_target": s["clean_target"],
                                      "clean_required": s["clean_required"],
                                      "required": [{k3: r.get(k3) for k3 in ("role", "label", "cls", "xy_mm", "key", "gt_key")}
                                                   for r in s["required"]]}
            # dev-only: what each checked object is in GT (for the verdict's precision / recall), scorer side
            gts = {g["label"]: g for g in gt}
            for lab, r in row["check"].get("objects", {}).items():
                o = bel_in.get(lab)
                if o is None:
                    continue
                c = fp.centre(o["key"], o["pos"], o["quat"])
                dmin, glab = 1e9, None
                for g in gt:
                    from dojo.perception.grasp_metric import gt_key
                    gk = gt_key(g)
                    try:
                        gc = fp.centre(gk, g["pos"], g["quat"])
                    except Exception:  # noqa: BLE001
                        gc = np.asarray(g["pos"], float)
                    d = float(np.linalg.norm(np.asarray(gc)[:2] - np.asarray(c)[:2]))
                    if groups.of(g["label"]) == groups.of(lab) and d < dmin:
                        dmin, glab = d, g["label"]
                r["dev_gt_mm"] = round(1000 * dmin, 1) if glab else None
                g = gts.get(glab)
                if g is not None:                   # identity / flip truth (dev only)
                    from dojo.perception.grasp_metric import _rot_err
                    r["dev_gt_key"] = gt_key(g)
                    r["dev_rot_deg"] = _rot_err(gt_key(g), o["quat"], g["quat"])
                    idr = r.get("identity") or {}
                    if idr.get("flip_pose") is not None or "flip_margin" in idr:
                        from dojo.perception.wrist_check import yaw_flip
                        cc = fp.centre(o["key"], o["pos"], o["quat"])
                        fpos, fq = yaw_flip(o["pos"], o["quat"], cc)
                        r["dev_rot_deg_flip"] = _rot_err(gt_key(g), fq, g["quat"])
                if r.get("applied") and r.get("pose7"):
                    c2 = fp.centre(o["key"], r["pose7"][:3], r["pose7"][3:7])
                    g = gts.get(glab)
                    if g is not None:
                        gc = fp.centre(gt_key(g), g["pos"], g["quat"])
                        r["dev_gt_mm_after"] = round(1000 * float(np.linalg.norm(np.asarray(gc)[:2] - np.asarray(c2)[:2])), 1)
            rows.append(row)
            tag = " ".join(f"{arm}:{int(v['clean_r']['10'])}" for arm, v in row["scores"].items())
            print(f"[e1] {ep} {e.kind} {k} {e.arm} {e.target} hover {h}({src}) {dt:.2f}s {tag}"
                  + (f" ERR {err[:200]}" if err else ""), flush=True)
    res = {"cfg": cfg.to_dict(), "sam": bool(a.sam), "embed": a.embed or None, "hover": a.hover, "head": a.head,
           "rows": rows,
           "wall_s": round(time.perf_counter() - t_all, 1)}
    if sam is not None:
        res["sam_info"] = sam.M.info()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(summary(rows))


def summary(rows) -> str:
    arms = [x for x in ARMS if rows and x in rows[0]["scores"]]
    lines = [f"events {len(rows)}"]
    for R in ("5", "10", "15"):
        cells = []
        for arm in arms:
            c = sum(bool(r["scores"][arm]["clean_r"][R]) for r in rows)
            fixed = sum(bool(r["scores"][arm]["clean_r"][R]) and not r["scores"]["rt"]["clean_r"][R] for r in rows)
            broke = sum(not r["scores"][arm]["clean_r"][R] and r["scores"]["rt"]["clean_r"][R] for r in rows)
            cells.append(f"{arm} {c}/{len(rows)} (+{fixed} -{broke})")
        lines.append(f"R={R}cm: " + " | ".join(cells))
    t = [r["wall_s"] for r in rows]
    if t:
        lines.append(f"wall per hover: median {np.median(t):.2f}s p90 {np.percentile(t, 90):.2f}s max {max(t):.2f}s")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
