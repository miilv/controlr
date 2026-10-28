"""Stream H, E2-style: the held object's in-hand pose read at the PRE-PLACE HOVER (``held.held_pose``), offline on stream
C's per-step sequences, scored with stream M's grasp-time metric at every grasp / place. 0 LLM calls.

    python scripts/perception/h95_hover_eval.py --seq-root SEQ_ROOT --out OUT.json [--episodes a,b]
        [--arms rt,oracle,read_ok,read_coarse] [--trigger all|swing|doubt|swing_or_doubt] [--set k=v ...]

For every place event (M's ``episode_events``: step k, arm, held label) the hover is the arm's last still step before
the place's descent (``hover_step``: the pre-place "still" the runtime's ``run_place`` holds, where the place refine and
the insertion held read already run; no extra env steps). There the belief's in-hand transform (M's ``BeliefTracker``,
riding the commanded TCP) is patched and rides on to the release:
  rt           the runtime's own belief;
  oracle       the GT in-hand transform at the hover (dev only: the ceiling of a perfect hover read);
  read_ok      ``held_pose`` on the hover frame, adopted when held_pose calls it ok (insertion-grade sigma);
  read_coarse  also an "ambiguous" fit whose position sigmas are all <= ``--coarse-sigma-mm`` at a best-view IoU >=
               ``--coarse-iou`` (non-insertion places tolerate mm-level, not sub-mm, in-hand error).
``--trigger`` restricts which places read (the runtime's ``place_held_read`` rule): ``swing`` = the object's centre of
mass sits >= ``--lever-mm`` off the pads' pivot axis against gravity at the hover (it can turn about the closing axis:
a mug by its wall, a broom by its handle), ``doubt`` = the pick refine of that grasp was not ok or fit below
``--doubt-iou``.
Reads are cached per place, so all arms share one read. Per read: prior / fit / GT in-hand error, ok / why, wall time.
Needs stream C's ``dojo.perception.seq_data`` and stream W's ``scripts/perception/wrist/e2_release.py`` (tracker patch).
DEV-ONLY: GT is read here, never by a policy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def hover_step(actions: list[dict], arm: str, k: int, max_back: int = 160) -> int | None:
    """The step (actions executed) of the arm's pre-place hover: walking back from the release command k: the settle
    hold (identical joint commands), the descent (moving), then the hover's still step(s); returns the last still
    step's index. None when the pattern is not there."""
    key = f"{arm}_arm_joint_state"

    def q(i):
        return np.asarray(actions[i - 1][key], float)

    i = k - 1
    lo = max(2, k - max_back)

    def still(i):
        return float(np.abs(q(i) - q(i - 1)).max()) < 1e-6

    while i > lo and still(i):                 # settle after the descent
        i -= 1
    moved = 0
    while i > lo and not still(i):             # the descent (and any turn)
        i -= 1
        moved += 1
    if i <= lo or moved < 2:
        return None
    return i                                   # the hover's still step (q(i) == q(i - 1))


def swing_lever_mm(T_tcp_obj, com_obj, T_tcp_world) -> float:
    """How far (mm) the centre of mass sits off the closing axis (TCP y) across gravity at this TCP pose: the lever
    gravity turns the object about the pads with."""
    c = np.asarray(T_tcp_obj, float)[:3, :3] @ np.asarray(com_obj, float) + np.asarray(T_tcp_obj, float)[:3, 3]
    g = np.asarray(T_tcp_world, float)[:3, :3].T @ np.array([0.0, 0.0, -1.0])
    return abs(float(np.cross(c, g)[1])) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="")
    ap.add_argument("--arms", default="rt,oracle,read_ok,read_coarse")
    ap.add_argument("--trigger", default="all", choices=("all", "swing", "doubt", "swing_or_doubt"))
    ap.add_argument("--lever-mm", type=float, default=20.0)
    ap.add_argument("--doubt-iou", type=float, default=0.85)
    ap.add_argument("--coarse-sigma-mm", type=float, default=3.0)
    ap.add_argument("--coarse-iou", type=float, default=0.75)
    ap.add_argument("--held-gap-mm", type=float, default=15.0,
                    help="patch only places whose GT object is in the jaws at the hover: the TCP within this of its box "
                         "(empty jaws are grip_check's, not a read's; <= 0: every place)")
    ap.add_argument("--min-change-mm", type=float, default=0.0,
                    help="adopt a read / the oracle only when it moves the body centre this far (or turns min-change-deg)")
    ap.add_argument("--min-change-deg", type=float, default=0.0)
    ap.add_argument("--read-cache", default="", help="a JSON file of hover reads: loaded if present, extended and saved "
                                                         "(re-score other gates / triggers without new reads)")
    ap.add_argument("--require-done", default="")
    ap.add_argument("--set", action="append", default=[], help="held_pose cfg field=value (JSON value)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(HERE, "wrist"))
    from e2_release import at_or_before, load_actions, load_report, patched_tracker, summary

    from dojo.perception import grasp_metric as G
    from dojo.perception.held import held_pose, tcp_T
    from dojo.perception.seq_data import Sequence
    from dojo.skills.grasps import mesh_com
    from dojo.skills.world import AssetDB

    cfg = {}
    for kv in a.set:
        k_, v = kv.split("=", 1)
        cfg[k_] = json.loads(v)
    root = os.path.expanduser(a.seq_root)
    eps = sorted(x for x in os.listdir(root) if x != "smoke" and os.path.exists(os.path.join(root, x, "meta.json")))
    if a.episodes:
        eps = [x for x in eps if x in a.episodes.split(",")]
    if a.require_done:
        eps = [x for x in eps if os.path.exists(os.path.join(os.path.expanduser(a.require_done), x, "_done"))]
    arms = a.arms.split(",")
    adb = AssetDB()
    com_cache: dict = {}

    def com_of(key):
        if key not in com_cache:
            try:
                typ, cat, idx = key.split("/")
                com_cache[key] = np.asarray(mesh_com(adb.get(typ, cat, int(idx), mesh=True)), float)
            except Exception:  # noqa: BLE001 (no metadata here: the catalogue mesh's bbox centre)
                from dojo.perception.render import load_asset
                V = np.asarray(load_asset(key).V, float)
                com_cache[key] = (V.min(0) + V.max(0)) / 2
        return com_cache[key]

    rows, reads = [], []
    disk = {}
    if a.read_cache and os.path.exists(a.read_cache):
        disk = json.load(open(a.read_cache))
    t_all = time.perf_counter()
    for ep in eps:
        seq = Sequence(os.path.join(root, ep))
        task = seq.task
        report, actions = load_report(seq), load_actions(seq.meta["actions"])
        if report is None:
            continue
        fp, groups = G.Footprints(), G.Groups(task)
        places = [(int(e.step), e.arm, e.target) for e, _ in G.episode_events(report, actions)
                  if e.kind == "place" and e.target]
        picks = {}                                         # (arm, label) -> [(step, ok, iou)] pick refines
        for e in report.get("events") or []:
            if e.get("kind") == "refine" and e.get("for") == "pick" and e.get("label"):
                picks.setdefault((e.get("arm"), e["label"]), []).append((int(e["step"]), bool(e.get("ok")),
                                                                         float(e.get("iou") or 0.0)))
        hover = {}
        for k, arm, lab in places:
            h = hover_step(actions, arm, k)
            hover[(k, arm)] = at_or_before(seq, h) if h is not None else None
        cache: dict = {}                                   # (k, arm) -> read record (shared by the read arms)

        def gt_world(j, lab, bt, arm=None):
            """Dev only: the GT pose of the object of ``lab``'s group in ``arm``'s jaws (nearest the TCP; interchangeable
            pieces: the one actually held), else nearest the belief."""
            o = bt.snapshot(j).get(lab)
            if o is None or o.get("pos") is None:
                return None
            c = fp.centre(o.get("key"), o["pos"], o["quat"])
            if arm is not None:
                c = tcp_T(arm, seq.state(j)[f"{arm}_arm_joint_state"][:6])[:3, 3]
            same = [g for g in G.gt_rows_asset(seq.gt(j)) if groups.of(g["label"]) == groups.of(lab)]
            if not same:
                return None
            g = min(same, key=lambda g: float(np.linalg.norm(fp.centre(G.gt_key(g), g["pos"], g["quat"]) - c)))
            return G.pose_T(g["pos"], g["quat"])

        def held_at(j, arm, lab, bt) -> bool:
            """Dev only: the GT object of the label's group nearest the belief is in the jaws at step j."""
            if a.held_gap_mm <= 0:
                return True
            Tg = gt_world(j, lab, bt, arm)
            key = (bt.belief.get(lab) or {}).get("key")
            if Tg is None or not key:
                return False
            from dojo.perception.render import load_asset
            V = np.asarray(load_asset(key).V, float)
            p = tcp_T(arm, seq.state(j)[f"{arm}_arm_joint_state"][:6])[:3, 3]
            loc = Tg[:3, :3].T @ (p - Tg[:3, 3])
            d = np.maximum(np.maximum(V.min(0) - loc, loc - V.max(0)), 0.0)
            return float(np.linalg.norm(d)) * 1000 <= a.held_gap_mm

        def place_read(bt, k, arm, lab, j):
            """The hover read of (k, arm) on frame j (cached): T_world of the fit and its record."""
            if (k, arm) in cache:
                return cache[(k, arm)]
            dk = f"{ep}|{k}|{arm}"
            if dk in disk:
                tw, rec = disk[dk]
                cache[(k, arm)] = (None if tw is None else np.asarray(tw, float).reshape(4, 4), rec)
                reads.append(rec)
                return cache[(k, arm)]
            lab0, T_rel, _ = bt.attached[arm]
            o = bt.belief.get(lab) or {}
            key = o.get("key")
            T_cmd = bt.T(arm, j)
            st = seq.state(j)
            T_meas = tcp_T(arm, st[f"{arm}_arm_joint_state"][:6])
            T_prior_meas = np.linalg.inv(T_meas) @ T_cmd @ T_rel
            lever = swing_lever_mm(T_prior_meas, com_of(key), T_meas) if key else None
            pk = [p for p in picks.get((arm, lab), []) if p[0] <= k]
            doubt = bool(pk) and (not pk[-1][1] or pk[-1][2] < a.doubt_iou)
            others = []
            snap = bt.snapshot(j)
            for l2, x in snap.items():
                if l2 != lab and x.get("pos") is not None and x.get("key"):
                    others.append({"key": x["key"], "pose7": list(x["pos"]) + list(x["quat"])})
            rec = {"episode": ep, "task": task, "label": lab, "arm": arm, "release": k, "hover": j, "key": key,
                   "lever_mm": None if lever is None else round(lever, 1), "doubt": doubt}
            t0 = time.perf_counter()
            try:
                r = held_pose(seq.frame(j), arm, key, T_prior_meas, {"others": others, **cfg})
            except Exception as ex:  # noqa: BLE001 (a dev eval: record and go on)
                r = {"ok": False, "why": f"error {type(ex).__name__}: {str(ex)[:160]}"}
            rec["wall_s"] = round(time.perf_counter() - t0, 2)
            T_fit = r.get("T_tcp_obj") if r.get("ok") else r.get("fit_T_tcp_obj")
            sig = r.get("sigma") or {}
            pos_sig = max([float(v) for v in sig.get("pos_mm") or []] or [math.inf])
            iou = max([float(v) for v in (r.get("iou") or {}).values()] or [0.0])
            rec.update(ok=bool(r.get("ok")), why=r.get("why"), views=r.get("views"), iou=round(iou, 3),
                       sig_pos_mm=None if math.isinf(pos_sig) else round(pos_sig, 2),
                       shift_mm=round(float(r.get("shift_mm") or 0), 1), turn_deg=round(float(r.get("turn_deg") or 0), 1))
            coarse = bool(r.get("ok")) or (r.get("why") == "ambiguous" and pos_sig <= a.coarse_sigma_mm
                                           and iou >= a.coarse_iou)
            rec["coarse"] = coarse
            Tw_fit = None if T_fit is None else T_meas @ np.asarray(T_fit, float)
            # dev only: in-hand error before / after against the GT object at the hover
            Tg = gt_world(j, lab, bt, arm)
            if Tg is not None and key:
                c = fp.centre(key, [0, 0, 0], [1, 0, 0, 0])

                def err(Tw):
                    if Tw is None:
                        return None
                    d = (Tw[:3, :3] @ c + Tw[:3, 3]) - (Tg[:3, :3] @ c + Tg[:3, 3])
                    ang = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(Tw[:3, :3].T @ Tg[:3, :3])) - 1) / 2))))
                    return [round(float(np.linalg.norm(d)) * 1000, 1), round(ang, 1)]
                rec["err_prior"] = err(T_meas @ T_prior_meas)
                rec["err_fit"] = err(Tw_fit)
            cache[(k, arm)] = (Tw_fit, rec)
            disk[dk] = [None if Tw_fit is None else np.asarray(Tw_fit).reshape(-1).tolist(), rec]
            reads.append(rec)
            print(f"   [hover] {ep} {lab} k={k} @{j} lever {rec['lever_mm']} doubt {doubt} ok={rec['ok']} {rec['why']} "
                  f"coarse={coarse} err {rec.get('err_prior')} -> {rec.get('err_fit')} iou {rec['iou']} "
                  f"{rec['wall_s']}s", flush=True)
            return cache[(k, arm)]

        def wanted(rec) -> bool:
            sw = rec.get("lever_mm") is not None and rec["lever_mm"] >= a.lever_mm
            return {"all": True, "swing": sw, "doubt": rec.get("doubt"), "swing_or_doubt": sw or rec.get("doubt")}[a.trigger]

        per_arm = {}
        for arm_name in arms:
            done = set()

            def patcher(bt, upto, arm_name=arm_name, done=done):
                if arm_name == "rt":
                    return
                for k, arm, lab in places:
                    j = hover.get((k, arm))
                    if j is None or (k, arm) in done or j > upto or arm not in bt.attached or bt.attached[arm][0] != lab:
                        continue
                    done.add((k, arm))
                    if not held_at(j, arm, lab, bt):
                        continue
                    T_cmd = bt.T(arm, j)
                    if arm_name == "oracle":
                        _, rec = place_read(bt, k, arm, lab, j) if a.trigger != "all" else (None, None)
                        if rec is not None and not wanted(rec):
                            continue
                        Tw = gt_world(j, lab, bt, arm)
                    else:
                        Tw, rec = place_read(bt, k, arm, lab, j)
                        if not wanted(rec) or not (rec["ok"] if arm_name == "read_ok" else rec["coarse"]):
                            continue
                    if Tw is not None:
                        lab0, T_rel0, pre = bt.attached[arm]
                        if a.min_change_mm > 0 or a.min_change_deg > 0:
                            key = (bt.belief.get(lab) or {}).get("key")
                            c = fp.centre(key, [0, 0, 0], [1, 0, 0, 0]) if key else np.zeros(3)
                            T0 = T_cmd @ T_rel0
                            dmm = float(np.linalg.norm((Tw[:3, :3] @ c + Tw[:3, 3]) - (T0[:3, :3] @ c + T0[:3, 3]))) * 1000
                            ddeg = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(Tw[:3, :3].T @ T0[:3, :3])) - 1) / 2))))
                            if dmm < a.min_change_mm and ddeg < a.min_change_deg:
                                continue
                        bt.attached[arm] = (lab0, np.linalg.inv(T_cmd) @ Tw, pre)

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
                                             "clean_required": s["clean_required"], "clean_target": s["clean_target"],
                                             "held_cls": s["required"][0].get("cls") if e.kind == "place" else None}
            per_arm[arm_name] = out
        miss = {"clean_r": {"5": False, "10": False, "15": False}, "reasons": {"5": ["not scored"], "10": ["not scored"],
                "15": ["not scored"]}, "clean_required": False, "clean_target": False, "held_cls": None}
        keys = sorted(set().union(*(set(v) for v in per_arm.values())))
        for key in keys:
            ev = next(per_arm[an][key] for an in arms if key in per_arm[an])
            rows.append({"episode": ep, "task": task, "arm": key[1],
                         **{k_: ev[k_] for k_ in ("kind", "step", "target", "reference")},
                         "scores": {an: {k_: per_arm[an].get(key, miss)[k_] for k_ in
                                         ("clean_r", "reasons", "clean_required", "clean_target", "held_cls")}
                                    for an in arms}})
        tags = " ".join(f"{an}:{sum(x['clean_r']['10'] for x in per_arm[an].values())}/{len(per_arm[an])}" for an in arms)
        print(f"[hover] {ep} places {len(places)} hovers {sum(v is not None for v in hover.values())} {tags}", flush=True)
    if a.read_cache:
        json.dump(disk, open(a.read_cache, "w"))
    res = {"arms": arms, "trigger": a.trigger, "lever_mm": a.lever_mm, "cfg": cfg, "rows": rows, "reads": reads,
           "wall_s": round(time.perf_counter() - t_all, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(summary(rows, arms))


if __name__ == "__main__":
    main()
