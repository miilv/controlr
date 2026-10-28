"""Stream H: the in-hand pose read from a wrist frame during the lift / carry (``held.held_pose``), against GT.

For every place event stream M scored (``m95_score.py score`` -> ``events.jsonl``) inside one of stream C's per-step
sequences (``~/dojo-data/bgperc/seq/<task>_<layout>/``: ``meta.json``, ``steps.jsonl`` with the official state and the
dev-only GT rows per step, ``frames/<step:04d>_<cam>.jpg``):

- prior = the runtime's belief at the grasp (``grasp_metric.episode_events``: the target's believed pose) in the TCP
  frame of the close: what the belief carries to the place today;
- read frame = the first step >= ``--read-after`` steps after the close where the TCP has risen ``--lift-mm`` (the
  runtime's held check moment: no extra steps), before the release;
- fit = ``held_pose`` on that frame (own wrist + head + other wrist), with ``--cfg``;
- truth = the GT object in the measured TCP frame at the read frame, and at the frame before the release.

Per event: the body-centre error (mm) and rotation (deg) of prior and fit in the hand at the read, and at the release
if the belief had adopted the fit at the read (``rel_*``: what M's metric would score; drift after the read included).

    python scripts/perception/h95_carry_eval.py --seq-root ~/dojo-data/bgperc/seq --events SCORE/events.jsonl \
        --out OUT.jsonl [--cfg '{"max_shift_mm": 10}'] [--tasks hang_mugs,stack_bowls] [--only-bad]

Finding (2026-10-28, journal 2026-10-28-dojo-h95-held-state): on hang_mugs / stack_bowls the in-hand error that M
scores at the place arises AFTER the lift (the mug swings in its wall grasp, the bowl turns), and held_pose on mugs fits
at IoU 0.6-0.8; a read at the lift is not a lever, so the runtime carry read was dropped (the code is in this PR's
first commit). A read belongs at the place hover (BACKLOG).

DEV-ONLY: GT is read here, never by a policy. CPU (perception's CPU silhouettes when no GL).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import time
from pathlib import Path

import numpy as np

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _load(p: str | Path):
    with gzip.open(p, "rt") as f:
        return json.load(f)


def _steps(seq: Path) -> dict[int, dict]:
    out = {}
    for ln in open(seq / "steps.jsonl"):
        if ln.strip():
            r = json.loads(ln)
            out[int(r["step"])] = r
    return out


def _frame(seq: Path, step: int, state: dict) -> dict:
    import cv2
    imgs = {}
    for c in CAMS:
        p = seq / "frames" / f"{step:04d}_{c}.jpg"
        if not p.exists():
            p = p.with_suffix(".png")
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(p)
        imgs[c] = im[..., ::-1].copy()
    return {"images": imgs, "state": {k: np.asarray(v, float) for k, v in state.items()}}


def _rotvec_deg(R) -> list:
    from scipy.spatial.transform import Rotation
    return np.round(np.degrees(Rotation.from_matrix(R).as_rotvec()), 1).tolist()


def _rot_deg(Ra, Rb) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(Ra.T @ Rb)) - 1) / 2))))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-root", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--only-bad", action="store_true", help="only places whose held object M scored unclean")
    ap.add_argument("--read-after", type=int, default=8)
    ap.add_argument("--lift-mm", type=float, default=30.0)
    ap.add_argument("--max-events", type=int, default=0)
    a = ap.parse_args(argv)
    from dojo.perception.grasp_metric import Footprints, episode_events, gt_rows_asset, pose_T, tcp_track
    from dojo.perception.held import held_pose, tcp_T
    from dojo.skills.kinematics import X5Kin

    cfg = json.loads(a.cfg)
    tasks = {t for t in a.tasks.split(",") if t}
    ev_all = [json.loads(ln) for ln in open(a.events) if ln.strip()]
    by_ep = collections.defaultdict(list)
    for e in ev_all:
        by_ep[(e["run"], int(e["layout_id"]))].append(e)
    kin, fp = X5Kin(), Footprints()
    done = set()
    if os.path.exists(a.out):
        done = {(r["run"], r["layout_id"], r["release_step"], r["arm"]) for r in map(json.loads, open(a.out)) if r}
    fo = open(a.out, "a")
    n = 0
    for seq in sorted(Path(os.path.expanduser(a.seq_root)).iterdir()):
        if not (seq / "meta.json").exists():
            continue
        meta = json.loads((seq / "meta.json").read_text())
        if tasks and meta["task"] not in tasks:
            continue
        act_p = Path(meta["actions"])
        run, lid = act_p.parent.name, int(meta["layout_id"])
        evs = sorted(by_ep.get((run, lid)) or [], key=lambda e: e["step"])
        if not evs:
            continue
        rep_p = act_p.with_name(act_p.name.replace("_actions.json.gz", "_report.json.gz"))
        report, actions = _load(rep_p), _load(act_p)["actions"]
        track = tcp_track(actions, kin)
        beliefs = {(ev.kind, ev.step, ev.arm): (ev, b) for ev, b in episode_events(report, actions, kin)}
        steps = _steps(seq)
        for i, e in enumerate(evs):
            if e["kind"] != "place" or (run, lid, e["step"], e["arm"]) in done:
                continue
            req = e["required"][0]
            if a.only_bad and req.get("cls") in ("ok", "not_held"):
                continue
            g = [x for x in evs[:i] if x["kind"] == "grasp" and x["arm"] == e["arm"]]
            if not g or not req.get("gt_label"):
                continue
            k0, k1, arm, lab = g[-1]["step"], e["step"], e["arm"], e["target"]
            gb = beliefs.get(("grasp", k0, arm))
            if gb is None or gb[0].target != lab or (gb[1].get(lab) or {}).get("pos") is None:
                continue
            bo = gb[1][lab]
            key = bo.get("key")
            T_prior = np.linalg.inv(track[arm][k0 - 1]) @ pose_T(bo["pos"], bo["quat"])
            if k0 - 1 not in steps:
                continue
            z0 = float(tcp_T(arm, steps[k0 - 1]["state"][f"{arm}_arm_joint_state"])[2, 3])
            f = next((k for k in range(k0 + a.read_after, k1 - 1) if k in steps and
                      float(tcp_T(arm, steps[k]["state"][f"{arm}_arm_joint_state"])[2, 3]) - z0 >= a.lift_mm / 1000), None)
            row = {"task": meta["task"], "run": run, "layout_id": lid, "arm": arm, "label": lab, "key": key,
                   "gt_label": req["gt_label"], "cls": req.get("cls"), "grasp_step": k0, "release_step": k1, "read_step": f}
            if f is None or (k1 - 1) not in steps:
                row["why"] = "no read frame (no lift before the release)"
                fo.write(json.dumps(row) + "\n")
                continue

            def gt_in_hand(k: int):
                rows = gt_rows_asset(steps[k]["gt"])
                gr = next((r for r in rows if r.get("label") == req["gt_label"]), None)
                if gr is None:
                    return None, None
                Tt = tcp_T(arm, steps[k]["state"][f"{arm}_arm_joint_state"])
                return np.linalg.inv(Tt) @ pose_T(gr["pos"], gr["quat"]), Tt

            T_gt, T_tcp_f = gt_in_hand(f)
            T_gt1, T_tcp_1 = gt_in_hand(k1 - 1)
            if T_gt is None or T_gt1 is None:
                row["why"] = "no GT row"
                fo.write(json.dumps(row) + "\n")
                continue
            c = fp.centre(key, [0, 0, 0], [1, 0, 0, 0])                 # body centre in the object frame

            def err(Ta, Tb) -> tuple[float, float]:
                pa, pb = Ta[:3, :3] @ c + Ta[:3, 3], Tb[:3, :3] @ c + Tb[:3, 3]
                return round(float(np.linalg.norm(pa - pb)) * 1000, 1), round(_rot_deg(Ta[:3, :3], Tb[:3, :3]), 1)

            t0 = time.time()
            try:
                r = held_pose(_frame(seq, f, steps[f]["state"]), arm, key, T_prior, cfg)
            except Exception as ex:  # noqa: BLE001 (a dev eval: record and go on)
                row["why"] = f"held_pose {type(ex).__name__}: {str(ex)[:120]}"
                fo.write(json.dumps(row) + "\n")
                continue
            T_fit = np.asarray(r.get("fit_T_tcp_obj") or r["T_tcp_obj"], float)
            row.update(ok=bool(r.get("ok")), fit_why=r.get("why"), views=r.get("views"), view_diag=r.get("view_diag"),
                       width_mm=r.get("width_mm"),
                       iou={k: round(float(v), 3) for k, v in (r.get("iou") or {}).items()},
                       sigma={k: np.round(v, 2).tolist() for k, v in (r.get("sigma") or {}).items()
                              if k in ("pos_mm", "rot_deg", "lead_lat_mm", "axis_deg")},
                       shift_mm=round(float(r.get("shift_mm") or 0), 1), turn_deg=round(float(r.get("turn_deg") or 0), 1),
                       t_s=round(time.time() - t0, 1),
                       prior_mm_deg=err(T_prior, T_gt), fit_mm_deg=err(T_fit, T_gt),
                       rel_prior_mm_deg=err(T_prior, T_gt1), rel_fit_mm_deg=err(T_fit, T_gt1),
                       drift_mm_deg=err(T_gt, T_gt1), prior_rotvec_tcp_deg=_rotvec_deg(T_gt[:3, :3] @ T_prior[:3, :3].T),
                       prior_dpos_tcp_mm=np.round((T_prior[:3, :3] @ c + T_prior[:3, 3] - T_gt[:3, :3] @ c - T_gt[:3, 3])
                                                  * 1000, 1).tolist())
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            n += 1
            print(f"{run} {lid} {lab} {req.get('cls')}: prior {row['prior_mm_deg']} fit {row['fit_mm_deg']} "
                  f"ok {row['ok']} ({row['fit_why']}) rel {row['rel_prior_mm_deg']} -> {row['rel_fit_mm_deg']} "
                  f"{row['t_s']} s", flush=True)
            if a.max_events and n >= a.max_events:
                return


if __name__ == "__main__":
    main()
