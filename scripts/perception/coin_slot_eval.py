"""DEV (stream C): the pick-hover read of a slot-pinned disc (deposit_coin's coin in its stand), offline.

Frames: per-step sequences (``dojo.perception.seq_capture``) of recorded deposit_coin episodes; GT per step (``gt``
rows, dev-only, scorer side). Selects the steps where a wrist camera looks down at the coin from 6-30 cm, then reads
the coin from that one frame with each method from priors made by perturbing the GT stand (yaw, xy) and re-deriving
the coin in its slot, as perception's belief does. Scores the coin's disc normal (deg, sign-free), its offset along
the normal (mm: what moves a pad onto the disc) and in the disc plane.

    python scripts/perception/coin_slot_eval.py SEQ_DIR [SEQ_DIR ...] --out rows.jsonl [--per-ep 6]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import cv2
import numpy as np

from dojo.perception.cameras import R_to_quat_wxyz, quat_wxyz_to_R, wrist_camera
from dojo.perception.inverse_layout import support_slot_pose, task_map

COIN, STAND = "Rigid/coin/00000", "Geometry/vertical_coin_stand/00000"
H_MAX = 0.17           # a pick hover: the wrist camera at most this far above the coin
Z_REST_MAX = 0.79      # the coin still in its stand
GRIP_OPEN_MIN = 0.3    # ... with the jaws open or pre-shaped
CENTRE_PX = 220        # ... and near the view's centre (a pick hover is planned over the coin; a place hover over the
                       # bank has it at the edge)
PERTURB = [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (-48.0, 0.0, 0.0), (48.0, 0.0, 0.0), (90.0, 0.0, 0.0),
           (0.0, 0.012, 0.0), (0.0, -0.012, 0.0), (0.0, 0.0, 0.012), (30.0, 0.008, 0.0),
           (0.0, 0.03, 0.0), (45.0, 0.0, 0.03)]    # the last two: beyond the read's 18 mm radius (it should refuse)


def gt_rows(row):
    g = {}
    for o in row.get("gt") or []:
        g[o["label"]] = o
    return g


def yaw_of(q):
    R = quat_wxyz_to_R(q)
    return math.atan2(R[1, 0], R[0, 0])


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def coin_from_stand(stand_pos, stand_q, placement):
    p, q = support_slot_pose(STAND, stand_pos, stand_q, COIN, placement)
    return np.asarray(p, float), np.asarray(q, float)


def disc_normal(q):
    from dojo.perception.render import load_asset
    a = load_asset(COIN)
    k = int(np.argmin(a.extents))
    return quat_wxyz_to_R(q)[:, k]


def score(pos, q, gpos, gq):
    n, gn = disc_normal(q), disc_normal(gq)
    ang = math.degrees(math.acos(min(1.0, abs(float(n @ gn)))))
    e = np.asarray(pos, float) - np.asarray(gpos, float)
    along = abs(float(e @ gn))
    plane = float(np.linalg.norm(e - (e @ gn) * gn))
    return {"normal_deg": round(ang, 1), "along_mm": round(1000 * along, 1), "plane_mm": round(1000 * plane, 1)}


def select_steps(rows, frames_dir, per_ep):
    out = []
    for r in rows:
        g = gt_rows(r).get("coin0")
        if g is None:
            continue
        c = np.asarray(g["pos"], float)
        for arm in ("left", "right"):
            f = glob.glob(os.path.join(frames_dir, f"{r['step']:04d}_cam_{arm}_wrist.*"))
            if not f:
                continue
            gj = np.ravel(r["state"].get(f"{arm}_ee_joint_state", [1.0]))
            if gj.size and float(gj[0]) < GRIP_OPEN_MIN:
                continue                           # closed jaws: not a pick hover (after a missed close)
            cam = wrist_camera(arm, np.asarray(r["state"][f"{arm}_arm_joint_state"], float))
            Tc = cam.T_wc
            pc = cam.T_cw @ np.r_[c, 1.0]
            h = float(Tc[2, 3] - c[2])
            if pc[2] <= 0.03 or not (0.06 <= h <= H_MAX) or c[2] > Z_REST_MAX:
                continue
            uv = cam.K @ (pc[:3] / pc[2])
            if math.hypot(uv[0] - 320, uv[1] - 240) > CENTRE_PX:
                continue
            out.append((r, arm, f[0], h))
    if len(out) > per_ep:
        idx = np.linspace(0, len(out) - 1, per_ep).round().astype(int)
        out = [out[i] for i in sorted(set(idx))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-ep", type=int, default=6)
    ap.add_argument("--methods", default="refine,slot_read")
    a = ap.parse_args()
    from dojo.perception.refine import refine
    tm = task_map()["deposit_coin"]["labels"]
    placement = tm["coin0"]["placement"]
    bank_key = next(iter(tm.get("piggy_bank", {}).get("candidates") or []), None)
    methods = a.methods.split(",")
    fo = open(a.out, "w")
    for sd in a.seq:
        rows = [json.loads(line) for line in open(os.path.join(sd, "steps.jsonl"))]
        gpath = os.path.join(sd, "gt.jsonl")
        if os.path.isfile(gpath):                  # the current capture keeps GT apart from the perceiver's inputs
            gts = {}
            for line in open(gpath):
                g = json.loads(line)
                gts[g["step"]] = g.get("gt") or g.get("rows") or []
            for r in rows:
                r["gt"] = gts.get(r["step"], [])
        sel = select_steps(rows, os.path.join(sd, "frames"), a.per_ep)
        print(f"{os.path.basename(sd)}: {len(sel)} hover frames", flush=True)
        for r, arm, fpath, h in sel:
            img = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
            g = gt_rows(r)
            gs, gc = g["vertical_coin_stand"], g["coin0"]
            gsp, gsq = np.asarray(gs["pos"], float), np.asarray(gs["quat"], float)
            for dyaw, dn, dp in PERTURB:
                # perturb the stand about its own z (yaw), then shift along / across the GT coin normal
                Rs = rotz(math.radians(dyaw)) @ quat_wxyz_to_R(gsq)
                n = disc_normal(gc["quat"])
                n2 = np.array([n[0], n[1], 0.0])
                n2 /= np.linalg.norm(n2)
                t2 = np.array([-n2[1], n2[0], 0.0])
                sp = gsp + dn * n2 + dp * t2
                sq = np.asarray(R_to_quat_wxyz(Rs), float)
                cp, cq = coin_from_stand(sp, sq, placement)
                base = {"seq": os.path.basename(sd), "step": r["step"], "arm": arm, "h_m": round(h, 3),
                        "perturb": [dyaw, dn, dp], "prior": score(cp, cq, gc["pos"], gc["quat"])}
                for m in methods:
                    t0 = time.time()
                    try:
                        if m in ("refine", "refine_r70"):
                            kw = {"search_radius": 0.07} if m == "refine_r70" else {}
                            o = refine(COIN, img, r["state"], list(cp) + list(cq), arm, empty_hand=True, **kw)
                            res = {"ok": o["ok"], "iou": round(o["iou"], 3), **score(o["pose"][:3], o["pose"][3:], gc["pos"], gc["quat"]),
                                   "turned": o.get("turned_deg")}
                        elif m == "slot_read":
                            from dojo.perception.slot_read import slot_read
                            bank = g.get("piggy_bank")
                            others = [(bank_key, list(bank["pos"]) + list(bank["quat"]))] if bank and bank_key else None
                            o = slot_read(STAND, list(sp) + list(sq), COIN, placement, img, r["state"], arm,
                                          others=others)
                            if "child_pose" in o:
                                res = {"ok": o["ok"], "iou": o["iou"], "hull_iou": o.get("hull_iou"), "margin": o["yaw_margin"], "why": o["why"],
                                       **score(o["child_pose"][:3], o["child_pose"][3:], gc["pos"], gc["quat"]),
                                       "stand_yaw_err": round(math.degrees((yaw_of(o["base_pose"][3:]) - yaw_of(gsq) + math.pi / 2) % math.pi - math.pi / 2), 1)}
                            else:
                                res = {"ok": False, "why": o.get("why")}
                        else:
                            raise ValueError(m)
                    except Exception as e:  # noqa: BLE001
                        res = {"error": f"{type(e).__name__}: {e}"[:200]}
                    res["time_s"] = round(time.time() - t0, 2)
                    fo.write(json.dumps({**base, "method": m, **res}) + "\n")
                fo.flush()
    fo.close()


if __name__ == "__main__":
    main()
