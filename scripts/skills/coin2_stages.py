"""Dev-only deposit_coin contact-stage report from GT replays (COIN2). Never a policy input.

    python3 scripts/skills/coin2_stages.py REPLAY_DIR [--reports REPORT_ROOT] [--json OUT]

REPLAY_DIR holds ``m95_replay.py --every 1 --full`` outputs (one ``<name>.json.gz`` per episode: GT objects, flange
poses and the applied gripper per step) or G3 ``--trace`` episode records converted to the same frame layout. Each
episode's executed actions give the gripper commands. Per close of a gripper near the coin it reports:

- ``tip_clear_mm``: the fingertips' height over the coin holder's top at the close (negative = inside the holder);
- ``depth_mm``: the coin centre below the TCP along the approach at the close, and after the close settles;
- ``left_holder``: the coin rose > 8 mm with this hand, and the stage at which the hold was lost (``lift`` while the
  TCP is within 40 mm of its close height, ``carry`` after, ``handover`` when the other hand closed on it, ``release``
  when this hand opened);
- ``slip_mm``: how far the coin moved in the TCP frame between the settled close and the end of the hold.

and per episode the coin's end state (``in_bank`` / ``in_slot_high`` / ``on_table`` / ``in_holder`` / ``held``).
GT is for scoring only.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys

import numpy as np

TCP_OFFSET = 0.145          # flange -> TCP along the flange x axis (dojo.skills.kinematics.TCP_OFFSET)
TIP = 0.0126                # fingertips beyond the TCP along the approach (finger model)
HOLDER_HALF = 0.00855          # vertical_coin_stand half height: its origin is its bbox centre (catalogue extents 17.1 mm)
TABLE_Z = 0.765
BANK_EXT = (0.148398, 0.20948, 0.115033)
ARMS = ("left", "right")


def qR(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tcp_T(fl) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = qR(fl[3:7])
    T[:3, 3] = np.asarray(fl[:3], float) + T[:3, 0] * TCP_OFFSET
    return T


def obj(frame, label):
    for o in frame["objects"]:
        if o.get("label") == label:
            return o
    return None


def grip_cmds(actions) -> dict:
    return {a: [float(np.asarray(act.get(f"{a}_ee_joint_state", [1.0]), float).reshape(-1)[0]) for act in actions]
            for a in ARMS}


def closes(cmd: list[float]) -> list[int]:
    """1-based action steps at which a gripper command closes below 0.5 after being open; the LAST closing command of
    each closed run is the grasp (a pre-shape opening is not one)."""
    out, k, n = [], 0, len(cmd)
    while k < n:
        if cmd[k] < 0.5 and (k == 0 or cmd[k - 1] >= 0.5):
            j = k
            while j + 1 < n and cmd[j + 1] < 0.5:
                j += 1
            # last strictly decreasing command inside the run: the final close
            last = k
            for i in range(k, j + 1):
                if i == k or cmd[i] < cmd[i - 1] - 1e-6:
                    last = i
            out.append((last + 1, j + 1))
            k = j + 1
        else:
            k += 1
    return out


def in_bank(c, bank) -> bool:
    R = qR(bank["quat"])
    loc = R.T @ (np.asarray(c, float) - np.asarray(bank["pos"], float))
    half = np.asarray(BANK_EXT) / 2
    # the bank's origin is at its base (placement height): the box spans z 0..h
    return bool(abs(loc[0]) <= half[0] and abs(loc[1]) <= half[1] and 0.0 <= loc[2] <= BANK_EXT[2])


def episode(rep: dict, actions: list[dict]) -> dict:
    fr = rep["frames"]
    steps = sorted(int(k) for k in fr)
    F = {s: fr[str(s)] for s in steps}
    s0 = steps[0]
    c0 = np.asarray(obj(F[s0], "coin0")["pos"], float)
    hold = obj(F[s0], "vertical_coin_stand")
    bank = obj(F[s0], "piggy_bank")
    holder_top = float(hold["pos"][2]) + HOLDER_HALF if hold else TABLE_Z + 2 * HOLDER_HALF
    cmd = grip_cmds(actions)
    rows = []
    for arm in ARMS:
        other = "right" if arm == "left" else "left"
        oc = closes(cmd[other])
        for k_close, k_end in closes(cmd[arm]):
            if k_close not in F:
                continue
            Tt = tcp_T(F[k_close]["flange"][arm])
            c = np.asarray(obj(F[k_close], "coin0")["pos"], float)
            rel = Tt[:3, :3].T @ (c - Tt[:3, 3])
            if np.linalg.norm(rel) > 0.06:
                continue                                   # not a close on the coin
            tips = Tt[:3, 3] + Tt[:3, 0] * TIP
            ks = min(k_close + 6, k_end)
            Ts = tcp_T(F[ks]["flange"][arm]) if ks in F else Tt
            cs = np.asarray(obj(F[ks], "coin0")["pos"], float) if ks in F else c
            rel_s = Ts[:3, :3].T @ (cs - Ts[:3, 3])
            # follow the hold until the coin leaves the hand (> 12 mm from its settled place in the TCP frame)
            lost_at, stage, rose, peak = None, None, False, float(cs[2])
            z_close = float(Tt[2, 3])
            last_rel = rel_s
            for s in range(ks, k_end + 1):
                if s not in F:
                    continue
                T = tcp_T(F[s]["flange"][arm])
                cc = np.asarray(obj(F[s], "coin0")["pos"], float)
                r = T[:3, :3].T @ (cc - T[:3, 3])
                peak = max(peak, float(cc[2]))
                if cc[2] > c0[2] + 0.008:
                    rose = True
                if np.linalg.norm(r - rel_s) > 0.012:
                    lost_at = s
                    taken = [k for k, _ in oc if k <= s and k > k_close]
                    stage = ("handover" if taken else
                             "lift" if T[2, 3] < z_close + 0.04 and np.linalg.norm(T[:2, 3] - Tt[:2, 3]) < 0.02 else
                             "carry")
                    break
                last_rel = r
            if lost_at is None:
                stage = "release"
            rows.append({"arm": arm, "close": k_close, "open": k_end,
                         "tip_clear_mm": round((float(tips[2]) - holder_top) * 1000, 1),
                         # at the settled grasp (after a snap rise from the walls)
                         "tip_clear_settled_mm": round((float((Ts[:3, 3] + Ts[:3, 0] * TIP)[2]) - holder_top) * 1000, 1),
                         "coin_up_mm": round((float(c[2]) - float(c0[2])) * 1000, 1),
                         "depth_mm": round(float(rel[0]) * 1000, 1),
                         "depth_settled_mm": round(float(rel_s[0]) * 1000, 1),
                         "lateral_mm": round(float(np.hypot(rel_s[1], rel_s[2])) * 1000, 1),
                         "left_holder": rose, "lost_at": lost_at, "lost_stage": stage if rose or lost_at else None,
                         "slip_mm": round(float(np.linalg.norm(last_rel - rel_s)) * 1000, 1),
                         "peak_rise_mm": round((peak - float(c0[2])) * 1000, 1)})
    # release: the last open of a hand holding the coin; the coin's pose in the bank's frame there
    rel = None
    for c in sorted(rows, key=lambda c: c["open"]):          # the last open of a hand still holding it
        if c["lost_at"] is None and c["open"] in F:
            k = c["open"] - 1 if (c["open"] - 1) in F else c["open"]
            o = obj(F[k], "coin0")
            if bank:
                Rb = qR(bank["quat"])
                loc = Rb.T @ (np.asarray(o["pos"], float) - np.asarray(bank["pos"], float))
                n_loc = Rb.T @ qR(o["quat"])[:, 2]
                yb = math.degrees(math.atan2(n_loc[1], n_loc[0])) % 180.0
                rel = {"step": k, "arm": c["arm"], "xyz_bank_mm": [round(float(v) * 1000, 1) for v in loc],
                       "normal_yaw_bank_deg": round(yb, 1),
                       # the slot's plane has its normal along the bank's x (successful coins: 0-3 deg)
                       "slot_plane_err_deg": round(min(yb, 180.0 - yb), 1),
                       "across_slot_mm": round(float(loc[0]) * 1000, 1),
                       "bottom_over_top_mm": round((float(o["pos"][2]) - 0.01425 - (float(bank["pos"][2]) + BANK_EXT[2]))
                                                   * 1000, 1)}
    last = F[steps[-1]]
    ce = np.asarray(obj(last, "coin0")["pos"], float)
    Re = qR(obj(last, "coin0")["quat"])
    flat = abs(float(Re[2, 2])) > 0.9
    if bank and in_bank(ce, bank):
        end = "in_bank" if ce[2] < float(bank["pos"][2]) + BANK_EXT[2] - 0.016 else "in_slot_high"
    elif np.linalg.norm(ce[:2] - c0[:2]) < 0.006 and ce[2] < c0[2] + 0.006:
        end = "in_holder"
    elif ce[2] < TABLE_Z + 0.02:
        end = "on_table_flat" if flat else "on_table"
    else:
        end = "elsewhere"
    return {"name": rep.get("name"), "layout_id": rep.get("layout_id"), "score": rep.get("score"),
            "steps": rep.get("n_actions"), "end": end, "closes": rows, "release": rel}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("replays")
    ap.add_argument("--reports", default=None, help="root with <name>/*_report.json.gz (adds the category)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--actions-dir", default=None, help="search here (recursively) for the replays' actions by file name")
    a = ap.parse_args(argv)
    out = []
    for p in sorted(glob.glob(os.path.join(a.replays, "*.json.gz"))):
        rep = json.load(gzip.open(p))
        cands = ([rep["actions"]] if os.path.exists(rep["actions"]) else []) + \
            glob.glob(os.path.join(a.reports or "", rep["name"], "*_actions.json.gz")) + \
            ([c for c in glob.glob(os.path.join(a.actions_dir, "**", os.path.basename(rep["actions"])), recursive=True)
              if c.split(os.sep)[-3:] == rep["actions"].split("/")[-3:]] if a.actions_dir else [])
        if not cands:
            print(f"{rep['name']}: no actions file, skipped")
            continue
        with gzip.open(cands[0], "rt") as f:
            acts = json.load(f)["actions"]
        e = episode(rep, acts)
        if a.reports:
            rp = glob.glob(os.path.join(a.reports, rep["name"], "*_report.json.gz"))
            if rp:
                r = json.load(gzip.open(rp[0]))
                e["result"] = str((r.get("result") or {}).get("result", ""))[:160]
                e["held_looks"] = [(h["step"], h.get("held"), h.get("arm")) for h in r["events"] if h["kind"] == "held_look"]
                e["consume"] = [(h["step"], h.get("arm"), h.get("method"), h.get("pose7_used")) for h in r["events"]
                                if h["kind"] == "consume" and h.get("label") == "coin0"]
        out.append(e)
    for e in out:
        print(f"{e['name']}: score {e['score']} end {e['end']} steps {e['steps']}")
        for c in e["closes"]:
            print("   ", {k: c[k] for k in ("arm", "close", "open", "tip_clear_mm", "tip_clear_settled_mm", "coin_up_mm", "depth_mm",
                                           "depth_settled_mm", "lateral_mm", "left_holder", "lost_at", "lost_stage",
                                           "slip_mm", "peak_rise_mm")})
        if e.get("held_looks"):
            print("    held_look", e["held_looks"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
