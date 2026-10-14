"""Replay a recorded leader-arm demo on the follower in JOINT space (no FK / IK).

Reads a so101_teleop_record.py recording, converts the follower's joint readings to degrees with
the recording's calibration file, and streams them through the server's `execute` (its stop
rules apply) in segments split at `--cuts` (the demo's gripper close), `--slow` times slower. The gripper
replays the leader's command (the follower stalls on the object, as in the demo).

    uv run python scripts/so101_replay.py runs/demos/ball_demo_4 --dry-run
    uv run python scripts/so101_replay.py runs/demos/ball_demo_4 --until 5.4      # approach, close, lift

Put the object where it was in the demo. The arm first moves (joint space) to the demo's first
pose.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from controlr.robot.so101 import protocol as P

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
STOP = {"track_stop_rad": float(np.radians(15)), "current_stop_ma": [400, 600, 600, 400, 300], "stop_ticks": 3}


def load(demo: Path, every: float):
    cal = json.loads((demo / "calib_so101_follower.json").read_text())

    def deg(name, v):
        lo, hi = cal[name]["range_min"], cal[name]["range_max"]
        return ((v + 100) / 200 * (hi - lo) + lo - (lo + hi) / 2) * 360 / 4095

    rows = [json.loads(line) for line in (demo / "joints.jsonl").read_text().splitlines()]
    rows = [r for r in rows if "follower" in r]
    out, last = [], -1e9
    for r in rows:
        if r["t"] - last >= every:
            out.append((r["t"], np.radians([deg(j, r["follower"][j]) for j in ARM]), float(r["leader"]["gripper"])))
            last = r["t"]
    return out


def segments(samples, cuts):
    """Split at the given demo times (e.g. where the jaws start and stop closing): the server
    times a path by joint distance, so a close with the arm still must be its own segment."""
    segs, cur = [], [samples[0]]
    edges = sorted(cuts)
    for s in samples[1:]:
        cur.append(s)
        if edges and s[0] >= edges[0]:
            segs.append(cur)
            cur = [s]
            edges.pop(0)
    if len(cur) > 1:
        segs.append(cur)
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", type=Path)
    ap.add_argument("--slow", type=float, default=2.0, help="time scale vs the demo")
    ap.add_argument("--every", type=float, default=0.1, help="waypoint spacing, s of demo time")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--until", type=float, default=1e9)
    ap.add_argument("--cuts", default="3.53,3.94", help="segment boundaries, demo seconds")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--port", type=int, default=7811)
    args = ap.parse_args()

    samples = [s for s in load(args.demo, args.every) if args.t_from <= s[0] <= args.until]
    segs = segments(samples, [float(c) for c in args.cuts.split(",") if c])
    for i, s in enumerate(segs):
        q0, q1 = np.degrees(s[0][1]), np.degrees(s[-1][1])
        print(f"seg {i}: t {s[0][0]:5.2f}-{s[-1][0]:5.2f}  grip {s[0][2]:5.1f}->{s[-1][2]:5.1f}  "
              f"q {np.round(q0, 1)} -> {np.round(q1, 1)}", flush=True)
    if args.dry_run:
        return

    conn = Client(("127.0.0.1", args.port), authkey=P.authkey_from_env())

    def call(op, **kw):
        conn.send(P.request(op, **kw))
        return P.unwrap(conn.recv())

    try:
        first = segs[0][0]
        print("to the demo's first pose", flush=True)
        r = call("execute", q=np.asarray([first[1]]), gripper=[first[2]], duration_s=4.0, stop=STOP)
        if r["stopped"]:
            raise SystemExit(f"STOP on the way to the first pose: {r['stop_reason']} {r['stop_joint']}")
        for i, s in enumerate(segs):
            q = np.asarray([x[1] for x in s[1:]])
            g = [x[2] for x in s[1:]]
            dur = (s[-1][0] - s[0][0]) * args.slow
            r = call("execute", q=q, gripper=g, duration_s=dur, stop=STOP)
            st = r["state"]
            print(f"seg {i} done: grip {st['gripper']:.1f} (cmd {g[-1]:.1f})"
                  f"{'  BLOCKED (holding something)' if r['gripper_blocked'] else ''}"
                  f"  err max {np.degrees(np.max(np.abs(r['final_err_rad']))):.1f} deg"
                  f"{'  STOP ' + r['stop_reason'] + ' ' + r['stop_joint'] if r['stopped'] else ''}", flush=True)
            if r["stopped"]:
                raise SystemExit("stopped: the arm froze where it is")
            time.sleep(0.3)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
