"""Measure the FK height error: lower the jaw tips onto the table and read FK z at contact.

The tool points straight down; from FK z = --start-mm the arm steps down --step-mm at a time
without the server's sag settle (it would push into the table). Contact = the measured TCP stops
following the command (the table holds the tips up) or the server's current/tracking stop. The
table is z = 0, so FK z at contact is the error at that spot. Each probe retracts to the start.

    uv run python scripts/so101_touch_probe.py --points 0.18,0 0.24,0 0.20,-0.08

Clear the table under the points first. Needs the server + tunnel (docs/SO101.md).
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from multiprocessing.connection import Client
from PIL import Image

from controlr.robot.so101 import kinematics as K
from controlr.robot.so101 import protocol as P

STOP = {"track_stop_rad": float(np.radians(15)), "current_stop_ma": [400, 600, 600, 400, 300], "stop_ticks": 3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="+", default=["0.18,0", "0.24,0", "0.20,-0.08"])
    ap.add_argument("--start-mm", type=float, default=30.0)
    ap.add_argument("--floor-mm", type=float, default=-80.0, help="never command below this FK z")
    ap.add_argument("--step-mm", type=float, default=2.0)
    ap.add_argument("--gripper", type=float, default=40.0)
    ap.add_argument("--out", default="runs/touch_probe")
    ap.add_argument("--port", type=int, default=7811)
    args = ap.parse_args()

    conn = Client(("127.0.0.1", args.port), authkey=P.authkey_from_env())

    def call(op, **kw):
        conn.send(P.request(op, **kw))
        return P.unwrap(conn.recv())

    lim = np.asarray(call("info")["limits_rad"])
    out = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    def q_now():
        return np.asarray(call("state")["q"], float)

    def go(p, correct=True, duration=0.0):
        R = K.reachable_rotation(p, K.R_DOWN)
        path, frac, why = K.line_path(q_now(), p, R, lim)
        if frac < 0.999:
            raise RuntimeError(f"{np.round(p * 1000)} mm unreachable ({why})")
        return call("execute", q=np.asarray(path), gripper=[args.gripper] * len(path),
                    duration_s=duration, stop=STOP, correct=correct)

    results = []
    try:
        for spec in args.points:
            x, y = (float(v) for v in spec.split(","))
            go(np.array([x, y, args.start_mm / 1000]))
            errs, contact = [], None
            z = args.start_mm
            while z - args.step_mm >= args.floor_mm:
                z -= args.step_mm
                r = go(np.array([x, y, z / 1000]), correct=False, duration=0.25)
                zm = float(K.fk(np.asarray(r["state"]["q"], float))[2, 3] * 1000)
                errs.append(zm - z)
                free = float(np.median(errs[:5]))
                print(f"  ({x * 1000:.0f},{y * 1000:.0f}) cmd z {z:6.1f}  meas {zm:6.1f}  lift {zm - z - free:+5.1f}"
                      + (f"  STOP {r['stop_reason']} {r['stop_joint']}" if r["stopped"] else ""), flush=True)
                if r["stopped"] or (len(errs) > 5 and zm - z - free > 3.0):
                    contact = {"x": x, "y": y, "z_fk_mm": zm, "cmd_mm": z, "stopped": r["stopped"],
                               "reason": r["stop_reason"], "q_deg": np.degrees(r["state"]["q"]).round(2).tolist()}
                    obs = call("observe")
                    Image.open(io.BytesIO(obs["jpeg"])).save(out / f"contact_{x * 1000:.0f}_{y * 1000:.0f}.jpg")
                    break
            go(np.array([x, y, args.start_mm / 1000]))
            print(f"contact at ({x * 1000:.0f},{y * 1000:.0f}): FK z = "
                  f"{contact['z_fk_mm']:.1f} mm" if contact else f"no contact down to {args.floor_mm} mm", flush=True)
            results.append(contact or {"x": x, "y": y, "z_fk_mm": None})
    finally:
        (out / "contacts.json").write_text(json.dumps(results, indent=1))
        print(f"saved {out}/contacts.json", flush=True)
        conn.close()


if __name__ == "__main__":
    main()
