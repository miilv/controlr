#!/usr/bin/env python
"""Leader -> follower teleop that writes the demo to disk as it goes (runs on the rig, LeRobot env).

lerobot-record keeps joint data in memory until the episode ends; one serial glitch at 90 s
lost a whole demo. Here every tick is one flushed JSONL line (leader + follower joints in
DEGREES, i.e. controlr's joint zero) and the wrist frame is saved every `--frame-every` s, so a
crash loses at most the current tick. Serial errors are logged and the tick is retried.

    python so101_teleop_record.py --out ~/controlr-so101/demos/ball_demo_2 --seconds 150

Stop: Ctrl-C, `touch <out>/STOP`, or the time limit. End with the leader in the rest pose:
the follower's torque goes off on exit.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import cv2
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig


class Camera:
    def __init__(self, path: str, width: int, height: int) -> None:
        self.cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {path} did not open")
        self.frame = None
        self.stamp = 0.0
        self.lock = threading.Lock()
        self.alive = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while self.alive:
            ok, img = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.stamp = img, time.time()

    def latest(self):
        with self.lock:
            return self.frame, self.stamp

    def close(self) -> None:
        self.alive = False
        time.sleep(0.1)
        self.cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=150.0)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--frame-every", type=float, default=0.1)
    ap.add_argument("--follower-port", default="/dev/ttyUSB1")
    ap.add_argument("--follower-id", default="follower_v1")
    ap.add_argument("--leader-port", default="/dev/ttyUSB0")
    ap.add_argument("--leader-id", default="leader_v1")
    ap.add_argument("--camera", default="/dev/video32")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max-relative-target", type=float, default=5.0)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    stop_file = out / "STOP"
    stop_file.unlink(missing_ok=True)
    log = open(out / "joints.jsonl", "a", buffering=1)

    def note(**kw) -> None:
        log.write(json.dumps({"t": round(time.time(), 4), **kw}) + "\n")
        log.flush()
        os.fsync(log.fileno())

    cam = Camera(args.camera, args.width, args.height)
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id, use_degrees=True))
    follower = SO101Follower(SO101FollowerConfig(
        port=args.follower_port, id=args.follower_id, use_degrees=True,
        max_relative_target=args.max_relative_target))
    leader.connect()
    follower.connect()
    note(event="start", args=vars(args))
    print(f"recording to {out} for {args.seconds:.0f} s (touch {stop_file} to stop)", flush=True)

    t0 = time.time()
    last_frame = 0.0
    errors = 0
    period = 1.0 / args.hz
    try:
        while time.time() - t0 < args.seconds and not stop_file.exists():
            tick = time.time()
            try:
                action = leader.get_action()
                sent = follower.send_action(action)
                obs = follower.bus.sync_read("Present_Position")
                errors = 0
            except Exception as e:  # serial glitch: log, back off, retry
                errors += 1
                note(event="error", n=errors, error=f"{type(e).__name__}: {e}")
                print(f"serial error {errors}: {e}", flush=True)
                if errors > 50:
                    raise
                time.sleep(0.05)
                continue
            rec = {"leader": {k.removesuffix(".pos"): round(v, 2) for k, v in action.items()},
                   "sent": {k.removesuffix(".pos"): round(v, 2) for k, v in sent.items()},
                   "follower": {k: round(v, 2) for k, v in obs.items()}}
            frame, stamp = cam.latest()
            if frame is not None and tick - last_frame >= args.frame_every:
                name = f"{tick - t0:08.3f}.jpg"
                tmp = out / "frames" / (name + ".tmp")
                cv2.imwrite(str(tmp), frame, [cv2.IMWRITE_JPEG_QUALITY, 90], )
                os.replace(tmp, out / "frames" / name)
                rec["frame"] = name
                rec["frame_age"] = round(tick - stamp, 3)
                last_frame = tick
            note(**rec)
            time.sleep(max(0.0, period - (time.time() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        note(event="stop", elapsed=round(time.time() - t0, 2))
        print("stopping: follower torque off", flush=True)
        for dev in (follower, leader):
            try:
                dev.disconnect()
            except Exception as e:
                print(f"disconnect: {e}", flush=True)
        cam.close()


if __name__ == "__main__":
    main()
