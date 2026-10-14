#!/usr/bin/env python
"""Leader -> follower teleop that writes the demo to disk as it goes (runs on the rig, LeRobot env).

The control loop is lerobot-teleoperate's (follower observation, leader action, send, 60 Hz
precise_sleep, default RANGE_M100_100 units, no max_relative_target), so it feels the same.
Recording is off the loop: a writer thread appends every tick to a JSONL (leader action +
follower Present_Position, both in lerobot's normalised units; degrees come later from the
calibration files) and saves the latest wrist frame every `--frame-every` s. lerobot-record
keeps joint data in memory until the episode ends; here a crash loses at most a few ticks, and a
serial error is logged and the tick retried.

    python so101_teleop_record.py --out ~/controlr-so101/demos/ball_demo_2

Stop: Ctrl-C, `touch <out>/STOP`, or `--seconds`. End with the leader in the rest pose: the
follower's torque goes off on exit. Nothing else may hold the serial ports (lerobot-teleoperate).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import cv2
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep

CALIB = Path.home() / ".cache/huggingface/lerobot/calibration"


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


def writer(out: Path, q: queue.Queue, cam: Camera, frame_every: float) -> None:
    """Drain tick records to the JSONL; attach and save a wrist frame every `frame_every` s."""
    log = open(out / "joints.jsonl", "a", buffering=1)
    last_frame = 0.0
    while True:
        rec = q.get()
        if rec is None:
            break
        frame, stamp = cam.latest()
        if frame is not None and rec["t"] - last_frame >= frame_every:
            name = f"{rec['t']:.3f}.jpg"
            tmp = out / "frames" / (name + ".tmp")
            cv2.imwrite(str(tmp), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            os.replace(tmp, out / "frames" / name)
            rec["frame"] = name
            rec["frame_age"] = round(rec["t"] - stamp, 3)
            last_frame = rec["t"]
        log.write(json.dumps(rec) + "\n")
        if q.empty():
            log.flush()
            os.fsync(log.fileno())
    log.close()


def soft_start(follower, leader, fps: int, rate: float, q: queue.Queue, timeout: float = 20.0) -> None:
    """Walk the follower from where it is to the leader's pose at `rate` units/s.

    The teleop loop sends the leader pose unclipped (as lerobot-teleoperate does): if the arms
    disagree at start, the follower would cross the whole gap in one tick.
    """
    cmd = {k: float(v) for k, v in follower.get_observation().items()}
    step = rate / fps
    t0 = time.perf_counter()
    print("soft start: follower walking to the leader's pose (hold the leader still)", flush=True)
    while time.perf_counter() - t0 < timeout:
        loop_start = time.perf_counter()
        target = leader.get_action()
        gap = max(abs(target[k] - cmd[k]) for k in cmd)
        if gap < 1.0:
            break
        for k in cmd:
            cmd[k] += max(-step, min(step, target[k] - cmd[k]))
        follower.send_action(cmd)
        precise_sleep(1 / fps - (time.perf_counter() - loop_start))
    else:
        raise RuntimeError(f"soft start: leader still {gap:.0f} units away after {timeout:.0f} s")
    q.put({"t": 0.0, "event": "soft_start_done", "s": round(time.perf_counter() - t0, 2)})
    print(f"soft start done in {time.perf_counter() - t0:.1f} s: teleop live", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--frame-every", type=float, default=0.1)
    ap.add_argument("--ramp-rate", type=float, default=25.0,
                    help="soft start speed, normalised units/s (200 = a joint's whole range)")
    ap.add_argument("--follower-port", default="/dev/ttyUSB0")
    ap.add_argument("--follower-id", default="follower_v1")
    ap.add_argument("--leader-port", default="/dev/ttyUSB1")
    ap.add_argument("--leader-id", default="leader_v1")
    ap.add_argument("--camera", default="/dev/video32")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for kind, typ, ident in (("robots", "so101_follower", args.follower_id),
                             ("teleoperators", "so101_leader", args.leader_id)):
        shutil.copy(CALIB / kind / typ / f"{ident}.json", out / f"calib_{typ}.json")
    stop_file = out / "STOP"
    stop_file.unlink(missing_ok=True)

    cam = Camera(args.camera, args.width, args.height)
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))
    follower = SO101Follower(SO101FollowerConfig(port=args.follower_port, id=args.follower_id))
    leader.connect()
    follower.connect()

    q: queue.Queue = queue.Queue()
    w = threading.Thread(target=writer, args=(out, q, cam, args.frame_every))
    w.start()
    q.put({"t": 0.0, "event": "start", "wall": time.time(), "args": vars(args)})
    print(f"recording to {out} for {args.seconds:.0f} s (touch {stop_file} to stop)", flush=True)

    t0 = time.perf_counter()
    errors = 0
    try:
        soft_start(follower, leader, args.fps, args.ramp_rate, q)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.seconds and not stop_file.exists():
            loop_start = time.perf_counter()
            try:
                obs = follower.get_observation()
                action = leader.get_action()
                follower.send_action(action)
                errors = 0
            except Exception as e:  # serial glitch: log, back off, retry
                errors += 1
                q.put({"t": round(loop_start - t0, 4), "event": "error", "error": f"{type(e).__name__}: {e}"})
                print(f"serial error {errors}: {e}", flush=True)
                if errors > 50:
                    raise
                time.sleep(0.02)
                continue
            q.put({"t": round(loop_start - t0, 4),
                   "leader": {k.removesuffix(".pos"): round(v, 2) for k, v in action.items()},
                   "follower": {k.removesuffix(".pos"): round(v, 2) for k, v in obs.items()}})
            precise_sleep(1 / args.fps - (time.perf_counter() - loop_start))
    except KeyboardInterrupt:
        pass
    finally:
        q.put({"t": round(time.perf_counter() - t0, 4), "event": "stop"})
        q.put(None)
        print("stopping: follower torque off", flush=True)
        for dev in (follower, leader):
            try:
                dev.disconnect()
            except Exception as e:
                print(f"disconnect: {e}", flush=True)
        w.join()
        cam.close()


if __name__ == "__main__":
    main()
