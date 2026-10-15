#!/usr/bin/env python
"""Leader -> follower teleop that writes the demo to disk as it goes (runs on the rig, LeRobot env).

The control loop is lerobot-teleoperate's (follower observation, leader action, send, 60 Hz
precise_sleep, default RANGE_M100_100 units, no max_relative_target), so it feels the same.
Recording is off the loop: a writer thread appends every tick to a JSONL (leader action +
follower Present_Position, both in lerobot's normalised units; degrees come later from the
calibration files) and saves the latest wrist frame every `--frame-every` s. lerobot-record
keeps joint data in memory until the episode ends; here a crash loses at most a few ticks, and a
serial error is logged and the tick retried.

    python scripts/so101_teleop_record.py --out ~/controlr-so101/demos/ball_demo_2

Stop: Ctrl-C, `touch <out>/STOP`, or `--seconds`. End with the leader in the rest pose: the
follower's torque goes off on exit. Nothing else may hold the serial ports (lerobot-teleoperate).

Ports: ttyUSB numbers swap on re-plug and both adapters share one by-id name. `--follower-port`
/ `--leader-port` default to `auto`: every /dev/ttyUSB* is read and each arm found by its encoders
against its calibration file (`controlr/robot/so101/arm_id.py`, loaded by path from the repo or
the deployed copy: same layout, `<root>/scripts/` + `<root>/controlr/robot/so101/`). Explicit
ports are checked the same way and refused on a mismatch. An arm whose motors do not hold its
calibration file is refused (lerobot would offer to overwrite the motors); when arm_id knows
which arm it is, `--restore-calibration` writes the file into its motors (torque off only).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep

CALIB = Path.home() / ".cache/huggingface/lerobot/calibration"


def _load_arm_id():
    """arm_id.py by path (no controlr install in LeRobot's env): <root>/controlr/robot/so101/ next
    to <root>/scripts/ (repo and rig deploy), or next to this file (an old flat copy)."""
    here = Path(__file__).resolve().parent
    tried = [here.parent / "controlr/robot/so101/arm_id.py", here / "controlr/robot/so101/arm_id.py"]
    for path in tried:
        if path.exists():
            spec = importlib.util.spec_from_file_location("so101_arm_id", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod
    raise SystemExit("arm_id.py not found (deploy it with this script): " + ", ".join(map(str, tried)))


A = _load_arm_id()


def resolve_ports(roles: dict[str, tuple[str, str]], calibs: dict) -> tuple[dict[str, str], dict]:
    """roles {role: (arm id, port or 'auto')} -> ({arm id: port}, arm_id readings). Explicit ports
    are verified, never trusted; refuses (SystemExit) on any doubt. Writes nothing."""
    explicit = {ident: port for ident, port in roles.values() if port != "auto"}
    ports = set(explicit.values())
    if len(explicit) < len(roles):
        ports |= set(A.candidate_ports())
    print(f"identifying the arms on {', '.join(sorted(ports)) or '(no /dev/ttyUSB*)'}:", flush=True)
    try:
        mapping, readings = A.find(calibs, sorted(ports))
    except A.ArmIdError as e:
        raise SystemExit(f"arm identification failed: {e}\nNothing was written.") from None
    for role, (ident, port) in roles.items():
        if port != "auto" and mapping[ident] != port:
            other = {p: i for i, p in mapping.items()}.get(port)
            what = f"that is {other}" if other else "that is not one of the arms"
            raise SystemExit(f"--{role}-port {port}: {what}; {ident} is on {mapping[ident]} (scores above). "
                             f"Pass --{role}-port {mapping[ident]} or auto. Nothing was written.")
    return mapping, readings


def check_calibration(dev, calibs: dict, reading, restore: bool) -> None:
    """The motors must hold dev's calibration file (lerobot's is_calibrated). arm_id already knows
    the arm is dev (resolve_ports), so a mismatch means its motors hold the wrong calibration:
    say whose, and with ``restore`` write the file into them (torque off only)."""
    if dev.is_calibrated:
        return
    held = A.holder(reading[1], calibs)
    what = f"{held}'s calibration" if held else "a calibration that matches no file"
    if not restore:
        dev.bus.port_handler.closePort()
        cmd = " ".join([Path(sys.executable).name, *sys.argv, "--restore-calibration"])
        raise SystemExit(
            f"{dev.id} on {dev.config.port}: this arm IS {dev.id} (its encoders fit {dev.id}'s ranges, see "
            f"above), but its motors hold {what} (Homing_Offset / position limits in EEPROM; an earlier "
            f"swapped-port connect wrote it). Nothing was written. To write {dev.id}'s file back into this "
            f"arm's motors (arm still; it does not move):\n  {cmd}")
    torque = {m: v for m, v in dev.bus.sync_read("Torque_Enable").items() if v}
    if torque:
        dev.bus.port_handler.closePort()
        raise SystemExit(f"{dev.id}: --restore-calibration refused, torque is on ({', '.join(torque)}). "
                         "Nothing was written.")
    before = dev.bus.read_calibration()
    dev.bus.disable_torque()                    # already off; unlocks the EEPROM (Lock = 0)
    dev.bus.write_calibration(dev.calibration)
    # a new Homing_Offset moves Present_Position; a stale Goal_Position register would then point
    # somewhere else and configure() turns the follower's torque on: goal := present first
    present = dev.bus.sync_read("Present_Position", normalize=False)
    dev.bus.sync_write("Goal_Position", present, normalize=False)
    print(f"{dev.id} on {dev.config.port}: wrote {dev.id}'s calibration file into the motors (were: {what}):",
          flush=True)
    for m, c in dev.calibration.items():
        b = before[m]
        print(f"  {m}: homing_offset {b.homing_offset} -> {c.homing_offset}, range [{b.range_min}, "
              f"{b.range_max}] -> [{c.range_min}, {c.range_max}]", flush=True)
    if not dev.is_calibrated:
        dev.bus.port_handler.closePort()
        raise SystemExit(f"{dev.id}: the motors still differ from the file after the write; stop and check.")


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
            last_frame = rec["t"]
            name = f"{rec['t']:.3f}.jpg"
            try:  # a frame failure must not stop the joint log
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise RuntimeError("imencode failed")
                tmp = out / "frames" / (name + ".tmp")
                tmp.write_bytes(jpg.tobytes())
                os.replace(tmp, out / "frames" / name)
                rec["frame"] = name
                rec["frame_age"] = round(rec["t"] - stamp, 3)
            except Exception as e:
                rec["frame_error"] = f"{type(e).__name__}: {e}"
        log.write(json.dumps(rec) + "\n")
        if q.empty():
            log.flush()
            os.fsync(log.fileno())
    log.close()


def soft_start(follower, leader, fps: int, rate: float, q: queue.Queue, timeout: float = 20.0,
               max_gap: float = 30.0) -> None:
    """Walk the follower from where it is to the leader's pose at `rate` units/s.

    The teleop loop sends the leader pose unclipped (as lerobot-teleoperate does): if the arms
    disagree at start, the follower would cross the whole gap in one tick. A gap over `max_gap`
    units (any joint) is refused: walking it could sweep the arm into the rig.
    """
    cmd = {k: float(v) for k, v in follower.get_observation().items()}
    target = leader.get_action()
    far = {k.removesuffix(".pos"): round(target[k] - cmd[k]) for k in cmd
           if not k.startswith("gripper") and abs(target[k] - cmd[k]) > max_gap}
    if far:
        raise SystemExit(f"the leader is far from the follower's pose (leader - follower: {far}); put the "
                         "leader in the follower's pose (usually rest) and start again. Nothing moved.")
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
    ap.add_argument("--follower-port", default="auto", help="auto = find it by its encoders (arm_id)")
    ap.add_argument("--follower-id", default="follower_v1")
    ap.add_argument("--leader-port", default="auto", help="auto = find it by its encoders (arm_id)")
    ap.add_argument("--leader-id", default="leader_v1")
    ap.add_argument("--restore-calibration", action="store_true",
                    help="an arm identified by arm_id whose motors hold the wrong calibration: write its "
                         "file into the motors (torque must be off) instead of refusing")
    ap.add_argument("--camera", default="/dev/video32")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    files = {args.follower_id: CALIB / "robots/so101_follower" / f"{args.follower_id}.json",
             args.leader_id: CALIB / "teleoperators/so101_leader" / f"{args.leader_id}.json"}
    calibs = {ident: A.load_calib(f) for ident, f in files.items()}
    ports, readings = resolve_ports({"follower": (args.follower_id, args.follower_port),
                                     "leader": (args.leader_id, args.leader_port)}, calibs)
    args.follower_port, args.leader_port = ports[args.follower_id], ports[args.leader_id]

    out = Path(args.out).expanduser()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for (ident, f), typ in zip(files.items(), ("so101_follower", "so101_leader")):
        shutil.copy(f, out / f"calib_{typ}.json")
    stop_file = out / "STOP"
    stop_file.unlink(missing_ok=True)

    cam = Camera(args.camera, args.width, args.height)
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))
    follower = SO101Follower(SO101FollowerConfig(port=args.follower_port, id=args.follower_id))
    # lerobot's connect() answers a calibration mismatch by offering to write the file into the
    # motors (ENTER): on swapped ports that wrote the follower's calibration into the leader and
    # drove it. Same steps as connect() (no cameras here), but a mismatch is refused (or, with
    # --restore-calibration and the arm identified, fixed from the file).
    for dev in (leader, follower):
        dev.bus.connect()
        check_calibration(dev, calibs, readings[dev.config.port], args.restore_calibration)
        dev.configure()

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
