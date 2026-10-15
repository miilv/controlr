"""SO-101 server: owns the Feetech bus and the wrist camera on the machine the arm is plugged
into, and speaks ``protocol.py`` to ``So101Robot``.

Runs in LeRobot's environment (python >= 3.10, numpy, opencv, lerobot[feetech]); imports
nothing from controlr but ``protocol`` and ``arm_id`` (loaded by path, so no controlr install is
needed). ``--port auto`` finds the follower among /dev/ttyUSB* by its encoders (``arm_id``):

    python controlr/robot/so101/server.py --port auto --robot-id follower_v1 \
        --camera /dev/v4l/by-id/usb-...-video-index0
    python controlr/robot/so101/server.py --fake          # no hardware: kinematic arm + test image

Safety lives here, below anything the client sends:
* every goal is clamped into the arm's CALIBRATED range (LeRobot's calibration file) minus a
  margin, and every motion is slowed to ``--max-joint-speed``;
* power-up writes Goal_Position = Present_Position before enabling torque (no jump);
* during a motion the arm stops (goal frozen at the present position) when a joint lags its
  goal by more than ``track_stop_rad`` or draws more than ``current_stop_ma`` for
  ``stop_ticks`` ticks — it is pushing against something;
* Ctrl-C: first press freezes the arm in place (and refuses motion until a client reconnects),
  a second press within 3 s disables torque (THE ARM FALLS) and exits. The hard e-stop is the
  arm's power switch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import signal
import sys
import threading
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("so101_protocol", Path(__file__).with_name("protocol.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)  # type: ignore[union-attr]
_spec = importlib.util.spec_from_file_location("so101_arm_id", Path(__file__).with_name("arm_id.py"))
arm_id = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arm_id)  # type: ignore[union-attr]

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM = MOTORS[:5]
TICKS_PER_REV = 4095            # LeRobot's STS3215 resolution - 1 (DEGREES normalisation)
# folded rest pose of the rig (deg, read 2026-10-14 with the arm resting on its stops)
REST_DEG = (0.0, -95.0, 92.0, 40.0, -90.0)


def _log(msg: str) -> None:
    print(f"[so101 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


BUS_TRIES = 3                   # attempts per bus transaction
BUS_RETRY_S = 0.005


def _bus(fn, what: str, *args, **kwargs):
    """``fn(*args, **kwargs)`` with up to BUS_TRIES attempts. A single garbled status packet is
    common on the Feetech bus (LeRobot: ConnectionError "Incorrect status packet") and must not
    abort a move; the next transaction is usually fine. Re-raises the last failure."""
    for attempt in range(1, BUS_TRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001  (not KeyboardInterrupt/SystemExit)
            if attempt == BUS_TRIES:
                raise
            _log(f"bus {what} failed ({type(e).__name__}: {e}), retry {attempt}/{BUS_TRIES - 1}")
            time.sleep(BUS_RETRY_S)


# ---------------------------------------------------------------------------- arm backends
class LerobotArm:
    """The real arm through LeRobot's FeetechMotorsBus with the saved calibration."""

    fake = False

    def __init__(self, port: str, robot_id: str, calibration_dir: str | None = None,
                 allow_limp_start: bool = False) -> None:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        def make(port: str):
            return SO101Follower(SO101FollowerConfig(port=port, id=robot_id, use_degrees=True,
                                                     calibration_dir=Path(calibration_dir) if calibration_dir else None))

        robot = make(port)
        if not robot.calibration:
            raise SystemExit(f"no LeRobot calibration for id {robot_id!r} ({robot.calibration_fpath}); "
                             "run lerobot-calibrate first")
        if port == "auto":      # ttyUSB numbers swap on re-plug: find the follower by its encoders
            _log(f"--port auto: looking for {robot_id} among {arm_id.PORT_GLOB}")
            try:
                port = arm_id.find({robot_id: robot.calibration}, log=_log)[0][robot_id]
            except arm_id.ArmIdError as e:
                raise SystemExit(f"cannot find {robot_id}'s port: {e}\npass --port /dev/ttyUSBn") from None
            robot = make(port)
        self.port = port
        self.bus = robot.bus
        self.calibration = robot.calibration
        self.bus.connect()
        if not self.bus.is_calibrated:
            self.bus.port_handler.closePort()
            raise SystemExit(f"{port}: the motors' calibration registers differ from {robot_id}'s calibration "
                             "file: wrong port, or the motors hold another arm's calibration. Nothing was "
                             "written. scripts/so101_teleop_record.py prints which (and --restore-calibration "
                             "fixes it); never let lerobot overwrite them")
        held = [m for m, v in _bus(self.bus.sync_read, "read Torque_Enable", "Torque_Enable").items() if v]
        if held and not allow_limp_start:
            self.bus.disconnect(disable_torque=False)
            raise SystemExit(f"torque is ON ({', '.join(held)}): the arm may be holding itself up and would fall "
                             "when this server configures the motors. Stop the previous process with its "
                             "shutdown (it parks the arm on its rest stops), or support the arm and pass "
                             "--allow-limp-start")
        # LeRobot's SO101Follower.configure(), but the goal is set to the present position
        # BEFORE torque comes on: a stale Goal_Position register would make the arm jump.
        self.bus.disable_torque()
        self.bus.configure_motors()
        for m in MOTORS:
            self.bus.write("Operating_Mode", m, 0)          # position mode
            self.bus.write("P_Coefficient", m, 16)
            self.bus.write("I_Coefficient", m, 0)
            self.bus.write("D_Coefficient", m, 32)
        self.bus.write("Max_Torque_Limit", "gripper", 500)
        self.bus.write("Protection_Current", "gripper", 250)
        self.bus.write("Overload_Torque", "gripper", 25)
        self.torque = False

    def limits_rad(self) -> np.ndarray:
        out = []
        for m in ARM:
            c = self.calibration[m]
            mid = (c.range_min + c.range_max) / 2
            out.append([(c.range_min - mid) * 2 * math.pi / TICKS_PER_REV,
                        (c.range_max - mid) * 2 * math.pi / TICKS_PER_REV])
        return np.array(out)

    def read(self) -> tuple[np.ndarray, float]:
        pos = _bus(self.bus.sync_read, "read Present_Position", "Present_Position")
        q = np.radians([pos[m] for m in ARM])
        return q, float(pos["gripper"])

    def read_current_ma(self) -> np.ndarray:
        cur = _bus(self.bus.sync_read, "read Present_Current", "Present_Current", normalize=False)
        return np.array([abs(int(cur[m])) * 6.5 for m in MOTORS])     # STS3215: 6.5 mA per unit

    def read_health(self) -> tuple[list[float], float]:
        temp = _bus(self.bus.sync_read, "read Present_Temperature", "Present_Temperature", normalize=False)
        volt = _bus(self.bus.sync_read, "read Present_Voltage", "Present_Voltage", normalize=False)
        return [float(temp[m]) for m in MOTORS], float(np.mean([volt[m] for m in MOTORS])) / 10.0

    def write(self, q: np.ndarray, gripper: float) -> None:
        goal = {m: float(np.degrees(v)) for m, v in zip(ARM, q)}
        goal["gripper"] = float(gripper)
        _bus(self.bus.sync_write, "write Goal_Position", "Goal_Position", goal)

    def set_torque(self, on: bool) -> None:
        if on:
            q, g = self.read()
            self.write(q, g)
            _bus(self.bus.enable_torque, "torque on")
        else:
            _bus(self.bus.disable_torque, "torque off")
        self.torque = on

    def close(self) -> None:
        try:
            self.bus.disconnect(disable_torque=False)
        except Exception:  # noqa: BLE001
            pass


class FakeArm:
    """Kinematic stand-in: each joint follows its goal at a bounded speed (no gravity, no
    contact). ``block`` = {joint index: angle} stops a joint there (tests the stop rule)."""

    fake = True

    def __init__(self, q0=None, speed_rad_s: float = 3.0) -> None:
        self.q = np.radians(REST_DEG) if q0 is None else np.asarray(q0, float)
        self.g = 2.0
        self.goal_q, self.goal_g = self.q.copy(), self.g
        self.speed = speed_rad_s
        self.t = time.monotonic()
        self.torque = False
        self.block: dict[int, float] = {}
        self.sag = np.zeros(5)           # rad the loaded joints stay short of their goal
        # failure injection (tests): after ``fail_after`` more reads, the next ``fail_reads`` bus
        # reads raise like a garbled status packet (below the retry, as on the real bus)
        self.fail_reads = 0
        self.fail_after = 0

    def _bus_read(self) -> None:
        if self.fail_reads <= 0:
            return
        if self.fail_after > 0:
            self.fail_after -= 1
            return
        self.fail_reads -= 1
        raise ConnectionError("fake: Incorrect status packet")

    def _advance(self) -> None:
        now = time.monotonic()
        dt, self.t = now - self.t, now
        if not self.torque:
            return
        step = self.speed * dt
        goal = self.goal_q - self.sag          # gravity: the loaded joints stop short
        self.q = self.q + np.clip(goal - self.q, -step, step)
        for i, lim in self.block.items():
            self.q[i] = min(self.q[i], lim)
        self.g += float(np.clip(self.goal_g - self.g, -300 * dt, 300 * dt))

    def limits_rad(self) -> np.ndarray:
        return np.radians([[-110, 110], [-100, 100], [-97, 97], [-95, 95], [-157, 163]])

    def read(self) -> tuple[np.ndarray, float]:
        _bus(self._bus_read, "read Present_Position")
        self._advance()
        return self.q.copy(), self.g

    def read_current_ma(self) -> np.ndarray:
        _bus(self._bus_read, "read Present_Current")
        self._advance()
        lag = np.abs(np.append(self.goal_q - self.q, (self.goal_g - self.g) / 100))
        return np.minimum(lag * 4000.0, 3000.0)

    def read_health(self) -> tuple[list[float], float]:
        return [25.0] * 6, 12.5

    def write(self, q: np.ndarray, gripper: float) -> None:
        self._advance()
        self.goal_q, self.goal_g = np.asarray(q, float).copy(), float(gripper)

    def set_torque(self, on: bool) -> None:
        self._advance()
        self.goal_q, self.goal_g = self.q.copy(), self.g
        self.torque = on

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------- camera
class Camera:
    """Grabs frames continuously on a thread, so ``fresh()`` can wait for a frame exposed after
    a given time (no stale buffered frame of the arm mid-motion)."""

    def __init__(self, path: str | None, width: int, height: int, fps: int, *, mjpg: bool = True,
                 strict: bool = True) -> None:
        self.width, self.height, self.fps = width, height, fps
        self.fake = path is None
        self._lock = threading.Condition()
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self._stop = False
        if self.fake:
            self._frame = self._test_image()
            self._t = time.monotonic()
            return
        import cv2

        self.cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit(f"camera {path} gives no frames (another program using it?)")
        h, w = frame.shape[:2]
        if (w, h) != (width, height):
            if strict:
                raise SystemExit(f"camera {path} delivers {w}x{h}, asked for {width}x{height}")
            _log(f"camera {path} delivers {w}x{h} (asked {width}x{height})")
            self.width, self.height = w, h
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _test_image(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), 200, np.uint8)
        img[:: 40, :, :] = 90
        img[:, :: 40, :] = 90
        return img

    def _run(self) -> None:
        import cv2

        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._t = time.monotonic()
                self._lock.notify_all()

    def fresh(self, after: float, timeout_s: float = 2.0) -> tuple[np.ndarray, float]:
        """The first frame whose capture finished after ``after`` (monotonic seconds)."""
        if self.fake:
            return self._frame.copy(), time.monotonic()   # type: ignore[union-attr]
        deadline = time.monotonic() + timeout_s
        with self._lock:
            while self._t <= after:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise RuntimeError(f"no camera frame within {timeout_s:.1f} s")
                self._lock.wait(left)
            return self._frame.copy(), self._t      # type: ignore[union-attr]

    def close(self) -> None:
        self._stop = True
        if not self.fake:
            self.cap.release()


def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    try:
        import cv2
    except ImportError:              # --fake in controlr's venv (tests): Pillow instead
        import io

        from PIL import Image

        out = io.BytesIO()
        Image.fromarray(rgb).save(out, format="JPEG", quality=int(quality))
        return out.getvalue()
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


# ---------------------------------------------------------------------------- motion
def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


class Server:
    def __init__(self, arm, camera: Camera, *, extra_cameras: dict[str, Camera] | None = None,
                 control_hz: float = 50.0, max_joint_speed: float = 1.0,
                 limit_margin_rad: float = math.radians(3.0), settle_s: float = 0.15,
                 settle_tol_rad: float = math.radians(0.5), settle_max_s: float = 1.5,
                 correct_tol_rad: float = math.radians(0.4), max_offset_rad: float = math.radians(8.0),
                 correct_iters: int = 8, gripper_max: float = 100.0) -> None:
        self.arm, self.camera = arm, camera
        self.extra_cameras = dict(extra_cameras or {})       # fixed scene cameras, e.g. the laptop's
        self.hz = control_hz
        self.max_speed = max_joint_speed
        lim = arm.limits_rad()
        self.lim = np.stack([lim[:, 0] + limit_margin_rad, lim[:, 1] - limit_margin_rad], axis=1)
        self.settle_s, self.settle_tol, self.settle_max = settle_s, settle_tol_rad, settle_max_s
        self.frozen = False          # Ctrl-C: refuse motion until the next client connects
        # Sag compensation: the servos' low position gain (P=16) leaves the loaded joints several
        # degrees short of their goal (first jog: 5-13 mm low per move). After a move the goal is
        # offset by the measured error until the joints arrive (integral settle, capped); the
        # offset carries into the next move (no drop at its start), also after a stop (a stop
        # without it made the next lift lag 8.7 deg and stop again); a torque change clears it.
        self.correct_tol, self.max_offset, self.correct_iters = correct_tol_rad, max_offset_rad, correct_iters
        self.offset = np.zeros(5)
        self.gripper_max = float(gripper_max)      # never open the jaws wider than this (0..100)
        self.last_cmd: tuple[np.ndarray, float] | None = None
        self.t0 = time.monotonic()
        self.last_still = time.monotonic()

    # -- helpers
    def clamp(self, q) -> np.ndarray:
        return np.clip(np.asarray(q, float), self.lim[:, 0], self.lim[:, 1])

    def state(self) -> dict:
        q, g = self.arm.read()
        temp, volt = self.arm.read_health()
        return {"t": time.monotonic() - self.t0, "q": q, "gripper": g,
                "current_ma": self.arm.read_current_ma(), "torque": bool(self.arm.torque),
                "temp_c": temp, "voltage_v": volt}

    def info(self) -> dict:
        return {"motors": list(MOTORS), "limits_rad": self.lim, "gripper_range": [0.0, self.gripper_max],
                "camera": {"width": self.camera.width, "height": self.camera.height, "fps": self.camera.fps,
                           "fake": self.camera.fake},
                "extra_cameras": {n: {"width": c.width, "height": c.height, "fake": c.fake}
                                  for n, c in self.extra_cameras.items()},
                "fake": bool(self.arm.fake), "control_hz": self.hz, "rest_q": np.radians(REST_DEG),
                "max_joint_speed": self.max_speed}

    def observe(self, jpeg_quality: int = 92) -> dict:
        frame, t = self.camera.fresh(after=self.last_still)
        extra = {n: encode_jpeg(c.fresh(after=self.last_still)[0], jpeg_quality) for n, c in self.extra_cameras.items()}
        return {"jpeg": encode_jpeg(frame, jpeg_quality), "frame_t": t - self.t0, "state": self.state(),
                "extra_jpegs": extra}

    def _write(self, q, g) -> None:
        q = self.clamp(np.asarray(q, float))
        self.arm.write(q, g)
        self.last_cmd = (q, float(g))

    def hold(self) -> dict:
        """Keep the arm where it is: re-send the last command (incl. the sag offset), or the
        present position after a torque change."""
        if self.last_cmd is not None and self.arm.torque:
            self.arm.write(*self.last_cmd)
        else:
            q, g = self.arm.read()
            self._write(q, g)
        self.last_still = time.monotonic()
        return self.state()

    # -- motion
    def run_path(self, q_rows, g_rows, duration_s: float, stop: dict | None = None,
                 correct: bool = True) -> dict:
        """Stream a min-jerk timed path through joint waypoints; stop on blocking.

        correct=False skips the integral sag settle (a probe that must not push into contact)."""
        if self.frozen:
            raise RuntimeError("the arm is frozen (Ctrl-C on the server): reconnect to resume")
        if not self.arm.torque:
            self.arm.set_torque(True)
            self.offset[:] = 0.0
        stop = dict(stop or {})
        track_stop = float(stop.get("track_stop_rad", math.radians(12.0)))
        cur_stop = stop.get("current_stop_ma")      # one value or one per arm joint
        cur_stop = np.broadcast_to(np.asarray(cur_stop, float), (5,)) if cur_stop else None
        stop_ticks = int(stop.get("stop_ticks", 3))
        q0, g0 = self.arm.read()
        rows = [q0] + [self.clamp(r) for r in np.asarray(q_rows, float).reshape(-1, 5)]
        g_in = np.asarray(g_rows if g_rows is not None else [], float).reshape(-1)
        grip = [g0]
        for i in range(1, len(rows)):
            gv = g_in[i - 1] if i - 1 < len(g_in) else math.nan
            grip.append(grip[-1] if math.isnan(gv) else float(np.clip(gv, 0.0, self.gripper_max)))
        Q, G = np.array(rows), np.array(grip)
        seg = np.max(np.abs(np.diff(Q, axis=0)), axis=1) if len(Q) > 1 else np.zeros(0)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        # min-jerk peak speed = 1.875 x average: keep the PEAK under max_joint_speed
        duration = max(float(duration_s), 1.875 * total / self.max_speed,
                       0.3 if abs(G[-1] - G[0]) > 1 else 0.0, 1.0 / self.hz)
        peak_cur = np.zeros(6)
        peak_err = np.zeros(6)
        over = 0
        stopped, reason, joint = False, "", ""
        bus_err = ""
        q, g = q0, g0
        t_start = time.monotonic()
        ticks = 0
        dt = 1.0 / self.hz
        while True:
            t = time.monotonic() - t_start
            s = min_jerk(t / duration)
            if total > 1e-9:
                d = s * total
                k = int(min(np.searchsorted(cum, d, side="right") - 1, len(Q) - 2))
                f = 0.0 if seg[k] < 1e-12 else (d - cum[k]) / seg[k]
                q_goal = Q[k] + f * (Q[k + 1] - Q[k])
            else:
                q_goal = Q[-1]
            g_goal = G[0] + s * (G[-1] - G[0])
            try:
                self._write(q_goal + self.offset, g_goal)
                q, g = self.arm.read()
                cur = self.arm.read_current_ma()
            except Exception as e:  # noqa: BLE001  (failed after the retries: stop, don't raise)
                bus_err = f"bus error: {type(e).__name__}: {e}"
                stopped, reason = True, bus_err
                _log(f"{bus_err}: stopping the move")
                try:
                    if self.last_cmd is not None:
                        self.arm.write(*self.last_cmd)      # freeze at the last goal sent
                except Exception as e2:  # noqa: BLE001
                    _log(f"freeze after the bus error failed: {e2}")
                break
            err = np.abs(np.append(q_goal - q, 0.0))
            peak_cur = np.maximum(peak_cur, cur)
            peak_err = np.maximum(peak_err, err)
            bad = err[:5] > track_stop
            if cur_stop is not None:
                bad |= cur[:5] > cur_stop
            over = over + 1 if bad.any() else 0
            ticks += 1
            if over >= stop_ticks:
                score = err[:5] / track_stop
                if cur_stop is not None:
                    score = np.maximum(score, cur[:5] / cur_stop)
                i = int(np.argmax(np.where(bad, score, 0.0)))
                stopped, joint = True, ARM[i]
                reason = "tracking" if err[i] > track_stop else f"current {cur[i]:.0f} mA"
                self.arm.write(q, g)            # freeze where it is, no push (the sag offset is
                self.last_cmd = (q, float(g))   # hold() must not re-send the blocked goal
                break                           # kept: it compensates gravity, not the obstacle)
            if t >= duration:
                break
            time.sleep(max(0.0, dt - ((time.monotonic() - t_start) - t)))
        try:
            self._settle()
            if not stopped and correct:
                self._correct(Q[-1], G[-1])
            q, g = self.arm.read()
            state = self.state()
        except Exception as e:  # noqa: BLE001
            if not bus_err:
                raise
            # the bus is still failing: report the last good reading (the client gets a STOP)
            _log(f"no state after the bus error: {type(e).__name__}: {e}")
            self.last_still = time.monotonic()
            state = {"t": time.monotonic() - self.t0, "q": q, "gripper": g, "current_ma": np.zeros(6),
                     "torque": bool(self.arm.torque), "temp_c": [math.nan] * 6, "voltage_v": math.nan}
        # the gripper was told to close but stopped short: something is between the jaws
        blocked = bool(G[-1] < G[0] - 1 and g > G[-1] + 5)
        return {"state": state, "stopped": stopped, "stop_reason": reason, "stop_joint": joint,
                "ticks": ticks, "wall_s": time.monotonic() - t_start, "duration_s": duration,
                "peak_current_ma": peak_cur, "peak_track_err_rad": peak_err, "gripper_blocked": blocked,
                "final_err_rad": Q[-1] - q, "offset_rad": self.offset.copy()}

    def _correct(self, target: np.ndarray, g: float) -> None:
        for _ in range(self.correct_iters):
            q, _ = self.arm.read()
            e = target - q
            if float(np.max(np.abs(e))) < self.correct_tol:
                break
            self.offset = np.clip(self.offset + 0.8 * e, -self.max_offset, self.max_offset)
            self._write(target + self.offset, g)
            time.sleep(0.12)
        self._settle()

    def _settle(self) -> None:
        t0 = time.monotonic()
        prev, still_since = None, None
        while time.monotonic() - t0 < self.settle_max:
            q, g = self.arm.read()
            q = np.append(q, math.radians(g))     # the jaws too (0..100 read as "degrees")
            if prev is not None and float(np.max(np.abs(q - prev))) < self.settle_tol:
                still_since = still_since or time.monotonic()
                if time.monotonic() - still_since >= self.settle_s:
                    break
            else:
                still_since = None
            prev = q
            time.sleep(1.0 / self.hz)
        self.last_still = time.monotonic()

    def move_joints(self, q, gripper=None, duration_s: float = 3.0) -> dict:
        g = [math.nan if gripper is None else float(gripper)]
        return self.run_path([q], g, duration_s, stop={"track_stop_rad": math.radians(25.0)})

    def torque(self, on: bool) -> dict:
        self.arm.set_torque(bool(on))
        self.offset[:] = 0.0
        self.last_cmd = None
        return self.state()

    def rest(self) -> None:
        if self.arm.torque and not self.frozen:
            _log("moving to the rest pose")
            self.move_joints(np.radians(REST_DEG), None, 4.0)
        self.arm.set_torque(False)

    def handle(self, op: str, args: dict):
        if op == "info":
            return self.info()
        if op == "observe":
            return self.observe(int(args.get("jpeg_quality", 92)))
        if op == "state":
            return self.state()
        if op == "execute":
            return self.run_path(args["q"], args.get("gripper"), float(args.get("duration_s", 0.0)),
                                 args.get("stop"), bool(args.get("correct", True)))
        if op == "move_joints":
            return self.move_joints(args["q"], args.get("gripper"), float(args.get("duration_s", 3.0)))
        if op == "hold":
            return self.hold()
        if op == "torque":
            return self.torque(bool(args["on"]))
        raise ValueError(f"op {op!r} not handled")


def serve(server: Server, host: str, port: int, authkey: bytes, *, listener=None,
          max_clients: int | None = None, handle_sigint: bool = True) -> None:
    """Serve clients one after another until ``shutdown`` (or ``max_clients`` sessions).
    ``listener``: an already bound ``Listener`` (tests); ``handle_sigint`` needs the main thread."""
    last_int = [0.0]
    accept_fails: dict[str, list] = {}      # message -> [last logged, not logged since]

    def on_sigint(signum, frame):  # noqa: ARG001
        now = time.monotonic()
        if now - last_int[0] < 3.0:
            _log("second Ctrl-C: torque OFF (the arm falls), exiting")
            server.arm.set_torque(False)
            raise SystemExit(130)
        last_int[0] = now
        server.frozen = True
        try:
            server.hold()
        except Exception as e:  # noqa: BLE001
            _log(f"hold failed: {e}")
        _log("Ctrl-C: arm FROZEN in place; motion refused until a client reconnects. "
             "Ctrl-C again within 3 s = torque off + exit")

    if handle_sigint:
        signal.signal(signal.SIGINT, on_sigint)
    listener = listener if listener is not None else Listener((host, port), authkey=authkey)
    with listener:
        _log(f"listening on {listener.address} (fake arm: {server.arm.fake}, fake camera: {server.camera.fake})")
        served = 0
        while max_clients is None or served < max_clients:
            served += 1
            try:
                conn = listener.accept()
            except (OSError, EOFError) as e:
                # something probes the port every ~15 s ("bad message length"): once a minute
                last = accept_fails.setdefault(str(e), [-math.inf, 0])
                if time.monotonic() - last[0] >= 60.0:
                    more = f" (+{last[1]} since the last report)" if last[1] else ""
                    _log(f"accept failed: {e}{more}")
                    last[:] = [time.monotonic(), 0]
                else:
                    last[1] += 1
                continue
            server.frozen = False
            _log("client connected")
            try:
                while True:
                    try:
                        req = conn.recv()
                    except (EOFError, OSError):
                        _log("client gone: holding position")
                        server.hold()
                        break
                    started = time.perf_counter()
                    op = req.get("op")
                    if req.get("version") != P.PROTOCOL_VERSION:
                        conn.send(P.error(f"protocol version {req.get('version')} != {P.PROTOCOL_VERSION}"))
                        continue
                    if op == "close":
                        conn.send(P.ok({}, started))
                        break
                    if op == "shutdown":
                        server.rest()
                        conn.send(P.ok({}, started))
                        return
                    try:
                        conn.send(P.ok(server.handle(op, req.get("args") or {}), started))
                    except Exception as e:  # noqa: BLE001
                        try:
                            server.hold()
                        except Exception:  # noqa: BLE001
                            pass
                        conn.send(P.error(f"{type(e).__name__}: {e}", traceback.format_exc()))
            finally:
                conn.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", default="auto", help="serial port of the follower arm; auto = find it among "
                    "/dev/ttyUSB* by its encoders (arm_id.py)")
    ap.add_argument("--robot-id", default="follower_v1", help="LeRobot calibration id")
    ap.add_argument("--calibration-dir", default=None)
    ap.add_argument("--camera", default=None, help="V4L2 device path (/dev/v4l/by-id/...)")
    ap.add_argument("--camera2", default=None, help="a fixed scene camera (e.g. the laptop's, through "
                    "libcamera's v4l2-compat preload): V4L2 device path")
    ap.add_argument("--camera2-name", default="front")
    ap.add_argument("--camera2-width", type=int, default=1280)
    ap.add_argument("--camera2-height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--host", default=P.DEFAULT_HOST)
    ap.add_argument("--listen-port", type=int, default=P.DEFAULT_PORT)
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--max-joint-speed", type=float, default=1.0, help="rad/s, peak")
    ap.add_argument("--gripper-max", type=float, default=75.0, help="widest jaw opening allowed, 0..100")
    ap.add_argument("--fake", action="store_true", help="no hardware: kinematic arm and a test image")
    ap.add_argument("--print-info", action="store_true", help="print info + state as JSON and exit")
    ap.add_argument("--allow-limp-start", action="store_true",
                    help="start even if torque is on (the arm goes limp while the motors are configured)")
    a = ap.parse_args(argv)
    arm = FakeArm() if a.fake else LerobotArm(a.port, a.robot_id, a.calibration_dir, a.allow_limp_start)
    camera = Camera(None if a.fake and not a.camera else a.camera, a.width, a.height, a.fps)
    extra = {}
    if a.camera2:
        extra[a.camera2_name] = Camera(a.camera2, a.camera2_width, a.camera2_height, a.fps, mjpg=False, strict=False)
    server = Server(arm, camera, extra_cameras=extra, control_hz=a.control_hz, max_joint_speed=a.max_joint_speed,
                    gripper_max=a.gripper_max)
    try:
        if a.print_info:
            info, st = server.info(), server.state()
            print(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in {**info, **st}.items()},
                             indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
            return
        serve(server, a.host, a.listen_port, P.authkey_from_env())
    finally:
        try:
            if not a.print_info:
                server.rest()
        finally:
            camera.close()
            for c in extra.values():
                c.close()
            arm.close()


if __name__ == "__main__":
    main(sys.argv[1:])
