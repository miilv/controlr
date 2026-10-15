"""Which SO-101 arm is on which serial port, from the motors themselves.

Both CH340 adapters of the rig share one /dev/serial/by-id name and /dev/ttyUSB0/1 swap on
re-plug, and the motors' Homing_Offset registers are no identity either: a wrong mapping lets
LeRobot write one arm's calibration into the other arm's EEPROM. What does identify an arm is
its encoders against each calibration file's range:

    actual = Present_Position (raw) + Homing_Offset held by the motor     # encoder, offset undone
    pred   = (actual - file.homing_offset) mod 4096                       # what the file would read

The right file puts every joint of an arm at a normal pose inside its calibrated
[range_min, range_max]; another arm's file puts several outside (its zero is elsewhere).

Stdlib only and no controlr imports: the recorder and the server load this file by path in
LeRobot's env on the rig (like ``protocol.py``). ``read_port`` / ``read_ports`` / ``find`` import
lerobot inside the function, so the pure part imports anywhere.
"""

from __future__ import annotations

import glob
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
RESOLUTION = 4096               # STS3215 encoder ticks per turn
TOL = 40                        # ticks (~3.5 deg) a joint may read past its range: an arm resting
                                # on a hard stop sits at the range's edge or a little past it
PORT_GLOB = "/dev/ttyUSB*"


class ArmIdError(RuntimeError):
    """No unambiguous arm -> port assignment."""


class Joint(NamedTuple):
    pred: int                   # the reading this calibration file would produce
    lo: int
    hi: int
    out: int                    # ticks outside [lo - tol, hi + tol]; 0 = inside


class Score(NamedTuple):
    out: int                    # joints out of range
    joints: dict[str, Joint]


def load_calib(path: str | Path) -> dict[str, dict]:
    """A LeRobot calibration file: {motor: {id, drive_mode, homing_offset, range_min, range_max}}."""
    return json.loads(Path(path).read_text())


def _get(c, key: str) -> int:
    """A calibration entry from a JSON dict or a LeRobot ``MotorCalibration``."""
    return int(c[key] if isinstance(c, Mapping) else getattr(c, key))


def score(raw: Mapping[str, int], motor_offsets: Mapping[str, int], calib: Mapping, tol: int = TOL) -> Score:
    """Out-of-range joint count (+ per-joint detail) if the arm read as ``raw`` with
    ``motor_offsets`` in its motors were the arm ``calib`` belongs to. Joints missing from
    ``calib`` are skipped."""
    joints = {}
    for m in raw:
        if m not in calib:
            continue
        c = calib[m]
        pred = (int(raw[m]) + int(motor_offsets[m]) - _get(c, "homing_offset")) % RESOLUTION
        lo, hi = _get(c, "range_min"), _get(c, "range_max")
        out = max(lo - tol - pred, pred - hi - tol, 0)
        joints[m] = Joint(pred, lo, hi, out)
    return Score(sum(1 for j in joints.values() if j.out), joints)


Reading = tuple[Mapping[str, int], Mapping[str, int]]          # (raw Present_Position, Homing_Offset)


def assign(readings: Mapping[str, Reading], calibs: Mapping[str, Mapping], tol: int = TOL) -> dict[str, str]:
    """{arm id: port}: the one assignment of every calibration to a distinct port under which no
    joint is out of range. Extra ports (an unknown arm, the other arm when only one calibration
    is given) are left out. Raises ``ArmIdError`` when no such assignment exists (an arm missing,
    unknown, or far outside its range) or when more than one does (ambiguous)."""
    ids, ports = list(calibs), list(readings)
    table = {(p, i): score(*readings[p], calibs[i], tol).out for p in ports for i in ids}
    if len(ports) < len(ids):
        raise ArmIdError(f"{len(ports)} readable port(s) for {len(ids)} arm(s) ({', '.join(ids)})\n"
                         + format_scores(readings, calibs, tol))
    fits, best, best_out = [], None, None
    for perm in itertools.permutations(ports, len(ids)):
        total = sum(table[p, i] for p, i in zip(perm, ids))
        if total == 0:
            fits.append(dict(zip(ids, perm)))
        if best_out is None or total < best_out:
            best, best_out = dict(zip(ids, perm)), total
    if len(fits) == 1:
        return fits[0]
    if fits:
        alts = "; ".join(", ".join(f"{i}={p}" for i, p in f.items()) for f in fits)
        raise ArmIdError(f"ambiguous: {len(fits)} assignments fit equally ({alts}). Move the arms into "
                         "different poses (e.g. one at rest, one raised) or pass the ports explicitly\n"
                         + format_scores(readings, calibs, tol))
    detail = ", ".join(f"{i}={p} ({table[p, i]} out)" for i, p in best.items())
    raise ArmIdError(f"no assignment fits: the best ({detail}) still has {best_out} joint(s) out of range. "
                     "An arm unplugged/unpowered, an unknown arm, or an arm far outside its calibrated "
                     "range (move it to its rest pose)\n" + format_scores(readings, calibs, tol))


def holder(motor_offsets: Mapping[str, int], calibs: Mapping[str, Mapping]) -> str | None:
    """The id whose calibration file's homing offsets the motors hold (None: none of them).
    Says what is IN the motors, not which arm it is (that is ``score``/``assign``)."""
    for ident, calib in calibs.items():
        if all(int(motor_offsets[m]) == _get(calib[m], "homing_offset") for m in motor_offsets if m in calib):
            return ident
    return None


def format_scores(readings: Mapping[str, Reading], calibs: Mapping[str, Mapping], tol: int = TOL) -> str:
    """One line per (port, calibration): out-of-range count and the offending joints."""
    lines = []
    for p in readings:
        held = holder(readings[p][1], calibs)
        for i in calibs:
            s = score(*readings[p], calibs[i], tol)
            bad = ", ".join(f"{m} {j.pred} not in [{j.lo}, {j.hi}]" for m, j in s.joints.items() if j.out)
            lines.append(f"  {p} as {i}: {s.out}/{len(s.joints)} joints out of range" + (f" ({bad})" if bad else ""))
        lines.append(f"  {p} motors hold: " + (f"{held}'s calibration" if held else "no known calibration"))
    return "\n".join(lines)


# ------------------------------------------------------------------ LeRobot side (rig only)
def read_port(port: str) -> tuple[dict[str, int], dict[str, int]]:
    """Raw Present_Position and Homing_Offset of the 6 motors on ``port``. Opens the port without
    LeRobot's handshake or calibration, reads, closes it; writes nothing."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(port=port, motors={m: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100)
                                              for i, m in enumerate(MOTORS, 1)})
    bus.connect(handshake=False)
    try:
        raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        offsets = bus.sync_read("Homing_Offset", normalize=False, num_retry=2)
    finally:
        bus.port_handler.closePort()
    return {m: int(raw[m]) for m in MOTORS}, {m: int(offsets[m]) for m in MOTORS}


def candidate_ports() -> list[str]:
    return sorted(glob.glob(PORT_GLOB))


def read_ports(ports: list[str]) -> tuple[dict[str, Reading], dict[str, str]]:
    """``read_port`` on each port: (readings, {port: error} for the ones that did not answer)."""
    readings, errors = {}, {}
    for p in ports:
        try:
            readings[p] = read_port(p)
        except Exception as e:  # noqa: BLE001  (not a Feetech bus, unpowered arm, busy port)
            errors[p] = f"{type(e).__name__}: {e}"
    return readings, errors


def find(calibs: Mapping[str, Mapping], ports: list[str] | None = None,
         log=print) -> tuple[dict[str, str], dict[str, Reading]]:
    """Read ``ports`` (default: every /dev/ttyUSB*), log the scores, ``assign``. Returns
    (mapping, readings); raises ``ArmIdError``."""
    ports = candidate_ports() if ports is None else ports
    if not ports:
        raise ArmIdError(f"no serial ports match {PORT_GLOB}: arms unplugged?")
    readings, errors = read_ports(ports)
    for p, e in errors.items():
        log(f"  {p}: no answer ({e})")
    mapping = assign(readings, calibs)          # its error carries the score table
    log(format_scores(readings, calibs))
    log("  arms: " + ", ".join(f"{i} = {p}" for i, p in mapping.items()))
    return mapping, readings
