"""Wire protocol between ``So101Robot`` (controlr's python) and the SO-101 server on the rig.

WHY stdlib + numpy only: the server runs in LeRobot's own environment on the machine the arm
is plugged into (``scripts/so101_server.sh``), the client in controlr's venv wherever the LLM
router is reachable — over an ssh tunnel. Neither side drags the other's dependencies along.

Transport: ``multiprocessing.connection`` with an HMAC authkey; the server listens on
localhost only (reach it through ``ssh -L``). Payloads are plain dicts of str / int / float /
bool / None / bytes / lists / numpy arrays.

Request:  ``{"op": <OPS member>, "args": {...}, "version": int}``
Response: ``{"ok": True, "result": {...}, "server_s": float}`` or
          ``{"ok": False, "error": str, "traceback": str}``

Ops (all blocking; the arm holds its last target between requests):

* ``info``     -> ``{"motors", "limits_rad" (5, 2), "gripper_range", "camera": {"width",
                  "height", "fps"} | None, "fake", "control_hz", "rest_q"}``
* ``observe``  args ``jpeg_quality`` -> OBS: a FRESH frame (captured after the arm settled) +
                  STATE
* ``state``    -> STATE (no image)
* ``execute``  args ``q`` (N, 5) rad arm waypoints, ``gripper`` (N,) 0..100 (NaN = unchanged),
                  ``duration_s`` (whole path, min-jerk timed), ``stop`` dict (tracking /
                  current thresholds) -> EXEC
* ``move_joints`` args ``q`` (5,), ``gripper``, ``duration_s`` -> EXEC (joint-space move for
                  reset / rest, not used by the turn loop)
* ``hold``     -> freeze at the present position
* ``torque``   args ``on`` (bool) -> STATE. ``on=False`` makes the arm go limp: it falls unless
                  supported (used by the hand-guided calibration)
* ``close``    -> ends this client session (server keeps running)
* ``shutdown`` -> server moves to ``rest_q`` slowly, disables torque, exits

STATE = ``{"t", "q" (5,) rad, "gripper" 0..100, "current_ma" (6,), "torque", "temp_c" (6,),
"voltage_v"}``
OBS = ``{"jpeg": bytes, "frame_t", "state": STATE}``
EXEC = ``{"state": STATE, "stopped": bool, "stop_reason": str, "stop_joint": str, "ticks",
"wall_s", "peak_current_ma" (6,), "peak_track_err_rad" (6,), "gripper_blocked": bool}``
"""

from __future__ import annotations

import os
import time
from typing import Any

PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7811
AUTHKEY_ENV = "CONTROLR_SO101_AUTHKEY"

OPS = ("info", "observe", "state", "execute", "move_joints", "hold", "torque", "close", "shutdown")


def authkey_from_env(env: dict | None = None) -> bytes:
    """The shared secret both sides read from the environment (the dev fallback is for a
    single-user box; the port is localhost-only either way)."""
    value = (env if env is not None else os.environ).get(AUTHKEY_ENV, "")
    return (value or "controlr-so101-dev").encode()


def request(op: str, **args: Any) -> dict:
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {OPS}")
    return {"op": op, "args": args, "version": PROTOCOL_VERSION}


def ok(result: Any, started: float) -> dict:
    return {"ok": True, "result": result, "server_s": time.perf_counter() - started}


def error(message: str, tb: str = "") -> dict:
    return {"ok": False, "error": message, "traceback": tb}


class RemoteError(RuntimeError):
    """The server raised while handling a request; carries its traceback."""

    def __init__(self, message: str, tb: str = ""):
        super().__init__(message + (f"\n--- server traceback ---\n{tb}" if tb else ""))
        self.server_traceback = tb


def unwrap(response: dict) -> Any:
    if not isinstance(response, dict) or "ok" not in response:
        raise RemoteError(f"malformed response: {type(response).__name__}")
    if not response["ok"]:
        raise RemoteError(response.get("error", "unknown error"), response.get("traceback", ""))
    return response["result"]
