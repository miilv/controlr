"""Wire protocol between ``IsaacRobot`` (controlr's python) and the Isaac server.

WHY a separate module with no controlr / Isaac imports: the same file is
imported by two different interpreters — controlr's venv (client) and Isaac
Sim's bundled python 3.12 (server, see ``scripts/isaac_server.sh``). Keeping it
stdlib + numpy means neither side drags the other's dependencies along.

Transport is ``multiprocessing.connection`` on localhost with an HMAC authkey
(same pattern as PHANTOM ``phantom/sim/remote_policy.py``). Payloads are plain
dicts of str / int / float / bool / None / lists / numpy arrays — never custom
classes — so either side can be upgraded without pickle-compatibility traps.

Request:  ``{"op": <OPS member>, "args": {...}}``
Response: ``{"ok": True, "result": {...}, "server_s": float}`` or
          ``{"ok": False, "error": str, "traceback": str}``

Ops (all blocking; physics is PAUSED between requests — nothing steps while
the model thinks):

* ``info``        -> static scene facts: camera K / T_cam_base, gripper
                     calibration, table/bin geometry, physics dt, timings.
* ``reset``       args ``task, seed, params`` -> ``{"obs": OBS, "episode": {...}}``
* ``observe``     -> OBS (renders the D435 view; does not step physics)
* ``state``       -> STATE (no rendering)
* ``execute``     args ``q`` (N,6) rad joint targets, ``gripper`` (N,) target
                  opening in m (NaN = unchanged), ``durations`` (N,) s,
                  ``force_stop_n``, ``settle`` dict -> EXEC report dict
* ``check_goal``  -> ``{"success", "progress", "message", "metrics"}``
* ``close``       -> ends this client session (server keeps running)
* ``shutdown``    -> server process exits

OBS = ``{"rgb": HxWx3 uint8, "state": STATE, "render_s": float}``
STATE = ``{"t", "q"(6,), "tcp_pos"(3,), "tcp_rotvec"(3,), "tcp_fk_pos"(3,),
"gripper_m", "gripper_closure", "gripper_closed", "holding"}``
"""

from __future__ import annotations

import os
import time
from typing import Any

PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7801
AUTHKEY_ENV = "CONTROLR_ISAAC_AUTHKEY"
READY_MARKER = "CONTROLR_ISAAC_READY"

OPS = ("info", "reset", "observe", "state", "execute", "check_goal", "close", "shutdown")


def authkey_from_env(env: dict | None = None) -> bytes:
    """The shared secret both sides read from the environment.

    A fixed fallback would let any local process drive the robot; we still
    allow it for single-user dev boxes but make it explicit and loud."""
    value = (env if env is not None else os.environ).get(AUTHKEY_ENV, "")
    return (value or "controlr-isaac-dev").encode()


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
