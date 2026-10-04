"""Wire protocol between controlr (``RoboDojoRobot``) and the shim that runs inside
RoboDojo's eval client (``shim/deploy.py``).

WHY this direction: RoboDojo's eval client owns the episode (layouts, step limit, scoring,
result json, videos) and calls ``XPolicyLab/policy/<name>/deploy.py::eval_one_episode`` once
per episode with the live ``TASK_ENV``; the sim is paused between its calls. Our shim
connects to ``controlr robodojo-serve`` (a ``multiprocessing.connection`` Listener on
localhost, HMAC authkey), announces the episode, and then SERVES robot requests until
controlr says the episode is over. So the shim is the connecting side but the request
handler; controlr sends requests.

Stdlib + numpy only: the same file is imported by controlr's venv and by RoboDojo's conda
python (the shim). Payloads are plain dicts / lists / numpy arrays.

shim -> controlr, once per episode: ``{"op": "hello", "episode": EPISODE}``
controlr -> shim: ``{"op": <OPS>, "args": {...}}``; shim -> controlr: ``ok(...)`` / ``error(...)``

Ops:
* ``observe``     -> OBS (renders all cameras)
* ``execute``     args ``steps``: list of ``{arm: {"flange": [x y z qw qx qy qz] | None,
                  "grip": 0..1 | None}}`` (world frame, RoboDojo's flange = link6 pose;
                  grip 1 = open) -> EXEC
* ``check_goal``  -> ``{"success", "ended", "score", "steps_used", "step_lim"}``
* ``done``        args ``outcome`` -> ends the episode on the shim side (an episode that
                  RoboDojo has not ended is marked failed, as RoboDojo's own policies do)

EPISODE = ``{"task", "layout_id", "seed", "instruction", "step_lim", "arms", "cameras",
"control_hz", "robodojo_root"}``
OBS = ``{"images": {cam: HxWx3 uint8}, "K": {cam: 3x3}, "T_world_cam": {cam: 4x4}
(OpenCV camera axes), "arms": {arm: {"q": (6,), "flange": (7,), "grip": float}},
"steps_used", "step_lim", "ended", "success"}``
EXEC = ``{"steps": [{"arms": {arm: {"status", "waypoints"}}, "env_steps"}], "env_steps",
"ended", "success", "sim_s", "wall_s"}``
"""

from __future__ import annotations

import os
import time
from typing import Any

PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7831
AUTHKEY_ENV = "CONTROLR_ROBODOJO_AUTHKEY"
HOST_ENV = "CONTROLR_ROBODOJO_HOST"
PORT_ENV = "CONTROLR_ROBODOJO_PORT"

OPS = ("observe", "execute", "check_goal", "done")


def authkey_from_env(env: dict | None = None) -> bytes:
    """Shared secret; the run script sets a random one per run."""
    value = (env if env is not None else os.environ).get(AUTHKEY_ENV, "")
    return (value or "controlr-robodojo-dev").encode()


def address_from_env(env: dict | None = None) -> tuple[str, int]:
    e = env if env is not None else os.environ
    return e.get(HOST_ENV, DEFAULT_HOST), int(e.get(PORT_ENV, DEFAULT_PORT))


def request(op: str, **args: Any) -> dict:
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {OPS}")
    return {"op": op, "args": args, "version": PROTOCOL_VERSION}


def ok(result: Any, started: float) -> dict:
    return {"ok": True, "result": result, "server_s": time.perf_counter() - started}


def error(message: str, tb: str = "") -> dict:
    return {"ok": False, "error": message, "traceback": tb}


class RemoteError(RuntimeError):
    def __init__(self, message: str, tb: str = ""):
        super().__init__(message + (f"\n--- shim traceback ---\n{tb}" if tb else ""))
        self.server_traceback = tb


def unwrap(response: dict) -> Any:
    if not isinstance(response, dict) or "ok" not in response:
        raise RemoteError(f"malformed response: {type(response).__name__}")
    if not response["ok"]:
        raise RemoteError(response.get("error", "unknown error"), response.get("traceback", ""))
    return response["result"]
