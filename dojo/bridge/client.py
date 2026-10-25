"""Client for robot-harness's robot servers: the ``harness/robot_client_v2.py`` protocol, stdlib only.

One JSON object per request, one per reply, each terminated by a newline, one request per connection, over the
server's Unix socket ``$BENCH/run/<server-id>.sock``. Every reply carries ``ok``, ``cmd``, ``t_wall``, ``step``,
``steps_left``, ``ended``; errors are ``{"ok": false, "error", "hint"}``. Verbs are methods
(``c.obs(cameras=[])``, ``c.move_to(arm="left", pos=[...])``), exactly like ``robot_client_v2.Robot``, so the same
object can be handed to robot-harness skills (``skill_robodojo.run(robot, **args)``) as their ``robot``.

The protocol mirrors robot_client_v2 (Mikhail Sannikov, robot-harness, 25 Oct 2026); it is re-implemented here
rather than imported so the bridge needs no robot-harness checkout on ``sys.path`` (the transport is replaceable:
``RobotServerClient(transport=fn)`` with ``fn(request_dict) -> reply_dict``, used by the tests).
"""

from __future__ import annotations

import json
import os
import socket
import time

BENCH_DEFAULT = "/home/anywherevla/data2/bench"


class ServerError(RuntimeError):
    """The server answered ``ok: false`` (or did not answer)."""

    def __init__(self, verb: str, reply: dict):
        self.verb, self.reply = verb, reply
        super().__init__(f"{verb}: {reply.get('error')} (hint: {str(reply.get('hint') or '')[:300]})")


def sock_path(server_id: str, bench: str | None = None) -> str:
    return os.path.join(bench or os.environ.get("BENCH", BENCH_DEFAULT), "run", server_id + ".sock")


def unix_request(path: str, req: dict, timeout: float = 1800.0) -> dict:
    """robot_client_v2.request: connect, send one line, read until the newline."""
    if not os.path.exists(path):
        return {"ok": False, "error": f"no server socket {path}", "hint": "robot list; robot-start-strict ..."}
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    buf = b""
    try:
        s.connect(path)
        s.sendall((json.dumps(req) + "\n").encode())
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
    except (ConnectionRefusedError, FileNotFoundError) as e:
        return {"ok": False, "error": f"server not answering at {path}: {e}", "hint": "see run/<id>/server.log"}
    except socket.timeout:
        return {"ok": False, "error": f"timeout after {timeout}s", "hint": "the command may still be running"}
    finally:
        s.close()
    if not buf:
        return {"ok": False, "error": "empty reply", "hint": "the server may have exited; see run/<id>/server.log"}
    return json.loads(buf.decode())


class RobotServerClient:
    """Verbs of one robot server. ``call`` returns the raw reply (robot_client_v2 semantics, never raises);
    ``ok`` raises ``ServerError`` on ``ok: false``. ``on_request(verb)`` runs before every request (the backend
    uses it to flush buffered joint steps before a foreign skill moves the robot)."""

    def __init__(self, server_id: str | None = None, sock: str | None = None, bench: str | None = None,
                 timeout: float = 1800.0, transport=None, log=None):
        if transport is None and sock is None and server_id is None:
            raise ValueError("server_id, sock or transport")
        self.id = server_id
        self.sock = sock or (sock_path(server_id, bench) if server_id else None)
        self.timeout = timeout
        self._transport = transport
        self.on_request = None
        self._log = log                     # _log(dict) per request: verb, ok, wall, step ("log" is a server verb)
        self.n_calls = 0
        self.wall_s = 0.0

    def request(self, req: dict) -> dict:
        t0 = time.perf_counter()
        rep = self._transport(req) if self._transport is not None else unix_request(self.sock, req, self.timeout)
        dt = time.perf_counter() - t0
        self.n_calls += 1
        self.wall_s += dt
        if self._log is not None:
            self._log({"verb": req.get("cmd"), "ok": bool(rep.get("ok")), "wall_s": round(dt, 4), "step": rep.get("step"),
                      "error": rep.get("error") if not rep.get("ok") else None})
        return rep

    def call(self, _verb: str, **kw) -> dict:          # _verb, not cmd: a cmd= keyword must reach kw (v2's fix)
        if self.on_request is not None:
            self.on_request(_verb)
        kw = dict(kw)
        if "cmd" in kw:                                 # gripper(cmd="open") -> action (the envelope key is the verb)
            kw["action"] = kw.pop("cmd")
        kw["cmd"] = _verb
        return self.request(kw)

    def ok(self, _verb: str, **kw) -> dict:
        rep = self.call(_verb, **kw)
        if not rep.get("ok"):
            raise ServerError(_verb, rep)
        return rep

    def __getattr__(self, verb):
        if verb.startswith("_"):
            raise AttributeError(verb)
        return lambda **kw: self.call(verb, **kw)


__all__ = ["RobotServerClient", "ServerError", "sock_path", "unix_request", "BENCH_DEFAULT"]
