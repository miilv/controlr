"""Thread bridge: a synchronous program (skills / expert programs) behind proxy_eval's ``next(obs)``.

The skills layer drives an env synchronously (``env.step(action)``, ``env.proprio()``); the proxy
harness calls the policy (``next(obs) -> {"actions": [...]}``) and renders one observation per call.
``Bridge`` runs the program in a worker thread:

* ``step(action)`` only buffers the action (the program keeps planning back to back);
* ``observe()`` (and ``proprio()``) flushes the buffer: the main thread's ``next()`` returns it to the
  harness, the harness executes it and calls ``next(new_obs)``, the worker resumes with that obs;
* when the program returns, a last flush hands over any pending actions, then ``{"done": True}``.

The env itself is touched only by the main thread (the harness); the worker does numpy + socket I/O.
If the episode ends (success, query failure, step limit) the harness stops calling ``next`` and the
worker stays blocked in ``observe``; ``cancel()`` (called on the next ``reset``) unblocks it with
``Cancelled`` (a BaseException, so no ``except Exception`` in a program swallows it).

Every executed action is kept in ``executed`` (the official action dicts, in order), so the runtime can
match an executed segment against the twin's verified candidates.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback

import numpy as np

_CANCEL = object()


class Cancelled(BaseException):
    """The episode is over (or a new one starts); the worker must unwind."""


class Bridge:
    def __init__(self):
        self._to_main: queue.Queue = queue.Queue()
        self._to_worker: queue.Queue = queue.Queue()
        self.pending: list[dict] = []
        self.executed: list[dict] = []
        self.obs: dict | None = None
        self.fresh = False
        self.thread: threading.Thread | None = None
        self.n_flush = 0
        self.result = None
        self.error: str | None = None
        self.done = False
        self.t_worker = 0.0                 # wall time the worker ran (planning, perception, verify calls)
        self._t_resume = None

    # ------------------------------------------------------------------ main thread
    def start(self, target, obs: dict) -> dict:
        self.obs, self.fresh = obs, True
        self.thread = threading.Thread(target=self._run, args=(target,), daemon=True, name="dojo-runtime-program")
        self._t_resume = time.perf_counter()
        self.thread.start()
        return self._wait()

    def resume(self, obs: dict) -> dict:
        if self.done:
            return {"done": True}
        self._t_resume = time.perf_counter()
        self._to_worker.put(obs)
        return self._wait()

    def _wait(self) -> dict:
        msg = self._to_main.get()
        self.t_worker += time.perf_counter() - (self._t_resume or time.perf_counter())
        if msg[0] == "actions":
            return {"actions": msg[1]}
        self.done = True
        return {"done": True}

    def cancel(self, timeout_s: float = 10.0) -> None:
        if self.thread is not None and self.thread.is_alive():
            self._to_worker.put(_CANCEL)
            self.thread.join(timeout_s)

    # ------------------------------------------------------------------ worker thread
    def _run(self, target) -> None:
        try:
            self.result = target()
            if self.pending:
                self.observe()              # hand the last actions over (the harness ends the episode)
            self._to_main.put(("done", None))
        except Cancelled:
            return
        except BaseException as e:  # noqa: BLE001 (reported, the episode ends with done)
            self.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
            self._to_main.put(("done", None))

    def observe(self) -> dict:
        """The observation after every buffered action has run (flushes if needed)."""
        if self.fresh:
            return self.obs
        acts = self.pending
        self.pending = []
        self._to_main.put(("actions", acts))
        obs = self._to_worker.get()
        if obs is _CANCEL:
            raise Cancelled()
        self.executed.extend(acts)
        self.obs, self.fresh = obs, True
        self.n_flush += 1
        return obs

    # ------------------------------------------------------------------ skills env protocol
    @property
    def step_lim(self) -> int:
        return int(self.obs["step_lim"])

    @property
    def steps_used(self) -> int:
        return int(self.obs["steps_used"]) + len(self.pending)

    def step(self, action: dict) -> bool:
        """Buffer one env step. True when this step uses the last one of the budget (the harness then
        ends the episode)."""
        self.pending.append({k: np.asarray(v, np.float32).copy() for k, v in action.items()})
        self.fresh = False
        return self.steps_used >= self.step_lim

    def proprio(self) -> dict:
        return self.observe()["state"]

    @property
    def n_executed(self) -> int:
        return len(self.executed) + len(self.pending)

    def actions_since(self, n0: int) -> list[dict]:
        allx = self.executed + self.pending
        return allx[n0:]


__all__ = ["Bridge", "Cancelled"]
