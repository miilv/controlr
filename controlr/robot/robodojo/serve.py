"""``controlr robodojo-serve``: controlr as RoboDojo's controller.

Listens on localhost for the shim (``shim/deploy.py``, inside RoboDojo's eval client). Every
connection is one RoboDojo episode: the shim announces it (``hello``: task, layout, instruction,
step limit), we run a normal controlr episode on it (``run_episode`` with ``RoboDojoRobot``; one
run directory per episode, named ``<cfg.name>_<task>_L<layout>``), then tell the shim the
outcome (``done``) so RoboDojo can score and save the episode. Runs until ``max_episodes`` or
until killed by the run script after the eval client exits.
"""

from __future__ import annotations

import copy
import os
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Callable

from controlr.robot.robodojo import protocol as P

READY_MARKER = "CONTROLR_ROBODOJO_READY"


def episode_config(cfg, episode: dict):
    """The per-episode config: task and seed come from RoboDojo (seed = layout id)."""
    c = copy.deepcopy(cfg)
    task = str(episode.get("task") or c.task.name)
    layout = episode.get("layout_id")
    c.task.name = task
    if layout is not None:
        c.seed = int(layout)
    c.name = f"{cfg.name}_{task}_L{layout if layout is not None else 'x'}"
    return c


def serve(cfg, *, llm_factory: Callable[[], object] | None = None, max_episodes: int | None = None,
          listener: Listener | None = None, on_episode: Callable | None = None,
          on_turn: Callable | None = None) -> list:
    from controlr.loop import run_episode
    from controlr.robot.robodojo.client import RoboDojoRobot

    own = listener is None
    if listener is None:
        listener = Listener(P.address_from_env(), authkey=P.authkey_from_env())
    ready = os.environ.get("CONTROLR_ROBODOJO_READY_FILE")
    if ready:
        Path(ready).write_text(READY_MARKER + "\n")
    print(f"{READY_MARKER} {listener.address}", flush=True)
    results = []
    try:
        while max_episodes is None or len(results) < max_episodes:
            conn = listener.accept()
            try:
                hello = conn.recv()
                if not isinstance(hello, dict) or hello.get("op") != "hello":
                    raise P.RemoteError(f"expected hello, got {hello!r}"[:200])
                ep = hello["episode"]
                c = episode_config(cfg, ep)
                print(f"episode {c.name}: {ep.get('instruction')!r} (step_lim {ep.get('step_lim')})", flush=True)
                robot = RoboDojoRobot(conn, ep, c.robot.params)
                res = run_episode(c, robot=robot, llm=llm_factory() if llm_factory else None, on_turn=on_turn)
                try:
                    conn.send(P.request("done", outcome=res.outcome))
                    P.unwrap(conn.recv())
                except (EOFError, OSError, P.RemoteError) as e:
                    print(f"  done not acknowledged: {e}", flush=True)
                results.append(res)
                print(f"-> {res.outcome} success={res.success} turns={res.turns} {res.run_dir}"
                      + (f"\n   error: {res.error}" if res.error else ""), flush=True)
                if on_episode:
                    on_episode(res)
                if res.outcome == "interrupted":
                    break
            finally:
                conn.close()
    finally:
        if own:
            listener.close()
    return results
