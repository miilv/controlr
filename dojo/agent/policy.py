"""AgentPolicy: the hybrid LLM agent as a proxy_eval policy (``--policy dojo.agent.policy:AgentPolicy``).

``RuntimePolicy`` does the setup (perception + looks + search in mode perception, the WorldState provider, the twin
verifier with ``verify=bundle``) and the tail (finish unless the program says ok, still steps, error handling);
in between, instead of the rule supervisor, ``AgentPolicy`` runs ``dojo.agent.loop.AgentLoop``: the model reads
the manual, the instruction, the scene (neutral ids) and the head frame, and answers with code cells over ``robo``
(skills, primitives, perception, verify inside the skills). The same model, manual and procedure for every task;
``info["task"]`` is never read (RuntimePolicy keeps it for logs only).

Config: ``AgentConfig`` fields as JSON in ``$DOJO_AGENT`` (``model``, ``extra_body``, ``max_calls``, ...); the
runtime's own config in ``$DOJO_RUNTIME`` / ``--config``. Logs per episode under ``$DOJO_AGENT_LOG`` (default
``./agent_logs``): transcript (images by sha), calls with usage, cells.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from dojo.agent.loop import AgentConfig, AgentLoop
from dojo.agent.robo import CellLimits, Robo
from dojo.agent.scene import assign_ids, scene_text
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy


class _AgentCtx:
    """``robo``'s view of RuntimePolicy's episode context (feat/dojo-integ's program seam)."""

    def __init__(self, c, assets):
        self.c, self.assets = c, assets
        self.events = c.events
        self.obs0 = c.obs0
        self.instruction = c.instruction
        perc = getattr(c, "perception", None) is not None
        self.refine = (lambda sk, label, arm, kind="pick": c.refine(label, arm, kind)) if perc else None
        self.search = self._search if perc else None

    def sk(self):
        return self.c.sk

    def new_skills(self):
        return self.c.make_skills(self.c.sk)

    def observe(self) -> dict:
        return self.c.bridge.observe()

    def steps_used(self) -> int:
        return int(self.c.bridge.steps_used)

    def log(self, msg: str) -> None:
        lg = getattr(self.c, "log", None)
        if callable(lg):
            lg(msg)

    def _search(self):
        rec = self.c.search()
        if rec is not None:
            self.c.prov.init_from_record(rec)
            self.c.make_skills(self.c.sk)
        return rec


class AgentPolicy(RuntimePolicy):
    def __init__(self, cfg: RuntimeConfig | None = None, agent_cfg: AgentConfig | None = None, client=None,
                 log_root: str | Path | None = None, **kw):
        super().__init__(cfg, program=self._program, **kw)
        self.agent_cfg = agent_cfg or AgentConfig(**json.loads(os.environ.get("DOJO_AGENT", "{}") or "{}"))
        self._client = client
        self.log_root = Path(log_root or os.environ.get("DOJO_AGENT_LOG", "agent_logs"))
        self.n_episodes = 0

    def client(self):
        if self._client is None:
            from dojo.agent.llmio import make_client
            self._client = make_client(timeout_s=self.agent_cfg.timeout_s, max_retries=3)
        return self._client

    def _program(self, c) -> dict:
        self.n_episodes += 1
        acfg = self.agent_cfg
        ctx = _AgentCtx(c, self.assets())
        world = ctx.sk().refresh()
        ids = assign_ids(world)
        objects_root = getattr(ctx.assets, "root", None)
        text, ids = scene_text(world, ids, objects_root=objects_root)
        robo = Robo(ctx, ids, objects_root=objects_root,
                    limits=CellLimits(acfg.max_cell_steps, acfg.max_cell_contacts), auto_frames=acfg.auto_frames)
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        log_dir = self.log_root / f"{stamp}_ep{self.n_episodes:03d}"
        loop = AgentLoop(acfg, self.client(), robo, log_dir=log_dir, log=self.log)
        rep = {"model": acfg.model, "extra_body": acfg.extra_body, "log_dir": str(log_dir), "outcome": None,
               "usage": {}, "cells": 0, "skills_called": []}
        self.report["agent"] = rep

        def live(lp):                     # the main thread may close the report while the worker is parked
            rep["usage"] = lp.usage()
            rep["cells"] = len(lp.cells)
            rep["skills_called"] = [e.get("skill") for e in ctx.events if e.get("kind") == "agent_skill"]
        loop.on_call = live
        frame0 = (ctx.obs0.get("images") or {}).get("cam_head")
        out = loop.run(ctx.instruction, text, frame0)
        live(loop)
        rep.update({"outcome": out["outcome"], "finish": out["finish"], "agent_config": asdict(acfg)})
        fin = robo.finished or {}
        ok = fin.get("status") == "done"
        return {"ok": ok, "stage": out["outcome"], "note": fin.get("summary", "")}


__all__ = ["AgentPolicy"]
