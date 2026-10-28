"""AgentPolicy: the hybrid LLM agent as a proxy_eval policy (``--policy dojo.agent.policy:AgentPolicy``).

``RuntimePolicy`` does the setup (perception + looks + search in mode perception, the WorldState provider, the twin
verifier with ``verify=bundle``) and the tail (finish unless the program says ok, still steps, error handling);
in between, instead of the rule supervisor, ``AgentPolicy`` runs ``dojo.agent.loop.AgentLoop``: the model reads
the manual, the instruction, the scene (neutral ids) and the head frame, and answers with code cells over ``robo``
(skills, primitives, perception, verify inside the skills). The same model, manual and procedure for every task;
the launcher's task name and the layout id never reach it (``RuntimePolicy.reset`` keeps the instruction and the step
limit only).

Config: ``AgentConfig`` fields in the runtime config's ``extra.agent`` (``--config '{"extra":{"agent":{...}}}'``)
or as JSON in ``$DOJO_AGENT`` (``model``, ``extra_body``, ``max_calls``, ...). Logs per episode under
``$DOJO_AGENT_LOG`` (default ``$OUT/agent``, the eval run dir): transcript (images by sha), calls with usage, cells.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from dojo.agent.loop import AgentConfig, AgentLoop
from dojo.agent.manual import SKILL_DOCS_VERSION
from dojo.agent.robo import CellLimits, Robo
from dojo.agent.scene import assign_ids, scene_text
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy


def _deployed_commit() -> str | None:
    """The commit this checkout was deployed from (``.deployed_commit``, written at deploy): runs map to code,
    including the perception version merged into it."""
    f = Path(__file__).resolve().parents[2] / ".deployed_commit"
    try:
        return f.read_text().strip()[:40] or None
    except OSError:
        return None


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
        # agent config: explicit > RuntimeConfig.extra["agent"] (``--config '{"extra":{"agent":{...}}}'``, what a
        # queue job can carry) > $DOJO_AGENT
        extra = (getattr(self.cfg, "extra", None) or {}).get("agent")
        self.agent_cfg = agent_cfg or AgentConfig(**(extra if extra is not None else
                                                     json.loads(os.environ.get("DOJO_AGENT", "{}") or "{}")))
        self._client = client
        default_root = (Path(os.environ["OUT"]) / "agent") if os.environ.get("OUT") else Path("agent_logs")
        self.log_root = Path(log_root or os.environ.get("DOJO_AGENT_LOG") or default_root)
        self.n_episodes = 0
        self.n_resets = 0

    def client(self):
        if self._client is None:
            from dojo.agent.llmio import make_client
            self._client = make_client(timeout_s=self.agent_cfg.timeout_s, max_retries=3)
        return self._client

    def episode_config(self, k: int) -> AgentConfig:
        """Episode k's agent config: ``models`` cycles by the episode counter (an experiment arm, never a
        decision input: no task / layout / instruction is read)."""
        from dataclasses import replace
        arms = self.agent_cfg.models
        if not arms:
            return self.agent_cfg
        arm = arms[k % len(arms)]
        return replace(self.agent_cfg, model=arm["model"], extra_body=dict(arm.get("extra_body") or {}), models=[])

    def reset(self, info: dict) -> None:
        super().reset(info)
        # the arm is fixed at reset (every episode the driver starts counts, even one whose setup fails)
        self._episode_cfg = self.episode_config(self.n_resets)
        self.n_resets += 1

    def _planner_first(self, c, acfg: AgentConfig) -> tuple[dict | None, str]:
        """``planner_first``: RuntimePolicy's own program (the rule planner's skill, its bounded retry), no LLM call.
        Returns (its result when the agent must not run, else None; the note the agent gets about what was tried)."""
        res = self._supervise(c)
        skill = self.report.get("skill")
        ok = bool((res or {}).get("ok"))
        w = c.sk.refresh()
        left = (w.steps_left / w.step_lim) if w.step_lim else 1.0
        c.events.append({"kind": "planner_first", "skill": skill, "ok": ok, "step": int(c.bridge.steps_used),
                         "budget_left": round(left, 3)})
        base = {"planner_first": True, "planner_skill": skill, "model": acfg.model, "usage": {}, "cells": 0,
                "skills_called": [skill] if skill else [], "code_commit": _deployed_commit(),
                "skill_docs_version": SKILL_DOCS_VERSION}
        if ok:
            self.report["agent"] = {**base, "outcome": "planner_done"}
            return res, ""
        if skill and left < acfg.planner_first_budget_frac:
            self.report["agent"] = {**base, "outcome": "planner_failed_low_budget"}
            return res or {"ok": False, "stage": "planner_first"}, ""
        if not skill:
            return None, ""
        last = next((h for h in reversed(c.sk.history) if not h.ok), None)
        why = f"{last.skill} {last.status}: {str(last.note)[:200]}" if last is not None else str(res)[:200]
        return None, (f"\n\nALREADY TRIED (no LLM): the library skill `{skill}` with the arguments the instruction "
                      f"implies, with one retry, did not complete the task ({why}). The scene above is the state now.")

    def _program(self, c) -> dict:
        acfg = getattr(self, "_episode_cfg", None) or self.episode_config(0)
        self.n_episodes += 1
        note = ""
        if acfg.planner_first:
            done, note = self._planner_first(c, acfg)
            if done is not None:
                return done
        ctx = _AgentCtx(c, self.assets())
        world = ctx.sk().refresh()
        ids = assign_ids(world)
        objects_root = getattr(ctx.assets, "root", None)
        text, ids = scene_text(world, ids, objects_root=objects_root, version=acfg.scene_version)
        text += note
        robo = Robo(ctx, ids, objects_root=objects_root,
                    limits=CellLimits(acfg.max_cell_steps, acfg.max_cell_contacts), auto_frames=acfg.auto_frames)
        from dojo.agent.loop import registry_skills
        reg_entries = registry_skills(acfg)
        if reg_entries is not None:
            robo.skills = {e.name: e for e in reg_entries}
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        log_dir = self.log_root / f"{stamp}_ep{self.n_episodes:03d}"
        loop = AgentLoop(acfg, self.client(), robo, log_dir=log_dir, log=self.log)
        rep = {"model": acfg.model, "extra_body": acfg.extra_body, "log_dir": str(log_dir), "outcome": None,
               "usage": {}, "cells": 0, "skills_called": [], "code_commit": _deployed_commit(),
               "manual": acfg.manual, "appendix": acfg.appendix, "skill_docs_version": SKILL_DOCS_VERSION,
               "registry": acfg.registry or None, "strict": acfg.strict,
               "skills_prompted": sorted(robo.skills) if robo.skills is not None else None,
               "planner_first": bool(acfg.planner_first), "planner_skill": self.report.get("skill") if acfg.planner_first else None}
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
