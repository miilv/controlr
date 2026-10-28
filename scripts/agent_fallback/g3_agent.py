"""Physics-only (G3-equivalent) planning ceiling of the LLM agent as the "Plan none" fallback.

AgentPolicy with planner_first (the rule planner first; the agent loop only when it names no skill), RuntimeConfig mode
gt (perfect perception), verify off, on simlab's ProxyEpisode (physics-only Isaac 5.1 RoboDojo EvalEnv: its reset,
step accounting, step_lim and reward latches) with a NULL renderer: no camera frames, so the agent works from the GT
scene text only ("GT-scene-text planning ceiling"). Rows / reports / actions as dojo.runtime.eval writes them.

    scripts/simlab/rdpy.sh -u scripts/agent_fallback/g3_agent.py --task T --layouts L.json ... --out DIR --agent '{...}'
Keys: DOJO_AGENT_DOTENV names the one dotenv with OMNIROUTE_* (never printed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


class NullRender:
    """RenderClient's interface without a renderer: frames are absent (``images`` {})."""

    def load_stage(self, path, spec) -> float:
        return 0.0

    def render(self, gt) -> dict:
        return {"images": {}, "renderer": "none"}

    def close(self) -> None:
        pass


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--layouts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent", default="{}", help="AgentConfig fields (JSON); planner_first is forced on")
    ap.add_argument("--episode-max-minutes", type=float, default=12.0)
    a = ap.parse_args(argv)
    from dojo.agent.policy import AgentPolicy
    from dojo.bridge.twin import BundleTwins
    from dojo.runtime import eval as E
    from dojo.runtime.policy import RuntimeConfig
    from dojo.simlab.proxy_eval import ProxyEpisode

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    agent = {**json.loads(a.agent), "planner_first": True}
    cfg = RuntimeConfig(mode="gt", verify="off", extra={"agent": agent})
    ep = ProxyEpisode(a.task, NullRender(), out)
    pol = AgentPolicy(cfg, log_root=out / "agent")
    twins = BundleTwins(a.task, 1, out / "twins")
    try:
        E._run(a.task, a.layouts, cfg, out, 1e9, "none", a.episode_max_minutes, f"{a.task}_gt_agentfb", None, ep,
               pol, twins)
    finally:
        twins.stop()


if __name__ == "__main__":
    main()
