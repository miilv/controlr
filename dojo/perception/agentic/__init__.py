"""Frozen agentic perceiver (D-C6): a fresh LLM agent per perceive call, launched by the policy.

- system prompt = a short runner preamble (``prompts/``) + robot-harness ``docs/PERCEIVER-PROTOCOL.md`` (vendored
  byte-identical under ``vendor/``, pinned by md5 in ``vendor/MANIFEST.md5``);
- tools = the protocol's own scripts (``vendor/robodojo/*.py``, ``vendor/libero/grid.py``) run in a per-call
  workspace, plus ``locate`` / ``project`` over the call's calibration (``geom.py`` math);
- model pinned by id through omniroute; no memory across calls or episodes; no human.

Entry points: ``j1.perceive`` / ``j1.held_look`` (robot-harness J1 interface), ``libero.perceive_cell`` (LIBERO picks.json).
"""
