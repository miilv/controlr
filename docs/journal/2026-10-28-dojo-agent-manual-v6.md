# 2026-10-28: agent manual v6, a compliance fix (no checker knowledge in prompts); the "Plan none" fallback harness

## Why

The hybrid LLM agent's default manual (`manual_v4`, since 2026-10-25) broke our own single-fixed-system rules
(robot-harness RULES-robodojo §14.5: no score ladders, reward predicates, checker thresholds or per-task recipes reach
the model). Found by the independent reviewer while setting up the agent as the "Plan none" fallback, confirmed by the
Legality stream's audit (two reads), ruled by the lead:
- **§3 "Instruction patterns and how they are judged"**, "they come from how the benchmark checks results": 14
  checker-derived recipes keyed by instruction phrasing, i.e. per-task text. At least 7 of the 19 "Plan none" cells
  had one (cover, general_pickup, classify x2, arrange x2, pour).
- **§2 numbers and reward predicates:** "within about 0.15 m and 20 degrees", "opening above 0.8", "checked to about a
  centimetre and a few degrees", "pressing a button one time too many can fail the task", "lifting it can fail the
  task", "a lift-and-hold goal stays satisfied only while holding".
- **Skill docs:** four magnitudes ("lifts it ~12 cm", "holds it up ~11 cm", "back up ~7 cm", "sits ~1 cm down") and one
  judging phrase (fill_container: "is judged by the end state only").

This is a compliance fix, not a tuning change: v6 is the default because v4 is not allowed, whatever it scores.

## What changed

- `dojo/agent/prompts/manual_v6.md` = v4 without §3, and with §2 rewritten as six generic action rules:
  1. end with `finish()` (both arms home, grippers open; leave enough steps);
  2. open both grippers after each completed stage (that banks it);
  3. counts and orders given in the instruction are exact;
  4. hold still while another arm moves in the scene;
  5. do not release an object unless the instruction asks you to put it somewhere (the generic form of v4's
     "hold what must be held"; ruled by the lead);
  6. record in code what the instruction asks you to remember.
  - The skill-library intro says "as it is stated" instead of "as it will be judged".
- `SKILL_DOCS_VERSION` 11 -> 12: the four magnitudes reworded as motion ("lifts it and holds it", ...), fill_container
  described by the end state the skill leaves, not by how it is judged.
- `skills/registry.yaml` (the J2 registry, whose docs fill the same prompt slot when `AgentConfig.registry` is set): the
  same four magnitudes and the fill_container phrase reworded; `notes` / `stats` fields never reach a prompt.
- `AgentConfig.manual` default `manual_v4` -> `manual_v6`; `manual_v4` stays in the tree, unedited (old runs).
- `tests/dojo/test_agent_manual_v6.py`: no §3 recipes; no number + unit / "opening above" in §2 or any skill doc; no
  "fail(s) the task" / checked / judged / score / reward in §2 or any skill doc; the six generic rules present; v6 is the
  default.

Kept, by the lead's ruling: §1's robot facts (reach, perception error), skill docs' instruction quotes ("For 'fold the
clothes'": routing by instruction phrasing, as the rule planner does), API argument defaults (`lift=0.08`).

## The fallback harness (`scripts/agent_fallback/`)

The "Plan none" fallback is the existing `AgentPolicy` with `{"extra": {"agent": {"planner_first": true}}}`: the rule
planner's skill first (no LLM call); the agent loop only when the planner names no skill (or the skill failed with
>= 35 % of the budget left). It stays OFF in the candidate preset until its measurement and a camera A/B.
- `g3_agent.py`: a G3-equivalent planning ceiling. `dojo.runtime.eval._run` on simlab's `ProxyEpisode` (physics-only
  Isaac 5.1 RoboDojo EvalEnv) with a NULL renderer (no frames: the agent works from the GT scene text), mode gt,
  verify off. Label its numbers "GT-scene-text planning ceiling".
- `job.sh` (one light-lane job: waits for the lane OUTSIDE its `timeout -k 60 1800`, scope 8G / 600 %),
  `run_all.sh` (a sequential queue on one lock, layouts 100000-04 settled), `summarize.py` (per-cell success / score /
  calls / tokens / Average points).
- The measurement (19 cells x 5 layouts, cx/gpt-6.1-sol, manual_v6) is running; its numbers go to a report under
  docs/experiments/ when done.
