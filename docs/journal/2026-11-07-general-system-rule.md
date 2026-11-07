# 2026-11-07: one general system is a hard rule

**Decision (Ilia, ~17:20Z, "our hard rule and the path to generalization; it cannot be changed").** Our RoboDojo
system is ONE general system, along the line Mikhail proposed:
- a model planner composes **general skills** (pick, place, push, pour, press, open, insert, handover) and
  perception queries;
- it writes code against the same primitives only when no skill fits or a skill fails;
- working code becomes a candidate general skill and is promoted only if it holds on other layouts (the ratchet);
- one perception path and one catalogue serve every episode.

**Forbidden from now on:**
- task names in prompts or code, task-named vocabularies, task-keyed switches and routes;
- per-task builds, entries and prompt banks;
- values taken from task configs or checkers;
- evaluator-termination feedback ("still running ⇒ failed");
- the three PyRUA-RSI tricks: picking blocks into the dustpan instead of sweeping, kong's slot→group table, and
  termination as a retry signal.

**Why.**
- The RoboDojo submission rules, read verbatim from the site bundle on 7 Nov: "Each submission shall represent a single
  fixed system ... exact same inference procedure across all tasks. An agent or planner inside that system is allowed
  ... We do not allow a router that switches between different policies or checkpoints by task. We also do not allow
  a router that selects, for each task, the policy already known to perform best on that task."
- A Codex legality audit on the partner team's own planner system found it "likely illegal". Their planner picked
  registry rows that were tasks in disguise, and programs copied task-config / checker logic.
- PyRUA-RSI's per-task programs (~85–87 in its own runs) are selected by task name, so they would not qualify as a
  board entry. We keep its scores only as a yardstick of what is physically reachable (REPRO,
  `docs/experiments/data/rsi-yardstick/`).

**What changes today.**
- **Every per-cell stream builds general skills or skill parameters.** COIN2, CLASSIFY3, ORG2, MATCH2 and IMIT2 have
  been told. The mechanisms found so far are general in nature: a moving-object intercept, close-after-clearance
  insertion of thin objects, mouth-relative pouring, reading demonstration order from frame changes, regrasp carries
  and base-origin perception.
- **SAMINT is re-scoped from per-task SAM routing to one general union perception** (v2b ∪ SAM). Its prompts come
  from the instruction plus the shared catalogue. The per-task routing (#424) stays default-off and diagnostic-only.
- **The KONG scoped re-test is parked** (42 unstarted rows,
  `compute4:~/dojo-data/kong-retest-parked-1720.txt`). Its lever is keyed on the task name
  (`watch_relocate_task="make_kong"`), so it cannot be adopted. KEY scoped (`key_scope_task`) is dead for the same
  reason.
- **The joint system with the partner team is accepted:** their model planner over a shared general skill library.
  A stronger model than Haiku is allowed. BACKLOG lists our current task-keyed routing (`supervisor.py`
  vocabularies, `routes.py` / `planner.py` text rules, task-keyed switches) as the items to retire.
