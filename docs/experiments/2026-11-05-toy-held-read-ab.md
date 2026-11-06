# 2026-11-05 — pre-registration: held-ring read at the peg hover

**Status at registration:** specification only; no new camera episodes. Do not execute while D-C19 owns compute4.

**Hypothesis:** a measured ring-to-TCP transform reduces peg misses caused by an unobserved attachment offset.
Base refinement stays enabled. This tests `peg_held_read`, not the rejected `peg_stuck_confirm`.

## Frozen comparison

Gain cell: `play_stacking_toy`, ten paired layouts **100040–100049**, one episode per arm per layout. Reserve these
ids for camera scoring after registration; do not tune on them. The existing permitted `c4b_playstackingtoysg*`
inventory contains no camera A/B on these ids. Before dispatch the lead must check the complete camera experiment
registry for this cell: if any id has already been A/B'd, stop before collecting either arm and amend this registration
to ten other never-used layouts. No replacement or expansion after examining outcomes.

- A: frozen integration candidate with `peg_base_refine: local`, `peg_base_tries: 8`, `peg_stuck_confirm: off`,
  `held_pose: request`, `peg_held_read: off`.
- B: identical bytes/config except `peg_held_read: on`.
- Pin the PR commit, candidate JSON hash, asset catalogue/layout hashes and perception-service version in the dispatch
  manifest. Both arms must use the same fixed non-Claude route (`cx/gpt-6-luna` for any agentic calls), model parameters,
  planner configuration, generator version, step budget, renderer and cameras. Do not compare against historical scores
  collected with a different model. No `claude/` or `cc/` calls.
- Pair arms in one stable layout state; order A/B on even ids and B/A on odd ids. Keep all outcomes and infrastructure
  failures. An infrastructure-only retry must use the same inputs, be recorded, and be independent of the task score.
- Switch scope is `stack_on_pegs` only. Guard cells: `stack_blocks` and `insert_tubes`, ten pairs each on the same id
  range (fresh per cell checked before dispatch). Expect identical action traces; report any divergence.

## Decision (default camera rule, fixed before collection)

Adopt-candidate only if **all** conditions hold:

1. Gain cell n >= 10 pairs; pooled score(B) − pooled score(A) >= **5 points** on the 0–100 scale.
2. Success flips: B-only wins >= 2 × A-only losses, and wins − losses >= 2.
3. Each guard: pooled score delta >= −5 points; flip losses <= wins.
4. Physics-only replay write audit (`scripts/perception/m95_score.py audit`) finds no excess harmful belief writes
   against A on the same episodes, per writer. Harmful means >2 mm extra 3-D tracked-object error; report z separately.
   Include held-pose reattachments and any logged before/after writes, not merely grasp-time accuracy.

Report every pair's score, binary success, steps, read attempts/adoptions, re-picks, no-pair failures, harmful/helpful
writes and wall time. Give pooled means and spread, paired deltas and flip counts. Camera frames/videos of one success
and a typical failure use the repository's video tooling. No adoption from G3 alone. A rejection with a positive pooled
delta allows one fresh-layout retest under a new registration; no same-id reruns.

## Budget and boundaries

60 camera episodes (30 pairs including guards), at most three initial agentic calls per episode plus existing recovery
calls: provision 180 initial calls and report actual calls/tokens per model, cache share and GPU hours. This document
spends zero calls. Dispatch is deferred until the sealed measurement has released compute4 (expected after
2026-11-06 06:00Z; check release with the lead, never by reading sealed directories). During implementation compute4 is
read-only, limited to historical dev rows under `~/dojo-data/integ-c4/runs2`; measurement ids and all sealed paths are
excluded from diagnosis. Use the standard GPU lock and 10 GB disk guard for later execution.
