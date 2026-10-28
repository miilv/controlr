# 2026-10-28 — the uniform alternative to D-C17's per-task tie switch (Legality stream)

The lead's single-fixed-system audit: `select_ties.TIE_OFF` is a per-vocabulary switch (the instruction-derived task),
which a reviewer can read as per-task tuning. The uniform alternative is no select-list tie for ANY task.

**Done** (branch `fix/legal-uniform-ties`): `EstimatorConfig.select_ties` = "dc17" (default, unchanged: ties kept where
the checker scores no identity, off for TIE_OFF) | "off" (no tie for any task: the identity step takes each member's own
best key, `_options` constrains nothing). `select_ties.tie_mode(task, mode, policy)`. Test:
`test_legality_select_ties.py::test_the_uniform_policy_turns_every_tie_off_and_the_default_is_unchanged`.

**Cost, offline frame 0** (B1 / eval_g1, `--bg-loo`, ids 100000-09 + 100020-24, compute3 light lane, ~8 min, 0 LLM).
Only the KEPT tasks change (TIE_OFF tasks are already off, INERT ones have nothing to tie); of those, stack_blocks and
stack_bowls have frame-0 sets (sweep_blocks and the `_random` variants have none: unmeasured).

| renderer | task | objects | identity right dc17 -> off | found, xy < 10 mm | xy median (mm) |
|---|---|---:|---|---|---|
| 5.1 | stack_blocks | 45 | 28 -> 21 | 35 -> 31 | 1.6 -> 6.4 |
| 5.1 | stack_bowls | 45 | 32 -> 37 | 27 -> 31 | 6.8 -> 5.7 |
| 6.0 | stack_blocks | 45 | 28 -> 27 | 34 -> 30 | 1.7 -> 2.1 |
| 6.0 | stack_bowls | 45 | 30 -> 38 | 26 -> 30 | 8.7 -> 6.0 |
| | **total** | **180** | **118 -> 123** | **122 -> 122** | |

- Free in aggregate, not per task. stack_bowls gains: the `same` tie (one asset for all three bowls) was hurting.
  stack_blocks loses: its single-category `hierarchical` option + distinct keys help its fits (5.1 xy median 1.6 ->
  6.4 mm). Neither checker scores identity, so the rendered effect runs through the fits (grasps), not the identities.
- Default unchanged until the lead decides.
