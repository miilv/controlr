# 2026-10-28 — a uniform colour map instead of a per-task one (Legality stream, single-fixed-system audit b2)

**Before:** B1's appearance cost used the colour map (`color_M`, the renderer's colour calibration) only when the
instruction-derived vocabulary was in `EstimatorConfig.color_tasks` (stack_blocks(_random), stack_blocks_by_language,
stack_bowls(_random)): a per-task switch ("it hurts pack", commit 5844ef7).

**Measurement** (lead-approved ~1 h light lane, compute3, offline B1 / eval_g1, frame 0, `--bg-loo`, ids 100000-09 +
100020-24, 0 LLM). Arms "all" (map for every task) and "none" (never); the current per-task setting is each task's arm.
Sets: d0_51 (genuine 5.1, 6 tasks) and d0_ext (6.0 proxy, 12 tasks). The 6.0 `d0` set repeats d0_51's tasks on the proxy
renderer and was dropped to fit the budget. Light lane 18:18-18:55Z.

| set | task | objects | all: within tol / identity | none | current |
|---|---|---:|---|---|---|
| d0_51 | build_tower | 120 | 99 / 118 | 99 / 118 | none |
| d0_51 | pack_objects_into_box_random | 75 | 51 / 57 | 52 / 58 | none |
| d0_51 | push_T | 30 | 25 / 30 | 25 / 30 | none |
| d0_51 | put_bottles_into_dustbin | 65 | 56 / 54 | 58 / 54 | none |
| d0_51 | stack_blocks | 45 | 35 / 28 | 34 / 28 | all |
| d0_51 | stack_bowls | 45 | 38 / 32 | 38 / 33 | all |
| d0_ext | stack_blocks_by_language | 30 | 28 / 24 | 28 / 21 | all |
| d0_ext | store_tools_in_toolbox | 50 | 49 / 50 | 48 / 49 | none |
| d0_ext | general_pickup | 10 | 6 / 10 | 7 / 10 | none |
| d0_ext | 9 more (align, cover, fill_pen, insert_key, kong, organize, plug, press, swap_T) | 553 | identical | identical | none |
| | **total** | **1023** | **858 / 676 = 1534** | **860 / 674 = 1534** | 861 / 676 = 1537 |

(within tol = eval_g1's `failure_class` ok or identity; identity = the right asset.)

**Choice: "all".** The two uniform settings tie in aggregate (1534 each, both 3 below the per-task one). Tie-break:
- "none" costs stack_blocks_by_language 3 identities of 30, and its checker scores the colour order;
- "all" costs 4 pose-within-tolerance objects across pack, bottles and general_pickup, and gains store_tools 1 / 1;
- the colour map is a property of the renderer, not of a task (color_calib's own docstring).

Per-task losers of "all" vs the current per-task setting: pack_objects_into_box_random (-1 / -1),
put_bottles_into_dustbin (-2 within tol), general_pickup (-1 within tol); gainer store_tools_in_toolbox (+1 / +1).

**Done** (branch `fix/legal-uniform-color`): `EstimatorConfig.color_map` = "all" (DEFAULT) | "tasks" (the old per-task
switch, kept for dev A/Bs) | "none"; `inverse_layout.colour_map_for(task, cfg)`; test `tests/dojo/test_legality_color_map.py`
(cv2: skips in CI, passes locally).

**Decision (the lead, 2026-10-28): the default is "all".** Reason: single-fixed-system compliance (no per-task switch in
the perception's appearance cost). The cost, -3 of 1023 objects against the per-task setting, is within noise; "all" over
"none" by the tie-break above.

**Rendered behaviour changes** for every task outside the old `color_tasks`: B1's appearance cost now uses the colour
map. Watch the three loser tasks (pack, bottles, general_pickup) in the next breadth run.
