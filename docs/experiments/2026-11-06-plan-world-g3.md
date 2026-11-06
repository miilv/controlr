# Belief-object CPU routing: paired G3 validation

**Hypothesis:** checked object/attachment-aware free-space routes reduce carry/approach losses without changing contact skills or rates.

**Setup:** compute6, physics-only mode A; ten paired settled dev layouts per cell, all other switches default. A: `plan_world=table`; B: `objects`. Motion freeze `c58b7aff`. See [journal](../journal/2026-11-06-dojo-plan-world.md) for design, budgets, manifests and launch details.

This is a bounded CPU skill-planner extension. It does not implement a dynamic cuRobo world or change simlab Mover. Existing skill collision models already include many objects in A. The added search includes Geometry, uses the existing held attachment, and tries Cartesian/midpoint routes before legacy fallback.

[Machine-readable results and provenance](2026-11-06-plan-world-results.json).

## Aggregate

Score is on a 0–100 scale; spread is sample SD. Planning totals include all candidate planning; added-search time is a subset. Fallback counts are speculative requests, including rejected candidates, not executed moves.

| Cell | Arm | n | Success | Score ± SD | Mean steps | Planning s | Added-search s | Fallbacks / requests |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fill_egg_holder | table | 10 | 0/10 | 23.5 ± 4.74 | 316.2 | 280.0 | 0 | 0/0 |
| fill_egg_holder | objects | 10 | 0/10 | 28.5 ± 12.48 | 353.3 | 552.6 | 373.15 | 1433/1603 |
| stack_blocks | table | 10 | 10/10 | 100.0 ± 0.0 | 153.6 | 17.1 | 0 | 0/0 |
| stack_blocks | objects | 10 | 10/10 | 100.0 ± 0.0 | 155.5 | 30.9 | 12.8 | 14/274 |
| store_tools_in_toolbox | table | 10 | 4/10 | 77.5 ± 21.89 | 453.4 | 979.6 | 0 | 0/0 |
| store_tools_in_toolbox | objects | 10 | 4/10 | 75.0 ± 26.35 | 456.7 | 1367.8 | 74.01 | 177/316 |

## Pairs

| Cell | ID | A score | B score | A steps | B steps |
|---|---:|---:|---:|---:|---:|
| fill_egg_holder | 100000 | 25 | 25 | 324 | 235 |
| fill_egg_holder | 100001 | 25 | 40 | 209 | 311 |
| fill_egg_holder | 100010 | 25 | 25 | 425 | 340 |
| fill_egg_holder | 100012 | 25 | 25 | 272 | 459 |
| fill_egg_holder | 100015 | 25 | 0 | 330 | 365 |
| fill_egg_holder | 100016 | 25 | 40 | 201 | 310 |
| fill_egg_holder | 100019 | 25 | 40 | 393 | 355 |
| fill_egg_holder | 100053 | 25 | 40 | 420 | 681 |
| fill_egg_holder | 100054 | 25 | 25 | 215 | 248 |
| fill_egg_holder | 100055 | 10 | 25 | 373 | 229 |
| stack_blocks | 100000 | 100 | 100 | 203 | 209 |
| stack_blocks | 100001 | 100 | 100 | 113 | 114 |
| stack_blocks | 100002 | 100 | 100 | 175 | 180 |
| stack_blocks | 100003 | 100 | 100 | 186 | 188 |
| stack_blocks | 100004 | 100 | 100 | 164 | 168 |
| stack_blocks | 100010 | 100 | 100 | 190 | 193 |
| stack_blocks | 100011 | 100 | 100 | 108 | 108 |
| stack_blocks | 100012 | 100 | 100 | 116 | 109 |
| stack_blocks | 100013 | 100 | 100 | 177 | 183 |
| stack_blocks | 100014 | 100 | 100 | 104 | 103 |
| store_tools_in_toolbox | 100000 | 100 | 100 | 429 | 438 |
| store_tools_in_toolbox | 100001 | 50 | 75 | 431 | 438 |
| store_tools_in_toolbox | 100002 | 50 | 50 | 479 | 474 |
| store_tools_in_toolbox | 100003 | 75 | 25 | 437 | 416 |
| store_tools_in_toolbox | 100004 | 100 | 100 | 536 | 540 |
| store_tools_in_toolbox | 100030 | 75 | 75 | 421 | 434 |
| store_tools_in_toolbox | 100031 | 75 | 75 | 432 | 436 |
| store_tools_in_toolbox | 100032 | 100 | 100 | 545 | 557 |
| store_tools_in_toolbox | 100033 | 50 | 50 | 382 | 381 |
| store_tools_in_toolbox | 100034 | 100 | 100 | 442 | 453 |

## Paired interpretation

- fill_egg_holder: n=10, mean delta +5.00 points (paired SD 12.91); score wins/losses 5/1, success flips 0/0.
- stack_blocks: n=10, mean delta +0.00 points (paired SD 0.00); score wins/losses 0/0, success flips 0/0.
- store_tools_in_toolbox: n=10, mean delta -2.50 points (paired SD 18.45); score wins/losses 1/1, success flips 0/0.

The egg result is a directional partial-score screen, not a full-success ceiling
break: +5 points with paired SD 12.91 and 0/10 full successes. Tools regresses
slightly; stack passes its guard. No default or camera-board entry changes.
The [prospective egg camera specification](2026-11-06-plan-routes-camera-ab.md)
keeps the repository's full score/flip gate; it is not a promotion.

Fallback rates: tools **56.0%**, egg **89.4%**, stack **5.1%** of added-search
requests. Counts include candidate evaluation, not just executed transits.
Timing load is uncontrolled across the shared machine; the planning-time
ratio is not a controlled latency benchmark.

## Limits and spend

Small dev sample, one deterministic physics run per layout/arm; no camera or board claim. Search uses sampled collision geometry, inherited contact exclusions and a cooperative wall-clock budget. It is not continuous collision certification, dynamic cuRobo planning or other-arm avoidance. A large fallback rate means much of B still uses A’s routing. No model calls/tokens/spend. Physics-only runs produce no rendered videos. Local artifacts remain below 1 GB; raw logs/actions/layouts stay on compute6 under `~/dojo-data/plan`.
