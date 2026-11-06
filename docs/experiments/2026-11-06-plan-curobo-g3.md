# Belief-object cuRobo planning: paired G3 validation

**Question:** can a belief-backed cuRobo world and gripper attachment improve the existing skills without changing their rates or contact commands?

**Setup:** source db8daddda4937402f6da1dcd2c19a3b650d4c021, compute6, physics-only G3 mode A; ten paired settled dev layouts each for tools, egg and stack. Both arms fix plan_solver=curobo; A uses plan_world=table, B objects. All other fields retain defaults. Inputs are the same frozen layout manifest as the [CPU comparison](2026-11-06-plan-world-g3.md). No held-out/reserved IDs, camera-policy episodes or model calls.

The native skill planner already checks objects and held geometry; this experiment changes the free-space proposer. cuRobo receives believed OBBs/container parts and attached-box covering spheres, then the existing Collision model validates its proposals. See the [journal](../journal/2026-11-06-dojo-plan-world.md) and [worker incident](../incidents/2026-11-06-plan-curobo-worker.md).

[Machine-readable results and provenance](2026-11-06-plan-curobo-results.json).

## Aggregate

Score is 0–100, with sample SD. Times are observed wall seconds on a shared machine, not a controlled latency benchmark. Planning time includes all candidate search; query/validation and cold startup are subsets.

| Cell | Arm | n | Success | Score ± SD | Mean / median steps | Planning s | Query s | Startup s | Fallback rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| store_tools_in_toolbox | table | 10 | 4/10 | 77.50 ± 21.89 | 453.4 / 434.5 | 2239.2 | 0 | 0 | n/a |
| store_tools_in_toolbox | objects | 10 | 4/10 | 80.00 ± 19.72 | 461.7 / 442.0 | 2272.6 | 86.11 | 90.79 | 88.7% (275/310) |
| fill_egg_holder | table | 10 | 0/10 | 23.50 ± 4.74 | 316.2 / 327.0 | 257.1 | 0 | 0 | n/a |
| fill_egg_holder | objects | 10 | 0/10 | 22.00 ± 6.32 | 315.4 / 307.5 | 921.2 | 645.98 | 69.94 | 97.5% (2850/2922) |
| stack_blocks | table | 10 | 10/10 | 100.00 ± 0.00 | 153.6 / 169.5 | 14.9 | 0 | 0 | n/a |
| stack_blocks | objects | 10 | 10/10 | 100.00 ± 0.00 | 153.8 / 169.5 | 216.4 | 115.04 | 85.84 | 59.1% (162/274) |

## Per-layout pairs

| Cell | ID | A score | B score | A steps | B steps |
|---|---:|---:|---:|---:|---:|
| store_tools_in_toolbox | 100000 | 100 | 100 | 429 | 430 |
| store_tools_in_toolbox | 100001 | 50 | 75 | 431 | 436 |
| store_tools_in_toolbox | 100002 | 50 | 50 | 479 | 476 |
| store_tools_in_toolbox | 100003 | 75 | 50 | 437 | 446 |
| store_tools_in_toolbox | 100004 | 100 | 100 | 536 | 542 |
| store_tools_in_toolbox | 100030 | 75 | 75 | 421 | 421 |
| store_tools_in_toolbox | 100031 | 75 | 75 | 432 | 433 |
| store_tools_in_toolbox | 100032 | 100 | 100 | 545 | 549 |
| store_tools_in_toolbox | 100033 | 50 | 75 | 382 | 446 |
| store_tools_in_toolbox | 100034 | 100 | 100 | 442 | 438 |
| fill_egg_holder | 100000 | 25 | 25 | 324 | 271 |
| fill_egg_holder | 100001 | 25 | 25 | 209 | 210 |
| fill_egg_holder | 100010 | 25 | 25 | 425 | 307 |
| fill_egg_holder | 100012 | 25 | 25 | 272 | 443 |
| fill_egg_holder | 100015 | 25 | 25 | 330 | 308 |
| fill_egg_holder | 100016 | 25 | 10 | 201 | 287 |
| fill_egg_holder | 100019 | 25 | 25 | 393 | 338 |
| fill_egg_holder | 100053 | 25 | 25 | 420 | 432 |
| fill_egg_holder | 100054 | 25 | 25 | 215 | 329 |
| fill_egg_holder | 100055 | 10 | 10 | 373 | 229 |
| stack_blocks | 100000 | 100 | 100 | 203 | 206 |
| stack_blocks | 100001 | 100 | 100 | 113 | 113 |
| stack_blocks | 100002 | 100 | 100 | 175 | 175 |
| stack_blocks | 100003 | 100 | 100 | 186 | 186 |
| stack_blocks | 100004 | 100 | 100 | 164 | 164 |
| stack_blocks | 100010 | 100 | 100 | 190 | 189 |
| stack_blocks | 100011 | 100 | 100 | 108 | 108 |
| stack_blocks | 100012 | 100 | 100 | 116 | 116 |
| stack_blocks | 100013 | 100 | 100 | 177 | 177 |
| stack_blocks | 100014 | 100 | 100 | 104 | 104 |

## Paired interpretation

- store_tools_in_toolbox: delta +2.50 points, paired SD 14.19; score wins/losses 2/1; success flips 0/0.
- fill_egg_holder: delta -1.50 points, paired SD 4.74; score wins/losses 0/1; success flips 0/0.
- stack_blocks: delta +0.00 points, paired SD 0.00; score wins/losses 0/0; success flips 0/0.

No full-success ceiling improved. Tools' +2.5-point change is below the
prospective five-point score gate; egg regressed and the stack control stayed
perfect. No GPU camera A/B is warranted by this sample. The earlier
[CPU egg camera specification](2026-11-06-plan-routes-camera-ab.md) remains
prospective and undispatched; its partial-score signal is not a ceiling gain.

## Planner coverage and failure limits

Counts include speculative candidate plans; an accepted proposal is not necessarily an executed move. A worker deadline stops the owned worker and caches legacy fallback for the remainder of that Motion instance (normally the episode). Thus repeated timeout-labelled fallbacks are not repeated expensive GPU calls.

| Cell | Worker-backed requests | Accepted proposals | Actual worker deadlines | Cached fallbacks |
|---|---:|---:|---:|---:|
| store_tools_in_toolbox | 117 | 35 | 10 | 193 |
| fill_egg_holder | 1376 | 72 | 5 | 1546 |
| stack_blocks | 153 | 112 | 8 | 121 |

The two-second GPU deadline is separate from up-to-60-second cold startup. Shutdown and individual CPU collision checks can add a small overrun; maximum observed query/validation duration is retained in the JSON. The robot remains still while planning. The existing safety envelope, contact/explicit-joint semantics, step limits and rate/easing rules are unchanged. The fixed mounting base is omitted from the GPU model; the original skill model validates every accepted path. No other-arm or continuous-collision guarantee is claimed.

Failed skill events remain slips, missed placements and unreachable pairs as recorded below. They overlap retries and are not an exclusive object-level causal taxonomy; an in-hand shift alone does not prove which contact caused it.

- store_tools_in_toolbox failed-skill events: A {'place_miss': 7, 'grasp_slip': 3, 'collision': 1, 'ik_fail': 2}; B {'grasp_slip': 3, 'collision': 1, 'place_miss': 5}.

- fill_egg_holder failed-skill events: A {'place_miss': 1, 'grasp_slip': 3}; B {'place_miss': 1, 'grasp_slip': 4}.

- stack_blocks failed-skill events: A {}; B {}.

## Spend and artifacts

60 completed G3 episodes; recorded episode wall 7357.9 s and load 314.2 s (2.13 lane-hours), plus app startup, API probes, cleanup and queue overhead. **0 LLM calls, 0 tokens, $0 model spend.** No camera-policy run or default change. Raw layouts, actions, logs and process receipts stay in compute6:~/dojo-data/plan. Local writes remain below 1 GB. Physics-only source episodes have no camera frames; any separately rendered action replay is diagnostic and excluded from these outcomes.


## Reproduction and diagnostic media

Run the G3 entrypoint with the frozen source and the manifest's copied layout
directory, once for each arm and task. Example arguments (inside the approved
light-lock, disk/VRAM admission and resource wrapper):

```bash
scripts/simlab/rdpy.sh -u -m dojo.skills.g3 run \
  --task store_tools_in_toolbox --seeds 100000,100001,100002,100003,100004 \
  --layouts ~/dojo-data/plan/layouts/store_tools_in_toolbox \
  --skill-attr plan_solver=curobo --skill-attr plan_world=objects \
  --out ~/dojo-data/plan/g3-curobo/<unique-output>
```

Use table for A and objects for B, with the same manifest and original task
step limits. The JSON records all per-layout results, fallback reasons, source
and asset fingerprints. The baseline's historical name table retains existing
native object checks; this is a comparison of free-space route proposers.

[Successful tools replay](../video/plan-world-tools-success.mp4) and
[typical partial failure](../video/plan-world-tools-failure.mp4) render the saved
candidate joint actions in Isaac with three RGB views. Source/replay outcomes
match: layout 100000 succeeds at 100; 100003 fails at 50. These clips were made
with scripts/turn_video.py from a separate diagnostic render export on merged
head 8f6a9d26. They contain no model call, are excluded from the paired sample,
and are not camera-policy evidence. A separate merged-head GPU stack smoke
also reproduced layout 100000's success and 206 steps.


A later, separately frozen [HANG/sweep extension](2026-11-06-plan-extension-g3.md)
also finds no ceiling gain: HANG 61.5→57.5 (4/10 successes both), sweep 20→20
(2/10 both). Its source, generator versions and stricter admission limits are
reported separately. The held-sphere representation covers the full box rather
than mesh concavities; representation and deadline effects are not isolated.
