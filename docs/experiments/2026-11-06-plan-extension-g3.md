# Belief-object cuRobo planning: HANG and sweep extension

**Question:** can a belief-backed cuRobo world and gripper attachment improve the existing skills without changing their rates or contact commands?

**Setup:** source 79bb72b3040b8e205b62440107fe5af6aedec00c, compute6, physics-only G3 mode A; ten paired settled dev layouts each for hang_mugs and sweep_blocks. Both arms fix plan_solver=curobo; A uses plan_world=table, B objects. HANG probes and all other fields retain defaults. [Inputs/configuration](2026-11-06-plan-extension-config.json) were frozen before outcomes. This is a separate-source extension of the [original study](2026-11-06-plan-curobo-g3.md), requested by the lead after its null ceiling results. No pooling, reserved IDs, camera-policy episodes or model calls.

The native skill planner already checks objects and held geometry; this experiment changes the free-space proposer. cuRobo receives believed OBBs/container parts and attached-box covering spheres, then the existing Collision model validates its proposals. See the [journal](../journal/2026-11-06-dojo-plan-world.md) and [worker incident](../incidents/2026-11-06-plan-curobo-worker.md).

HANG uses v15 variant-2 layouts 100030–39; sweep uses the shared v13 settled
panel 100000–04 and 100010–14. Generators are fixed within each paired cell.
The new HANG table baseline is 61.5 / 4 successes, compared with the earlier
[HANG study](2026-11-06-hang-skill-ceiling.md)'s 67.5 / 5. The difference is
layout 100038 (100→40). Its cause is not established. Use the new same-source
control, and do not pool or substitute the historical baseline.

[Machine-readable results and provenance](2026-11-06-plan-extension-results.json).

## Aggregate

Score is 0–100, with sample SD. Times are observed wall seconds on a shared machine, not a controlled latency benchmark. Planning time includes all candidate search; query/validation and cold startup are subsets.

| Cell | Arm | n | Success | Score ± SD | Mean / median steps | Planning s | Query s | Startup s | Fallback rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hang_mugs | table | 10 | 4/10 | 61.50 ± 34.00 | 355.6 / 344.0 | 1576.9 | 0 | 0 | n/a |
| hang_mugs | objects | 10 | 4/10 | 57.50 ± 38.82 | 356.9 / 338.5 | 1585.2 | 88.31 | 78.77 | 93.7% (743/793) |
| sweep_blocks | table | 10 | 2/10 | 20.00 ± 42.16 | 491.5 / 494.5 | 1304.5 | 0 | 0 | n/a |
| sweep_blocks | objects | 10 | 2/10 | 20.00 ± 42.16 | 417.3 / 431.5 | 1607.0 | 82.56 | 84.28 | 45.2% (47/104) |

## Per-layout pairs

| Cell | ID | A score | B score | A steps | B steps |
|---|---:|---:|---:|---:|---:|
| hang_mugs | 100030 | 100 | 100 | 325 | 329 |
| hang_mugs | 100031 | 100 | 0 | 396 | 328 |
| hang_mugs | 100032 | 15 | 40 | 309 | 319 |
| hang_mugs | 100033 | 40 | 40 | 472 | 472 |
| hang_mugs | 100034 | 40 | 15 | 376 | 417 |
| hang_mugs | 100035 | 100 | 100 | 357 | 361 |
| hang_mugs | 100036 | 40 | 40 | 272 | 272 |
| hang_mugs | 100037 | 100 | 100 | 331 | 348 |
| hang_mugs | 100038 | 40 | 100 | 404 | 406 |
| hang_mugs | 100039 | 40 | 40 | 314 | 317 |
| sweep_blocks | 100000 | 100 | 100 | 573 | 584 |
| sweep_blocks | 100001 | 0 | 0 | 105 | 105 |
| sweep_blocks | 100002 | 0 | 0 | 666 | 410 |
| sweep_blocks | 100003 | 0 | 0 | 441 | 486 |
| sweep_blocks | 100004 | 0 | 0 | 132 | 133 |
| sweep_blocks | 100010 | 0 | 0 | 972 | 369 |
| sweep_blocks | 100011 | 0 | 0 | 407 | 404 |
| sweep_blocks | 100012 | 0 | 0 | 507 | 453 |
| sweep_blocks | 100013 | 100 | 100 | 630 | 630 |
| sweep_blocks | 100014 | 0 | 0 | 482 | 599 |

## Paired interpretation

- hang_mugs: delta -4.00 points, paired SD 40.40; score wins/losses 2/2; success flips 1/1.
- sweep_blocks: delta +0.00 points, paired SD 0.00; score wins/losses 0/0; success flips 0/0.

Neither cell raises its tested ceiling. HANG loses four score points with one
full-success flip in each direction; sweep ties all ten scores. No new camera
A/B specification is triggered by this extension. Defaults stay off.

## Planner coverage and failure limits

Counts include speculative candidate plans; an accepted proposal is not necessarily an executed move. A worker deadline stops the owned worker and caches legacy fallback for the remainder of that Motion instance (normally the episode). Thus repeated timeout-labelled fallbacks are not repeated expensive GPU calls.

| Cell | Worker-backed requests | Accepted proposals | Actual worker deadlines | Cached fallbacks |
|---|---:|---:|---:|---:|
| hang_mugs | 110 | 50 | 10 | 683 |
| sweep_blocks | 97 | 57 | 2 | 7 |

The two-second GPU deadline is separate from up-to-60-second cold startup. Shutdown and individual CPU collision checks can add a small overrun; maximum observed query/validation duration is retained in the JSON. The robot remains still while planning. The existing safety envelope, contact/explicit-joint semantics, step limits and rate/easing rules are unchanged. The fixed mounting base is omitted from the GPU model; the original skill model validates every accepted path. No other-arm or continuous-collision guarantee is claimed. Held spheres conservatively cover the full OBB; mesh concavities are filled. The current data do not isolate representation rejection from the budget and cached-fallback limits.

Failed skill events remain slips, missed placements and unreachable pairs as recorded below. They overlap retries and are not an exclusive object-level causal taxonomy; an in-hand shift alone does not prove which contact caused it.

- hang_mugs failed-skill events: A {'grasp_slip': 1, 'place_miss': 2}; B {'grasp_slip': 2, 'place_miss': 4, 'collision': 1}.

- sweep_blocks failed-skill events: A {'grasp_slip': 2, 'place_miss': 19}; B {'grasp_slip': 2, 'place_miss': 20, 'collision': 2}.

## Spend and artifacts

40 completed G3 episodes; recorded episode wall 7267.6 s and load 206.7 s (2.08 lane-hours), plus app startup, API probes, cleanup and queue overhead. **0 LLM calls, 0 tokens, $0 model spend.** No camera-policy run or default change. Raw layouts, actions, logs and process receipts stay in compute6:~/dojo-data/plan/extension. Admission: used VRAM <=20 GiB, available RAM >=12 GiB, load1 <=26; at most four PLAN jobs, light locks only, 8 GB/400% CPU per job. No compute3 job was started. Local writes remain below 1 GB. Physics-only source episodes have no camera frames; any separately rendered action replay is diagnostic and excluded from these outcomes.


## Stage-event audit and diagnostic replays

The existing HANG audit helper counts 36→38 pick attempts, 7→10 loose picks,
1→2 failed lifts, 2→1 no-plan exits and 1→2 reported hang misses. Neither arm
exhausted its budget. These events overlap retries and differ from official
checker outcomes. Original phase traces were not enabled, so the audit cannot
classify carry/insertion/release losses; no such causal attribution is made.
Counts and this limitation are retained in the JSON.

Recorded-action replays show both full-success flip directions:
- layout 100031: [table success](../video/plan-world-hang-table-success.mp4)
  versus [objects failure](../video/plan-world-hang-objects-failure.mp4);
- layout 100038: [objects success](../video/plan-world-hang-objects-success.mp4),
  while the primary table episode scored 40.

All three clips reproduce their source success/score (100, 0, 100). They use
real Isaac RGB and scripts/turn_video.py, contain no model call, and are excluded
from the G3 sample. They are diagnostic replays, not original camera frames or
a camera-policy comparison. Source action hashes and render metadata remain
under the private extension replay-video directory.

All four G3 chains, 20 recorded worker PIDs and the replay process exited.
Robot/cuRobo fingerprints match before/after the extension. All eight G3
admissions pass: maximum used VRAM 18568 MiB, minimum available RAM 38981 MiB,
maximum load1 23.99. The separate replay also uses the strict admission wrapper.
CI on integrated controller/docs head 62c677ae passed 1932 tests / 191 skipped;
43 focused planner/hook checks also passed after HANG integration.

Reproduce with the frozen config/source and the primary report's G3 entrypoint,
using extension/layouts/<task>, the config's ten IDs, and unique output paths.
Use identical plan_solver=curobo and all default skill switches in both arms;
vary only plan_world. Do not overwrite the retained episode records.
