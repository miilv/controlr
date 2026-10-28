# 2026-10-28: the post-release pose of placed objects (stream W, E2)

Follows the wrist-check E1 (journal 2026-10-28-dojo-wrist-check): there, the "neighbour off 17-27 mm" failures were
objects let go earlier whose belief kept the PLANNED place pose. The lead moved stream W to the release pose.

## Ceiling: a perfect post-release read on stream M's replays

`scripts/perception/wrist/e2_oracle_m95.py` (GT only, every replay of M's set, 0 LLM calls):
- After every release (M's `episode_events`), the released object's belief becomes its GT pose ~5 steps later and
  stays until the report updates it again.
- Every grasp / place is scored with M's metric, exactly as `m95_score.py score` scores it (held drift / gap included).

| rounds | events | clean@R10 now | with a perfect release read | fixed / broken |
|---|---|---|---|---|
| latest per task | 810 | 543 (67.0 %) | 598 (73.8 %) | +57 / -2 |
| older | 2028 | 1441 (71.1 %) | 1534 (75.6 %) | +100 / -7 |

Latest rounds, by task (R10 fixed):
- play_tic_tac_toe +14 / 112
- sweep_blocks +10 / 47
- play_Xylophone +8 / 16
- fasten_screws +8 / 150
- fill_pen_holder +5 / 34
- hang_mugs +4 / 24
- store_tools +3 / 38
- deposit_coin, swap_blocks +2 each
- stack_blocks +1

The broken events are attribution artefacts. With the object where it really is, a later close no longer attributes
to it in M's rebuilt belief.

Caveat: the oracle finds the object wherever it went. A read near the release pose catches only what stayed within a
few cm of it.

## Why today's post-place refine misses it

`post_place` (on in `submit_v*`) re-fits the released object as a resting object: (x, y, yaw) on its support, at the
prior's height. Two consequences:
- A mug let go on a rack hook swings 14-38 degrees about the hook and lands 13-22 mm off. The hang_mugs post_place
  refines report "perception not ok" at iou 0-0.3.
- A mouse laid on a mousemat ends ~10 mm off with a 7-9 degree turn.

Side finding, now fixed in M's tracker (#205): the metric's belief tracker did not apply post_place moves, because the
event logged from / to and no pose7. The runtime now logs pose7 for every adopted post-place pose.

## The release read

`dojo/perception/release_read.py` reuses `held.held_pose`'s machinery, without the gripper.
- **Views:** the releasing arm's wrist, the head and the other wrist, each used when enough of the predicted
  silhouette is visible.
- **Unknown pixels:** the robot from FK, plus known objects in front of the released one (a z-buffer at the prior).
  Elsewhere another object counts as background, e.g. the mousemat around a mouse.
- **Colour model:** a two-class Lab model per view, learnt around the prior's silhouette, eroded by the expected
  release error.
- **Fit:** 6-DoF Nelder-Mead over the weighted soft-IoU loss. Starts swing ±20 / 40 degrees about world x / y and
  shift ±12 mm. The colour models are re-learnt at the fit, then a polish runs.
- **Acceptance (ok):**
  - best view IoU ≥ 0.75, every used view ≥ 0.70;
  - the releasing wrist among the views, or at least two views (the head alone moved a tower block from 1 to 22 mm);
  - the fit beats the prior's cost by 0.03, inside ±50 mm / 75 degrees.
- **Runtime:** `RuntimeConfig.post_place_read = "6dof"` (default off). Inside `post_place`, it runs when the refine is
  not ok or fits below `post_place_read_iou` (0.85). An ok read moves position and orientation and logs `read6` plus
  pose7. Service op `release_read`.

## E2 on stream C's frames

Setup: `scripts/perception/wrist/e2_release.py`, 19 episodes (build_tower 3, deposit_coin 4, hang_mugs 5, organize 5,
stack_bowls 2). The read runs on the frame of the runtime's own post_place step, else release + 8.

Reads (48):

| outcome | count | detail |
|---|---|---|
| ok | 13 | 9 improved by > 3 mm, 0 worse |
| no_gain | 17 | the prior was already right: build_tower blocks at 0.3-2 mm |
| fit_poor | 15 | mostly objects 80-700 mm from where they were let go: they fell or never arrived |
| one far view / no views | 3 | |

Improved reads:

| object | before | after |
|---|---|---|
| mug on a hook | 16 mm | 0.5 mm |
| mug on a hook | 19 mm | 1.9 mm |
| mouse on its mat (5 reads) | 9.6-10.5 mm | 0.4-1.9 mm |
| alarm in the drawer | 21 mm | 1.1 mm |
| bowl | 18 mm | 9 mm |

Events: clean@R10 88 → 89 of 110 events scored in every arm (+1 / -0). The oracle gives +2 / -0 on the same events,
so the read realises 1 of the 2. Most of these tasks' failures are not release poses.

Cost: median 1.9 s per read, max 3.4 s (GPU, GL silhouettes).

An earlier acceptance rule (ok_iou 0.70, any single view) broke 6 build_tower events: head-only fits moved accurate
blocks by ~20 mm. The view rules above came from that.

## Next

- Stream C renders per-step sequences of the high-ceiling tasks (tic_tac_toe, sweep, Xylophone, fasten_screws,
  fill_pen, store_tools).
- Report the realised fraction of the ceiling per task. That decides a rendered A/B.

Data: compute3 `~/dojo-data/w-wrist/` (`e2m_oracle.jsonl`, `e2_read_v3.json`).
