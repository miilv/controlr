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

## The wide search (`post_place_read` "6dof_search", PR #213) and the per-task table

The local read reaches only what stays within ~50 mm of the release. The ceiling with the oracle restricted to that
(`e2_oracle_m95.py --max-mm 50`) is +16 of 817 latest-round events, against +57 for the full oracle.

Where the rest goes, latest rounds, 369 releases:

| where the object ended | releases |
|---|---|
| within 50 mm of the release | 272 |
| fell 50-150 mm away | 45 |
| still where it was grasped (a missed grasp) | 21 |
| elsewhere | 31 |

When the read finds nothing at the release (`fit_poor` / `no_models` / `no_views` / `out_of_range` / `one_far_view`),
`release_read.release_search` tries two things in order:
1. **The pose the object had when the jaws closed** (`belief.grasp_z`). It is taken when the head's static-background
   foreground covers its silhouette there (prec ≥ 0.6), or when a head (x, y, yaw) fit stays within 20 mm at
   IoU ≥ 0.65.
2. **A head-view search within 150 mm of the release.** It looks at foreground no other believed object explains,
   blobs nearest first, each fitted with the asset's resting shapes. A wrist that sees the fit polishes it with the
   6-DoF read; otherwise a head fit at IoU ≥ 0.65 that covers ≥ 50 % of its blob is taken.

Composition with stream H's grip_check (#209), covered by a unit test:
- **Closed on air:** grip_check puts the object back at its grasp pose and releases nothing, so no read runs.
- **A slip:** grip_check's drop under the jaws is the read's prior, and the object moves once.

E2 on stream C's renders:
- 31 episodes, 276 events scored in every arm (12 changed attribution), R10.
- Same code and metric for every task (`e2_final_{a,b,c}.json`).
- 0 LLM calls, GT-free.

Clean events at R10, rt = the runtime's own belief:

| task (episodes) | events | rt | oracle | search | oracle fixed / broken | search fixed / broken |
|---|---|---|---|---|---|---|
| play_tic_tac_toe (4) | 92 | 48 | 65 | 69 | +17 / -0 | +21 / -0 |
| play_Xylophone (3) | 12 | 6 | 12 | 6 | +6 / -0 | +0 / -0 |
| store_tools_in_toolbox (3) | 22 | 10 | 12 | 12 | +2 / -0 | +2 / -0 |
| hang_mugs (5) | 24 | 14 | 16 | 14 | +2 / -0 | +0 / -0 |
| fasten_screws (3) | 24 | 18 | 19 | 18 | +1 / -0 | +0 / -0 |
| fill_pen_holder (3) | 16 | 9 | 9 | 10 | +0 / -0 | +1 / -0 |
| sweep_blocks (3) | 13 | 9 | 9 | 9 | +0 / -0 | +0 / -0 |
| build_tower (3), organize (5), stack_bowls (2), deposit_coin (4) | 86 | 74 | 74 | 74 | +0 / -0 | +0 / -0 |
| **all** | 289 | 188 (65.1 %) | 216 (74.7 %) | **212 (73.4 %)** | +28 / -0 | **+24 / -0** |

Other radii:

| R | rt | search | oracle |
|---|---|---|---|
| R5 | 198 | 218 (+20 / -0) | 224 |
| R15 | 172 | 198 (+27 / -1) | 202 |

Reading the table:
- **The search realises 24 of the oracle's 28 events at R10 and breaks none.**
- **tic_tac_toe (+21, more than the oracle's +17):**
  - The runtime "placed" player_piece0 again and again after missed grasps. The pre-grasp check finds it still in
    the pool 13 times (head prec 0.89-0.96, IoU 0.84-0.88).
  - The oracle snaps the released piece onto the nearest GT piece of its group, often another piece already on the
    board, so it is not a true ceiling for interchangeable pieces.
- **Xylophone (0 of 6):** the mallet is found at its grasp pose, but that pose was itself 20-27 mm off: a thin
  object, ~5 head px. Fasten nuts are ~10 head px. Both are in BACKLOG ("needs a wrist look").
- **hang_mugs (0 of 2):** with M's newer tracker (#205, it applies post_place moves) the remaining two events are not
  fixed. The earlier tracker scored 1 of 2.
- **sweep_blocks:** the search finds the broom 46 → 0.5 mm and 9 → 1.3 mm, but the oracle gains nothing on these
  three episodes. The sweep frames are the 6.0 render of M's reproduced physics of genuine-5.1 runs.

Reads (119):

| outcome | count |
|---|---|
| local 6-DoF read | 24 |
| pre-grasp | 23 |
| head search | 4 |
| no_gain (prior already right) | 14 |
| not found | 54 |

Cost: median 1.85 s per release, max 5.2 s.

Next: the lead's rendered arm 9 (submit_v3, `post_place_read` "6dof_search" vs off, ttt / hang / fasten / fill_pen /
store_tools / sweep + build_tower as a no-regression check).

Data: compute3 `~/dojo-data/w-wrist/e2_final_{a,b,c}.json` (`e2_report.py` pools them), `e2m_oracle50.jsonl`.

## Rendered arm 9 and the fix (fix/release-read-small)

Integration's arm 9 (compute4, submit_v3, `post_place_read` "6dof_search" vs off, 35 pairs) lost fasten_screws: 5/5 -> 1/5
(overall 13/35 vs 11/35). The reviewer's trace (runs2 `c4b_screws_rs*` vs `rsoff*`, dev_belief_probe):
- the screwing skill turns the nut in place, releasing and re-closing; its post-place refine was ok at iou 0.69-0.78, below
  `post_place_read_iou` 0.85, so the 6-DoF read ran, fit poorly (0.68-0.74), and the search's "never left" seed (the pose at
  the close = the nut one turn higher) wrote the nut back up 3 mm, twice; the thread-on then read "belief_mismatch";
- another episode: a head-only search moved a ~10 px nut 107 mm (place miss); one "pre_grasp" write had head IoU 0.0
  (accepted on foreground precision alone).
The offline E2 could not see it: the metric scores events at the task tolerance (15 mm) and replays keep the recorded
actions, so a 3 mm creep is neutral there.

Rules now:
- an ok post-place refine is never re-read (`post_place_read_iou` default 0.0; 0.85 keeps the old trigger for an A/B);
- foreground precision never places an object: the head fit must explain the view (IoU >= 0.65);
- a small object (mesh: every side < 45 mm, or the thinnest < 5 mm: nuts, coins, blocks) is never placed from the head
  alone: the wrist's 6-DoF read must agree;
- "never left" (the pose at the close) is a seed only after a real lift (the TCP rose >= 3 cm with the object attached:
  `KinematicBelief.released_lift`) that nothing showed the object rose with (no grip_check / held_look "held" in that
  grasp). An in-place turn has no lift.
Regression test: `test_release_read_never_ratchets_a_nut_being_threaded` (the recorded numbers of rs_100000 nut1 @684).
The A1 base drops 6dof_search; these rules carry into A1's write path.
