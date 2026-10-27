# 2026-10-27 — Perception-2 (round 3): agentic overrides, merged blobs, failure attribution

Stream: Perception-2 (perception core). The lead's brief and the round rules are in `/tmp/dojo-team/`. No LLM calls in
this stream: every agentic number below replays logged agent answers.

## 1. Conveyor (handed to Memory)

The conveyor scenes are now the Memory stream's: background, region mapping and the tracker. What I found before the
handoff:
- **No table in the scene.** B1's backgrounds are the wooden-table median (`bg_folds`). In the conveyor scene the
  whole belt reads as foreground: one blob of 175k px, 57 % of the head image.
- **Regions read as world coordinates.** Labels resting on the belt (`relative_plane "conveyor/conveyor/0"`) have
  their xlim/ylim offsets from the conveyor root applied as world coordinates.
- **One label group for all five labels.** `label_groups` merges them because the candidate lists are identical.
  The group takes target0's placement, so `_placement_prior` gives `fixed_xy = (1.88, -0.13)`, and `_propose` drops
  every blob more than 8 cm from it.
- **Objects ride the belt.** target0 spawns at x -0.2 and is at 0.234 by frame 0, so a belt label's x range must
  cover the whole belt.
- **The belt top is about 0.768**, which is TABLE_Z + 3 mm.
- **Frame 0 always shows target0 at nearly the same pixel.** A median over frame-0 images keeps the object in the
  background.

## 2. The agentic perceiver overrode confident B1 poses (plug_in_charger)

**Cause (Precision's read of breadth2 plug: 3 of 5 chargers turned 80-180 deg, 23-47 mm off).**
- The agent's charger point is not snapped to B1's. It lies 23 mm away, and B1's calibrated risk is >= 0.2 (flagged
  ambiguous / near the home grippers).
- H2 (`seeded_refine`) then either:
  - rejects its fit, which keeps the agent's raw reading. That reading carries a yaw the agent TYPED in its own
    convention ("the prong direction as the heading": -118.8 deg against B1's correct -15.3); or
  - fits from that yaw into a flip (100003: -158.8 deg, 47 mm).
- `merge_into_record` adopts any agent pose unconditionally.

**The same mechanism elsewhere (all 243 logged calls, breadth1 + breadth2).**
- Logged runtime result: the agent changed a label's pose 100 times. 26 better, 35 worse, 14 both ok, 25 both gross
  (15 mm / 15 deg vs the settled truth).
- make_kong 100004 (agentic0): the agent re-laid the whole tile row with a wrong pitch and moved 14 exact config-slot
  tiles 21-28 mm. This is the lead's "make_kong 100004 roles miss".
- Raw agent points also carry the agent's head-plane height. 30 of 99 agent-changed objects were 15-154 mm off in z
  (organize_table's garage figure 80-100 mm under the table, the drawer 154 mm up, deposit_coin's coin -16 mm).

**Fix (`seed_refine.arbitrate`, run by `j1.finish_answer` after H2; `AGENTIC_ARBITRATE`, default on).**
- **Height.** An unrefined agent point is set down so its lowest point is at its paired B1 object's (the same
  physical object). An unpaired point goes on the table.
- **Pose.** An agent pose that moves (> 3 mm) or turns (> 5 deg) one of B1's objects is scored against B1's pose by
  silhouette loss in frame 0 + the scan views. Each pose is scored in its own scene: the other objects of the same
  reading are masked out.
  - The agent's pose stands only when its loss is lower.
  - An object B1 placed at its config slot (`flags.slot`) needs a 0.15 lower loss.
- `finish_answer` is the old post-processing of `perceive_impl`, factored out. `agentic/replay.py` replays logged calls
  through it with no LLM.

## 3. build_tower: attribution of the scan-trigger losses, and the merged-blob fix

**Attribution.** The 7 perception_gross losses of the trigger arm (`B_build_scantrigger_*`, 3ac3597, before #62),
with identical pillars matched by position:
- **Phantom stacked pillar** (z +38 mm, added by the scan-primary record): 100011, 100014, 100022. #62 removes this
  class.
- **Missing piece**: 100000, 100010, 100022, 100024. In all four the piece is fully visible: known 0.83-1.00,
  foreground 0.96-1.00 in the head and wrist views, and no estimated object covers it.
  - 100024 is already fixed on main.
- **Shape / place confusion**: 100008.

**Causes of the missing pieces (main B1 on the runs' saved frame 0).**
- The pinned base boards (block0 / block7, config slots) are placed before the blob proposals, but their pixels stay
  in the head foreground. A piece touching a board merges into the board's blob, more than 3x the piece's silhouette,
  and `_propose` skips such blobs.
- Three pieces within 10 cm (100022) form one blob that no candidate can take whole.

**Fix (perception core; `EstimatorConfig.slot_blob_cut`, `split_merged`, both default on).**
- The pinned objects' silhouettes are cut out of the head foreground before blobbing, and in the residual rounds and
  the unexplained list.
- A blob larger than 2.5x the largest pending candidate's silhouette also proposes object-sized pieces
  (`_split_blob`). The whole blob stays a proposal.

## 4. Catalogue items (PR #82, merged)

- **Egg holder links.** `catalog_check` now fails on an egg holder / toaster / laptop candidate without
  `links.json` (the skills' lid hinge / lever). `--sync` copies it from a source catalogue, else writes it from the
  USD. SpringButton is exempt: it is pressed at perception's annotated point, and adding a file would change what
  press_by_number loads.
- **Xylophone crash.** `policy._search_refine` keyed every record entry as `Rigid/<cat>`, so the Geometry xylophone
  crashed refine. It now keeps the entry's type.
- **Box catalogues.** compute3 and compute5-8 are identical after the lead's copy: 1732 files, md5 227ab515069b,
  egg-holder links 5bcd2513, coin-stand support frame present. `catalog_check --sync --assets` changed nothing on any
  box.

## 5. PR #86 review (Memory's conveyor)

- OK from the perception side.
- Not measured: the non-belt groups whose regions now get mapped (fill_egg_holder eggs, store_laptop, solve_equation
  mats, the by_image fixtures).
- Moved-wrist backgrounds are still warped through the table plane.
- `j1_adapter` (the robot-harness server's perceive route) never applies `with_scene`.

## 6. build_tower merged blobs: results (report: experiments/2026-10-27-tower-merged-blobs.md)

- **d0_51 build_tower:** within tolerance 164 -> 173 / 200, missed 13 -> 4, 0 lost. 17 other G1 sets are identical
  object for object.
- **The first version** also cut the pinned objects' pixels in the residual rounds. That cost organize_table 100000's
  figure in front of the pinned cushion (0.4 -> 40 mm), so the final version keeps those rounds as on main.
- **An additive variant** (proposals from both the cut and the uncut blobs) was worse on the tower: 170, with 6 new
  gross pillars.
- **New defaults:** `EstimatorConfig.slot_blob_cut` and `split_merged`, both on.

## 7. hang_mugs: wrong mug models (B1's identity polish)

**Traced `_polish` on breadth2 100004 (GL, compute6).**
- Silhouettes are flat across the 10 mug models (loss 0.30-0.36). Only the mean colour separates them.
- **Fault 1: paired texture overrides a large colour difference.** It weighs 0.8 against mean colour's 0.3, and it
  turned the white mug 00004 (distance 0.10) into the dark 00002 (0.46).
- **Fault 2: the pink mug 00006 never made the 4 polish trials.** They are pre-ranked by area ratio + 0.3 x blob
  colour against a big wrong incumbent (00000), and 00006 came 5th of 9. Scored at that spot, 00006 has the lowest
  total cost (0.475 vs 0.66).

**Fix (branch feat/dojo-perception2-mug-identity).**
- Trials are ranked by the colour inside the incumbent's silhouette, plus the 2 best-coloured candidates.
- Paired texture cannot pick a trial whose mean colour is > 0.2 worse.

**Breadth frames.**
- 100004: 1/3 -> 3/3 models right.
- 100003's cream mug, half behind the right gripper, still reads 00002.
- 100002's missing mug is a different case.

**Next.** G1 regression on all sets is running. A hang_mugs D0 render was requested from the lead (heavy lock).

## 8. play_Xylophone: the mallet offsets came from the agent, and #92 removes them

**Tools' rendered 5.1 run** (compute7 `xylo_r51_main`, 2a6e30d, before #92): 3/10. In 5 of the 7 failures the mallet
was "found" 93-100 mm off.

**Per record stage.**
- B1's mallet is right at initial / scan (0-5 mm).
- The agentic step moves it 95-100 mm, at yaw 0 or ~90. The agent's point is the mallet's visual middle; the merge
  takes it as the asset origin, which sits near one end of the 213 mm mallet.

**Replay of the run's 17 agentic calls through main's arbitration** (5.1 frames, GL, no LLM). Mallet error:
- 100005: 97 -> 1 mm
- 100006: 100 -> 0 mm
- 100008: 99 -> 4 mm
- 100002 agentic2: 95 -> 5 mm
- 100009: 95 -> 30 mm (B1's own error)
- 100003: 99 -> 23 mm (B1's own error)

**Genuine B1 cases left.**
- 100001: the stand 75 mm off at frame 0; the slot reads empty; the search finds the mallet.
- 100007: missing at frame 0, fine after the scan.
- 100009: a 180-deg stand flip in the frame-0 replay.

**Tried, not kept.** A support-slot base kept only in place frames whose slot faces up, plus the assembly's own
symmetry. It changed nothing on the 10 frames: the stand's "up_side" frame is a valid upright pose, and the bad cases
are yaw / position errors.

## 9. refine(): the "ok for a wrong fit" item re-measured (BACKLOG P1)

- **The item's own set** (6.0 D0, frames 1 / 2 / L1, build_tower + stack_blocks, priors +-20 mm, `WITH_OTHERS=1`, main):
  ok-but->15 mm fell from 10 of 291 to 3 of 271 ok fits.
  - The one gross case is build_tower 100009 L1 block2, 62 mm at iou 0.67. The runtime's big-shift bar (iou >= 0.7
    above 30 mm) refuses it.
  - The others are 15 and 25 mm.
- **Genuine 5.1 d0_51** (frames 1 / 2; tower, stack_blocks, bowls, push_T; 334 fits): ok 321, none beyond 15 mm, 4 at
  10-25 mm.
- The item stays open as a watch item only.
