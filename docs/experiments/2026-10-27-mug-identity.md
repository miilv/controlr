# 2026-10-27 — hang_mugs: wrong mug models from B1's identity polish

**Question.** Breadth-2 hang_mugs (proxy B) was attributed "mug0 / mug1 identity swapped in 3/5".
- Which of those are real perception losses?
- Why does B1 read the wrong mug model?
- Does a fix in the identity polish hold up on the G1 sets?

**Data.**
- **Breadth-2 hang frames.** The look frames of the 5 episodes (6.0 proxy), frame 0 and frame 0 + scan.
- **hang_mugs D0 on genuine 5.1** (`d0_hang51`). 25 settled dev layouts, frame 0 + one left-wrist look, rendered for
  this report (make_d0, heavy lock on compute5).
- **G1 regression.** d0_51 and d0_ext.
- **Truth and matching.** Our settled layouts. Mugs are matched to truth by position (the three labels are
  interchangeable). Yaw is scored without symmetry: the handle matters; the catalogue index wrongly calls mugs
  z-symmetric.
- 0 LLM calls.

## Re-attribution of breadth-2

| layout | what B1 + the runtime read | real? |
|---|---|---|
| 100000, 100001 | all three mugs right after the scan (1-5 mm, 0-2 deg, right models); labels permuted | no: the report compared interchangeable labels by name (Integration-3 fixes report.py) |
| 100002 | one mug (00004) never found; a phantom 560 mm away | yes: missing mug |
| 100003 | the cream mug 00008, half behind the right home gripper, read as 00002 / 00003 | yes: wrong model |
| 100004 | the pink mug 00006 read as the larger cream 00000, 21-24 mm / 59-68 deg off | yes: wrong model drags the pose |

## Cause

`_polish` traced on 100004 (GL).
- **The silhouette cannot separate the 10 mug models** (loss 0.30-0.36 at the pink mug's place). The mean colour can:
  0.25 for 00006 against >= 0.54 for every other model.
- **Paired texture overrides a large colour difference.** It weighs 0.8 against the mean colour's 0.3, and it replaced
  the white mug 00004 (distance 0.10) with the dark 00002 (0.46).
- **The pink mug never made the 4 polish trials.** The trials are pre-ranked by area ratio + 0.3 x the blob's colour
  against a big wrong incumbent (00000), and 00006 ranked 5th of 9.

## Fix

Three `EstimatorConfig` fields:
- `polish_fit_color`: rank the trials by the colour inside the incumbent's own silhouette.
- `polish_color_candidates = 2`: also try the 2 best-coloured candidates the ranking left out.
- `polish_color_veto = 0.2`: paired texture cannot pick a trial whose mean colour is more than 0.2 worse.

## Results

**hang_mugs D0, genuine 5.1.** 25 layouts, 75 mugs, main vs this branch, paired. Right model = the catalogue model;
pose ok = <= 15 mm and <= 15 deg with the handle counted.

| frames | right model main -> fix | pose ok | model and pose ok | changed |
|---|---|---|---|---|
| frame 0 | **47 -> 54** | 42 -> 44 | **40 -> 43** | +7 models, 0 lost; +2 poses, 0 lost |
| frame 0 + wrist look | **52 -> 56** | 49 -> 50 | 43 -> 45 | +4 models, 0 lost; +1 pose, 0 lost |

**Breadth-2 frames (6.0 proxy, 15 mugs).**
- 100004 goes from 1/3 to 3/3 right models.
- 100003's cream mug, half behind the right gripper, still reads 00002.
- 100002's missing mug is a different case.

**First version, two losses.** In the first version a colour-vetoed trial fell through to the plain-loss comparison.
- The d0_ext regression lost organize_table 100000's garage figure (0.4 -> 148 mm) and one fill_pen identity.
- An ablation pinned both on the veto alone.
- Final version: a vetoed trial is skipped and the incumbent stays. Both losses are gone (rows below).

**G1 regression, main (5878646) vs this branch (final), same box, frame 0.**

| set | task | n | within tol main -> fix | missed | gross |
|---|---|---|---|---|---|
| d0_51 | put_bottles_into_dustbin | 125 | **104 -> 108** | 3 -> 3 | 11 -> 10 |
| d0_51 | build_tower / stack_blocks / push_T / stack_bowls | 200 / 75 / 50 / 75 | 164 / 60 / 41 / 67, identical | | |
| d0_51 | pack_objects_into_box_random | 125 | 89 -> 89 | 2 -> 2 | 28 -> 28 |
| d0_ext | organize_table / store_tools / general_pickup / fill_pen_holder | 90 / 50 / 10 / 50 | identical | | |
| d0_ext | make_kong / press_by_number / cover_blocks | 220 / 50 / 60 | identical | | |
| d0_ext | swap_T / align_blocks / plug_in_charger / insert_key / stack_blocks_by_language | 20 / 40 / 20 / 20 / 30 | identical | | |

- **put_bottles:** 6 bottles change class, all for the better (4 identity / overshoot -> ok, 1 gross -> ok, 1
  overshoot -> identity at 9 mm).
- **Every other set** is identical object for object.

## Caveats and next

- One right model of three is still missing in 21 of 75 frame-0 mugs. The worst cases are mugs half hidden by a home
  gripper; those need the look (+4 more with it) or the agent.
- The missing mug (breadth 100002) and the handle yaw when the model is right are not addressed here.
- The catalogue index's `symmetry_z = 36` for mugs hides handle yaw in index-based evals (eval_g1, breadth_eval). B1
  itself uses the shape cache (1).
