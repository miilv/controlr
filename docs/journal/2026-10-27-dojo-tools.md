# 2026-10-27: dojo tools-articulated: the runtime belief dropped every narrow pick; eggs into the egg holder

Stream: Tools-Articulated (round 3): fill_egg_holder, play_Xylophone, make_toast, hang_mugs. Branches: `fix/dojo-runtime-narrow-attach`
(PR #84, merged), `feat/dojo-tools-egg` (this PR), `feat/dojo-tools-xylo` (an experiment, not merged), `feat/dojo-tools-hang-held`
(the hang analysis on #77 + #79).

Boxes:
- compute7 light slots 1-4 and heavy lock 1 (shared with Official-Path);
- before compute7 was ready: compute3 overflow slot 4.

Tiers:
- G3 = the TRUE state in the physics-only twin;
- G3 mode B = `g3 run --mode-b [--held-belief] [--belief-offset ...]`;
- proxy B = breadth2 (`submit_v1`, compute3, Isaac 6.0 render);
- rendered B = `integ_eval.sh --renderer isaac51 --preset submit_v1` (genuine 5.1, compute7).

## 1. The runtime belief never attached a narrow pick (PR #84)

**Symptom.** breadth2 (proxy B, 5 layouts per task) scored 0/5 on three tasks:
- make_toast: grasp_slip 3;
- play_Xylophone: grasp_slip 2;
- hang_mugs: grasp_slip in most picks.

The failing picks all read "object rose 0 mm (planned L), in-hand drift ~L mm". The beliefs were good:

| object | belief error |
|---|---|
| bread | 0.9-2.7 mm |
| mugs | 0.6-1.9 mm |
| the mallet | 2.6 mm / 4.8 deg |

The commanded grasp of toast 100002 was within 1.5 mm of G3 mode A's grasp on the same seed, which held.

**Cause.** `KinematicBelief.step` attached an object only when the gripper command crossed 0.5 from above. A narrow pick opens to less than 0.5 before it closes:
- a bread slice: 0.37;
- a 30 mm handle opening: 0.42-0.46;
- a 3-5 mm mug wall: 0.42-0.44.

So the object was never attached, and what followed was:
1. the belief left it on the table;
2. `verify_pick` read "rose 0 mm, drift = lift" and called it grasp_slip;
3. the bank opened the gripper over a held object.

G3's `heldsim` attaches on any close below 0.5, so G3 mode B never showed this.

**Evidence: breadth2 picks with the belief-stuck signature** (rose ~0 mm, drift = the planned lift):

| task | belief-stuck picks |
|---|---|
| hang_mugs | 21 / 24 |
| make_toast | 18 / 20 |
| play_Xylophone | 8 / 8 |
| the other 14 tasks | 0 |

In hang, every 15.2 mm wall pinch (37 mm gap, g 0.53) held, and every 3-5 mm wall pinch failed.

**Evidence: replays.** breadth2's proxy-B actions, replayed in the physics-only twin from the same layouts (`replay_probe.py`, compute7):

| episode | in the twin |
|---|---|
| toast 100002 | bread_0 rose +125 mm with the hand (planned 127), then fell at the bank 2 steps after the false grasp_slip; bread_1 the same (+178) |
| hang 100004 | mug2 rose 0.803 -> 0.898 m, dropped at the re-open |
| xylophone 100002 and 100004 | the mallet rose with the hand on both picks (100004's belief was 12.3 mm / 7.2 deg off), then dropped |

**Fix (PR #84, merged).**
- Attach on any closing command below 0.5, as heldsim does.
- Release on an opening command above 0.6, or 0.15 above the tightest command since the grasp. That is Integration-3's review: a narrow release must let go.
- Its unit test fails on main at g_open 0.46.

**After.** Rendered B, play_Xylophone on main 2a6e30d, settled layouts 100000-04: 2/5 (100000 and 100004 succeed). breadth2 had 0/5 on the proxy tier.

All three failures are perception:
- 100001: the mallet was not found;
- 100003: the mallet was 93 mm and 90 deg off;
- 100002: the mallet was believed on its stand, 88 mm off; the jaws closed on nothing, and the held-look saw the empty jaws (opening 0.0) yet the runtime kept the grasp.

(layouts 100005-09: see below)

## 2. play_Xylophone: the experiment branch is not merged

`feat/dojo-tools-xylo` tested two mode-B-only changes:
- handle grasps opened to 70 mm;
- the believed in-hand transform snapped to the pads' constraint (handle centred, its axis perpendicular to the closing axis).

| G3 mode B (--held-belief), 10 layouts | main | branch |
|---|---|---|
| mallet belief 3 mm + 5 deg off | 10/10 | 10/10 |
| mallet belief 9 mm + 8 mm + 8 deg off | **10/10** | 8/10 |

The narrow handle grasp already tolerates these errors; the mode-B misses were #84. The branch stays unmerged. G3's `--belief-offset LABEL:dx,dy,dyaw` (a yaw error about the bbox centre) is in this PR, because it is useful for any long tool.

## 3. make_toast

presettle dropped every make_toast record ("settled record not stable"), so G3 runs on raw records, as skills-4 did. Main 6dd647c, layouts 100000-09:

| tier | success | mean |
|---|---|---|
| G3 mode A | 9/10 | 0.925 |
| G3 mode B (bread_0..3 and the shelf 2 mm, 2 mm, 1.5 deg off: the shelf's error is shared by its slices) | 7/10 | 0.725 |

With #84 the toast skill needs no change. The rendered pair is Integration-3's.

## 4. fill_egg_holder (Long-Horizon 2.5 %, best entry 21.2)

Report: [experiments/2026-10-27-egg-holder-drop.md](../experiments/2026-10-27-egg-holder-drop.md). G3 mode A, the 11 settled
layouts: 3 -> 17 eggs in, mean 0.027 -> 0.186.

- **Why skills-6 scored 0.03.** Its traces, replayed offline:
  - hands stopped on the basket rim on the way down: the finger root block straddled the rim between the corner samples;
  - transits ran through the rim: the basket, a Geometry object, was no obstacle;
  - eggs fell out while being turned upright.
- **Built.**
  - Picks are checked with dense finger samples against the wall profile (calibrated on those traces).
  - The basket becomes a height-map obstacle once the egg is over its rim.
  - The egg is carried lying and let go 12 mm off a hole's centre, so it tips in.
  - An egg against the wall is pushed off it first; one left lying across a hole is pressed.
- **Tried and dropped** (no gain on the same layouts):
  - drops ordered toward the outer rim, with the egg turned about its own axis: 11 eggs;
  - a wall check of -13 mm: 12 eggs;
  - a second press: no change.
- **Left.**
  - Eggs on the wall's foot with neither a pick nor a nudge plan.
  - Eggs under the basket's flared rim, on the table.
  - Eggs wedged across a hole.
  - Mode B: the runtime predicts a released egg on the table, because the holder is not a fixture (BACKLOG).

## 5. hang_mugs: no "held" rule can see a mug fall; a fallen mug is retried

The lead asked for a fix to #77's fixture "held" rule. Integration-3's G3 found mugs that fell read "held by a fixture".

**Measurement.** G3 mode B, `--belief-drop fixtures`, Integration-3's offsets, every release's believed pose and the truth logged (heldsim events). 12 releases, layouts 100000-03:
- the believed handle frame was 7-15 mm from a hook frame in all 12;
- 7 mugs hung (3.7-24 mm off the belief);
- 5 fell (63-342 mm).

The believed distances overlap completely, so no rule on the believed pose can separate them. #77's rule says "held" for the wrong reason: every mug is under the rack's upper arms or pole top. It never changed an outcome here, so it stays.

**What can help.**
- (a) The runtime's release resync (frames vs the predicted state). Integration-3 checks its divergence events on the rendered hang pairs.
- (b) G3 `--belief-drop fixtures_look` emulates it: a fixture-held prediction more than 20 mm off the truth is re-estimated.
- (c) `hang_on_hooks` retries a mug that is not on its hook after the release, once. In G3 mode A the retried mugs lay on their side, and none had a plan (2 retries, 0 hung). See the PR numbers.

## Spend

- LLM calls: only the rendered xylophone episodes (submit_v1's agentic perceiver).
- GPU: compute7 light slots 1-4, compute7 heavy lock 1, compute3 overflow slot 4. No downloads.

## 6. After #98: the confirmation, rendered xylophone, eggs in mode B (later the same night)

**#98 confirmation.** compute7, G3 mode A, the same 11 layouts:
- main 5878646: 2 eggs (0.018);
- the merged config: 16 eggs (0.173); its first run had 17 (0.186), so the run-to-run noise is about 1 egg;
- paired: 9 layouts up, 1 down, 1 same.

**Rendered xylophone (genuine 5.1, submit_v1, compute7).**

| main | layouts | success | failures |
|---|---|---|---|
| 6dd647c (before #84) | 100000-04 | 0/5 | four "rose 0 mm, drift 57 mm" |
| 2a6e30d (#84) | 100000-09 | 3/10 | every miss was perception, six of them the mallet 93-100 mm off |
| 1f7e965 (#84 + #92 + #93) | 100000-04 | 4/5 | |

- 2a6e30d: Perception-2 traced the 93-100 mm offsets to the agentic perceiver merging the mallet's visual middle as its origin. #92 fixes that.
- 1f7e965: the only miss is the mallet not found.
- 100005-09 on 1f7e965 waits for the heavy lock (yielded to Official-Path until 05:00Z).

**Eggs in G3 mode B** (`--held-belief --belief-drop fixtures`, 2 mm belief offsets, 10 layouts): 8 eggs (0.077), against mode A's 16. GT traces of 3 episodes:
- at the release the true egg sits 10-14 mm lower in the pads than believed;
- it lands only ~6 mm off the hole centre instead of 12, so it wedges across the hole 3-4 mm over the plate;
- mode A re-measures the held egg after the pick and at the place; submit_v1 has held_pose off.

The release rule is not what limits mode B. It predicts the egg on the table under the hole's xy, which the skill's in-hole check happens to accept: no phantom re-picks. A post-drop `confirm_pose` of the expected tipped-in pose was adopted once in 10 layouts and changed no outcome (8 eggs either way).

**The slide (follow-up PR).** An egg let go across its hole is pushed along its axis toward the side it was let go on, until one end is over the hole. In mode B this happens after every drop, from the release pose; in mode A, only when the egg is measured lying across. The wall nudge may also graze the wall profile by 20 mm.

| G3, the 11 layouts | mode B eggs | mode A eggs |
|---|---|---|
| main (#98) | 8; repeat 9 | 16 |
| + slide | 13 | 17 |
| + slide + nudge 20 mm | 12 | 20 |

Details: [experiments/2026-10-27-egg-holder-drop.md](../experiments/2026-10-27-egg-holder-drop.md).

Dropped variants:
- held_pose auto: no help. Precision is writing a generic show-pose fallback.
- Pushing at the believed middle height: the fingertips passed over the egg.
- The nearer arm: the other hand was in the way.

## 7. hang_mugs, the numbers

- **G3 mode A, 25 layouts:** main 0.440 (skills-7 had 0.44).
- **Retry of a mug not on its hook after the release:** 0.470 on 23 layouts. Paired: 2 up, 1 down, which is noise. Not merged.
  - Most retried mugs lay on their side.
  - A lying mug only plans with a rim-to-base pinch, and that pinch slips.
- **G3 mode B** (`--belief-drop fixtures`, Integration-3's offsets, 10 layouts):
  - fixtures: 0.370;
  - fixtures_look (the emulated release resync, 1-3 per episode, PR #117): 0.395;
  - fixtures_look + retry: 0.385.
- **Regression set on the retry branch:** plug 8/10 (the same two fail), kong 5/5, store_tools 0.45, stack 5/5. Every reference holds.

## 8. hang_mugs: why mugs fall in mode A, and a fallen mug cannot be stood up

**Where the 18 lost mugs of main's 25 layouts are lost** (place and hang notes):

| what | mugs |
|---|---|
| hung | 44 |
| slipped in the hand on the carry (place missed by 23-54 mm, tilted 27-81 deg) | 4 |
| fell at the release, the place within tolerance | 14 |

Of the 14 that fell at the release:
- 6 were placed within 0.2-2.6 mm, with an in-hand shift of at most 5 mm, and still fell after the slide, the lower and the release;
- the other 8 had shifted 11-35 mm in the hand (the slide uses the measured grasp).

So in mode A the hang motion itself loses ~1 mug in 4. That is the lever; traces of the falls are being read.

**Standing a fallen mug up: not reachable.** G3 snippet:
- a mug laid on its side in front of the rack;
- every box, antipodal and auto grasp;
- 32 upright place poses at 2 relay spots.

Result:
- right arm: 498 grasps, 0 pick + place pairs;
- the place IK fails for every approach within 13 deg of horizontal (5705 checks);
- the wrist limit fails 1255, and the palm hits the table 1204.

Standing a mug up puts the X5's wrist horizontal low over the table, and the X5 does not reach that. A re-orient through an in-air handover or a handle grasp would be needed. Both rank after the falls.

**The re-plan after #119.** Integration-3's fallen check re-plans the episode. `hang_on_hooks` then starts again with every mug, and would take hung mugs off their hooks. Fix (separate PR): mugs already on a hook count as done and keep their hooks.

## 9. The runtime belief dropped what was let go in an articulation to the table (follow-up PR)

Perception-3 found breadth3's make_toast failure: every failed place read "xy err 0.0 mm, dz -62.6 mm (centre 0.8231 vs 0.8858)". That is the belief's prediction, not perception:
- `drop_rest` drops a released slice to the table;
- the toaster is an Articulation, and #77's fixture rule takes static Geometry only;
- the skill then re-planned every slice.

The egg holder has the same landmine.

**Fix.** An articulation's static links are a fixture, in the runtime belief and in heldsim:
- the links that are no joint's child: the egg holder's body, the toaster's base;
- each mesh vertex goes to the link with the nearest `links.json` sample;
- the lid and the lever are left out.

**G3 mode B, `--belief-drop fixtures`, main e3e4bb5 vs the fix:**

| task | layouts | score | places ok | belief on the table |
|---|---|---|---|---|
| make_toast (raw records, 2 mm / 1.5 deg offsets) | 10 | 0.725 -> 0.725 | 7/19 -> 19/19 | 19 -> 0 |
| fill_egg_holder (2 mm offsets) | 11 | 12 eggs -> 12 eggs | | |

- G3 never failed on the wrong belief: it re-measures after a place. The rendered runs did, so the rendered pair decides.
- Eggs: the score is unchanged, with 25 picks instead of 21. The belief now puts an egg let go over a hole lying across it, 2 mm into the hole, and the skill tries again more often.

