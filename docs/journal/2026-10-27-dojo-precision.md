# 2026-10-27: dojo precision: mode B's screw depth, the coin handover's hand, a relay for lying tubes

Stream: Precision (round 3). Tasks by points: plug_in_charger, deposit_coin, fasten_screws, insert_tubes (insert_key
moved to Pegs at 00:00Z). Branches from main: `feat/dojo-precision-screws`, `feat/dojo-precision-coin`,
`feat/dojo-precision-tubes`. G3 runs come from commit-stamped copies (`scripts/skills/deploy_skills2.sh DIR`,
`CONTROLR_HOST=` for the box) with `SIMLAB_USD_SYNC_LINKS=1`. They ran on compute3 light slot 4 until compute6 opened,
then on compute6 light slots 1-4. Data: `<box>:~/dojo-data/precision/g3/<run>_<commit>/`. G3 = the TRUE state in the
physics-only twin; G3 mode B = `--mode-b` plus the hooks named per row. No task names. No LLM calls so far.

## 1. breadth2 (proxy B, submit_v1, n 5 per task): what lost the points

Read from `compute3:~/dojo-data/integ/runs/B_<task>_breadth2/*_report.json.gz` against our settled layouts' truth.

- **plug_in_charger 0/5: perception and runtime, not skills.**
  - 3/5 (100000 / 01 / 03): the agentic perceiver (agentic0, then fused) replaced a correct B1 / scan charger with a
    wrong yaw:
    - scan -15.3 deg (truth -15.4) became -118.8;
    - scan 8.2 (truth 7.9) became -72.4;
    - B1 21.5 (truth 21.1) became -158.8 (a flip).
  - 4/5: a wrist refine that turned the charger more than 90 deg kept the prior orientation but took the flipped
    fit's root position.
    - The charger's root sits ~24 mm from its centre of symmetry, so the belief's centre went 47 mm off. The refine
      then flipped between the two modes on every re-hover.
    - The as-is relay put the real charger ~47 mm from the belief, and the other arm grasped air (both grasp_slips).
    - In 100003 / 100004 the flipped fit was the truth.
    - Routed by the lead: the runtime part went to Integration-3 (#80 keeps the fit's centre and refuses a big turn
      at low iou); the agentic yaw went to Perception-2.
  - `held_pose` is off in submit_v1, so every place ran the blind probing tail. G3 mode B: 2/10 without the read,
    8/10 with it (skills-9). An A/B on rendered 5.1 is approved (section 5).
  - 100002: no relay plan (it fails in G3 too).
- **deposit_coin 0/5: perception.**
  - "no rim grasp of coin0 (axis 2)": the belief has the coin lying flat at its stand (body z up) while it stands in
    the stand. #78 (the coin / stand pair in `slot_settle`) fixes exactly that when the stand is found.
  - The other 3: the coin was not found.
- **insert_tubes 3/5:** both failures are the third tube with no one-arm plan; 100002's "tube0 not found" did not
  decide it. Section 4.

## 2. fasten_screws mode B: the belief turned the nut down with the hand

Integration-3's G3 mode B on #77 + #79 (5ef91bb; 2 mm nut offsets, `--belief-drop fixtures`, 100000-09):
- 0/10, mean 0.39;
- every nut placed and screwed, `success_flag` true, `stages_left` 1;
- final poses "equal to mode A's".

Those poses were the belief: in mode B, g3's `final_objects` comes from `sk.refresh()`, which goes through the provider.

**Cause.** With #77's fixtures rule a released nut stays where the jaws let it go, so the belief moves it down with
the hand at every sweep. `screw_on` read its depth from that belief:

| sweep | mode A, measured (GT) | mode B, believed |
|---|---|---|
| 1 | 0.7-0.9 mm (the thread engages) | 2.6 mm |
| 2 | 2.2-2.5 mm | 2.7 mm |
| 3 | 2.5 mm | 2.8 mm |
| 4 | 2.4-2.5 mm | (stopped) |

- Mode B stopped after 3 sweeps at a believed 11.3 mm below the bolt's top. The truth was about 8.5 mm, and the
  checker wants the nut's bbox bottom more than 9 mm under the bolt's bbox top. One or more nuts fail it (score 0.5 /
  0.2 / 0.0), and the success stage never latches.
- The belief also drove the pitch re-measure (3.12 -> 3.35 mm).

**Fix** (`precision.screw_on`, mode B only: `not sk.remeasure`; the mode-A branch is unchanged):
- the nut's depth is the lower of the belief's and a sweep model's: seated at most `SCREW_SEAT_MAX` (4 mm), then
  `SCREW_FIRST_EFF` 0.28 / `SCREW_EFF` 0.88 of the thread's descent per sweep (`SCREW_PITCH` x sweep / 360),
  calibrated on mode A's measured sweeps;
- each sweep plans on the nut at that depth on the bolt's axis (`_screw_model`);
- the pitch stays `SCREW_PITCH`.

The lower of the two is safe both ways: a belief that measures the nut (a perception re-measure) wins when it is
shallower, and over-turning costs only steps.

| G3 mode B, 2 mm nut offsets, `--belief-drop fixtures`, 100000-09 | success | mean | steps median / max (1900) |
|---|---|---|---|
| before (Integration-3, 5ef91bb = main + #79) | 0/10 | 0.39 (0.5 x7, 0.2 x2, 0.0) | 1048 / 1121 |
| after (deaeab7 = 5ef91bb + the fix) | **10/10** | **1.00** | 1266 / 1353 |
| G3 mode A reference (skills-5, 25 layouts) | 24/25 | 0.98 | 1245 / 1365 |

Every nut now gets 4 sweeps, as in mode A. Data: compute3 `~/dojo-data/precision/g3/screws_mbfx_model_deaeab7/`, and
Integration-3's `~/dojo-data/integ3/g3/screws_mb2_dfx_5ef91bb/`.

Also in the PR:
- g3 hook runs record `final_true` next to `final_objects`;
- BACKLOG: G3 mode B's `final_objects` is the belief, and skills that measure their own progress on an object the
  belief cannot see must model it.

Open (fasten_screws): proxy / rendered B after #77 lands, i.e. whether perception's post-place re-measure of a nut on
its bolt agrees with the model, and the nut colour pairing from perception.

## 3. deposit_coin: the handover's taker put its hand on its side at the slot

G3 (skills-5's coin_final) has 9 failures in 25. I read each failure's trace (object poses + measured joints per step,
`--trace`).

**Cause of the 5 handover misses** (100001 / 09 / 13 / 16 / 22).
- The taker takes the coin's top rim from above with its hand straight down (0 deg). At the slot hover the same hand
  leans 76 / 71 / 46 / 68 / 90 deg from vertical.
- `plan_disc_handover` and `run_disc_handover` take the FIRST of the 48 slot targets (4 insert frames x 12 spins of
  the coin about its normal) that plans, and the first spin can turn the hand onto its side.
- On its side, the hand's 28 mm finger-root blocks hang over the pig's back (offline from the traces:
  `fingers_raise`'s worst point is 41 mm along the slot and 66 mm across it). The `fingers_out` raise lets the servo
  descend only 0-7.5 mm, so the coin is let go 0-1.5 mm into the slot and falls flat on the bank or onto the table.

**Fix:** `precision.hand_down_first`: the targets are ordered by how far the hand holding the coin would lean from
straight down there. Both the handover's plan and its re-plan from the measured grasp take the first that plans.

| G3, all 25 settled layouts | success | mean | steps median |
|---|---|---|---|
| before: main 48f5dde | 16/25 | 0.688 | 163 |
| after: + hand_down_first | **20/25** | **0.816** | 168 |

- Rescued: 100001, 09, 13, 16, 22 (exactly the five leaning hands). Lost: 100003.
- 100003 is the checker's bottom-bound landmine, not a regression: the coin went in (xy 1.1 mm) and lies on the
  bank's floor 0.8 mm lower than the lowest of the 20 successes (z 0.7747 vs 0.7755). In the before run it landed
  leaning and passed. Every coin on the floor sits within ~1 mm of that bound.
- The after arm also carried one-arm coins at 0.06 rad/step instead of 0.09. No one-arm outcome changed, and that
  change is not in the PR: the one-arm drop it was meant for (100002) is the coin sliding 5.2 mm down in the pads
  during the lift out of its stand (cover 9 -> 3.8 mm), then falling out early in the carry.

Data: compute6 `~/dojo-data/precision/g3/coin_g3main_48f5dde/`, `coin_hdown_97ea679/` (traces of the failures).

Left (G3, 5 of 25):
- 100002 / 100019: the coin slides in the pads as it leaves the stand (100019: back into the stand);
- 100000: the giver's pick turned loose (21.7 mm);
- 100017: the taker's descent is out of reach;
- 100003: the bottom bound.

## 4. insert_tubes: a relay for a tube lying on the table, places within reach, one-arm tubes first

G3 on main has 9 failures in 25, all at 0.4 (2 of 3 tubes in):
- 4: one tube with no one-arm plan;
- 4: the only one-arm plan held the tube 21 mm off its centre of mass, and it missed;
- 1: a far place missed.

The fallback relay never ran: `place_oriented`'s as-is relay needs `rests_stably` of the object as it lies, and a
lying tube's lowest points are a line.

**Changes** (`precision.insert_each`; `library.place_oriented(as_is_lying=)` + `lies_on_table`; `skills.plan_transfer(place_reach=)`,
default `PLACE_REACH`, so no other caller changes):
1. **The as-is relay of an object lying on the table.** It is put down as it lies on a spot between the arms, clear of
   the rack (`spot_avoid`), then put into a hole from there. The relay's holes come from all the free ones, ranked by
   reach from either base: each arm reaches only its own side of the rack (holes 0.49-0.61 m out).
   - Half the relays end with the same arm: the put-down works as a re-grasp, giving a CoM grasp that reaches the
     hole.
2. **A one-arm plan holding the tube off its CoM runs only after the relay found nothing** (G3: such places 2 in /
   6 missed).
3. **Places at most `EACH_REACH` 0.56 m from the placing arm's base** (all legs). From per-layout hole distances,
   CoM-held tubes:
   - 143/144 in below 0.56 m;
   - 0/4 at 0.562-0.567 m: the right arm into the far-left front hole, held 46-56 mm off its command;
   - the left arm's mirror place at 0.596 m 2/2: the arms are not mirror images here. Thin evidence: 2 independent
     bad cases.
4. **An object only one arm can reach goes first** (`EACH_PICK_REACH` 0.62 m). In every relay case the stranded tube
   was a one-arm tube, left for last (nearest to the rack first) after objects either arm could carry had taken its
   arm's holes.
5. **A tube counts as in only at its depth** (leading end >= depth - `seat_tol` below the mouth). With #77 a probing
   tail that met the rack short and let go left the tube believed "held by the rack" where the jaws opened
   (Integration-3's G3 runtime-belief run: one 11.9 mm short counted as in, and the checker did not count it).

**G3 numbers** (compute6; before = main 48f5dde; after = the PR head 2681f3d on main 0127871 with #93):

| | layouts | success | mean | steps median / max (500) |
|---|---|---|---|---|
| mode A, before | all 25 | 16/25 | 0.784 | 336 / 430 |
| mode A, after | all 25 | **24/25** | **0.976** | 344 / 485 |
| mode A, relay + off-CoM only (7effcc9) | all 25 | 21/25 | 0.904 | 345 / 485 |
| mode B, 2 mm offsets, `--belief-drop fixtures`, before (main 0127871) | 100000-14 | 6/15 | 0.613 | 305 / 473 |
| mode B, the same, after (2681f3d) | 100000-14 | **9/15** | **0.760** | 452 / 481 |

- Mode A: rescued 8 (100005 / 06 / 08 / 09 / 10 / 11 / 19 / 22), lost 0. Left: 100013, two relays in one
  episode, out of steps at 485.
- Mode B: rescued 3 (100002 / 06 / 10), lost 0. Five of the six left are relay episodes out of steps (449-481).
- Mode B and the step budget: a relay leg costs ~110 steps (a put-down + a re-pick at mode B's ~47 steps per pick).
  The mode B failures left are relay episodes that ran out of steps (471-481). Proxy B spends ~440 steps on three
  direct tubes (breadth2: looks, refines, scan), so in proxy / rendered B a second relay never fits and a first one
  rarely does. Change (4) is the step-free part.

## 5. plug_in_charger: held_pose in the runtime (rendered A/B) and the in-hand flip

**held_pose A/B** (rendered 5.1 on compute6, submit_v1 vs + `held_pose: auto`, main 2a6e30d, 100000-09 paired):
off 6/10, auto 4/10. Auto rescued none and lost two:
- 100004: the flip loop below, with no held_pose read at all;
- 100007: both reads not ok, then the same blind probing tail the off arm used.

The protocol never got past the hover read:
- 14 hover reads: 1 ok, 0 adopted (8 ambiguous, 4 fit_poor, 1 out_of_range).
- Not one observer look or show pose ran. Every candidate from perception's `show_pose.propose` was infeasible for our
  planner: "palm / link at table" (looks proposed around the plug's hover, 2 cm over the strip) or "arms closer than
  4 cm".
- G3's 2/10 -> 8/10 came from `heldsim`'s generic candidates, which plan.

The lead's decision: `held_pose` stays off. Report: experiments/2026-10-27-plug-held-pose-ab.md.

**The in-hand flip.** The charger's yaw is a coin toss between the top-down pick view and the in-hand ("held") view.
Rendered 100004 (truth -9.7 deg):
- pick -11.4 (right);
- held 170.6 (wrong), adopted: the held review called the grasp "loose", put the charger down and re-planned;
- pick -10.9 (right), but turned 179 deg from the flipped belief, so "prior kept";
- held -8.5 (right), but loose against the flipped re-plan;
- ... 4 cycles, 604 s.

The first plan was right. Perception-2's exact replay: in-hand, the flipped side does not even converge; no perception
tweak beats main.

**Fix** (runtime, `RuntimeConfig.held_flip_confirm`, default on = a default change; `held_flip_agree_deg` 30):
- A held refine turning the object more than `refine_max_turn_deg` is not applied: "pending a second view, plan
  kept". The skills' held review then sees no turn, so there is no put-down loop.
- A later view applies it when it finds the same turn in the object's own frame: another arm's pick refine (after a
  relay) or a held refine at another step. A frame-relative turn survives a carry.
- A view within `refine_turn_check_deg` refutes it.
- Pick views alone never flip the belief, as before (#80).

**Rendered A/B** (5.1, submit_v1, compute6, 100000-09 paired; main a3dc7e1 vs + the fix 404cc5c):
- main **4/10**, the fix **8/10**: rescued 4, lost 0. Left: 100002 (no relay plan) and 100005 (perception).
- On main, the in-hand flips in 100000 / 03 / 08 (1 / 3 / 1 of them) each put the charger down, and all three failed.
- With the fix, the same layouts' flips (one each) waited, with no put-down, and all three went in.
- 100009: its in-hand flip was confirmed by the other arm's pick view after the relay and applied (main failed it on
  an unrelated grasp slip).
- Wall median 171 -> 141 s.
- Report: experiments/2026-10-27-plug-held-flip.md.

## 6. Checks: fasten_screws at 25 layouts, coin and tubes on main, an infrastructure landmine

- **SDF cooking failures** (#125). Two physics-only Isaac processes that start the same SDF task at once can fail to
  cook its SDF colliders ("UjitsoMeshCookingContext: cooking failure"; nuts, bolts, the tube rack).
  - fasten_screws G3 mode B in two jobs started the same second came out 1/15: every place missed, the arm held 100+
    mm off its command, 11 s per episode. The same layout alone scores 1.0.
  - g3 now reads its own Kit log per episode (`KitLogWatch`): such episodes are `infra_failure`, out of n. The twin
    service flags its load. Kit keeps 200 logs (was 30).
  - Stagger same-task SDF jobs; the race did not reproduce on demand (2 jobs together: 1.0 each).
- **fasten_screws G3 mode B on main** (#85; 2 mm offsets, fixtures belief):
  - 100010-24 as one job: 14/15 (100024 at 0.5);
  - with 100000-09: **24/25**, equal to mode A's 24/25.
- **Post-merge, G3 mode A on main 13a28b6, all 25 each** (no infra failures): deposit_coin 20/25 (0.816) and
  insert_tubes 24/25 (0.976), per layout the same as the PR runs.
- **deposit_coin G3 mode B on main** (2.5 mm coin offset, `--held-belief`, fixtures; #101 in): 17/25. The extra losses
  against mode A are rim grasps whose coin falls out at the lift (100006: lifted 24 mm, then fell, and the handover went
  on with an empty hand). A fixed 2.5 mm offset with a null refine is pessimistic; the runtime refines the coin at the
  hover.
- **Tubes, a quicker relay put-down** (3 cm pre-place, 0.10 rad/step): no measurable gain (408 -> 400, 485 -> 487
  steps). The steps go into homing / parking between the legs (~124 per relay episode outside the skill records), not
  the put-down. Dropped.

## 7. The held-pose protocol in the runtime: a read on request, our own show pose (#136)

- `held_pose: request` (runtime) + `plan_place(held_read=True)`: only places that ask read (Pegs' insert_key).
- When none of perception's show-pose candidates plans (in the runtime, none ever did), our own: the show point x 8
  yaws, the first observer that plans.
- Rendered plug, the same flip code: held_pose off **8/10** vs auto + the generic show **4/10**.
  - The show poses ran (26-34 steps), with 2 of 3 reads ok (~0.6 mm lateral).
  - The read + servo seated 1.6 mm short of the checker's 15 mm blade depth where the blind probing tail went in, and
    two charger grasps slipped (the extra carry).
  - So `insert_into` does not ask by default: a skill that opts in shows it on rendered frames first.

## 8. insert_tubes mode B: the guarded descent's stride

Mode B's per-tube cost (G3 mode B, the actions' segments): a tube's insertion is the guarded / probing tail (no in-hand
read), 23 strides of 3 mm with a still step each, so 46 env steps for the hole's 52 mm, plus 7 for the release. That is
a third of the ~140 steps a tube costs in mode B. A relay leg (~110) then never fits in 500, and the step-budget
failures missed by 1-30 steps.

**Change:** `plan_place(guard_step=, guard_rate=)` (default unchanged: 3 mm, 0.02 rad/step); `insert_each` descends in
6 mm strides at 0.04 rad/step, one move step each. The holes are 29-30 mm for a 28 mm tube. Mode A never runs this
tail (it re-measures and servos).

| G3 mode B, 2 mm offsets, fixtures belief, 100000-14 | success | mean | steps median / max (500) |
|---|---|---|---|
| main d2e64f1 | 9/15 | 0.760 | 452 / 481 |
| + guard stride (2f80385) | **11/15** | **0.840** | 388 / 485 |

- Rescued 100000 / 100011, lost 0. Left: 100005 / 09 / 13 (two relays in one episode, still out of steps) and 100008
  (no plan for the last tube).
- Rendered 5.1 B (submit_v1, compute6, 100000-09 paired, chunks of 5 interleaved; main d2e64f1 vs 2f80385):

| rendered 5.1 B, 100000-09 | success | mean | steps median | wall median |
|---|---|---|---|---|
| main | 6/10 | 0.760 | 466 | 180 s |
| + guard stride | **7/10** | **0.820** | 393 | 132 s |

  - Rescued: 100000 (budget). Lost: none.
  - Still failing:
    - 100002 went from budget to belief_mismatch;
    - 100008: no plan, in both arms;
    - 100009: budget, in both arms.
  - Every success took fewer steps: 45-67 fewer per episode.

## 9. plug_in_charger 100002: relay spots toward the socket's side (#163)

100002 had no relay plan in any mode. The socket is 0.39 m to the right and only the left arm reaches the charger. Every
relay chain from spots near `relay_near` (0, -0.28) failed to plan: from there, no grasp the right arm can take
inserts.

G3 snippets with `relay_near` moved:
- (0.12, -0.28): 1.0 (2 of 2);
- (0.2, -0.28) and (0.15, -0.24): 1.0;
- (0.15, -0.28): 0 (the left arm's as-is put-down landed 26 deg tilted).

**Change:** when no chain plans near `relay_near`, `place_oriented` runs a second round of spots near a point moved
toward the targets' mean x, by up to `RELAY_SHIFT` = 0.12 m. The first round is unchanged, so only an episode with no
relay plan at all can change.

G3 mode A, paired against main:
- plug 100002: 0 -> 1.0;
- plug's other failures (100008 / 13): identical;
- the regression set (store_tools 5, insert_tubes 100005-14, make_kong 5, stack_blocks 5): identical scores and
  steps.

The second round fired only on 100002. The first version (0.15 m) over all 25 plug layouts was identical to main
except 100002.

## Spend and machines

- LLM calls: only the agentic perceiver's in the rendered plug A/Bs: 4 arms x 10 episodes, ~3 calls each, ~120 calls
  (Sonnet 5.5 via omniroute). Every G3 run: none.
- The rendered tubes guard A/B (§8): 20 episodes, ~3 agentic-perceiver calls each, ~60 calls.
- compute3 light slot 4 until compute6 opened (screws, tubes chunk 1); then compute6 light slots 1-4 and heavy lock 1
  (rendered A/Bs).
