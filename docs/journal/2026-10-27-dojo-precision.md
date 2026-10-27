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
