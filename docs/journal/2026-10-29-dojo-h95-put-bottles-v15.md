# 2026-10-29: put_bottles on v15 layouts: the candidate 9/16, plus unblock + gentle 13/16 (stream H)

Method: G3 mode gt (physics-only twin, GT world, 0 LLM, traced).
- Layouts: my own v15 dev draws, put_bottles_into_dustbin 100030-49 variant 1 (`gen_layouts.generate(..., variant=1)`,
  simlab/v15, main b131264e).
  - G3 refuses the canonical ids of the training block (D-C10c), so the draws are variants.
  - My 100005 regenerates byte-identical to Integration's v15 record.
  - Presettled with `dojo.simlab.presettle` defaults (physics-only, 2 passes) in a private DOJO_DATA
    (compute3:~/dojo-data/h95/v15dev). 16 of 20 settled.
- No measurement ids (100005-09, 100020-29) were used: the lead's rule.
- Base and fix ran on the same code. The candidate = `put_into_opts` spot, patient, sequence +
  `joint_limits=measured`.

## Diagnosis

| arm | success | mean |
|---|---|---|
| today's defaults | 1/16 | 0.23 |
| candidate | **9/16** | **0.70** (12 up, 0 down) |

The candidate's ceiling carries over from the old layouts to v15. Its 7 failures all end in "no arm can pick", with
the budget left.

- **2 handover blocks** (100035, 100045).
  - The helper puts a lying bottle down at a handover spot in front of a standing bottle. In a CPU replay at the end
    state (kinematic env), each makes the other unpickable (most grasps collide), and moving either one away frees
    the other.
  - The spot passed `can_pick_at` for the handed-over bottle itself. Nothing checked the others.
  - At the decision the check has to use the place plan's own pose: `plan_place(..., yaw_sym=8)` picks the yaw, not
    the in-hand one. The bottle then rests up to ~13 mm off that plan, and 100035 blocked only at the off-plan pose.
- **3 bottles flung past the bin** (100030, 100033, and 100035 once unblocked).
  - The dropper carries a side-pinched standing bottle to the drop point at CARRY (0.14 rad/step, ease 3). The TCP
    moves 0.09-0.16 m/step.
  - The bottle turns 14-45° in the fingers during the carry and leaves them in its stop. It lands on the floor
    0.3-1.1 m past the bin.
- **2 knock-overs / slips** (100032, 100037, plus 100049).
  - In 100032 the right arm's reset home after a failed pick swept a standing bottle over.

## The options (`RuntimeConfig.put_into_opts` / `Skills.put_into_opts`; G3 `--put-into`; default empty = today)

- `unblock` (`library._spot_blocks`):
  - A handover spot is skipped when an object that some arm can pick from home now could be picked by no arm with the
    handed-over object at the place plan's pose, or at that pose shifted 15 mm in x or y (`SPOT_SLACK`).
  - A skipped spot is logged as "[put_into] handover spot ... would block [...]".
- `gentle` (`library.DROP_GENTLE`): the drop carry at MEDIUM (0.10 rad/step) with a 6-step ease, instead of CARRY with 3.
- Only put_bottles reaches put_into's drop and handover (journal 2026-10-28-dojo-h95-put-bottles), so neither option
  touches another task.

## Screen (v15 dev draws, n = 16, same code)

| arm | success | mean | vs candidate |
|---|---|---|---|
| candidate | 9/16 | 0.70 | |
| + gentle | 11/16 | 0.78 | 3 up, 0 down |
| + unblock | 9/16 | 0.73 | 2 up, 0 down (100035 0.25 → 0.4, 100045 0.1 → 0.4) |
| **+ unblock + gentle** | **13/16** | **0.89** | **5 up, 0 down** |

- Rescued to success: 100030 and 100033 (gentle alone does it too); 100035 and 100045 (only with both options).
  100037 went 0.25 → 0.4.
- Left:
  - 100032 and 100049: a standing bottle side-pinched by the left arm slips (rose -59 / -92 mm). One of them had been
    knocked over by the reset sweep first.
  - 100037 is 0.4, not_home.
- Steps: the successes take 286-611 of 700. Gentle adds ~20-30 steps per episode.

Spend: 0 LLM calls. G3 on compute3: 8 jobs of 2-16 episodes (~11 min each), 1 presettle, CPU replays. One job at a
time.

Expected Average: the cell's score goes 0.70 → 0.89 in G3 on v15, i.e. +0.19 of one cell (~+0.45 Average at
~2.5 points per cell), if it transfers to the rendered run. The rendered A/B decides.

Next (BACKLOG): the reset-home sweep that knocks bottles over, and the left arm's side-pinch slips on standing bottles.
