# 2026-10-30: fill_pen_holder on v15 dev draws: the holder is dragged over by the hand that put it down; `fill_move_lift` (stream H)

## Method

- G3 mode gt: physics-only twin, GT world, 0 LLM, traced.
- Layouts: S's v15 dev draws, variant 1 of 100030-49 (compute3 `~/dojo-data/s-fillpen/simlab/layouts/fill_pen_holder`,
  read only). All 20 load stable.
- Base = main 0c5cddfb. It reproduces S's modeA rows exactly: 0.125, 0/20, every score equal (journal
  2026-10-29-dojo-insert-key-fill-pen).
- Score tiers: 1 / 2 / 3 / 4 pens = 0.10 / 0.25 / 0.40 / 0.90, in transition mode. Every tier needs the holder
  upright (≤ 45 deg) and all grippers open.

## Classes (base 0/20, mean 0.125)

**The holder ends on its side in 5/20 episodes.** Each scores 0, and in each all four pens end "0 grasps": the drop
targets of a lying holder admit no grasp.
- 12 of the 20 episodes carry the holder to a spot near the robot ("comfy" move).
- **4 of the 5 tips (100030, 100040, 100043, 100047) are one mechanism**, read from the traces:
  - The hand held the 80 × 69 × 120 mm holder top-down across its walls ("top axis 1 tilt 0").
  - It set the holder down upright and opened.
  - The place's retreat backs out 50 mm, which leaves the fingertips ~15 mm below the rim (tips at z 0.870-0.871,
    rim 0.885).
  - The hand's next move then dragged the holder over, 3-10 steps in, tilt 10 → 90 deg. That next move was the first
    pen's pick approach in 100030, and the right arm's way back in 100040 / 43 / 47.
- **100031 is different.** A tilted side grasp ("box close 1 approach +2 tilt -25") stalled 43 mm above the table and
  let go there ("dz +53 mm"). The holder stayed hooked on the fingers and fell when the arm moved off.

What the pens did (80 in total):

| outcome | pens |
|---|---|
| inside | 21 |
| no grasp (20 of these in the tipped episodes) | 27 |
| dropped and not inside | 11 |
| no grasp-place pair | 9 |
| placed "ok" but not inside | 5 |
| slip | 5 |
| collision | 2 |

## `fill_move_lift` (new switch, default "off")

- **What it does.** "clear" (`library._lift_clear_of`, `RuntimeConfig` / `Skills` / G3 `--skill-attr`) acts right
  after `fill_container`'s container move, before anything else moves. Every hand whose fingertips are within 60 mm
  of the container's footprint and below its top goes straight up until they are 30 mm above it.
- **Fallback.** The right arm, reaching across to the holder's spot at x ≈ -0.11, has no straight 44 mm lift there:
  its line IK fails 2/3 of the way. The lift then takes the first of 30 / 15 / 5 mm over the rim that a straight line
  reaches. In practice that is 19-29 mm up.
- The first version had no fallback. It cleared only the left-arm cases, and gave 0 up / 0 down.
- **Where it acts.** Only in the move branch, and only after a move that ran steps. In the 8 episodes that never move
  the holder it changes nothing.

## Screen (G3 mode gt, the same 20 ids, 0 LLM)

| arm | mean | successes | up / down | holder on its side at the end |
|---|---|---|---|---|
| base (main 0c5cddfb) | 0.125 | 0/20 | | 5 |
| fill_move_lift=clear | 0.190 | 0/20 | 5 / 0 | 3 |

All 20 ran under the flag. The 8 episodes that never move the holder are bit-identical to base (steps, scores and
classes).

| seed | base | clear | |
|---|---|---|---|
| 100040 | 0 | 0.10 | holder saved |
| 100043 | 0 | 0.40 | holder saved; a later pen drop knocked it over after the 3-pen tier |
| 100047 | 0 | 0.40 | holder saved |
| 100041 | 0 | 0.25 | its holder never tipped; the changed start moved the pens' plans |
| 100042 | 0.10 | 0.25 | same |

- 100030's holder is saved too, but its pens still fail: 3 dropped beside it, 1 with no pair. It stays at 0.
- 100031 (the high release) is unchanged.
- 100032 is equal at 0.25: its holder was knocked over by a pen drop late, after its tier.

## Left

- Pen drops that knock the standing holder over (100032, 100043 under the flag).
- The high release of a tilted holder grasp (100031).
- The pen classes: dropped and not inside 14, no grasp 14, no pair 12, slip 7. pen0 is still the worst asset
  (BACKLOG).

## Spend

0 LLM calls. G3 on compute3, one job at a time: 4 base chunks of 5, 6 fix chunks of 4, and 2 identity chunks.
