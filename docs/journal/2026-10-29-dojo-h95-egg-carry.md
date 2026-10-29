# 2026-10-29: fill_egg_holder, a slow egg carry is null; parked (stream H)

Method: G3 mode gt (physics-only twin, GT world, 0 LLM), traced.
- v1 draws of 100030-49, presettled into a private DOJO_DATA. 10 of the 20 settled stably, so n = 10.
- Dev 100000, 01, 06, 08, 09, 10, 12, 15, 16, 19 (default layouts).
- Base and fix ran on the same code.
- Caveat (Integration, 2026-10-29): the fill_egg measurement layouts came from generator v6. v15 regenerations are
  coming, so these numbers may not transfer.

## Why the carry

Traced base (journal 2026-10-28-dojo-h95-fill): 11 of 32 drops had no egg in hand. The egg had left the pads during the
lift or the carry (MEDIUM 0.10 rad/step up, CARRY 0.14 rad/step across), from holds the pick had called "ok" or loose.

## The switch (branch `feat/dojo-h95-egg-carry`, not merged)

- `RuntimeConfig.egg_carry` / `Skills.egg_carry`: "v1" (default) | "slow".
- "slow": `holders._fill_drop` lifts and carries at 0.05 rad/step (`EGG_CARRY_RAD`, the plan_pick lift and the
  plan_place approach).
- G3: `--skill-attr egg_carry=slow`.

## Screen

| set | v1 | slow | up / down |
|---|---|---|---|
| v1 draws, settled (n 10) | 0.150 | 0.150 | 3 / 3 |
| dev (n 10) | 0.235 | 0.205 | 1 / 3 |

Drop and pick outcomes, totals over the episodes (v1 → slow):

| | v1 draws | dev |
|---|---|---|
| eggs dropped "in a hole" | 13 → 13 | 19 → 17 |
| eggs dropped "NOT in a hole" | 7 → 5 | 5 → 3 |
| picks that slipped | 9 → 5 | 4 → 2 |
| eggs with no pick | 19 → 22 | 10 → 14 |
| picked, then no drop plan | 6 → 5 | 11 → 9 |

- The slower carry loses fewer eggs in transit and misses fewer holes.
- But no more eggs land in a hole, and the score does not move. Eggs with no pick and picks with no drop plan bind,
  as `egg_pair` already showed (BACKLOG).
- Parked. No PR for a flag with no measured gain (the lead's ruling).
