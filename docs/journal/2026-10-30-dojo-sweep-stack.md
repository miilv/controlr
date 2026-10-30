# 2026-10-30: sweep_blocks in G3: a stack of failure classes, three levers (stream N)

G3 mode gt, 30 episodes: dev 100000-04 (settled) and v1 draws 100030-39 (generated with `variant=1`), both cells. The
simlab std 100030-49 records are D-C10c's calibration block and G3 refuses them. The task is binary. 0 LLM calls.

**Classes** (`/tmp` classifier on the episode rows: the result stage plus the dustpan's final pose):

| arm (commit) | ok | broom (relay) | dustpan carried away | cubes short at the lip | sweep other (strokes, steps) |
|---|---|---|---|---|---|
| base (46540867) | 1 | 16 | 5 | 1 | 7 |
| + sweep_relay_grip (875258d2) | 1 | 13 | 5 | 3 | 8 |
| + relay_grip + sweep_deep (0270c8cf) | 2 | 13 | 5 | 0 | 10 |
| + relay_grip + deep + sweep_hold_pan (0e2a6125) | 2 | 13 | 5 | 0 | 10 |

- **Broom relay.** The broom lies out of the right arm's reach, so the left arm puts it down nearer, through the table.
  The left arm's "tilt -25 off -17mm" pinch rose 2-43 mm, or slipped 5 steps into the carry (replay of dev 100004: the
  jaws closed to 0.0 at step 36). The relay's place then moved an empty hand, and the right arm reached 0 of 40 broom
  grasps. Two more relays found no spot. The camera rehearsal's "rose 0 of 60 mm" is this class.
- **The dustpan carried away.** The drag holds the dustpan without a `Held`, so a later pick of the other arm parks
  this arm home with its jaws still on the handle. Replay of std 100003: the dustpan rode home in the left hand from step
  56 to 414, and every cube was swept to where it had been.
- **Short at the lip.** The skill's own test (the centre within 81 mm, read right after the push) passed, and the
  checker failed one to three objects 87-126 mm from the circle's centre.
- **Sweep other.** Stroke IK ("above: no seed converged / out of reach"), pushes that leave the object 3-7 cm short,
  and episodes out of steps (961-990 of 1000).

**Levers** (all default off):
- `sweep_relay_grip`: the relay giver's straight-down grasps, ranked by `grasps.hold_quality`, are planned first. It
  fixed 3 relays (std 100004, `_random` 100004 and 100035), but in most episodes the firm set has no plan and the
  tilted pinch is used as before. The fixed episodes then failed on the next class.
- `sweep_deep`: an object not 15 mm inside the circle gets up to 2 more pushes, 60 mm past the lip. It converted std
  100032. Two more "short" episodes became "sweep other", because the deeper pushes left the object short again.
- `sweep_hold_pan` (`Skills.no_park`): the dustpan's arm is never parked. **As run in G3 it did not act**: it covered
  `_park_if_blocking`, but the park comes from `Skills.parallel`'s single-plan branch (also `pick_place`'s
  `park_other`). Both are covered now, with unit tests, and not yet run in G3.

**Stack.** 2 / 30 with everything on, so it stops per the box rule. Each lever only uncovers the next class. The open
items, in order of size:
1. A firm broom hand-over (a pinch that holds a 40 g broom on a 30 cm carry, or the instruction's in-air hand-over).
2. The dustpan kept in place: `sweep_hold_pan` as fixed, run once.
3. Stroke reach: where the dustpan goes relative to the cubes and the right arm.
