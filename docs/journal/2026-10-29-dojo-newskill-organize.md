# 2026-10-29 — stream N: organize_table at true state, place order and press strokes

**Baseline** (G3 mode gt, v1 100030-39, compute3, main 6f2f6309, no flags): 0/10 successes, mean 0.375. Per item:
- **mouse:** mostly placed.
- **alarm → drawer:** misses with a 10-17° tilt in about half the rows.
- **garage → cube_cushion:** never placed. Either ik_fail ("left: N grasp-place pairs", no full plan) or a slice-grasp
  slip.
- **keyboard → frame:** the push ends "no push stroke" in 10/10.

This matches H's diagnosis (journal 2026-10-28-dojo-h95-organize): the garage is carried after the alarm and sweeps
through the drawer and the alarm on every place approach.

**`arrange_order`** "near" (today) | "big_first" | "push_big_first" (`RuntimeConfig`, `Skills`, G3 `--skill-attr`;
default "near").
- **big_first** places the largest object first, while the table is emptiest. In big_first rows the garage's transfer
  plans directly, which no base row did. The 40° slice grasp then slips: the garage drops, sometimes by the keyboard.
- **push_big_first** runs the pushes before the places.

**`push_stroke`** "face" (today) | "press" (default "face").
- **What it does:** when no face stroke plans on either axis, the keyboard is dragged with the closed fingertips pressed
  3 mm into its top.
- **The first version** kept the hand straight down. It never planned: after the first +y strokes the keyboard sits
  ~0.48 m from both bases, where no straight-down pose has an IK (CPU replay of bigpress 100030, kinematic world from
  the row's final poses).
- **Now:** the hand tilts 65 / 50° towards its heading (`PRESS_PITCHES`), and the replay finds a plan (left arm, along
  the long axis).

**G3 screen** (v1 100030-39, compute3, 0 LLM):

| arm | n | successes | mean | garage placed | keyboard push |
|---|---|---|---|---|---|
| base (6f2f6309) | 10 | 0 | 0.375 | 0/10 | no stroke 10/10 |
| push_order lookahead | 4 | 0 | 0.31 | 0/4 | no stroke 4/4 |
| big_first (3011a32b) | 6 | 0 | 0.375 | 0/6 (plans direct, slips) | no stroke 6/6 |
| big_first + press, tilted (017a115f) | 10 | 1 | 0.40 | 1/10 | press ran 8/10, 19-129 mm left |
| push_big_first + press, tilted | 10 | 0 | 0.375 | 0/10 | ok 2/10, press ran 9/10, 18-138 mm left |

Every arm is within noise of the base. Neither flag is worth an A/B as it stands.
- **Tilted press.** It turns "no push stroke" into strokes that move the keyboard, but the drags do not converge on the
  frame (18-138 mm left).
- **The garage.** It stays the blocker. With big_first its transfer plans, then the 40° slice grasp slips in 5-6 of 10.
  The next lever is the hold of that grasp: tilt, slice height against the figurine's mass, squeeze.
