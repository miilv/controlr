# 2026-10-27: fill_egg_holder: eggs carried lying and let go off a hole's centre, pushed off the basket wall first

**Question.** Can `fill_holes` put eggs into the egg holder reliably enough to beat the board's best entry (21.2, PyRUA)?
skills-6 measured G3 mode A 0/11, mean 0.03.

**Tier and setup.**
- G3 mode A: the TRUE state in the physics-only twin, compute7 (Isaac 5.1 physics).
- The 11 settled dev layouts of fill_egg_holder (100000-100024 with an ok settled record).
- `SIMLAB_USD_SYNC_LINKS` on; catalogue: perception's, md5-identical to skills-6's egg meshes, + `links.json`.
- No LLM calls.
- Score = the checker's ladder: 1 egg 0.1, 2 eggs 0.25, 3 eggs 0.4, 4 eggs 0.9, + the lid and home 1.0.

**Why skills-6 failed.** Its 11 traces, replayed offline (`fill_egg_holder_*_trace.json.gz`; measured vs commanded TCP at every close):

| what | how often |
|---|---|
| the hand stopped 15-65 mm short of the egg on the way down | 9 picks |
| a transit at rim height stopped on the basket (the basket is Geometry and was no collision obstacle) | 7 moves |
| picked, then lost while the egg was turned upright in the pads (it pivots about the pad axis) | ~6 of 10 picks |

In the blocked picks the dense finger samples were 17-34 mm into the basket's wall profile, but the corner samples cleared it: the 38 x 56 mm finger root block straddled the rim. Hands that went down had dense clearance >= -12.6 mm.

**What changed (`dojo/skills/holders.py`, `fill_holes` mode "drop").**
1. Picks: top-down level and rolled pinches, filtered by dense finger samples against the basket's wall profile (>= -10 mm, from the calibration above).
   - Neighbour eggs are mesh parts the fingers may push 3 mm.
   - The egg is lifted until its centre is 45 mm over the rim.
   - Then the basket becomes a height-map obstacle for every transit.
   - Tip-to-tip pinches are off: they slipped.
2. An egg with no pick that rests against the wall is pushed 3 cm inward first (`nudge_from_wall`):
   - closed fingertips go down into the gap on its wall side, at or up to 14 mm above its widest height;
   - the path is checked against the wall profile.
3. The egg is carried lying, as picked, and let go over a free hole:
   - its centre sits 12 mm (or else 7 mm) along its long axis off the hole's centre;
   - it is as low as its underside clears the far rim (+2 mm).
   - Nothing is under its near end, so it tips in. Centred, a lying egg spans the hole and can stay.
4. An egg left lying across the hole (centre over the plate) is pressed from above, at its middle and then at its high end. If it is still across, it does not count as in, and it is retried.
5. Back holes under the open lid fall back to the old stand-up place when no lying drop plans.

**Results** (same 11 layouts, eggs in = the sum of the ladder's egg counts):

| version | eggs in | mean | per layout (100000 01 06 08 09 10 12 15 16 19 23) |
|---|---|---|---|
| main / skills-6 (egg_v6, compute3) | 3 | 0.027 | 0 0 0 0 0 .1 0 0 0 .1 .1 |
| drop + nudge | 7 | 0.077 | 0 0 0 0 0 0 .25 0 .25 .25 .1 |
| + press (7 mm offset first) | 14 | 0.150 | .1 .25 .1 .4 .1 0 .25 0 .25 .1 .1 |
| + second press, lying-across not counted | 13 | 0.141 | .1 .25 .1 .4 .1 0 .25 0 .25 .1 0 |
| **+ 12 mm offset first (this PR)** | **17** | **0.186** | .1 .25 .25 .25 .1 0 .25 .25 .25 .25 .1 |
| 7 mm first, wall check -13 mm | 12 | 0.132 | |
| drops toward the outer rim first + egg turned about its axis | 11 | 0.114 | |

(final confirmation and main on compute7: see below)

**Reading.**
- The lying drop puts an egg in when its pick holds: tilted 16-51 deg, centre 25-32 mm over the hole's floor, inside the checker's 40 mm.
- What is left, in order:
  1. Eggs against the basket wall with neither a pick nor a nudge plan: 2-3 per layout. Eggs on the table under the basket's flared rim have no grasp at all.
  2. Eggs that land wedged across a hole, 2-5 mm over the plate. Pressing did not free them.
  3. Picks that come up loose and drop on the carry.
- The lid never ran, since 4 eggs never went in.
- n = 11, at 0.1-0.25 per layout: the variants 0.13-0.19 are within noise of each other. The step from 0.03 to ~0.15 is not.

**Mode B (not measured here).**
- The nudge and the press move an egg without a grasp, so their predicted outcomes go to `provider.confirm_pose`, as a topple's do.
- The runtime belief drops a released egg to the table: the holder is an Articulation, not a #77 fixture (BACKLOG). The post-place refine has to find it in its hole.

**Spend.** No LLM calls. About 14 GPU-hours on compute7 light slots and compute3 overflow slot 4.
