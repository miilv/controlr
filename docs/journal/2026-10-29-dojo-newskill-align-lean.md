# 2026-10-29: stream N, align_blocks in camera mode ran out of steps; a leaner mode B behind a flag

Rendered align arm (compute4 `c4b_alignblockssg_port_100005-07`, #282 code): 1/3. Both failures were the 200-step
budget, not perception.
- The runtime's look macro took 2 × 14 steps at the start. Each block's slide then took ~62-64 steps: a look pose 8 cm
  up, the refine, then the slide's own approach, grasp and slide.
- The second slide needed 63 / 68 steps with 26 / 34 left (reserve 30).
- Every hover refine was ok (shift 0.8-2.8 mm, turn 0.3-3.4°). In 100006 the one finished slide ended 6.8 mm off, and
  no steps were left for the fix pass.

`Skills.align_measure` / `RuntimeConfig.align_measure` / G3 `--skill-attr align_measure=` (default "seq" = today):
- "parallel": every hover is the slide's pre-grasp, so the slide starts where the refine was taken.
- Two movers on two arms whose hovers keep 0.12 m between the arms (`motion.min_arm_distance`; the links are 7-8 cm
  thick) go there in one move and refine with their own wrists. Their slides run in the same steps when the merged
  schedule keeps the arms apart too.
- Otherwise one block at a time, the other arm parked when it is in the way.
- A first version with a 5 cm clearance ran two slides 9 mm apart in G3 100031, and neither block moved.

G3 mode B (`--mode-b`: a refine hook that sees nothing), v1 draws of 100030-39, compute3:

| align_measure | commit | n | success | steps (min / median / max) |
|---|---|---|---|---|
| seq | 1e87a1db | 10 | 9 | 120 / 150 / 175; 100030 ran out of steps |
| parallel | 29338585 | 10 | 10 | 67 / 107 / 159 |

Kinematic env (cramped scene, hovers 2.4 cm apart): 175 → 130 steps.
