# 2026-11-06 — post-D-C19 camera A/B wave: verdicts

Plan and pre-registered rules: [the wave plan](../experiments/2026-11-06-dojo-ab-wave.md), and for the default rule [EXPERIMENTS.md, "Camera A/B decision rule"](../EXPERIMENTS.md). Rows: compute4 runs2, families `c4b_<cell>sg_<lever>{A,B}r1_<id>`. A = CANDIDATE.txt at deploy 416a2a03; B = A + one switch. Each verdict is read off the lane logs' per-episode success / score lines once the block is complete.

## peg: `peg_held_read=on` (#388), play_stacking_toy — NOT ADOPTED (rule miss by one net win)

| cell | pairs | A pooled | B pooled | Δ | B-only wins | A-only losses | successes A / B |
|---|---|---|---|---|---|---|---|
| play_stacking_toy (gain) | 10 | 64.0 | 70.0 | +6.0 | 2 | 1 | 3 / 4 |
| stack_blocks (guard) | 10 | 91.5 | 100.0 | +8.5 | 1 | 0 | 9 / 10 |
| insert_tubes (guard) | 10 | 88.0 | 88.0 | 0.0 | 0 | 0 | 8 / 8 |

- **What passed:** the pooled gain (+6.0 ≥ +5), the flip ratio (2 ≥ 2 × 1) and both guards.
- **What failed:** the rule also needs ≥ 2 net wins, and there is 1. So the lever stays default-off.
- **Re-test:** with a positive pooled delta it qualifies for ONE re-test on layouts it has never run on (EXPERIMENTS.md), ideally with 20 gain pairs.
- 60 / 60 jobs had rc 0.
