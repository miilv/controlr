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

## waist: `belt_waist_first=on` (#389), match_and_pick_from_conveyor — NOT ADOPTED (rule miss by one net win)

The gain set is 100005-09 + 100020-24 with 100008 → 100025: 100008 failed raw-check, under the pre-collection amendment.

| cell | pairs | A pooled | B pooled | Δ | B-only wins | A-only losses | successes A / B |
|---|---|---|---|---|---|---|---|
| match_and_pick_from_conveyor (gain) | 10 | 50.0 | 60.0 | +10.0 | 1 | 0 | 5 / 6 |
| pick_from_conveyor_by_image (guard) | 5 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 / 0 |

- **What passed:** the pooled gain (+10 ≥ +5), the flip ratio (1 ≥ 2 × 0) and the guard.
- **What failed:** net wins are 1, and the rule needs ≥ 2. The lever stays default-off.
- **Re-test:** it qualifies for one fresh-layout re-test; with peg, a 20-gain-pair re-test is the natural next booking.
- 30 / 30 jobs had rc 0.

## disc: `disc_hold_evidence=on` (#382), deposit_coin — NOT ADOPTED (rule miss by one net win)

| cell | pairs | A pooled | B pooled | Δ | B-only wins | A-only losses | successes A / B |
|---|---|---|---|---|---|---|---|
| deposit_coin (gain) | 10 | 12.0 | 26.0 | +14.0 | 2 | 1 | 1 / 2 |
| insert_key (guard) | 3 | 43.3 | 43.3 | 0.0 | 0 | 0 | 1 / 1 |
| play_tic_tac_toe (guard) | 3 | 83.3 | 83.3 | 0.0 | 0 | 0 | 2 / 2 |
| stack_blocks (guard) | 3 | 100.0 | 100.0 | 0.0 | 0 | 0 | 3 / 3 |

- **What passed:** the pooled gain (+14 ≥ +5), the flip ratio (2 ≥ 2 × 1) and all guards (identical).
- **What failed:** net wins are 1, and the rule needs ≥ 2. The lever stays default-off; it qualifies for one fresh-layout re-test.
- 38 / 38 jobs had rc 0.

**Pattern across the wave so far.** peg (+6), waist (+10) and disc (+14) each have a positive pooled score, no guard harm, and fail only the "≥ 2 net success flips" clause by one.
- In low-success cells, success flips are rare at n = 10.
- The board scores the mean SCORE, not the success rate.
- These verdicts stand as registered. The refined rule for future pre-registrations is in EXPERIMENTS.md.

## reloc re-test: `watch_relocate=true`, make_kong (fresh salted layouts) — NOT ADOPTED (guard failed)

| cell | pairs | A pooled | B pooled | Δ | B-only wins | A-only losses | successes A / B |
|---|---|---|---|---|---|---|---|
| make_kong (gain) | 10 | 40.0 | 70.0 | **+30.0** | 4 | 1 | 4 / 7 |
| imitate_sorting_sequence (guard) | 8 | 13.8 | 5.0 | −8.8 | 0 | 1 | 1 / 0 |

- **What passed:** the gain cell, clearly (+30; flips 4 vs 1, net 3).
- **What failed:** the imitate guard (Δ −8.8 < −5, losses 1 > wins 0), so the lever is not adopted. This was its one fresh-layout re-test.
- **Missing guard pairs:** two imitate guard pairs (b1 100003, b2 100001) were lost to infra-only startup stalls (rc 4, 0 steps, retry exhausted), so the guard has n = 8.

**Follow-up (a new lever, not a third test):** make_kong's gain is large, and the only harm shows up in a different task. The natural next lever is `watch_relocate` scoped to make_kong only, so it can't change imitate by construction. It needs its own pre-registration on new fresh layouts, under the current score-based rule.
