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

## seg v2n additive (#387), compute6 c6b_ — DOES NOT PASS

This is SEGAB's report.py, run once after all 140 rows completed (23:08Z), under the spec's rule: equal-cell pooled gain ≥ +5, wins ≥ 2× losses with ≥ 2 net, and guards.

| cell | pairs | Δ (B − A) | wins / losses |
|---|---|---|---|
| make_toast_random (gain) | 10 | 0.0 | 0 / 0 |
| hang_mugs_random (gain) | 10 | −1.0 | 0 / 0 |
| sort_nesting_dolls_by_size_random (gain) | 10 | 0.0 | 0 / 0 |
| store_laptop_and_headphones_random (gain) | 10 | 0.0 | 0 / 0 |
| pour_liquid_into_cup_random (gain) | 10 | −10.0 | 2 / 3 |
| stack_blocks (guard) | 10 | 0.0 | 0 / 0 |
| insert_tubes (guard) | 10 | 0.0 | 0 / 0 |

- The pooled gain's bootstrap 95% CI is [−10.5, +6.7].
- Cost: 833 agentic calls (claude/claude-sonnet-5-5), 12.2M prompt / 0.75M completion tokens, 9.9M cache-read.

**Reading.** v2n names 24 / 49 new-class objects on dev frame-0 (vs 0 for v2b) without hurting old classes, yet four of five gain cells are byte-for-byte equal in score between arms. So naming the new objects is not what limits these cells. Their G3 ceilings are low or zero (store_laptop 0, hang_mugs_random ~20, dolls_random ~50), and make_toast_random was earlier blocked by the toaster vocabulary in routing. The segmenter stays at v2b, and `seg_additive` stays default-off. The next Generalization work should start from each cell's G3 failure class, not from perception.

## key re-test: `key_handover=v2, key_lean=on`, insert_key (fresh salted, n = 9) — NOT ADOPTED (guard failed)

| cell | pairs | A pooled | B pooled | Δ | B-only wins | A-only losses | successes A / B |
|---|---|---|---|---|---|---|---|
| insert_key (gain) | 9 | 43.3 | 70.0 | **+26.7** | 3 | 0 | 3 / 6 |
| plug_in_charger (guard) | 10 | 80.0 | 70.0 | −10.0 | 1 | 2 | 8 / 7 |

- **What passed:** the gain cell (+26.7; flips 3 vs 0, net 3).
- **What failed:** the plug_in_charger guard (Δ −10 < −5, losses 2 > wins 1), so the lever is not adopted. This was its one fresh-layout re-test.
- 38 / 38 jobs had rc 0.

**Follow-up (a new lever):** the switches change the shared insertion / handover path that plug_in_charger also uses. As with kong, scope them to the routed insert_key task only, then pre-register a new A/B on new fresh layouts. The gain is worth ~0.67 Average points.

## pen: #397 conditional inset, fill_pen_holder (compute4, moved from compute3) — NOT ADOPTED (gain below +5)

| cell | pairs | A pooled | B pooled | Δ | success flips (B / A) | score wins / losses (≥ 10 pts) | successes A / B |
|---|---|---|---|---|---|---|---|
| fill_pen_holder (gain) | 10 | 4.0 | 7.0 | +3.0 | 0 / 0 | 4 / 1 | 0 / 0 |

- The 100020 pair ran once as the disclosed `__rerun_c4`. One infra attempt (100029 A) was retried under the rule.
- The pooled gain of +3 is below the +5 bar, so the lever is not adopted. The direction matches G3 (+6 pooled), but the camera effect is smaller.

## 7 Nov 02:14Z: KONG / KEY scoped A/Bs moved to the candidate perceiver

The compute6 rows for both scoped levers (#410/#411 KONG, #412 KEY) pinned
`AGENTIC_MODEL=cx/gpt-6-luna`. The lead's "no Claude-route calls" brief was meant
for dev agents, but it was carried over to the camera rows. Adoption is
measured on the candidate perceiver (`claude/claude-sonnet-5-5`). The tie and
seg rows already used it.

- **The fix:** `STOP` was set and the not-yet-started KEY rows were rewritten.
  All 40 KONG rows were re-queued blind as `__sonnet`. The single started Luna KEY
  arm (`keyAr1b1_100001`) is excluded and was re-run as `__sonnet` at the head
  of the queue.
- **Data status:** no Sonnet result had been read. The Luna KONG panel was
  complete but had no verdict, so it stays supplementary. See the amendments in
  both preregs.
- **Restart:** heavy2 on compute6 exited on the `STOP`. It was restarted by the
  dispatcher's own command (sources `dojo-env.sh` + `lane.env`, segab lane
  script).
- **compute4 caps:** `lane2.caps` raised from `1 4` to `2 4`. Since D-C19 there
  are no other GPU users, and the 32 GB card fits two Isaac jobs. The VRAM and
  boot gates are unchanged, and heavy2 was started on the same 416a2a03
  deployment. This roughly halves the peg / waist re-test tail.
