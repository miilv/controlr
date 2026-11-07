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

## KEY scoped (#412): insert_key-only handover/lean, candidate Sonnet — DOES NOT PASS

[Preregistration and full verdict](../experiments/2026-11-07-key-scoped-prereg.md).
This uses the current **score-based** rule (pair win/loss >=10 points), not
success flips for the gain gate. The lead's route amendment `74d9ed80` is
applied: excluded Luna `c6b_insertkeysg_keyAr1b1_100001` is replaced by its
`__sonnet` A arm. Every analysed arm verifies Sonnet in its actual call logs,
transcript start and report agentic metadata; no further route violations.
Static deployment `code.model` remains stale Luna and is not route evidence.

| cell | pairs | A pooled | B pooled | Δ | success flips (B / A) | score wins / losses (>=10 pts) | successes A / B |
|---|---|---|---|---|---|---|---|
| insert_key (gain) | 10 | 23.5 | 23.5 | 0.0 | 1 / 1 | 1 / 1 | 1 / 1 |
| plug_in_charger (guard) | 5 | 80.0 | 80.0 | 0.0 | 0 / 0 | 0 / 0 | 4 / 4 |
| stack_blocks (strong guard) | 5 | 100.0 | 100.0 | 0.0 | 0 / 0 | 0 / 0 | 5 / 5 |

- **What passed:** all 40 analysed arms rc 0, complete 10/5/5 paired panel,
  actual Sonnet route and frozen config/input checks; both guards meet delta
  >=−5 and success-flip losses <=wins. No time caps or missing outcomes.
- **What failed:** gain delta 0 < +5, score wins 1 < 2×1 losses, and net
  score wins 0 < 2. Paired bootstrap 95% CI **[−25.5, +25.5] points**
  (10,000 paired-layout resamples, seed 0). The scoped lever is not adopted;
  no candidate or default change. The earlier global verdict stands.
- **Spend:** **121 logical Claude calls**; **1,725,976 prompt / 84,499
  completion / 49,981 reasoning / 1,419,036 cache-read tokens**. Reasoning /
  cache are reported separately without double counting. Excludes Luna.
  **Actual recovered transport retries are unobservable:** OmniChat drops
  attempt counters/status. **Zero surfaced transport failures or logged
  429s**, not a certified zero transient-retry count. Summed lane locks:
  5,234 s; analysed episode wall: 3,160.5 s, excluding admission waits.
- **Write audit:** not triggered under the preregistration: these are existing
  motion-search/waypoint-budget switches, with no changed perception read,
  belief writer or acceptance predicate. Downstream observations/writes may
  differ with motion; no physics-audit pass is claimed.

Per-pair scores, per-row model proof, raw source hashes and spend are retained
in [the snapshot](../experiments/data/key-scoped/verdict.json) and
`compute6:~/dojo-data/key-1107/`. No further run is authorized by this verdict.

## peg re-test: `peg_held_read=on`, fresh salted panel — CAMERA PASS; NOT ADOPTED (write audit fails)

Uses exactly the pre-7-Nov registration's success-flip gate; the later
score-win refinement is supplementary only. Final arm completed at 06:03:17Z.
[Full verdict, route/spend evidence and audit](../experiments/2026-11-06-peg-waist-retest-prereg.md#peg-re-test-camera-verdict-7-november).

| cell | pairs | A pooled | B pooled | Δ | B-only wins | A-only losses | successes A / B |
|---|---|---|---|---|---|---|---|
| play_stacking_toy (gain) | 20 | 55.5 | 63.5 | +8.0 | 5 | 2 | 3 / 6 |
| stack_blocks (guard) | 10 | 100.0 | 100.0 | 0.0 | 0 | 0 | 10 / 10 |
| insert_tubes (guard) | 10 | 82.0 | 82.0 | 0.0 | 0 | 0 | 7 / 7 |

- **Camera gate passed:** +8 >= +5; success flips 5 >= 2×2, net 3 >= 2;
  both guards pass. Supplementary >=10-point score wins/losses: gain 7/4,
  guards 0/0 each. These do not change the registered verdict.
- Sample SD A/B: gain 23.73/28.52, blocks 0/0, tubes 28.98/28.98.
  Paired bootstrap gain 95% CI **[−6.0, +22.5] points** (10,000 paired-layout
  resamples, seed 0).
- **Integrity:** all 80 arms rc 0, complete 20/10/10 pairs, actual Sonnet
  model verified in every call log and report. No infra exclusions/retries
  or time caps; 19 included parsing errors, with recovered HTTP retry counts
  unobservable. Frozen effective configs and loaded-layout hashes match.
- **Spend:** **331 logical Claude calls**; **5,022,186 prompt / 313,569
  completion / 188,655 reasoning / 3,989,714 cache-read tokens**. Reasoning
  and cache are subsets, not additions. No new audit/verdict LLM calls.
- **No adoption:** the required write audit has excess harmful held-pose
  writes, A 0 / B 22. Its full coverage and per-writer counts are recorded in the
  linked verdict. No candidate or default changed. The camera-only candidate
  delta would be exactly `--config.peg_held_read: off → on`; the failed write
  gate prevents promoting it. This consumes the once-only fresh-layout re-test.

Waist is excluded from this analysis and remains a separate verdict request.

Peg audit completion: **80/80 trajectories, 1,696/1,696 writes**, zero uncertain/missing frames; 80/80 terminal scores and 532/532 live grasp probes agree. The full per-writer harmful/helpful/neutral counts and separate z metrics are in the linked prereg verdict and [audit snapshot](../experiments/data/peg-retest/audit-summary.json).

## KONG scoped (#410/#411): make_kong-only relocation, candidate Sonnet — DOES NOT PASS

[Preregistration, full write audit and verdict](../experiments/2026-11-07-kong-scoped-relocate-prereg.md).
The explicitly registered **score-based** gain rule applies (pair score W/L
≥10 points). Under lead amendment `74d9ed80`, only the 40 `__sonnet` rows are
primary. Every row verifies `claude/claude-sonnet-5-5` in actual call logs,
transcript starts and report agentic metadata: zero primary route violations.
Static deployment `code.model` remains stale Luna and is not route evidence.

| cell | pairs | A pooled | B pooled | Δ | success flips (B / A) | score wins / losses (≥10 pts) | successes A / B |
|---|---|---|---|---|---|---|---|
| make_kong (gain) | 10 | 40.0 | 60.0 | **+20.0** | 2 / 0 | 2 / 0 | 4 / 6 |
| imitate_sorting_sequence (guard) | 5 | 23.0 | 7.0 | **−16.0** | 0 / 1 | 0 / 2 | 1 / 0 |
| stack_blocks (strong guard) | 5 | 100.0 | 100.0 | 0.0 | 0 / 0 | 0 / 0 | 5 / 5 |
| Luna supplementary: make_kong | 10 | 40.0 | 50.0 | +10.0 | 1 / 0 | 1 / 0 | 4 / 5 |
| Luna supplementary: imitate | 5 | 7.0 | 11.0 | +4.0 | 0 / 0 | 1 / 0 | 0 / 0 |
| Luna supplementary: stack | 5 | 100.0 | 80.0 | −20.0 | 0 / 1 | 0 / 1 | 5 / 4 |

- **What passed:** gain delta +20 ≥+5, score wins 2 ≥2×0 losses and net 2;
  stack guard. All 40 primary arms rc 0 and complete 10/5/5 pairs, actual
  route and frozen-input/config checks. Primary paired bootstrap 95% CIs:
  kong **[0, +50]**, imitate **[−44, 0]**, stack **[0, 0]** points
  (10,000 paired-layout resamples, seed 0; descriptive).
- **What failed:** imitate delta −16 <−5 and success losses 1 >0 wins;
  the full raw write gate also fails. The lever is not adopted. Guard configs,
  resolved skill parameters and excluded relocation paths match by construction
  and nine focused tests; both guards have zero relocation writes. Independent
  stochastic camera outcomes still vary, and the registered guard is retained.
  B imitation 100001/100003 wall caps count as outcomes. No episode reruns or exclusions.
- **Sonnet write audit:** **249/249 writes**, 40/40 arms, zero uncertain, 40 stable
  resets; frozen hashes unchanged. Pooled harmful A/B: perception 31/39, predicted
  13/13, watch_lay 6/6, watch_relocate 0/2. Relocation B has 16 helpful / 11 neutral / 2
  harmful (A zero). Same-label diagnostics are supplementary; raw nearest-object gate
  counts stand.
- **Primary spend:** **142 logical Claude calls**; **2,182,217 prompt /
  150,450 completion / 99,674 reasoning / 1,687,743 cache-read tokens**.
  One surfaced internal-server stream error, 11 no-JSON parser errors,
  zero logged 429s. Recovered HTTP retries and their usage are unobservable
  because OmniChat drops attempt/status metadata; zero transport retries is
  not claimed. Summed lane locks 14,283 s; episode wall 12,080.9 s.
- **Luna supplementary, never pooled/adopted:** gain misses net wins; stack
  guard fails. CIs: kong [0, +30], imitate [0, +12], stack [−60, 0]. Its raw
  audit remains 228/228 writes with harmful excesses. **74 logical Luna calls**;
  **763,614 prompt / 25,666 completion / 13,342 reasoning / 588,672 cache-read**.
  Three disconnect failures explicitly exhausted three attempts each (≥6 HTTP
  retries); recovered retries remain unknown; zero logged 429s.

The final primary arm completed at **06:16:41Z** (lane log). Per-pair scores,
spread, per-row route proof, raw source hashes and reported spend are in
[the evidence snapshot](../experiments/data/kong-scoped-relocate/verdict.json).
No candidate/default field changes, new camera dispatch or adoption.

## Lead, 7 Nov ~07:00Z: two refinements to the adoption gates (prospective only)

These were written after the peg re-test and the KONG scoped verdicts. They apply only to pre-registrations from
2026-11-07 07:00 UTC; every earlier A/B keeps the rule it registered, and none is re-judged. See EXPERIMENTS.md.

- **Write audit, per writer.**
  - An existing writer passes when its harmful RATE is no more than 5 percentage points above A's. Raw totals of
    downstream writers mostly track trajectory length after divergence.
  - A writer new in B passes when helpful ≥ 2 × harmful. Against A's zero, every new read failed by construction.
- **Parity guards for scoped levers.** A guard on which the lever is proven inactive (identical resolved config and
  skill parameters, zero lever actions and writes in every guard episode) is gated by that proof, not by its noisy
  camera score. One strong guard (≥ 90 % cell) stays camera-gated.

Why:
- KONG scoped (make_kong only) failed on an imitate guard of −16 at n = 5, with the code path identical by
  construction and zero relocation writes.
- Its new relocation writer (16 helpful / 2 harmful) and peg's held_pose writer (11 helpful / 22 harmful) were judged
  the same way, by "any harmful write against A's zero".
- The refined gates still reject peg (its new writer fails 2×), so they separate the two cases on mechanism, not on
  outcome.

make_kong under scoping has been positive in every test so far: +20 at n = 15 historically, +30 global, +10 Luna,
+20 Sonnet. **KONG scoped gets its one fresh-layout re-test under the refined rule**: new layouts, 20 make_kong pairs,
imitate as a parity guard, stack_blocks as the strong guard. It is queued on compute6 after the tie block. Peg's one
re-test is used up; a mechanism study (offline, no LLM calls) decides whether a new lever follows.

## Lead, 7 Nov ~08:00Z: audit instrument fixed (#417); peg stays NOT ADOPTED

#417 fixes two instrument bugs that TOY found (journal 2026-11-07-dojo-peg-mechanism):
- held-pose writes logged a stale, pre-sync `before` pose that included the arm's transport;
- the audit took the tracked object as the nearest GT in the label group, which misassigns multi-instance labels.

The threshold and metric are unchanged.

Supplementary re-scores with the fixed instrument, which change no registered verdict:

| writer | harmful / helpful / neutral |
|---|---|
| peg B `held_pose` | 0 / 13 / 134 (was 22 / 11 / 114) |
| KONG Sonnet `watch_relocate` | 0 / 18 / 11 (was 2 / 16 / 11) |
| peg `perception`, harmful A / B | 49 / 50 |
| KONG `perception`, harmful A / B | 10 / 12 |

**peg_held_read stays NOT ADOPTED.** Its prereg uses the raw-count gate, which still fails by one perception write
even with the fixed instrument. Adopting it would need both the fixed instrument and the prospective refined gate
applied retroactively to data collected under the old registration. The value at stake is about +0.18 Average points,
which does not justify that. The hover-only transactional read TOY proposed would remove the show-time reads, so it is
a different lever. It is a candidate for a fresh A/B under the refined gates when a camera slot frees up; see BACKLOG.

The KONG scoped re-test (prereg 2026-11-07-kong-scoped-retest) starts after #417 merged, so its registered audit uses
the fixed instrument.

## waist re-test: `belt_waist_first=on`, fresh salted panel — FAIL / NOT ADOPTED

Exactly the original success-flip rule; newer score-based counts remain
supplementary. [Full verdict and applicability proof](../experiments/2026-11-06-peg-waist-retest-prereg.md#waist-re-test-camera-verdict-7-november).

| cell | pairs | A pooled | B pooled | Δ | B-only wins | A-only losses | successes A / B |
|---|---|---|---|---|---|---|---|
| match_and_pick_from_conveyor (gain) | 20 | 50.0 | 35.0 | −15.0 | 0 | 3 | 10 / 7 |
| pick_from_conveyor_by_image (guard) | 10 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 / 0 |

- **Failed gain:** −15 < +5, wins 0 < 2×3, net −3 < +2. Guard passes its
  registered score/flip bounds. Supplementary >=10-point score wins/losses:
  gain 0/3, guard 0/0; these do not change the rule.
- Sample SD A/B: gain 51.30/48.94, guard 0/0. Paired bootstrap 95% CI
  **[−30.0, 0.0] points** (10,000 paired-layout resamples, seed 0).
- All **60/60 arms rc 0**; actual Sonnet route verified from every call log
  and report, frozen configs/RAW hashes match. Zero infra retries, exclusions
  or time caps. Six included no-JSON parsing errors; recovered HTTP retries
  remain unobservable. Final arm done at **09:16:08Z**.
- **Write audit does not apply:** frozen source confines the flag to motion/
  intercept candidate selection. All read/writer configs match; gain writers
  are perception/tracked/refine_pick in both arms, guard perception/tracked
  in both; zero held-pose/relocation writes. Logged write totals A/B 3,247/3,528
  differ with trajectories, not changed writer code. The linked proof reports
  all counts and the exemption scope; no harmlessness or physics-audit pass
  is claimed. Registered nearest/raw instrument and #417 supplementary fixed
  identity are both N/A, not retroactively substituted.
- **Spend:** **145 logical Claude calls**; **1,959,279 prompt / 75,055 completion /
  42,712 reasoning / 1,633,672 cache-read tokens**. Reasoning/cache are subsets,
  not additions. No new analysis LLM/GPU calls. Camera locks 13,813 s; episode
  wall 10,395.5 s, excluding admission waits.

No candidate or default changed. This was the once-only fresh-layout re-test;
no further re-test or adoption is authorized. Peg's registered verdict remains
unchanged, including the separately reported #417 supplementary re-score.

## POUR camera — budget pacing — FAIL / NOT ADOPTED

[Full registered verdict](../experiments/2026-11-07-pour-camera-prereg.md#pour-camera-verdict--fail--not-adopted).

| cell | pairs | A pooled | B pooled | Δ | success flips (B / A) | score wins / losses (≥10 pts) | successes A / B |
|---|---|---|---|---|---|---|---|
| pour_by_language (gain) | 20 | 0.0 | 43.0 | +43.0 | 7 / 0 | 15 / 0 | 0 / 7 |
| pour_liquid_into_cup (guard) | 10 | 60.0 | 50.0 | -10.0 | 1 / 2 | 1 / 2 | 6 / 5 |
| play_Xylophone (strong guard) | 10 | 80.0 | 90.0 | +10.0 | 1 / 0 | 1 / 0 | 8 / 9 |
| language, full-success-only | 20 | 0.0 | 35.0 | +35.0 | 7 / 0 | 7 / 0 | 0 / 7 |

- **Gain passes** official +43,15/0 score wins/losses and full-success-only
  +35,7/0. Eight 20-point partial scores are not counted as full success.
- **Cup guard fails** −10 <−5 and two paired losses >one win; Xylophone passes.
  No parity waiver was registered.
- Sample SD A/B: language 0/43.66,cup 51.64/52.70,Xylophone 42.16/31.62.
  Paired bootstrap 95% CIs (10,000,seed 0): language [25,62],full [15,55],
  cup [−40,20],Xylophone [0,30].
- **Audit FAIL (both gain writers exceed their harmful-rate bounds): 81/81 exact-step writes**,
  0 uncertain, 46/46 stable
  prefixes, nearest fallback 5/81.
  Identity/fidelity coverage limits remain unwaived (one 4.9 mm close mismatch;
  76 initial-tight/five fallback associations, no first-pick anchor sources).
  Pacing introduces no writer implementation/config change, but observed count
  changes trigger the registered audit. Per-writer rates/z and prefix-fidelity
  limits are in the linked evidence; guard writes are reported.
- **80/80 rc 0 and actual Sonnet routes**; frozen configs/input hashes match.
  Nine pre-episode ownership refusals excluded; no controller rerun, exclusion
  or wall cap. Final valid arm 2026-11-07T13:05:41Z.
- **Spend:** **467 logical Claude calls**; **6,537,401 prompt / 401,947 completion / 242,154 reasoning / 5,357,603 cache-read tokens**. Reasoning/cache are subsets. All 46 included no-JSON error
  entries retained; recovered HTTP retries unobservable. Analysis/audit 0 calls.
- **Absolute B 43 versus D-C19 candidate 0**, +43 points (unpaired historical
  aggregate context). The actual registered paired A is also 0.
- **Disclosure:** lead saw original 100000–09 outcomes during refusal diagnosis.
  Replacement used eligibility/validity only, unchanged salts/recipe.

No adoption or candidate/default change. [Evidence](../experiments/data/pour-camera/camera-verdict.json).
