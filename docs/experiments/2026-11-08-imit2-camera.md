# 2026-11-08 — IMIT2 camera A/B verdict: home wait + kept sides (C vs A), IMIT2 switches (B vs C)

Pre-registration: [2026-11-07-imit2-camera-prereg](2026-11-07-imit2-camera-prereg.md) (on main `e649a2c1`, inputs
frozen `6d10fa0c`, queued 16:06:53Z, [receipt](data/imit2-camera/dispatch.json)). Code `89b194be` on compute6, fresh
salt `imit2-camera-20261107-7494a61e…` over ids 100000–09 / 100020–29, `claude/claude-sonnet-5-5`. Collection
2026-11-07 ~18:50Z – 2026-11-08 04:03Z, 90 run dirs (87 rows + 3 infra retries). Data read-only from
`compute6:~/dojo-data/integ-c6/runs2/c6b_*_imit2*` and the lane logs; no new runs; scores from each row's episode
record.

## Verdicts

| Comparison | Gain cell (imitate, n = 16 pairs) | Strong guard stack_blocks (5) | Parity guard classify_objects (5) | Write audit | Verdict |
|---|---|---|---|---|---|
| **C vs A** (home wait + kept sides) | 12.19 → 10.00, **Δ −2.19**; W/L/T **2 / 2 / 12**; flips 0 / 0; bootstrap 95 % [−10.3, +3.8] | 83 → 100 (+17), flips 1 / 0: pass | proof holds; score 66 → 54 (reported) | not computed (no replays) | **FAIL** (gain gate) |
| **B vs C** (four IMIT2 switches) | 10.00 → 23.75, **Δ +13.75**; W/L/T **7 / 3 / 6** (net +4, 7 ≥ 2 × 3); flips 2 / 0; bootstrap 95 % [+1.3, +29.7] | 100 → 100, flips 0 / 0: pass | proof holds; score 54 → 71 (reported) | **not computed** (registered audit needs physics replays; none authorized) | **gain + guards PASS; not passed as registered** (missing audit coverage fails the gate) — and **not adoptable** under the hard rule (§ Hard-rule check) |

Score wins / losses are per-pair score differences ≥ 10 points. Bootstrap: 10 000 resamples of the paired ids, seed 0.
Two of B's seven wins are exactly at the 10-point threshold (100000, 100008); the pass is not robust to small changes.

Per-pair imitate scores (A → C → B), included ids:
100000 5 / 5 / 15 · 100003 15 / 15 / 5 · 100004 15 / 15 / 15 · 100005 5 / 5 / 5 · 100006 30 / 30 / 50 ·
100007 0 / 0 / 0 · 100008 5 / 5 / 15 · 100009 5 / 5 / 5 · 100020 0 / 15 / 0 · 100021 0 / 15 / 0 · 100022 15 / 0 / 15 ·
100023 0 / 0 / **100** · 100026 50 / 50 / **100** · 100027 0 / 0 / 5 · 100028 0 / 0 / 0 · 100029 50 / 0 / 50.

### Exclusions (prereg rules)

- **100001: infrastructure in both comparisons.** Every attempt:

  | Row | rc | steps | outcome |
  |---|---|---|---|
  | imit2Cr1_100001 | 4 | 0 | watchdog: "no env step for 600 s (at 0:), scope CPU 4.73 cores: killing the scope" |
  | imit2Cr1_100001__retry | 0 | 951 | perception_gross, score 0 (lock 841 s) |
  | imit2Br1_100001 | 4 | 0 | watchdog: "no env step for 601 s (at 0:), scope CPU 7.79 cores" |
  | imit2Br1_100001__retry | 4 | 0 | watchdog: "no env step for 601 s (at 0:), scope CPU 7.6 cores" |
  | imit2Ar1_100001 | 4 | 0 | watchdog: "no env step for 600 s (at 0:), scope CPU 7.96 cores" |
  | imit2Ar1_100001__retry | 4 | 0 | watchdog: "no env step for 600 s (at 0:), scope CPU 7.53 cores" |

  A's and B's retries also failed on infrastructure, so 100001 leaves C vs A (no A) and B vs C (no B). Every failed
  attempt stalls before the first env step (the reset / demo setup), consistent with the demo-timing landmine below.
- **100024, 100025: RoboDojo flagged the episode unstable** (its post-demo check: the demonstration did not leave
  every aim in basket1; "not a valid eval sample"). 100025 is unstable in all three arms; 100024 in A and B (C hit the
  25-minute wall cap at 0). Both leave both comparisons. Counting them as zeros instead changes nothing (all ties).
- 100002 was excluded at prep (no valid demo); the gain cell is 16 pairs per comparison, not 20.

### Guards

- **Strong guard, stack_blocks (gated):** A 83 (100002: belief_mismatch, 15), C 100, B 100. C vs A +17, B vs C 0; no
  flip losses. Pass in both comparisons.
- **Parity guard, classify_objects (proof gated, score reported):** on all 15 guard episodes (and all 15 stack
  episodes) zero `[imitate]` / re-find / far-side log lines, zero `twin_costs` events, zero `confirm_pose` events. The
  resolved configs differ only in the imitate fields (checked at staging: all 87 rows construct, C − A = `imitate_wait`,
  `imitate_keep_sides`; B − C = the four IMIT2 fields). The proof holds. Scores A 66, C 54, B 71 (classify's own
  perception spread; n = 5).

### Route and spend

All 85 rows that reached perception have agentic call records, **every one `claude/claude-sonnet-5-5`**; the 5 rows
with no records are the 100001 infra attempts (killed before the first perceive). No HTTP 429 anywhere; 36 recoverable
"no JSON object" turn errors. **441 Sonnet calls** (172 perceive records): prompt 6.40 M, completion 0.43 M,
reasoning 0.27 M, cache read 5.08 M tokens. By arm: imitate A 95 / C 116 / B 103, stack 14 / 14 / 14,
classify A 28 / C 27 / B 30. **This exceeded the registered estimate of ≤ 261 calls**: the estimate counted perceive
invocations (≤ 3 per episode), but one invocation makes 1–4 model turns (`calls` per record).

### Write audit

The registered audit (`m95_score.py audit` with the #417 identity instrument on physics-only replays of both arms)
was **not run**: it needs simulator replays, and this verdict was to use logs only. Per the prereg, missing audit
coverage fails the gate, so neither comparison can pass on the audit until it is run.

Log-only supplement for the new writer (not the registered audit): `imitate_retry=refind` proposed a start spot through
`confirm_pose` 6 times in B (100000 ×1, 100008 ×4, 100022 ×1). Perception adopted **one** (100000, `t1` at step 1122).
The nearest layout object to the proposed spot (`t4`) never rose (`dev_gt_probe` rise 0 mm). The write moved the
belief from 52.2 mm to 22.9 mm from that object's start pose, so on that reading it was helpful. The other five were
rejected, so no belief changed. The 100008 proposal repeated 4 times; a repeated rejected proposal should not be
retried.

## Stage split (the 16 included ids; dev scorer: spatial GT from the fresh layouts, demo order from the frozen pickles)

| Stage | A | C | B |
|---|---|---|---|
| departure order exactly right | 13 / 16 | 14 / 16 | 12 / 16 |
| all 5 departures read (B: with elimination) | 13 | 15 | 15 |
| twin pairs right | 59 / 80 (74 %) | 58 / 80 (73 %) | **65 / 80 (81 %)** |
| all five twins right | 5 | 5 | **9** |
| plan right (order and twins) | 5 | 5 | **9** |
| objects placed (sum) | 21 | 18 | **29** |
| air closes / closes | 18 / 63 | 11 / 46 | 32 / 77 |
| RoboDojo query endings | 8 | 5 | 5 |

- On these fresh layouts the departure order is mostly right in every arm (13–14 / 16, vs 18 / 42 on the old rows).
  `persist_elim` did not improve it here (12 / 16; elimination fired in 4 B episodes). The order misses' causes were
  not diagnosed from the logs (per-read departure frames are not saved).
- `cues` raised the all-right twin plans from 5 to 9, as offline (fewer than the offline 90 % pairs: 81 %).
- B places more objects but needs more grasp attempts (`refind`'s retries: 77 closes, 32 on air). Wall time is now a
  limit: five included episodes ended on the 25-minute wall cap with objects still to copy (A 100006 / 100026,
  C 100026, B 100006 / 100029).
- C's home wait and kept sides did not show a gain: query endings fell 8 → 5 and C gained 100020 / 100021, but it
  lost 100022 and 100029 (no object placed: failed puts of the first twin).

## Hard-rule check (Ilia, 7 Nov: one general system; no task names / task-keyed code / checker values)

- `imitate_carry=reach` contains `CROSS_TRIES_KEYS = ("phone",)` in `dojo/skills/interactive.py`: a category key
  taken from the task teardown ("phones by a table handover"). This is task-shaped. **B cannot be adopted while it
  exists.** No adopted switch depends on it: nothing from IMIT2 is adopted, and C's switches (`imitate_wait`,
  `imitate_keep_sides`) do not touch it. Only `imitate_carry=reach` reads it.
- **Proposed general replacement:** no category list. For an object on the far side, the container arm tries centred
  grasps first. A table handover is chosen by evidence: no plan for the container arm, or a failed direct attempt on
  this object (slip / air close: `grip_check` or no rise). Shape could also be used, from the perceived footprint's
  aspect and height (a long, flat object), with thresholds set from data and not from the task.
- Other constants in the four switches are generic (step counts, read counts, a reach margin in metres, a 40 mm
  re-find radius) and come from no task config or checker. The `cues` read uses no category. One more point for the
  lead: the skill itself is reached through the runtime vocabulary entry keyed `imitate_sorting_sequence`
  (`dojo/runtime/supervisor.py`), a task-name route outside IMIT2.

## The bar

REPRO's PyRUA-RSI yardstick on our first five canonical dev layouts: imitate **20.0** (n = 5, one success). Its four
losses started manipulation before the demonstrator was home, because our generated demos last 669–728 actions and
its wait is capped at 640 (journal 2026-11-07-rsi-repro). B here is 23.75 on 16 fresh-salt pairs (A 12.2, C 10.0).
These are different layouts with the same demo-timing confound, so the numbers are not comparable as a bar. Official
demos are likely ≤ 640 actions; until our dev demos match official timing, absolute imitate numbers may not transfer.
The arm comparisons stay valid (identical demos per id).

## Conclusions (n = 16 pairs; one repeat)

1. Home wait + kept sides (C) show no camera gain over the candidate. Δ −2.2 with 2 / 2 wins; rejected.
2. The four IMIT2 switches on top of C clear the score gain rule and both guards (Δ +13.8, 7 / 3, flips 2 / 0). This
   comes mostly from two full copies (100023, 100026) and the twin plans (5 → 9).
3. As registered, B vs C is **not passed**: the write audit has no coverage. Whatever the audit says, B is **not
   adoptable** until `CROSS_TRIES_KEYS` is replaced by a general rule.
4. Spend overran the registered call estimate (441 vs ≤ 261): the estimate counted perceive invocations, not model
   turns.

## Next (on the lead's go)

- The demo-timing work: match official demo length; expected to also clear the 100001 reset stall.
- Replace `CROSS_TRIES_KEYS` with the evidence-based handover rule; stop repeating a rejected re-find proposal.
- If B is to be re-tested: run the registered write audit on physics replays (needs the lead's authorization), then a
  fresh-layout re-test of the general version.
