# 2026-11-08 — CLASSIFY3 camera A/B verdict: the sort skill's lowest-point container test

**Verdict: FAIL** under the pre-registered rule ([prereg](2026-11-07-classify3-camera-prereg.md)). `sort_contain`
stays `root` (default). Evidence that motivated it: [CLASSIFY3 report](2026-11-07-classify3.md).

## What ran

- **Panel:** 50 camera episodes (`c4b_*_cls3{A,B}r1_*`), compute4 lanes heavy1 / heavy2, 23:33Z–01:57Z.
- **Inputs:** deployment `2f198642`, all rows rc 0. Fresh salt `classify3-cam-20261107-040c…`, ids 100020–29 (sort cells)
  and 100020–24 (stack_blocks). One episode record per row, 0 unstable, 0 infrastructure failures.
- **Arms:** A = the candidate + `instance_evidence=log`; B = A + `sort_contain=bottom`.
- **Data:** per-row data in [verdict_rows.json](data/classify3-camera/verdict_rows.json), read-only via
  [verdict_collect.py](data/classify3-camera/verdict_collect.py). It reads only these 50 run directories under
  compute4 `~/dojo-data/integ-c4/runs2`.

## Decision (as registered)

| Cell | Pairs | A pooled | B pooled | Δ | Wins / losses / ties (≥ 10 pts) | Success flips |
|---|---|---|---|---|---|---|
| classify_objects | 10 | 30.5 | 25.0 | −5.5 | 1 / 3 / 6 | 0 / 0 |
| classify_objects_by_language | 10 | 7.0 | 11.0 | +4.0 | 3 / 1 / 6 | 0 / 0 |
| **gain cells pooled** | **20** | **18.75** | **18.00** | **−0.75** | **4 / 4 / 12** (net 0) | 0 / 0 |
| stack_blocks (strong + parity guard) | 5 | 100.0 | 100.0 | 0.0 | 0 / 0 / 5 | 0 / 0 |

1. **Gain: FAIL.** Pooled +5 is not met (−0.75); wins ≥ 2 × losses is not met (4 vs 4); net ≥ 2 is not met (0).
2. **Strong guard: PASS.** stack_blocks B 100.0 vs A 100.0, no flip losses.
3. **Parity guard: PASS.**
   - Every pair's resolved config is identical apart from `sort_contain` (hash over all other fields, all 25 pairs).
   - No stack_blocks episode ran a sort skill: histories are only pick / place / bank / jobs.
4. **Write audit: PASS** (registered gate: no B-only writer).
   - Sort cells: A `perception` 138 / `predicted` 72; B `perception` 139 / `predicted` 68.
   - stack_blocks: A 4 / 13; B 7 / 11.
   - No other writer in either arm.
   - The #417 identity-association instrument scores harmful or helpful writes against physics replays. The
     registered gate is the writer inventory, B adds no writer, and no replays were run (no new runs): it was not
     needed and not applied.

Per-pair scores (A → B):
- classify: 100020 0 → 0; 100021 .40 → 0; 100022 .40 → .40; 100023 .40 → .40; 100024 1 → 1; 100025 .15 → .15;
  100026 .40 → .15; 100027 .15 → 0; 100028 0 → 0; 100029 .15 → .40.
- language: 100020 .10 → 0; 100021 0 → 0; 100022 .40 → .40; 100023 0 → 0; 100024 0 → .10; 100025 0 → .10;
  100026 .10 → .10; 100027 0 → 0; 100028 0 → 0; 100029 .10 → .40.

## Route and spend

- **Route.** Every agentic call record (`agentic/call_*/calls.jsonl`), every transcript request and every report
  `agentic` entry of all 50 rows names `claude/claude-sonnet-5-5`. No row used another model, every row made calls,
  and there were 0 HTTP 429s.
- **Spend:** **273 Sonnet calls** (A 134, B 139; the prereg estimated ≤ 300).
  - Prompt 4,064,236 tokens, of which 3,286,129 were cache reads.
  - Completion 243,069 tokens, including 133,049 reasoning.
  - List-price equivalent at $2 / $10 / $0.20 per M (input / output / cache read): **$4.64**, or $5.03 if every
    uncached prompt token was a 5-minute cache write. The route is omniroute's Claude subscription, so the
    incremental bill is $0. The reading and this verdict made no calls.

## What the data say beyond the gate

- **The lever did change the skill's own claims.** In the sort cells A's records read "in the container" 58 of 117
  times; B's read it 74 of 103 times. So B had fewer records and more "in" claims: fewer re-dos, as predicted offline.
  It did not turn into score. Without physics replays (no new runs) this verdict cannot split true from false "in"
  claims on these rows.
- **Three rows crashed at skill step 0 in shared p2 code:** `KeyError: 'garage_15_0'`, raised before any placement.
  - classify 100020 crashed in both arms.
  - classify 100021 crashed in **B only**, which costs B that pair (0.40 → 0). The crash depends on the episode's
    perception, not on the lever.
  - These are outcomes under the prereg (rc 0, not infrastructure), so the verdict counts them.
  - Diagnostic only: without the two affected pairs the gain cells read 18.61 → 20.00 (+1.4), 4 wins / 3 losses,
    which still fails every gain condition. The verdict does not change.
- **The ceiling is elsewhere.** A's classify scores (30.5) and language scores (7.0) on fresh layouts sit where the
  D-C19 rows did (35 / 13).
  - The CLASSIFY3 report's stage audit puts the language loss at frame-0 instance count.
  - It puts classify's at the step budget, plus carries refused by the idle arm.
  - The container test alone cannot move either.

## Under the hard rule (one general system)

On these cells the instance count still comes from the task config's range (`_series_count`'s `[lo, hi, step]`), and
the hard rule forbids that count. Whatever this verdict had been, adopting `sort_contain=bottom` would only change the
general `sort_into_containers` skill's library code. It would not license or depend on the forbidden count: the skill
reads whatever instances the world holds. The general count belongs to the evidence-ledger / general-union plan in
the CLASSIFY3 report's addendum.

## Conclusions (n = 20 gain pairs)

- `sort_contain=bottom` does not raise the camera score of the sort cells (pooled −0.75, 4 / 4). It stays off. The
  pooled delta is negative, so §Re-tests allows no re-test; it is never re-run on these ids.
- The guards hold, the write audit holds, and the route is clean.
- **Next:**
  1. Fix the p2 `KeyError` crash: 3 of 40 sort episodes died at step 0 (BACKLOG).
  2. Build the general count before more sort-skill levers. The language cell scores 7 with a pinned count of 6.
