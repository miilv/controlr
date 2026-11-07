# COIN once-only fresh-layout camera re-test: disc hold evidence

Prospective registration before generation or camera collection. This consumes
`disc_hold_evidence=on` (#382)'s ONE permitted re-test. The first panel's
positive pooled gain qualified it: deposit_coin 12 → 26 (+14), success flips
2 / 1; all three guards identical. It missed the old net-success-win bar.
See [wave verdicts](../journal/2026-11-06-ab-wave-verdicts.md) and the
[original registration](2026-11-05-coin-disc-hold-evidence-prereg.md).
No further re-test, tuning, optional stopping or outcome-based selection.

## Arms and freeze

Reuse the original disc deployment on compute4 (haptics):
`~/dojo-integ-c4-416a2a03`, clean tracked tree at
`416a2a0374e06628350816a019862caf5156e730`. Merged #382
(`c04335990e2b36c06fa5642998c4c2908d5f59b5`) is its ancestor.
Both arms use the captured candidate `scripts/dojo/queues/2026-11-06-candidate.txt`,
SHA256 `5cde3f9e684c0937f21a0f28609437e6d317a433e000bc171198349417cd398e`.
A retains disc_hold_evidence=off; B changes ONLY disc_hold_evidence=on.
Keep slot_read, held_thin and every other resolved field identical.
Both pin `AGENTIC_MODEL=claude/claude-sonnet-5-5` explicitly on every row.
Use submit_v3 / RuleSupervisor, Isaac 5.1 CAMERA belief, seg v2b,
digit_v2, kong_v1b, grip v2, original background and frozen color calibration.
No implementation-agent model probes. Stop on HTTP 429 via existing lane policy.
Freeze complete effective A/B configs, queue rows, code, inputs, catalog,
backgrounds, calibration, models and versions before camera dispatch.
The final manifest and this registration must be on main before queue append.

## Sample and fresh identities

| cell | role | pairs | five-record blocks |
|---|---|---:|---:|
| deposit_coin | gain | 20 | 4 |
| insert_key | guard | 10 | 2 |
| play_tic_tac_toe | guard | 10 | 2 |
| stack_blocks | guard | 10 | 2 |

One repeat per identity, 50 pairs / 100 camera rows. Guard set follows the
original disc panel; 10 pairs per guard mirrors the peg/waist re-test's policy,
expanding the initial three-pair guards rather than weakening regression coverage.
[Literal salts](2026-11-07-coin-disc-retest-salts.json) are fixed before generation.
Private roots: `~/dojo-data/coin-disc-retest-1107/blocks/<block>`;
each uses ids 100000–04 only. Identity is task + salt + id, not numeric id alone.
Fresh salts and new roots must not collide with the historical camera registry;
raw content hashes are checked against original disc inputs before dispatch.
Use unchanged generator and default two-pass presettle for settled cells;
if a task's task_info says settled=false, use raw plus unchanged --raw-check.
Keep all attempts, preparation indices and drops. Exactly one five-record refill
block per cell is reserved; only preparation drops permit ascending valid refill
ids to restore n. If still short, block dispatch; never substitute after outcomes.
Order AB for even zero-based sequential layout index and BA for odd, per cell,
continuing across blocks. Both arms start fresh episodes on the same box.
Infrastructure failures allow one identical-input retry per arm via lane2;
record all attempts; missing terminal pairs block adoption.

## Score decision rule (7 Nov)

The gain B pooled score must be at least A + 5 points. A paired score win is
B − A >= 10 points (0.1); loss is A − B >= 10. Require wins >= 2 × losses
and wins − losses >= 2. Report success flips alongside, not as the gain rule.
Each guard must have B pooled >= A − 5 points and no more success-flip losses
than wins. Full registered n must complete. Report per-cell mean/sample SD,
successes, paired score counts, success flips, infra exclusions and spend.
Adoption also requires the standard physics-only replay harmful belief-write
audit (>2 mm added 3-D error), no excess harmful writes in B by writer.
This task freezes and dispatches only; verdict and write audit are later work.

## Lane and budget

Append complete `c4b_` rows AFTER all queued work under
`flock ~/dojo-integ-c4-state/queue2.txt.lock`; do not rewrite existing bytes.
Existing heavy1/heavy2 `lane2.sh` retain admission gates, 50 GB disk floor,
1980-second job cap / 25-minute episode cap, private locks and retry policy.
Preparation serializes through our wave-start and heavy1 locks, respects the
same VRAM/disk limits, and never opens coord. Only lane2 accesses its boot stamp.
No sealed data, reserved ids >=100100, other team's paths, system changes or
process termination. Retain GPU/occupancy logs; timing is shared-box timing.

Budget estimated only from first disc A/B call logs (`agentic/**/calls.jsonl`,
summing `calls`, never outcomes): deposit_coin 134 calls / 20 episodes;
insert_key 23 / 6; play_tic_tac_toe 12 / 6; stack_blocks 14 / 6.
For 40 gain + 20 episodes per guard: 268 + 76.7 + 40 + 46.7 = **432 logical
Claude calls** (round up to 433). Allow about 650 for adaptive variation;
one infra retry per arm could increase this further. Historical token scaling
predicts 6.17M prompt, 276k completion, 143k reasoning (included in completion),
4.95M cache-read (included in prompt). Claude has a separate weekly limit;
remaining headroom is unknown, so this is not a claim of sufficient quota.
No live calls are needed for preparation. Actual calls/tokens at dispatch and
queue positions/hashes will be appended as a receipt without computing a verdict.

## Preparation freeze and non-dispatch receipt — 7 Nov 03:18Z

**Dispatch blocked before camera collection.** All 20 deposit_coin, 10
play_tic_tac_toe and 10 stack_blocks layouts passed default two-pass presettle.
insert_key primary blocks passed 0/5 and 3/5; its one reserved refill passed
2/5. Thus only 5/10 key pairs are available. Ten key records failed pass 2.
No tolerances were relaxed and no raw fallback was applied. All four cells use
settled inputs, matching the original disc panel; none is a raw-running cell
requiring --raw-check. Reserved unused raw blocks remain uncollected.

The registered single-refill limit therefore blocks the whole dispatch. A
prospective request to the lead to authorize additional fresh key-only refill
blocks is pending; no such block has been generated. No camera outcome was read.
This is a preparation shortfall, not a camera verdict or an additional re-test.

- [Preparation manifest](data/coin-disc-retest-1107/manifest-preparation-blocked.json):
  SHA256 `0290417dee87fd6c76f3b1c4e97265bd898292650409ec8d50b9717724c329be`. Explicitly complete=false / dispatch_blocked=true;
  freezes 70 generated raw records, preparation indices, 45 valid selected layouts,
  resolved configs, code and 2,536 catalog/background/model asset file hashes.
- [Provisional rows](data/coin-disc-retest-1107/rows-preparation-blocked.txt):
  90 rows for available pairs only, **not queued or authorized as a partial panel**.
  Every provisional row passed the explicit Sonnet route check; resolved A/B
  configs differ only in disc_hold_evidence. Full requested panel remains 100 rows.
- [Non-dispatch receipt](data/coin-disc-retest-1107/non-dispatch-receipt.json):
  **0 rows appended; no positions; ETA not scheduled**. Under queue flock at
  2026-11-07T03:18:42Z, the queue contained 117 existing rows.
  SHA256 before and after (unchanged): `c62fcb6894b505d27177bef065f142016efb8f0927547f4a14f4c77a228aa4d3`.
- Preparation/main registration commit: `a4563552`, before any generation.
  Deployment remains original clean `416a2a03`, containing merged #382.
  Salts file SHA256 `df7dd668feff16f9c749fe3ab9e0f806bc6e8cd98e150faf4430c11652e493c3`.
- Actual re-test/model-probe spend: **0 logical calls, 0 tokens**.
  Estimate for eventual complete panel stays **433 logical Claude calls**.
  Historical disc jobs occupied 11,226 seconds / 38 rows (mean 295.4 seconds):
  approximately 8.2 lane-hours for 100 rows, before retries and queue waits.
  Two lanes would need roughly 4.1 elapsed hours for this panel alone.

Raw/loaded records, indices and all preparation logs remain under
`compute4:~/dojo-data/coin-disc-retest-1107/`. Physics preparation is finished,
its locks released, and the existing lanes continue unchanged. Parent prep
processes were stopped only by their own recorded PID while waiting with no
child active; the completed first block was retained without regeneration.

## Lead-authorized prospective key refill amendment — 7 Nov

The lead explicitly authorized this amendment before any camera collection.
Reason: 7/10 primary insert_key layouts failed the unchanged second stability
pass. The original reserved refill yielded only 2/5 stable records, leaving
5/10 key pairs. The prior block-on-shortfall clause is superseded ONLY as follows.

Generate at most TWO additional five-record, fresh salted **key-only** blocks:
`disc-keyrefill2` then `disc-keyrefill3`, with literal salts fixed in the
[amendment salt file](2026-11-07-coin-disc-retest-amendment-salts.json).
Use new private roots and ids 100000–04; identity includes the fresh salt.
Verify never-used identities and raw hashes. Keep the unchanged generator,
default two-pass presettle, and unchanged --raw-check for raw-running cells
(none of this panel's cells runs raw). No stability threshold changes or fallback.
Select stable records ascending by block number then id, based on stability
only. Retain all original valid records; stop at ten key pairs. No outcomes
may be inspected or used for selection. No third additional block is authorized.

After these blocks, dispatch ten key pairs if available; otherwise dispatch
all available stable key pairs if **at least eight**, disclosing reduced guard n.
If fewer than eight, stop and report to the lead without queueing any panel rows.
Full registered n now means this prospectively selected key n (8–10), with
20 gain, 10 tic-tac-toe and 10 stack pairs unchanged. Gain/guard decision
thresholds, Sonnet in both arms, deployment, config parity and every other
condition remain unchanged. Record each block's failure rate, including which
pass failed. If the key second-pass failure rate is much higher than other
cells, record the preparation landmine in BACKLOG. Freeze the selected panel
and manifest on main before appending rows. This amendment and its salts must
also reach main before generating the two new blocks.

## Amended panel freeze — before camera dispatch

The lead amendment reached main at `96965dc2` before the two additional key
blocks were generated. Both are now prepared with unchanged two-pass checks.
The final panel restores **all 10 key pairs**, selecting disc-key2: 100000, 100001, 100003; disc-keyrefill1: 100000, 100001; disc-keyrefill2: 100000, 100001, 100003, 100004; disc-keyrefill3: 100002.
The exact identities and AB/BA order are in the manifest and row file. Other cells retain
their original frozen inputs. No camera outcome or transcript was read.

| block | prepared | stable | failed | failure rate | second-pass failures |
|---|---:|---:|---:|---:|---:|
| disc-gain1 | 5 | 5 | 0 | 0% | 0 |
| disc-gain2 | 5 | 5 | 0 | 0% | 0 |
| disc-gain3 | 5 | 5 | 0 | 0% | 0 |
| disc-gain4 | 5 | 5 | 0 | 0% | 0 |
| disc-gainrefill1 | 0 | — | — | not prepared; reserved raw only | — |
| disc-key1 | 5 | 0 | 5 | 100% | 5 |
| disc-key2 | 5 | 3 | 2 | 40% | 2 |
| disc-keyrefill1 | 5 | 2 | 3 | 60% | 3 |
| disc-ttt1 | 5 | 5 | 0 | 0% | 0 |
| disc-ttt2 | 5 | 5 | 0 | 0% | 0 |
| disc-tttrefill1 | 0 | — | — | not prepared; reserved raw only | — |
| disc-stack1 | 5 | 5 | 0 | 0% | 0 |
| disc-stack2 | 5 | 5 | 0 | 0% | 0 |
| disc-stackrefill1 | 0 | — | — | not prepared; reserved raw only | — |
| disc-keyrefill2 | 5 | 4 | 1 | 20% | 1 |
| disc-keyrefill3 | 5 | 3 | 2 | 40% | 2 |

Every rejection above occurred in the second pass. Key failures total 13/25
(52%); other prepared cells total 0/40 (0%). This is a preparation-selection
landmine, not evidence about controller performance. Added the corresponding
item to [BACKLOG](../BACKLOG.md). All rejected inputs and indices are retained;
no tolerance changes, raw fallbacks or extra blocks were used.

[Final manifest](data/coin-disc-retest-1107/manifest.json) SHA256:
`9eea63ddebf016dcfd71b1c9f21cd7169dad8de14c0868de6a6a64e76153947d`.
[Final 100 rows](data/coin-disc-retest-1107/rows.txt) SHA256:
`90acca0d52de076720b85c6291d2af7d6b493aef4e51bd8a8eb06d5cc5f2951e`.
The manifest is complete, selecting 50 pairs (20 gain, 10 per original guard),
with raw/loaded/index and candidate asset hashes, complete effective A/B
configs and all 80 generated raw records, including unused reserves.
All **100/100** rows explicitly pin `AGENTIC_MODEL=claude/claude-sonnet-5-5`;
only disc_hold_evidence differs between resolved arms. Reused deployment
`416a2a0374e06628350816a019862caf5156e730` remains clean and contains #382.
The previous preparation-only freeze and non-dispatch receipt are retained
as history and superseded by this complete freeze. Projected spend remains
433 logical Claude calls; actual before dispatch: zero calls and zero tokens.
This complete freeze must be on main before any row is appended.

## Dispatch receipt — 2026-11-07T03:29:40Z

Complete freeze was pushed to main at `e7ca8286` before append.
Appended **100 rows at queue positions 116–215**
(1-based positions at append), after all 115 existing queued rows,
under `flock ~/dojo-integ-c4-state/queue2.txt.lock`. Existing bytes were preserved
exactly. No existing row was reordered, rewritten or removed by this task.

- [Receipt with every row and its append position](data/coin-disc-retest-1107/dispatch-receipt.json).
- Manifest SHA256: `9eea63ddebf016dcfd71b1c9f21cd7169dad8de14c0868de6a6a64e76153947d`.
- Row-file SHA256: `90acca0d52de076720b85c6291d2af7d6b493aef4e51bd8a8eb06d5cc5f2951e`.
- Queue SHA256 before: `735f9d6fcbad1d78e262e566115e4afd74d4aae2a2409c41b716a00cd1c24e19`.
- Queue SHA256 after: `4dde2fdcea6a108c8ea4d011e4ee162088299c8cdc22b2bcb13f3fbfc39514bc`.
- Route rechecked on **100/100** rows before append and again after append:
  `AGENTIC_MODEL=claude/claude-sonnet-5-5`, with no Luna/cx route.
- Post-append verification: all 100 frozen rows present in their frozen order;
  heavy1 and heavy2 lane processes alive. No panel row had started.
- Actual spend at dispatch: **0 logical calls, 0 prompt/completion/reasoning/cache
  tokens**. Estimated full-panel spend remains **433 logical Claude calls**.
- Occupancy-only historical estimate: 9.0 lane-hours
  ahead, 9.3 lane-hours for this panel; about
  9.2 elapsed hours on two lanes from append,
  approximately **12:40 UTC on 7 Nov** for completion. Admission waits, active jobs,
  adaptive perception and infra retries can extend this. It is not a deadline.

The panel is dispatched at the full original n: 20 deposit_coin pairs and
10 pairs each of insert_key, play_tic_tac_toe and stack_blocks. This task ends
at dispatch; no verdict, adoption recommendation or camera outcome analysis.

## COIN re-test verdict — 7 Nov: FAIL / NOT ADOPTED

The once-only fresh-layout re-test fails the registered camera gain rule.
The original panel is not pooled into this result. No adoption, candidate/default
change, tuning or further re-test is authorized by this verdict.

| cell | pairs | A pooled | B pooled | Δ | score wins / losses (≥10 pts) | success flips (B / A) | successes A / B |
|---|---:|---:|---:|---:|---:|---:|---:|
| deposit_coin (gain) | 20 | 17.0 | 12.0 | -5.0 | 2 / 4 | 1 / 2 | 2 / 1 |
| play_tic_tac_toe (guard) | 10 | 90.0 | 90.0 | +0.0 | 0 / 0 | 0 / 0 | 8 / 8 |
| stack_blocks (guard) | 10 | 100.0 | 100.0 | +0.0 | 0 / 0 | 0 / 0 | 10 / 10 |
| insert_key (guard) | 10 | 23.5 | 32.0 | +8.5 | 2 / 1 | 2 / 1 | 1 / 2 |

Gain checks: pooled Δ **−5.0 < +5**; score wins **2 < 2 × 4** losses;
net score wins **−2 < +2**. All three guard cells pass: B pooled is at least
A − 5 and success-flip losses do not exceed wins (key 1 loss / 2 wins;
tic-tac-toe and stack 0 / 0). Success flips are supplementary for the gain,
not substituted for its score rule. All 50 registered pairs / 100 arms are
complete, with exactly one terminal episode and rc 0 per arm, no retries,
infra exclusions, cooking failures or time-capped episodes.

### Spread, paired bootstrap and absolute score

| cell | A sample SD | B sample SD | paired mean Δ 95% CI, points |
|---|---:|---:|---:|
| deposit_coin | 29.93 | 22.85 | [-21.0, +11.0] |
| play_tic_tac_toe | 21.08 | 21.08 | [+0.0, +0.0] |
| stack_blocks | 0.00 | 0.00 | [+0.0, +0.0] |
| insert_key | 26.88 | 35.84 | [-17.0, +34.0] |

CIs are paired percentile bootstrap intervals: 10,000 resamples of the salted
layout pairs within each cell, NumPy default_rng(0), B − A mean score in points.
One repeat per identity; equal numeric ids across salts remain distinct pairs.
The deposit_coin CI **[−21, +11]** includes zero. The CI is descriptive; it adds
no unregistered significance criterion and does not replace the threshold rule.

B's absolute deposit_coin mean is **12.0** (1/20 successes), versus the lead's
D-C19 score **8**: **+4.0 points**; versus PyRUA-RSI **79.6**: **−67.6 points**.
These are cross-panel benchmark comparisons, not paired improvements: this
fresh salted panel differs from those benchmark evaluation populations.
Its matched A baseline is 17.0, so the lever's actual paired estimate is −5.0.

### Registered write audit and supplementary #417 instrument

The registration reached main at a4563552 before #417. Its frozen
EXPERIMENTS.md defined the tracked object as the nearest pre-write GT object
in the label group, and required **no excess raw harmful writes per writer**.
Therefore the primary instrument is the frozen `416a2a03` scorer, not the later
persistent-identity instrument or its newer rate-based gate. A harmful write
adds **>2 mm** of unrounded 3-D body-centre error; helpful removes >2 mm;
z is reported separately. Both arms and all registered cells are included.
The primary gate is not retroactively replaced by a new-writer or rate rule.

Saved numeric action trajectories were replayed physics-only on compute4 using
the original frozen deployment and settled inputs. Normalized audit reports
union the unmodified extractor's explicit writes with its legacy before/after
writes (deduplicated by step, label and poses rounded to four decimals), while
preserving original consume/probe identity evidence. Original reports, actions,
call logs and layouts were not modified. Replays request every write step
exactly, plus the standard grasp/gripper snapshots and terminal state; audit
uses --back 0 / --harm-mm 2. No model is queried.

Primary coverage: **582/582 writes**, 100/100 trajectories, zero uncertain writes and zero frame lag. All 100 replay starts were stable. Terminal-score matches: **100/100**; grasp-close probe agreement within 2 mm: **262/262**, worst deviation **0.0 mm**.

| writer | writes A / B | harmful A / B | helpful A / B | neutral A / B | uncertain A / B |
|---|---:|---:|---:|---:|---:|
| perception | 112 / 111 | 31 / 34 | 51 / 50 | 30 / 27 | 0 / 0 |
| post_place | 5 / 5 | 1 / 0 | 3 / 3 | 1 / 2 | 0 / 0 |
| predicted | 82 / 88 | 44 / 46 | 37 / 40 | 1 / 2 | 0 / 0 |
| refine_look | 6 / 8 | 0 / 0 | 3 / 4 | 3 / 4 | 0 / 0 |
| refine_look+confirm | 1 / 2 | 0 / 0 | 1 / 2 | 0 / 0 | 0 / 0 |
| refine_pick | 73 / 81 | 1 / 3 | 3 / 7 | 69 / 71 | 0 / 0 |
| refine_pick+confirm | 4 / 2 | 0 / 0 | 4 / 2 | 0 / 0 | 0 / 0 |
| refine_place | 0 / 2 | 0 / 0 | 0 / 2 | 0 / 0 | 0 / 0 |

Registered write gate: **FAIL**. Raw harmful counts exceed A for `perception` (31 → 34), `predicted` (44 → 46), `refine_pick` (1 → 3). Helpful writes cannot offset these excesses under the registered gate. On deposit_coin alone, harmful `perception` writes rise 2 → 4 and `predicted` writes 0 → 4; the failure is also present in the gain cell.

Mean absolute z below uses the scorer’s rounded 0.1-mm z rows; the harmful gate uses unrounded 3-D error changes.

| writer | A mean absolute z before → after, mm | B mean absolute z before → after, mm |
|---|---:|---:|
| perception | 0.97 → 4.72 | 1.17 → 3.80 |
| post_place | 1.02 → 1.02 | 1.52 → 1.52 |
| predicted | 9.28 → 23.73 | 11.52 → 26.00 |
| refine_look | 0.00 → 0.00 | 0.00 → 0.00 |
| refine_look+confirm | 0.00 → 0.00 | 0.00 → 0.00 |
| refine_pick | 0.07 → 0.07 | 0.22 → 0.22 |
| refine_pick+confirm | 1.25 → 1.25 | 2.50 → 2.50 |
| refine_place | — | 0.00 → 0.00 |

Supplementary #417 re-score uses exact merge `77b45719`, the same frozen write
ledger and replay frames, and persistent first-pick identity associations.
There are no held_pose writes, so no legacy show-prior transport reconstruction
is needed. This instrument is supplementary and cannot change the registered
verdict. Its raw harmful counts are reported on the same basis below.

| writer | harmful A / B | helpful A / B | neutral A / B |
|---|---:|---:|---:|
| perception | 12 / 14 | 70 / 70 | 30 / 27 |
| post_place | 1 / 0 | 3 / 3 | 1 / 2 |
| predicted | 4 / 7 | 78 / 80 | 0 / 1 |
| refine_look | 0 / 0 | 3 / 4 | 3 / 4 |
| refine_look+confirm | 0 / 0 | 1 / 2 | 0 / 0 |
| refine_pick | 1 / 4 | 3 / 6 | 69 / 71 |
| refine_pick+confirm | 1 / 0 | 3 / 2 | 0 / 0 |
| refine_place | 0 / 0 | 0 / 2 | 0 / 0 |

The supplementary raw-count comparison also has excesses: `perception` 12 → 14, `predicted` 4 → 7 and `refine_pick` 1 → 4. This does not replace the primary instrument or apply the newer rate gate.

Supplementary coverage: **582/582 writes**; uncertain **0**; nearest fallbacks **55/582**. Body-geometry cache has zero missing assets. Consumed-pick evidence lacks an exact replay frame at 41/179 steps; #417 uses other available tight/grasp evidence or its explicitly reported nearest fallback. These sparse identity-anchor limitations are not silently treated as exact first-pick associations.

[Audit summary](data/coin-disc-verdict-1107/audit-summary.json) contains per-cell
and per-writer counts, rates, z, fidelity and replay hashes. Full frozen ledgers:
[registered](data/coin-disc-verdict-1107/writes-registered.jsonl) and
[#417 supplementary](data/coin-disc-verdict-1107/writes-fixed417.jsonl).
[Body bounds/group cache](data/coin-disc-verdict-1107/body-geometry.json) preserves
the public asset-derived body-centre metric. Raw sources, normalized inputs,
replays and reproduction scripts remain in
`compute4:~/dojo-data/coin-disc-verdict-1107/`. No sealed data were read.

### Actual route proof and spend

All 100 arms' actual call logs were checked, independently of the queued route:
**164 model-labelled records all name claude/claude-sonnet-5-5**. The other
21 unlabelled records have zero logical calls (local held-evidence paths),
so there is no unlabelled positive-call route, Luna or cx request in this panel.
The report's aggregate logical-call and available token counters reconcile with
the call logs on every arm. Source SHA256s and pair-level results are retained in
[the camera snapshot](data/coin-disc-verdict-1107/verdict-camera.json).

| arm | logical Claude calls | recorded prompt | recorded completion | recorded reasoning | recorded cache-read |
|---|---:|---:|---:|---:|---:|
| A | 214 | 3,052,629 | 136,411 | 72,481 | 2,425,768 |
| B | 214 | 3,045,704 | 139,825 | 76,837 | 2,428,810 |
| total | 428 | 6,098,333 | 276,236 | 149,318 | 4,854,578 |

**Token coverage caveat:** 28 positive-call held_look records (28 logical calls)
have tokens=null (A 13 calls, B 15); their usage is unavailable. Therefore the token columns are
**recorded lower bounds**, not complete billing totals or zeros for missing
usage. All 428 logical calls are counted; token usage covers 400. Another 21
zero-call records also lack token fields. Seventeen surfaced error entries
remain included in the logs and accounting; successful hidden HTTP retries
cannot be separately enumerated. Reasoning is included in completion and
cache-read in prompt; do not add these subset columns again. The pre-dispatch
estimate was 433 logical calls. Verdict/physics audits made **zero new LLM calls**. The missing held-look usage is recorded as a landmine in [BACKLOG](../BACKLOG.md).

**Final disposition:** FAIL / NOT ADOPTED. The once-only re-test is exhausted.
No candidate field changes: disc_hold_evidence remains off; all other fields
remain unchanged. No automatic adoption or new dispatch follows.
