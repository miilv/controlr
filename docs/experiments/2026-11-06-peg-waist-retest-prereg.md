# Compute3 peg and waist once-only camera re-tests

Prospective registration: no re-test generation or camera data exist at commit.
The initial tests had positive pooled deltas and one net win each; see
[wave verdicts](../journal/2026-11-06-ab-wave-verdicts.md). This consumes each
lever's ONE permitted fresh-layout re-test; no further re-test is authorized.

## Arms and freeze

Both arms use main `86ad1987` in `compute3:~/dojo-integ-c3-86ad1987`,
family prefix `c3b_`. Baseline is the exact captured candidate in
`scripts/dojo/queues/2026-11-06-candidate.txt`, SHA256 `5cde3f9e684c0937f21a0f28609437e6d317a433e000bc171198349417cd398e`,
with only host paths relocated. Peg B changes only `peg_held_read=on`;
waist B changes only `belt_waist_first=on`. A keeps both off.
All arms use `submit_v3`, Isaac 5.1 camera-only belief, RuleSupervisor,
`AGENTIC_MODEL=claude/claude-sonnet-5-5`, and the candidate's perception,
seg v2b, digit_v2, kong_v1b, grip v2 and frozen calibration settings.
This explicitly supersedes prior cx-route overrides, identically in A/B.
No separate implementation-agent Claude calls. Stop dispatch on HTTP 429.
Freeze complete effective configs, queue, catalog/background/calibration/model,
raw/loaded/index hashes and package/version differences before dispatch.

## Sample and fresh identities

| lever | cell | role | pairs |
|---|---|---|---:|
| peg | play_stacking_toy | gain | 20 |
| peg | stack_blocks | guard | 10 |
| peg | insert_tubes | guard | 10 |
| waist | match_and_pick_from_conveyor | gain | 20 |
| waist | pick_from_conveyor_by_image | guard | 10 |

One repeat per identity. Four five-record blocks per gain; two per guard.
All ids are 100000–04 in independent private block roots under
`~/dojo-data/integ3-1106/blocks`. Never pool equal numeric ids across salts.
The [literal salts and SHA256s](2026-11-06-peg-waist-retest-salts.json) are frozen
before generation; file SHA256 `d468aea30f7d0917a941b78218a90c75b896d37fbf45dc874f0a14264cf5b0a7`. Fresh salts distinguish these layouts
from all original tests. Check the historical identity registry before dispatch.
Use unchanged generator and two-pass presettle for peg, RAW plus unchanged
`--raw-check` for conveyors. No tolerance relaxations. Keep all attempts/indices.
For each cell exactly ONE five-record refill block is reserved in the salt file.
Only if preparation drops records, select valid refill ids ascending to restore n;
record exclusions and selection before collection. If still short, block that
family; do not lower n or substitute after outcomes.
Alternate AB/BA by sequential pair ordinal, keeping both arms on compute3.
No tuning, optional stopping or selection based on camera outcomes.
Infrastructure failures permit one identical-input retry per arm via lane2;
missing terminal pairs block adoption. Report every attempt separately.

## Default decision rule

For each lever independently: its gain B pooled score must be at least A + 5,
success flips wins >= 2 * losses, and wins - losses >= 2. Each guard must have
B pooled >= A - 5 and no more flip losses than wins. Full registered n must
complete. Positive delta without the flip bar is rejection, not adoption.
Report per-cell mean, sample SD, successes, paired flips, all infra exclusions,
call/token totals, occupancy and harmful belief-write audit. Perception changes
also require EXPERIMENTS.md's physics replay write audit before adoption.

## Lane and budget

Exactly one camera/Isaac job of ours, using the existing heavy lock. Before
each GPU job record `who` and nvidia-smi; admit only at <= 14336 MiB used,
with at least 16 GiB free. Log power.draw every five minutes. No system changes
or touching colleagues' processes, compute4, dc19 or coord. Private offline Kit
extscache and 600-second early/hard watchdog follow #402; retain 1980-second
job cap and 25-minute episode cap. Package patch differences are recorded for
this paired same-box comparison, never silently described as exact parity.
140 re-test episodes; historical occupancy predicts about 12 lane-hours before
prep/retries/waits. Perception calls are adaptive; record actual logged usage.
No automatic controller adoption follows from this registration.

## MOVE34 pre-collection host amendment

Box moved compute3 → compute4 before any valid episode; reason: Isaac startup
crashes on compute3. Both arms now use `~/dojo-integ-c4-86ad1987`, exact source
`86ad1987ac4750dad187a45bf7add1dbcd25b98a`, with `c4b_` names and inputs under
`~/dojo-data/wave-1106-c3moved/`. Frozen raw, loaded and index bytes, salts,
configs, pair identities and remaining arm order are preserved. No layouts are
regenerated and no outcomes are used to select this move.

INTEG3's five failed startup attempts produced zero valid episodes or model
transcripts: no valid episode or fresh-layout identity was consumed. The
infrastructure-attempt history remains recorded. This amendment authorizes
moving the **158 remaining rows exactly**, retaining the existing retry suffix;
it does not silently refill the absent PEN A/B 100020 rows or reset retries.
The frozen manifest remains 80 pairs, while the dispatched remainder covers
79 pairs. PEN cannot pass the full-panel rule with 100020 missing; restoring
that pair would require an explicit prospective authorization.

The compute4 lane's existing policy supersedes compute3-specific operational
clauses: one lane, <=16384 MiB used admission, the existing
`~/coord/lane_gate.stamp` protocol and 50 GB free-disk floor. Keep compute4's
working renderer/startup recipe and current jobs unchanged; do not carry over
compute3's offline Kit/watchdog environment. Append after all current wave
reloc/key rows. Both arms retain the frozen common perception model; MOVE34
makes no separate Claude-route calls. Existing decision rules and sample
requirements remain in force. Compute3 remains under STOP, with its queue
archived as moved to compute4.
## Peg re-test camera verdict (7 November)

The final peg arm, `c4b_insert_tubes_pegAr1_pegtubes2_100004`, reported
`done rc=0` on heavy1 at **06:03:17Z**. This verdict was computed only after that
barrier. Waist is excluded. This is the once-only fresh-layout re-test, not
an additional selection round.

**Camera decision: PASS under this registration's success-flip rule.**
The 7 November score-based refinement does not apply retrospectively.
Gain +8.0 >= +5; success-flip wins 5 >= 2×2 losses; net wins 3 >= 2.
Both guards meet B >= A−5 and flip losses <=wins. This is a camera-gate
result; the separate required write-audit result is below.

| cell | pairs | A mean ± sample SD | B mean ± sample SD | Δ points | success flips B / A | supplementary score wins / losses (>=10 points) | successes A / B |
|---|---:|---:|---:|---:|---:|---:|---:|
| play_stacking_toy (gain) | 20 | 55.5 ± 23.73 | 63.5 ± 28.52 | +8.0 | 5 / 2 | 7 / 4 | 3 / 6 |
| stack_blocks (guard) | 10 | 100.0 ± 0.00 | 100.0 ± 0.00 | 0.0 | 0 / 0 | 0 / 0 | 10 / 10 |
| insert_tubes (guard) | 10 | 82.0 ± 28.98 | 82.0 ± 28.98 | 0.0 | 0 / 0 | 0 / 0 | 7 / 7 |

Scores and SDs use the 0–100 point scale; SD uses n−1. Paired bootstrap 95%
percentile CI for the gain is **[−6.0, +22.5] points** (10,000 resamples,
NumPy default_rng seed 0, sampling 20 paired salted layouts with replacement).
The CI includes zero; the registered decision is a threshold rule, not a
claim of a statistically resolved positive population effect. The supplementary
7/4 score count would not meet the newer ratio; it does not change this rule.

### Integrity, infrastructure and actual route

All 80 registered arms have exactly one terminal episode and exactly one
`done rc=0` entry, across heavy1/heavy2. **Zero peg infrastructure failures,
zero arm retries, zero excluded peg attempts, zero time caps.** No missing
pairs or refill substitutions. The five historical compute3 startup failures
were PEN, not peg; they contribute no peg episodes or spend. The lead later
increased compute4 camera caps to two lanes; both arms remain on compute4
with the same frozen source and inputs.

Each report's full resolved RuntimeConfig equals its frozen config at
`86ad1987`; all loaded-layout SHA256s match the frozen salted manifest. The
renderer/mode are Isaac 5.1/perception. Every analysed arm verifies
`claude/claude-sonnet-5-5` in **both its actual `agentic/**/calls.jsonl` records
and report `agentic[].model` metadata**, with exact per-row call/token agreement.
Queue text and static deployment model labels are not the route evidence.
No route violations or missing route proofs were found.

There are 19 surfaced agentic parsing errors (16 “turn 0: no JSON object”,
3 “turn 1: no JSON object”); these are included in logical call/token totals,
not classified as infrastructure episode retries. Recovered HTTP transport
attempts are unobservable in this logger, so zero infrastructure retries does
not certify zero transient HTTP retries. Summed camera lane-lock occupancy:
22,509 s (6.25 lane-hours); episode wall time: 17,866.1 s (4.96 hours), excluding
startup/admission waits.

### Recorded spend

| arm | logical Claude calls | prompt tokens | completion tokens | reasoning tokens | cache-read tokens |
|---|---:|---:|---:|---:|---:|
| A | 164 | 2,477,462 | 152,126 | 92,698 | 1,963,901 |
| B | 167 | 2,544,724 | 161,443 | 95,957 | 2,025,813 |
| total | **331** | **5,022,186** | **313,569** | **188,655** | **3,989,714** |

Reasoning is reported separately within completion accounting, and cache reads
within prompt accounting; do not add them again. The verdict/audit makes no
new LLM calls. Per-pair scores, route proofs, attempt ledger, source hashes,
coverage and spend are retained in
[data/peg-retest/verdict-camera.json](data/peg-retest/verdict-camera.json) and
`compute4:~/dojo-data/peg-retest-1107/`.

### Required physics write audit: FAIL; overall not adoptable

The camera gate passes, but the independent no-excess-harmful-write gate fails.
No candidate or default is changed. The sole hypothetical camera-only candidate
change would be `--config.peg_held_read: off → on`; the failed write gate blocks
promoting it. The once-only fresh-layout re-test is exhausted; this result
authorizes no further run.

All **80/80 action trajectories** were replayed physics-only on compute4 from
the exact frozen settled inputs and recorded actions, using source `86ad1987`.
Replays cover every logged write step exactly, not a nearest sparse frame.
`m95_score.py audit --back 0` uses the registered **>2 mm added 3-D body-centre
error** threshold and nearest pre-write tracked object in the label group.
No tolerance or harmful-write threshold was changed.

The standard extractor suppresses legacy refine/release events whenever any
explicit `belief_write` exists. For coverage, separate analysis reports union
its explicit-write output with its legacy-event output (the latter excludes
explicit `belief_write` events); duplicates use step, label and before/after
poses rounded to four decimals. Both original reports and action files remain
unchanged. These normalized reports are inputs to the unmodified audit scorer.
This includes held reattachments, perception updates, accepted refines,
release/post-place updates and predicted writes; rejected read attempts are
not pose writes.

Coverage: **1,696/1,696 logged writes**, A 757 / B 939; zero missing write-step frames, zero uncertain writes, and zero frame lag. Terminal score fidelity: **80/80**. The live grasp-close GT probe agrees within 2 mm on **532/532** closes (worst reported difference 0.0 mm). These fidelity checks accompany the write gate; they are not a replacement camera result.

| writer | writes A / B | harmful A / B | helpful A / B | neutral A / B | uncertain A / B |
|---|---:|---:|---:|---:|---:|
| held_pose | 0 / 147 | 0 / 22 | 0 / 11 | 0 / 114 | 0 / 0 |
| perception | 252 / 266 | 118 / 125 | 80 / 87 | 54 / 54 | 0 / 0 |
| post_place | 104 / 116 | 15 / 8 | 22 / 25 | 67 / 83 | 0 / 0 |
| predicted | 145 / 138 | 70 / 64 | 74 / 72 | 1 / 2 | 0 / 0 |
| refine_held | 9 / 9 | 2 / 0 | 0 / 0 | 7 / 9 | 0 / 0 |
| refine_pick | 238 / 250 | 4 / 4 | 21 / 17 | 213 / 229 | 0 / 0 |
| refine_pick+confirm | 1 / 4 | 0 / 0 | 1 / 4 | 0 / 0 | 0 / 0 |
| refine_place | 8 / 9 | 0 / 0 | 0 / 0 | 8 / 9 | 0 / 0 |

In particular, **held-pose harmful writes increase from 0 to 22**. This excess is present in explicit held-pose writes and does not depend on the supplemental legacy-event extraction. Helpful writes cannot offset it under the registered gate. This is an aggregate gate audit, not a tuning diagnosis.

The z component is reported separately below as mean absolute z error before → after (mm), using the scorer's reported 0.1-mm z values. The decision above uses the unrounded 3-D error change, not z alone.

| writer | A mean absolute z before → after, mm | B mean absolute z before → after, mm |
|---|---:|---:|
| held_pose | — | 13.66 → 25.18 |
| perception | 3.27 → 11.33 | 3.52 → 11.52 |
| post_place | 8.14 → 6.89 | 6.48 → 5.80 |
| predicted | 11.23 → 24.21 | 10.95 → 24.34 |
| refine_held | 2.27 → 2.14 | 2.87 → 2.78 |
| refine_pick | 2.74 → 2.62 | 2.53 → 2.56 |
| refine_pick+confirm | 0.00 → 0.00 | 27.80 → 1.20 |
| refine_place | 0.00 → 0.00 | 0.00 → 0.00 |

Summed replay execution time: 5,420.9 s, excluding startup/admission waits; no LLM calls. Replays used existing physics light slots with bounded scopes and the shared startup/VRAM/disk constraints; camera jobs were not changed. Only verified idle audit launchers were restarted to improve admission polling.

[Audit coverage, per-writer z metrics, fidelity and replay hashes](data/peg-retest/audit-summary.json) are retained with the camera snapshot. Full per-write records, normalized audit inputs, replay jobs and analysis scripts remain in `compute4:~/dojo-data/peg-retest-1107/`. Raw source hashes were rechecked after the audit. Waist outcomes are not part of this report.

## Waist re-test camera verdict (7 November)

**FAIL / NOT ADOPTED.** All 60 waist arms had completed before analysis; the
last `done rc=0` is **09:16:08Z**. The registered success-flip rule is primary:
gain **−15 < +5**, wins **0 < 2×3** losses, net **−3 < +2**. The guard meets
both registered bounds. This was the once-only fresh-layout re-test; it is
exhausted. No candidate, preset or default changes are made, and no further run
is authorized by this verdict.

| cell | pairs | A mean ± sample SD | B mean ± sample SD | Δ points | success flips B / A | supplementary score wins / losses (>=10 points) | successes A / B |
|---|---:|---:|---:|---:|---:|---:|---:|
| match_and_pick_from_conveyor (gain) | 20 | 50.0 ± 51.30 | 35.0 ± 48.94 | −15.0 | 0 / 3 | 0 / 3 | 10 / 7 |
| pick_from_conveyor_by_image (guard) | 10 | 0.0 ± 0.00 | 0.0 ± 0.00 | 0.0 | 0 / 0 | 0 / 0 | 0 / 0 |

The gain's paired bootstrap 95% percentile CI is **[−30.0, 0.0] points**:
10,000 resamples, NumPy default_rng seed 0, resampling the 20 paired salted
layouts. Scores/SDs use 0–100 points; SD uses n−1. Score wins/losses are
supplementary and do not replace this preregistration's success-flip gate.
The guard is camera-gated under its original rule; no later parity-guard
exemption is applied.

### Actual route, full infrastructure accounting and frozen inputs

Every arm has exactly one START, one terminal episode and one `done rc=0`
entry across heavy1/heavy2: **60/60**, with complete 20/10 paired panels.
**Zero waist infrastructure failures, retries, exclusions or time caps**; no
missing pairs, post-outcome refills or rerun directories. The five old compute3
startup failures were PEN and contribute no waist episodes/calls.

Actual route is **claude/claude-sonnet-5-5 in all 60 arms**, verified in each
arm's saved `agentic/**/calls.jsonl` and report `agentic[].model`, not queue
text or static deployment metadata. All per-row logical-call/token totals
agree between both sources. All 60 call bundles contain every requested usage
field. The six surfaced parser errors (two turn-0, three turn-1, one turn-2
“no JSON object”) are included in spend; they are not episode retries. Recovered
HTTP transport retries and any unreported usage are not observable in this
logger, so no zero-transient-retry claim is made.

All actual resolved RuntimeConfigs equal the frozen configs at `86ad1987`;
each pair differs in **only `belt_waist_first`**, A off / B on. Loaded RAW
records match the salted manifest's SHA256s; RAW filenames and raw indices
remain the registered inputs, not substituted settled records. All rows are
`isaac51` / perception. All 90 RAW/loaded/index references (36 distinct files), 270 evidence-file
hashes and 1,317 deployment source-file hashes were verified unchanged. Camera lane-lock occupancy sums to
**13,813 s (3.84 lane-hours)**; episode wall time is **10,395.5 s (2.89 hours)**,
excluding startup/admission waits.

### Write-audit applicability: motion/intercept-only, not applicable

The prereg requires physics harmful-write audits for perception changes. Under
the lead's explicit applicability instruction for this verdict, waist-first is
a **motion/intercept grasp-selection change with no new or changed belief
writer**, so that audit does not apply. This is supported by the frozen code,
all resolved configs and the full write logs, rather than inferred from scores:

- At source `86ad1987`, runtime references only declare, validate and forward
  the flag to Skills (`policy.py:474,680,1014`). Its only execution reads are
  in `interactive.plan_intercept` (`interactive.py:880,909`): prefer existing
  mesh-waist grasp candidates before surface/rim candidates, and filter skipped
  grasp methods. The same waist helper was already the A fallback. This adds
  neither a perception request nor a belief-write/acceptance predicate. The
  collision-search helper temporarily projects obstacles and restores them in
  `finally`; it does not commit a belief relocation.
- There are **zero flag references in the frozen perception modules**. All
  read/writer settings match, including `belt_wrist_veto=on`, `track_moving=auto`,
  `track_read=belt`, `tracker=auto`, `track_expire=off`, `refine=true`,
  `held_pose=request`, `watch_relocate=false`, and the same held/refine/scan
  gates and numeric thresholds.
- The observed writer families are identical by cell: gain `perception`,
  `tracked`, `refine_pick`; guard `perception`, `tracked`. Both arms have zero
  `held_pose` and zero `watch_relocate` events. Perception operation types also
  match by cell. Existing write/read counts vary with trajectories; this proof
  does **not** claim identical beliefs, identical observations or zero harmful
  writes from the unchanged baseline writers.

| cell / existing writer | logged writes A | logged writes B |
|---|---:|---:|
| gain / perception | 8 | 10 |
| gain / tracked | 3,127 | 3,413 |
| gain / refine_pick | 12 | 9 |
| guard / perception | 9 | 4 |
| guard / tracked | 91 | 92 |
| total | 3,247 | 3,528 |

Counts cover explicit write records plus accepted legacy refine events, using
the frozen extractor and de-duplicating step/label/before/after poses to four
decimals, as for peg. They are operational write counts, **not harmful-write
classifications**. Gain `belt_read` operations A/B are 1,431/1,535 and refine
operations 38/27; guard `belt_read` is 70/70. These differences are disclosed;
no writer implementation, accepted read type or acceptance threshold changes.

| instrument | role under this registration | waist result |
|---|---|---|
| frozen nearest-in-label-group identity, raw no-excess-per-writer gate | primary if a perception audit applies | not applicable to this motion-only change; no GT harmful-write score claimed |
| #417 fixed persistent identity | supplementary only for this older registration | not applicable; no supplementary GT re-score run |

No later harmful-rate/new-writer gate or fixed-identity result is substituted
for the preregistered instrument. No physics replay, camera job, model call or
lane/queue change was needed for this applicability review. Source references,
function hashes, per-row config proof and per-cell write/read logs are retained
in [the applicability snapshot](data/waist-retest/audit-applicability.json).
The camera rejection stands independently of this applicability classification.

### Recorded spend

| arm | logical Claude calls | prompt tokens | completion tokens | reasoning tokens | cache-read tokens |
|---|---:|---:|---:|---:|---:|
| A | 73 | 995,315 | 37,052 | 20,831 | 825,615 |
| B | 72 | 963,964 | 38,003 | 21,881 | 808,057 |
| total | **145** | **1,959,279** | **75,055** | **42,712** | **1,633,672** |

Reasoning is reported separately within completion accounting, and cache-read
within prompt accounting; do not add either twice. No new LLM calls were made
for this verdict. Per-pair scores, per-row actual route proof, attempt ledger,
source hashes, configs and spend are retained in
[the camera snapshot](data/waist-retest/verdict-camera.json) and
`compute4:~/dojo-data/waist-retest-1107/`. No adoption, candidate change, default
change or additional fresh-layout re-test follows from this report.
