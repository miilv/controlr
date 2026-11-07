# 2026-11-07 — KONG scoped, one fresh-layout re-test

**Hypothesis:** the unchanged make_kong-only relocation lever improves the
camera gain cell under the prospectively refined score, parity and per-writer
write-audit gates. This is its **one** fresh-layout re-test, authorized by the
lead after the [first scoped verdict](2026-11-07-kong-scoped-relocate-prereg.md).
That earlier verdict keeps its original rules and is not re-judged.

## Frozen treatment, deployment and route

The [machine-readable design](data/kong-scoped-retest/design.json) is committed
with this document before generation or queue insertion. Both arms use exactly
#410/#411, deployment `/home/robotics/dojo-integ-c6-kong-ffb2c151`, merged code
`ffb2c151f1d48f5bc68f67b10ab4430b0fd4685e`. Candidate
`scripts/dojo/queues/2026-11-06-candidate.txt`, SHA256
`5cde3f9e684c0937f21a0f28609437e6d317a433e000bc171198349417cd398e`;
box-local calibration paths map `/home/haptics/` to `/home/robotics/` only.

A: `watch_relocate=false, watch_relocate_task=""`.
B: exactly A with `watch_relocate_task="make_kong"`.
Both retain `watch=true`, perception mode, submit_v3, grip v2, v2b segmentation,
digit_v2 / kong_v1b glyphs, the same color calibration/backgrounds, perception
JSON, service recipe, managed venv and native simulator. No other lever,
threshold, budget, default, candidate or motion policy changes.
Every queue row explicitly sets **`AGENTIC_MODEL=claude/claude-sonnet-5-5`**.
Actual route must later be verified per row from call logs/transcript starts
and report agentic metadata, never queue text or stale `code.model` alone.
A route violation prevents adoption; it does not become another model panel.

| Cell | Pairs | Role | Camera gate |
|---|---|---|---|
| make_kong | 20 | gain | registered score rule |
| imitate_sorting_sequence | 5 | parity guard | post-collection parity proof; score descriptive |
| stack_blocks | 5 | strong guard | ordinary score and success-flip guard |

One repeat per arm per identity: **30 pairs / 60 camera rows**. Stack is the
strong guard (the first Sonnet scoped panel was 100/100); fresh layouts are
not selected using their camera success. Task order is gain, imitation, stack.
Pair jobs are adjacent; AB when `(id−100050)` is even, BA when odd. Independent
fresh environments, identical frozen layout/demo per pair. Keep existing
25-minute episode/job caps and the existing lane wrapper/watchdogs.

## Freshness, preparation and selection (no sealed data)

One new salt, frozen before any generation:

`kong-scoped-retest-20261107-53a0ae76903fe9ae1cb24eac61cc45fe74f086fc02819f7795d535629fce55c2`

SHA256 of its literal UTF-8 bytes, no newline:
`e3ffabe8f9b9b5a0e909b26c358e9eb82b6dc03ab1d1f78c9f139ea371ccc071`.

Use private `compute6:~/dojo-data/kong-retest-1107/kong-scoped-retest/`.
Candidate ids are **100050–100099**, all below 100100, per task. These are
new salted dev identities; never use old KONG/reloc salts, records, demos,
measurement inputs or sealed data. A score-blind metadata/history collision
check precedes generation; a collision blocks prep and requires amendment.
Identity is (task, salt, id, raw hash, loaded-record hash), not numeric id alone.

Use the same pinned `dojo.simlab.gen_layouts --seed-salt` and unmodified
presettle tools. Generate the first 20 gain / five guard ids ascending;
if settling drops a record, generate the next consecutive id and continue
until the first 20/5/5 valid records have been selected, or 100099 is exhausted.
Selection uses **stability only**, before camera collection. Retain every raw
record, dropped attempt and validation index. No score, appearance, semantic
layout difficulty or later policy outcome may influence selection.

All three tasks run settled records: default two-pass physics-only presettle
is the selection check (250 extra steps, 100-step rest window, 1 mm/1 degree
rest test, default 5 mm reload drift limit, no extra passes or tolerance changes).
The existing `--raw-check` is used only for tasks running raw records; this
panel has none. Keep the RAW originals and settled provenance for all draws.

For the first five stable imitation layouts, use the existing
`demo_gen --speed 1.25`, normal validation and its normal seeded order tries,
as in the previous scoped panel. Both arms use the identical frozen demo.
A demo failure blocks launch; do not replace an otherwise stable selected
layout based on demo outcome, escalate speed, reduce n or use a stub. Exhausted
candidate ids, missing records or invalid demos require a pre-data amendment.

Prep uses a free compute6 light lock, VRAM used ≤20 GiB and load ≤26 at
admission, at most two own jobs (one serial prep job intended), 20G memory /
600% CPU scope and ≥50 GB disk floor. Existing lane jobs/queue are untouched.
Freeze raw/settled/demo and index SHA256s, generator/catalogue/calibration/
checkpoint/background hashes, packages, deployment, effective A/B configs,
script provenance and expanded pair order into a manifest before dispatch.

## Registered gates (43c1801d, prospective after 07:00 UTC)

Apply [EXPERIMENTS.md §3](../EXPERIMENTS.md), including the lead's
[prospective rationale](../journal/2026-11-06-ab-wave-verdicts.md).

- **Gain:** complete 20 pairs. B−A pooled mean ≥5 points on 0–100 scores.
  A pair score win is B−A ≥10, loss ≤−10; smaller differences tie.
  Require wins ≥2×losses and wins−losses ≥2. Report success flips alongside.
- **Imitation parity guard:** all five pairs must complete. After collection,
  prove identical **task-resolved effective configuration** and config-resolved
  Skills parameters in both arms, plus **zero relocation actions and zero
  relocation writes in every episode**, using the episode/service logs. Resolve
  the raw task selector into `watch_relocate_for(instruction_routed_task)` before
  comparing effective configs: the raw selector alone differs by treatment,
  while its effective value must be false for imitation. All other fields must
  match their frozen arm and each other. Freeze constructor/default/override
  skill-parameter byte snapshots from the pinned code, then reconstruct and
  check them from every observed report config after collection, including
  rebuilt Skills. Check routed vocabulary and watcher code-path exclusion.
  A mismatch or missing proof fails parity. Report imitation scores/spread,
  flips, time caps and all writes, **without camera-score or write-rate gating**.
- **Strong stack guard:** complete five pairs, B−A pooled mean ≥−5 points,
  success-flip losses ≤wins. Report score wins/losses and relocation exclusion.
  Its camera gate remains even though relocation is structurally inactive.
- **Write audit:** full matched physics replays of all 60 arms, exact write-step
  snapshots, all logged belief writers covered, missing/uncertain coverage
  prevents adoption. Threshold is unchanged: >2 mm added/removed 3-D error;
  report z separately. Judge the refined writer gates on **make_kong only**,
  pooled across its 20 pairs, where this lever acts. Guard writes are reported.
  For each writer with writes in A, require B harmful/writes ≤A harmful/writes
  +0.05 (five percentage points). Report raw counts, helpful and neutral rates
  too; raw excess is not the gate. If a writer has no writes in B, report its
  rate as undefined with zero harmful writes, rather than adding pseudocounts.
  For a writer new in B (zero A writes), require helpful ≥2×harmful. A writer
  absent from both arms is reported inactive. No helpful/harmful trade-off is
  applied to the existing-writer rate check.

All gates must pass; the camera gain alone does not authorize adoption.
No optional stopping, score-selected exclusions or sample expansion. Policy
failures, order violations, time caps and exhausted budgets remain outcomes.
One classified infrastructure-only retry is allowed with identical inputs;
retain both attempts and its reason. Missing planned camera pairs block adoption.
Report paired bootstrap 95% CIs (10,000 paired-layout resamples, RNG seed 0),
means/sample SD, individual paired deltas, successes/flips, steps and occupancy.
The CI is descriptive, with no extra gate. No adoption/default change is
part of this dispatch; a verdict is a later lead request.

## Registered write-audit instrument (lead addition, before data)

The lead identified stale pre-sync `before` poses on held-pose writes and
nearest-pre-write-GT association swaps on multi-instance labels, including
KONG's two prior relocation flags. The fix is being prepared on
**`fix/audit-identity`**: persistent identity association and the stale-before
fix, with **the 2 mm threshold and 3-D/z metric unchanged**.

The primary audit uses that PR's identity-association instrument, **pinned to
its merge commit**, provided it is merged before the first re-test row starts.
If it is not merged by that start, use the current instrument pinned to
`ffb2c151f1d48f5bc68f67b10ab4430b0fd4685e`. Either way the instrument is fixed
before any re-test outcome exists; never choose it after observing outcomes.
Record the applied instrument, full commit and selection/start timestamps in
the dispatch receipt. This affects offline measurement only; camera candidate,
deployment, lever and route stay fixed. **Nearest-GT results are supplementary**
and must be reported alongside, never silently substituted for the registered
primary identity instrument. If fallback applies, label the current nearest-GT
instrument as primary and report its association diagnostics separately.
The nearest-GT supplementary column is always retained; if fallback applies,
it coincides with the fallback primary counts. A metadata-only recorder,
started before queue insertion, reads only public PR merge metadata and lane
start timestamps, never outcomes. It fixes the eligible merge SHA immediately
when available (or the fallback at the first-start cutoff), records
`compute6:~/dojo-data/kong-retest-1107/audit-instrument.json`, and amends the
remote dispatch receipt with the applied instrument. The timestamp cutoff and
selection rule above are immutable even if GitHub evidence is temporarily
unavailable. No retroactive re-scoring of the previous verdict under this
registration.

## Call estimate, queue and receipt

The previous Sonnet panel used 142 logical calls for 40 rows: 36 kong,
62 imitation, 44 stack. Doubling only kong gives a **178-call estimate**;
linear row scaling gives **213**. Plan roughly **180–215 Claude calls**, with
300 as planning headroom, not a new episode cap or score-based stop. At the
previous overall usage per row, 60 rows would be ~3.27M prompt / 226k completion /
150k reasoning / 2.53M cache-read tokens; report actuals and retry observability.
Preparation and physics audits make zero model calls. Worst-case camera cap
is 25 summed lane-hours before infra retries (12.5 h on two lanes); expected
occupancy and queue ETA are recorded after prep from recent lane timings.

Append all 60 rows **after every currently queued row** in
`~/dojo-integ-c6-state/queue2.txt` under `flock queue2.txt.lock`. Preserve the
existing bytes, order, names and started jobs; never move or rewrite tie rows.
Verify every new row's explicit Sonnet route, deployment, frozen layout/demo,
A/B config and whole-pair adjacency. Record queue positions/counts, byte hashes
before/after, manifest SHA256, insertion UTC, ETA and audit-instrument selection
in this document and remote `notes.log`. This prereg/index/design commit must
already be on main before any row is queued.

**Dispatch receipt:** pending preparation; no re-test camera rows queued by
this registration commit.

## Lead-authorized demo-input amendment (before any row is queued)

The lead prospectively amended input eligibility after fresh imitation id
**100050** failed normal demo validation at speed 1.25 after all four standard
order tries: **unstable, only 3/5 support objects placed**. This is input
preparation evidence, not a camera outcome. No re-test row has been queued.

For the imitation parity guard (and any cell requiring a demo), select the
**first five ascending fresh ids that pass BOTH default two-pass presettle /
applicable raw-check AND the standard demo validation**, using input validity
only, never a camera/policy outcome. Keep speed **1.25**, normal order tries,
validation and the existing fresh salt / 100050–100099 candidate pool. Retain
every failed raw/settled/demo attempt and its exact failure reason in the
manifest. This explicitly supersedes the earlier fixed-five/demo-failure
launch blocker and permits ascending refill to restore the registered n=5.
No speed escalation, cherry-picking or n reduction is authorized.

The gain-cell, guard and decision-rule sections remain unchanged. If the
imitation demo failure rate exceeds 30% of attempted demo inputs, record the
input-validity landmine in BACKLOG with numerator/denominator and reasons.

## Preparation freeze (before queue insertion)

All **20 gain / 5 parity / 5 strong** identities are prepared and frozen on
`ffb2c151f1d48f5bc68f67b10ab4430b0fd4685e`. Gain ids **100050–100069** and
stack ids **100050–100054** all passed unchanged two-pass settling without
refill. Imitation ids **100051–100055** pass both settling and standard demo
validation at 1.25. Rejected **100050** is retained with all four normal order
tries, the unstable flag and 3/5 supports placed. Demo failure rate **1/6
(16.7%)** does not exceed the registered 30% BACKLOG threshold.

[Manifest](data/kong-scoped-retest/manifest.json), SHA256
`844d0ea503d8be6cf3aaf879a711f6eb4144fe813f0e053e7cd0a95a2a360a70`.
It freezes all attempted raw/settled records, validation indices, five valid
demos, every failed id/reason, all 60 planned rows and AB/BA order.
[Shared pins](data/kong-scoped-retest/shared-freeze.json) retain both design
versions, prereg/amendment commits, deployment source hashes, current package
parity, catalogue/background/calibration/checkpoint hashes and operational
scripts. Current packages have `diff={}`; all eleven checked deployment source
files match the frozen commit. Nine focused scoping tests pass on that exact
deployment. [Effective-config/Skills snapshots](data/kong-scoped-retest/parity-proof.json)
match for both guards; the required post-collection proof remains pending.
[All input selection attempts](data/kong-scoped-retest/selection.json).

The instrument PR is not yet published/merged at this freeze. The registered
merge-before-first-start rule and pinned fallback are unchanged. The public
metadata recorder is healthy and will put the applied instrument/merge SHA in
`~/dojo-data/kong-retest-1107/audit-instrument.json` and the remote dispatch
receipt before the verdict; it never reads episode outcomes. No re-test row
has yet been queued, and no camera verdict is made here.

## Dispatch receipt — 2026-11-07 07:38:49Z

queue position moved later by the lead for POUR; no input/config change

**Appended 60 rows, queue positions 79–138 (one-based at insertion), after all
78 existing tie rows.** The append held `queue2.txt.lock`; the prior queue
bytes are an exact unchanged prefix. No existing row was reordered, rewritten
or held, and no lane/cap/STOP settings changed. All 30 new pairs remain adjacent
in the frozen AB/BA order. Both already running tie arms remain untouched.

- Preregistration/index/design on main **25ab63a0**, prospective input
  amendment **0de2aad4**, complete preparation freeze **73560a60**: all before
  insertion. [Machine-readable receipt](data/kong-scoped-retest/dispatch.json).
- Existing rows **78**, appended **60**, queue total **138**. **60/60** new
  rows verified for explicit `claude/claude-sonnet-5-5`, pinned deployment,
  arm config, loaded layout and pair adjacency. This is a dispatch route check;
  actual call-log/transcript/report route proof remains required after collection.
- Queue SHA256 before:
  `5a2ea474ac721ac936908bd8f1cf9502dd4d2b2e8f8870029a178350289aa2b9`.
- Queue SHA256 after:
  `ac2d1f602234b41a8ae7d9b67d883fb363985cd014ae2e0cffe958d66eb86aae`.
- Manifest SHA256:
  `844d0ea503d8be6cf3aaf879a711f6eb4144fe813f0e053e7cd0a95a2a360a70`.
- Appended-row file SHA256:
  `97e50bd318d55f7e92a9bd1c95e41e89a3e5baf9b93af3e90aef38b47b13368c`.
- Rough two-lane ETA: **14:25Z first re-test row, 16:47Z final row**, 7 November.
  Estimated existing queued/active occupancy **48,749 lane-seconds**, new block
  **17,065 lane-seconds (~4.74 lane-hours)**. Based on recent tie timings and
  prior Sonnet per-cell means; admission, startup, time caps and infra retries
  can delay it. The preregistered cap ceiling remains 25 new camera lane-hours.

### Audit instrument status at dispatch

No `fix/audit-identity` PR was published/merged at insertion and no re-test row
had started. Accordingly the [dispatch-time instrument snapshot](data/kong-scoped-retest/audit-instrument-at-dispatch.json)
is **registered_pending_cutoff**, not a claim that the future cutoff already
occurred. Its fallback is pinned to
`ffb2c151f1d48f5bc68f67b10ab4430b0fd4685e`; the immutable registered rule uses
the identity PR's full merge SHA if merged before the first re-test start.

The metadata-only recorder remains active. The authoritative applied-instrument
receipt is `compute6:~/dojo-data/kong-retest-1107/audit-instrument.json`;
it also amends that directory's `dispatch.json` and `notes.log` with the applied
instrument, full commit, merge/selection/start timestamps, without reading
outcomes. Once fixed, it stops. The instrument choice is determined by the
precommitted merge/start cutoff, never by outcomes; the final applied pin must
be included with the later verdict. Nearest-GT results remain supplementary
under the registered conditional instrument rule. No camera outcome was read
for this dispatch and no verdict/adoption/default change is made.

## Lead protocol correction — eligible rendered ids, before replacement collection

At 10:41Z the lead removed all 60 refused-id rows under the queue lock to
`compute6:~/dojo-data/pour-cam/held-refused-ids-1041.txt`. None started;
**no KONG re-test camera outcome exists**. The old 100050–69 preparation and
receipts above remain historical only and are superseded by this amendment.
Those ids are Mikhail's D-C10a slice and integ refuses them rendered (rc=2).

Regenerate every cell under a **new salt**, frozen in
[the eligible design](data/kong-scoped-retest/design-eligible.json):
`kong-scoped-retest-eligible-20261107-0a5495833a42acc6b37f50426c3c4cae3ce155a466c61f4c897fad0cbba7f4b9`. Literal UTF-8 salt SHA256:
`8ee55262d8806828ef6678fa99e77ea07898bc5e85d544f3977dcf3d8083ad24`.
Private replacement root: `compute6:~/dojo-data/kong-retest-eligible-1107/`.
No refused layout or demo is reused. Freshness is the new salt; the positive
rendered allowlist is `dojo.simlab.paths.OUR_MEASUREMENT_SLICES` only.

- make_kong: exactly 20 pairs, ids **100000–09 and 100020–29**.
- imitation parity: the first five ids ascending within **100000–09** passing
  BOTH unchanged two-pass presettle and standard demo validation at **1.25**.
- stack strong: first five ascending stable ids in the eligible allowlist.

All failures/attempts remain in the replacement manifest. No tolerance,
validation, camera policy, sample-size, guard or refined decision-rule change.
The >30% demo-failure BACKLOG rule remains. A shortage of eligible valid inputs
blocks launch rather than borrowing from another slice.

**Instrument:** #417 merged at 07:51:39Z, before any re-test start. The primary
identity-association instrument is now fixed to full merge commit
`77b4571907f6ce49eec11c0685db8f0c7ba05d64`.
Nearest-GT remains supplementary; the camera deployment stays the same #410/#411
`ffb2c151` deployment, with Sonnet explicitly pinned in every arm.

The prep script must refuse any candidate/selected/rendered id outside
`is_our_measurement_id` before generating or staging it. Recheck the allowlist
when constructing and appending rows. Append the replacement 60 rows at the
queue END under `queue2.txt.lock`, preserving all existing bytes and priorities.
Freeze a new manifest and publish replacement positions, hashes and ETA.
