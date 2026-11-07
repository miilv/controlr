# 2026-11-07 — POUR camera, once-only fresh-layout re-test

Lead-authorized under EXPERIMENTS.md's one re-test allowance for a rejected
lever with positive pooled delta. The [original verdict](2026-11-07-pour-camera-prereg.md#pour-camera-verdict--fail--not-adopted)
remains **FAIL / NOT ADOPTED** under its registered gates. This is a new panel
and prospective rationale, not a re-judgment of that result. No adoption or
candidate/default change is authorized by preparation or dispatch.

## Frozen comparison

[Design](data/pour-camera-retest/design.json). Reuse the identical implementation
066e9c8f34c727d9bbfdfa1c219e7c6a78366a45 (#415/#417), isolated compute6 deployment
/home/robotics/dojo-integ-c6-066e9c8f. The current board candidate file
scripts/dojo/queues/2026-11-06-candidate.txt is still SHA256
5cde3f9e684c0937f21a0f28609437e6d317a433e000bc171198349417cd398e.
Recheck package/catalogue/candidate/native source and calibration/background/
checkpoint pins before generation. No simulator, driver or shared service change.

Language: A candidate with aim b10, profile off, pace v1; B identical except
pace budget. The candidate already enables pour_order/pour_by_language equally.
Strong-guard Xylophone retains the same raw A/v1 and B/budget flags and is
camera-gated. Single-cup is scoped to **v1 in both arms**, explicitly in each
row: this makes its complete resolved RuntimeConfig identical, and agrees with
the existing single-cup planner resolver even when the global flag is budget.
No application code changes implement this scope; queue configs are explicit.

Both retain submit_v3, grip v2, v2b/digit_v2/kong_v1b, unchanged candidate
perception JSON, host-mapped frozen calibration and backgrounds. Every row pins
AGENTIC_MODEL=claude/claude-sonnet-5-5; verify actual call/transcript route per
row after collection, not copied deployment labels. A mismatch blocks
qualification. Same 25-minute episode/job caps, native watchdog600 and lane recipe.

## New inputs and fixed sample

20 language gain pairs, 10 single-cup parity pairs, 10 Xylophone strong-guard
pairs: **40 pairs /80 rows**, one repeat per arm. The strong guard qualifies
from public candidate camera100 >=90; the first POUR panel's B also scored90.
Each task has a new independently drawn salt, frozen in the design. Fresh
identity is task+salt+raw/loaded hashes; numeric ids must be reused because our
only rendered slices are **100000–09 and100020–29**. Never reuse an old POUR
record or salt. No sealed data, Eval_Layout, compute4 or old measurement record.

Select ascending, validity only, from those20 eligible ids. Generate the first
required count (split generator batches at the slice gap), raw-check, then
default two-pass presettle:250 extra steps,100 rest steps,1mm/1degree rest test,
5mm reload drift. Keep first n passing both. Refill only with next eligible ids
in batches of the missing count; retain every failure/raw/settled/validation.
No fallback, relaxed thresholds, new salt search, n reduction or outcome-based
selection. Pool exhaustion blocks launch pending a prospective lead amendment.
Freeze all input/attempt hashes and expanded rows on main before queueing.

AB/BA alternates by selected pair index in each cell; same loaded record in
each pair, independent resets. No controller reruns or optional stopping.
One classified infrastructure-only retry on the same input is allowed; retain
both attempts. Missing planned pairs block qualification. Reports preserve
policy budget failures, slips and checker misses as outcomes.

## Registered decision rules

Language official 0–100 score: mean B−A >=5, score wins (delta>=10) >=2×losses
(delta<=−10), and at least2 net wins, over20 complete pairs. Apply the **same
screen to full-success-only scores** (100 on full success,0 otherwise). Report
success flips, official partials, mean/sample SD and all paired deltas. No idle
motion to manufacture score edges. Descriptive paired-layout percentile
bootstrap95% CI:10,000 resamples, RNG seed0, not another gate.

Single-cup **PARITY guard**: its gate is identical resolved task config and skill
parameters, v1 pacing in both arms, and post-collection zero budget-pour actions
and zero lever-specific writes in **every** episode. Report camera scores,
successes, spread and paired wins/losses, without a score gate. Missing parity
proof blocks qualification; noise is not a reason to drop a row.

Xylophone **STRONG guard**, ordinary EXPERIMENTS.md camera gate: B−A mean>=−5
and success-flip losses<=wins. Report score wins/losses and full-success-only
scores without adding another guard gate. No parity exemption on this guard.
This wording follows the lead's “camera-gated as usual” instruction; the gain
cell retains its explicitly unchanged additional full-success-only screen.

## Motion-only scope: write gate N/A, supplementary audit retained

The lead prospectively applies the same motion-only rationale as the waist
verdict. EXPERIMENTS.md's write-adoption gate addresses perception/belief changes.
Pacing changes motion dispatch, overlap, lift/untilt speeds and return reserve;
it does not change a writer implementation, settings, acceptance rule or schema.
The scope proof must pin source/read sites, resolved config/Skills parameters
and per-row writer inventory. Only the motion pace differs on language; writer
settings and code are identical. Check that inventory remains within those
unchanged writer paths after collection.

**Write audit N/A as an adoption gate, by this proof.** Different observed writer
counts/outputs following different trajectories do not trigger a gate. If source,
writer settings/acceptance/schema differ unexpectedly, the scope proof fails;
do not relabel that as motion-only. There is deliberately no observed-count/
output trigger.

Still run/report the **frozen #417 identity-instrument audit as supplementary**,
with raw harmful/helpful/neutral/uncertain counts, per-writer rates, z separately,
exact write-step replay coverage, stable resets, nearest fallbacks and live-probe
fidelity. Preserve the original instrument; do not repair anchors from outcomes.
The original panel's provisional identity/ported-grasp and4.9mm fidelity
limitations remain disclosed; audit limitations cannot turn into a new gate or
be represented as harmlessness clearance. GT is offline-only, never policy input.

Single-cup source proof: planner.pour_bottle_call(b10,None,off) resolves
pour_bottle with spout_back0.010; it does not select pour_bottle_lean. The adapter's
two-arm budget path is enabled only for pour_bottle_lean. Freeze identical
single-cup config/constructor/skill snapshots; verify zero budget-pour actions
and lever writes in traces. Perception/predicted ordinary writes are reported
and may differ; they are not lever-specific writes.

## Queue, compute and spend

User robotics, compute6; write only within the existing isolated deployment,
~/dojo-data/pour-cam/retest and authorized lane state. One admitted light prep
job, VRAM used<=20GiB/load<=26,20G/600% scope,disk>=50GB, native physics-only
raw/settle. Zero prep model calls. Stop only own PIDs; no lane/service/STOP/cap
changes and never touch others' processes.

Under admission then queue2.txt.lock, insert **immediately after remaining
MATCH2 screen rows** (m2scr/match2 screen family), before every unstarted tie.
Protect queued mates/retries of started pairs. Preserve every old row's bytes
and relative order. If MATCH2 has not yet queued, insert before first unstarted
tie and reserve its20-row priority ahead of POUR operationally—no dummy jobs.
Record MATCH2 rows/positions or already-started/completed evidence, protected
prefix, queue before/after hashes, manifest hash, insertion UTC and rough ETA.

Estimate **about470 logical Claude calls**, using the completed original panel's
467 as planning context, not a cap: roughly6.6M prompt/0.4M completion/0.24M
reasoning/5.4M cache-read tokens. Actual calls/tokens/errors are reported;
reasoning/cache are subsets. Supplementary physics audit makes zero model calls.
No new camera outcomes are used before freezing this panel.

Queue family: c6b_<taskslug>_pourrt[AB]r1_<id>, distinct from the original pourcam rows.

**Receipt:** dispatched 2026-11-07 15:43:21 UTC, positions9–88, after all eight remaining MATCH2 rows. The initial registration commit queued no rows; see the receipt below.

## Preparation freeze before queue insertion

The prospective registration/design/index/rationale reached main in **d956bc00**.
Before any camera row is queued, the strong-guard wording is aligned literally
with the lead's ordinary EXPERIMENTS.md score/flip rule; it has no extra score-win
gate. Language's additional full-success-only gate is unchanged. This prospective
clarification changes no inputs, motion, source or observed outcomes.

| cell | attempted layouts | selected | validity failures retained |
|---|---:|---:|---:|
| pour_by_language | 20 | 20 | 0 |
| pour_liquid_into_cup | 13 | 10 | 3 |
| play_Xylophone | 12 | 10 | 2 |

All selected records passed raw-check and unchanged default two-pass settling.
Selection is ascending and validity-only. Every failed record/reason/hash remains
in the [selection ledger](data/pour-camera-retest/selection.json) and manifest;
none is silently refilled by another salt or an off-slice id. All selected numeric
ids are in our two slices, with the new registered task salts and hashes distinct
from every original POUR input.

The [manifest](data/pour-camera-retest/manifest.json) SHA256 is
3334e3eba4862f99c7b0f53e40b7e4c52602122445ec13802028406975745218. It freezes all80 expanded rows, paired loaded inputs, attempts, complete
resolved configs and operational script hashes. Also committed:
[pre-generation freeze](data/pour-camera-retest/pre-generation-freeze.json),
[scope proof](data/pour-camera-retest/scope-proof.json) and
[exact rows](data/pour-camera-retest/queue-retest.txt).
The immutable old deployment's130 source/native/candidate/package/catalogue/
calibration/background/checkpoint descriptors rehash unchanged. Writer source/
settings/acceptance/schema remain common and have no pace read. Constructor/
dispatcher source is frozen with the resolved configs.

Single-cup's complete RuntimeConfig is identical across arms (pacev1); both
resolve pour_bottle with spout_back0.010 and no pair-motion mode. Post-collection
zero budget actions and lever writes on every cup episode remain its gate.
Observed downstream writer counts/outputs are reported without a write gate.
The frozen#417 audit is supplementary only, with its known limitations retained.
No supplementary audit has run on this panel before camera collection.

Every row explicitly pins Sonnet5.5, aim b10, profileoff and the common route,
with fresh private DOJO_DATA and the same episode caps. All preparation made
zero model calls. This complete freeze reaches main before insertion.

## Dispatch receipt — 2026-11-07 15:43:21 UTC

[Machine-readable receipt](data/pour-camera-retest/dispatch.json). Under admission
then queue locks, **80 re-test rows were inserted at positions9–88**, immediately
after the eight remaining MATCH2 screen rows (four intact pairs, ids100006–09).
Both active MATCH2 arms at100005 were untouched; no started pair was split.
There were **no unstarted tie rows in the queue** at this snapshot, so none can
precede POUR. All eight old rows retain exact bytes and relative order; the
queue had88 rows after insertion. No lane, STOP, cap, service or other process
setting changed.

- Registration/design/index/journal rationale on main: **d956bc00**.
- Complete frozen inputs/configs/source proof/rows on main: **fb3f70db**, before
  any re-test row was queued.
- Queue SHA256 before: a83a2027d3c2e94e230e5d3c178fc018efe6c8f34a27f59af42ab8f759a25d36.
- Queue SHA256 after: 88ddd338fd135332566ff357fde3bfb5ae39cd5079102b5c890d3ed548c56102.
- Manifest SHA256: 3334e3eba4862f99c7b0f53e40b7e4c52602122445ec13802028406975745218.

Post-insertion snapshot hash/order verification passed. All80 new rows remained
queued at verification with one explicit Sonnet pin each. The bounded, read-only
route monitor PID1259069 is alive with no violations and zero model calls of
its own; actual per-row call proof is pending as collection starts. It also
recognizes classified retry paths and the task-suffixed report format.
Preparation exited normally and left no own simulator running.

Timing estimate at insertion: first POUR work around **16:10 UTC**, completion
around **19:30 UTC on7November**. This uses preceding MATCH2 timing plus original
19121 summed lane-seconds and the lane's serialized150-second startup cadence
(~3h20 for the block). Admission, caps and infra retries can delay it. Planned
spend remains about470 logical Claude calls; actuals are reported later.

The final selected sample is40 pairs from45 attempts: language20/20 valid,
cup10/13, Xylophone10/12. Three cup settles and two Xylophone raw reset failures
are retained. Raw checks passed43/45 attempts; two-pass settling passed40/43.
Every selected input passed both checks, with only ascending eligible refills.
No camera outcome selected a replacement, and no previous POUR record was used.

Write gate remains **N/A by motion-only scope proof**; #417 audit is
supplementary. Cup is gated by its post-collection parity proof, Xylophone by
the ordinary camera rule, language by both registered gain screens. No verdict,
adoption, candidate or default change is made by dispatch. The original POUR
verdict remains **NOT ADOPTED**.
