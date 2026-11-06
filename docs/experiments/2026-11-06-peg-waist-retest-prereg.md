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
