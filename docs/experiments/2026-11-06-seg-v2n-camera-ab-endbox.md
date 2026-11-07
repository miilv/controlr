# Additive v2n camera A/B: end-of-box snapshot

> **Final verdict (lead, 2026-11-07): DOES NOT PASS.** The complete 70/70 panel and its reading are in
> [journal 2026-11-06-ab-wave-verdicts](../journal/2026-11-06-ab-wave-verdicts.md) (seg v2n section). The snapshot
> below is kept as recorded at the 21:51Z cutoff.

**Verdict: pending.** At 2026-11-06 21:51:16 UTC, the runner had 57 paired
results from the 70 registered pairs. Toast had not started; mugs had seven
pairs. One pour result uses an automatic retry after an unscored unstable
reset whose infrastructure eligibility is unresolved. This report does not
qualify adoption or a sealed board score. Production defaults remain off.

The monitoring task was deleted at the end of the approximately eight-hour
box. Authorized camera lanes were left running. This is a frozen cutoff report;
later outcomes require a new report, rather than edits to these results.

## Setup and evidence

Question: does independent new-class v2n segmentation improve the five gain
cells without regressing two strong guards? Follow the [registration](2026-11-06-seg-v2n-camera-ab.md)
and [launch journal](../journal/2026-11-06-segab-compute6-launch.md), merged in
[PR #403](https://github.com/miilv/controlr/pull/403) with green CI. Code is pinned
to `8473130a8fdb45ba586b4454ea71cd771b85d4e1` on compute6, genuine Isaac 5.1,
frozen candidate `submit_v3`, Claude Sonnet 5.5 under the lead's host/model
amendments. Both arms load v2b+v2n privately; only B enables additive requests
(IoU 0.1, minimum score 0.5). Reader/pad settings are retained.

All 70 original fresh salted layouts passed prep: ten raw checks for toast and
sixty two-pass settles. No layout was excluded and no authorized c refill was
needed. Each cell has two blocks a/b, IDs 100000–04, one repeat; registered
AB/BA parity is retained. Both arms use identical records on the same box.
No frozen input, controller, tolerance or model setting changed after outcomes.

Base v2b MD5: `2f546196145fc424f76a5e2e47dd3210`.
V2n MD5: `be0817051b8efd0565218033cb647cec`.
Aggregate input manifest SHA256:
`92689cca1d1260c01082ae04eef183ecf414728d3ba2a8f6ba089415cd76094c`.
The end audit reverified 280 distinct recorded file paths, including layouts,
indices, queues, shared backgrounds/calibration, weights, catalogue index,
generator, candidate and operational freeze files. All 114 completed source
row matches its registered task/layout and pinned code. Independent paired
arithmetic and 10,000-draw within-cell bootstrap checks match the snapshot.

Artifacts:

- [Frozen cutoff JSON](data/seg-v2n-camera-ab/endbox-2151.json), SHA256
  `d360f5f7b7a098d5430a43b20eda52a4dab351c06af96c2da965d9eb9cd4e6d1`.
- [Independent audit](data/seg-v2n-camera-ab/endbox-audit.json), including the
  unscored attempt, per-arm success/steps/runtime and terminal diagnostics.
- [114 completed episode rows](data/seg-v2n-camera-ab/endbox-episodes.csv),
  each with source-file SHA256. The no-row attempt is recorded separately in
  the audit and incident, not silently inserted as a scored episode.

Remote evidence remains under `compute6:~/dojo-data/segab-1106/` and
`~/dojo-data/integ-c6/runs2`. The frozen remote snapshot is
`checkpoints/20261106T2151Z-endbox.json`; subsequent lane files may grow.

## Paired scores and uncertainty

Scores are 0–100. Arm values show mean ± sample SD. Delta SD is computed on
paired B−A values, rather than separate-arm SDs. The descriptive 95% bootstrap
CI resamples distinct layouts within cell, 10,000 draws, seed 0. Partial mugs
results retain n=7; no missing layout is imputed. Pour statistics are provisional
because of the retry issue below. W/L means B-only / A-only success flips.

| Cell | Pairs | A score ± SD | B score ± SD | B−A ± paired SD | Paired 95% CI | Success A; B | W/L |
|---|---:|---:|---:|---:|---|---|---:|
| make_toast_random | 0/10 | — | — | — | — | — | 0/0 |
| hang_mugs_random | 7/10 | 5.71 ± 15.12 | 6.43 ± 8.02 | 0.71 ± 13.36 | [-8.57, 8.57] | 0/7; 0/7 | 0/0 |
| sort_nesting_dolls_by_size_random | 10/10 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | [0.00, 0.00] | 0/10; 0/10 | 0/0 |
| store_laptop_and_headphones_random | 10/10 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | [0.00, 0.00] | 0/10; 0/10 | 0/0 |
| pour_liquid_into_cup_random* | 10/10 | 50.00 ± 52.70 | 40.00 ± 51.64 | -10.00 ± 73.79 | [-50.00, 30.00] | 5/10; 4/10 | 2/3 |
| stack_blocks | 10/10 | 90.00 ± 31.62 | 90.00 ± 31.62 | 0.00 ± 0.00 | [0.00, 0.00] | 9/10; 9/10 | 0/0 |
| insert_tubes | 10/10 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 | [0.00, 0.00] | 10/10; 10/10 | 0/0 |

The two guard cells meet their observed registered guard conditions: delta 0,
no success-flip losses, ten pairs each. The full equal-weight five-cell gain
mean and pooled CI are **not computed** because toast and three mug pairs are
missing. Available gain flips are 2 wins / 3 losses using the provisional retry;
this is not the full 50-pair success-flip test. Missing pairs retain a pending
verdict under the frozen rule. No score-based stopping, cell removal or rerun
was introduced. No Generalization projection or new sealed score is claimed.

## Unscored reset and automatic retry

The end audit found a gap in the routine checkpoint collector. At 17:12:19Z,
`c6b_poursg_segA_b_100003` started and ended 47 seconds later with rc=0 but no
`*_episodes.jsonl` row. Its summary has `n=0, unstable=1`; the driver log says
`Unstable envs: [0]`. A nearby PhysX message says a non-GPU-compatible convex
mesh cannot collide with a particle system. That warning alone does not prove
the reset's root cause or establish a classified infrastructure failure.

At 17:13:04Z the existing lane automatically queued `__retry` on its general
"no finished episode" condition. The identical-input retry started 17:13:34Z
and ended 17:17:20Z with success/score 100. Both attempt artifacts remain retained;
resolved runtime configs match and all frozen layout hashes are unchanged.
The retry count stayed at one, but eligibility under the registration's
**classified infrastructure only** retry rule is not established. The
[incident](../incidents/2026-11-06-segab-unscored-reset.md) records the source
logic and evidence. No post-hoc classification or further retry is authorized
by this report.

The table reproduces the runner-selected retry provisionally. As a sensitivity
example only, assigning the original reset a failed zero-score outcome would
change pour from 50→40, W/L 2/3, to 40→40, W/L 3/3. That hypothetical assignment is
not adopted after looking at outcomes. The lead must adjudicate this issue
before a final comparable cohort verdict. Routine status messages that reported
no infrastructure failures described parsed episode rows and missed this
no-row attempt; the end audit corrects that incomplete attempt accounting.

## Steps and recorded episode runtime

Mean ± sample SD; wall times are the episode row's recorded wall time. Paired
B−A wall differences include changed actions, outcomes, perception and machine
load. They are not an isolated estimate of extra-detector inference overhead.

| Cell | n | A steps ± SD | B steps ± SD | A wall seconds ± SD | B wall seconds ± SD | Paired wall B−A seconds |
|---|---:|---:|---:|---:|---:|---:|
| make_toast_random | 0 | — | — | — | — | — |
| hang_mugs_random | 7 | 413.71 ± 147.50 | 379.29 ± 85.97 | 409.14 ± 107.79 | 338.90 ± 64.81 | -70.24 |
| sort_nesting_dolls_by_size_random | 10 | 554.70 ± 156.31 | 635.20 ± 182.23 | 696.64 ± 217.50 | 680.33 ± 247.02 | -16.31 |
| store_laptop_and_headphones_random | 10 | 62.20 ± 34.87 | 71.90 ± 44.76 | 207.29 ± 45.00 | 190.20 ± 54.53 | -17.09 |
| pour_liquid_into_cup_random* | 10 | 335.40 ± 80.09 | 306.10 ± 107.12 | 166.58 ± 22.35 | 160.56 ± 29.73 | -6.02 |
| stack_blocks | 10 | 219.10 ± 67.92 | 220.90 ± 68.94 | 104.91 ± 30.73 | 111.46 ± 40.49 | 6.55 |
| insert_tubes | 10 | 456.70 ± 17.70 | 457.10 ± 18.11 | 140.00 ± 29.46 | 134.25 ± 22.07 | -5.75 |

## Diagnostics, spend and capacity

The 114 completed rows comprise 47 normal terminal classifications and 67
`perception_gross` classifications (A 33 / B 34). They report zero time caps,
zero in-episode instability, zero cooking-failure entries and zero missing-mesh
entries. Separately, the original no-row pour attempt reports one unstable
reset. There are 115 completed launcher attempts including that reset, plus two
in-flight attempts:117 STARTs at cutoff. No episode-row `infra_failure` is
recorded; the unscored reset remains explicitly unclassified.

Terminal grader diagnostics contain 216 per-object rows in each arm: A has 6
not-found and 113 found-with-wrong-identity rows; B has 7 and 116. These are
terminal diagnostics, not detector proposal/phantom totals. Extra proposal
counts, phantoms and detector-only runtime overhead have not been established
by this snapshot. No harmful-belief-write qualification is claimed. Success
and typical failure videos, full diagnostic aggregation and any descriptive
Generalization projection remain for the final completed-cohort report.

Recorded camera usage at cutoff, including completed call logs from in-flight
episodes: **659 calls;9,477,400 prompt tokens;610,089 completion tokens**, including
386,658 reasoning and 7,774,943 cache-read tokens. Cache reads are 82.04% of recorded
prompt tokens. Reasoning is included in completion totals. Calls or usage not
yet logged at cutoff are additional; there is no dollar-cost estimate from the
subscription route. Development, validation and reporting made no model calls.

Completed launcher lock occupancy is **38,762 seconds (10.77 lane-hours)**,
including the47-second reset attempt. The two in-flight runs had 340 seconds
combined elapsed since START, reported separately because this omits their
pre-START service initialization. Physics preparation occupied 1,626 seconds
(27m06s) earlier, as recorded in the launch journal.

Five-minute power samples during the booking span 11.23–398.14 W; the latest
sample before cutoff is 103.67 W at 21:47:33Z. These are sampled readings, not peak
power measurements. There was no power-triggered lane reduction or active STOP
at cutoff. Both camera managers and the existing power watcher were alive.
The unchanged 16 GiB admission, offline Kit overlay, 600-second watchdogs and
HTTP 429 global dispatch stop remain in force. Shared services and other streams
were untouched.

## Handoff

At cutoff there are **26 scheduled episodes outstanding**: two mug attempts
in flight,24 rows queued (four more mugs and twenty toast). Thirteen paired
results are missing. At-cutoff manager PIDs are 287133 / 277591 and power logger
247816; verify command-line ownership before acting on a recorded PID.

The bounded monitoring task has been deleted. Authorized lanes continue with
the frozen queue; no completed prep, dispatch, finalizer or observer was
restarted. There is no reliable completion ETA under admission waits and
heterogeneous episode durations. The outstanding final report must reconcile
the reset retry, include every remaining pair, apply the unchanged gain/guard
rule, and retain these cutoff artifacts. No adoption or default change occurs
at this handoff.
