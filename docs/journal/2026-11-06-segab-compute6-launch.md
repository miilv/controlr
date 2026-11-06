# Additive v2n camera A/B on compute6

SEGAB is preparing the [registered comparison](../experiments/2026-11-06-seg-v2n-camera-ab.md)
on compute6, using merged PR #387 at `8473130a8fdb45ba586b4454ea71cd771b85d4e1`.
Both arms use this deployment, immutable v2b and the frozen candidate. A disables
additive segmentation; B enables v2n at IoU 0.1 / minimum score 0.5. All reader,
pad and candidate runtime options are retained. This is measurement, not adoption.

## Lead amendments before collection

The lead assigned compute6 instead of the prospective compute4 host, the candidate's
`claude/claude-sonnet-5-5` route instead of cx, and interleaved A/B queue rows.
No compute4 connection, forbidden data directory or coordination path is needed.
Compute3 is read-only and supplies only the registered checkpoint bytes.

After reconciling the spec's existing `-a` and `-b` blocks with its no-replacement
rule, the lead explicitly authorized one `-c` refill block per affected cell.
Salt strings are `seg-v2n-camera-20261106-<slug>-c`, with slugs toast, mugs,
dolls, laptop, pour, blocks and tubes. All a/b/c strings and SHA256 values were
frozen in compute6 `~/dojo-data/segab-1106/salts.json` before generation; file SHA256
`68863000aad54050e74c169ed61f97d9276d3095b231f0ed5ba8cd5a0ac5d991`.
The amendment was recorded in `notes.log` at 13:54:50Z, before any SEGAB camera job.
Take passing refill IDs in ascending order, retain every exclusion and its reason,
and generate no blocks after c. If still short, retain the cell with explicit n
and the unchanged decision rule. Ready cells need not wait for another cell's refill.

The decision remains pooled gain B−A at least five points, gain success flips
wins at least twice losses with at least two net wins, and each guard B−A at least
−5 points with flip losses no greater than wins. Missing pairs leave a verdict
pending. Report per-cell n/spread and paired bootstrap CI (10,000 draws, seed 0).

## Deployment and preflight

The source archive is isolated at `~/dojo-integ-c6-8473130a`. A new managed Python
3.12.13 venv reproduces the existing compute6 lane's exact package freeze, including
CUDA torch 2.11.0+cu128 and timm 1.0.30, with the editable project pointed at the
new deployment. The first dependency-resolution attempt lacked the CUDA wheel
index and failed before installing; the corrected installation uses that index.
Package/candidate parity passes with `diff={}` against the live 416a2a03 lane's
package reference. The catalogue gate passes. Cross-version resolved candidate
comparison preserves every existing field; nine new RuntimeConfig fields default
`off`. Full evidence is under `~/dojo-data/segab-1106/`.

Checkpoint MD5s on compute3 and compute6 match:

- v2b: `2f546196145fc424f76a5e2e47dd3210`.
- v2n: `be0817051b8efd0565218033cb647cec`.

V2n and its 32-class map are isolated under `segab-1106/weights`; existing v2b,
digit and kong files remain untouched. The dev-only server check used the prior
INTEG6 dev frame 100002, never a scoring frame. It reports 511 base / 32 extra
classes and `merge=add-only`; nine base detections have an exact serialized prefix
under additive requests. This dev image produces no extra detections. The server
was stopped by its checked-PID wrapper after the check. No LLM call was made.

All 70 original salted layouts were generated using the pinned v15 generator.
Every original gain record contains at least one registered new asset. Preparation
holds an existing heavy lane lock between jobs, with bounded CPU/RAM and VRAM
admission; no light lock is taken. Toast uses the existing raw-check path; the
other tasks use two-pass presettle without tolerance changes. The offline Kit
extension overlay and both 600-second watchdog settings are inherited unchanged
from INTEG6's dev-qualified recovery recipe.

The two managers drain existing jobs normally before reloading an isolated copy
of the operational launcher. Only SEGAB rows request its `v2n-ab` private service:
both checkpoints load for both arms, while request config selects A/B. Existing
shared services and tie-row deployments stay unchanged. The queue update must
preserve a queued mate of any already-started tie pair before the SEGAB block,
hold `queue2.txt.lock`, and record before/after hashes and exact inserted rows.
HTTP 429 retains the existing global dispatch stop; power monitoring retains
heavy2's drain after two consecutive samples over 700 W.

Preparation and launch evidence will be added after the first camera START.
