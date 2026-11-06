# Additive v2n camera A/B on compute6

SEGAB is preparing the [registered comparison](../experiments/2026-11-06-seg-v2n-camera-ab.md)
on compute6, using merged PR #387 at `8473130a8fdb45ba586b4454ea71cd771b85d4e1`.
Both arms use this deployment, immutable v2b and the frozen candidate. A disables
additive segmentation; B enables v2n at IoU 0.1 / minimum score 0.5. All reader,
pad and candidate runtime options are retained. This is measurement, not adoption.

## Lead amendments before collection

The lead assigned compute6 instead of the prospective compute4 host, the candidate's
`claude/claude-sonnet-5-5` route instead of cx, and interleaved A/B queue rows. The registered AB/BA parity by task, block and
layout index is retained; interleaving does not require A first.
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
admission; no GPU preparation uses a light lock. Toast uses the existing raw-check path; the
other tasks use two-pass presettle without tolerance changes. The offline Kit
extension overlay and both 600-second watchdog settings are inherited unchanged
from INTEG6's dev-qualified recovery recipe.

The two managers drain existing jobs normally before reloading an isolated copy
of the operational launcher. Only SEGAB rows request its `v2n-ab` private service:
both checkpoints load for both arms, while request config selects A/B. Existing
shared services and tie-row deployments stay unchanged. The [operational diff](../experiments/data/seg-v2n/lane2-private-service.patch)
is retained for review; the deployed launcher SHA256 is
`c904b0d3f6e7bfabd9c6355e69588d975ad0e3315a97d2938bc3043ccde0df43`. The queue update must
preserve a queued mate of any already-started tie pair before the SEGAB block,
hold `queue2.txt.lock`, and record before/after hashes and exact inserted rows.
HTTP 429 retains the existing global dispatch stop; power monitoring retains
heavy2's drain after two consecutive samples over 700 W.

Local focused validation passed 28 tests with seven optional skips. The pinned
deployment passed all 35 of these tests with no skips. Shell syntax and all
140 original RuntimeConfigs plus both merged estimator configurations validate.

Toast prep passed all ten raw checks. Its first staged queue incorrectly used
A first throughout; before any SEGAB camera START, a locked correction restored
the registered AB/BA parity without changing inputs or configs. The original
manifest is archived; `toast-order-correction.json` records both queue hashes.
The corrected toast manifest SHA256 is
`634c48a3e5bafcb286071ebbff4990e6cd9cbe4a30f1f7dfd1c7a04b34afae04`.
At 14:03:15Z the first 20 rows were inserted, retaining the queued
`c6b_imitatesg_tieAr2_100005` mate ahead of SEGAB. Heavy2 reloaded at 14:04:04Z,
then waited at its unchanged camera memory gate. The power log
`lane2_draining` flag also reflects our temporary manual drain marker; it is
not evidence of a high-power event. Power-driven reduction is identified by
the watcher reducing `lane2.caps` to one.

## First launch checkpoint

At **14:34:04Z**, heavy1 started `c6b_poursg_segA_a_100000` from the pinned
8473130a deployment, with `seg_additive=false`. Its private SEG server was up
in 12 seconds. The recorded start sample was **29.71 W / 12,847 MiB used**.
The actual perception worker PID **296659**, read through an environment
allowlist, confirms the Claude model, complete A estimator config, private
SEG socket, shared digit/kong addresses and frozen background/calibration paths.
The live private server reports 511 base / 32 extra classes and `add-only`.
Evidence: `first-A-live-verification.json`. No scoring frame was inspected.

Ready cells are prepended individually as they qualify; each insertion retains
any queued mate of an already-started pair at the front. AB/BA parity is fixed
by the original task/block/id indices. Thus the exact cell order reflects
resource readiness, with each cell's explicit frozen schedule retained. The
pending tie A mate started at 14:15:58Z and completed normally at 14:36:43Z;
both original tie arms finished without being interrupted. Every queue mutation
has a locked receipt with before/after hashes and the inserted commands.

At **14:36:43Z**, SEGAB had no completed episodes or paired result. Completed
call logs recorded **4 camera calls**, **52,083 prompt / 2,135 completion** tokens,
including **1,283 reasoning / 42,036 cache-read** tokens. In-flight calls are
additional. Implementation, validation and the dev smoke made zero model calls.
The verdict is **pending**, not a negative result or adoption decision.

All five gain cells and stack_blocks are frozen at ten pairs each. These
per-cell manifests under `~/dojo-data/segab-1106/` include raw/loaded SHA256+MD5,
all attempted index statuses, resolved runtime/perception configs, exact queues,
and shared generator/salt/catalogue/calibration/background/weight identities:

| Cell | Pairs ready | Manifest SHA256 |
|---|---:|---|
| make_toast_random | 10 | `634c48a3e5bafcb286071ebbff4990e6cd9cbe4a30f1f7dfd1c7a04b34afae04` |
| hang_mugs_random | 10 | `2bf85de3eb0e569a6ea736ba3d195ab126f5b9df47b2e63d8a752ff8de066308` |
| sort_nesting_dolls_by_size_random | 10 | `5464c026127bc5e2b2d0baba71f7fd66fd41ce00a23f387f5216c9116180fd4c` |
| store_laptop_and_headphones_random | 10 | `c394ecf076657f8c8c783cc18782074bfb4b5a9f4120f43a7eff03429f43a833` |
| pour_liquid_into_cup_random | 10 | `4542aa8609c99bfeed59aece8ca9210a1f5afe4dc20bbb2e8419fb8c6e93e9f6` |
| stack_blocks | 10 | `b9aef4e9bcd0999155110dfbcbc9d83ecc0e80445a55079035173cb29e46ffc2` |
| insert_tubes | pending | heavy2 prep started 14:36:51Z |

Sixty selected original layouts passed (ten raw checks and fifty two-pass
settles); no exclusions or c refills were needed. Tube originals are still
being prepared. Their rows will not be queued until they qualify. `finalize.py`
will verify all recorded file digests and write the aggregate `manifest.json`
when the seventh cell has been inserted. No aggregate hash is claimed yet.

Operational ownership: camera managers **287133 / 277591**, existing power
logger **247816**; own prep **286386**, dispatch **277483**, manifest finalizer
**289672**. Check PID command lines before acting. Prep uses a free heavy slot,
limits CPU to six cores/RAM to 20 GiB, and preserves roughly 6 GiB free VRAM.
A 14:36Z light-slot check found used VRAM 23,832 MiB and load 98.77, so no GPU
prep was started on those slots. Existing light streams were untouched.

The nominal full budget is **35 lane-hours**, or **17.5 wall-hours at two
continuously admitted camera lanes**, before retries and admission waits.
Capacity has fluctuated above the unchanged 16 GiB-used camera gate, so there
is **no reliable wall-clock completion ETA**. For reference only, uninterrupted
two-lane service from the first START would finish around 08:04Z the next day;
this is not a measured forecast. The original registered 7,000-call budget
remains an estimate; actual call/token logs are retained. A 14:36Z instantaneous
sample was 104 W; no >700 W power-triggered reduction or HTTP 429 was observed.

A same-thread follow-up is scheduled every 30 minutes (next 15:07:31Z), bounded
to the approximately eight-hour monitoring box ending **21:51Z**. It continues
remaining prep/dispatch, verifies the first B worker, reports progress and
publishes an end-of-box report, then removes itself. Missing pairs retain a
pending verdict. Authorized camera lanes may continue after the monitoring box.

