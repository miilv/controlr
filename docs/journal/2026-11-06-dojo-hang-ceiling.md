# 2026-11-06 — HANG: handle grasp and hook attainability

Completed negative G3 study: no ceiling gain under these probes. All switches
remain off; camera prerequisite fails. Lead review requested; do not merge.

## Protocol

Read CLAUDE, DEVELOPMENT, ARCHITECTURE, EXPERIMENTS including the Camera A/B
rule, hang journals and BACKLOG, G3 tooling, and EGG PR #399's journal.
Physics-only G3 mode A on compute6, no verifier or model calls. Original dev
ids 100000-04, followed by private v15 variant-2 draws 100030-39 generated
under `~/dojo-data/hang/simlab/layouts/`. Settling exclusions are separate from
skill failures. No tuning on 100005-09, 100020-29 or >=100100. No compute4 or
D-C19 data access. Private deployment `~/controlr-hang`; retained data under
`~/dojo-data/hang`.

Initial source: `62f22c22`. Candidate snapshot: `1e79bef4`, separately deployed
under `~/controlr-hang/candidate` so initial jobs retain their original source.

## Candidates, default off

- `hang_half_turn`: alternate handle orientation about its upward axis. Hole
  center, threading line and insertion translation remain fixed. The original
  candidates precede the alternate within each lift strategy. Does not change
  hook eligibility or cradle placements.
- `hang_handle_grasp`: outer-bar pinches across the handle plane, confirmed by
  mesh rays finding opposed contact faces, before existing body-wall grasps.
  Normal IK, finger and carry collision checks remain active.

No checker, physical properties, safety threshold, episode limit, model route
or default is changed. The random mug14 frame/checker limitation is reported
separately; no alternative pose is selected to exploit it.

## Initial observations

Standard 100000-03: scores 100, 100, 40, 100. In 100002 the third mug was
already out of the pads before release (84 mm in-hand shift at the approach).
Random 100000: score zero, 374/800 steps, four loose-pick put-downs; one mug
reported on-hook by the skill was mug14, which receives no credit under
the documented checker limitation. These are partial observations, not a gain
estimate or a new ceiling.

## Resource receipts

Initial baseline launchers exited before simulation while deployment was still
copying; restarted after the clean deployment completed. Initial PIDs 212364 /
212365 held light locks 1 / 2. PEN was using light3, so this briefly made three
G3 jobs across streams; corrected to a single HANG slot. Checked HANG waiters
214082/214083 were stopped, then checked HANG simulator 212365 was killed.
Only random 100000 had been recorded; the interrupted next layout is not an
outcome. Resume PID 215942 requests 100001-04, with completed rows retained.
No other stream's process was signalled. Until the capacity update below, subsequent HANG jobs used light1 only,
8G / 600% scopes, a 20 GB MemAvailable floor, disk guard and <=24 GB used VRAM
admission. No heavy/camera lock is used.

## Completed initial baseline and source freeze

Initial original dev panel, five layouts per cell (scores in points):

| Cell | Success | Mean ± sample SD | Scores in id order |
|---|---:|---:|---|
| hang_mugs | 3/5 | 76 ± 32.86 | 100,100,40,100,40 |
| hang_mugs_random | 0/5 | 0 ± 0 | 0,0,0,0,0 |

Attempt counts from histories (retries overlap; not mutually exclusive episode classes):

| Event | Standard | Random |
|---|---:|---:|
| Pick attempts | 16 | 16 |
| Loose picks put down | 1 | 8 |
| Failed lifts | 0 | 0 |
| No-plan exits | 0 | 3 |
| Failed hang: carry loss | 1 | 4 |
| Failed hang: insertion miss | 0 | 0 |
| Failed hang: release/exit loss | 1 | 0 |
| Budget exhausted | 0 | 0 |
| Mug14 instances | 0 | 4 |
| Skill-reported on-hook mug14 outcomes | 0 | 2 |

The phase bins are post-hoc physical diagnostics, not official checker terms:
>30 mm pick-to-pre-release in-hand shift is carry loss; otherwise a failed
hang whose hole is along the hook shaft (x -12 to tip+2 mm, radial error <=12
mm) is release/exit loss, and the rest insertion misses. Continuous coordinates
are retained. Random retries must be deduplicated: only the final assignment
feeds a completed hang. Success traces are intentionally absent in G3, so these
trace bins diagnose failures only. `scripts/skills/hang_phase_report.py` uses
cumulative trajectory segment lengths: the trace's `seg` field names the
PREVIOUS completed action. The two actual standard failures are carry loss
100002 and release/exit loss 100004.

CPU geometry-only check on the initial 15 mugs per cell: mesh-confirmed bar
candidates exist on 9 standard / 4 random mugs; 8 standard / 4 random have
an IK-reachable candidate from home. This does not check collision or physics.
All four random cases are mug14; the grasp probe alone has little prospect of
raising the random checker ceiling. No pose or checker term was changed to
circumvent that limitation.

Rebased over merged EGG/TOOLS work after the initial PR was conflicted (which
prevented CI from starting). Frozen paired source:
`8ff324fd72265ffda0eb296f2e4aeecc6da37255`, `~/controlr-hang/paired`.
Both A (off/off) and B (half-turn + handle-bar) use it. The earlier candidate
snapshot was not measured. The unstarted exploratory waiter 218148 was
checked and stopped during this source transition; no episode was lost.

Generated standard settling: 20/20 stable, variant 2, ids 100030-49. The
first ten stable draws per cell were to be frozen before either paired arm.
At this checkpoint random settling and the pair were pending shared-slot access.
To prevent polling starvation, checked unstarted HANG waiters 219200, 224477
and 222924 were replaced by blocking single-lock waiters. Remaining pipeline
PID 227675; screen waiter PID 227676. The launcher also waits if two other
module-based G3/settling jobs are active. Never more than one HANG simulator
after the initial correction and before the dedicated-capacity update below.
No other stream's process was signalled.

Rebased local full suite: **1842 passed / 187 skipped**, 17m56s. The new
retry-deduplication audit test passes separately. Camera specification fixes
fresh owned-namespace variants 21/22 with stability-only reserves 20/19;
no camera data or render preparation has been collected for the spec.

## Capacity update and first probe

The lead explicitly replaced the shared two-job cap with dedicated HANG
locks `~/dojo-gpu-light5.lock` / `light6.lock`, up to two HANG physics jobs.
Admission is now used VRAM <=20 GiB, free RAM >=12 GiB, load1 <=26;
`admissions.jsonl` logs every check and admission, including both MemFree
(the enforced RAM floor) and MemAvailable. Old unstarted waiters 227675,
227676, 227677 and their checked flock/sleep children were stopped. No
simulator was running for HANG during that switch. The first two admissions
at 13:03:32Z both read 4813 MiB VRAM used, 32.255 GiB free RAM, load1 12.67.
The launchers retain the 8G/600% scopes and disk guard. No light1-4 or camera
lock is used after this capacity update.

Single lever screen, PR policy source `8ff324fd`, original dev standard
100002, `hang_handle_grasp=on`: **40**, the same as baseline, at 359 steps.
All three selected grasps were the original wall pinches; the added handle
pinches were not chosen by the full transfer planner. This is a null screen,
not evidence that the bar pinches hold in physics. Paired B remains the two
switches together; no policies are changed after the source freeze.

Standard dev10 manifest is frozen at `~/dojo-data/hang/dev10/hang_mugs/manifest.json`:
variant 2 ids 100030-39. The standard A0 chunk runs while random settling
finishes; each cell's manifest precedes either of its arms. Standard chunk
order is A0,B0,B1,A1; random order B0,A0,A1,B1. Both use the same pair source
and each chunk contains five distinct layouts. Workers 240158 / 240159
schedule the remaining jobs only through the admission-logged launcher.

Before any paired B result, the camera prerequisite is made quantitative:
G3 must gain >=5 score points in each cell with no full-success decrease,
in addition to lead review. This is a dispatch prerequisite, not camera
adoption; the camera default rule remains unchanged. Implementation-source
CI passed (1842 tests, 187 skipped locally); latest audit/docs CI is separate.

## Independent carry-rate probe

The first new grasp screen selected wall pinches, and generated A/B rows still
show in-hand/carry failures. A third independent default-off switch,
`hang_carry_slow`, halves only the threaded-hook pre-hover carry rate from
0.06 to 0.03 rad/step. It does not alter the pick, slide, release or step
budget, and does not apply to cradles. Hypothesis: lower angular motion rates
reduce losses while a wall-held cup turns. It may instead consume budget or
leave contact/reach failures unchanged. Original dev standard 100002 is the
first same-source off/on screen. A separate immutable source snapshot will
keep it independent of the ten-layout bundle pair at `8ff324fd`.

## Carry-rate screen result

Pace source `eac4a8b1de9630fac5c0b982049b8ac1c86b3081`, stored under
`~/controlr-hang/pace`, was verified and stamped before either screen was
admitted. Both arms use that same snapshot, with only `hang_carry_slow` on
in B. Original dev standard 100002: A **40**, 359 steps; B **40**, 455 steps.
Both lose the third cup during carry (in-hand shifts 83.9 / 78.5 mm at the
failed place). The screen gives no score gain and costs 96 steps. It is a
single-layout negative screen, not a population estimate; park the probe.
The two-switch geometry pair remains unchanged at `8ff324fd`.

The added positive cradle test initially failed because its stub also sees
relay calls with no place options; the test-only fixture was corrected.
Final full local suite: **1848 passed / 187 skipped**, 18m59s. Latest PR CI
passes. No simulator result used the transient failing test fixture; policy
code is identical between the pace snapshot and its test-only descendants.

Another planning landmine: a hook can remain reserved after a failed lift
or a second loose-pick put-down, although the object never reached it. That
reduces later objects' hook choices. No gain is attributed to fixing this:
it is recorded in BACKLOG as a separate future lever, not folded into the
frozen pair.

## Final paired result and stop

Both generated panels finished: twenty stable draws per cell, with the first
ten frozen before their arms (variant 2, ids 100030–39). Twenty layout pairs,
forty paired episodes, all usable. The same-source geometry bundle loses:

| Cell | A mean ± sample SD; successes | B mean ± sample SD; successes | Paired delta | Score W/L/tie |
|---|---|---|---:|---|
| hang_mugs | 67.5 ±35.06; 5/10 | 50.5 ±35.86; 3/10 | -17 | 0/3/7 |
| hang_mugs_random | 26.5 ±30.65; 1/10 | 25 ±31.62; 1/10 | -1.5 | 0/1/9 |

Standard A/B failed lifts rise 1→3; random 1→4. Random loose picks rise 7→11
and carry losses 1→3. Standard 100037/39 bar pinches fail the lift with
38/33 mm drift; later assignments change. Random's nine mug14 instances
in six layouts retain the documented pose/checker limitation; no exception
was added. No arm exhausts its budget. No-plan exits are 1/2 standard and
6/5 random, combining grasp, transfer and hook attainability rather than
claiming every hook physically unreachable. The initial standard release/exit
failure remains distinct from carry loss.

A same-source diagnostic on random 100031 gives A 15, half-turn only 15,
bundle 0. Removing bar grasps restores A on this one layout conditional on
half-turn; this is not a ten-layout qualification of either lever. Pace is
also parked after its null same-source screen. The G3 dispatch threshold
(+5 in each cell without success loss) fails. Stop here under the user’s
early-stop instruction; no camera batch or adoption claim.

The [final experiment report](../experiments/2026-11-06-hang-skill-ceiling.md)
and [result manifest](../experiments/2026-11-06-hang-g3-results.json) retain
every paired score, phase diagnostic, source/config/layout receipt, 86 asset
hashes, and all 13 admissions. Bootstrap intervals are descriptive. Host
load varies during runs despite admission bounds; wall times are not a
controlled speed comparison. The [camera registration](../experiments/2026-11-06-hang-camera-ab-spec.md)
preserves the default rule and remains deferred.

Total: 54 recorded G3 episodes (10 initial, 40 paired, 4 screens), one
interrupted initial partial attempt with no outcome, and 40 settling draws.
Recorded episode/load plus settling time is 3.71 lane-hours, excluding startup,
queue and the partial attempt. LLM calls/tokens/spend: 0/0/$0. No camera runs
or rendered videos. All 13 dedicated-lock admissions meet the updated limits:
max VRAM 18033 MiB, min free RAM 13.915 GiB, max load1 25.69; 175 waiting
checks retained. All HANG workers and simulators exited; light5/light6 are
unheld. Final local validation: 1848 passed/187 skipped, focused 29 passed;
final-head CI is checked before handoff. PR #400 stays open for lead review.
