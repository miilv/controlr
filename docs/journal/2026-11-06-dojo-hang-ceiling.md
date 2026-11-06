# 2026-11-06 — HANG: handle grasp and hook attainability

Investigation in progress; no adoption claim. Lead review requested; do not merge.

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
No other stream's process was signalled. Subsequent HANG jobs use light1 only,
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
first ten stable draws per cell will be frozen in `dev10-manifest.json` before
either paired arm. Random settling and the pair are pending shared-slot access.
To prevent polling starvation, checked unstarted HANG waiters 219200, 224477
and 222924 were replaced by blocking single-lock waiters. Remaining pipeline
PID 227675; screen waiter PID 227676. The launcher also waits if two other
module-based G3/settling jobs are active. Never more than one HANG simulator
after the initial correction. No other stream's process was signalled.

Rebased local full suite: **1842 passed / 187 skipped**, 17m56s. The new
retry-deduplication audit test passes separately. Camera specification fixes
fresh owned-namespace variants 21/22 with stability-only reserves 20/19;
no camera data or render preparation has been collected for the spec.
