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
judged physically hung by the skill was mug14, which receives no credit under
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
