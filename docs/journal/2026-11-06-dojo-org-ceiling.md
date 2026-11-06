# ORG: organize-table reach and contact approaches

Development study, no camera adoption claim. Source base `18d97e56`.
Read CLAUDE, DEVELOPMENT, ARCHITECTURE, EXPERIMENTS including the current
score-based camera rule, prior organize/v15/cover/ordering and A1/bottles
notes, BACKLOG, and EGG #399 / TOOLS #398 / HANG #400 / PEN #397 / PLAN #401.

Compute6 initially exceeded the requested 20 GiB used-VRAM admission threshold
(22078 MiB). The explicitly authorized compute3 fallback is used after reading
the driver incident, checking `who` and GPU users: 3482 MiB used, 54185 MiB
available RAM, load1 0.17. No system changes or other processes are touched.
Private deployment `~/controlr-org`, data `~/dojo-data/org`, at most two
physics-only light jobs (light7 / light2), 8G / 600% scopes. Subsequent wrappers
retain per-start who/GPU checks, <=14000 MiB used VRAM and >=25 GiB available
RAM; disk guard before every launch. Timeouts include a 60-second KILL tail.

Original dev ids 100000-04 use existing settled records copied by explicit id.
Fresh private v15 variant-3 records 100030-44 are generated and presettled,
with no protected ids used. Select the first ten stable generated ids before
candidate outcomes. Initial baseline stamp accidentally says `ebcbca`; actual
source is this checkout's unmodified `18d97e56`, copied before implementation.
Later snapshots are separately deployed so running jobs never see source edits.

Two independent default-off switches:

- `push_approach=tilted`: after the existing straight-down face pushes, try
  65/50-degree approaches with the TCP compensated to preserve fingertip
  contact position. Same face height, stroke length, collision checks and speeds.
  Unlike the old press stroke, this contacts the side rather than dragging on top.
- `arrange_hover=low`: 5 mm pre-height and 3 mm support clearance for arrange
  placements, with existing checked transit and tail. Hypothesis: the final
  pose reaches but the original pre-place is outside IK reach.

Initial baseline 100000: score 0.50, 223/1000 steps. Figurine has two
screened pairs but its full approach target is unreachable. Keyboard fails
both face axes after 50 push steps. Neither is step-budget exhaustion.
Focused tests cover candidate tip geometry and refusal, config validation,
hover forwarding and legacy order. Full suite and physical screens pending.

LLM calls/tokens/spend: 0/0/$0. No camera jobs, compute4, or dc19 access.

## Lateral-route probe

The first five generated pairs tie (mean 45, no successes), with several
keyboard no-plan exits turning into unscored precision misses. No camera gate.
A CPU reconstruction of original 100000, with drawer/stand obstacles and low
hover, rejects the legacy route; wider 55/65-degree grasps increase pairs
12 -> 32 but still find no full route. A bounded lateral waypoint probe does
find a complete checked route for the original slice grasp at z 178 mm,
dir 135, tilt 40. The successful waypoint offset is (+80, -100) mm in world xy.
This is feasibility only, not a physical pick or score gain.

A third independent default-off switch `arrange_route=lateral` forwards
`approach_lateral` to placement. Legacy transit is tried first; only a collision
failure tries four fixed lateral offsets, each with fully checked legs. Failed
IK endpoints remain refused. No partial route executes. The physical pilot
combines this route with the separately tested low hover, without tilted pushes.

## Completed paired result and disposition

[Report](../experiments/2026-11-06-org-skill-ceiling.md) and
[raw records/manifest/admissions](../experiments/2026-11-06-org-g3-results.json):
ten stable variant-3 pairs, **42.5 ±12.08 -> 47.5 ±14.19** (sample SD),
**3 score wins / 1 loss / 6 ties**, 0/10 successes both, no success flips.
Median steps 279 -> 539, maxima 379 -> 679 of 1000. No budget exhaustion.
All 29 recorded G3 episodes have complete meshes and no infrastructure failure.
The gain meets the current score rule exactly; it does not establish a higher
full-task success ceiling. The [parked camera registration](../experiments/2026-11-06-org-camera-prereg.md)
explicitly uses score-pair wins despite its November 6 filename. No camera runs.

The lateral pilot (`low + lateral`, no tilted pushes) picks original 100000's
figurine, then loses it during transfer: 169.2 mm attachment shift, 152.7 mm
final xy error / 82.7 degrees tilt. Score 50, 300 steps. Park this lever; exclude
it from the camera candidate. Merged-main bundle smoke reproduces generated
100030's score 50 and 506 steps exactly, separate from the comparison.

Local full suite on pre-lateral source: 1961 passed / 191 skipped. Latest
route/switch/legacy focused suite: 60 passed. The full suite started before the
lateral additions; final source is covered separately by CI. The earlier suite
was stopped after the main merge by checked own PID; TERM did not stop it, so
that exact old pytest PID was killed. Other worktrees' tests were untouched.
Pyflakes: only existing unused imports / unrelated f-string warning, no new
undefined names. Local footprint ~160 MB. Zero model calls/tokens/spend.

The original baseline stamp error and missing tilted-pilot stamp are called out
in the report; paired rows have the correct immutable source. Main was merged
into this branch to resolve PR CI eligibility; paired deployments stayed frozen.
PR #407 remains open for the lead; do not merge.
