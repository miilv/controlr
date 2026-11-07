# Laptop G3 round 2e — independent failure diagnosis

The frozen centre-stage panel has random B at 1/5 completed credits, so its
maximum is 6/10 and it cannot meet the 7/10-per-cell stopping rule. Finish and
archive that panel without changing source or inspecting individual panel
failures for development. Standard B is 5/8 at this decision point.

Generate a new independent pilot bank: ids **100045–49**, v15 variant 17,
salt `laptop-round2e-pilots-2026-11-07-88c48b74`, both cells. These ids have not
been generated/run by this stream. Settle once and retain all stable records.
Run frozen source **7faef88e**, seat/staged/dense/mesh/center bundle, on all five
pilots per cell before choosing another candidate. The seven earlier pilots
per cell all passed; they do not cover the current fresh-panel failure rate.

Use measured traces to distinguish staging refusal, physical pickup failure,
loose hold, carry contact and final support. Hypotheses, not selected fixes:
(1) narrow cup proposals should be a fallback when a reliable wide hold fits,
rather than displacing it after every staging move; (2) a failed physical
pinch should exclude a site rather than the entire antipodal method; (3) a
checked staging pose should be chosen for a demonstrated re-grasp geometry,
not solely table free-space. No relaxed physical verifier, joint-limit or
collision exemptions. Any change must have pilot evidence and unit tests.

No new panel is dispatched by this registration. A later candidate/source
and new salt must be registered before a new panel; ids **100000–14** are still
unused by this stream and reserved for that purpose (first ten stable,
reserve only for stability/infra, not policy scores). Keep all prior panels
separate, no selected pooling. Two concurrent physics-only LAPTOP jobs maximum
on light2/light8, unchanged who/GPU/disk admission and 8G/600% scopes. Zero
LLM calls/tokens/spend. No camera dispatch or board change.

## Coverage finding before candidate selection

The earlier seven random development records all contain headset asset 00008.
This new bank contains 00008 (45/46), 00010 (47/48) and 00009 (49). Standard
contains 00004 (45/46/49), 00005 (47), 00002 (48). This is a development coverage
gap, not evidence that any particular unopened panel failure has that cause.
Public metadata: 8/9/10 all mass 0.22 kg; friction 0.32/0.45/0.45; depth
72.28/80.64/78.30 mm; bridge local z 85/80/90 mm. Do not infer grip-force limits
from mass/friction alone. Complete the baseline and use actual per-shape lift,
carry and support measurements to select the next mechanism.

## Selected pilot prototype before new actions

Frozen baseline on all ten new pilots: **standard4/5, random2/5**. The failures
are standard48 (asset2), random47/48 (asset10), random49 (asset9).
Standard48: right26 mm cup pinch has patch1.0/depth and quality0.17 yet is
loose14.9 mm; subsequent left39 mm picks rise3/80 mm. Random47 stages with a
35 mm right pinch (4.9 mm lift drift), then left35 mm rises39/80 mm with40 mm
drift; the existing whole-method rejection discards every untried antipodal
site. A no-action-change repeat's stage lands17.6 mm off and remains a
place_miss under the original15 mm verifier. Random48 has only16 wide grasps,
one endpoint pair and no full approach. Random49's80 mm pinch drifts37 mm;
no alternative antipodal site is tried. These are pilot observations only.

Prototype two default-off switches: `headset_grasp=cups` adds the existing
geometry-verified18–35 mm broad/deep cup proposals to the wide bank even when
no stage succeeded; all IK/collision and physical verification remain. It
must not mark a failed stage as successful or infer a hold. `headset_retry=sites`
remembers failed **object-frame site + approach + closing axis per arm**, not
an entire method, and allows up to four physical staging attempts and four
hang attempts, each checked anew from the measured state. Candidate sites
within15 mm, closing axes within20 deg (finger flips equivalent) and approaches
within30 deg of a failed site are excluded. Existing800-step/reserve bounds
remain; no retry after budget/end or a placement miss, and no loosened hold,
place or table tolerance. Legacy modes keep their exact prior budgets/filters.

First screen the four failed pilot records, then all new pilots plus the
original seven/cell if supported. Any final bundle/source and fresh panel
must be registered separately before generation. No camera or model calls.

## Full-opening enclosure probe before new actions

Source d8387f61 first probes remain zero: standard48 repeats26 mm loose grips
then a bad third site knocks the object off the table; random47 tries18/19 mm
cup sites, all loose/slipped. Do not interpret more attempts as a solution.
The original standard48 diagnostic shows an earlier mechanism: while g=0.629
(~47.9 mm commanded open gap), descent moves the object from z0.8056 to0.8215
before closure. TCP target error is only0.03 mm/0.004 deg; this is approach
contact, not failed IK. During closure the relative root moves another11 mm.
The nominal26 mm visual-mesh slice is insufficient opening clearance for this
cup's physical envelope (asset depth88.78 mm; exact collider section not measured).

Probe **full88 mm opening** for narrow cup proposals, via existing Grasp.clear_w.
Keep the same TCP, closure, native finger/table/scene validators and original
physical verifier. A too-low/wide pose must be refused, never exempted. Test
standard48 and random47 first; screen the other failures only if supported.
This is an enclosure-clearance test, not a larger hold/slip tolerance. Source
remains d8387f61 plus a retained probe recipe until production proof.

## Enclosure and actual-support rest extension

The full-opening standard48 probe changes right26 mm pickup drift14.9→6.6 mm
(clean), followed by1.1 mm staging carry shift. Left26 mm remains loose8.4 mm;
a subsequent41 mm grasp opened by its nominal width is loose23.9 mm. Extend
full88 mm opening to the complete cup-mode bank, retaining all native checks.

Also probe an actual-mesh support target for the staging leg only. The current
bbox rest leaves sparse-headset mesh15–23 mm over the table; random47's release
falls15.5 mm and tilts11.5 deg (another repeat misses xy by17.6 mm). Set the
desired root z from the minimum of all transformed mesh vertices plus3 mm
clearance, rather than empty bbox corners; retain the original release gap,
full-mesh table tolerance and hand/scene validators. Missing/invalid geometry
retains the old target. This is a planned pose, never a forced scene pose.
Compare full-bank opening alone on standard48 and full-bank opening + mesh rest
on random47; then cross-check any physical improvement. No verifier relaxation.

## Checked measured-attachment re-plan requirement

Full-opening standard48 eventually lifts a38 mm pinch cleanly (6.8 mm drift),
but its measured-attachment place re-plan fails. The existing nominal-plan
fallback then carries and loses258 mm of attachment, missing by202 mm xy.
For `headset_retry=sites` only, refuse that fallback: put the verified hold
back down through the existing re-grasp path, record the executed pickup site
as unusable for this transfer, and try another checked site within the same
four-attempt/800-step bounds. Legacy retry keeps its existing fallback.
No failed re-plan may be reclassified as a valid carry. Preserve failed
put-down/budget/end status rather than pretending the gripper is empty.

## Gravity-aligned bridge-carry feasibility probe

If the remaining heavy/cup cases still fail, test a distinct mechanism on
pilots only: checked upright centre rest, then a front/back bridge pinch and
carry without the90-degree object turn. An earlier asset8 upright-rest pilot
was physically stable (0.6 mm xy,-0.5 mm z,1.6 deg tilt through30 steps).
Require the existing static stability screen; retain native checked staging,
full opening and physical verification. Generate bridge-centre/nearby grasps
with mesh opposition, approaches70/80/90 deg from down, closing across bridge
depth, and no forced poses. Native IK/collision selects the arm/approach.

The stand's top25 mm mesh spans ~105 x75 mm. To keep band-grasp fingers above
it, carry to bridge point ~60 mm above its functional frame, release40 mm above
the frame (~25 mm above the physical top), then hold30 steps before grading.
A prior numeric IK screen finds left80–90 deg bridge approaches reachable at
that height; vertical approaches are not. This is a genuine gravity seating
attempt: require final physical support, stable orientation and open hands,
not transient proximity while falling. Do not change checker thresholds.
First probe standard48, then random47 if supported. No new panel/source freeze.

## Upright release-height compatibility probe

The upright-table screen has34–36 endpoint pairs but its first24 full plans
cannot pick; the original clean cup grasp's upright endpoint puts the palm
near the table. Probe a50 mm release gap over the true upright rest, using
the release-height pose for endpoint screening and rebuilding the checked
place plan against the actual ground rest with `release_gap=.05`. This
preserves the original place verifier's ground target; no widened tolerance
or forced pose. Full pick, measured re-plan and place paths remain checked.
Then allow30 stationary steps for a genuine upright rest before the bridge
pickup. Restrict the initial feasibility test to two centre spots rather than
repeating the same refused grasp list over a large grid. If this does not
physically rest upright, stop this mechanism and retain the null.

To separate endpoint-screen coverage from physical feasibility, also run one
cheap standard48 snippet using its previously demonstrated clean right26 mm
object-frame grasp (not an id-specific production rule). Plan/check that pick,
then plan the upright ground rest directly from the measured hold with50 mm
release gap, retaining the ground target for verification. Stop if either
checked plan or physical hold fails. If upright after30 steps, test the same
mesh-verified bridge bank/raised release above. This isolates actual physics
from a large coupled-plan search; retain the exact known grasp in the recipe.

## Mesh-derived gravity-balanced bridge site

The upright-release standard48 prototype physically rests upright:10.6 mm xy,
-1.1 mm z,4.7 deg tilt after30 steps. Its first off-centre(-45 mm) bridge grip
is loose13.6 mm; another shoulder grip slips. Measured arm tracking in these
bridge picks is <=0.7 deg, not a joint-stop explanation.

The initial band sampler used feature z85 mm and lower. Actual central-band
vertices span z79.5–99 mm; opposed-face rays at z85 mm all miss, at89 mm find
31.5 mm only for a horizontal approach, and at97 mm find42.7–44.2 mm. Probe
mesh-derived upper-band levels and the horizontal site above the mesh-estimated
COM (minimising gravity moment about the closing axis), retaining nearby
alternatives. Carry with the current upright orientation to the cradle, rather
than correcting the4.7-degree rest tilt; this stays within the task's alignment
and avoids introducing another wrist torque. All native checks, original pick
verification and30-step released-support proof remain. No scene pose changes.

One final bridge feasibility variant approaches from above (35/50/65 deg
from down) instead of tangent-horizontal, to avoid pushing an upright headset
sideways before enclosure. Rotate the closing axis0/15/25 deg within the band
plane, verify opposed mesh faces again, and recompute the gravity-balanced site
for each axis. This trades a small face angle for a reachable downward approach;
native IK/collision decides feasibility. Keep full opening, original hold
verification, current-orientation carry and released-support proof. If this
variant also fails, park the upright-bridge mechanism and proceed with the
physically supported enclosure/rest improvements rather than adding more
unregistered variants. No final-panel record is involved.

## Downward bridge result and cross-shape check

The final tilt variant succeeds on standard48: 20 points, bridge pickup3.3 mm
drift, after an upright rest. The earlier balanced horizontal grip was loose
18.1 mm, followed by a186 mm slip. Before any production integration, cross-check
the unchanged tilt recipe on standard45 (a lighter shape) and random49 (asset9).
These are pilot records, not final-panel observations.

## Production enclosure/rest screen registration

Freeze the implementation of full-opening `cups`, `mesh_center` support
height, and `sites` retry with refusal of an unchecked measured-attachment
place fallback. Screen the original seven pilots30,31,65–69 plus new45–49
per cell (12/cell), all fixed settled files. Bundle: straddle/on/staged/cups/
mesh/mesh_center/sites. No bridge prototype or changed hold tolerances.
Record every outcome, including regressed old successes; this screen is
development, not a fresh paired estimate. Final source and fresh-panel
registration remain separate. All switches retain their original defaults.

Bridge cross-check standard45 fails the native static stability predicate
before any action. The heavy standard48 success is feasibility evidence,
not grounds for a universal bridge strategy. Another original cup diagnostic
measures closing-axis shift9.0 mm and perpendicular shift13.7 mm after lift;
a closing-axis recenter alone would not explain that entire failure.

## Numerical development record

These are repeated development records; they do not estimate fresh reliability.
All use physics-only G3 with original physical-verification thresholds.

| Prototype | Standard48 | Random47 | Random48 | Random49 |
|---|---:|---:|---:|---:|
| Frozen centre-stage baseline | 0 | 0 | 0 | 0 |
| Cup bank + site retries, nominal opening | 0 | 0 | 0 | 0 |
| Full opening, narrow proposals | 0 | 20 | unrun | unrun |
| Full opening, all cup-mode proposals + mesh rest | 0 | 20 | 20 | 0 |

Standard48's first right pickup improves14.9→6.6 mm lift drift with full
opening, then carries to the centre with1.1 mm attachment shift. The old
opening disturbed the object15.9 mm upward before closure although TCP error
was0.03 mm/0.004 degrees. This is physical pre-contact clearance, not failed
IK or evidence that the8 mm verifier should be relaxed. Its later left38 mm
pick is clean2.3–6.8 mm, but measured place re-planning fails; the nominal
fallback then loses258 mm. The production `sites` guard refuses that carry.

Mesh-ground staging on random47 lands5.2 mm off in xy,+1.3 mm z and6.3
degrees tilt. The original bbox release was17.6 mm off,−12.7 mm z and15.3
degrees. A subsequent18 mm left grasp lifts clean7.6 mm and genuinely hangs
13 mm from the cradle frame. Its23.6 mm carry shift is compensated by the
existing measured-place re-plan; it must not be described as a rigid hold.
Random48 lifts7.3 mm drift directly and earns20 without a staging transfer.
Random49 still fails: first30 mm pickup loose8.5 mm, then wider holds slip.

The upright-bridge feasibility variant is separate from production:
standard48 upright rest10.6 mm xy,−1.1 mm z,4.7 degrees; downward bridge
pick3.3 mm drift and20 points after30 released-support steps. Horizontal
bridge pick18.1 mm loose, next site186 mm slip. Cross-shape standard45
fails static rest stability without moving; random49 lifts7.8 mm and carries
with1.8 mm shift but misses its rest by19.8 mm/9.3 degrees, ending at0.
Thus one successful upright-bridge mechanism exists, with no reliable
generic rest/re-grasp policy yet. It is not included in the panel candidate.

Production source **bf6ff63c** implements the registered enclosure/rest
bundle. Focused CPU tests:31 passed. The full suite and12/cell production
screen are running. The earlier source d8387f61 passed2101 tests with191
skips in1093.08 s; that result does not validate the added source by itself.

## Lift-speed probe registered from continuous pilot evidence

Production random45 regresses to0: four loose9.5/12.8/9.2/13.9 mm pickups.
Its first grip shifts6.4 mm during closure, then5.1 mm during the lift, with
4.37 degrees rotation. The first lift step moves TCP17.2 mm; object rise11.4
mm, while the last three lift steps hold their relative pose much better.
This is actual load-onset motion in the jaws, not just the wrist endpoint.
Before another source change, test a slower checked lift (`lift_rad=.03`
instead of the original default) on random45 and standard48 pilots. Keep full
opening, mesh rest, original8/30 mm drift thresholds and measured-plan refusal.
No loaded-hold exemption or changed checker. Preserve the current12/cell
production screen unchanged; these prototype jobs queue behind each lock's
current episode, with no more than two actual scopes.

The slower random45 probe earns20 after rejecting an8.3 mm first pickup;
the next is clean7.5 mm and carries with6.1 mm shift. Before additional
production-screen dispatch, stop the old12/cell parent drivers (only their
verified own pidfiles), allow in-flight episodes to finish and retain all
completed rows. The candidate already has two regressions; finishing every
old pilot cannot erase them. This truncated development screen is not a
paired panel and will be reported at actual n, not the originally planned12.
Cross-check the same slow recipe on standard45 and random49 in addition to
standard48. No final layout is opened/generated under this amendment.

## Broader independent development coverage registration

Before the last fresh paired panel, generate an additional independent pilot
bank under `~/dojo-data/laptop/coverage18`: numeric training ids100030–44,
v15 variant18, salt `laptop-round2e-coverage18-2026-11-07-88c48b74`, both
cells. These reuse numeric pilot ids in a separate namespace with new draws;
they are **not** fresh never-used ids and must never be presented as the final
paired panel. Do not overwrite any prior raw/settled record. Select first ten
stable ascending per cell, with later records only stability reserves. Preserve
all raw files, settled index and selected hashes before policy outcomes.

Source **fb39b381** (includes current main) mirrors the opt-in slower lift.
Condition further production screening on the cross-shape slow probes before
spending the broader bank: straddle/on/staged/cups/mesh/mesh_center/sites plus
headset_lift=slow, unchanged800-step limit, full native checks and physical
verification. Screen the original12 pilots/cell and this10/cell independent
bank; report them separately at actual n. Development outcomes may diagnose
and improve the mechanism, but final ids100000–14 remain untouched.

No final-panel generation or paired candidate selection is authorized by
this development registration. Same two light slots, scopes and admission
gates; zero model calls and no camera jobs/board change.

Cross-shape slow results: standard45 now earns20 with right7.3 mm and left5.6
mm pickup drift, then0.6/2.5 mm carry shifts. Heavy standard48 stays0 (first
slow pickup8.8 mm loose); it also scored0 with the frozen centre-stage baseline.
Integration source **fc90c405** merges current main's unrelated opt-in pour
code; the headset implementation is identical to fb39b381. Use fc90c405 for
the registered production/coverage screen after all older-source jobs exit.
Both observed enclosure/rest regressions have been rescued by slowing lift,
without changing the verifiers. Random49's slow cross-check remains pending.

The superseded production screen completes at **standard2/4** and
**random3/5** (ids45–48 /45–49 respectively); all full-task successes0.
Additional ids were not dispatched after the parent drivers stopped. Keep
the original12/cell intent above and this actual-n amendment together.
The added bank's first ten random draws all use asset8; its independent
positions do not close the rare-shape gap. Existing new45–49 pilots retain
assets9/10 coverage. Do not replace the predeclared first ten for better
policy outcomes or describe them as shape-stratified sampling.

The random settling batch completes15/15 stable. Its freed light8 slot is
used for standard settling: cancel only the verified own queued light2 bash
wrapper/flock child, record their pidfile/commands and requeue the same
untouched standard bank on light8. No active runner or other stream is stopped.

Slow random49 completes at0:30 mm pickups loose8.3/8.9 mm,80 mm pickup47 mm
slip. Both added banks settle15/15. The fb39b381 full suite finds one default
contract regression: legacy hang passes an empty `pick_kw` dict instead of
omitting it (2136 passed/191 skipped, one failure). Preserve the original
omitted keyword when seat/lift are legacy; keep the existing regression
test unchanged. This does not change any seat-on candidate motion. Re-run
the focused suite and full suite against the integrated source before merging.

Freeze production screen action source **504daaef**: integrated main plus
legacy-keyword fix,45 focused tests passed. Source fc90c405 was not deployed.
The original12/cell and first-ten-stable coverage18 screen use source504daaef
with exactly the registered slow enclosure/rest flags, two concurrent slots
and fixed settled files. Full integrated local suite starts before dispatch.
No new fresh-panel id or checker/verifier threshold is changed.

Before any coverage18 policy outcome, set the screening gate for spending
the remaining never-used panel ids: at least9/12 credits in the original
pilot bank and8/10 in coverage18 **in each cell**. Complete both registered
screens and report all outcomes, rather than stopping on an early favourable
count. If this gate fails, diagnose pilots and continue development; do not
reroll the final panel or alter its future selection rule. These are
development gates, not a formal estimate of population reliability.
