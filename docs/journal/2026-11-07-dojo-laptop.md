# Laptop stream: headset seating

Read CLAUDE, DEVELOPMENT, ARCHITECTURE, EXPERIMENTS; BACKLOG laptop/headset
items; October 29 hang/laptop/overhang journal; 62007227; PR #350; branch
histories (`origin/feat/dojo-gennew-laptop` predates the store module,
`wip/store-laptop` retains the closed-slab approach); November 5 points table;
and both compute3 driver/render incidents.

The live task checker sources confirm headset-only 20 points, laptop 80 as
close + upright + bottom in rack. Close alone cannot score. The existing
store module's opening prose incorrectly suggests three independent tiers.
Prioritise the headset. A default-off RuntimeConfig / Skills / G3 switch
`headset_seat=on` filters slipping narrow cradle grasps, squeezes for eight
steps, and seats the measured bridge before release. No task-name dispatch,
checker access, rendering or LLM calls in the action path.

Pre-registration: [panel](../experiments/2026-11-07-laptop-g3.md).
Private salted variant-11 records, pilots separate from panel. Deployment
`~/controlr-laptop`, all outputs `~/dojo-data/laptop`. Light7 serial jobs,
8G/600% user scopes, admission logs, timeout with KILL tail. Root service
and colleague processes untouched. Local pyflakes reports only pre-existing
unused imports and a pre-existing placeholder-free f-string.

The first local pytest invocation used a global pytest because dev extras
were missing; `uv sync --extra dev` corrected the environment. Focused
headset/laptop tests pass. Full suite started with `uv run pytest -q`.

The first candidate's 41 mm pinch holds pilot 100031 at lift (3.7 mm drift)
but shifts 113.4 mm in carry. The pre-panel amendment halves the cradle carry
rate; it still shifts 97.6 mm in that pilot and scores zero. Candidate frozen
at deployed `58dcf38e` for the panel; later changes are tests/docs only.
The G3 deployment stamp is explicitly set before the final pilot/panel;
initial pilots have null G3 `code.commit` (deploy.sh writes `.deployed.json`,
whereas G3 reads `.deployed_commit`). No ambiguous pilot is in the panel.

Local full suite: 2057 passed / 191 skipped (1089.38 s), started on the first
candidate before the carry amendment. Final candidate/config/defaults-focused
suite: 10 passed. Final source also receives PR CI validation. Main was merged
for current documentation; the running physics deployment remained frozen.

Completed panel: ten pairs per cell, A 0 ± 0 -> B 2 ± 6.32 points in each;
1 score win / 0 losses / 9 ties each, full success 0/10 in every arm.
No infrastructure failures or missing meshes. No camera qualification (pooled
+2 < +5), no camera spec/dispatch or board update. Own scopes/processes exited.
Footprint 130 MB deployment / 2.6 MB data; zero model calls/tokens/spend.

## Round 2: contact during the turn

Lead requested sustained development and two LAPTOP jobs (light7/light8),
with the same box gates and zero LLM spend. Numeric round-1 pilot audit:
relative position remains within ~1 mm through 58 degrees of TCP turn.
Separation from sampled stand mesh falls to ~5 mm at step 95; the headset
then stays nearly stationary against the stand while TCP-relative drift grows
past 90 mm. A stationary-pivot fit leaves 44 mm maximum residual; this is
not explained by a simple free pendulum. The leading mechanism is premature
stand contact during wrist reorientation, supported by the staged-route probe.

Default-off `headset_carry=staged` separates turn and approach. Standard
100031's carry drift falls 97.6 -> 1.2 mm and earns 20 points. Random 100030/31
also score 20 (20.0/5.2 mm carry shift). Standard 100030 remains loose at lift.
High carries, thin bridge pinches, cup-site ranking, loose continuation and
excluding failed pinch sites were probed on pilots only; none improved the
frozen bundle. The weak loose-hold continuation is explicitly not implemented.

The laptop physical closure probe shuts to 0.99 but slides the slab off its
stand onto the table. Opposite-side tray targets then have left reach failure
and right table collision. Closed geometry ~179 x 126 x 10 mm; stand top
sample bbox ~305 x 119 mm. Authored laptop mass is **0.1 kg**, friction **0.45**,
so the old journal's informal ~1 kg-class description is not an established
physics fact. Rack endpoint IK/native collision results are reported separately.

Fresh variant-12 salted ids 100070–84 are generated and settled independently
per cell; selection/hashes and final candidate are pre-registered in the
[round-2 report](../experiments/2026-11-07-laptop-g3-round2.md). Existing panel
records are never used for tuning. No camera/board changes or external work
are touched. Full local suite runs concurrently with settling after source freeze.

Round-2 staged panel (source 43d5882b, fresh variant-12 ids 100070–79):
standard 4 ±8.43 -> 12 ±10.33 points (4 wins / 0 losses / 6 ties), random
2 ±6.32 -> 14 ±9.66 (6/0/4). Headset credits A/B 2/10->6/10 and 1/10->7/10;
full-task success 0 in every arm. All 40 valid, no mesh/infra exclusions.
This passes the score gain gate, but misses the lead's standard 7/10 target.
Full suite on staged source: 2061 passed / 191 skipped, 987.22 s.

Returned only to the original loose-hold pilot (not panel failures). A denser
opposed-face/approach bank finds a 44 mm pinch with 3.0 mm lift drift and
1.3 mm carry shift, rescuing standard 100030. The standard six approach angles
miss zero tilt; thirteen include it. New default-off `headset_grasp=dense`
samples 400 faces and thirteen approaches, retains box/slice proposals at
>=35 mm and screens 24 full hangs. Original lift/loose checks remain. Fresh
round-2b panel registration uses new salt, variant 13 and ids 100050–64.
Another stream owns light7 now; only our waiting wrapper PID was stopped,
and subsequent work uses light2/light8. No other process is signalled.

The dense bank scores 4/4 original pilots before the second panel. Lift drift
is 3.0/6.0 mm on standard 100030/31 and 5.3/5.4 mm on random; carry drift
1.3/3.3 and 4.9/2.5 mm. This is pilot development, not a held-out claim.
The laptop rack's four right-arm endpoints also pass the rack/scene check;
thus the blockers are acquiring/supporting the slab and a complete route,
not a universal upright-pose IK failure. An unbraced horizontal push is
predicted to slide at ~0.44 N before tipping at ~6–10 N; a fixture changes
that assumption and is not ruled out. No physical laptop hold was demonstrated.

Production wiring reproduces all four dense-bank pilot credits before panel
launch. Continuous post-lift to pre-place audits have maximum displacement
1.66/3.36 mm (standard) and 6.79/2.50 mm (random), not just a small final
residual after a large swing. These traces are retained under the private
production-pilot directories. The second fresh panel starts only afterward.
Final-source local suite: **2063 passed / 191 skipped**, 1041.26 s. Source
473d64cf also has green GitHub CI; subsequent edits are documentation/data only.

Dense fresh panel source 473d64cf: standard A0 ±0 -> B12 ±10.33, 6 wins /
0 losses /4 ties; random A2 ±6.32 -> B18 ±6.32, 8/0/2. Headset credits
0/10->6/10 and 1/10->9/10; all full-task successes zero. Forty valid rows,
no exclusions/mesh/infra failures. This misses standard reliability despite
passing the paired score gate. Continue on new pilot ids 100065–69 with
variant 14 and an independent salt; no individual panel failures are opened
for tuning. Next panel is reserved for previously unused 100085–99.

New pilots baseline dense+staged: standard 2/5, random 4/5. Standard 100067/69
never pick; increasing full-plan limit to 96 still produces zero, with 403/389
pick-plan calls. Instrumented standard 100067/68 shows 119/120 false held-table
lift refusals respectively. The guard uses held OBB corners even for the sparse
headset. At a refused first lift waypoint, OBB min z=0.75929 m while full mesh
min z=0.78473 m; table z=0.765 m. No wrist-stop cause: commands <=0.83 rad,
tracking <=0.08 rad on these pilots.

Prototype full-mesh table checks rescue both 100067/68 to headset credit.
Pilot 67 lifts with 5.1 mm drift / carries with 4.4 mm shift; pilot 68 retains
the first loose rejection, then re-grasps with 2.7 mm drift / 2.6 mm carry.
New default-off `headset_table=mesh` scopes the guard to cradle hangs. All mesh
vertices, unchanged tolerance, conservative XY overlap test; missing/invalid
mesh falls back to bbox. Arm/finger checks remain; scope restores on exceptions.
Unit coverage includes real penetration, triangle crossings with no inside
vertex, unsampled low vertices, disjoint footprint, invalid mesh, hand guard,
scope cleanup, and unchanged defaults. Wider search is not implemented.

Mesh-table fresh panel source 18e70412: standard A0 ±0 -> B18 ±6.32,
9/0/1 W/L/T; random A4 ±8.43 -> B10 ±10.54, 3/0/7. Headset credits
0->9 and 2->5 of ten. All full-task successes zero; 40 valid rows, no
exclusions. Random reliability fails; continue on independent pilot 100066,
not individual panel failures. Full local suite 2071 passed /191 skipped,
934.65 s. No model calls, camera dispatch or board changes.

Independent random-100066 cup diagnosis: 16 physical sites total (eight per arm) screened,
reset each time. Two right 25 mm cup pinches remain suspended through 30 steps
(patch 1.0, one clean 6.2 mm lift drift), but right cannot reach the cradle;
left pickups reach commanded poses yet slip. The 13 mm bridge pinch falls
with jaws closing to zero: stationary pinch-site error ~10->82 mm, rather
than a maintained pendulum. No verifier exemption is adopted.

A checked centre table rest lets right pick/put down, then left re-grasp and
hang. Prototype earns 20 in 412 steps with 0.3 mm staging carry shift and
0.8 mm final carry shift. An upright rest also works physically but is not
included. One wrapper-registration bug is retained as a direct-hang null,
not a staging result. New default-off `headset_stage=center` scopes staging
to >0.40 m transfers whose pickup/cradle favor different arms. Only verified
stages enable narrow broad-patch cup proposals for hanging; ordinary retry,
loose-hold and method-slip rules remain. Runtime/Skills/G3 mirror the field.

Production source 7faef88e screens 7/7 standard and 7/7 random on original
30/31 plus independent 65–69, including random66 in 362 steps. Focused suite
27 passed; full local suite running. Freeze and register variant-16 fresh
ids 100015–29 (first ten stable), new salt, ten paired layouts per cell.
All earlier panels remain separate, no selected pooling or camera dispatch.

Frozen 7faef88e full local suite completes: **2094 passed /191 skipped**,
974.86 s. The first ten stable fresh records per cell are selected and hashed
before any panel outcome: ids 100015–24. Both arms use exact settled files;
paired source/config remain frozen, no laptop route, camera or model calls.

Centre-stage fresh panel source7faef88e: standard A0±0 ->B14±9.66,
7/0/3 W/L/T; random A0±0 ->B10±10.54,5/0/5. Headset credits0->7 and0->5
of ten, full successes zero. Forty valid rows, no exclusions; random misses
reliability. Continue on new independent45–49 variant17 pilots, not individual
panel failures. Their random shapes8/9/10 and standard2/4/5 expose coverage
missing from the earlier seven all-asset8 random pilots.

Measured-joint continuous audit of production random66: max attachment drift
1.97 mm staging,2.74 mm final carry through99.2-degree turn. A transient j1
command/measurement gap13.9 degrees (-1.745/-1.502 rad) remains; do not claim
ideal kinematic tracking. Original lift/loose verification stays unchanged.

Independent shape pilots45–49 expose narrow-opening contact: standard48's
TCP reaches0.03 mm/0.004 degrees yet the object rises15.9 mm before closure.
Full88 mm opening changes its first lift drift14.9→6.6 mm and staging carry
shift to1.1 mm. Full-opening plus a true mesh-support rest rescues random47
and48, while standard48/random49 still fail. A later clean standard48 pickup
cannot re-plan its measured place; nominal fallback loses258 mm. Opt-in sites
now puts that hold down rather than carrying an unchecked nominal attachment.

The heavy headset can also be hung via a genuinely different mechanism:
checked upright table rest, downward bridge pinch (3.3 mm pickup drift),
fixed-orientation carry, release above the cradle and30 support steps. Its
horizontal bridge pinch was loose18.1 mm then slipped186 mm. This is a pilot
success only: standard45 fails static upright-rest stability before moving;
random49's upright rest misses19.8 mm and9.3 degrees. Keep the bridge recipe
as feasibility evidence, outside production.

The full-opening/rest source bf6ff63c regresses old-success standard45 and
random45 at pickup, despite rescuing random47/48. Stop additional dispatch
from its12/cell development drivers, keeping in-flight/completed rows. The
continuous random45 trace separates6.4 mm closure movement from5.1 mm load-
onset movement/4.37-degree rotation; first lift TCP step17.2 mm. A slower
0.03 rad/step lift rescues the same pilot after rejecting its8.3 mm first
pickup, then a7.5 mm pick and6.1 mm carry shift earn20. Add a separate opt-in
headset_lift=slow axis, mirrored in RuntimeConfig/Skills/G3; lift height,
closure, physical drift thresholds and all original defaults are unchanged.
