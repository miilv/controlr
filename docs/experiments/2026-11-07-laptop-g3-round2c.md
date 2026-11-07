# Laptop G3 round 2c development registration

The dense-bank panel B reaches 6/10 standard and 9/10 random, missing the
lead's 7/10-per-cell stopping criterion. Preserve that result; do not tune on
its individual failure records or pool selected panels.

Create a **new pilot bank**, ids 100065–69, v15 variant 14, salt
`laptop-round2c-pilots-2026-11-07-88c48b74`, both cells. These ids were not
previously generated or run by this stream. Settle once and retain all stable
records. First run the frozen dense+staged candidate, source 473d64cf, on all
five pilots per cell. Use numeric traces to distinguish lift loss, contact
while rotating, diagonal approach contact, native joint tracking/limits and
planning refusal. All subsequent tuning stays on this pilot bank and the
original 100030/31 pilots, never panel records.

Predeclared hypotheses: (1) approach the cradle normal to the headset plane
via a clear frontal waypoint after the far rotation, rather than a diagonal
line that can sweep one ear cup into the pole; (2) improve pad insertion depth
by adding geometry-verified deeper pinches; (3) use the existing measured-joint
bounds if traces show stop-limited tracking. Keep loose-hold rejection, native
collision checks and the original retry/step budgets. Only mechanisms supported
by pilot measurements can enter the next bundle.

A later panel must be registered separately after pilot development: new salt
and never-used ids 100085–99, first ten stable ascending per cell, shared settled
records, ten A/B pairs per cell, same score and 7/10-per-cell reliability rules.
No panel is dispatched by this development registration. Two concurrent
physics-only LAPTOP scopes max, light2/light8, 8G/600%, who/GPU admission,
<=24 GiB GPU used and >=10 GB disk. All existing box/path/zero-LLM rules apply.

## Pilot-only amendment before the coverage probe

Frozen dense+staged baseline on the new pilots: standard 2/5, random 4/5.
Standard 100067 has 77 grasp/place IK pairs but no full pick/approach in the
first 24 screened pairs; standard 100069 also never attempts a pick. Standard
100068 lifts loose (16.6 mm drift), regrasp is retained, then no complete plan.
Random 100066 tries 71 mm grasps, rises zero and is rejected. Measured joint
tracking is within 0.08 rad and j5 commands <=0.83 rad: no stop-limited evidence.
Before motion changes, test up to **96 full plans** on pilots only, retaining
all IK/collision/physical verification and the existing search/step budgets.
This is a new bounded-coverage hypothesis motivated by pilot refusals, not an
individual panel failure. If it finds a pick, record the selected rank/site,
physical hold and score rather than equating IK pairs with actual coverage.

## Geometry amendment before mesh-table physics probes

The 96-plan probes remain zero: 403 and 389 pick-plan calls across direct,
high and relay searches on standard 100067/69. Detailed default-budget
instrumentation on standard 100067/68 identifies the dominant refusal as
`pick_lift: held object below the table` at the first waypoint (119 and 120 such
failures respectively). The current guard uses held OBB corners, even for a sparse headset
whose empty box corners extend below the support plane. Arm/finger endpoints
pass, and no joint-stop evidence appears.

Probe an optional **full-mesh** table guard. It may reclassify only that held-
object table refusal after every arm/finger/obstacle check already passed.
Use all mesh vertices, retain the existing table tolerance, and conservatively
check global minimum z whenever the mesh XY bbox overlaps the table; this
also covers triangles crossing the table with no vertex inside. Missing or
invalid mesh falls back to the original bbox guard. Record OBB and mesh min z,
override counts, physical lift/hold and final score on pilot records only.
No blanket collision-disable, smaller safety margin or unchecked lift is allowed.

## Frozen final panel registration (before generation/outcomes)

Production source **18e70412** screens 7/7 standard and 6/7 random development
pilots (original two plus five new per cell). On the new pilots alone,
standard rises 2/5→5/5 and random stays 4/5. The remaining random 100066 71 mm
grasp rises zero and remains a physical failure. Do not count pilots as panel
rows. No front-waypoint, deeper-grasp, wider-search or joint-limit change is
included: the numeric evidence supports the conservative mesh table guard.

Fresh final panel: v15, **variant 15**, salt
`laptop-round2c-meshtable-2026-11-07-88c48b74`, ids **100085–99** per cell.
These ids have never been generated/run by this stream. Settle once and pick
the first ten stable ascending before any policy outcome; hash exact settled
records and use those files in both arms. Expected ids 100085–94 if stable;
reserve 95–99 only for predeclared stability/infra exclusions, never score.
No sealed/protected ids >=100100 and no rendered/proxy jobs.

A: `hang_cradle=straddle`, `headset_seat=off`, `headset_carry=legacy`,
`headset_grasp=legacy`, `headset_table=bbox`.
B: same straddle plus `headset_seat=on`, `headset_carry=staged`,
`headset_grasp=dense`, `headset_table=mesh`.
No laptop route in either arm; source and all remaining flags identical.

Ten pairs per cell, A/B order alternated by id, separate from every earlier
panel. Report n, score mean/sample SD, headset credits, full successes, score
wins/losses/ties, success flips, steps and all infra exclusions. Stopping rule:
**B earns headset credit on >=7/10 in each cell**. Also require the current
paired score gain gate for the prospective camera spec: pooled +5 points,
wins >=2*losses and >=2 net wins; do not silently pool selected panels.
Two concurrent LAPTOP scopes max, light2/light8, same admission/resource/path
rules. Zero LLM calls/tokens/spend. No camera dispatch or board change.

## Completed mesh-table panel

Frozen action source **18e70412**, first ten stable 100085–94 per cell. All
30 draws settled; all 40 episodes valid, no exclusions, missing meshes,
infrastructure errors or exhausted budgets.
[Rows, hashes, admissions and recipes](2026-11-07-laptop-g3-round2c-results.json).
Points are 0–100; SD is sample SD.

| Cell | n pairs | A mean ± SD | B mean ± SD | Headset credits A/B | Full successes A/B | Score W/L/T |
|---|---:|---:|---:|---:|---:|---:|
| standard | 10 | 0 ± 0 | 18 ± 6.32 | 0/10 / 9/10 | 0/10 / 0/10 | 9 / 0 / 1 |
| random | 10 | 4 ± 8.43 | 10 ± 10.54 | 2/10 / 5/10 | 0/10 / 0/10 | 3 / 0 / 7 |

Pooled gain +12 points, 12 wins / 0 losses passes the score gate; **random
5/10 fails reliability**. Preserve this panel separately, with no selected
pooling. Only aggregate outcomes were used to decide continuation; return to
independent random pilot 100066 for the [band diagnosis](2026-11-07-laptop-g3-round2d.md).
Steps median/max: standard A63.5/138, B253.5/280; random A69/158,
B146.5/337. Full-success flips 0/0. Laptop route remains unattempted.

Local full suite on the exact action source: **2071 passed / 191 skipped**
(934.65 s); focused suite 24 passed. Zero model calls/tokens/spend (0/0/$0),
no camera dispatch and no board change.
