# Laptop G3 round 2: separate wrist rotation from cradle approach

## Diagnostic and development protocol

Zero LLM calls. Compute3 physics only; no camera dispatch or board changes.
Two concurrent LAPTOP jobs maximum, light7/light8, admission <=24576 MiB used
VRAM and >=10 GB disk; who/GPU admissions, 8G/600% scopes, KILL-tail timeout.
Deployment stays within `~/controlr-laptop`, data within `~/dojo-data/laptop`.
BYIMG's light3/light4 processes are untouched.

Use round-1 pilot records 100030/31 only for mechanism probes. Never tune on
round-1 panel records or round-2 panel outcomes. Probe high carry, thin bridge
pinches, staged rotation/carry, site ranking, loose-hold continuation, and
bounded re-grasp selection. Do a separate short physical/IK laptop study.

Fresh panel draw recipe fixed before generation: v15, variant 12, salt
`laptop-round2-staged-2026-11-07-88c48b74`, ids 100070–84 per cell. These ids
were not used in round 1, and salted draws are new. Settle once, select the
first ten stable ids ascending before policy outcomes, hash the exact settled
files, use those files in both arms. No ids >=100100. The 100050–99 restriction
is on rendered/proxy episodes; these are physics-only G3 draws.

Candidate switches will be frozen and committed after pilot-only development,
before any panel episode. A is round-1 control (`hang_cradle=straddle`,
`headset_seat=off`, `headset_carry=legacy`). B will retain the same straddle
setting and use a default-off staged carry bundle. Ten pairs per cell, order
alternated by id, report mean/sample SD, headset credits, full successes,
score wins/losses/ties, steps and infrastructure exclusions. Reliability
criterion: at least 7/10 headset-tier credits **per cell**. Camera advancement
also requires the current paired score rule; write only a spec, dispatch none.

## Frozen candidate (before panel episodes)

Use B: `headset_seat=on`, `headset_carry=staged`; A keeps both defaults.
Both use `hang_cradle=straddle`; neither enables the laptop route. The staged
carry turns the held object to the target orientation above its current xy,
then uses a checked Cartesian segment at fixed wrist orientation to the hover.
Both legs must plan before anything executes. Existing finger/arm checks,
cradle contact allowance, measured attachment, grasp-slip/loose rejection,
release and retry budgets remain in force. Defaults remain off/legacy.

Development: staged carry scored headset credit on 3/4 pilot records (std
100031, random 100030/31; std 100030 fails loose holds). Higher 12/20 cm
approaches and bridge-centre thin pinches found no complete transfer. Preferring
low ear-cup sites did not fix the failed standard pilot. Continuing its loose
22 mm hold lost the headset, so loose rejection is retained. Rejecting the
failed pinch site did not rescue that pilot and is excluded too. No pilot
result is included in the panel; these are a mechanism screen, not a reliable
success-rate estimate. Actual production wiring will receive pilot-only smoke
checks before the fresh panel; any correction is recorded before outcomes.

## Completed staged-carry panel

Source `43d5882b`; first ten stable salted variant-12 layouts 100070–79
per cell; no exclusions (all 30 generated layouts settle). Forty valid paired
G3 episodes, no missing meshes, no infrastructure failures or budget exhaustion.
[Rows, layout hashes, admissions, recipes and diagnosis](2026-11-07-laptop-g3-round2-results.json).
Scores below are 0–100 points with sample SD; raw rows use 0–1.

| Cell | n pairs | A mean ± SD | B mean ± SD | Headset credits A/B | Full successes A/B | Score W/L/T |
|---|---:|---:|---:|---:|---:|---:|
| standard | 10 | 4 ± 8.43 | 12 ± 10.33 | 2/10 / 6/10 | 0/10 / 0/10 | 4 / 0 / 6 |
| random | 10 | 2 ± 6.32 | 14 ± 9.66 | 1/10 / 7/10 | 0/10 / 0/10 | 6 / 0 / 4 |

The pooled +10-point gain and 10 score wins / 0 losses pass the score gate.
The lead's **7/10 per-cell reliability criterion is not met**: standard is
6/10. This is not a stopping result; continue pilot-only grasp development,
then use another newly registered panel. Do not pool a selected later panel
with this one or relabel this candidate as reliable. No camera jobs/board change.

Steps median/max: standard A 93.5/195, B 201/306; random A 139.5/156,
B 210/279. No success flips (the laptop stage is not attempted).
Full local suite on this source: 2061 passed / 191 skipped, 987.22 seconds.

## Numeric mechanism evidence

Measured-joint audit of pilot standard 100031: TCP-relative displacement
remains ~1 mm through a 58.5-degree wrist turn. At steps 95–110, sampled
mesh proximity is ~5 mm while the object origin remains nearly stationary
against the stand and the TCP continues moving. Shift then reaches 97.6 mm.
A single fixed-pivot fit has 43.6 mm maximum residual; maximum joint tracking
error is 0.68 degrees. This favors premature stand contact during reorientation
over a pure pendulum or a joint-stop explanation. Mesh subsampling reports
proximity, not contact force: the causal interpretation is an inference.
Separating turn/carry reduces drift to 1.2 mm on that pilot and earns credit.

## Laptop feasibility (not a laptop panel)

A physical standard-100031 probe shuts the lid to 0.99 but slides the laptop
from its stand onto the table: asset origin ends at z 0.76489 m. Closed
geometry is 179.3 x 125.95 x 10.07 mm. The stand's sampled top bbox is
304.87 x 118.95 mm, so simply inserting two trays under opposite side edges
is obstructed at this supported pose; both hand targets pass IK but hit the
stand. After the closure slide, the left tray target is out of reach and
the right passes IK but has a link at table z 0.762 m (table is 0.765 m).
No unsafe contact plan executes.

Rack reach is **not globally infeasible**. From 240 generated slab-edge grasps
per arm, 58 left and 84 right start grasps pass IK. Their six upright targets
give 0/348 left endpoint successes (reach bound) and 4/504 right successes.
Eight additional 32-random-seed queries are recorded per arm. Four right
endpoints also pass the native self/table/scene check, including rack parts;
representative TCP ~[0.318, -0.185, 0.7814] m, endpoint error 0.38 mm / 0.14 deg.
This establishes endpoint feasibility, not a supported pick or full path.
The remaining mechanism needs underside clearance/pivot support or a re-grasp;
no new laptop transfer has been physically demonstrated, so no laptop panel runs.

Authored metadata: laptop mass 0.1 kg, friction 0.45; headset mass 0.25 kg,
friction 0.45. The old informal “~1 kg-class slab” is not supported by this
metadata. A symmetric Coulomb calculation for a 0.1 kg load at mu=0.45 needs
about 1.09 N normal force per pad; this is **not a measured grip-force limit**
and cannot establish physical impossibility. Geometry/contact access, not
mass alone, blocks the tested tray. A fixture-pivot maneuver remains unproven.

All studies are simulator-only, all inputs are private dev records, LLM
calls/tokens/spend 0/0/$0. No real robot, rendering, system changes or other
stream's processes are touched. Another stream later acquired light7; the
queued own pilot was terminated by its recorded PID and moved to light8.
