# Laptop G3 round 2h — strict-transfer loaded proof

## Completed fresh-panel decision

**The requested mechanical stopping rule and paired gain rule both pass.**
Source **10f3e0a8**, perfect-state G3 mode A, physics only, ten fresh pairs
per cell. All40 official rows are valid, stable and complete: no episode
exclusions, retries, reserve replacements or missing meshes. Each A/B pair
uses identical frozen settled bytes. Earlier panels remain separate.

| Cell | Pairs | A mean ± sample SD /100 | B mean ± sample SD /100 | Headset credits A → B | Score W/L/T | Full successes A/B |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 10 | 0.00 ±0.00 | 18.00 ±6.32 | 0/10 →9/10 | 9/0/1 | 0/0 |
| Random | 10 | 0.00 ±0.00 | 18.00 ±6.32 | 0/10 →9/10 | 9/0/1 | 0/0 |

Paired delta18.00 ±6.32 points in each cell; pooled delta18,18 score wins,
0 losses,2 ties. Headset-credit flips18/0; full-task success flips0/0.
Both B rates exceed7/10. Net **+0.30 G3 Average points** at1/120 per cell,
without camera adoption or board change.

A/B steps median/max: standard60/145 versus280.5/422; random133.5/169 versus
241.5/338. No primary episode exhausts800 steps. Raw/proven pick entries
overlap phases; these are actual episode counters, not summed skill steps.

The [frozen manifest](2026-11-07-laptop-g3-round2h-manifest.json),
[raw archive](2026-11-07-laptop-g3-round2h-results.json) and
[validated summary](2026-11-07-laptop-g3-round2h-summary.json) retain44
development plus40 primary rows, all30 exact settled JSON texts/hashes,
source hashes, profiles, recipes, admissions and attempts. Both drivers
complete20/20 attempts withrc0. Policy zeros are retained.

## What works and what remains

The bundle combines clear rotation/fixed-orientation carry, geometry-filtered
full-opening grasps, checked table rest/re-grasp, actual release clearance,
bounded site retries and a loaded proof only where the original loose rule
would put down. Original hard rise/30 mm slip limits remain. The
[diagnosis](2026-11-07-laptop-g3-round2g.md) separates a one-pad pendulum
(fixed-pivot residual<=0.93 mm) from the earlier97.6 mm carry loss. Two
unsupported holds follow20 mm raises by19.56/19.69 mm with<=0.62 mm/0.42
degrees of relative movement. This does not claim a rigid nominal grasp.

Both retained B zeros are100005. Standard rejects26 mm pickups at18.9,
16.1 and23.5 mm nominal drift, then a21 mm pickup rises0/80 mm and drifts
98 mm. Random rejects a39 mm pickup at12.2 mm drift, puts it down and has
no later hang plan. These were inspected only after all40 rows completed,
for reporting. No retuning or further physical experiment follows.

Fresh coverage: standard asset4 seven times,2 once,5 twice; all ten random
records are asset8. Older development includes9/10, but this panel does not
establish fresh rare-shape reliability. The result meets the operational
target on this registered draw, without selected pooling.

The laptop80-point tier still needs a supported pickup. The
[tray/rack study](2026-11-07-laptop-g3-round2.md) records a179.3 x125.95 x10.07
mm closed slab, stand interference for bimanual tray targets, and0/348 left
versus4/504 right upright endpoints passing native checks. Right reach exists;
a supported pickup is unproven. The1.09 N/pad ideal friction requirement is
not a measured force-capacity claim. Closing alone earns0; no laptop panel.

The [camera spec](2026-11-07-laptop-camera-ab-spec.md) remains blocked on a
real held observer and compatible renderer. Cached/projected belief is not
physical evidence; this helper lacks a timestamp/provenance contract for
camera use. No observer is implemented here, no camera jobs/model calls,
no board change. Physics-only numeric traces replace rendered videos on
driver595.99. Model calls/tokens/spend **0/0/$0**.

Full local **2200 passed,191 skipped**,1025.96 s;72 focused mocks pass.
Defaults stay legacy/off. Native refusal, other objects/threaded hooks and
unchanged tolerant relay execution are covered. Controller/tests remain
byte-identical to10f throughout this panel; later main integration changes
research/docs only. All LAPTOP physics jobs exited; other streams/root3613
untouched. Footprint296 MB deployment +43 MB data,91 GB disk free.

## Development admission completed before primary

All44 registered source10f3e0a8 development episodes complete. Original bank:
12/12 standard and11/12 random headset credits. Independent coverage:
8/10 standard and9/10 random. Both original gates pass unchanged. All44
rows are stable with complete meshes, no episode exclusions or replacements.
Full-task successes0.

| Bank | Cell | n | Headset credits | Mean ± sample SD /100 | Steps median / max |
|---|---|---:|---:|---:|---:|
| Original | Standard | 12 | 12 | 20.00 ±0.00 | 247.5 /632 |
| Original | Random | 12 | 11 | 18.33 ±5.77 | 245.5 /533 |
| Independent | Standard | 10 | 8 | 16.00 ±8.43 | 356.0 /789 |
| Independent | Random | 10 | 9 | 18.00 ±6.32 | 289.0 /399 |

The scope correction rescues standard34 in789/800 steps. Independent
standard33/38 and random35 remain0; all outcomes are retained separately
from the fresh A/B. Full local2200 passed/191 skipped;72 focused mocks pass.
All new defaults remain legacy/off.

The registered30-file variant21 raw draw is settling. Freeze both first-ten-
stable manifests before either policy condition. The requested fresh7/10
target is not yet tested. No camera/model calls or board change.

## Registration before revised actions

The source041 loaded-proof development screen remains frozen and completes
in full. Its conditional fresh variant20 remains **unexecuted**, with ids
100000–14 still ungenerated. No new-panel outcome is used for development.

Code review finds a scope problem: `run_transfer` also attempts the loaded
proof on tolerant table relays (`_run_leg` tol_xy30 mm). Those already allow
an originally successful loose lift under the legacy20 mm tolerance rule.
This spends30 stationary plus checked raise/return steps unnecessarily, and
on refusal returns to that legacy loose-carry path. The helper does not
promote a refused proof, but the earlier registration's no-carry-after-refusal
claim is too broad for the actual caller. Retain all actual G scores and raw
verdicts; do not treat that protocol claim as established. Standard34 prototype
also demonstrates the cost: its final checked place needs219 steps with212
available after multiple proofs/relays. This motivates scope correction,
not a change in drift threshold or controller step budget.

Revise only the proof caller: attempt it **only when the original loose rule
would put the object down** (`tol_xy <20 mm` or original in-hand-view rejection).
Keep the original tolerant relay path exactly, without added proof actions.
Every invoked proof that refuses then goes through the existing strict
put-down rule, so no refused new proof can become a carry. Keep all original
hard slip/lift checks and all loaded-proof motion/measurement thresholds.
Default/other-object behavior remains unchanged. Add mock regression coverage
for identical tolerant-relay execution with legacy/loaded settings and strict
rejection after a refused proof. No task/id/asset routing, forced state or
collision exemption.

After implementation/focused tests, freeze one new action source and the
same common bundle (F wide/clear plus headset_hold=loaded). Complete original
12/cell and independent coverage18 first ten/cell, same retained settled files,
new manifests/output paths. Gate>=9/12 original and>=8/10 independent headset
credits in each cell before any unused final layout is generated. Complete
all44, no replacing policy zeros. Keep G source041's entire screen separate;
it may qualify its numerical gate but is superseded before fresh generation
for this code/protocol reason, not silently relabelled.

Use a separate isolated deployment under ~/controlr-laptop so remaining G
jobs cannot import changed source. Two actual physics-only8G/600% scopes
maximum, explicit light2/light8, unchanged who/GPU<=24576 MiB/disk>=10GB
admission. Finish each old cell before its slot starts the revised screen.
No model calls, renderer, board change, camera dispatch or sealed data. A
fresh salt/source/pair registration follows only after this source is frozen;
none is generated by this development registration.


Freeze action source **10f3e0a8** after72 focused mock
tests pass. Flags are exactly the F wide/clear profile plus headset_hold=loaded.
Original hard thresholds,30 stationary/20 mm checked challenge and all native
checks stay unchanged. Retain source hashes before the first new outcome.
Source041 jobs continue to completion in their frozen deployment; new H drivers
wait for the corresponding completed G cell before claiming its slot.

## Conditional fresh paired-panel registration — before generation

Only execute if the complete source10f3e0a8 development screen passes both
>=9/12 original and>=8/10 independent credits in each cell. Otherwise keep
this registration unexecuted and continue pilot development. Do not substitute
source041 rows into this gate. The source041 conditional panel is superseded
before any generation because of the caller-scope issue above.

Both conditions are perfect-state G3 mode A, physics only, same rule plan,
800-step task budget and frozen source10f3e0a8. Common hang_cradle=straddle.
A explicitly sets hang_wide=off, seat=off, carry/grasp/retry/lift/hold=legacy,
table=bbox, stage=off. B is the exact registered wide/clear+loaded bundle:
hang_wide=on, seat=on, carry=staged, grasp=cups, table=mesh, stage=mesh_clear,
retry=sites, lift=slow, hold=loaded. No laptop route, model call or belief noise.

Generate previously unused root-namespace ids **100000–14**, v15 **variant21**,
salt `laptop-round2h-strictload-2026-11-07-88c48b74`, in each cell. Preflight
refuses any existing raw/settled record. Neither variant19 nor20 was drawn;
this is source correction before generation, not a salt reroll after outcomes.
Settle each once, select first ten stable ids ascending before policy outcomes,
retain later stable records as infrastructure reserves only. Freeze BOTH cell
manifests and layout/source hashes before launching either condition. Each
pair uses identical settled bytes in A/B; assert hashes before every job.
No id>=100100, sealed source, renderer or camera output.

Complete ten pairs/cell (40 valid rows), alternating A/B order by selected
index. Single-episode7200 s ceilings preserve the controller's800 steps.
Use only our light2/light8 slots, at most two actual8G/600% scopes, who/GPU
before batches and <=24576 MiB used VRAM/>=10GB disk admission for each job.
Record every attempt. Confirmed external process/driver/OOM or pre-action
reset failures may retry the identical pair once, then use the next stable
reserve with both arms. Do not replace completed policy no-plans, slips,
place misses, budget failures or action-induced simulation failures. Missing
official score after policy failure means incomplete panel, not an invented0.

Report n/exclusions, A/B actual mean/sample SD on0–100, headset-tier credits,
full-task successes and score wins/losses/ties/success flips per cell. Keep
all earlier candidates/panels separate, no selected pooling or tuning from
individual fresh-panel failures. Mechanical stopping rule: **B>=7/10 headset
credits EACH cell**. Separate real-gain gate: pooled B−A>=5 points, score wins
(>=10-point increase)>=2*losses and>=2 net wins. A real gain authorizes only
a prospective camera spec with fresh physical held-observation prerequisites;
no camera dispatch or board change. If the mechanical target fails, retain
this null and resume pilot work without rerolling this salt.

## Development scheduling amendment before standard outcomes

The H random screen runs on light8 while G standard finishes on light2.
H standard has produced no manifest or outcome yet. Replace its waiting-only
parent with a two-worker batch scheduler: after the G standard completion
marker, write BOTH H standard bank manifests, assign the predetermined eleven
two-episode groups alternately to light2/light8, and queue through the same
locks/admission wrappers. The existing H random driver still competes for
light8; there are at most two actual scopes, with unchanged8G/600% limits.
This keeps a released slot useful when only one cell remains. Controller,
flags, source, layouts, group membership, gate and all policy outcomes stay
unchanged; bank order is not an experimental condition. Retain the cancellation
of only our verified waiting parent PID and both scheduler recipes. Never
stop an in-flight physics job or overwrite an existing outcome.

Before the standard scheduler leaves its G-completion wait (no H standard
manifest exists), refine worker allocation to claim the next predetermined
group from a shared FIFO, rather than fixing every other group to one slot.
This avoids putting all three known slow standard coverage groups on light2.
Membership/order in the pending list, all22 layouts, source and gates stay
fixed; completion time only determines which free slot claims the next group.
Retain both waiting-parent cancellations and scheduler versions. No in-flight
physics process or outcome is stopped or replaced.

Full final local suite on the strict-transfer implementation: **2200 passed,
191 skipped**,1025.96 s.72 focused mocks also pass. Action source remains10f;
subsequent changes are experiment documentation/orchestration only.

H random completes11/12 original and9/10 independent, freeing light8 while
G standard's last group still owns light2. Before any H standard manifest,
allow its FIFO parent to start when **either G standard or H random is done**.
Its light2 worker queues behind G; its light8 worker immediately uses the
released slot. Both H manifests still precede outcomes and both sources stay
isolated. This removes a scheduling-only dependency, without changing any
layout, controller, group, gate or outcome. Retain the verified waiting-parent
replacement; no physics process is stopped.


The last scheduling amendment was not applied: G standard finished and the
existing FIFO parent created both H manifests before the replacement PID
check. That parent was resumed/preserved. The additional launcher refused
the existing manifest before creating any episode or scope. Retain its
startup refusal separately; no physical job was stopped, duplicated or
excluded. The original two-worker FIFO is the actual standard scheduler.

The complete screen passes: original12/12 standard,11/12 random;
independent8/10 standard,9/10 random. Both original admission gates are met
without amendment. Source/config remain10f and the registered bundle.

Light2 was acquired by IMIT2 after our screen ended. Cancel only our verified
waiting generator wrapper/orphan flock (no physics or generated records), then
use free light8 for the registered draw. All30 variant21 raw files verify
complete. Use the existing free light7/light8 pool for settling and the paired
panel, rather than pre-empting light2's owner. A lock acquired by another
stream queues our job normally. At most two LAPTOP jobs, unchanged who/GPU,
<=24GiB VRAM/>=10GB disk and8G/600% limits; no other process is touched.
This is resource admission only, before any A/B policy outcome.


## Frozen primary execution

Both settlers exit0 with15/15 stable records in each cell. Before either
condition, freeze the [complete manifest](2026-11-07-laptop-g3-round2h-manifest.json):
selected100000–09 and infrastructure reserves100010–14 in each cell, variant21,
registered salt/source10f and complete source/layout hashes. No reserve is a
policy-failure replacement.

Launch the two cell drivers on light7/light8. Each condition loads the exact
settled record (explicit SIMLAB_DEV_LAYOUT_SUFFIX=.settled), verifies hashes,
and validates source/config/task/id/stability/mesh/official finite score and
settled filename. Refuse malformed or incomplete outcomes for explicit
adjudication; completed policy zeros are never rerun. The40-row primary
validation is running. No controller/test changes after source10f, no camera
or model calls, no board change.


## Primary completion

Both drivers finish20/20 official episodes withrc0, no retries or reserves.
The mechanical target is met; stop physical work. Finish reports, BACKLOG
and PR checks only. Defaults off, camera spec blocked, board unchanged.
