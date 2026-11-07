# Laptop G3 round 2g — loaded-hold diagnosis

## Registration before diagnostic actions

The frozen3866 wide/clear candidate passes its original banks (11/12 standard,
10/12 random) and independent random bank9/10, but standard coverage18 reaches
only2/5 (maximum7/10). Finish its full registered screen and retain this null;
its conditional fresh registration remains unexecuted. Ids100000–14 are still
ungenerated. No salt reroll or fresh-panel failure is used for development.

Standard coverage18 **100030** starts with a43 mm left pinch whose original
verifier reports16.3 mm nominal in-hand drift. Subsequent pickups/relays fail.
The opposite-arm pilot also fails: first right26 mm pinch is loose15.5 mm,
then later attempts slip/fall. Neither result distinguishes initial contact
centering or pendulum settling from a hold that continues to slide.

On this existing pilot only, execute the **same first coupled checked pick**
under source3866 and the frozen wide/clear profile. Stop the transfer after
that pickup, preserve its original verdict, and record actual measured robot
joints, object transform in TCP, original material pinch point in TCP, actual
jaw joints and full-mesh table clearance. Hold30 stationary closed-jaw steps.
If the original pickup is still provisionally held and physically clear of
the table, plan/check a20 mm vertical raise and return with closed fingers,
recording the same measurements. Re-run the **unchanged original pick verifier**
at the end; do not reclassify the result, change a threshold or continue the
hang. This is a diagnostic, not a scored episode or a new admission rule.

Report root-relative translation/rotation drift and pinch-site drift during
the loaded stationary/probe phases, clearance and jaw gap. A steady suspended
hold would support a separately registered feedback/grasp hypothesis; continued
sliding or falling rejects that explanation. No synthetic attachment, object
pose write or collision exemption. Both motions require native full checks.

Compute3 physics-only; idle light8 beside the frozen standard screen, at most
two actual8G/600% scopes with unchanged who/GPU/disk admissions. All writes
under the existing permitted roots; no camera jobs, board change or model
calls. No new controller or fresh panel is authorized by this diagnostic.

## Initial diagnostic and loaded challenge amendment

The first run completes the registered lift and30-step hold. Original verdict
is loose16.3 mm; unchanged final verifier is loose17.7 mm. During30 stationary
steps root-in-TCP moves at most0.87 mm, rotates1.48 degrees and the original
material pinch point moves0.49 mm. Last20 steps:0.31 mm/0.23 degrees/0.11 mm.
The nominal-to-measured rotation is66.9 degrees, and material pinch offset is
[11.93,-0.79,11.31] mm in approach/closing/cross axes: movement along the
closing axis is small, but the contact has moved toward the pad end while the
body rotates. Measured jaw joints are20.91/21.35 mm. Mesh bottom is only5.6 mm
over the table, so the pre-registered30 mm clearance guard correctly skips
the challenge. This does not yet prove an unsupported transferable grip.

Retain that trace unchanged. On the **same pilot** repeat the unchanged pick
and30-step hold, then allow the checked20 mm raise/return when mesh clearance
is>=3 mm; require the measured peak mesh clearance>=20 mm before interpreting
this as an unsupported challenge. Record initial/peak/final relative transforms
and every step, with the original verifier unchanged. Plan both legs before
execution, no hand/table exemption. Also run the same diagnostic on existing
failed coverage18 standard100031, in the second idle slot, to distinguish a
site-specific steady grip from general drift. These remain unscored diagnosis;
no admission criterion, source action or fresh-panel gate changes. Two initial
recipe import errors occurred before runner creation/actions; both are retained
as infrastructure recipe failures, not policy scores.

## Unsupported challenge measurements and selected prototype

Both native raise/return plans pass. Standard30: measured root rises19.56 mm,
mesh bottom reaches25.08 mm above the table; throughout the challenge root-
in-TCP deviates at most0.53 mm/0.40 degrees, ending0.54 mm below its starting
height. Standard31: root rises19.69 mm, bottom reaches92.39 mm, relative drift
<=0.62 mm/0.42 degrees; original nominal drift14.1→13.6 mm remains loose.
The original verifier is unchanged in both diagnostics. These physically
unsupported, following holds justify a separate **loaded-hold proof**, not a
larger nominal drift threshold or a claim every loose pickup is sound.

Prototype default-off `headset_hold=loaded` (legacy default): only a headset
pickup already classified `ok=True,status=loose` by the original verifier may
try it. Keep original hard slip/lift gates. Hold30 closed stationary steps,
then pre-plan/check a20 mm vertical raise and return, with8 hold steps at
each endpoint, unchanged orientation and all native scene/hand/mesh checks.
Sample actual refreshed object/TCP pose at each physical step (preserve any
existing step observer). Require full-mesh initial clearance>=3 mm, peak>=20 mm,
measured object rise>=17 mm, all sampled attachment translations<=3 mm and
rotations<=3 degrees relative to the settled pre-challenge reading. Re-run
unchanged original verifier at the end; hard slips still reject. Only after
this loaded proof may a loose result be recorded as `ok` for a measured-
attachment re-plan; retain the raw nominal drift in its note. Any invalid
mesh, missing hold, unstable sample, infeasible leg, budget/end or failed
measured place re-plan must not become a successful hold/carry. No synthetic
attachment, scoring change or pose write. Normal pickups/other objects/default
modes execute exactly their old path. Fresh camera observations would be needed
to transfer this proof beyond G3; a cached belief cannot establish following.

First prototype on the two diagnosed failed pilot records only. If they hang,
implement and mock-test the default-off mirrored field, then freeze one common
profile and screen both full original12 and coverage18 ten/cell again. Admission
remains>=9/12 original and>=8/10 coverage in each cell. Fresh panel needs a
separate salt/source registration after that gate; all unused ids remain
untouched. No change to frozen3866 screen or its null decision.

Standard31 prototype earns20 in304 steps: its14.1 mm nominally loose hold
passes the unsupported proof, measured carry shifts28.1 mm but the measured
tail/release leaves the bridge23 mm from the cradle. This is a successful
physical hang, not a rigid-attachment claim. Standard30's first hold passes
the proof but its measured attachment has no checked place; it is correctly
put down rather than carried on the nominal plan. Its remaining attempts
continue unchanged. To avoid mistaking one rescued site for generality,
add the two other already-failed independent standard pilots34 and38 as
separate loaded-proof prototypes, on the freed light2 slot. Preserve30's
outcome regardless. Any integration will be justified as a measured loaded
proof, not a guaranteed rescue of30, and must pass the full registered
original12/coverage10 screen before touching any unused layout.

## Production candidate registration before screen outcomes

Freeze action source **0dc59c72**, isolated deployment
`~/controlr-laptop/frozen-round2g`. The same common F profile gains the single
new field `headset_hold=loaded`; all defaults remain legacy/off.65 focused
mock tests pass, including stale observations, sliding, rotation, support,
invalid mesh, hard-slip rejection, both native-leg refusals and observer
restoration. The full local suite is running. Implementation additionally
re-runs the original hard verifier after the stationary phase, updating the
measured held geometry before native challenge planning; this enforces the
registered hard gates rather than using a stale held box. Preserve this
source/recipe difference from3866 prototypes.

Complete both original12/cell and coverage18 first ten/cell (same frozen
settled hashes as E/F), source0dc in both cells, new r2g-production output
paths and manifests. Gate>=9/12 original and>=8/10 independent credits in
**each** cell; complete all44 regardless of early failures. Every policy
zero is retained, no shape-specific routing, no pooling old candidate rows.
The remaining3866 prototypes finish on their frozen deployment; a production
driver may start only once the corresponding LAPTOP slot is free. Two actual
8G/600% scopes on light2/light8, unchanged who/GPU/disk admissions. No salt,
layout generation, new panel inspection, camera job, model call or board change
is authorized by this development registration. A fresh panel still requires
a separate source/salt registration and both gates first.
