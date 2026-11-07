# POUR — parked camera A/B specification

Development specification only. No camera job is queued or dispatched, and no
submission/candidate file is changed. This is not evidence of camera gain.

## Question

Does the G3 budget profile's saved motor time let the camera controller finish
three correctly grounded pours within 800 steps without increasing slips?
The earlier overlap camera 0/10 remains a negative result; this tests the new
lift/untilt pacing and last-profile reserve policy, not a relabeled old panel.

## Freeze before future execution

The lead/Mikhail assigns a rendering-capable authorized host and camera-eligible
fresh dev layouts with a new salt, and records source, baseline config, model,
layout hashes and registration timestamp before outcomes. Do not use this G3
panel's 100080–89 as our camera measurement slice. Do not use sealed records.
Compute3's current 5.1 RTX renderer is known to crash; this specification does
not authorize a render probe, compute4 access or any host/driver changes.

Hold perception, model, prompts, grasp settings, joint limits, geometry,
new_routes and all other candidate settings identical between arms. Keep
`ported_pour_profile=off` and `ported_pour_aim=b10` in both arms; aim b10 is
already an adopted lever in the points table. If the actual frozen baseline
differs, resolve and register that comparison explicitly before execution.
The existing pour_order route must be enabled equally in both arms.

- A: legacy `ported_pour_pace=v1`.
- B: `ported_pour_pace=budget`.

Pacing affects ordered liquid pours only; single-cup and ball routes preserve
their pacing. Runtime uses belief state, with the existing truth guard. G3
particle/checker diagnostics never supply state to the camera policy. Do not
alter an idle arm to manufacture a home-trigger score edge.

## Paired design

At least ten fresh language pairs, ten single-cup reference guard pairs and ten
`play_Xylophone` strong-guard pairs. The public points table records the latter
at camera 100; confirm the frozen candidate still qualifies at >=90 before
registration. If it does not, the lead must name another already-qualified
strong guard prospectively. This is a parked design, not an executed registration.
Use the same presettled record within each pair, balanced AB/BA ordering,
frozen controller settings and identical perception configuration. Keep source
and input hashes in every row. Replacements are only for verified infrastructure
failures, selected from a predeclared fresh list; retain all failed attempts.
Do not retry controller failures, stop after early wins or tune on panel ids.

Report official and full-success-only scores, means/sample SD, full successes,
score wins/losses at >=10 points, success flips, final checker states and stage
counts. Apply the standard >=5 point gain, wins>=2×losses and >=2 net wins;
require the same on the full-success-only score for this task. Each guard must
lose <=5 mean points with no excess of paired losses over wins. Both guards
retain ordinary score gates; this design claims no parity score waiver. Log
resolved skill arguments and verify that pacing contributes zero actions or
belief writes on either guard.

The registered G3 full successes took 655–750 steps, leaving 50–145 steps;
the slowest has little room after historical 42–46 camera look steps.
A pilot G3 budget success at 698 steps would retain room for the historical 42–46
camera look steps; a 785-step success would not. Measure that overhead rather
than extrapolate a success rate. Break failures into wrong pair/grounding,
missing pick, grasp/lift, flow slip, profile budget, untilt lease/budget/pose,
and fluid/checker failures. Record whether companion moves actually executed,
which arm order was selected, and the final upright reserve consumed.

Capture a success and typical failure video only on the future authorized
native renderer. Count actual model calls/tokens/spend and separate perception
steps from motor steps. No model calls, images or videos were produced for
this specification. Adoption requires the rendered paired result; otherwise
keep the flag off and report the null without retuning this panel.

## Prospective belief-write audit

Follow the current DEVELOPMENT/EXPERIMENTS audit recipe on physics-only replays
of the future camera episodes. Report helpful/harmful/total writes per writer
and z error separately. On language, require each existing writer's harmful
rate in B <= A +5 percentage points; a writer new in B must have helpful >=2x
harmful writes. Report guard writers as well. Perception is held fixed and no
new writer is intended; trajectory divergence still warrants inspecting the
observed writes. Freeze these rules in the future registration before data.
This prospective design does not change the earlier G3 panel's registered gates.
