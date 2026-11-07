# POUR — one fresh paired G3 panel

Registered **2026-11-07T05:40:53Z**, before any controller run on panel layouts.
Dev-only, physics-only; no submission/candidate change, camera job or model call.

## Evidence used for selection

Main aac2a4b4 language baseline: 0/10 full successes, 0±0 points; eight
third-pour budget stops and two second-pour slips. Six exploratory mechanisms
on fresh pilot records 100070–72: aim-only, sparse, via and checked v5 each 0/3;
overlap 1/3; budget 2/3. Budget's successes took 698 and 785 steps and have all
three final fluid predicates true. Single-cup exploratory aim and checked
versions both succeed in 285 steps on their one shared record. The separate
ten-layout main reference baseline remains pending and finishes before panel
dispatch. No paired-panel outcome informed this choice.

## Frozen source and arms

Both arms run commit **bcc8d4117e5ebaa53a7f597f780033a70fcd6e7a** from private
`~/controlr-pour/candidate4`. This incorporates current main's unrelated BYIMG
work. POUR controller files are unchanged by that merge. Pilot snapshots remain
frozen separately. A retains legacy POUR behavior, verified by dispatch and
kinematic command/step equivalence tests; both arms share new G3 diagnostics.
The earlier ten-layout baseline is exact main at the study's start, not this
instrumented comparison source.

Common: GT mode, `--routes pour_order --skill-attr ported_routes=on`,
`ported_pour_profile=off`, no mode-B hooks, default native physics and limits.

- **A:** `ported_pour_pace=v1`, `ported_pour_aim=v1`.
- **B:** `ported_pour_pace=budget`, `ported_pour_aim=b10`.

Budget retains the overlap flow angles/speed, hold count, drain and staged
untilt angles. Lift is normal speed; untilt uses the existing full held-speed
limit. Only the last flow profile may spend the put-back reserve. Its own
completion test refuses partial or not-upright pours. Move leases, step limits
and safety checks are unchanged. Single-cup routing ignores pacing: its B
change is aim only, independently covered by unit tests.

## Fresh records and ordering

Gain: **10 pairs pour_by_language**. Guard: **10 pairs pour_liquid_into_cup**.
Raw ids 100080–89 were freshly generated with salt
`pour-panel-20261107-6c42a9`; [raw hashes](2026-11-07-pour-g3-prereg-layouts.json).
No old record sharing an id is used. Select the first ten stable presettled
records in increasing id order before any controller outcomes. If an initial
record is unstable, use the first stable unseen replacement from 100090–99,
generated with the same salt. Publish settled hashes/selection before dispatch.
No selection by success, score or controller diagnostics.

Run both arms on the identical selected settled file. Alternate AB then BA by
layout index, five of each per cell. One episode per layout/arm, no retries of
controller failures, no early stopping. The registered driver verifies source,
settings and hashes, rejects duplicate/mixed rows and skips already recorded
episodes on resume. Admission/process failures stop for diagnosis. If either
arm has a verified infrastructure failure, retain both attempt records, exclude
that pair from paired inference, and use the next predeclared fresh stable
replacement for both arms without changing either configuration. Report all
such attempts and replacements. An implementation change invalidates this panel;
it does not authorize a retuned second panel.

## Outcomes and decision

Report official score on 0–100, mean, **sample SD**, full successes, and paired
score wins/losses at **>=10 points**; report success flips separately. Also
report the full-success-only score (100 on full success, otherwise 0), as the
prior task-specific lead ruling does not seek intermediate home-trigger credit.
No idle-arm move is added to manufacture a home edge. A positive feasibility
result must include a full checker success with all three final fluid predicates
true. Preserve the distinction between latched checks and current predicates.

The gain screen must pass on both official and full-success-only scores:
B mean >= A mean +5 points; wins >=2×losses and >=2 net wins. The reference
guard must lose no more than 5 mean points and have no more losses than wins,
on both score definitions. This is stricter than the default score-only rule
because this checker has the documented home-edge loophole. Ten pairs establish
feasibility, not a precise general success probability or camera improvement.
If the screen passes, write a parked camera A/B spec; do not dispatch it.

Classify each actual failure by the first blocking stage: routing/planning;
grip/lift; tilt/in-hand slip; flow profile/budget; untilt lease/budget/pose;
fluid capture/residual; remaining checker/home condition. Use motion traces,
pending upright labels and final particle predicates. A successful episode is
a success even if its conservative internal skill verdict preceded a successful
final home action. Report steps and final physical capture alongside scores.

## Execution and budget

40 scored episodes, zero LLM calls/tokens/$; one light7 job at a time, 8G/600%
scopes, <=24 GiB used VRAM and >=10 GB free disk at every start. Run through
`scripts/skills/pour_g3_panel.sh` after publishing the private registration JSON
and after the main reference baseline completes. Keep raw/settled files, logs,
actions and failed traces under `~/dojo-data/pour`; code under
`~/controlr-pour`. No sealed data, compute4, RTX rendering or video generation.
There is no statistical time box or optional stopping based on observed gains.
