# BYIMG R3 target-hold isolation — development registration

Registered before outcomes. This is one component screen, not a task-score
panel or an adoption test. The single full-task paired panel is registered
separately after the transfer bundle is frozen.

Generate ten fresh v15 variant-23 draws, ids 100080–100089, salt
`byimg-r3-target-isolation-20261107`, under the private BYIMG R3 root.
These ids/draws have never been used by BYIMG. They are physics-only G3 inputs;
no rendered, reserved or sealed slice is read. Keep all stable draws, report
unstable loads separately, and do not replace by target identity.

Run the implemented lift-first prefix with `belt_first=on,belt_target=match,
belt_first_recover=on,belt_first_phase=on`: basket end-wall intercept and
measured elevated hold, phase-predicted picker, matchbelt 125 mm target lift,
and at most one picker-only grasp-family retry. Stop before loading or any
loaded-lift variant; hold both commanded arms and closed jaws for ten more
actions. A successful target hold requires target-origin rise >75 mm both
immediately after the pick and after that hold. Report n/N, rise, actual finger
gap, picker, grasp family, contact phase and steps. Infrastructure/refusal is
not scored as a physical miss. This screen is used only to diagnose target
interception; these ten inputs are excluded from the final paired panel.

Source prefix: `af7ec47b`. A later change to transfer/loaded-lift code does not
change this prefix. Runs use the shared compute3 light locks, admission checks,
8G/600% scopes, at most two jobs, physics only and zero model calls.

## Frozen-prefix confirmation registration

After the spent-layout component screens, validate frozen source `fab6ca6a`
once on ten additional fresh raw v15 draws 100090–100099, variant **21**, salt
`byimg-r3-frozen-target-confirm-20261107`. Same >75 mm immediately and after
10-action hold criterion, no loading, no tuning from these outcomes. Report
this independent n/N separately from the original 5/10 screen and from the
full-task settled panel. A first launch specified invalid variant 29 and was
refused during generation before writing any input or running an episode;
variant 21 is fixed here before outcomes. The task panel's variant 19 is valid.

## Component results

Initial registered screen: **5/10** retained targets (81,83,86,88,89), all stable
loads. The others end at one basket intercept, two target slips and two
no-intercept plans. Conditional 5/9 after the basket miss is descriptive only.

Frozen-prefix independent confirmation: **6/10**, all stable loads, no episode
replacement. Six successful targets rise 122.75–125.95 mm before the hold and
122.83–125.99 mm after it. Failures are three no-intercept plans (90,97,99) and
one -5 mm slip (98). This does **not** establish reliable interception. The
spent-layout combined screen is 7/10 and is not substituted for this fresh
result. The task panel remains a separate settled-input comparison. Exact
component rows are in the R3 probe archive.
