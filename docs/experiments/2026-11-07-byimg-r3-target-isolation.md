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
