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
