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
