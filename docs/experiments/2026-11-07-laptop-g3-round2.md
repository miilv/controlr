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
