# PEN round 2: conditional holder/drop profile in G3

**Question.** Can pose and shaft geometry retain the fresh-layout gain without
losing any of the original baseline's successful paths?

**Setup.** Physics-only G3 on compute6, no controller/perception LLM. Twenty
paired development layouts: five original (100000–04), ten prior private variant-2
(100030–39), and five added private variant-2 draws (100040–44). All twenty use
fixed presettled records. Source freeze `3fda84cd`, checked source hashes and
matching deployment stamps in both arms. A uses defaults; B changes only
`fill_guard=geometry`. The four independent fill switches remain off in the
resolved config. No defaults changed, no sealed records used, no camera run.

The condition applies `rolled_tilt` + stationary holder + inset packing + safe
insertion when a rigid holder lies inside the arm bases' lateral span, beyond
one arm's comfortable reach, with nearest-arm distance >0.40 m, and at least one
flat horizontal slender shaft. It uses current world poses and geometry, with
no extra provider refresh and no layout/asset IDs. Other scenes execute the
baseline. This is a development heuristic learned on the old fifteen layouts;
the five added layouts produced no scores before the final code freeze.

## Original-five loss diagnosis

The [broad-profile experiment](2026-11-06-pen-g3.md) scored 40/25/0/10/10 versus
baseline 25/100/10/25/10. Saved-action physics replays reproduce **all ten scores**,
including the baseline full success. The critical differences are:

| ID | Broad minus A | Pen and phase | Observed mechanism |
|---|---:|---|---|
| 100000 | +15 | Slim-pen insertion | Three pens logged inside versus two; no lost baseline tier |
| 100001 | -75 | target2 oil_pen/00001, fit; target0 pen/00001, transfer planning | First oil changes wall side/yaw; subsequent inset attempts refuse a fit. Baseline's 2 mm padding fallback fits on the exact same recorded post-drop world. Target0 then has no executable transfer despite 330 screened pairs; baseline eventually inserts all four |
| 100002 | -10 | target2 oil release; target3 oil transfer | Stay skips relocation; first oil remains at the rim, second has zero drop pairs. Baseline relocates and inserts target3 despite joint-5 tracking lag |
| 100003 | -15 | target3 oil_pen/00001, transfer | Baseline's reported relocation failure physically moves the cup to a reachable pose; both oils land. Stay leaves target3 with zero drop pairs; both rolled pen/00000 shafts still slip |
| 100004 | 0 | Left-arm parking after attempts | Bundle tips the holder near step 360; banked score conceals a physical regression. Baseline remains upright |

On 100001, the first oil's tip settles **113.8 mm below the rim**, 16.9 degrees
from vertical; this is not a persistent high release. On the same post-drop
world, inset's 3/4 mm occupied-pen padding attempts fail, while baseline's 2 mm
fallback finds a **1 mm-clearance** target for the next oil. This is a CPU packing
counterfactual, not an execution success. On 100002, the bundle's first oil tip
settles only **0.1 mm below the rim**, after a +21.4 mm servo height residual.
The baseline's scored target3 has **15.6 mm TCP lag** and nevertheless lands.
On 100003, the cup shifts from (+.2342,+.0043) to (-.1281,-.2456) m even though
relocation reports failure. The diagnosis concerns the four-switch bundle;
these traces are not four independent causal ablations.

Full event details, physical measurements and reproduction commands are in the
[round-2 journal](../journal/2026-11-06-dojo-pen-conditional.md). The conditional
screen deliberately forgoes some known broad-profile gains rather than treating
score alone as a certificate of safe geometry.

## Completed 20-pair comparison

| Cohort | n | A mean ± SD | B mean ± SD | Paired gain ± SD | Wins / losses / ties | Success A / B |
|---|---:|---:|---:|---:|---|---|
| Original dev | 5 | 34.00 ± 37.65 | 34.00 ± 37.65 | +0.00 ± 0.00 | 0 / 0 / 5 | 1 / 1 |
| Previous fresh variant-2 | 10 | 6.50 ± 8.18 | 18.50 ± 12.92 | +12.00 ± 9.49 | 7 / 0 / 3 | 0 / 0 |
| Added fresh variant-2 | 5 | 12.00 ± 16.43 | 12.00 ± 16.43 | +0.00 ± 0.00 | 0 / 0 / 5 | 0 / 0 |
| All development layouts | 20 | 14.75 ± 22.85 | 20.75 ± 22.44 | +6.00 ± 8.97 | 7 / 0 / 13 | 1 / 1 |

Score scale is 0–100; SD is sample SD. The requested pooled comparison gains
**6.0 points**, with **7 wins / 0 losses / 13 ties**; paired-delta SD is 8.97.
Full successes remain **1/20 in each arm**, with **0 success flip wins / losses**.
No original layout loses a point; all five trajectories exactly
match both final A and the round-1 baseline. All eleven inactive-condition
trajectories match A exactly. Every previous development A trajectory matches
round 1, and every previous B trajectory matches its selected earlier branch.

The five added layouts were not scored before the final condition was frozen.
They are **all ties**, including the baseline's 40-point 100043. Thus this panel
shows no regression on five additional draws, but **no new-draw score gain**.
The pooled gain comes from seven of the ten previously observed fresh layouts.
The condition was learned on those fifteen earlier development layouts; this
is development evidence, not a population ceiling estimate or camera result.

Final holder tips (>45 degrees): **5/20 → 2/20**, with **3 tip fixes / 0 new tips**.
The remaining tips, 100031 and 100042, occur in both arms with the condition
inactive. The original 100004 tip regression is absent. All nine active-profile
holders end upright.

| Layout | A | B | Delta | Condition active |
|---|---:|---:|---:|---|
| 100000 | 25 | 25 | +0 | no |
| 100001 | 100 | 100 | +0 | no |
| 100002 | 10 | 10 | +0 | no |
| 100003 | 25 | 25 | +0 | no |
| 100004 | 10 | 10 | +0 | no |
| 100030 | 10 | 10 | +0 | no |
| 100031 | 0 | 0 | +0 | no |
| 100032 | 10 | 25 | +15 | yes |
| 100033 | 0 | 25 | +25 | yes |
| 100034 | 10 | 25 | +15 | yes |
| 100035 | 25 | 40 | +15 | yes |
| 100036 | 0 | 0 | +0 | no |
| 100037 | 0 | 10 | +10 | yes |
| 100038 | 10 | 25 | +15 | yes |
| 100039 | 0 | 25 | +25 | yes |
| 100040 | 10 | 10 | +0 | yes |
| 100041 | 0 | 0 | +0 | no |
| 100042 | 0 | 0 | +0 | no |
| 100043 | 40 | 40 | +0 | no |
| 100044 | 10 | 10 | +0 | yes |

Last logged per-pen outcomes across 80 pens (not independent trials or final
checker counts):

| Last event | A | B |
|---|---:|---:|
| Inside | 20 | 30 |
| Place miss / released outside | 20 | 13 |
| No complete transfer | 32 | 23 |
| Grasp slip | 7 | 5 |
| No opening fit | 0 | 7 |
| Collision refusal | 1 | 2 |

The profile improves placement and holder stability; it does not solve pen/00000
or the full-task ceiling. Inactive scenes retain the baseline's joint-limit
behavior, including its known tracking lag. Inset packing still refuses some
capacity-constrained placements. Defaults remain off.

Exact per-episode fields, condition measurements, action hashes, layout hashes,
runtime/asset versions and all ten original replay outcomes are in
[the JSON artifact](2026-11-06-pen-conditional-g3.json). Raw logs/actions/traces
remain under `~/dojo-data/pen/r2/guard_A20` and `guard_B20` on compute6.


## Validation and limits

All **40** final episodes are stable, with no missing mesh substitutions; both
jobs exit 0 and record source commit `3fda84cd4d2579d25d35039c8e083eba2e2d731b`.
Final episode wall totals: A **2924.3 s**, B **3030.4 s**,
excluding reset/startup; planning totals 2284.9 / 2393.1 s.
Original replay episode time totals 405.4 s. LLM calls/tokens: **0 / 0**.
Both arms use at most two physics light locks, admission used VRAM <=24576 MiB,
10G RAM / 600% CPU scopes and single-thread BLAS. No camera video was made on
these lanes. All 40 raw/settled layout files are hashed in the artifact.

The frozen implementation passes **48 focused tests** and its initial CI passes
**1871 / 187 skipped**. Main was subsequently merged into the feature branch;
all four PEN production files remain SHA256-identical to the measured freeze.
The combined local suite initially reports **1890 passed / 191 skipped / 1 failed**
in 18m37s: an unchanged SO-101 fake-arm test reports an elbow blockage during
reset, before the test installs its block. The entire fake-arm file then passes
**14/14** in 18.13s. The final combined-tree full-suite rerun passes
**1891 / 191 skipped** in **19m25s**. No robot code was changed. Final-head CI
is published in PR #397’s checks.

The separately instrumented broad-profile 100001 rerun reproduces **25 points,
365 steps and exactly the original actions**. At step **74**, target0’s first
full pick approach intersects `pen_holder`; the raised via-pose at z=1.078 m
is unreachable under measured limits. Other approaches intersect the inserted
`target3`. Its 57 pick-plan calls include 14 collision refusals, 12 IK refusals
and 31 feasible picks whose full transfer/relay still does not execute. These
are internal planning calls, not independent trials. The diagnostic is excluded
from the 20 pairs; its exact recorded refusals are in the JSON artifact.
Read-only CPU planning probes are likewise not score trials.


The [conditional camera specification](2026-11-06-pen-conditional-camera-prereg.md)
remains parked. It supersedes the unrun broad-arm proposal, retaining the default
Camera A/B rule: >=10 gain pairs, >=5 pooled score points, success flip wins >=2x
losses and >=2 net wins, plus corresponding guard limits. G3 cannot substitute
for that rendered rule. The generic fill route requires a frozen routing audit
and any additional shared-route guard cells to be registered before camera data.
PR #397 is for lead review and remains unmerged.
