# ORG: organize-table reach approaches

**Result:** a modest partial-score gain, without any full-task success.
Ten generated dev pairs: **42.5 ±12.08 -> 47.5 ±14.19**, sample SD in points.
Score wins/losses/ties (difference >=10 points): **3/1/6**, net gain **5 points**.
Successes **0/10 -> 0/10**, success flips 0/0. Median steps **279 -> 539**,
maxima 379 -> 679, limit 1000. This qualifies only as a candidate for the
[current score-based camera rule](../EXPERIMENTS.md), not camera adoption.

## Method and provenance

Physics-only G3 mode A, measured object state, no verifier or LLM. Compute6
initially exceeded 20 GiB used VRAM. Used the explicitly authorized compute3
fallback after checking who/GPU and reading the driver incident. RTX 5090,
driver 595.99.02. No host changes; both members of every pair stay on this host.
Private source deployments under `~/controlr-org`, artifacts `~/dojo-data/org`.
At most two workers, light7/light2, 8G RAM / 600% CPU scopes, disk guard,
30-minute timeout with KILL grace. Subsequent admission receipts record who,
GPU users, <=14000 MiB used VRAM and >=25 GiB available RAM before launch.

Original dev 100000-04: existing settled records copied by explicit id.
Fresh private simlab/v15 variant-3 draws 100030-44: 10/15 pass two-pass settling;
excluded 100032-35 and 100041 before outcomes. The first ten survivors and
SHA256 manifest were frozen before paired runs. No protected IDs were used.
Both arms run source `aa6a7751`, in one immutable candidate deployment.
A is all defaults; B sets only `arrange_hover=low`, `push_approach=tilted`.
Each arm runs two five-layout chunks with a 90-second yield. Chunk order was
A then B, both chunks; execution times are not a controlled speed comparison.

The initial five-layout all-off baseline predates the flags (`18d97e56`), but
its deployment stamp mistakenly reads `ebcbca`. The isolated tilted pilot lacks
a stamp; its actual policy source matches `aa6a7751`. These receipt errors are
retained, not silently repaired. Neither affects the same-source ten-pair panel.
Merged-main smoke `aaf7df34` repeats B 100030 exactly: 50 points, 506 steps.
Later lateral-route source `6e2315e8` is a separate one-layout pilot.
All [raw episode records, manifests and admissions](2026-11-06-org-g3-results.json)
are retained; actions/traces and private layout bytes remain on compute3.

## Switches, all default off

- `arrange_hover=low`: arrange uses 5 mm pre-height and 3 mm support clearance,
  retaining checked approach and placement tail. Final targets do not change.
- `push_approach=tilted`: append 65/50-degree face approaches after original
  vertical candidates, compensating TCP position to preserve fingertip contact.
  Face height, stroke length, execution rates and existing checks remain.
- `arrange_route=lateral`: independent, excluded from B. After legacy transit
  collides, try four fixed xy detours with both legs checked. Unreachable
  endpoints refuse directly; no partial detour executes.

## Paired scores and steps

| Variant-3 ID | A | B | Steps A/B |
|---|---:|---:|---:|
| 100030 | 50 | 50 | 279/506 |
| 100031 | 50 | 50 | 256/572 |
| 100036 | 50 | 50 | 211/280 |
| 100037 | 50 | 50 | 279/632 |
| 100038 | 25 | 25 | 133/129 |
| 100039 | 25 | 50 | 379/615 |
| 100040 | 50 | 75 | 286/679 |
| 100042 | 50 | 50 | 323/586 |
| 100043 | 25 | 50 | 224/407 |
| 100044 | 50 | 25 | 308/159 |

B's gain is not attributable to either switch alone: only the bundle has ten
pairs. The original 100000 single-switch screens both score 50 vs A's 50:
tilted 223 steps (identical), low hover 213 vs 223. No additional favorable
repeats or outcome-based exclusions were used.

## Failure census

These are the final arrange-call outcomes per item, **not checker credit**.
The checker scores the best simultaneous item tier with open grippers at any
step, so a later failed skill or fallen item can still have earned credit.
No-plan categories combine native IK and checked-path refusal; they do not prove
global physical unreachability. Place misses use the skill's own target tolerance.

| Item / outcome | Original five A | Generated ten A | Generated ten B |
|---|---:|---:|---:|
| Mouse: skill ok | 5 | 10 | 10 |
| Figurine: no complete plan | 3 | 4 | 4 |
| Figurine: executed pick failed lift | 2 | 6 | 5 |
| Figurine: placement precision miss | 0 | 0 | 1 |
| Alarm: no complete plan | 0 | 3 | 3 |
| Alarm: executed pick failed lift | 0 | 0 | 1 |
| Alarm: carry loss | 0 | 0 | 1 |
| Alarm: placement tilt/position miss | 3 | 6 | 1 |
| Alarm: skill ok | 2 | 1 | 4 |
| Keyboard: no finishing stroke | 5 | 9 | 3 |
| Keyboard: placement/convergence miss | 0 | 1 | 7 |
| Episode step budget exhausted | 0 | 0 | 0 |

Original A is **50 ±17.68**, 0/5 success, 223-368 steps. Figurine failed lifts
rise -41/-15 mm instead of 80, with 224/127 mm in-hand drift. Alarm's three
misses tilt 15.2-17.8 degrees. In original 100000 the figurine has two screened
pairs but its approach target fails IK. Low hover raises that to twelve pairs,
still with no complete route. Wider 55/65-degree candidates reach 32 pairs in
a separate CPU reconstruction, but remain blocked in the full approach.

B 100037 reaches placement but misses by 50.1 mm / 28 degrees, with a 26.8 mm
hover attachment shift. B 100044 loses the alarm during its giver carry (35 mm),
separate from B 100030's failed alarm lift (11/100 mm, 87 mm drift).

The lateral CPU probe finds a complete route for the original high slice grasp
(z 178 mm, dir 135, tilt 40) through (+80,-100) mm from the goal. Physical
`low + lateral` on original 100000 picks it (6.8 mm initial drift), then loses
it during transfer: 169.2 mm hover attachment shift, final xy error 152.7 mm,
tilt 82.7 degrees. Score remains 50, 300 steps. This is a **rejected pilot**,
not a validated ceiling gain; it is excluded from the camera candidate.

## Disposition, validation and spend

Retain defaults off. The [parked camera pre-registration](2026-11-06-org-camera-prereg.md)
uses only low hover plus tilted pushes and the current score-pair rule, as
explicitly requested. There is no camera adoption or full-success ceiling claim.
The figurine's hold and the keyboard's final convergence remain dominant gaps.

29 completed G3 episodes: 5 initial, 20 paired, 2 single-switch pilots, 1
merged-head smoke, 1 lateral pilot. All stable, no infrastructure failure or
missing meshes. Recorded episode+load time 2429.39 seconds (~0.675 lane-hours),
plus settling, startup, queues and CPU diagnosis. LLM calls/tokens/spend **0/0/$0**.
No camera frames/videos were generated. Local footprint ~160 MB, below 1 GB.

Local pre-lateral full suite: **1961 passed / 191 skipped**; latest focused
switch/route/legacy suite **60 passed**. Compilation passes. Pyflakes has only
pre-existing unused imports and one unrelated placeholder-free f-string, no
new undefined names. Final-head CI is checked before handoff. PR stays open
for lead review; do not merge.
