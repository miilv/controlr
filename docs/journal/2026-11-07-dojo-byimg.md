# BYIMG: braced hook development

Read CLAUDE, DEVELOPMENT, ARCHITECTURE, EXPERIMENTS, prior H/C/belt-blind notes,
BACKLOG, points-table board context, PR #264 and the historical hook branch.
The installed checker has no partial-credit ladder; see the
[registration](../experiments/2026-11-07-byimg-g3.md).

Private deployment/data: compute3 `~/controlr-byimg`, `~/dojo-data/byimg`.
Physics only after reading both driver incidents. No renderer, model calls,
sealed data, compute4, system changes or other users' processes touched.
The first attempt queued on light7; cancelled only own waiting shell and its
flock child, then used light3/light4. At most two jobs. Admissions logged.

Ordinary settling rejects common belt travel. A separate belt-relative snapshot
script retains that diagnostic and uses existing raw-at-rest checks instead.
Fresh salted variant-7 draws are disjoint from historical episode inputs.
All new runtime behavior is opt-in. Perception and candidate are unchanged.

Pilot 100021 also failed its target intercept (donut slip followed by no
intercept). Isolated lift probe 100020: hook bottom rise 4.4 mm; near-wall-only
stop dragged basket to y=-0.071 but hook rise was -2.1 mm. Deeper handle pinches
(40/60/80 mm) closed but their 130 mm lifts failed IK. None is a working chain.
The registered comparison remains frozen at e68dd094; exploratory probes are
not panel observations and cannot be counted as successes.


Completed ten-pair panel: A/B mean 0/0, sample SD 0/0, successes 0/10 each,
score wins/losses/ties 0/0/10, no success flips. Both have eight target-pick
endings and two lift endings. Hook rises 5.2/0.7 mm. No missing mesh or infra
failures. No camera spec: the gain gate fails. Raw episodes and exact frozen
inputs are checked into the report's companion JSON; traces remain in private
data. Footprint after panel ~1.1 MB data / 41 MB deployment.

Opposite-wall-only probe: all 80 grasps fail IK. Farther pull toward y=-0.20
slides through the wall pinch (actual basket y=-0.055) and still has no far-wall
IK. A working physics chain remains unresolved; do not infer success from the
new route or mocked tests. Settled files keep post-load belt phase and reset
advances it again; this explicitly limits historical raw-v15 comparability.

## Round 2 — active development

Lead explicitly requested sustained, multi-approach work and kept #414 unmerged.
Fresh isolation draws: v15 variant 9, salt `byimg-r2-isolation-20261107`, ids
100000–09; separate fresh pilots 100022–23. All private data under
`~/dojo-data/byimg/r2`, same 8G/600% scopes, per-start admission and light3/light4
locks. Physics-only, zero model calls. Round-1 files remain frozen.

Fifty physical isolation episodes (ten fresh layouts × five profiles): free
both-arm 125 mm lift 6/10, free left 125 mm 6/10, free left 150 mm 5/10;
after stopping basket, left 125 mm **9/10**, left 150 mm **8/10**. At 125 mm
the one miss is a slip; at 150 mm there is additionally a no-intercept ending.
The old matchbelt reference is `2026-10-27-dojo-memory` §6: 24/25 after a
second grasp family and partial-lift top-up. Its lift is 125 mm, not the image
chain's 150 mm. Narrowing intercept to a central arm window is also being
tested; early data do not support it as a universal fix.

Basket-first: six lift/pull combinations on right with inherited bounds, then
six on left and the later corrections/regrasps with measured bounds. No bound
was loosened. Origin lift alone is misleading: a 111 mm
origin lift left bottom only 31 mm above start at ~27 degrees tilt. Right-arm
tracking error there is <0.4 mrad. An actual measured-pose correction to
x=±0.05,y=-0.18 with intended bottom 0.88–0.90 does hold bottom at
0.858–0.878 m with 4–5 degrees tilt. Three subsequent left-arm target picks
succeeded. No checker success yet: drops/carries miss. Extra 30 settling steps
did not recover them.

Transfers/regrasps after the rim lift: band 210 sites / 96 endpoint-reachable,
arch 13 / 6; tested first reachable grasps closed but lost the basket after
release, despite final joint tracking within microradians. A hook insertion
returned partial. Level-pose correction alone also sometimes slips the wall.
Two left-arm rim examples tilt >40 degrees and do not retain a level hold.

Accessed actual cooked geometry through PhysX's
`request_convex_collision_representation`: **64 convex hulls**, valid result.
The older get_nb_convex_mesh_data/get_convex_mesh_data methods are unavailable
in this installed API. Collider analysis uses returned hulls, not an assumed
solid convex basket. No collider or physics parameter was modified.

Current probes distinguish target loss during transfer from release-point
misses. A slow center-drop trial measured empty picking fingers before opening:
the target was already back on/near the belt. Also found `belt_container=on`
leaks the basket-specific box-rim grasp family into target picks; the next
controlled trial scopes it off for the target while retaining the basket grip.
A depth/tilt/end-wall matrix is testing whether the stable raised basket grip
can generalize beyond one pilot.

Round-1 full local suite completed: **2058 passed, 191 skipped, 18m19s**.
CI passed on e68dd094; the newer evidence-only head needs a fresh CI trigger.

The complete end-wall prototype on pilot 100023 scores **1.0 at step 179**
(the environment terminates during placement). The clean implementation
`belt_first=on,belt_target=match` reproduces **1.0 at step 177** in G3, with
no instability or missing mesh. This is one tuned pilot, not a rate claim.

[Round-2 registration](../experiments/2026-11-07-byimg-g3-r2.md) is committed at
`db52accf` before any paired outcomes. Ten new variant-11 draws, ids 100010–19,
passed native-phase settling; A/B launch from that exact frozen source. No
camera dispatch, board change or merge. Full local suite on this implementation
is running. The central-window alternative completed 5/10 in both native
profiles and is rejected; no central-window code enters the candidate.

First registered round-2 panel: **10.0 ±31.62 → 20.0 ±42.16**, successes
1/10→2/10, score/success wins 2, loss 1, ties 7. B succeeds on 100012/17;
A on 100016. B ends at target pick (3), carry/drop (4), basket intercept (1).
The mean improves, but one net win fails the registered two-net-win condition.
A once-only independent 14-pair retest is registered in a separate report;
no tuning from these paired inputs enters it. All new behavior stays off.
