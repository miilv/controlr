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
