# Laptop G3 round 2f — wide priority and checked release clearance

## Production-screen registration before outcomes

Action source **3866ad73**; current main integration73a783bf changes docs only.
The full integrated local suite is running;54 focused mock tests passed.
All new behavior retains off/legacy defaults. No model calls, camera dispatch
or board-candidate change. This file registers development screening only;
a separate frozen fresh-panel registration is required before generation.

The preceding [development report](2026-11-07-laptop-g3-round2e.md) identifies
two separate regressions. A narrow cup can displace an effective wide hold:
standard31 loses237.7 mm from a25 mm cup but wide priority completes a41 mm
hang after original physical rejection/re-grasp. Standard66 similarly improves
from four loose narrow pickups to a44 mm pickup1.1 mm drift/carry1.2 mm and20
points. Separately, mesh-ground staging can eliminate reachable release pairs:
random66 has zero pairs at all8 spots until an explicit25 mm release clearance
is planned. Its checked two-arm transfer then earns20, with1.0/4.7 mm carry
shifts. Cross-shape standard45 retains20 with that clearance; random47 stays0.
These are separate pilot probes, not a validation of the combined bundle.

Freeze this single common bundle in both cells:

```yaml
hang_cradle: straddle
hang_wide: 'on'
headset_seat: 'on'
headset_carry: staged
headset_grasp: cups
headset_table: mesh
headset_stage: mesh_clear
headset_retry: sites
headset_lift: slow
```

`mesh_clear` screens at release height and rebuilds a full checked place
against the true mesh-ground rest. Its pickup repair repeats that process;
it cannot silently keep the raised screening goal. Every physical lift,
put-down, carry and release retains the native validators and original
8/30 mm drift limits. No learned/id-specific routing, forced poses or step
budget change. Existing wide priority retains narrow feasible fallbacks.

Screen all **12 original pilots/cell** (30,31,45–49,65–69) and the **first ten
stable coverage18 records/cell** (30–39 in the separate coverage18 namespace),
using exactly the previously settled/hash-retained files. The latter's random
bank is all asset8: independent placement coverage, not rare-shape coverage.
The original new45–49 bank includes9/10. Neither bank is a fresh primary panel.
Keep both banks and all earlier candidates separate; never pool selected runs.
Write new source/config/hash manifests before the first outcome in each bank.

Complete all22 episodes/cell. Gate for spending the remaining unused panel ids:
**>=9/12 original and>=8/10 coverage credits in each cell**. If either gate
fails, continue development on pilot records; do not reroll or inspect any
unused final-panel layout. The ongoing source504daaef screen remains frozen
and completes in full, with its nulls/regressions retained.

Compute3 physics-only, deployment `~/controlr-laptop`, all data under
`~/dojo-data/laptop`. At most two actual jobs on light2/light8; each admission
records who/GPU, used VRAM<=24576 MiB and disk>=10GB, then an8G/600% user
scope with bounded timeout. No renderer, system changes, other-process kills,
sealed data or compute4 access. Calls/tokens/spend:0/0/$0.
