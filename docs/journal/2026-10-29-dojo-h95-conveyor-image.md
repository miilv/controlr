# 2026-10-29: pick_from_conveyor_by_image, taken over from C: the classes, and a hook lift that rarely engages (stream H)

Method:
- G3 mode gt (physics-only twin, GT world, 0 LLM), traced, on main 680ab86e.
- Layouts: my own v15 dev draws 100030-49 (variant 1; raw records, like the measurement set), with C's sequence:
  `belt_container=on`, `belt_deposit=on` (journal 2026-10-29-dojo-conveyor-deposit).
- Dev ids only. Time-boxed at 3 h (the lead).

## Base (C's belt_deposit, n = 20): 1/20

| where it ends | n |
|---|---|
| the lift (14), of which: | 14 |
| - the handle pinch rose the basket 0-2 mm while the hand rose 9 cm | most |
| - no grasp of the handle (ik_fail) when the drag left the basket out of reach | several |
| the target pick: 4 "no intercept for target", 1 slip | 5 |
| success: 100037, latched at the drop (as in C's run) | 1 |

- "target in basket after the drop" holds in most of the 14 lift endings. The pick, drop and stop mostly work, and the
  lift is what binds.
- The basket asset is one: `Rigid/basket_grasp/00002`, 0.2 kg.
  - The handle bar is 9-11 mm thick and 146 mm long, ~9.5 cm above the rim.
  - Its collider is a single `/collision/model` mesh with `convexDecomposition` (stage query in the twin). So the bar
    the pads close on is whatever hulls the decomposition made of the arch, not the visual 11 mm bar.

## Tried: `belt_lift` "hook" (branch `feat/dojo-h95-conveyor-image`, not merged)

- After the drop, the picking arm slides its closed fingers horizontally under the middle of the handle bar, across
  it, aimed where the bar will be when the hand arrives (the basket drifts). The stopping arm lets go, and the hand
  lifts straight up: the bar should rest on the fingers, with no friction needed.
- Snippet prototypes on 6 layouts:
  - one lift of 95 mm (100031, first version);
  - one full success (100043, the checker latched on the hook lift).
- The G3 screens (20 draws each) do not repeat them:

| arm | success | lift endings |
|---|---|---|
| base (handle pinch) | 1/20 | 14 |
| hook | 1/20 | 14: rose 0-13 mm (11), hook pre ik_fail (1), hook in partial (1), in ik_fail (1) |
| hook + drag to y -0.16 (from -0.10) | 1/20 | 13: more ik_fail (the horizontal approach from the robot side runs out of room) |

- Why it misses: the basket is not held still. Between the drop and the hook it moves 6-22 cm, partly the belt pulling
  it through the wall pinch and partly the hook's own approach. So the fingers arrive beside the bar, not under it. The
  hand reached its target exactly (0 mm TCP error); the basket had moved.
- **Parked.**

## Open, if picked up again

- Hold the basket still before any lift. It must be off the belt or truly clamped: the wall pinch slips along the
  belt.
- A collider-aware handle grasp: read the cooked hulls of `/collision/model` (convexDecomposition) and grasp or hook
  what is really there.
- The 5 target-pick endings ("no intercept for target").
- C's journal notes the camera rows ran on stale layouts. Perception must find the basket on 5.1 before any of this
  scores (BACKLOG).

Spend: 0 LLM calls. G3 on compute3: 3 screens of 20 episodes and 7 snippet probes; one job at a time.
