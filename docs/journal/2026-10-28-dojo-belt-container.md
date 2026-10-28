# 2026-10-28 — pick_from_conveyor_by_image: lifting the basket off the belt (stream C)

## Why it was 0 / 10

- On current code (main 4309013), G3 on the v13 layouts (100000-04 + 100020-24) is 0/10. Every episode fails at
  "lift basket": `belt_pick` reports "basket rose 0 mm".
- A traced episode shows the cause. `rim_grasps` treats every container as round: a circle of radius min(x, y) / 2,
  24 mm under the bbox top. The basket's bbox top is its handle arch, 9 cm above the real rim, so the jaws closed on
  air 7 cm above the rim.
- The camera rows of this task ran on stale v12 layouts with the basket inside the conveyor; see BACKLOG and the
  layout audit in PERCEPTION_95 ### C.

## The fix, behind `belt_container` (a skill switch, default off; G3 `--skill-attr belt_container=on`)

- `grasps.box_rim_grasps` is a rectangular container's real rim, taken from the mesh:
  - the rim height is the highest slice spanning 85 % of the bbox in both axes, so a handle above it does not count;
  - each wall's thickness is measured there;
  - the pads straddle the wall 12 mm under the rim, the nearer walls first, approached from above (tilt 25-40 deg).
- `grasps.top_band_grasps` is a narrow top (the handle arch). It is a fallback: the handle is out of reach on the
  belt.
- In `pick_into_held`, the basket is no longer pulled 18 cm toward the base during the lift. A partial rise (≥ 3 cm)
  counts as taken. `hold_up` tries farther-out and lower spots.

## Screen (G3, physics-only, compute4 light1; variant records v1 of 100030-39, D-C10c-legal)

| arm | basket taken | basket rise | hold_up | target picked | dropped in | success |
|---|---|---|---|---|---|---|
| off (main) | 0 / 10 (9 "rose 0 mm", 1 no intercept) | 0 | — | — | — | 0 / 10 |
| on, tilts 0-25, pulled 18 cm | 10 / 10 | 46-63 mm | 0 / 10 (IK) | 5 / 10 | 0 / 5 | 0 / 10 |
| on, tilts 25-40, no pull | 10 / 10 | 71-77 mm | 0 / 10 (IK) | 5 / 10 | 0 / 5 | 0 / 10 |
| on, tilts 40-50 | 1 / 10 | — | — | — | — | 0 / 10 |

## Still blocking

- The checker wants the basket more than 8 cm up. The hand at the near rim, about 30 cm out at about 1 m, has no IK
  higher: lift-more and every `hold_up` spot fail on IK.
- The second arm picks the depicted object in 5/10; the other 5 slip on waist / axis grasps.
- The drop over a basket held at the belt has no plan in 5/5.

## Other users of `box_rim_grasps`

- Today only `pick_into_held` lifts a container.
- stack_bowls uses round `wall` / `rim` grasps: unaffected, and its bowls are round.
- classify_objects and pack_objects_into_box have no skill (Plan none). If they ever lift a basket or box, this is
  the grasp to use; fill_egg_holder does not lift its basket.
- Not wired anywhere else.
