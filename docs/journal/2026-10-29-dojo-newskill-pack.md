# 2026-10-29 — stream N: pack_objects_into_box (+ _random), route "pack" (default off)

**The skill** (`dojo/skills/pack.py`, `new_routes` "pack", route-only). The instruction "objects into the box with their
front sides facing left|right" routes to `pack_box`. Each object goes into the box:
- lying as it lies, along the box's y axis, with its front turned to the named side;
- the front is the active "checkpoint" frame's axis, else a shoe's toe (the lower end of its mesh);
- the slots are lanes across the box's floor, widest first; an object with no lane left goes on top in the middle;
- the grippers open after each object, since the checker credits 1 / 2 / 3 / 4 objects as 10 / 25 / 50 / 100.

**The old blocker** (the first G3 run produced no episode in 13 min). Every object had 0 grasp-place pairs, so
`place_oriented` went into relay searches of 35-190 s each, which also failed. The cause: the box was a `grip_only`
obstacle, and the grip_only check uses its solid bbox, so a gripper anywhere inside the box "hits box". In a CPU
replay of 100030 that was 0 of ~2000 pairs.

**The fix.** The box is an obstacle by its mesh's height map (`container_obstacle`, its open inside free), with no
`grip_only`. Release is 8 mm over the floor, and at most 2 relay spots. The CPU replay of 100030 then has 2 of 4 in, in
112 s of planning.

**G3 mode gt, v1 100030-39, compute3** (32c8294c, `--routes pack`; the cells are at 0 today, Plan none):

| cell | successes | mean | partial credit |
|---|---|---|---|
| pack_objects_into_box | 0/10 | 0.125 | 7/10 rows (1 item x5, 2 items 100038, 3 items 100030) |
| pack_objects_into_box_random | 0/10 | 0.085 | 7/10 rows (1 item x6, 2 items 100038) |

A second arm (pre-place with the object 2 cm over the rim, and no second try after a planning failure) gave 0.10 /
0.071 at n=8 / 7: not better, so it was dropped.

**What is lost.**
- The shoe never fits. It is 184 mm long and the floor is 200 mm, so only the exact centre at 0° clears the walls.
- The box sits 0.45-0.55 m from both bases (y ≈ 0 in every layout). Objects on the far arm's side need a relay, and
  those often fail.
- Every failure is ik_fail at planning.

**Routing guard** (`scripts/skills/route_guard.py`, offline, CPU). Inputs: 1018 compute4 camera planning worlds across 43
tasks, our 40 v1 pack layouts, and all 66 public instruction templates of the 54 tasks.
- The route fires 20/20 on each pack cell and 0 times anywhere else.
- In the cross check (every instruction × one world per task) it fires only for pack instruction × pack world.
- No camera report has a pack row or a Rigid "box".
