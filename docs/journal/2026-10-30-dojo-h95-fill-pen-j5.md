# 2026-10-30: fill_pen_holder drops: the joint-5 stop; `fill_joint_limits` (stream H)

Follows journal 2026-10-30-dojo-h95-fill-pen-lift. Method:
- G3 mode gt, 0 LLM, traced, on S's v15 dev draws 100030-49.
- Base = `fill_move_lift=clear` on 165e0076: 0.190, 0/20.

## The drops (52 pens let go over the holder, base)

From the traces, per release: the holding arm's joint-5 command against the measured joint, and the TCP's distance
from its plan (FK).

| at the release | pens | ended inside |
|---|---|---|
| \|q5\| commanded within pi/2 | 31 | 17 |
| \|q5\| commanded past pi/2 (1.58-1.90; the sim holds 1.57) | 21 | 3 |
| of those, TCP > 20 mm off the plan | 15 | 0 |

- The stop is the one journal 2026-10-28-dojo-h95-fill measured.
- A pen let go 20-77 mm off its planned spot lands on the rim:
  - its leading end catches the rim, it pivots and falls out (100038: the pen tilts 0 → 107 deg at the rim's height);
  - or it drops beside the holder.
- `joint_limits` "measured" fixes this for the whole episode, but it is not in the candidate: it costs planning time
  on every task.

## `fill_joint_limits` (new switch, default "inherit")

"measured" runs `fill_container` with joint 5 inside the sim's stop (`kinematics.JOINT_LIMITS_MEASURED`, ±1.56). The
episode's range is restored afterwards, also on an exception. The holder move runs under it too.

## Screen (the same 20 ids)

| arm | mean | up / down | pens inside | let go but not in (place_miss) | plan_s total |
|---|---|---|---|---|---|
| main (no flags) | 0.125 | | 21 | 11 | |
| fill_move_lift clear (base) | 0.190 | | 29 | 14 | 2031 |
| + fill_joint_limits measured | 0.247 | 7 / 2 vs base (12 / 2 vs main) | 37 | 7 | 2770 (x1.36) |

- **Losers.** 100048 0.25 → 0.10 and 100049 0.10 → 0. Both are high-hover drops, the class below: a pen let go 22 mm
  over the rim leaned out; one stuck at the rim.
- **What remains: the leaning drops.** Under the flag, drops whose hover is ≤ 8 mm over the rim end inside 31/34.
  Those that hover 22-25 mm, 8/17.
  - The 22-25 mm hovers are `_leaning`'s poses: 15 / 25 deg, the leading end 12 mm inside the opening. The servo
    hovers `hover_clear` above them, and `fill_container` takes them only when no upright pose plans.
  - Next: a leaning drop that holds the pen until its leading end is inside, or upright poses for more grasps.

## Spend

0 LLM calls. G3 on compute3: 5 chunks of 4, one job at a time.
