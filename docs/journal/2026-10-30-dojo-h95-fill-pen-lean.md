# 2026-10-30: fill_pen_holder leaning drops: upright-first is null; what is left (stream H)

Follows journal 2026-10-30-dojo-h95-fill-pen-j5. Method:
- G3 mode gt, 0 LLM, on S's v15 dev draws 100030-49.
- Base = `fill_move_lift=clear` + `fill_joint_limits=measured` (be7a798b): 0.247, 0/20.

## Tried: `fill_lean` "fallback" (branch `feat/dojo-h95-fillpen-upright`, not merged)

- **Why.** Leaning drop poses (`_leaning`: 15 / 25 deg) ended inside 8/17 under the base; upright ones 31/34.
- **What it does.** `plan_transfer` ranks the leaning poses only 0.05 per pose behind the upright one, so a better
  grasp score picks a leaning drop. "fallback" plans the upright pose alone first, and the leaning ones only when it
  does not plan.
- **What it changed.** It acts in 17 drops ("no upright drop plans" logged 17 times):
  - drops that end inside: 39/51 → 41/53, only 3 leaning drops left;
  - pens that never got a grasp: 17 → 20, and slips 3 → 8, because the upright-only grasps are worse ones;
  - pens inside at the end: 37 → 36.
- **Result:** 0.247 → 0.260, 4 up / 4 down (each loser lost one tier, ≤ 0.15). Noise; parked.

## What is left (base, 80 pens: 37 inside)

- **No reachable grasp: 17 pens.**
  - 7 lie 0.52-0.58 m from both arm bases on their first try: out of reach for every candidate.
  - 10 come after the holder has gone down: 100031's high release, and holders that a late pen drop knocked over. The
    lying holder's drop targets admit no grasp.
- **No grasp-place pair: 11. Slips: 3. Let go but not inside: 7.**
- Levers in the order I'd try them:
  1. Bring the far pens into reach first: a drag stroke, or a relay spot for "0 reachable".
  2. Keep a standing holder from being knocked by the drops: the drop's approach clear of the holder's rim.
  3. The tilted-grasp holder release (BACKLOG).

## Spend

0 LLM calls. G3 on compute3: 5 chunks of 4, one job at a time.
