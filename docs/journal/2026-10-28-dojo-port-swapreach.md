# 2026-10-28 — Stream P: swap_T's far spot, a carry height the arm can reach (`ported_swap`, default fixed)

**Done** (branch `feat/dojo-port-swapreach`, on top of the merged port #259):
- `RuntimeConfig.ported_swap` / G3 `--skill-attr ported_swap=reach` (default "fixed" = Mikhail's 0.86 m).
- With "reach", the ported swap route passes swap_poses a `carry_z` argument (`planner.swap_carry_z`). It is the highest
  of 0.86 / 0.84 / 0.82 / 0.80 m at which:
  - the left arm reaches the higher-x object's spot top-down, and the right arm the lower-x one's;
  - some tool yaw the skill may use there works (object yaw + 0 / 90 / 180 / 270 deg);
  - IK runs with the harness TCP, from home.
- The skill file is unchanged: `carry_z` is one of its own arguments.

**Why:** G3 mode gt of the merged port (5bfe623, 100000-09) scored 8/10 on swap_T. Both fails were the left arm unable to
place the low-x T on the high-x T's spot (x 0.091 m) at carry_z 0.86:
- IK has no solution from any seed or yaw;
- top-down reach at y -0.16 ends at x ~0.09 m at z 0.86, and at ~0.12 m at z 0.78.

**Result:** G3 mode gt, compute7, settled v0 100000-09, 955afcf, `ported_routes=on ported_swap=reach`: **10/10** (fixed:
8/10). Steps 242-353 of 400. Only layouts whose far spot is out of reach change: 100000 and 100002 get carry_z 0.84.

**Spend:** 0 LLM; 1 G3 job of 10 layouts on compute7.
