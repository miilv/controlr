# 2026-10-27: swap_blocks re-measure and the twin's none_ok (Memory stream)

## Genuine-5.1 B on main (submit_v1, 10 settled layouts 100000-09, compute5)

**10/10.** 128-164 s and 353-419 steps per episode. Before: breadth-2 (proxy B, pre-#78) 2/5.
- The cube pinned on its mat (#78) removed the 18 mm frame-0 bias. The pick-hover refine handles the yaw.

G3 mode A on main: 9/10 (100000-09). The miss is 100000: the first right-arm pick slides 7 mm, the retry holds, then
the place comes in at 7.5 deg relative tilt (place_miss). It fails the same way on 48600bf, so it is not a regression.

## The twin's none_ok
- Official-Path saw every verified segment none_ok on both twin paths.
- The judges now say which criterion each outcome failed (`why`, counted in `Skills.vreasons`, reported as `rejects` /
  `reject_examples` in G3 and runtime verify reports).
- In G3 `--verify 2`, every pick rollout is "rise 0 mm", and both cubes stay at their start poses.
- A re-assignment of identical-asset labels by position never fired, which rules out a label swap.
- stack_blocks verifies fine (6/6 first_ok).
- In the twin envs, the swap cube does not move: a non-simulated pose read for that asset, or the cushion mat holding it
  (BACKLOG). It costs ~72 s of twin time per rendered episode, not points (10/10).
