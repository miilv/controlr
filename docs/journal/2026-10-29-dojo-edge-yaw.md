# 2026-10-29: align_blocks, a cube's yaw from its outline edges (stream W)

**Camera failures** (candidate c4b_alignsg_par, 7/10). All three end with the belief reading the row square while the
checker's 8° in-line test fails:
- 100021: cube1's frame-0 yaw was 17.8° off;
- 100024: cube1's frame-0 yaw was 35° off;
- 100020: after a push the follow-up check didn't adopt the fix, yet the row was still reported 0/0 (skill/runtime
  accounting, passed to N).

A 4-5 cm cube's silhouette barely constrains its yaw.

**`EstimatorConfig.edge_yaw`** (default off, `dojo/perception/edge_yaw.py`). For a four-fold-symmetric shape whose
config rotates it (`rotate_rand`), each yaw in the 90° period is scored by the image gradient along the rendered outline
band, pooled over the views, with ±2 mm xy jitter and other objects / the robot masked. The yaw changes only when all of
these hold:
- the best yaw beats the current one by 5%;
- it beats the best yaw ≥ 20° away by 5%;
- the silhouette loss rises by at most 0.02.

**Gate** (d0_51 / d0_ext / p60, both arms on current defaults; found / ok10 / phantoms identical everywhere):

| task | cubes >8° yaw error | better / worse (>3°) | mean yaw error |
|---|---|---|---|
| align_blocks (30) | 3 → 0 | 4 / 0 | 5.0 → 3.7° |
| stack_blocks_by_language (29) | 4 → 4 | 2 / 1 | 4.7 → 4.2° |
| stack_blocks, cover_blocks | unchanged | 0 / 0 | unchanged |

**How the gate shaped it.**
- The first version (no guards) turned 7 stack_blocks blocks by 27-40°. Those blocks are placed unrotated by the config,
  so their yaw is known; config-unrotated objects are now skipped.
- The silhouette guard keeps a fit that is 8-32 mm off from taking a wrong turn.
- The one remaining worse case is a stack_blocks_by_language cube already 20 mm off in xy, whose yaw error goes 19 → 32°.

**Not done yet.** The post-slide confirm (the runtime's `confirm_pose`) does not use the edge yaw.
