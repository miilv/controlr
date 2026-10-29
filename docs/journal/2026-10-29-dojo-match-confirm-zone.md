# 2026-10-29 — match_and_pick_from_conveyor: commit to a match only from reads where the head view sees it whole (stream C)

`Skills.belt_confirm_zone` / `RuntimeConfig.belt_confirm_zone`: off (default) | on | uncertain. 0 LLM.

## Why

- **Camera failure classes.** The only rows that ran the current belt skill are c4b_matchbelt2 off + emb (7f29156,
  10 episodes):
  - 3 successes;
  - 5 closes on air: a phantom or stale track, the nearest real object 190-360 mm away;
  - 1 close on another rider;
  - 1 slip on the match.
- **G3 ceiling.** 24 / 25 on dev, 18 / 20 on v15 variants. The 2 v15 failures are grasp slips (camera, ice cream).
  The 3 lying bottles of v15 (bottle/00047, the public sampler's) are all taken.
- **Where the head reads are right.** Identity is 100 % right at x 0.4-0.7 (beyond the arms) against 5-53 % at the
  belt's entry and around the home grippers (journal 2026-10-28-dojo-belt-wrist-veto).
- **The tracks that reach that zone in those rows (5).** Their identity is right:
  - 5 / 5 from their first ≤ 3 zone reads;
  - 5 / 5 from all reads up to the zone;
  - 1 / 5 from the reads before it alone.

## Change

- **on.** A match counts only once it has ridden `ZONE_MARGIN` (10 cm) past the downstream arm's home hand along the
  belt (robot geometry, not a task number). The downstream arm then picks it.
- **uncertain.** Only a match whose tracker identity share is under `ZONE_SURE` (0.9) waits for the zone. Under G3
  (identity share 1) this is exactly today's path.

## G3 (v15 variants 100030-49, compute3 light)

| arm | success | note |
|---|---|---|
| off | 18 / 20 | 2 slips |
| on | 13 / 20 | the downstream arm's far-reach rim pinches slip on cameras / a plate (5), episodes take 500-650 of 700 steps |
| uncertain | = off under G3 by construction | |

## Use

- Not a default.
- **on** costs 5 / 20 at G3.
- **uncertain** is the variant for a camera A/B, if the belt_wrist_veto A/B (queued) leaves identity or run-out cases
  as the main class.
