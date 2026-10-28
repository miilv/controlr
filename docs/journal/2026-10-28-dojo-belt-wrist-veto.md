# 2026-10-28 — match_and_pick_from_conveyor: a wrist look that finds nothing vetoes the belt pick (stream C)

## Diagnosis (compute4 camera rows c4b_matchbelt2_off, 7f29156, 2/5; G3 is 96)

- **Episode by episode.**
  - 100001: B1 split the first rider at frame 0 into an ice cream plus a phantom "camera" leading it, so the
    remembered identity was wrong.
  - 100001 / 03 / 04: the pick went to a believed "same" rider whose track had no real object under it. dev_gt_probe
    puts the nearest object 269, 38 and 193 mm from the closing jaws.
- **The head-view belt reads are poor where the riders arrive and where they are picked** (physics-only replays of the
  5 episodes, GT every 5 steps, `track_read` entries matched to the nearest GT rider):

| rider's GT x | reads | identity right | xy error p50 / p90 (mm) |
|---|---|---|---|
| −0.8 … −0.1 (entry, our left arm) | 253 | 5-53 % | 24-70 / 63-157 |
| 0.0 … 0.3 (our right arm) | 29 | 6-80 % | 5-22 / 38-145 |
| 0.4 … 0.7 (beyond the arms) | 71 | 100 % | 7-9 / 10-13 |

  Our home grippers sit between the head camera and the belt at x ≈ ±0.3, and the entry is at the edge of the view.
  The tracker's per-track line fit and identity votes inherit that noise: identity flips (phone ↔ camera ↔ ice cream),
  tracks off by 5-15 cm.
- **The wrist look at the pick's hover is already there and separates the outcomes.** Over the 14 belt picks of all
  10 c4b match episodes:
  - every pick after a look that found nothing missed (8/8: not ok, IoU < 0.1, not "truncated");
  - no pick of the 3 successful episodes had such a look (0/3).

## Change (`RuntimeConfig.belt_wrist_veto` / `Skills.belt_wrist_veto`, default off)

- `pick_from_belt`: with the switch on, a wrist look that finds nothing at the track (`wrist_look_absent`) aborts the
  pick before the descent, with status "absent".
- `match_and_pick` then marks that track absent and waits for the next match, up to 3 rounds. Off: exactly as before.
- This is the negative-evidence rule (the lead, from A1): no descent on a track the close view does not support.

## Not yet measured

- What the robot does after a veto (the next match, picked) needs the rendered A/B: offline, the actions are fixed.
- Left for later: the frame-0 split of the first rider (100001), and identity votes only from the good zone
  (x > 0.4).
