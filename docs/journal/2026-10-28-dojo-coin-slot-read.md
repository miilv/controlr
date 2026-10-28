# 2026-10-28 — deposit_coin: the coin read with its stand at the pick hover; a thin coin is not "dropped" (stream C)

Two flags, both default off: `RuntimeConfig.slot_read` and `RuntimeConfig.held_thin`. 0 LLM calls.

## Diagnosis (compute4 camera rows c4b_coin_off, ad5b50e, 0/5, score 8; G3 is 82)

Compared with the layout GT, and with GT per step from 5.1 seq replays of the same episodes. The replays reproduce the
recorded GT probes: the same closes, the same coin rise.

- **(a) The coin pose at the pick is wrong, and nothing is lifted: 100002, 100004.**
  - The coin is the stand's slot child: perception places it from the stand (`settle_slot_children` /
    `support_slot_pose`), so the coin inherits whatever the stand read gives.
  - **100002.**
    - After the agentic read, the stand's xy is right to 0-1 mm, but its yaw is 48.5 deg off (frame-0 conf 0.14).
    - The edge grasp closed 48 deg across the disc.
    - The coin's own hover refines got IoU 0.24 / 0.09: not ok, not applied.
  - **100004.**
    - The stand is 12 mm off, 11 mm of it along the coin's normal, so one pad lands on the disc.
    - The hover refine wanted a 97 deg turn into a flat-coin pose at IoU 0.68, rightly rejected ("prior pose kept").
- **(b) The coin is held, and the belief drops it: 100000, 100003.**
  - The stand and the bank are on opposite sides, so the handover is forced (in G3 too).
  - The 5.1 GT trace of 100000:
    - the left arm picks the coin (step 65), carries it to the handover pose (0, −0.28, 0.93) and holds it there
      (GT z 0.93 up to step 128);
    - at step 111 `held_look` reads "not held" at confidence 1.0, evidence "opening" (measured 0.0, commanded 0.0);
    - `_jaws_closed` takes it as geometry ("jaws closed") and the belief drops the coin back to the stand;
    - the handover skill then reports "the disc fell out on the carry (373 mm off the handover pose)", the right arm
      goes home, and the left jaws open at step 130: the coin falls only then.
  - The coin is 1.9 mm thick. The opening verdict needs the jaws `HELD_OPEN_MARGIN` (0.1, ~5.4 mm per finger) above the
    closed floor, so nothing thinner than 10.8 mm between the pads can read "held".
  - grip_check hands objects thinner than `grip_thin_mm` (4 mm) back to this vision check.
- **(c) The coin is never found: 100001.** In the stereo arm, 100002 / 04 as well. Not addressed here.

## (a) Why the existing reads cannot fix it

Measured on the frames below, with priors made from the GT stand: turned 0 / ±20 / ±48 / 90 deg, shifted 12 mm, and
the coin re-derived in its slot, as perception's belief does it.

- **The coin's own refine (today's hover refine).** The 2 mm disc never fits.
  - From the GT pose: IoU 0.25-0.36, not ok.
  - When it is ok, it is ok on a flat-coin pose 89.5 deg off: 0 right of 52 ok reads (6.0 render) and 0 of 7 (5.1).
- **A refine of the stand alone with yaw starts over its whole period** (tried, dropped). It lands ~50 deg off at
  IoU 0.80-0.86, better than the GT yaw's 0.73. The stand's top is a table-coloured insert in a thin white rim.

## `dojo.perception.slot_read` (`RuntimeConfig.slot_read`: off | on)

- **The fit.**
  - Base and child are fitted together: the union of their convex hulls against the wrist frame's own foreground
    (`refine.local_foreground`).
  - Foreground gaps are closed up to ~1/4 of the pair's size, and enclosed holes filled (the table-coloured insert).
  - The search is a grid over the base's yaw period (10 deg) and xy within 18 mm (6 mm), plus a coordinate polish of
    the best 3.
- **Acceptance.**
  - On 5.1 RTX the stand's rim shows only at its ends and the coin's face reflects the table, so the foreground covers
    a third of the pair: IoU 0.2-0.3 at the right pose.
  - A read is accepted on shape instead: the silhouette against the convex hull of the foreground next to it
    (IoU ≥ 0.75; 0.95 at the right pose).
  - Plus a yaw margin ≥ 0.04 over the best fit ≥ 45 deg away.
  - Refused when the pair is cut by the image border. Other believed objects within 25 cm are masked out.
- **Speed.** ~1-2 s per read, with hull meshes (the stand's own mesh has 17 k faces).
- **Service.** Op `slot_read` (`slot_read_for`) finds the base from the task config's `support_slot`. It reads only
  while the believed child is still within 4 cm of its slot.
- **Runtime.**
  - At a pick hover, after A1 / wrist_check. An ok read writes the child's and the base's poses and skips the refine.
    Otherwise the refine runs as today.
  - `execute_pick` re-plans the same grasp in the object frame at the new pose.

### Offline

Per-step sequences, pick-hover frames: the wrist camera 10-17 cm above the coin, jaws open, the coin within 220 px of
the view's centre, the coin still in its stand. "Right" means the disc normal is within 10 deg and the offset along it
within 4 mm.

| frames | priors | reads | applied: right / wrong | refused | applied error p50 / p90: normal (deg), along (mm) |
|---|---|---|---|---|---|
| 5.1 RTX, c4b_coin_off 100000 / 02 / 03 / 04 (19 frames) | within the radius (0-90 deg, 12 mm) | 171 | 162 / 0 | 9 | 1.6 / 2.4, 1.9 / 2.5 |
| 5.1 RTX, same frames | 30 mm off (beyond the 18 mm radius) | 38 | 24 / 1 | 13 | 2.1 / 7.7, 2.2 / 3.0 |
| 6.0 render, deposit_coin 100002 / 03 / 06 (23 frames) | within the radius | 204 | 177 / 1 | 26 | 2.4 / 4.7, 2.2 / 2.7 |
| 6.0 render, same frames | 30 mm off | 45 | 36 / 2 | 7 | 2.5 / 6.2, 2.2 / 2.7 |

- **The failing episodes' own pick hovers.** 100002 step 59 (stand 48.5 deg off in the camera row) and 100004 step 63
  (11 mm along the normal): every read from within the radius was applied and right.
- **The coin's own refine on the same frames:** 0 right of its ok reads.
- **The hull threshold.** At 0.6, 14 wrong reads of 200 got through on 5.1; at 0.75, one did, the right ones unchanged
  (186). On 6.0, 0.75 let 2 through, both ~12 deg / 1 mm.
- **The remaining wrong reads** are ≤ 12 deg / 4.4 mm.
- **Coverage.** The tool is `scripts/perception/coin_slot_eval.py`. The 5.1 frames are Integration's replays on
  compute4 (`~/dojo-data/cells_c/coin/seq`, png). The 6.0 frames are the earlier compute3 sequences (jpg).
  - 100001 has no hover over the coin: it was never found.

## `RuntimeConfig.held_thin` (off | on)

- **The rule.** A held_look "not held" resting on the jaw opening alone is unknown (the object stays in the jaws, the
  place / post-place checks decide) when the held object's narrowest extent is below the opening's resolution.
- **The resolution.** `j1_adapter.opening_resolution_gap(cmd)` = `grip_to_gap(max(cmd, HELD_CLOSED_FLOOR) +
  HELD_OPEN_MARGIN)`, 10.8 mm at a full close. It is derived from the verdict's own margin and the finger geometry,
  not tuned.
- **Where it can act** (catalogue assets below 10.8 mm): deposit_coin's coin (1.9 mm), insert_key's key (10.0),
  play_tic_tac_toe's chessmen (10.0), pour_balls' spheres (10.0, not held singly), general_pickup's bottle opener /
  watch / wrench (4.4-10.3).
- **Tests.** The 100000 step 111 reading drops the coin with the flag off and keeps it with it on. A cube with the same
  reading, and the coin with a colour-test verdict, still drop.

## Next

- The rendered A/B on the candidate (Integration):
  - A = candidate, B = candidate + slot_read + held_thin, deposit_coin v0 100005-09 + 100020-29;
  - guards: insert_key and play_tic_tac_toe (thin held objects), plus one unrelated task.
- (c) The coin not found at frame 0 remains (BACKLOG).
