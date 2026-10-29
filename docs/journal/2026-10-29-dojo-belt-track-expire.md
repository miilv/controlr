# 2026-10-29 — belt tracking: a track expires (stream C, from Misha's G5 review)

`RuntimeConfig.track_expire` / `RuntimeProvider.track_expire`: off (default) | on. 0 LLM.

## The finding

Misha's review (robot-harness docs/notes/G5-VERIFICATION-2026-10-29.md, item 8 for Ilia) found three faults in
`belief.py` `_track` / `_predict`:

- a belt track no read matches is dropped only after `track_lost_reads` reads with the arms still (and nothing in the
  jaws). While the arms move it never expires;
- `_predict` extrapolates it on from its old fit for as long as it lives;
- a rider read again away from that prediction joins as a newcomer, and the same-asset ghost stays as a second
  "next one" of its identity.

This is the likely mechanism behind match_and_pick's main camera class: 5 of the 7 failures of c4b_matchbelt2 closed
the jaws on air at a stale or phantom track (journal 2026-10-29-dojo-match-confirm-zone).

## Change (on)

- **Expiry.** A track no read matched for `track_window` (40) steps expires, whatever the arms do (event
  `track_expired`).
- **No extrapolation past the window.** `_predict` leaves such a track where it is until the next read expires it.
- **Ghost retirement.** When a newcomer joins, a same-asset track that this read did not match and whose prediction lies
  within `track_ghost_m` (0.15 m) of the newcomer retires (event `track_ghost_retired`).
- **Off.** Unchanged.

## Tests (`tests/dojo/test_track_expire.py`, Misha's case)

One rider at 4 mm / step. The reads lose it for 40 steps while our arms keep moving, then read it again 10 cm short of
its track's prediction (the arm held it back):

| setting | objects of that asset afterwards |
|---|---|
| off | 2 (the ghost + the newcomer) |
| on | exactly 1, where the rider is |

- **Short gap (16 steps).** The ghost retires when the newcomer joins.
- **Lost for good.** It expires; nothing is extrapolated.

## Next

A camera A/B on match_and_pick, stacked with the belt_wrist_veto A/B (queued).
