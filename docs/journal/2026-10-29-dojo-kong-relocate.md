# 2026-10-29: make_kong's discard lies 48 mm from where the belief put it; `watch_relocate` (stream W)

Context: the make_kong candidate row (kongon2, b131264e) was 9/15. The 6 failures stopped at "roles: the discard's
glyph was not read". The lead asked whether the head view or the wrist look failed, and for the cheapest fix, sized on
dev frames. 0 LLM calls.

**Provenance (measurement-integrity rule, 29 Oct)**
- The failure class was read from the 6 failing kongon2 run reports (100005 / 06 / 22 / 23 / 26 / 28). That was before
  the rule that measurement ids are for scoring only.
- Everything after that is dev only: the cause, the fix, its thresholds and its checks used make_kong 100000-04 /
  100030-49.

**What the 6 rows show**
- The read ran at step 64.
  - Head view: "back" (4) or low_margin (2).
  - Home left wrist: "edge", the tile's silhouette cut by the frame.
  - The straight-down look never ran: "no look pose for either arm".
- So it is not kong_v1b's weak wrist read.

**Cause: the belief lays the knocked tile where it stood**
- `RuntimePolicy._lay_face_up` lays the knocked push tile face up at its slot (y 0.05).
- Dev renders of all 25 dev layouts: physics-only Isaac 5.1 with the arms held to step 64 / 90, then the Isaac 6.0
  proxy, scripts in `~/dojo-w-kong` on compute3. In 25 of 25 the tile lies 47.9-48.3 mm toward the robot (y 0.002),
  flat, turned ~178 deg.
- The read crops the head view at the slot, so it saw a standing neighbour's back ("back"), or only part of the tile.
- S's v3 assumption ("past the standing ones, y >= 0.1") came from the same belief.
- The dev renders reproduce the camera row: at the believed pose the head read is right in 15/25 at step 64 (60 %).
  kongon2 was 9/15.
- Other views do not help:
  - Home wrists: the left one has the tile cut by the frame or misses it; the right one does not see it.
  - A straight-down look: no IK at y 0.002 either, for either arm at 18 / 14 / 11 cm and 4 wrist yaws.

**What the flag does: `RuntimeConfig.watch_relocate` (default off; needs `watch`)**
- After the watcher lays a knocked object face up where it stood, the perception service locates it in the head view at
  each later watch read (every `watch_every` steps). The op is `locate_laid` (`task_state.laid_pose`).
- The search: a coarse 10 mm then a fine 2.5 mm grid within ±70 mm of the laid pose, in both directions. No fall
  direction is assumed. Two turns about z.
- The score: IoU of the silhouette with the pixels that changed since frame 0 and are now tile-face white (Lab change >
  12, HSV value > 150, saturation < 70), counted only near the laid pose.
- A location with IoU >= 0.4 is written to the belief. Two in a row within 3 mm end the search; so do 12 reads.
- Uses images and the laid (config-slot) pose only. No task name and no fall offset.

**Dev checks (100000-04 / 100030-49 only)**
- Static, 50 frames (25 layouts × step 64 / 90): located 50/50 within 3.5 mm, IoU >= 0.83, 0.08 s per call.
- Head read at the located pose:

| frame | right | wrong | unread | of |
|---|---|---|---|---|
| believed (slot) pose, step 64 | 15 | 2 | 8 | 25 |
| located, step 64 | 19 | 4 | 2 | 25 |
| located, step 90 | 22 | 1 | 2 | 25 |

- The watcher replayed on frames every 4 steps, 10 layouts (100000-04, 100030-34):
  - The generic watch read lays the tile at step 24-32.
  - Locations while the tile still moves are off by 41-57 mm, but none agree twice.
  - The search settles at step 36-48, and the belief is within 0.3-3.5 mm of the step-64 GT in 10/10.
- The first version scored the whole view. The opponent's white arm, mid-push, left nothing locatable until step 56-64,
  so the match now counts only new white pixels within the window plus 3 cm.
- Remaining errors are head misreads of a ~20 px face-up tile: 0→2, 1→2, 8→2, 29→19, 35→40, 12→11. Averaging 27
  shifted crops or two frames does not help. That is a reader matter (S), parked until the A/B.

**Not changed**
- Flag off: the tile is laid where it stood, as before.
- The read itself (`kong_read`, `_glyph_view_read`) is not changed. It crops at the belief pose, or at the wrist
  refine's pose when that refine is ok, max 3 cm.

Next: the lead's make_kong A/B (candidate vs + `watch_relocate`) on 100005-09 + 100020-29.

## 2026-10-30: camera A/B verdict: not adopted, stays default off

Camera A/B (lead's pre-registered rule), make_kong, 15 ids each arm:

| | all | unseen ids | identified ids |
|---|---|---|---|
| relocon (`watch_relocate`) | 9/15 (60) | 2 wins / 1 loss | 2 wins / 0 losses |
| relocoff (candidate) | 6/15 (40) | | |

- The bar was 0 success losers on the unseen ids, so it is not adopted. One unseen pair lost.
- The overall signal is real: +3 successes net.
- The losing id is a measurement id. Nobody may look at it to design a fix.
- The make_kong measurement ids are used up for this lever. Its next real test is D-C19 tranche B (a later freeze).
- A future version needs fresh evidence, on dev frames only (100000-04, 100030-49).
- The same rows also cleared the #330 boundary alert: relocoff 6/15 vs kongon2 9/15 is above the ≤ 4/15 revisit line, so
  it is noise. Code, inputs and env were checked inert.

Ways relocate could HURT, to check on dev before a v2:
- **A mislocated tile:** a location that is wrong but agrees twice (the opponent's arm or a neighbour's new white pixels
  inside the window plus 3 cm) is written to the belief. The read then crops elsewhere.
- **Settling before the fall ends:** two locations within 3 mm while the tile still slides (a slow topple). The search
  stops early on a stale pose.
- **A crop on a neighbour:** a located pose next to a standing hand tile. The head crop takes the neighbour's back or
  face, which turns a wrong read into "no evidence" or a different glyph.
- Cheap hardening to test:
  - require agreement on frames >= 8 steps apart;
  - refuse a location whose silhouette overlaps a believed neighbour by more than ~30 %;
  - keep the slot pose when the located read's margin is below the slot read's.
