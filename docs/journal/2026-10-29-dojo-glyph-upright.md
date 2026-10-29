# 2026-10-29: arrange's wrist looks read the digits upside down; `glyph_upright` (stream W)

Context: the arrange std rows that end "arranged" still failed 7 of 10, and arrange_random stopped at "read: digits
unknown" in 5 of 9 (c4b_largestnumber*_glyph, 3ad78f28). The lead asked for a classification by GT before any build.
0 LLM calls.

**Classification (ids 100005-09 / 100020-24; tiles matched to GT by the episode's perception map, every `digit_read`
event traced)**
- Frame-0 head reads were wrong in 1 of 10 layouts. 100006 read a 3 as a 9 (one head view, margin 0.783), and the confirm
  look corrected it.
- Wrist 6 / 9 flips hit 3 of 10 layouts:
  - 100007 failed on this alone: the confirm look turned a right head 9 into a 6.
  - In 100009 and 100021 the flip came on top of other failures.
- Counting hit 4 of 10: on 5-digit layouts (100008 / 09 / 23 / 24) a digit and a mat were never found, the count bug
  `series_link_raise` addresses.
- Slips hit 2 of 10 (100005, 100006): every value right, a tile lost on the carry. 100021 also had a grasp that never
  lifted.
- arrange_random "digits unknown", 5 rows:
  - 4 are an asymmetric digit (a 3, a 4, a 1, a 3) that stayed low_margin through 6-12 wrist looks.
  - 1 (100021) is infra: every glyph read was "error: RuntimeError". At 3ad78f28 the glyph client was not loud. A
    server error reply became a plain RuntimeError, and the reader turned that into "no read". Since be2e6b6b it raises
    `SideServerDown` (infra row, re-queued).
- Across the 10 std rows, 17 of 18 wrist reads of a 6 / 9 named its twin. 0 / 8 were "confirmed", and 2 / 3 / 4 / 5 / 7
  were mostly low_margin.

**Cause: the wrist camera's roll**
- `regrasp._look_move` puts the hand straight down 18 cm over the tile and tries wrist yaw 0 first. On the URDF's wrist
  camera that pose shows world +y pointing DOWN in the image. Measured roll of world +y (camera FK + IK):
  - head camera: within 16 deg;
  - home wrist: 0-70 deg;
  - look-over at yaw 0: 180 deg.
- digit_v2 learned the training rolls: head crops, and wrist crops from seg_render hover / look poses (the home flange
  tilted). Its rotation augmentation is small ("a 6 turned over is a 9").
- Dev check (digit_v2 on 2394 wrist and 1184 head crops of 100030-49 train / val, free tiles, CPU), wrist crops turned
  180 deg:
  - 6 / 9: read wrong 0.4 % -> 59 %;
  - other asymmetric digits: read 92 % -> 14 %;
  - 0 / 8: still 71 % read.
- That is the runtime pattern. A margin threshold cannot fix it: the wrong reads clear the 0.88 wrist margin.

**What the flag does: `EstimatorConfig.glyph_upright` "on" (default "off")**
- In the perception service's runtime wrist read (`_glyph_view_read`: look-overs, pick hovers, held and post-place
  refines), a flat glyph of the digit model (numerals and operators) is turned before the classifier sees it.
  `glyph_read.upright_deg` measures the angle of world +y at the object in that camera, and `turn_upright` rotates the
  crop by it: a numpy patch around the mask, bilinear, border-replicated like `crop_square`.
- The edge and arm checks still run on the view's own mask. The read and the `digit_read` event carry `turned` (deg).
- Why world +y: the layouts place numerals / operators face up within 45 deg of it (`rotate_deg` 45), and our places
  keep them within 34 deg.
- Untouched:
  - Frame-0 reads.
  - make_kong. Its standing tiles have no world up, and kong_v1b is roll-invariant on the dev wrist crops: 36.2-36.6 %
    read right, 0.2-0.6 % wrong, at 0 / 90 / 180 / 270 deg.
- Offline check, dev crops only:
  - Each crop was turned by 0 / 60 / 120 / 150 / 180 / 205 / 90 / 270 deg to stand for the look-over's roll, then put
    through `turn_upright` and the server's crop.
  - With the turn the reads match the as-rendered crops: 6 / 9 read 0.90-0.91 right, 0 / 8 0.95, others 0.92, wrong
    0.0-0.3 %.
  - Without it, at 180 deg: 6 / 9 0.2 % right and 59 % wrong, others 11 % right.
- Not done: a rendered look-over of 100030-49 on compute3. compute3 renders RGB only through the Isaac 6.0 proxy, a
  two-step job, and the rendered A/B on compute4 measures the real thing.

**Tasks whose skills take runtime wrist glyph reads (the A/B's cells)**
- arrange_largest_number, arrange_largest_number_random: numerals, by `tiles._look_value` / `_confirm` (look-overs), pick
  hovers, held and post-place refines. Gain cells.
- solve_equation: numerals and the five operators (plus, minus, multiplication, division, equal), by
  `complete_equation`'s looks and picks. Guard cell (10/10 now). The operators are symmetric under 90 or 180 deg; its
  numerals may gain.
- make_kong: the discard read (head + nearer wrist), the discard's straight-down wrist look, and the draw read's front
  wrist look. All use kong_v1b and are not turned. Not an A/B cell for this flag.
- None: press_by_number (Geometry number_card, B1's texture compare, no crop classifier); xylophone and every other task
  (no numeral, operator or mahjong object in any dev layout 100000-04 / 100030-49).

**Also in this PR**
- `tests/dojo/test_side_server_loud.py` inspected `_glyph_read_numeral` for the SideServerDown handler. That handler
  had moved into `_glyph_views`, so the test failed wherever cv2 is installed (CI skips it). It now checks `_glyph_views`.
- A behavioural test: an erroring glyph server raises SideServerDown through `classify_mask`, the frame-0 path
  (`_glyph_views`) and the runtime wrist read (`_glyph_view_read`).

**Left for others**
- `arrange_digits` looked at one unread tile up to 6 times from the same pose (12 looks in random 100007 / 08 / 20 /
  22). N's `tile_look_cap` (#328, default off) caps it at `LOOKS_PER_TILE`.
- Random 100021: re-run (Integration).

Next: the rendered A/B on compute4 (Integration), glyph_upright combined with the tile fixes and series_link_raise, on
arrange std + random, with solve rows as the guard.
