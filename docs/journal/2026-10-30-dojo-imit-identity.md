# 2026-10-30: imitate's frame-0 identity of small objects; `wrist_identity` (stream W)

Context: every imitate miss traced so far comes down to frame-0 identity of small near objects (watch vs garage /
action_camera; dev 100001: our watch read as a garage). The lead asked for the confusion, what separates the classes, and
the cheapest lever behind a flag. 3 h box, dev / training frames only, 0 LLM calls.

**Data: 26 frame-0 views, 260 movers**
- 16 new training-slice draws: seeds 100033-48, variant 3, `seg_render` + the 6.0 proxy render on compute3.
- 4 earlier training draws: 100030 / 31 / 32 / 49 v1.
- Dev 100002 / 100004 renders.
- Runtime frame-0 dumps of the dev rows 100000-03.
- Every layout holds the five categories (watch, garage figure, toy_car, phone, action_camera) once per side.
- B1 runs at the candidate config. "B" = `support_mask` + `wrist_pool` "support_masked" (the bundle).
- Script: `imit_ident_eval.py` (dev only, not committed). It gives per GT mover the nearest estimate within 30 mm, its
  category / asset, the confidence, and the GT and chosen assets' silhouette losses.

**Confusion at B (rows GT, columns estimate; ours + demonstrator)**

| GT \ est | watch | garage | toy_car | phone | action_camera | not found |
|---|---|---|---|---|---|---|
| watch | 26 | 0 | 2 | 0 | **14** | 10 |
| garage | 0 | 32 | 4 | 0 | 4 | 12 |
| toy_car | 0 | 0 | 45 | 0 | 1 | 6 |
| phone | 0 | 0 | 0 | 50 | 0 | 2 |
| action_camera | 0 | 1 | 0 | 0 | 37 | 14 |

- Our side: 16 of 130 read with the wrong category, 8 not found. Demonstrator side: 10 wrong, 36 not found (the
  Franka, 2026-10-29-dojo-imit-detection).
- The dominant error is watch -> action_camera (14). Next is garage figure -> toy_car / action_camera (8).

**What separates them**
- Size doesn't: the watch and the action camera have ~4-5 cm footprints, and the figure and the car ~6-9 cm.
- The head silhouette doesn't: misread objects cover 230-1700 px in the head (median 370). In 12 of 26 misreads the
  true asset at its true pose explains the views better than the chosen fit, so it never got proposed there. In the
  other 14 the wrong shape fits better.
- Colour overlaps: black watch straps vs the black camera.
- B1's own flags don't separate them: `ambiguous` marks 23 of 26 misreads but also 50 of 63 correct watches / cameras.
  A margin-triggered wrist look (`imitate_look` "ambiguous") would therefore fire on most objects.
- **The home wrist views do separate them.** At frame 0 they see each misread object at 0.9-17k px (median 3.1k). The
  watch's band and face, the camera's box and lens, and the figure's face are plain there.
- A DINOv2-B crop-vs-catalogue-render similarity (`embed_id`, MUSE recipe, already in the repo) gives the right
  category a 0.13-0.41 margin over the fitted wrong one.

**What the flag does: `EstimatorConfig.wrist_identity`** ("off" default | "embed" | "support_masked")
- Each chosen table object that a wrist view sees with >= `wrist_id_min_px` (800) gets a read in the wrist view that
  sees it largest. The crop is the foreground component overlapping its silhouette dilated by 15 px, with the other
  objects and the robot removed. The read is the learned identity over its label group's candidates (`embed_model`,
  default dinov2_b).
- If a category beats the fitted one by `wrist_id_margin` (0.12), it is re-fitted at the object's head blob. A
  re-fit more than `wrist_id_max_shift` (25 mm) away is rejected. The re-fit replaces the fit when the fused score
  (similarity minus `embed_weight` x loss) is higher.
- "support_masked" runs only in a frame where `support_mask` masked a support robot. That is a scene-content
  condition, not a task key.
- Diagnostics: `record["_perception"]["wrist_identity"]` (checked / changed / small / time_s), and per object
  `_perception.wrist_identity`.
- Runtime: the perception service needs `DOJO_EMBED_MODELS` holding `dinov2-base` (compute4:
  `/home/haptics/dojo-data/models/embed`).

**Result (same 26 frames, B -> B + wrist_identity "embed")**

| | ours wrong category | ours not found | demo wrong category | demo not found | watch right |
|---|---|---|---|---|---|
| B | 16 | 8 | 10 | 36 | 26 / 52 |
| B + wrist_identity | **5** | 8 | **5** | 34 | **39 / 52** |

- 19 objects fixed, 1 broken. The break: rt_100000 t2, a toy car read as an action camera at margin 0.24.
- Cost: 6 s per frame median (max 24 s), 8 objects checked per frame, on compute3's GPU shared with the
  segmenter.
- **The thresholds were tuned on these frames.**
  - An exploratory pass at margin 0.03 / 600 px / no shift limit fixed 25 and broke 12.
  - 6 of the breaks were re-fits that jumped 50-90 mm onto a neighbour; the shift limit addresses those.
  - The other breaks were wrong reads at margins of 0.06-0.10, except the 0.24 one that remains (0.12 sits above
    the others).
  - So 19 / 1 is in-sample. The held-out check is the dev camera rows.
- Remaining misses: 5 of ours. Watch -> action_camera 2, toy_car -> action_camera 2, figure -> action_camera 1.

Next: dev camera rows (100000-04, `_dev_` family) on the bundle-B config plus `wrist_identity` "support_masked" (and
`imitate_repair`), against bundle B. The held-out measure is the frame-0 confusion of those rows and the copy score.

Tests: `tests/dojo/test_wrist_identity.py`.

**Held-out dev rows and a time budget (same day)**
- Runtime frame 0 at 693f32ec vs dev_home at ae4eaca4:
  - 100002: 2 misreads (watch -> action_camera, ours and demo) -> 0. wrist_identity changed exactly those two.
  - 100003: 0 category misreads; one change, garage -> watch at our watch.
  - No wrong changes.
- 100001 + wrist_identity froze 4 / 4 on compute4. The cause is the runner's watchdog: no env step for 300 s while frame 0's estimate had not returned.
  - Kit's zenity "not responding" dialogs also appear in the successful rows, during any long perception wait.
  - On compute4 the frame-0 estimate took 88-99 s with the step (100002 / 03) against 53 s without (dev_home 100001).
  - On compute3 the service's own estimate on 100001's runtime frame takes 74 s with the step and 56 s without, with no hang. So what makes 100001 exceed 300 s on compute4 is not known.
- `wrist_id_budget_s` (45 s): no further object is checked once the step has run this long.
- The step prints `[wrist_identity] checked n changed m in t s` to the service log, so a frozen row shows whether frame 0 got past it.
