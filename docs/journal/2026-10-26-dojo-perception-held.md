# 2026-10-26 — perception-held: the held object's in-hand pose (held_pose) and where to look (show_pose)

Report with the numbers: [../experiments/2026-10-26-held-pose.md](../experiments/2026-10-26-held-pose.md).
- Branch `feat/dojo-perception-held` (PR #59).
- No LLM calls.
- GPU: physics captures on one light lane; Isaac 6.0 renders on two light lanes (the lead's rule: two slots, scope
  12G / 600 %, start at >= 24 GB and >= 10 GB VRAM).
- Every evaluation ran CPU-only (`DOJO_GL=0`, nice, in a scope).

## Why

Several skills reach 90-100 % in G3 (GT state) and about 0 in mode B. Their final insertion needs the held object's
pose in the gripper to sub-millimetre accuracy.
- G3 re-measures it from GT at the pre-insert hover.
- Mode B has only the transform predicted at grasp time, off by 1-4 mm and a few degrees.
- Examples: plug_in_charger (the blades need about +-0.1 mm) and play_stacking_toy (rings 1.5-2 mm off bind on the
  peg).

## Done

- `dojo/perception/held.py`: `held_pose(frame, arm, asset_key, prior_T_tcp_obj, cfg)`.
  - Render-and-compare of the catalogue mesh in every camera that sees the object, constrained by the gripper.
  - The contract is in the module doc and in the perception service (op `held_pose`, `PerceptionClient.held_pose`).
- `dojo/perception/show_pose.py`: where to look.
  - Observer looks and show poses, ranked by the predicted information about the in-pad DoF (silhouettes, both arms as
    occluders), with IK hints and step estimates (op `show_pose`, `PerceptionClient.show_pose`).
  - Generic: TCP frame + mesh + cameras; it never keys on a task.
- DEV data and eval: `dojo/perception/held_data.py`, `held_eval.py`, `scripts/perception/held_*.sh`.
  - G3 physics captures at the place hover, kinematic variants, Isaac 6.0 render.
  - Bridge-isr's genuine 5.1 held_look frames go through the same scorer.
- Tests: `tests/dojo/test_held_pose.py`.
  - In CI: geometry and data math.
  - Where cv2 + scipy exist: the estimator on a synthetic scene.

## Results at a glance (report for the tables)

Leading point across the insertion axis, median / p90, over all fits. Hit = lateral <= 0.5 mm AND axis <= 1 deg.

| object | hover (0 steps) | ranked observer look (13-17 steps) | ranked show pose + look (24-42 steps) |
|---|---|---|---|
| plug charger | 0.49 / 1.29 mm, hits 11/24 | 0.25 / 0.37 mm, 38/42 | 0.15 / 0.31 mm, 24/24 |
| stacking rings | 0.12 / 0.21 mm, 41/48 | 0.12 / 0.21 mm, 92/96 | 0.06 / 0.11 mm, 48/48 |
| tubes | 0.64 / 1.80 mm, 17/42 | 0.36 / 1.14 mm, 55/84 | 0.22 / 0.63 mm, 36/42 |
| pens | 0.72 / 2.57 mm, 14/38 | 0.27 / 0.81 mm, 50/72 | 0.25 / 0.58 mm, 30/38 |
| coin | 0.93 / 3.37 mm, 5/14 | 0.22 / 0.61 mm, 15/22 | 0.13 / 0.25 mm, 11/12 |
| genuine 5.1 stack blocks (held_look pose) | 0.32 / 0.60 mm, 26/56 | — | — |

- The sigma is calibrated per geometry class; held-out coverage at 2 sigma is 94-96 %.
- Per object it ranges from 0.77 (coin, own camera only) to 0.99.

## Findings

1. **Physical in-hand drift is tiny.** The G3 transform measured at the lift and the GT at the place hover agree
   within 0.1 mm / 0.05 deg (plug, rings). The mode-B problem is the PREDICTION at grasp time, not slip in the jaws.
   With a GT belief, the planned transform was already 0.1-1.4 mm off; the largest offset was along the closing axis,
   because the object is not exactly centred between the pads.
2. **The own wrist camera is free and sub-mm on position, but it cannot see the turn about the closing axis.**
   - It looks along the approach, so a turn about the closing axis and a slide along the approach trade off in a flat
     valley of the cost.
   - This holds for box-like objects (the charger) and for blocks seen from the back.
   - The sigma reports it, as a large along-approach / axis sigma, and such fits come back `ambiguous`.
3. **Any second view that sees the object fixes it.** A look from beside or across the pads, level or tilted toward
   the leading end, keeps the holding gripper out of the line of sight.
   - The naive look from the observer's own side was blocked by the holding gripper in about half the plug hovers.
   - Hence the ranked looks with both arms as occluders.
4. **Segmentation.**
   - A one-class "table colour" model missed a light-wood block on the wood table, so the model is now two-class (object
     vs background, learnt in the frame).
   - Scoring the IoU over the whole ROI let a similar-colour background (a light wall behind a white tube seen from the
     side) at P(object) ~ 0.5 inflate the union, and good observer views were dropped. The IoU is now scored near the
     expected outline.
5. **Box meshes have vertices only at their edges.** A vertex test of what the pads touch found nothing on the
   charger's flat sides and fell back to the wrong width; the finger mask then covered the object. It now uses surface
   samples (`held.surface_points`).
6. **Symmetric objects** (tubes, pens, coins): a turn about the symmetry axis changes nothing in the image. Left in,
   it inflated the lateral sigma 3x; it is now excluded from the uncertainty (`held.spin_axis`).
7. **Light lanes.** The g3 chains re-take a freed light lock within seconds, so two at once never came free by
   polling. The render now polls all four every 0.5 s on fds its shell holds (DEVELOPMENT pitfall).

## Tried, not kept

- A flat-shading NCC term (lit top vs darker sides) for the closing-axis turn: on 16 genuine 5.1 block fits it helped
  2 and hurt 6.
- `show_pose`'s predicted sigma (idealised silhouette information) is far below the measured error. It is used for
  ranking only (BACKLOG).

## Not done / next

- insert_key: the skill is parked on main (`place_oriented` ik_fail at every layout), so no held hover was reached and
  nothing was measured.
- The skills-side protocol (report, "Protocol") goes into the shared insertion tail. Integration-2 wires
  `provider.held_pose(arm)`.
- Mode-B end-to-end: G3 `--mode-b` with the measured transform (needs the provider) on plug / rings / tubes.
