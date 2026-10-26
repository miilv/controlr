# 2026-10-26 — perception-held: the held object's in-hand pose (held_pose) and where to look (show_pose)

Report with the numbers: [../experiments/2026-10-26-held-pose.md](../experiments/2026-10-26-held-pose.md).
Branch `feat/dojo-perception-held`. No LLM calls. GPU: Isaac 6.0 renders on two light lanes (the lead's rule), physics
captures on one light lane; every evaluation CPU-only (`DOJO_GL=0`, nice, in a scope).

## Why

Several skills reach 90-100 % in G3 (GT state) and about 0 in mode B. Their final insertion needs the held object's
pose in the gripper to sub-millimetre accuracy. G3 re-measures it from GT at the pre-insert hover; mode B only has
the transform predicted at grasp time, which is off by 1-4 mm and a few degrees (plug_in_charger: the blades need
about +-0.1 mm; play_stacking_toy: rings at a 1.5-2 mm offset bind on the peg).

## Done

- `dojo/perception/held.py` — `held_pose(frame, arm, asset_key, prior_T_tcp_obj, cfg)`: render-and-compare of the
  catalogue mesh in every camera that sees the object, constrained by the gripper. Contract in the module doc and in
  the perception service (op `held_pose`, `PerceptionClient.held_pose`).
- `dojo/perception/show_pose.py` — where to look: observer looks and show poses, ranked by the predicted accuracy of
  the in-pad DoF (Fisher information of the silhouettes, both arms as occluders), with IK hints and step estimates
  (op `show_pose`, `PerceptionClient.show_pose`). Generic: TCP frame + mesh + cameras, no task names.
- `dojo/perception/held_data.py`, `held_eval.py`, `scripts/perception/held_*.sh` — DEV data (G3 physics captures at
  the place hover + kinematic variants + Isaac 6.0 render) and the evaluation; Bridge-isr's genuine 5.1 held_look
  frames through the same scorer.
- Tests: `tests/dojo/test_held_pose.py` (geometry and data math in CI; the estimator on a synthetic scene where
  cv2 + scipy exist).

(TBD: findings, decisions, landmines, next — filled at the end of the stream.)
