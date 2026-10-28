# 2026-10-28 — D-C10c: 100030-49 is a training / priors / calibration block, never evaluated (Legality stream)

Ruling D-C10c (robot-harness 1c70562) widens D-C10b's calibration-only 100040-49 to 100030-49: training data, priors
and calibration come from it, so no evaluation run uses it.

**Done** (branch `fix/legal-dc10c-training-block`):
- `paths.TRAINING_BLOCK = (100030, 100049)`, `is_training_id`, `training_refusal(id, record, allow_variants)`;
  `OUR_MEASUREMENT_SLICES` = 100000-09 + 100020-29. `CALIBRATION_SLICE` stays 100040-49 (the calibration builder).
- Refused for evaluation: `integ_eval.sh` and `official_eval.sh` (by id, before any lock), `dojo.runtime.eval`,
  simlab `proxy_eval` and `dev_eval` (the official client on our layouts), and G3 runs (`g3.py run`) unless the record
  is a variant draw (`_simlab.variant` >= 1: a new layout from the config, not the id's own). seg_render, the
  calibration build and variant rollouts keep working.
- Not changed: offline perception tools whose docstrings name "100020-49" (`m95_*`, `sam/bake.py`): they score
  perception on recorded frames, not a policy evaluation; their next use should drop 100030-49 by hand.

**Test:** `tests/dojo/test_legality_calib.py::test_the_drivers_refuse_to_evaluate_the_training_block` (integ_eval.sh
refuses 100030 / 35 / 43 / 49 before any lock; the refusal helper, incl. variants allowed only for G3).
