# 2026-10-28 — legality P1-1: the layout id no longer picks the perception calibration (Legality stream)

Source: robot-harness `docs/notes/ILIA-POLICY-LEGALITY-AUDIT-2026-10-28.md`, P1 finding 1 (their refs are to the
imported copy of controlr db48652; here: `dojo/runtime/eval.py`, `firewall.py`, `policy.py`, `perception_service.py`).

**What was wrong.** The dev drivers put `dev_calib_fold = crc32(layout id) % 2` into the reset info
(`dojo/runtime/eval.py`; `scripts/dojo/official_eval.sh` passed the same through `bridge.driver --calib-fold`). The
policy stored it and sent it with every perception call; the service picked the backgrounds (`DOJO_BG_FOLDS/fold<k>`)
and the colour map (`DOJO_COLOR_CALIB["fold<k>"]`) by it. A one-bit, layout-id-derived policy input: against the
benchmark rule even though it carries no pose. The firewall also passed `task` and `layout_id` through to `self.info`
("logging only").

**Done** (branch `fix/legal-calib-fold`):
- `firewall.INFO_KEYS` = `instruction`, `step_lim`. Task, layout id, the reset's `stable` / timings and the fold are
  dropped before `RuntimePolicy.reset` stores anything (`info_dropped` lists them). `log_task` is gone.
- `RuntimePolicy.fold` is the constant `policy.CALIB_FOLD`. The service maps every request to its ONE frozen set,
  `DOJO_CALIB_SET` (a dir under `DOJO_BG_FOLDS`, with its own `color_M.json`, else that key of `DOJO_COLOR_CALIB`),
  whatever the request's `fold` says. Without `DOJO_CALIB_SET` the legacy `fold<k>` of the request (offline dev tools).
  `identity.load_color_M()` (wrist check gallery, identity fallback) follows the set too.
- The drivers no longer compute a fold. `bridge.driver --calib-fold` is removed (an old caller fails loudly), the
  summary / official rows carry `calib_set`. `scripts/dojo/calib_set.sh` (sourced by integ_eval, official_eval,
  bridge_episode; isr_env sets the default) exports `DOJO_CALIB_SET=frozen_d0_100010_19` and refuses a perception run
  whose set is missing.
- Dev hygiene, scorer side: `dojo.perception.calib_set` builds a set from an explicit id list with a `manifest.json`,
  refusing ids in our measurement slices (`paths.OUR_MEASUREMENT_SLICES`: 100000-09, 100020-49) and reserved ids.
  `dojo.runtime.eval` and `official_eval.sh` refuse to evaluate a layout the set's manifest lists; every integ row has
  `calib_set` and `calib_disjoint` (null for a legacy dir without a manifest).

**The frozen set.** `frozen_d0_100010_19`: D0 ids 100010-19 only (6 tasks: build_tower, pack_objects_into_box_random,
push_T, put_bottles_into_dustbin, stack_blocks, stack_bowls; 58 episodes; colour map from 239 objects at GT poses,
segmented against the set's own backgrounds). D0 holds 100000-24; 100010-19 are the only D0 ids outside every slice our
policy is evaluated on (our rendered / proxy runs refuse them: they are Mikhail's development slice, D-C10a). So the
set must never calibrate an evaluation of 100010-19 (manifest + driver check). The official benchmark's Eval_Layout
ids are disjoint from all of D0 anyway. Built on compute3, CPU only (`DOJO_GL=0`, no GL context, no GPU job), ~30 s each,
from commit 8b9ef51:
- genuine 5.1 (`d0_51`): `compute3:~/dojo-data/calib_sets/bg_folds_51/frozen_d0_100010_19`, copied to
  `~/dojo-perception-data/bg_folds_51/` (tree md5 `ecc80808189c`);
- 6.0 proxy (`d0`): `compute3:~/dojo-data/calib_sets/bg_folds/frozen_d0_100010_19`, copied to
  `~/dojo-perception-data/bg_folds/` (tree md5 `17c141b04925`).
- `scene/` (conveyor, basket, cube_cushion backgrounds) is copied from the legacy `fold0/scene` and the manifest says
  so: those were built by their own streams from other layouts (journals 2026-10-27-dojo-perception3,
  2026-10-28-dojo-conveyor-belt-blind) and are NOT covered by `layout_ids` (BACKLOG).

**How different is it** (compute3, offline):
- backgrounds: mean abs difference to fold0 / fold1 0.2-0.3 grey levels (wrists, head), 0.8-1.1 (robot-free head),
  inside the fold0-vs-fold1 spread (0.3-0.4 / 1.2);
- colour map, on the 237 objects of the EVALUATED ids 100000-09 (held out from every map, frame 0): mean abs colour
  error frozen 0.1105 (median 0.092) vs the per-episode fold maps that ran 0.1147 (0.108) on 5.1; 0.1106 (0.090) vs
  0.1142 (0.111) on 6.0. Uncalibrated: 0.16.
- So the change is small and not for the worse, but every perception input changed: rendered numbers are not
  comparable across this change. Integration should re-measure the baseline.

**Tests:** `tests/dojo/test_legality_calib.py`:
- `test_the_layout_id_never_changes_the_calibration_or_the_decisions` (the audit's regression test: identical frames,
  proprio and instruction; reset ids 100000 vs 100004 (old folds 1 / 0) and different task names; a fake perception
  whose answer depends on the set). It fails on a mutant that restores `self.fold = info["dev_calib_fold"]`
  (folds `{1}` vs `{0}`), and passes on the fix (same folds, decisions, actions, initial record);
- the sanitized info, a static check that no runtime / agent / skills source reads `task` / `layout_id` /
  `dev_calib_fold` from info, the bridge driver's reset info, the service's one set, the colour map default,
  the builder's refusals and manifest, the drivers' overlap check.

**Not done / next:**
- Copy `frozen_d0_100010_19` into every box's `DOJO_BG_FOLDS` roots before deploying this (compute4's
  `~/dojo-data/bg_folds_51_ilia` / `bg_folds_ilia`, compute2, compute5-8, isr-lab's `$ILIA/bridge/pdata/bg_folds_51`);
  until then integ / official runs there refuse in mode perception (by design). `DOJO_CALIB_SET=fold0` is a labelled
  legacy choice (constant, but `calib_disjoint` null: fold0 is built from D0 ids whose crc32 is odd, incl. 100000-03,
  08-09, 20-23).
- Offline dev evaluations (`eval_g1`, `stereo_eval`, `look`, `scripts/perception/*`) still use the 2-fold D0
  cross-validation: that is perception research, not the policy, and is unchanged.
- The J1 adapter (Mikhail's path) reads one `DOJO_BG` dir: never per episode, unchanged.
