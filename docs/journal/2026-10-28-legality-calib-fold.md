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
  bridge_episode; isr_env sets the default) exports `DOJO_CALIB_SET=frozen_d0_100040_49` and refuses a perception run
  whose set is missing.
- Dev hygiene, scorer side: `dojo.perception.calib_set` builds a set from an explicit id list with a `manifest.json`,
  only from the calibration-only slice (below). `dojo.runtime.eval` and `official_eval.sh` also refuse to evaluate a
  layout the set's manifest lists; every integ row has
  `calib_set` and `calib_disjoint` (null for a legacy dir without a manifest).

**The frozen set** `frozen_d0_100040_49` (the lead's ruling, D-C10b in robot-harness `coord/DECISIONS.md`): built
only from the calibration-only ids 100040-49 of our slice, which are never evaluated. `paths.CALIBRATION_SLICE`; the
builder refuses every other id; `integ_eval.sh`, `official_eval.sh`, `dojo.runtime.eval` and simlab's `proxy_eval`
refuse to evaluate them. Our rendered measurement slices are now 100000-09 and 100020-39. (A first build from the D0
ids 100010-19, the only D0 ids outside our measurement slices, was withdrawn: those are Mikhail's development slice,
D-C10a; its dirs on compute3 are deleted.)
- Frames: D0 format (`make_d0`: home + 3 scripted arm moves, frames 0-3; the robot-free head background needs the
  moves to see under the home grippers), the 6 D0 tasks (build_tower, pack_objects_into_box_random, push_T,
  put_bottles_into_dustbin, stack_blocks, stack_bowls) x 100040-49, records generated from the seed (as D0 was), in
  scratch `DOJO_DATA` roots so the shared `simlab/layouts` dirs are untouched. `make_d0` now stops at the first id that
  is not ours to render, so a failed layout can never push it into 100050+.
  - 6.0 proxy (`bg_folds`): rendered on compute3 under `~/dojo-gpu.lock` (physics-only 5.1 states + Isaac 6.0 RGB),
    `~/dojo-data/d0_calib60`.
  - genuine 5.1 (`bg_folds_51`): rendered by stream S on compute2 (5.1 RTX), `~/dojo-data/segtrain/d0cal_51`.
- Build: `python -m dojo.perception.calib_set build --d0 <root> --out <bg root> --scene-from <legacy fold0>`, CPU only
  (`DOJO_GL=0`). `scene/` (conveyor, basket, cube_cushion backgrounds) is copied from the legacy `fold0/scene` and the
  manifest says so: those were built by their own streams from other layouts and are NOT covered by `layout_ids`
  (BACKLOG).
- Every box gets it the same way: `scripts/infra/box_steps.sh` step `calib_sets` (after `catalog_post`; the catalog
  relay brings `bg_folds{,_51}` from compute3) checks each set against its md5 in `scripts/infra/pins/calib_sets.txt`
  and installs it into the integration roots (`~/dojo-data/<dir>_ilia`, `CALIB_EXTRA_ROOTS`), verified; a missing or
  different set fails the step (compute4's silently missing scene background was this failure mode). On a box:
  `bash ~/dojo-bootstrap/lib/box_steps.sh run calib_sets`, or `scripts/bootstrap_gpu_box.sh HOST --only catalog,calib_sets`.

**How different is it.** The first (withdrawn) 100010-19 build gave the yardstick: backgrounds within the fold0 /
fold1 spread (mean abs difference 0.2-0.3 grey levels, robot-free head 0.8-1.1 vs fold0-fold1 1.2), colour map on the
237 held-out objects of 100000-09 slightly better than the per-episode fold maps that ran (abs error 0.110 vs 0.115 on
both renderers). The 100040-49 set:
- 6.0 proxy (compute3, frames rendered 10:00-10:30Z under `~/dojo-gpu.lock`: 58 episodes, put_bottles_into_dustbin 8
  of 10 ids fit, the guard stopped it at 100050; 231 MB; set 2.4 MB, tree md5 `661f0fcf9970228d55c5e0dd1ab6cded`):
  backgrounds vs fold0 / fold1 0.30 / 0.31 (head), 0.24-0.28 (wrists), 1.17 / 1.71 (robot-free head; fold0-vs-fold1
  1.21), unseen head pixels 13,906 vs fold0's 13,721; colour map on the 237 held-out objects of 100000-09 abs error
  0.1128 (median 0.093) vs 0.1142 (0.111) for the per-episode fold maps that ran.
- genuine 5.1 (compute2, stream S rendered the frames 10:41-10:59Z, 5.1 RTX, make_d0 @ 1aa7b7a; 58 episodes,
  put_bottles_into_dustbin 100040 / 100049 unstable at reset; 209 MB; set built on compute2 CPU-only, copied to
  compute3's `~/dojo-perception-data/bg_folds_51/`, tree md5 `dbb2a685ad265e2b05b238b217142630`, its `scene/` identical to
  compute3's `fold0/scene`): backgrounds vs fold0 / fold1 0.35 / 0.37 (head; fold0-vs-fold1 0.39), 0.29-0.33 (wrists),
  1.18 / 1.61 (robot-free head; 1.24); colour map on the 237 held-out objects of 100000-09 abs error 0.1130 (median
  0.094) vs 0.1147 (0.108) for the per-episode fold maps that ran.
- `box_steps.sh run calib_sets` on compute3 (from this branch): both sets verified against their pins.

Every perception input changes, so rendered numbers are not comparable across this change: Integration re-measures
the baseline.

**Tests:** `tests/dojo/test_legality_calib.py`:
- `test_the_layout_id_never_changes_the_calibration_or_the_decisions` (the audit's regression test: identical frames,
  proprio and instruction; reset ids 100000 vs 100004 (old folds 1 / 0) and different task names; a fake perception
  whose answer depends on the set). It fails on a mutant that restores `self.fold = info["dev_calib_fold"]`
  (folds `{1}` vs `{0}`), and passes on the fix (same folds, decisions, actions, initial record);
- the calibration slice: the builder refuses every id outside 100040-49, integ_eval.sh refuses 100043 before any
  lock; `box_steps.sh calib_sets` installs a pinned set into an `_ilia` root, repairs a drifted copy, fails on a
  different set;
- the sanitized info, a static check that no runtime / agent / skills source reads `task` / `layout_id` /
  `dev_calib_fold` from info, the bridge driver's reset info, the service's one set, the colour map default,
  the builder's refusals and manifest, the drivers' overlap check.

**Not done / next:**
- Run `calib_sets` on every box before deploying this (compute2/3/4, compute5-8, isr-lab's
  `$ILIA/bridge/pdata/bg_folds_51` via `CALIB_EXTRA_ROOTS`); until then integ / official runs there refuse in mode
  perception (by design). `DOJO_CALIB_SET=fold0` is a labelled legacy choice (constant, but `calib_disjoint` null: fold0
  is built from D0 ids whose crc32 is odd, incl. 100000-03, 08-09, 20-23).
- Offline dev evaluations (`eval_g1`, `stereo_eval`, `look`, `scripts/perception/*`) still use the 2-fold D0
  cross-validation: that is perception research, not the policy, and is unchanged.
- The J1 adapter (Mikhail's path) reads one `DOJO_BG` dir: never per episode, unchanged.
