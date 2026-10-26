# 2026-10-26 — what IS there (held_look what, refine best), hover identity bar, catalogue probes

**Done:** branch `feat/dojo-perception-what`.
- **`dojo/perception/identity.py`:** rank interchangeable candidates by observed colour vs catalogue colour;
  `pick` (label = None when twins share the best key); `asked_verdict`; `refine_what` (refine + `best` /
  `asked_present` / `scored`); calibrated confidence tables (`CONF_FIT`, `CONF_PRIOR`, `CONF_HELD`).
- **held_look:** with `candidates` (the asked one first, at its expected in-jaw pose), held_look also returns `what`,
  `asked_held`, `scored` and `what_region`.
  - The B1 part is `j1_adapter.held_what`.
  - `agentic.j1.held_look_impl` forwards the candidates and keeps B1's `what` when the agent answers.
- **refine** returns `obs_fit` / `obs_prior` (the mean colour and pixel counts at the fit and at the prior).
- **`refine_identity`:**
  - a CHANGE of identity needs a fit with iou >= 0.9. Otherwise it returns the current identity's own fit, with
    `identity_ok=False` and `identity_rival`;
  - `look_alike` keeps the texture when the size changes;
  - the belief's identity survives candidate screening.
- **held_look** checks any arm commanded below 0.95. A board or pillar grasp commands 0.55-0.58; under the old 0.5
  rule, 27 of 58 5.1 tower holds got no B1 verdict.
- **refine:**
  - the table colour model excludes known non-support objects;
  - a support's pixels stay foreground only around the target's expected silhouette;
  - a fit that slides out of the view is rejected.
  - Results: 5.1 post-place well-placed ok, stack 5 → 20 of 23 and tower 10 → 19 of 67. D0: ok 284 → 306, ok > 10 mm
    off 17 → 5.
- **Catalogue / X5 dir probes:** any OSError counts as absent, and `DOJO_CATALOG` / `DOJO_X5_DIR` take an
  `os.pathsep` list. A colon-list `DOJO_CATALOG` on compute3 fell through to /root/... and crashed every perception
  episode of 7 scan A/B jobs.
- **Calibration:**
  - `dojo/perception/val_what.py` (VAL_SIZE=1 for the hover size check);
  - `heldsyn_synth.py` / `heldsyn_eval.py` (synthetic 6.0 held frames);
  - `val_replay51.py` (Bridge-isr's 5.1 set).
  - Report: docs/experiments/2026-10-26-what-is-there.md.

**Decisions:**
- held identity uses unmapped catalogue colours with lightness weight 1. The head-view colour map hurt in the jaws,
  and a held-view map fitted on 6.0 frames only gained 1 pp, at the risk of not carrying over to 5.1.
- The hover identity-change bar is iou 0.9. Below it, 24 % of size changes were wrong. 5.1 build_tower s0L9 was lost
  to a pillar → prism change at iou 0.67.

**Found along the way:**
- The runtime's `shape_classes` offers the whole task vocabulary to `refine_identity`, so in build_tower a pillar
  could become a prism. It should offer the label's own config group (BACKLOG; integration-2).
- 5.1 frame-0 identity on our 5 dev51 runs was 11/15. The misses are one size-group error (100006, all three blocks)
  and one dark-texture error (100005). Neither is a permutation.

**Not done / next step:**
- More 5.1 held frames (Bridge-isr regenerated checks) to confirm `CONF_HELD` on 5.1.
- identity for categories without texture differences (build_tower pillars are identical keys: what.label = None).
