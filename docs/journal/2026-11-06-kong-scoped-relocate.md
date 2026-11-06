# 2026-11-06 — KONG: scope actor relocation to make_kong

The [global relocation re-test](2026-11-06-ab-wave-verdicts.md) gained 30 points on
make_kong but lost 8.8 on imitate (n=8). It remains rejected. Added the independent,
default-off `RuntimeConfig.watch_relocate_task="make_kong"` lever, taking precedence
over the legacy bool and matching only the instruction-routed vocabulary. Empty
preserves the boolean, including legacy global-on runs. No presets/defaults changed.

## Consumption and shared paths

`RuntimePolicy._setup` resolves `perception_vocabulary(instruction)` (optionally
vocab_explain), passes its name as `scene` to RuntimeProvider and `_make_watcher`.
With watch on in perception mode, both make_kong and imitate get that watcher.
`RuntimeProvider.refresh` calls it only while no believed object is held.
Two repeated knocked-label reads call `_lay_face_up`; the sole consumer of the
old `cfg.watch_relocate` was the conditional immediately afterward. It starts
`_relocate_laid`, which calls `perception.locate_laid` and writes object position,
quaternion and source until two locations agree or the bounded tries run out.

Imitate and kong also share the actor-hold setup (no scan/look/search before the
actor-waiting skill), provider refresh, refine plumbing and skill construction.
Their task skills differ: copy_demonstration versus declare_kong. Imitation's
container `watch_read` is a separate service path, unaffected by watch_relocate.
The new gate resolves once inside `_make_watcher`; no skill parameter changes.

**Causality limit:** shared watcher installation is a possible route for global
side effects, not proof of the historical guard loss. The current
`perception_service.watch` calls `task_state.standing_slots/actor_change`; fewer
than two standing movable fixed slots returns unsupported and stops the watcher.
The [generic-watch audit](2026-10-28-legality-generic-watch.md) says only make_kong
has eligible slots. On that catalogue imitate cannot reach relocation at all.
Thus code inspection alone cannot explain the observed −8.8 as a relocation
write; deployment/catalogue/routing differences or run variability need existing
episode evidence. No episode data or remote box was inspected for this task.
The scoped gate excludes imitate even if a future/service read reports a knock.

## Verification and handoff

Tests compare baseline/scoped resolved JSON-valued Skills parameters byte for
byte for kong, imitate and stack; the fake launch metadata says stack even for
kong/imitate, verifying instruction routing. A forced-knock watcher regression
compares runtime function-call paths, every event byte, pose bytes, sources and
perception reads: kong matches legacy global-on; imitate/stack/unknown match
global-off. Default, exact-name, env loading, precedence and typo rejection are
covered. Existing relocation convergence/give-up tests remain intact.

[Camera preregistration](../experiments/2026-11-07-kong-scoped-relocate-prereg.md)
and JSON freeze two new salts/hashes, 10 gain + 5 imitate + 5 stack pairs under
the current score-based rule. No layouts generated, dispatch, GPU, remote writes,
model calls or system changes. The lead owns booking and review; do not merge.

Audit landmine: current relocation events contain only the resulting position,
not before7/after7; m95_score.audit does not count them. Full relocation-write
coverage is a pre-collection blocker, explicitly recorded in the preregistration
and backlog rather than claiming an incomplete audit passed.

Validation at implementation commit: targeted relocation/scoping tests **12 passed**;
production policy pyflakes clean; full CPU suite running. Test-file pyflakes reports
only pre-existing unused `env` and `T_from` at original lines 990/1220. Frozen design
validation verifies candidate and salt SHA256s, distinct non-reloc salts and 20 pairs /
40 episodes. Local environment is ~96 MB; no large dependencies downloaded.
