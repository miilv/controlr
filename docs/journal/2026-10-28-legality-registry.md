# 2026-10-28: agent registry: candidates never reach select unless opted in (robot-harness a4f4d5e)

Legality work, registry stream. Source: robot-harness's routing-integrity finding 1 ("unpromoted skill versions must
not reach a chooser"), fixed there in `d918f85` (main: `status: candidate` in `skills/registry.yaml`;
`harness/registry.py`'s `filter_strict` / `filter_benchmark` / `--docs` leave candidates out unless the caller opts in)
and in `a4f4d5e` (branch night/S2, not on GitHub; described in their `docs/notes/S2-VERIFICATION-2026-10-27.md`), which
changed THEIR imported copy of our loader. A J4 re-import from controlr would have overwritten that, so the rule now
lives upstream here.

**Done** (`dojo/agent/registry.py`):
- `Registry.select(strict, benchmark, backends, candidates=False)` skips `status: candidate` entries unless
  `candidates=True`. This holds for every backend and for strict and non-strict selection alike. Absent and
  `promoted` are both selectable. `SkillEntry.is_candidate` and module `is_candidate(e)` mirror
  `harness/registry.is_candidate`.
- The validator mirrors `harness/registry.validate`. `status` must be absent, `candidate` or `promoted`; anything else
  is an error. A candidate with `default: true` is an error too ("promote it first").
- `catalog_mesh` is now a contract read, as in `harness/registry.py` since `f7f9017` (25 Oct) and in their imported
  copy. It means grasps / geometry from the public catalogue mesh of the perceived asset at the perceived pose
  (`mesh_grasps` stays non-contract). Without it, robot-harness's current file did not load here at all: the 21
  J4-imported ilia entries list `catalog_mesh` in `reads`, so validation gave 42 errors and `load()` raised.
  `DOJO_SKILL_REGISTRY=<their skills/registry.yaml>` therefore failed at load before this PR. Our own
  `skills/registry.yaml` does not use the token.

**Parity check** (local, not a test: CI has no robot-harness checkout). On their `skills/registry.yaml` at main
`9b42433` (144 entries, 25 candidates), `select(strict=True, benchmark=b, candidates=c)` equals `filter_strict(es, b, c)`
for b in (None, robodojo, libero_pro) and c in (False, True): 32 / 36 strict. `select(benchmark="robodojo",
candidates=c)` equals `filter_benchmark`: 62 / 87.

**Exposure before the fix: none found in controlr.** `loop.registry_skills` is the only caller of `select`, through the
prompt (`system_text`) and `Robo.skill`'s callable set in `AgentPolicy`. It passes `backends=("dojo_skills",)`. All 25
candidates of theirs are `robot_server`, our own file has no `status` field, and their file could not be loaded anyway.
Our own registry's manual is byte-identical before and after (md5 of `system_text`, strict and not).

**Decisions:**
- No `AgentConfig` opt-in. No dev path here needs candidates: theirs are `robot_server` skills, which the dojo runtime
  does not call in process. If one is ever needed, it should be a new `AgentConfig` field with default `False`, passed
  to `select(candidates=...)` and recorded in the episode report next to `registry` / `strict`.
- Fixture: an 8-entry verbatim excerpt of their file at `9b42433`
  (`tests/fixtures/robot_harness_registry_excerpt_9b424334ac66.yaml`), not the whole 268 KB file. It covers strict and
  non-strict candidates, a `default: true` entry and the imported ilia entry with `catalog_mesh`. The live file is
  checked only when `HARNESS_ROOT` points at a checkout (the test is skipped otherwise).

**Tests** (`tests/dojo/test_agent_registry_candidates.py`; all six fail on main's `registry.py`):
`test_the_runtime_select_leaves_candidates_out_for_every_backend`, `test_opt_in_lists_candidates_and_keeps_the_other_filters`
(the opt-in does not lift strict, the backend filter or `fallback`), `test_validator_status_rules`,
`test_robot_harness_excerpt_selects_like_filter_strict`, `test_registry_skills_prompts_no_candidate` (it also adds a
`dojo_skills` candidate, so the guard does not rest on the backend filter), `test_current_robot_harness_registry_if_present`
(`HARNESS_ROOT`).

**Not done / for the founders:** their imported copy (`runtime/agent/registry.py`) still differs from ours in the
docstring and the `HARNESS_ROOT` default (their repo root). The candidates rule and `catalog_mesh` are now the same on
both sides, so a re-import keeps `a4f4d5e`'s behaviour. Robot-harness's open question stays theirs: whether the
26 Oct night versions without a status are candidates.
