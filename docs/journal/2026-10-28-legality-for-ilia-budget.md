# 2026-10-28: legality, "For Ilia" items: agent budget, episode state, data identity, fake J2 diagnostics

Source: robot-harness `docs/notes/P1S-VERIFICATION-2026-10-27.md` (For Ilia 1) and
`docs/notes/G1F-VERIFICATION-2026-10-28.md` (For Ilia 2, 12, 13), read-only; refs re-located from the imported copy
(controlr db48652) onto main 1e0f7b0. Sibling PR: held / pose unknowns (journal 2026-10-28-legality-for-ilia-unknowns,
which also lists every deferred item).

Tests: `tests/dojo/test_legal_for_ilia_budget.py` (real functions, fakes only: FakeChat, a branching fake IK, the
fake strict server, the vendored `j1_schema`; no Isaac, GPU, LLM or catalogue). One test needs cv2 + scipy (skipped in
CI; passes locally with `uv run --with opencv-python-headless --with scipy`).

## Fixed

| item | what was wrong | fix | test |
|---|---|---|---|
| P1S-1 (I1-1) | `gate_decision` said `episode_budget_spent`, but with B1 failed (`b1 is None`) `perceive_impl` fell through to an agent call; the count reset whenever `prior` was None and lived only in the restartable worker | `perceive(..., episode=None)` names the parameter (robot-harness's host passes `episode` only then) on both adapters. `episode_begin`: a new `episode["id"]` starts the count from the server's `agent_calls`; the same id keeps the larger count; no id = the old `prior is None` rule. A spent budget is checked first; with no B1 output the reply fails closed (no objects, `error`, `result=budget_spent_no_b1`), never an agent call. Every reply has `agent_call` true / false | `test_adapters_name_the_episode_parameter`, `test_spent_budget_without_b1_fails_closed_and_never_calls_the_agent[off/fails]`, `test_budget_follows_the_episode_id_not_the_prior`, `test_without_an_episode_the_old_gate_is_unchanged` |
| G1F-12 (G2-11) | the IK seed cache `_Q_LAST` crossed episodes in a persistent worker (missing joints: the last episode's branch seeded the self-mask) | `j1_adapter.episode_begin`: a new episode id clears `_Q_LAST`, first thing in `_perceive_impl`; the agentic worker passes its `episode` to B1 | `test_ik_seed_cache_never_crosses_an_episode`, `test_perceive_impl_starts_the_episode_before_anything` (cv2) |
| G1F-2 (G2-1, worker half) | the worker honoured DOJO_* overrides and fallback roots silently | `data_identity()` (once per worker): per input (catalog, x5, bg, colour calib) the configured value, the path the loaders resolved, `fallback`, sha256 (index.json, X5A.urdf, every bg file, the colour file) and an `id`; in every B1 / agent perceive output as `data` (the strict schema strips it; the server's raw log keeps it). `J1_REQUIRE_DATA=1` refuses a fallback / missing root, before B1 in the agentic worker too (a B1 failure there would let the agent run without it) | `test_data_identity_reports_effective_paths_and_refuses_fallbacks` |
| G1F-13 (G1-8 consumer) | `bridge/fake_server.py` reported whole-list `grip_change_while_moving` / `max_step_dq` | executed-prefix values, whole-list ones as `requested_*`, as robot-harness's shim now does | `test_fake_server_reports_executed_prefix_diagnostics` |

Also (G1F-10 / G2-9, deferred): a B1-gated reply of an episode that had an agent scene this worker never saw (a
restart) says `tracker=lost` in its notes, so the server's log shows the incidence. The tracker itself is not restored.

## Deferred / elsewhere

- G1F-3, G1F-5: the card-digit stream. Env / credential handling in `install_env` / `agentic/llm.py`: the SEC stream
  (`data_identity` reads the worker's environment only; `install_env` is untouched).
- G1F-1, G1F-4, G1F-11, P1S-2, P1S-3: the sibling PR. G1F-6..10: deferred there (rendered validation).
- P1S item 4 (pins): robot-harness's process.
- G2-11's alternative "reject missing joints": not done; the id reset covers it and the J1 server sends joints.

## Behaviour changes, and can rendered numbers change?

- The budget gate with the server's `episode`: a call without `prior` in the middle of an episode no longer resets
  the count (the point of I1-1). Rendered numbers: only where that happened (a worker restart, or a driver that
  perceives without `prior_n`): fewer agent calls, B1-gated replies instead. Without `episode` (in-process callers,
  older servers): unchanged, except `max_per_episode` 0 now means no agent at all (it still called one on the
  episode's first call).
- Budget spent and B1 failed: a fail-closed empty reply instead of an agent call. Rendered: only in that case.
- `_Q_LAST` reset per episode id: changes the self-mask only for a worker reused across episodes AND called without
  measured joints (robot-harness's J1 servers start one worker per episode and send joints): no change on their
  published path.
- `data` / `agent_call` in the outputs: additive (stripped by the schema). `J1_REQUIRE_DATA` defaults off.
- fake_server: tests / dry runs only.
