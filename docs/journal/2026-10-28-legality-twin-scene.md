# 2026-10-28 — legality P1-4: the launcher's task no longer decides whether the verifier runs (Legality stream)

Source: robot-harness `docs/notes/ILIA-POLICY-LEGALITY-AUDIT-2026-10-28.md`, P1 finding 4 (their
`runtime/policy.py:333-347,380-391`, `twin/twin_service.py:100-106,124-126`, `runtime/scripts/integ_eval.sh` = here
`dojo/runtime/policy.py` `_setup` / `twin`, `dojo/simlab/twin_service.py`, `scripts/dojo/integ_eval.sh`).

**What was wrong.** integ_eval.sh / proxy_eval.sh `--bundle` start the twin service for the launcher's `--task`.
`RuntimePolicy._setup` read `tc.info()["task"]` and, when it differed from the scene it had derived from the
instruction, logged `verify_unavailable` ("twin runs X, the instruction names scene Y") and ran without a verifier.
So a non-contract input, the launcher's task, changed the inference procedure (verify on / off).

**Done** (branch `fix/legal-twin-scene`):
- `_setup` builds the `BundleVerifier` for the instruction-derived scene and never calls `info()` on a twin.
- `RuntimePolicy.twin(scene)`: with a host factory (`_twin_factory`), one twin per scene (started on first use, kept
  for later episodes of that scene; before, the first episode's twin was kept for every later scene). Without a
  factory, the given client / the bundle's running service, which its host must have started for the instruction's
  scene.
- Harness side, `dojo.bridge.twin.BundleTwins` (installed by `dojo.runtime.eval` with verify=bundle, and by simlab's
  `proxy_eval` for a RuntimePolicy in a `--bundle`): the bundle's pre-started twin when the policy's scene equals the
  launched task, else a `TwinProcess` for the policy's scene inside the job's lock and scope (as the bridge driver
  behind the official client always did); stopped at the end. Rows carry `twins_served` (dev side).

**Behaviour change.** Only episodes whose instruction-derived scene differs from the launcher's task change: they
used to run without verification (event `verify_unavailable` "twin runs ..."), now they verify against a twin for
their own scene (one extra twin start, ~1-2 min, inside the episode's wall time). Episodes whose scene equals the
task are unchanged. To find affected historical episodes: rows / reports with that event reason. The bridge path
(official client) already started the twin from the scene; unchanged.

**Tests:** `tests/dojo/test_legality_twin.py`:
- `test_the_launcher_task_never_decides_verification_or_the_decisions` (the audit's regression test): the same
  observations and instruction, the twin behind the policy launched for `stack_blocks` vs `press_by_number`: verifier
  active in both, `info()` never called, the same verify calls, decisions and executed actions. A mutant that restores
  the `tc.info()["task"] != scene` check fails it ("unavailable (see events)").
- a factory twin is per scene; `BundleTwins` serves the bundle's twin only for the launched scene and spawns /
  stops one for another; a static check that no runtime / agent / skills source calls `.info()` on a twin client.
- `test_runtime.py::test_twin_is_started_for_the_scene_the_policy_chose` now expects a second twin for a second scene.

**Not run:** a Isaac episode with a mismatching scene (no GPU in this stream). The spawn path is the bridge driver's
`TwinProcess`, which runs on every official-path episode; under integ_eval it runs inside the same scope (24 GB): a
second physics-only twin fits, but watch `scope_stats.json` on the first such job.
