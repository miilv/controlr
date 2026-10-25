# 2026-10-25: dojo bridge: our full system behind robot-harness's strict server (the official-client path)

Stream: bridge (branch `feat/dojo-bridge`, code `dojo/bridge/`, tests `tests/dojo/test_bridge.py`, runbook
[../BRIDGE-RUNBOOK.md](../BRIDGE-RUNBOOK.md)). Goal: run RuntimePolicy / AgentPolicy (perception + skills + twin
+ agent) behind robot-harness's strict robot server, i.e. behind the OFFICIAL RoboDojo eval client, using only the
official observation contract. Before this, our system ran only in our own proxy eval. CPU only today; 1 live LLM
call.

**Done:**
- `dojo/bridge/client.py`: the robot_client_v2 protocol (JSON lines, one request per connection, Unix socket),
  re-implemented stdlib-only so no robot-harness checkout is needed on `sys.path`. Verbs are methods, so the same
  object can serve as `robot` for robot-harness skills (issue #5, adapter (b)).
- `contract.py`: our side of the strict rule, fail-closed.
  - `check_tier` refuses a non-strict server.
  - `check_reply` raises on any banned key at any depth: the strict server's BANNED list plus dev / grader keys.
  - `contract_obs` builds the runtime's observation from exactly: frames, measured q, measured flange, COMMANDED
    gripper, instruction, `steps_used`, `step_lim`.
  - Calibration never comes from a reply: our perception computes it from the public config + measured joints.
- `backend.py` `RobotServerBackend`:
  - `observe()`;
  - the skills `Env` protocol: buffered `step()`, flushed through `joint_steps` in chunks of ≤ 500;
  - `execute()`, `n_executed` / `actions_since` (the verifier's bookkeeping);
  - wrappers for `hold`, `gripper`, `return_home`, `perceive`, `held_look`;
  - a flush before any foreign verb, so a robot-harness skill can run after our buffered steps.
- `perception.py`: two routes behind the runtime's perception-client interface.
  - `local` (default): our service on contract frames, the same as the proxy runs.
  - `server`: the J1 server's `perceive` verb → `record_from_perceived` → layout record; `look_plan` comes from
    `needs_look`; the local-only ops are delegated or turned off.
- `twin.py`: the twin service is spawned for the scene the INSTRUCTION names (`perception_vocabulary`), never the
  launcher's task. `BundleVerifier` loads it from our estimate + measured proprio, unchanged.
- `driver.py` (`python -m dojo.bridge.driver`):
  - reset → `policy.reset({instruction, step_lim})` → `next` / `execute` / `observe` until the policy is done or
    `ended`;
  - outputs `bridge.json`, `report.json.gz`, `actions.json.gz`, `calls.jsonl`; no verdict.
- `fake_server.py`: a CPU fake of the strict J1 server on the kinematic env.
  - Same reply shapes, plus the proposed `joint_steps`.
  - Validation before any step; refusals cost 0 steps.
  - A 0.2 rad/step joint-jump cap, and a success latch (stacked + home).
  - Leak / tier switches for the firewall tests.
- `scripts/dojo/bridge_episode.sh`: one episode end to end.
  - Steps: session lock → `robot-start-j2` (official client) → perception service → driver (spawns the twin) →
    close → wait for `launch_done.json`.
  - `--fake` runs the same path on the fake server.
- 14 tests (18 s): protocol, strict observation = contract, leaking servers refused (objects / finger_q / calib),
  non-strict and verbless servers refused, `joint_steps` bit-exact and refusals, flush before foreign verbs, and
  three episode tests:
  - RuntimePolicy through the server, local route, GT patched to raise;
  - the full system: the server `perceive` route → twin loaded from those poses + proprio → AgentPolicy (FakeLLM)
    → library skill with twin verify on every segment → the latch ends the episode;
  - a static check that the bridge sources read no GT / eval files / grader verbs.
- `tests/dojo/test_bridge_rh.py` (opt-in, `CONTROLR_ROBOT_HARNESS=<checkout>`): our backend against robot-harness's
  REAL strict J1 server code.
  - His server files and frozen offline mock shim are copied into a temp `$BENCH`; his repo is untouched. The J1
    mock perceiver is used, and only `strict_calib.py` is stubbed (it reads RoboDojo's config at import).
  - Passed first try on robot-harness `d279be1`: tier check, reset, obs → contract observation (and `check_obs`),
    proprio, perceive → record, B1 held_look, hold, gripper, close.
  - `joint_steps` is refused as expected. Once #8 lands (with a mock-shim op), the same test executes it.
- Issue robot-harness #8 (posted by the lead): the `joint_steps` verb, plus a `robot-start-j2` start script.

**Numbers (mock, kinematic env; not a benchmark):**
- `scripts/dojo/bridge_episode.sh --fake`, RuntimePolicy, server perceive route: latch at 172/550 steps, 3.5 s
  wall, 15 server calls.
- The first in-process run (local route): 203 steps; 0 of 203 actions hit the fake's 0.2 rad/step cap.
- Live smoke: AgentPolicy on the fake server, `cx/gpt-6.1-sol`, through the CLI driver.
  - **1 call**: 4478 prompt / 57 completion tokens, 0 cached, 3.7 s.
  - The model's first cell called `stack_objects`; the latch ended the episode at 145/550 steps, inside the skill's
    `finish()`.
  - Live calls today: 1, in this smoke.

**Decisions:**
- **The driver plays the harness role**: proxy_eval's `reset` / `next(obs)` loop. RuntimePolicy and AgentPolicy
  run unchanged, so their own firewall (`check_obs`, mode perception) checks every observation a second time.
  No change to `dojo/runtime/` or `dojo/agent/`.
- **One new verb, not a translation.** Our skills emit per-step joint actions (0.18 rad/step free, 0.05 near
  contact, gripper as a step command). The twin verifies those exact lists.
  - `move_to` re-plans with cuRobo at ≤ 0.08 rad/step: different paths, ~2x the steps.
  - The CLI has no joint-target verb: the shim's `op_move` accepts `q`, but `cmd_move_to` sends only `flange`.
  - So we asked for `joint_steps` instead of forking his server (contract in the runbook §9).
- **`reserve=0` per call.** The runtime keeps its own fixed reserve (30, the same every episode). Actions past the
  step limit are never sent, as in the proxy.
- **Perception route default `local`**, because it has every op (refine / check / search / look poses), as in the
  proxy numbers. The `server` route exists for the audit trail (`perceive.jsonl` on Mikhail's side). It loses the
  stripped `_perception` flags, and its task routing covers 6 tasks.
- **Backgrounds / colour map: the 6.0-built sets by default.** The perception journal measured the 5.1-built ones
  within noise on genuine 5.1 frames; both stay selectable in `bridge_episode.sh`.
- **Frames**: every `obs` writes 3 PNGs. They go into our own `frames/` dir, and every 20th is kept (about 1 MB per
  observation otherwise).

**Found along the way (→ BACKLOG):**
- robot-harness's launchers hard-code `/home/anywherevla/data2/bench/env.sh` (BENCH, RD_HOME). Running his stack on
  our GPUs needs that path or an override.
- Two instruction routers for perception: `j1_adapter.infer_task` and `supervisor.perception_vocabulary`.
- The proxy's per-task twin launch is resolved for this path (the twin follows the instruction). BACKLOG item
  annotated.
- The strict server's `info.strict` holds a `calib` key (a string), so the banned-key scan must not run on `info`.
  The bridge checks only the tier there.
- With a success latch that needs home, the episode can end inside `finish()` before the agent's cell returns, so
  `agent.outcome` stays None. That is expected (the RoboDojo latch); read the `agent_skill` events instead.

**Not done / next step:**
1. Mikhail's side: `joint_steps` + `robot-start-j2` (#8), and the env path override.
2. On our GPU box (26 Oct): the §2 setup, then one practice episode on seed 0 layout 9 (stack_blocks):
   - RuntimePolicy first (0 calls), then AgentPolicy;
   - measure VRAM with the eval client + twin + perception on one card, plus observation latency and twin boot.
3. Unify the two instruction routers.
4. Scored blocks only on the lead's go: md5 freeze of our checkout + data, blind, ledger lines.
