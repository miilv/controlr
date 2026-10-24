# Teardown: coding agents with persistent sessions, skills and RSI on RoboDojo

This teardown covers EmbodiedEvo, RAX, OpenETA (RoboDojo branch), RACaP and Noematrix RoboRSI. It was written on 2026-10-24 by a read-only research agent. No API calls were made and nothing was downloaded beyond shallow source clones, which were deleted afterwards.

Every claim is labelled with its source:

- **[code]**: a file in a shallow clone, cited as `repo/path:line`.
- **[doc]**: a markdown document in the repository.
- **[paper]**: the arXiv HTML. These are the authors' numbers; we did not reproduce them.

| Repo | Branch / commit | arXiv |
|---|---|---|
| Einsia/EmbodiedEvo | main `470278c` (2026-10-14) | 2610.18867 |
| robot-agent-exploration/rax | `robodojo-behavior/native` `16e75db` (2026-10-24) | 2610.22496 |
| OpenMOSS/OpenETA | `robodojo-v112` `880522b` (2026-10-16) | — |
| Robo-Harness/racap | main `f93c37a` | 2609.29394 |
| nssmd/RoboRSI | main `90c8569` | 2610.12424 |

The context for these notes is in four places:

- [docs/ROBODOJO.md](../../docs/ROBODOJO.md): our XPolicyLab shim.
- [the brainstorm](../../docs/journal/2026-10-24-robodojo-brainstorm.md): the scoring math.
- [deep research](../sources/deep-research-robodojo-2026-10-24.md): rows 4, 5, 7 and the EmbodiedEvo rows.
- [robodojo.md](../sources/robodojo.md).

---

## 0. Summary

| | EmbodiedEvo | RAX | OpenETA v112 |
|---|---|---|---|
| Agent CLI | Codex CLI 0.159.2, `codex exec` (a Claude preset exists) | Claude Code or Codex, native on the host | Codex CLI, `codex exec`, **shell tool off** |
| Model | GPT-6 Astra, xhigh | Claude Opus 5.5 (3 tasks), GPT-6 Codex (4 tasks), xhigh | GPT-6 Astra, high |
| Robot interface | Python code cells sent over a Unix socket to a persistent host namespace; 4 primitives (raw native `step(action)`) | MCP tools (joint/EEF chunks of ≤50 rows, cuRobo planning, pixel→3D, FK preview) plus Python scripts that use the same MCP | MCP tools only (`observe`, `mark_point`, `move_to` with contact modes, `gripper_control`, `check_task_completion`) |
| Observation | RGB only, 3×640×480 plus state; no depth or calibration | **RGB-D plus intrinsics and extrinsics by default**; `rgb-only` is a profile | RGB-D (`mark_point` returns the measured depth point), calibrated TCP |
| Learning | Playground on 1 scene/task, then `skills/` and `lessons/` frozen; a fresh agent per Test scene | N exploration episodes on collection 0, compiled into `controller.py`, register → rehearse → submit; the frozen bundle runs alone | Human-assisted, cross-version harness repair (v108→v112); private frozen "experience" bundles |
| Held-out | Test seeds sampled after the freeze, Playground seed excluded | Formal layouts from another collection, scheduled up front | Seed 0 only; not held-out |
| RoboDojo number | **official board 38.95% SR / 45.08**; paper full set 37.82% / 44.48 | 7 tasks, score avg **78.3** (paper), not on the board | **23/42 tasks first-pass** (54.8% "coverage", seed 0) |
| Cost/episode | **$5.73** Test, about 4M tokens (mostly cache), 29 code execs (paper) | Session 4.9–14 h for one task (paper); a compiled program needs 0 LLM | not reported |

**Shared conclusion:**

- All three top "coding agent" entries put the agent in a sandbox that has **no simulator, no GPU and no ground truth**.
- They talk to a **host-owned session server** over a Unix socket.
- They persist knowledge as **files in a workspace**, frozen by hash before evaluation.
- The difference between them is what crosses the boundary:
  - raw actions (EmbodiedEvo);
  - rich planning tools plus depth (RAX, OpenETA);
  - a compiled program, with no LLM at test time (RAX code mode).

---

## 1. EmbodiedEvo (Einsia): official #4, 38.95% SR

### 1.1 How Codex is attached to a persistent RoboDojo session

**Agent side.** The agent is a thin Docker container that runs `codex exec`.

- The CLI argv is built in `orchestrator/launch_agent.py:26-46`. It is `codex exec --sandbox workspace-write --skip-git-repo-check -c model=... -c model_reasoning_effort=...`, or `claude --print --permission-mode acceptEdits`.
- Inside Docker, Codex's own sandbox is swapped for `--dangerously-bypass-approvals-and-sandbox`, because the container is the boundary (`:131-152`).
- The container flags are `--read-only`, `--cap-drop ALL`, `no-new-privileges`, `--pids-limit 512`, plus memory and CPU caps (`:78-128`).
- The network is a bridge named "filtered", which actually does **not** restrict destinations (`:7-8`).
- The Dockerfile is 26 lines: node22 plus `@openai/codex@0.159.2`, python3.11-slim, ripgrep, ffmpeg, numpy, pillow and imageio (`docker/agent/Dockerfile:1-26`).

**Socket.** The host binds a Unix socket with mode 0600 (`orchestrator/session.py:346-348`). It is mounted read-only into the container at `/run/embodiedevo/session.sock`.

- Also mounted read-only: `instruction.md`, `primitives/`, `scripts/` and `observations/`. In Test, `skills/` and `lessons/` are added (`orchestrator/session_runner.py:174-195`).
- The kickoff prompt is one line. Notably it says "inspect no more than two current frames per iteration to keep model requests focused" (`session_runner.py:152-164`).

**Client** (`tasks/RoboDojo/Workspace/scripts/env.py:1-84`, stdlib only):

- Commands: `exec`, `status`, `observe`, `instruction`, `reset`, `finish`.
- `exec submission/solution.py --include skills/x.py` *concatenates* source files and sends them as one JSON line.
- Only `.py` files under `submission/` and `skills/` are accepted (`:46-55`).
- "Never retry a request: a broken connection does not imply rollback" (`:38`).

**Host executor.** Code runs in **one persistent Python namespace** with restricted builtins.

- The AST validator forbids `_`-names, file access and imports other than `numpy` and `math` (`simulators/interactive.py:126-205`).
- The physics advances only inside `step(action)` (`orchestrator/session_client.md:65-67`), so agent thinking is free.

**Primitives** (`tasks/RoboDojo/Workspace/primitives/primitives.md:8-96`):

- `get_instruction()`.
- `get_observation()`: 3 RGB frames at 480×640, joint/EE state, `frequency` 25 Hz. "Depth and camera calibration are not enabled" (`:22`).
- `step(action)`: one complete native action dict, either absolute joints or absolute EE poses plus gripper 0..1. Native IK "may leave an arm unchanged".
- `reset()`: Playground only.

**Native-evaluator bridge** (`integrations/robodojo_policy/`). This is an XPolicyLab policy (`policy/EmbodiedEvo/model.py`).

- On the first `get_action` it spawns a worker process holding the agent's code session and talks to it over a `multiprocessing.Pipe` (`model.py:45-60`).
- **Each agent `step(action)` blocks on the pipe.** While the agent is thinking, `get_action` returns an *empty chunk* every 0.5 s, and "the deploy loop polls without advancing physics, so long coding turns do not hit RPC timeouts" (`model.py:79-84`).
- After the agent finishes, the bridge holds the measured poses until the official success check or timeout (`model.py:64-72`).
- `bridge.py:11-16` whitelists the observation keys passed to the agent: `instruction`, `vision`, `state`, `action`, `additional_info`.
- `bridge.py:19-31` validates action keys, shapes and finiteness.

> Pattern: **the empty-chunk poll** lets an arbitrarily slow agent run under XPolicyLab's synchronous loop. Our `robodojo-serve` shim already blocks on our side, but the empty-chunk trick is what keeps XPolicyLab's websocket alive if we move to the native policy-server path.

### 1.2 The Playground → Test protocol

The protocol is specified in `docs/PROTOCOL.md:26-60` and `AGENTS.md:134-150`, and enforced in `session_runner.py`.

**Playground**

- It runs on one fixed scene, `playground.seed = 0` (`tasks/RoboDojo/task.toml:183-188`).
- It has 100 code-execution requests in total. `reset()` renews only the native `step_lim`, never the exec budget or wall time.
- Variables survive resets. `skills/` and `lessons/` are writable.
- The instructions demand that knowledge be written *as soon as it appears*, not at the end (`Workspace/playground.md:10`, `:126-146`).

**Freeze**

- This happens only after a valid Playground. An infrastructure failure blocks the freeze "even if it wrote partial lessons" (`session_runner.py:656-669`).
- `skills/` and `lessons/` are copied to `frozen_harness/` and hashed into `protocol.json` (`:672-684`).
- `frozen_source.py:74-136` re-verifies the hashes before every Test. Test refuses to run without a freeze (`:756`).

**Test**

- Test seeds are drawn on the host *after* the freeze. The Playground seed is excluded (`orchestrator/files.py:221-233`).
- Paired conditions share one locked plan (`files.py:199-211`).
- Each scene gets a fresh container, workspace and simulator. Knowledge is mounted read-only, and `reset` is forbidden: the client rejects it and a program call raises (`session_client.md:41-43`).
- Each scene has 100 execs plus the native `step_lim`. There are 25 scenes by default (`task.toml:190-194`).
- The official `success` signal is returned in every exec response, both in Playground and in Test. The run fails to start if `expose_success_feedback` is false (`session_runner.py:45-50`). This is the native episode-end latch, not a shaped reward.

**Baseline.** A separate template (`tasks_baseline/`) that must not contain skills or lessons (`session_runner.py:70-73`).

**Recipes.** All 42 RoboProbe "recipes" are copied verbatim, including the **official score ladders**: see `tasks/RoboDojo/recipe/stack_bowls.md:17-23`. They are appended to the instruction **by default** (`recipe/README.md:8-9`). The paper's full-set/board number uses "task description + wiki" [paper §4, Table 4].

> **Leakage note:** the per-task score ladder is public wiki text, so the leaderboard accepts it. It is still task-specific prompting, which our ROBODOJO.md currently forbids ("no recipes, no score ladders"). The wiki is worth about +5 SR points for EmbodiedEvo on dev: 30.17 → 35.08 [paper Table 4]. It *hurt* Open (50→45) and Precision (15→12.5).

### 1.3 What the frozen skills and lessons look like

**No learned skills or lessons are released.** The repository ships only format guides:

- `tasks/RoboDojo/Workspace/skills/example.md:1-28` says "This directory initially contains no validated controllers".
- `lessons/example.md` gives the format `## name / Signature / Instead / Evidence / Status: verified|scene-specific|hypothesis`.
- "Cluster campaigns ... result archives are outside this source distribution" (`README.md:104-105`).

The required skill shape (`skills/example.md:6-24`):

- a closed loop: read the observation → compute the error against explicit targets → take a bounded action → observe → stop on tolerance, no progress, success or budget;
- parameterised, with no scene constants;
- a companion `.md` giving preconditions, units, frames, **observation/execution IDs as evidence**, and whether it was tested beyond one scene.

From the paper [App. E]:

- 146 files and 388 KB across the transfer packages.
- Named examples: `ee_move.py`, `pick_from_above.py` (3 mm alignment tolerance, budget check, descend/close/lift, returns `'lifted_needs_visual_check'`), `rim_pick.py`, `lessons/downward_approach.md`.
- Lessons such as "bound each translation (8 mm)", "inspect the load after carry" and "reserve actions for the home return before the terminal check".

### 1.4 Docker

See §1.1. The simulator, assets, GPU and ledger stay on the host. The container sees only the workspace, a staged `CODEX_HOME` and the socket (`docs/PROTOCOL.md:47-50`).

Credentials:

- `auth.json` and a filtered `config.toml` are copied into a per-session home (`launch_agent.py:210-262`).
- The model key is therefore *inside* the container, which has network access to the model endpoint.

### 1.5 Cost per episode [paper Table 8, §4.5–4.6, App. E]

These are Test episodes only, on dev scenes, with Astra/xhigh, failures included. Playground acquisition is **excluded**.

| Condition | $/episode | Code execs | Native actions |
|---|---:|---:|---:|
| No experience | 4.058 | 22.3 | 723 |
| Raw history | 12.203 | 23.7 | 705 |
| Skills/Lessons | **5.732** | 29.0 | 723 |

- Tokens per Test episode: **3.96M** (Skills/Lessons) against 4.46M (Raw history).
- Cache reads are 97.6% of input for Skills/Lessons and 82.6% for Raw history.
- The rates used for the estimate: $10/M input, $1/M cache read, $12.5/M cache write, $50/M output. "Estimates, not invoices."
- Wall time is not reported. Token use is tracked from Codex rollout JSONL (`orchestrator/files.py:32-60`).

> At 6,300 episodes a full board submission is about **$36k** of Test inference alone at $5.7/episode.

### 1.6 Structured memory vs raw logs

"Raw history", called "condition C" in code (`orchestrator/program_records.py:1-60`), exports the full ordered Playground execution history.

- It contains code, stdout, `result.json`, observations and frames.
- It is **never selected by success** (`:31-33`).
- Fields `score`, `reward`, `shaped_reward`, `ledger`, `skills` and `lessons` are rejected recursively (`:20-28`).
- It is mounted read-only as `program_records/` (`session_runner.py:185-191`).

Result [paper App. E]: Skills/Lessons beat Raw history by **+0.39 SR points, 95% CI [−5.11, +5.94]**, and by +2.23 Score, CI [−2.60, +7.25].

- n = 210 episodes per condition: 5 dev scenes × 42 tasks. The CI comes from a paired task-bootstrap.
- This is **no measurable SR difference**. The real gain is **cost: −53%**.
- Against no experience: 8.50 → 30.17% SR, +21.7 points.

Transfer [paper §4]:

| Harness source | ΔSR over no experience |
|---|---:|
| Same task | +21.39 |
| Mismatched task | +16.67 |
| Related task | +14.72 |

- The authors credit generic content ("bounded motion, retention checks, terminal return-home checks").
- An Astra-written harness lifts Astra/medium from 4.33% to 23.06%. Sol and Luna gain little.

> **Knowledge written by the strong model is reusable by a cheaper executor.** That is a route to cutting cost per episode.

Board breakdown [paper Table 3, 2026-10-09 snapshot]:

| Dimension | Score / SR |
|---|---|
| Memory | **71.9 / 69.7** |
| Open | 55.5 / 53.0 |
| Precision | 23.7 / 13.3 |
| Long-horizon | 37.2 / 23.5 |

- Memory is where a code-writing agent with persistent variables dominates: you just store the observation.
- Precision is where it loses badly against WAMs (VPP2 35.2% SR).

---

## 2. RAX (robot-agent-exploration): `robodojo-behavior/native`

### 2.1 Harness: CLI, tools, sandbox, observation

**Topology** (`docs/ARCHITECTURE.md:25-59`):

- The CLI runs natively on the host as the user.
- It starts a stdio MCP server, `harness/mcp_bridge.py:1-34`: 34 lines that pipe stdio to the session's Unix socket.
- Behind the socket are the MCP frontend, a supervisor, a gateway, per-environment native workers, the simulator, and a robot-only preview simulator.
- The agent's own Python scripts reach the *same* MCP through `from api.runtime import Context; ctx.call("robodojo_observe")` (`ARCHITECTURE.md:253-265`).
- Calls are serialized; scripts are not.

**Claude Code launch** (`harness/claude_cli/auto_research.py`):

- Command (`:257-273`): `claude -p --model M --effort xhigh --mcp-config home/mcp.json --strict-mcp-config --disallowedTools WebFetch WebSearch [Agent Task] --resume <id> --output-format stream-json`.
- Settings (`:196-226`):
  - `permissions.defaultMode=acceptEdits`, allow `mcp__auto_research`;
  - `sandbox.enabled=true, failIfUnavailable=true, autoAllowBashIfSandboxed=true, allowUnsandboxedCommands=false`;
  - filesystem `allowWrite=[workspace, socket dir]`, `denyRead=` the repo, other sessions and `~/.ssh`/`~/.claude`/...;
  - network `allowedDomains: []` with Unix sockets allowed.
- Environment (`:278-298`):
  - `MCP_TOOL_TIMEOUT` is set large, because "one robot tool call may block for days";
  - **`CLAUDE_CODE_AUTO_COMPACT_WINDOW=250000`** ("every call re-reads the whole context", `:31-33`);
  - **`ENABLE_PROMPT_CACHING_1H=1`**, because "agents often wait 5-30 minutes on rehearsals and episodes, and a lapsed 5-minute cache re-writes the whole context".
- The workspace `.claude/skills` is a symlink to `.agents/skills` (`:45-50`), so one skill tree serves both CLIs.
- HOME must differ from the workspace, or Claude Code skips the project skills (`:281`).

**Codex.** The same hide list goes through a Codex permission profile. Subagents use `multi_agent` with `max_concurrent_threads_per_session` (`harness/codex_cli/configuration.py:13-15`).

**RoboDojo MCP tools** (`services/robodojo/mcp_server.py:770-1060`):

- `robodojo_status`.
- `robodojo_observe`: RGB, depth, calibration and state; does not step.
- `robodojo_pixel_to_position`: takes the `step_id` of the frame the pixels came from, and **rejects stale selections**. It uses depth, or triangulation from ≥2 views.
- `robodojo_pose_math`.
- `robodojo_step`: 1..50 absolute 14-D joint rows.
- `robodojo_step_eef`: 1..50 EEF waypoints, bounded to 0.02 m / 0.1 rad per row, DLS clamp 0.05 rad.
- `robodojo_free_space_move`: single-arm cuRobo plan, ≤50 rows, with an optional `preview_only` mode.
- `robodojo_execute_motion_plan`: runs only if the state is within 0.002 of the plan start, otherwise `Plan_Stale`.
- `robodojo_fk_preview`.
- `gemini_generate` (Gemini 3.8 Flash, $10/session [paper]).
- Lifecycle: `start_episode` (no seed/layout argument), `evaluate` (trusted binary `task_complete`), `finish`, `register`, `rehearse`, `submit` (`ARCHITECTURE.md:189-207`).

**Replies** (`ARCHITECTURE.md:215-233`):

- JSON first, then images. Depth is a uint16 mm PNG.
- **A motion call attaches only its final frame**; the full sequence is saved to disk.
- The gateway strips reward, score, intermediate success, seeds, layout IDs and host paths.
- **Errors pass through whole** (type, stage, reason, innermost traceback, `simulator_state`) and never lock the episode.

**Observation profiles** (`ARCHITECTURE.md:242-251`):

- `rgbd` is the **default**: RGB, depth, intrinsics, extrinsics and pixel→3D.
- `rgb-only` removes depth, calibration and pixel→3D *before encoding*.

**Isolated runs** of frozen programs (`ARCHITECTURE.md:158-179`):

- bubblewrap with no network, all capabilities dropped, 8 GiB, no GPU;
- the robot is reached over stdin/stdout MCP with robot tools only;
- **the agent CLI is SIGSTOPped during register and rehearse**, so nothing can modify the bundle.

### 2.2 Online exploration scheme

The cycle is **Explore → Compile → Evaluate** (`README.md:40-52`; agent text `auto_research_agent/AGENTS.md:54-134`).

1. **Solve by hand.** The agent drives the robot through MCP and confirms with `evaluate`/`task_complete`.
2. **Automate.** It writes `code/<project>/controller.py:main(ctx)` from "observation-driven steps that check their preconditions, verify their results and report when they fail".
3. **Pressure-test across new layouts.** Every `start_episode` is the *next* layout of collection 0.
4. **Submit.** `register` hashes the bundle. `rehearse` runs it unattended once, in a fresh episode, under formal limits; it must end cleanly but need not succeed. `submit` closes exploration, and there is no second submission (`ARCHITECTURE.md:270-284`).

Memory (`agent_skills/task/experience-memory/SKILL.md:1-77`):

- `memory/TIPS.md`: write as soon as a local effect is visibly successful.
- `memory/skills/<slug>/SKILL.md` plus `evidence.json` holding manifest, step and frame IDs.
- `memory/INDEX.md` with tags, confirmation level (local-visible vs task-confirmed) and last episode.
- `memory/PROGRESS.md` for compaction recovery (`AGENTS.md:179-197`).

Retrieval: read INDEX first and "retrieve only matching learned skills ... do not load the entire history".

The exploration budget N is 1..1000 episodes, and **the session itself has no wall-clock or model-call budget** (`ARCHITECTURE.md:290-301`).

Parallelism:

- up to 8 exploration environments (separate processes), with one ledger for all episode starts;
- infrastructure errors are refunded, and the layout is replayed;
- up to 8 subagents.

Agent-formal mode is an alternative: live copies of the agent each get one held-out layout and carry `memory/`, `code/` and `handoff/` (≤100 MB) plus the conversation (`resume|compact|clear`) (`ARCHITECTURE.md:333-358`).

### 2.3 RoboDojo results [paper §5.2, Tables 8 and 13]

- **7 tasks** only: toast-r, mugs-r, sweep-r, plug_in_charger, swap_blocks, make_kong, pour_by_language.
- Code mode: the compiled bundle runs alone on 30–50 unseen layouts.
- Mean score **78.3**, against 24.9 for the best board baseline on those 7 tasks (VPP2).
- Per task: plug 94.0, swap 100, kong 96.0, pour 100, sweep-r 71.7, mugs-r 39.2, toast-r 47.2.
- Exploration scaling, N = episodes:
  - swap: 34 → 74 → 100 at N = 4, 8, 16;
  - kong: 0 → (none submitted) → 64 → 96 at N = 4, 8, 16, 32;
  - toast-r: 2.5 → 31.7 → 47.2 at N = 16, 32, 64;
  - mugs-r: flat at 33–39.
- Agent-in-the-loop formal numbers are **not** reported for RoboDojo.
- It is **not on the official board**, and covers only 7 of 42 tasks.

Privileged inputs [paper]:

- the public rubric plus "short clarifications";
- **a summary of the official task code** without thresholds, for plug and pour;
- one image of the completed task;
- the evaluator's `task_complete` during exploration.
- The parallel make-toast study *includes the numeric thresholds* plus keyframes.

Wall time on make-toast [paper]:

| Setup | Time | Score |
|---|---|---|
| Codex 1×1 | 14.2 h | 66.4 |
| Claude Code 1×1 (48 episodes) | 13.7 h | 63.3 |
| **Claude Code 4 agents × 4 envs** | **4.9 h** | **92.2** |

- The first success took 17 min of inference.
- Compiled, an episode takes **6 s** of inference.
- The episode wall cap is 6 s per native step (`services/robodojo/timeouts.py:7`).

### 2.4 Depth

**Yes.** The RoboDojo study runs `rgbd` [paper §5.2 "RGB-D observations"; `ARCHITECTURE.md:246-249`].

RGB-D vs RGB-only ablation [paper Table 8]: **31.1 vs 13.8 score, +17.3**, n = 3 task/budget pairs. Without depth, the agent falls back on pixel-to-table calibration, wrist servoing and hard-coded constants. On toast, 36 of 45 RGB-only episodes failed at the lever.

> The official RoboDojo policy observation has no depth (our brainstorm, §1). If we want RAX-like geometry legally, we need multi-view triangulation from RGB plus calibrated extrinsics. RAX's `pixel_to_position` already has a triangulation fallback.

The HF dataset card (read via WebFetch, not downloaded) covers 34.9 GB of **real-robot** Claude Code sessions (Franka and RB-Y1), not RoboDojo.

- Per session: `claude-home/projects/*.jsonl` transcripts, `mcp.jsonl` tool calls, `agent-workspace/{memory/PROGRESS,RUNS,TIPS, skills, code}` and `runtime/frames/<id>/{rgb, depth, meta.json}`.
- Runs take 2.7–6.9 agent-hours with 1.6k–5.8k tool calls.

---

## 3. OpenETA, `robodojo-v112`

### 3.1 Wiring

`scripts/codex_plugin_smoke.py:195-230, 405-430` launches `codex exec -C work --ephemeral --json -s read-only -m gpt-6-astra -o final.txt`, with the RoboDojo host as a **Codex plugin MCP server**. The config sets:

- `shell_tool = false`
- `memories = false`
- `multi_agent = false`
- `web_search = "disabled"`
- `history.persistence = "none"`
- `mcp_optional_startup_grace_ms = 0`, because "a tool-less turn" otherwise starts before the simulator is up.
- `OPENAI_*` and `CODEX_*` variables are stripped from the plugin environment.

> **Codex is used as a pure tool-calling loop** (no shell, no files), just with Codex's auth and context management. This is architecturally the same as our own API loop.

Run parameters (`docs/robodojo-v112.md:110-127`): 200 tool calls max, `--timeout 0` (no wall limit), seed 0, one fresh session per attempt.

### 3.2 Tools (`tools/codex_robodojo_host.py:186-300`)

| Tool | What it does |
|---|---|
| `read_skill` | Fixed, hash-bound workflow docs (pick-place, stack-support, align-insert, push-slide, tool-use, articulated, cloth, moving-target; `tools/robodojo_skill_library.py:10-51`, ≤32 KB each, sha-verified against the config). Costs one request and zero actions. |
| `episode_status` | Episode state. |
| `observe` | RGB and state. |
| `mark_point` | A pixel becomes the measured **RGB-D** world point plus a marked crop for verification. |
| `move_to` | Waypoints and contact modes `contact_mark`/`support_contact`/`push_contact`/`tool_contact`, each bounded: push ≤5 cm, adjust ≤3 cm/15°. `max_duration_s` ≤2. |
| `gripper_control` | Opens or closes the gripper. |
| `finish_episode` | `success=true` requires official success. |
| `check_task_completion` | **opt-in oracle**: semantic deviations without coordinates or thresholds, "valid only at its reported action index" (`:279-286`). |
| `review_experience` | ≤12 frozen historical segments (before/after images, state, request, feedback), "never executable replay" (`:288-292`; `tools/harness_experience.py:1-40`, ≤48 MB, schema-validated, hash-verified). |

### 3.3 Observation freshness

- **Contact marks expire.**
  - A contact point that predates a gripper/contact action gives `contact_point_stale`, with "mark a fresh visible surface" (`tools/codex_atomic.py:115`).
  - Contact authorisation needs the TCP within 0.10 m of the measured point (`:125`).
  - After any free-space or contact motion with unknown outcome, the old marks are retired (`:352-358`).
  - Handles expire on release or regrasp and cannot cross episodes (`codex_robodojo_host.py:196`).
- **Fresh-observation obligation.** If a world mutation returned no fresh observation, the host *substitutes* an observation refresh for the next requested stage. It returns `fresh_observation_required`, and the request is **not replayed** (`tools/codex_host.py:346-390`).
- **Unknown remote outcome.** The episode stops: "Do not resend that action or infer completion from position". A new session is required (`codex_host.py:375-378`). The release says the same thing: "An unknown action outcome must not be blindly replayed" (`docs/robodojo-v112.md:21-23`).
- **Every image is tagged** with `action_index` and `observation=fresh|cached|outcome_unknown_last_known` (`codex_robodojo_host.py:600-608`, `:761`).
- **Capture clock.** The physics tick and sim time at capture are recorded, with `camera_exposure_time_verified: False` stated explicitly (`sim/envs/robodojo/observation_clock.py:1-19`).

### 3.4 "Repair tools" and versioned evidence

**Host-side recovery modules** live in the env adapter:

- `boundary_recovery.py`: a non-worsening collision exit.
- `clearance_recovery.py`: retreat from sub-margin gaps.
- `separating_recovery.py`: a provable upward separation from finger/table contact.
- `checked_control.py`: monitored joint execution.
- `bounded_wait.py`: billed settling of ≤2 s through the native action path.

Each docstring states what it does *not* guarantee, for example "NOT a continuous certificate" (`sim/envs/robodojo/*_recovery.py:1-8`).

**Versioned configs.**

- Each preset is a combination of `{benchmark, embodiment, execution, interface}` components (`configs/harness/presets.json`, 15 `*_repair_v112` rows).
- The interface component pins the **sha256 of about 105 source files**, including skill docs and recovery modules (`configs/harness/components.json:2160-2260`).
- `registry.resolve(id).require_executable()` refuses to run if any hash differs (`docs/robodojo-v112.md:44-56`).
- "Older configurations ... may deliberately reject current source hashes. Do not repin them and call the result the same historical experiment" (`:17-19`).

**Task-wording supplements** are versioned separately (`tools/task_rule_supplements.py`). The rewards are not rewritten.

**Tool promotion dossiers** (`docs/generated/promotion-dossiers/*.json`). These are per-tool maturity records. Example from `compile-grasp-seed.json`:

- "verified", "three_person_review_approved";
- evidence: valid and invalid live requests seen, result conformant, collected in shadow mode, integration canary session.

### 3.5 23/42 [doc `docs/robodojo-rsi-results.md:1-56`]

| Pass | Tasks | Coverage |
|---|---|---|
| Pass@1 (v108) | 23/42 | 54.8% |
| RSI Pass@2 | 27/42 | 64.3% |
| RSI Pass@3 | 28/42 | 66.7% |

What the figures mean:

- "Coverage" means ≥1 official success among the first n attempts of a task, on **seed 0**, with ≤200 tool calls, **no wall-clock and no native-action limit**.
- The RSI is "human-assisted, cross-version": humans diagnose, approve and correct wording between attempts.
- `check_task_completion` is "additional, privileged semantic feedback".
- **Not comparable to board SR.** 23/42 is *task coverage at one layout*, not an SR over 50 layouts × 3 seeds.
- Token accounting is incomplete, and no cost is reported.

---

## 4. RACaP and Noematrix RoboRSI (brief)

### RACaP [code + paper; LIBERO, not RoboDojo]

- **Split.** A coding agent evolves typed Policy APIs, a ReAct harness and memory *before* deployment. At runtime a visual ReAct agent only selects APIs and arguments, "without generating or repairing source code on the robot's execution path" (`README.md:19`).
- There are six APIs: `pickplace`, `insert`, `stack`, `push`, `articulate`, `actuate_control`.
- **Loop** (`docs/EVOLUTION.md:31-50`):
  1. The scheduler picks a failure cohort.
  2. The parent is evaluated.
  3. A **visual critic** reads traces plus before/after frames and names "the earliest likely physical divergence".
  4. The coder edits `solution/`, `memory/` or tests in a **fresh git worktree**.
  5. The candidate must pass syntax, import, unit and smoke checks.
  6. **Parent and candidate run on the same tasks, seeds and budgets.**
  7. Promotion happens only if native successes strictly increase.
- **Selection rule** (`evolution/harness/selection.py:17-60`):
  - It raises unless the two episode key sets are identical.
  - Success up → promote. Success down → reject.
  - On a tie, a lexicographic cost `(mean sim steps, VLM calls, tool calls)` decrease gives an "efficiency" promotion.
- **Guards** (`evolution/harness/guards.py:27-31`) regex-reject candidate code containing `oracle`, `task_completed|predicate_status`, or **`task_id ==` / `seed in (...)`** (task-ID specialisation).
- **Memory** is plain rule bullets, e.g. `memories/transport_react.md`: "a retry must change one supported control", "never invent prose nudges".
- Results [paper]: LIBERO-PRO zero-shot 32.8 → 45.0%, n = 180, 33 rescued and 11 lost.
- Caveat: promotion is selected on the development cohort, so the transfer suite is the honest number.

### Noematrix RoboRSI [code + paper; LIBERO/RoboTwin]

- **Roles.** Manager, Planner, Engineer and Reviewer over a task–skill tree: `long_horizon/` → `atomic/` → `base/<robot>/<prim>/`. The Reviewer traces a failure to the "earliest responsible node" (`README.md:17-24, 50-70`).
- **Promotion gate.**
  - A change must succeed at least once on its source task and do at least as well as the current skill on dev seeds 21, 22, 23.
  - It is installed **task-local** (`skills/task_local/<task>/`) unless a cross-task gate on 2 other tasks that use the skill passes.
  - A `human` mode lists diffs for approval.
  - Every change is a git commit (`docs/EVALUATION.md:162-164`).
- **Modes.** `evolve` vs `frozen`: in frozen mode no skill, wiki, plan or training data is written (`docs/EVALUATION.md:1-10`).
- But the headline LIBERO and LIBERO-PRO numbers are "**improving online**", i.e. evaluated while evolving (`README.md:33-38`). Only LIBERO-Plus is on a frozen library.
- **Consolidation.** Stable atomic paths are compiled into code. Code on gives 29.0% vs 21.5% off (n = 600 matched pairs), with −29% tokens, −27% calls and −17% wall time (`README.md:42-48`).
- **State-access firewall** (`roborsi/agents/safety/gt_firewall.py:1-114`). The most instructive file in this survey:
  - It records a real incident: the Manager could `cat` env source and wrote success criteria into task wikis, and "**fourteen solved tasks turned out to have been solved while knowing the criterion**".
  - Rule 1: "**guard the read, not the write**". The filter sits at `read_wiki`.
  - Rule 2: "a criterion is a spec, not an observation". Measured object state may be written down; rules read out of the env may not.
  - Mechanism: fingerprint the task's `check_success` source (numbers with ≥3 decimals, asset IDs) plus a coarse vocabulary ban (`check_success`, `get_actor_pose`, `ground truth`, `envs/*.py`, ...). **Whole entries** are dropped, not clauses (`:96-114`).
- **CLI sessions are a switch.** Planner and Reviewer run as persistent Codex/Claude CLI sessions, or as stateless API calls with `ROBORSI_ROLE_SESSION=0` (`docs/EVALUATION.md:162`).
- Median episode is about 22 min, p90 about 50 min (`docs/EVALUATION.md:24`).

---

## 5. Cross-cutting patterns

### 5.1 How a coding-agent CLI is wired to a sim robot

All three RoboDojo entries use the same three layers. The **host owns the simulator** and serves a **Unix-socket API**. The **agent sandbox** has no GPU, no simulator source and no ground truth.

| Variant | Who uses it | What crosses the boundary |
|---|---|---|
| A. Code cells | EmbodiedEvo | Python source → host `exec` in a persistent namespace; the agent's shell does image analysis locally |
| B. MCP tools + scripts | RAX | MCP tools for the CLI *and* for the agent's Python (`api.runtime.Context`) |
| C. MCP tools only | OpenETA | No shell; the CLI is just a tool-calling loop |

Practical details that recur:

- Physics is paused while the agent thinks (EmbodiedEvo empty-chunk poll; RAX "simulator paused").
- No blind retry on an unknown outcome (EmbodiedEvo `env.py:38`, OpenETA `codex_host.py:375`).
- Motion replies attach only the final frame; the full sequence goes to disk (RAX).
- Long-blocking tool timeouts (RAX `MCP_TOOL_TIMEOUT`).
- A 1-hour prompt cache plus a 250k compaction window for long waits (RAX).
- Errors pass through whole (RAX).

### 5.2 How skills are stored, retrieved and verified

| | Storage | Retrieval | Verification before reuse |
|---|---|---|---|
| EmbodiedEvo | `skills/*.py` + `.md` (preconditions, evidence IDs); `lessons/*.md` with Status | the agent reads the files; `--include` concatenates | **none automatic**: Playground evidence only; "verified" means reproduced in *this* scene |
| RAX | `memory/TIPS.md`, `memory/skills/<slug>/{SKILL.md,evidence.json}`, `INDEX.md`; `code/<project>/` | INDEX-first, match on tags and conditions | register (hash) → **unattended rehearsal in a fresh episode** → submit |
| OpenETA | hash-pinned skill docs plus frozen experience segments | `read_skill(id)` and `review_experience(seg)` tools | sha256 binding plus a three-person promotion dossier per tool |
| RACaP | typed API code plus memory bullets in git | the ReAct agent selects an API | **paired parent/candidate on identical episode keys, strict improvement** |
| RoboRSI | task–skill tree in git; task-local first | the Planner reads the tree and wiki (firewalled) | dev-seed gate ≥ current; cross-task gate for global promotion |

### 5.3 How train/test leakage is prevented

- **Separate layouts.**
  - EmbodiedEvo samples Test seeds *after* the freeze and excludes the Playground seed.
  - RAX explores collection 0 and evaluates collection 1, with the schedule fixed up front.
  - RoboRSI uses dev seeds 21–23.
- **Freeze plus hash before test.** EmbodiedEvo `frozen_harness`, RAX bundle hash and checkpoint, OpenETA config sha.
- **Strip private fields at the gateway.** Reward, score, seeds and layout IDs (RAX `ARCHITECTURE.md:223-225`, EmbodiedEvo `program_records.py:20-28`).
- **Read-only knowledge in Test, and no reset** (EmbodiedEvo).
- **Content firewalls.** RACaP regexes against oracle and task-ID specialisation. RoboRSI fingerprints predicate source on read.
- **Weak spots seen:**
  - EmbodiedEvo default recipes include score ladders.
  - RAX gives task-code summaries, and sometimes thresholds.
  - OpenETA has an oracle `check_task_completion`, humans in the loop, and seed 0 only.
  - RoboRSI's online-evolving headline numbers.

### 5.4 Cost and wall time per episode

| System | LLM $/episode | Tokens/episode | Wall |
|---|---|---|---|
| EmbodiedEvo Test | $5.73 (Astra xhigh, est.) | 3.96M (97.6% cached) | n/r |
| RAX compiled bundle | ~$0 (Gemini ≤ $10/session shared) | ~2k (real robot) | 6 s inference; sim time dominates |
| RAX acquisition (one task) | n/r | n/r | 4.9–14.2 h/session |
| RoboRSI (LIBERO) | n/r | n/r | median 22 min, p90 50 min |
| OpenETA | n/r | n/r | no limit |

---

## Patterns to adopt for controlr's skill library + repair loop

1. **Host-owned session server; the agent only sees a socket.** Our `robodojo-serve` is already the host.
   - Add a session API (observe / exec-or-act / status / finish) on a Unix socket.
   - Strip seed, layout, reward and score before anything reaches the model (EmbodiedEvo `bridge.py:11-16`, RAX gateway).
2. **A skill is code plus a contract plus evidence.**
   - Use `skills/<name>/{skill.py, SKILL.md, evidence.json}`.
   - SKILL.md covers preconditions, units and frames, stop conditions and known failures.
   - evidence.json covers run, turn and frame IDs and the confirmation level (`local-visible` | `task-confirmed` | `rehearsed-unattended`).
   - Keep one `INDEX.md` for retrieval, and load only the matches (RAX experience-memory).
3. **Lessons are typed.** Use `Signature / Instead / Evidence / Status: verified|scene-specific|hypothesis` (EmbodiedEvo `lessons/example.md`).
   - Expect cost savings, not SR gains, from structured memory vs raw logs: +0.39 points, CI ±5.5, cost −53%.
   - Measure both.
4. **The promotion gate is paired and on fixed keys** (RACaP `selection.py`).
   - Parent and candidate run on identical (task, layout, seed) sets.
   - Promote only on a strict native-success increase. A tie plus lower steps/calls counts as an efficiency promotion.
   - Archive the rejects with their diffs.
   - New skills start **task-local**, and global promotion needs a cross-task no-regression gate (RoboRSI).
   - Development layouts must be disjoint from the eval layouts. RoboDojo's eval layouts are consumed in order from seed S, so develop on other seeds or collections.
5. **An unattended rehearsal before freeze** (RAX).
   - Any compiled skill or program must run end to end once in a fresh episode, with the agent paused, before it can enter the frozen set.
6. **Freeze by hash; refuse to run on mismatch** (EmbodiedEvo `frozen_source.py`, OpenETA `require_executable`).
   - The frozen set ID goes into every run log.
   - This fits our rule 8 (reproducibility) and the "defaults never change silently" rule.
7. **A firewall on the read path** (RoboRSI `gt_firewall.py`).
   - Any text that reaches the runtime model (lessons, wiki, handoff) passes a filter.
   - The filter uses fingerprints of the task's success predicate plus a coarse vocabulary ban, and drops whole entries.
   - Add a regex guard on skill code for `task ==`, `layout`, `seed` and `reward` (RACaP `guards.py`).
8. **Observation freshness.**
   - Every frame and derived point carries `action_index`/turn and `fresh|cached|unknown`.
   - Pixel→3D selections and plans are rejected if they are stale (RAX `step_id`, `Plan_Stale` at 0.002).
   - An unknown action outcome becomes a forced re-observe, never a replay (OpenETA `fresh_observation_required`).
   - These belong in `SafetyEnvelope` and feedback, as code, consistent with rule 7.
9. **Critic → minimal patch** (RACaP).
   - The repair agent gets traces plus before/after frames, and must name the *earliest* physical divergence before editing.
   - Each patch changes one thing, in its own git worktree.
10. **Let the strong model write knowledge and a cheaper one execute it.**
    - EmbodiedEvo: an Astra-written harness takes Astra/medium from 4.3% to 23.1%.
    - Compiling to code removes most runtime LLM cost (RAX 6 s/episode; RoboRSI −29% tokens).
    - With 6,300 board episodes, **runtime cost per episode is the binding constraint** (≈$36k at EmbodiedEvo's $5.7).
11. **Keep supplied knowledge separate from learned knowledge.**
    - Recipes and score ladders are a separate config switch with its own provenance record (EmbodiedEvo `recipe.json`).
    - Report results with and without them; they help Memory and Generalization and hurt Open and Precision.
12. **Depth-like geometry is worth about +17 score points** (RAX Table 8).
    - Within the official RGB-only observation, build calibrated multi-view triangulation (`pixel_to_position`-style, ≥2 views, residual ≤3 cm) as a tool.

## Should our runtime agent be a CLI coding agent (Codex / Claude Code) or our own API loop?

**For a CLI coding agent**

- All three published RoboDojo coding-agent results use one:
  - EmbodiedEvo: board 38.95% SR, Memory about 70–79;
  - RAX: 78.3 on 7 tasks;
  - OpenETA: 23/42 first pass.
- The CLI provides context compaction, resume, subagents, file tools and caching for free.
- Shell plus local Python means the agent can analyse images, write helpers and persist variables. That is exactly what wins **Memory**: 71.9 vs ≤56 for every WAM/VLA [EmbodiedEvo Table 3].
- Skills and lessons as files are native to these CLIs (`.claude/skills` symlink, AGENTS.md).
- Parallel subagents cut RAX acquisition from 13.7 h to 4.9 h *and* raised its score from 63 to 92.
- Session logs are rich for later analysis.

**For our own API loop**

- OpenETA *disables* Codex's shell and uses it only as a tool-calling loop (`shell_tool=false`). Its 23/42 came from host tools, not from coding.
- EmbodiedEvo's cost is about 4M tokens and $5.7 per Test episode. CLI agents re-read huge contexts, and RAX had to add a 250k compaction cap and a 1 h cache just to stay sane.
- Per-episode wall time is unreported or hours. Our brainstorm requires ≤5 min/episode × 6,300.
- CLI internals (system prompt, compaction, cache breakpoints, effort) are not under our control. That breaks our cache invariants (rule 6) and the "every axis is a config field" rule (rule 1).
- Version pinning is fragile. EmbodiedEvo pins Codex 0.159.2, and RAX documents several Claude Code quirks (HOME vs project skills, a sandbox list that fails beyond about 100 paths).
- Our API loop already has deterministic feedback, prefix caching, decision heads and run logs.
- The model matters more than the harness: Astra 72.7% vs Opus 25.3% on Astranomy. Astra is reachable via our router without Codex.
- The rules restrict credentials: only the `OMNIROUTE_*` keys. A CLI login would be a new credential path.

**Recommendation**

Split the roles, as RACaP and RoboRSI do.

- **Acquisition/repair (offline, between episodes): a CLI coding agent.**
  - Claude Code or Codex in a sandbox, with no network except the model endpoint.
  - It works on dev layouts with a host session API and writes `skills/`, `lessons/` and compiled programs.
  - Its output passes a paired gate plus an unattended rehearsal, and is then frozen by hash.
  - Wall time and tokens here are a one-time cost per task.
- **Runtime (the 6,300 board episodes): our own API loop.**
  - The model selects and parameterises frozen skills and programs through host tools.
  - It keeps a persistent per-episode scratch state for the Memory tasks: EmbodiedEvo's edge is the persistent namespace, not the CLI.
  - It falls back to low-level actions.
  - Cost stays controlled and cached, and every axis stays a config field.
- **Validation.** Run one A/B on 3–4 tasks before committing (Memory plus one Precision): EmbodiedEvo-style Codex-in-container vs our loop with the same frozen skills, same layouts, same model (Astra). Report SR, score, $/episode and wall time per episode.
