# Open-source LLM-robot-control implementations on GitHub: catalogue and ranking (as of 2026-10-01)

Scope: open-source code that puts a hosted or local LLM/VLM in the robot control loop, 2023-2026, with emphasis on 2025-2026. Covered: agentic robot "OSes", ROS/MCP bridges, LLM + learned-policy harnesses, frontier-model-as-direct-policy harnesses, coding-agent self-improvement frameworks, desktop/social robots, and the LeRobot ecosystem.

Teammates have deep-dived the user's own sources: quackd, innate-os PR #817, metal-arm-harness, piper-astra-jev, llm-robotics-playground, GPT-as-Policy, GPT-Policy, EmbodiedSWE, RoboDojo and Robocurve (see `research/sources/*.md`). The model-level, planner/codegen, VLA-head and benchmark landscapes are in sibling files under `research/landscape/`. This note only places those items in the catalogue.

Star counts and "last push" dates come from the GitHub REST API (`gh api repos/<r>`), queried 2026-10-01. Repos marked "read" were shallow-cloned to `research/repos/<owner>_<repo>` and their README plus key code were inspected.

---

## 0. TL;DR

- **Volume vs. signal.** About 130 `gh search repos` queries and 10 `gh search code` queries surfaced roughly 935 unique repos. Most were noise: chatbots named "robot", "SO-101" course repos, and dozens of non-robotics "awesome-gpt-6-astra" lists. About 45 are real LLM-in-the-loop robot-control codebases, and fewer than 15 have non-trivial evidence (trial counts, benchmarks or real-robot runs).
- **Four architectural families dominate:**
  1. **Runtime/OS + skills + MCP**: dimOS, OM1, PhyAgentOS, ROSClaw, AgenticROS, innate-os.
  2. **ROS introspection bridges**: ROSA, ros-mcp-server, RAI, ros2_mcp.
  3. **LLM planner + frozen learned primitive (VLA)**: RPent, Zetta, Safari SDK agent, Strands Robots, RoboCrew, EmbodiedSkills, Thea, OpenETA.
  4. **Frontier VLM as fine-grained policy through a discrete action vocabulary**: Show-Harness, RoboICL, DrivingBench, metal-arm-harness, Jev workbenches.
- **The 2026 shift.** The best-evidenced systems (RPent, Zetta, Show-Harness, OpenETA, RoboICL) all appeared Jul-Sep 2026. Their common recipe:
  - The frontier model never streams motor commands. It calls a few typed tools: frozen-VLA primitives, bounded Cartesian steps, perception queries.
  - A deterministic host enforces contracts.
  - Every call returns evidence.
  - Experience is kept in files: memory markdown, skills, critics.
- **The closest existing analogue to "Opus API backbone + light action head" is RLinf/RPent.** Its planner is literally the Claude Agent SDK (`rpent/planner/claude_code.py`). A frozen π0.5 is exposed as the retryable tool `pi0_pick`. The paper reports +38.6 pp on LIBERO-Pro and ~~+25.4 pp~~ [corrected: +27.1 pp in the current v5 paper body and PDF abstract (Codex/GPT-5.5 57.1% vs the frozen RLDX-1 baseline at 30.0%). The arXiv listing abstract still shows the older 25.4. The +38.6 pp LIBERO-Pro headline was measured with the **Claude Code (Opus-4.7)** planner: 82.4% vs RATS 43.8%.] on RoboCasa365 over the strongest baselines ([arXiv 2607.08448](https://arxiv.org/abs/2607.08448)).
- **The closest analogue of "distil the backbone into a light head" is showlab/Show-Harness.** The same discrete action units are driven zero-shot by frontier VLMs: Gemini-3.1 Pro, GPT-5.6-sol and Opus 5 reach 86-96% success. LoRA-fine-tuned Qwen3.5-0.8B…9B models run the same units at 12-33 Hz ([project page](https://showlab.github.io/Show-Harness/)).
- **MCP is now the default integration surface**, used by ros-mcp-server, dimOS, AgenticROS, OmniSim, DrivingBench and quackd. But every robot-MCP server that only exposes raw ROS publish/subscribe leaves timing, safety and verification to the LLM. Use them as southbound adapters, not as the harness.

---

## 1. Taxonomy (what the LLM emits → what executes it)

| LLM role | Action interface | Typical rate | Representative repos |
|---|---|---|---|
| Conversational agent / introspection | ROS CLI or rosbridge publish/subscribe/service/action, parameter set | per user turn | ROSA, ros-mcp-server, ros2_mcp, ros-skill, ros2ai |
| Skill selector over a fixed menu | named skills with typed args, behind capability locks or precondition governors | 0.1-1 Hz | dimOS, OM1, AgenticROS, innate-os, quackd, piper-astra-jev, Reachy Mini app |
| Orchestrator of learned policies | `run_instruction_until_done(instruction)` / `run_policy` / `pi0_pick` hand control to a VLA for a bounded segment | VLA 10-50 Hz inside; LLM per segment | Safari SDK agent, Strands Robots, RPent, RoboCrew, EmbodiedSkills, Zetta |
| Direct fine-grained policy via discrete units | `MV_LEFT`/`MV_FWD`/`GRASP`…, bounded Cartesian deltas, timed motion leases | 0.2-1 decision/s (frontier); 12-33 Hz (distilled small VLM) | Show-Harness, RoboICL, DrivingBench, metal-arm-harness, jev-robot-control |
| Coding agent that writes and evolves controllers/skills | Python skill scripts, critics and recovery programs, then validated | offline, between rollouts | ENPIRE, ASPIRE, Zetta (evolution loop), CaP-X, EmbodiedSWE |
| Typed "System One" decider | probability over enumerated options from text state | 2-10 Hz | dimOS `typesafe` agent, jev-robot-control, embodied-jev, RoboICL Jev gate |

---

## 2. Catalogue

★ = stars on 2026-10-01; "push" = last push date. Maturity scale: **P** = production-oriented releases/CI/docs; **R** = research code with paper and numbers; **D** = demo/hobby; **S** = stale (no push for more than 6 months).

### 2.1 Agentic robot runtimes / "robot OSes"

| Repo | ★ / push | LLM role, action interface, perception, hardware | Maturity | Notable design idea |
|---|---|---|---|---|
| [dimensionalOS/dimos](https://github.com/dimensionalOS/dimos) (read) | 4601 / 2026-10-01 | **LLM:** an MCP-client agent that calls skills served by an in-process MCP server (`dimos/agents/mcp/mcp_client.py`). LangChain `init_chat_model`; default model `"gpt-5.6-luna"`; Ollama variant available.<br>**Skills:** e.g. `move_to`, `navigate_with_text`, `move_to_pose`, `open_gripper`.<br>**Perception and memory:** spatio-temporal memory, detectors, VLMs.<br>**Hardware:** Go2 (stable), G1, xArm, Piper (beta), MAVLink drones (alpha). | P (v0.0.14, 2026-09-19; 45 contributors; "Pre-Release Beta") | **Capability locks:** `@skill(uses=[CAP_MOVEMENT])`. The server refuses conflicting calls with "Cannot start X: capability Y is held by Z" (`dimos/agents/capabilities.py`).<br>**Jev "System One" nav agent:** `dimos/agents/typesafe/`, `DEFAULT_MODEL="jev-latest"`, `NAV_MAX_HZ=2.0`, `PUBLISH_HZ=10.0` for cmd_vel, deadman zeroing, `MIN_PROBABILITY=0.5`. It converts a lidar/detection scene into words ("ahead_left", "near") for a text-only decider.<br>**`AGENTS.md`** targets coding agents ("vibecode your robots"). |
| [OpenMind/OM1](https://github.com/OpenMind/OM1) (read) | 2939 / 2026-09-30 | **LLM:** a "cortex" LLM (OpenAI, xAI, DeepSeek, Anthropic, Gemini, Ollama…) runs at a configured `hertz` (`config/unitree_go2_autonomy.json5`: `hertz: 1`).<br>**Prompt:** a "fuser" paragraph (`internal/fuser/fuser.go`): "Current observations:", KB/RAG context, "MEMORY:", "Available actions:", MCP tool descriptions, ending "What will you do next?".<br>**Actions:** labeled actions (`speak`, `emotion`, `Move`) routed over Zenoh/ROS2/CycloneDDS connectors. | P (Go runtime; Python branch deprecated; 100+ contributors) | Natural-language data bus. The paper ([arXiv 2412.18588](https://arxiv.org/abs/2412.18588)) reports 4 LLMs over a language bus, 1 Hz fusion and about 40 bits/s. Good for social/quadruped behaviour; too slow for manipulation. |
| [PhyAgentOS/PhyAgentOS-core](https://github.com/PhyAgentOS/PhyAgentOS-core) (read) | 2639 / 2026-10-01 | Agent plans tool calls through a single **Forge Gateway** HTTP contract (`/tools`, `/invocations`) to Dora dataflows, then robot/sim. | P/R (v1.0.0, 2026-09-05; [arXiv 2607.16636](https://arxiv.org/abs/2607.16636)) | Keeps **execution, evidence and verdict** as separate records. Before/after images are SHA-256-stamped. The verifier has modes off/audit/enforce/recovery. Recovery is an append-only `PlanRevision`; "unknown effects are reconciled and never retried blindly". State-as-a-File (Markdown+YAML) sits at the cognition/physics boundary. |
| [ros-claw/rosclaw](https://github.com/ros-claw/rosclaw) (read) | 214 / 2026-10-01 | Runtime under Codex / Claude Code / Hermes / OpenClaw / VLAs. Path: Intent → Body → Capability → Authority → Action. | D/P ("v1.3.0 Internal Alpha", 2026-09-17) | Every action returns an evidence-bearing `ExecutionReceipt`. Memory/evolution are async consumers of receipts. |
| [agenticros/agenticros](https://github.com/agenticros/agenticros) (read) | 161 / 2026-10-01 | One MCP server for Claude Code/Desktop/Dispatch, Codex, Hermes, Antigravity; an OpenClaw plugin; Gemini function-calling CLI.<br>**Tools:** `ros2_publish`, `ros2_camera_snapshot`, `ros2_depth_distance`, `ros2_find_object` (YOLOv8n), `ros2_navigate_to_place`, `ros2_estop`, `run_mission`, `follow_me_*`. | D/P | Transport-agnostic core (Zenoh, rosbridge, DDS, WebRTC). Local Ollama `qwen3-vl:8b-instruct` path. |
| [PlaiPin/rosclaw](https://github.com/PlaiPin/rosclaw) (read) | 625 / 2026-03-03 | OpenClaw agent → rosbridge, driven from Telegram/WhatsApp/Discord/Slack. | S/D. README: "undergoing a major re-architecture" | Capability auto-discovery node (`rosclaw_discovery`). |
| [automatika-robotics/embodied-agents](https://github.com/automatika-robotics/embodied-agents) + [emos](https://github.com/automatika-robotics/emos) | 69 / 80, both 2026-09/10 | ROS 2 component graph: VLM/LLM/STT/TTS components triggered by topics. Local models via Ollama. | P | Event-driven, self-reconfiguring components; cloud/local model switching at runtime. |
| [FlagOpen/RoboOS](https://github.com/FlagOpen/RoboOS) | 627 / 2025-12-18 | "Brain-Cerebellum": a RoboBrain planner plus Redis master/slaver skill execution; v2.0 adds MCP on the `stand-alone` branch. | S/R ([arXiv 2505.03673](https://arxiv.org/abs/2505.03673)) | Multi-robot. The v2 manual is still "coming soon". |
| [HorizonRobotics/HoloAgent](https://github.com/HorizonRobotics/HoloAgent) (read) | 395 / 2026-07-17 | AgentOS + 3D spatial memory + skills; FSR-VLN navigation; OpenClaw harness. | R, code partial (HoloAgent-1 "expected" 2026-12) | Cites "dimos-inspired Skill/Blueprint mechanisms". |
| [innate-inc/innate-os](https://github.com/innate-inc/innate-os) | 81 / 2026-10-01 | MARS home robot. Skills (`innate.Skill`) are "the unit of action"; agent + browser simulator. | P (product) | See teammate note `sources/innate-os-pr817.md` (GPT-6 Astra demo imitation, bounded `act` steps). |

### 2.2 ROS bridges and MCP servers (southbound adapters)

| Repo | ★ / push | What it does | Maturity | Notable |
|---|---|---|---|---|
| [nasa-jpl/rosa](https://github.com/nasa-jpl/rosa) (read) | 1647 / 2026-03-17 | LangChain `create_tool_calling_agent` + `AgentExecutor` (`max_iterations=100`). Tools in `src/rosa/tools/ros2.py` shell out to the `ros2` CLI: `ros2_topic_echo`, `ros2_service_call`, `ros2_param_set`, `ros2_doctor`, `roslog_list`. | P (v1.0.10; [arXiv 2410.06472](https://arxiv.org/abs/2410.06472)) | `RobotSystemPrompts(critical_instructions=…, constraints_and_guardrails=…)` and a tool `blacklist`. Mostly diagnostics/operations (NeBula-Spot demo). |
| [robotmcp/ros-mcp-server](https://github.com/robotmcp/ros-mcp-server) (read) | 1479 / 2026-10-01 | FastMCP over rosbridge. Tools: `get_topics`, `subscribe_once`, `publish_for_durations`, `send_action_goal`, `call_service`, `set_parameter`, image tools returning MCP `ImageContent`. Robot spec YAMLs (`robot_specifications/unitree_go2.yaml`). | P (v3.1.2, 2026-09-23; 18 contributors) | Works with Claude Code, Codex CLI, Gemini CLI, ChatGPT. Showcases: Claude diagnosing an industrial gripper from manuals; the "Wilson" mobile manipulator (Gemini + Nav2 + MoveIt). No timing or safety layer. |
| [RobotecAI/rai](https://github.com/RobotecAI/rai) (read) | 597 / 2026-09-10 | LangChain multi-agent framework. Tools: generic ROS 2 topics/services/actions, Nav2, manipulation `MoveToPointTool`/`GetObjectPositionsTool` (Grounded SAM 2). `rai_whoami` builds an embodiment description from docs/URDF. | P (2.12.4; [arXiv 2505.07532](https://arxiv.org/abs/2505.07532)) | `rai_bench`: a tool-calling-agent benchmark (subtask → validator → task scoring), O3DE manipulation, VLM bench. `rai_finetune` is listed as unfinished. |
| [wise-vision/ros2_mcp](https://github.com/wise-vision/ros2_mcp) (read) | 89 / 2026-09-30 | ROS 2 MCP server: multi-topic pub/sub, map-to-image, point-cloud BEV. | P | **Fail-closed read-only mode.** State-changing tools are not registered; a test fails if a new tool is not classified in `server/tool_safety.py`. |
| [lpigeon/ros-skill](https://github.com/lpigeon/ros-skill) | 27 / 2026-02-27 | Agent Skill (`SKILL.md` + `ros_cli.py` over rosbridge), e.g. `topics publish /cmd_vel … --duration 3`. | D | "Skill file + CLI" instead of MCP; the same pattern metal-arm-harness converged on. |
| [lpigeon/unitree-go2-mcp-server](https://github.com/lpigeon/unitree-go2-mcp-server) | 87 / 2026-06-10 | MCP → `unitree_ros2`. | D | |
| [omnilink-tech/omnisim](https://github.com/omnilink-tech/omnisim) (read) | 186 / 2026-09-30 | Newton-physics simulator with a first-party MCP server (37 tools) for Claude Code/Cursor. Cursor-paged event stream of contacts/limits/prints. | R/P (sim only) | Bitwise-deterministic replays on the CPU (`mujoco`) solver, so agent-debugged failures are reproducible. |
| [omni-mcp/isaac-sim-mcp](https://github.com/omni-mcp/isaac-sim-mcp) | 193 / 2025-04-25 | MCP extension for Isaac Sim scene/robot manipulation. | S | |
| [Auromix/ROS-LLM](https://github.com/Auromix/ROS-LLM) | 830 / 2023-07-10 | GPT-4/ChatGPT function interface to ROS motion/navigation. | S (lineage) | |
| [fujitatomoya/ros2ai](https://github.com/fujitatomoya/ros2ai) | 332 / 2026-08-04 | `ros2 ai` CLI extension (OpenAI API). | P (niche) | |
| [mgonzs13/llama_ros](https://github.com/mgonzs13/llama_ros) | 264 / 2026-09-29 | llama.cpp/llava.cpp GGUF inference as ROS 2 nodes. | P | Local-model path. |

### 2.3 LLM planner + learned-policy (VLA) harnesses: the most relevant family

| Repo | ★ / push | LLM role / action interface / perception / hardware | Evidence | Notable |
|---|---|---|---|---|
| [RLinf/RPent](https://github.com/RLinf/RPent) (read) | 1336 / 2026-10-01 | **Planner:** Claude Agent SDK (`ClaudeCodePlanner`: `model="sonnet"`, `allowed_tools="Bash Read Write Glob Grep"`, `timeout_s=600`, `max_budget_usd=10.0`) or Codex.<br>**Tools** (`robots/libero/tools.py`): `pi0_pick` (frozen π0.5 as a contact primitive), `move_to`, `move_pose`, `rotate_wrist`, `set_gripper`, `segment`, `back_project`, `view_env_state`.<br>**Other VLA backends:** RLDX-1, LingBot-VLA.<br>**Memory:** `GLOBAL_MEMORY.md` + task-specific files (HF `RLinf/RPent-memory`).<br>**Hardware:** Franka, dual Franka; SO-101 and YAM listed. | Harness VLA: +38.6 pp LIBERO-Pro, ~~+25.4 pp RoboCasa365~~ [corrected: +27.1 pp RoboCasa365 per v5 body (57.1% vs RLDX-1 30.0%; 250 rollouts, 5 held-out seeds per task)] over the strongest baselines; 58.4% RoboTwin C2R; dual-Franka demos ([arXiv 2607.08448](https://arxiv.org/abs/2607.08448)). README: Codex / GPT-6 Astra / low reasoning, **92.63% (741/800)** across all 8 LIBERO-PRO suites. A "non-reasoning mode" cuts execution time by about 40%. | The VLA is kept for local contact phases. Re-grounding, staging and transport are lifted to the planner plus analytic primitives. A "flywheel" exports episodes for training. |
| [air-embodied-brain/Zetta-Embodiment](https://github.com/air-embodied-brain/Zetta-Embodiment) (read) | 1269 / 2026-09-19 | Frozen base VLA. The LLM (mostly `gpt-5.6-sol`/`gpt-5.6-terra` in configs) evolves **code-based runtime critics and recovery skills**. Roles: Critic proposes → `Role1` accepts → recovery actor executes bounded programs. "Only the environment actor may write simulator actions." | LIBERO-Pro 90.8%, RoboCasa 93.6%, "11.1x inference speedup" ([arXiv 2608.16590](https://arxiv.org/abs/2608.16590)). Sim only. | Evolution protocol: 50 dev rollouts ("never use seeds 1..20") → failure cluster → causal diagnosis → candidate → shadow replay → paired same-seed gate → held-out seeds 1..20. `THIRD_PARTY_NOTICES.md`: "RPent-derived agent framework code"; no LICENSE file. |
| [google-deepmind/gemini-robotics-sdk](https://github.com/google-deepmind/gemini-robotics-sdk) "Safari SDK" (read) | 613 / 2026-09-18 | `safari_sdk/agent/framework/`: a Gemini Live orchestrator (flag `agent.model_name` default `"gemini-live-2.5-flash-preview"`).<br>**Tool `run_instruction_until_done`** (`behavior=NON_BLOCKING`) sends an "atomic" language instruction ("put the red dice in the green tray") to the Gemini Robotics VLA.<br>**Success detection** by `gemini-robotics-er-1.5-preview`, using start frames + timestamped history + current frames.<br>**Embodiment:** ALOHA. | Most features need Trusted-Tester access (On-Device model from SDK v2.4.1). | **The canonical 3-model split:** a live orchestrator, a VLA executor and an ER success detector polled every `sd_async_sd_interval_s`. Also `run_instruction_for_duration` and `scene_description`. |
| [google-gemini/robotics-samples](https://github.com/google-gemini/robotics-samples) | 118 / 2026-10-01 | `Getting Started/gemini_robotics_er.ipynb` (moved from the cookbook). `live-api/agent`: Live API server with embodiments `spot`, `tinybot` and **`human`**. | Samples | A human operator as a pluggable "embodiment", useful for Wizard-of-Oz testing of the harness. |
| [strands-labs/robots](https://github.com/strands-labs/robots) (read) | 171 / 2026-10-01 | `Agent(tools=[Robot("so100")])("pick up the red cube")`. One `Policy` ABC: LeRobot ACT/Pi0/SmolVLA/Diffusion/GR00T N1.7, Cosmos 3, MolmoAct2, cuRobo, MoveIt2. MuJoCo by default; `mode="real"` is an explicit opt-in. | P (v0.5.2, 2026-09-17); no success-rate evals in the README | `run_policy` shares the RTC "control-frequency / observed-delay contract" with `start_task`. Exclusive command bus. Human-in-the-loop **motion grants** (`_motion_grants.py`). Zenoh mesh with E-STOP broadcast. |
| [Grigorij-Dudnik/RoboCrew](https://github.com/Grigorij-Dudnik/RoboCrew) (read) | 139 / 2026-09-30 | LLM agent (default `google_genai:gemini-3-flash-preview`). Tools: wheel moves, `look_around`, **`create_vla_single_arm_manipulation`** (any LeRobot policy as a tool). Camera frames get an angle-grid overlay (`basic_augmentation`); LiDAR top-down map. Hardware: XLeRobot/LeKiwi, Earth Rover, Go2, Tello. | D | The cheapest hardware path for "LLM + LeRobot VLA as tool". |
| [ZJU4EmbodiedAI/EmbodiedSkills](https://github.com/ZJU4EmbodiedAI/EmbodiedSkills) | 234 / 2026-09-02 | VLM AgentLoop with 6 stages: observation, planning, preflight, bounded execution, verification, recovery. Persistent OpenPI π0.5 backend; RoboTwin 2.0/RMBench/LIBERO adapters. | R | **Trains a Qwen3-VL LoRA on full AgentLoop trajectories**, i.e. distils the planner. |
| [EIT-HAI/Thea](https://github.com/EIT-HAI/Thea) (read) | 130 / 2026-09-17 | Coding-agent-style loop over robot tools; LIBERO / RoboTwin; Lark chat channel. | R ([arXiv 2608.11246](https://arxiv.org/abs/2608.11246)) | "Scene Graph as Context" and "Evaluation as Exit Codes": termination, success judgement and failure diagnosis per tool call. |
| [OpenMOSS/OpenETA](https://github.com/OpenMOSS/OpenETA) (read) | 193 / 2026-09-14 | Planner picks one tool call at a time. 35 verified tool contracts (Stage 2). ~~The Codex plugin exposes only `observe`, `mark_point`, `move_to`.~~ [corrected: the paper abstract describes a 3-tool plugin. But the evaluated Codex release (branch `openeta-for-codex`, "OpenETA-Light"; the README's `openeta-light` link returns 404) exposes **six** MCP tools: `observe`, `mark_point`, `move_to`, `report_issue`, **`check_task`** (which queries the native LIBERO success checker) and `finish_episode`.] | README: Codex + `gpt-5.6-sol` (medium) **70.8% Pass@1 / 90.0% Pass@5 on all 130 LIBERO tasks** ([arXiv 2608.03924](https://arxiv.org/abs/2608.03924)) | "Every world-changing action creates a fresh-observation obligation." Tools are host-owned; Skills are reviewable text. |
| [RoboClaw-Robotics/RoboClaw](https://github.com/RoboClaw-Robotics/RoboClaw) | 164 / 2026-04-10 | VLM high-level controller across collection, training and deployment; Agibot G01. | README: +25% success over baseline pipelines ([arXiv 2603.11558](https://arxiv.org/abs/2603.11558)) | "Entangled Action Pairs" for self-resetting data collection. |
| [amap-cvlab/ABot-Claw](https://github.com/amap-cvlab/ABot-Claw) | 213 / 2026-04-14 | OpenClaw-based; VLN/VLA/WAM unified through a "VLAC" (Vision-Language-Action-Critic) loop. | R | |
| [rokbenko/quackd](https://github.com/rokbenko/quackd) | 246 / 2026-09-30 | LLM skill selector with an executor contract; `manipulate(instruction)` hands off to a LeRobot policy for 10 s. | D. Teammate note: no learned policy has driven the real arm. | See `sources/quackd.md`. |

### 2.4 Frontier VLM as the fine-grained policy (direct control)

| Repo | ★ / push | Interface | Evidence | Notable |
|---|---|---|---|---|
| [showlab/Show-Harness](https://github.com/showlab/Show-Harness) (read) | 497 / 2026-09-30 | **Discrete semantic units** (`MV_LEFT/RIGHT/FWD/BACK/UP/DOWN`, `GRASP`, `RELEASE`, `DONE`, rotation) plus per-robot deterministic interpreters (`interpreters/`).<br>**Duck-typed arm API:** `get_ee_pose`, `get_gripper_position`, `control_gripper`, `update_desired_ee_pose`.<br>**Hardware:** Franka (Polymetis), AgileX Piper (50 Hz joint streaming), ManiSkill, Isaac Lab. | Zero-shot top tier 86-96%: Gemini-3.1 Pro, GPT-5.6-sol, **Opus 5**. Fine-tuned Qwen3.5-2B: cross-task 86% vs best baseline 57%. Naming ablation: 20/20 with names + explained convention vs **1/20** with neither. Plugin ablations: −38 pp without multi-view guidance, −36 without subtask planning. Distilled models run at 12-33 Hz (39 ms/decision at 2B) ([project page](https://showlab.github.io/Show-Harness/), [arXiv 2609.10522](https://arxiv.org/abs/2609.10522)). | **GUMI:** humans *or computer-use agents* "play" the robot in a browser (`prompts/web_operator.txt`), producing training pairs in the same vocabulary. LoRAs on HF `showlab/Show-Harness-VLMs` (Qwen3.5 0.8/2/4/9B, Gemma4-E4B). |
| [Mosi-AI/RoboICL](https://github.com/Mosi-AI/RoboICL) (read) | 153 / 2026-10-01 | GPT-6 Astra sees a triptych (left wrist / head / right wrist) plus proprioception and emits a bounded sequence of dual-arm Cartesian actions. Context grammar `observation -> action -> receipt -> observation`. | 30 RoboDojo tasks: Overall 50.64 vs 33.68 for the strongest baseline. Real robot: 14.45 → 63.33 → 78.89 mean progress at 0/1/3 shots. Jev-gated action reuse cuts Astra calls by 33-48% ([arXiv 2609.34261](https://arxiv.org/abs/2609.34261)). | "Bounded anchored memory": fixed early anchors plus the latest interaction. |
| [aditya-ramabadran/drivingbench_harness_v1](https://github.com/aditya-ramabadran/drivingbench_harness_v1) (read) | 51 / 2026-09-26 | Real Toyota via comma/openpilot. MCP server `drivingbench_sandbox` with 3 tools: `observe`, `set_motion(direction, steering_percent, speed_mps, duration_s, reason)`, `stop_now`. | Safety-driver-supervised; parking-lot speeds | **Motion lease:** `DURATION_RANGE_S=(5,60)`, `SPEED_RANGE_MPS=(0.5,3.5)`, `NATIVE_WATCHDOG_S=2.0`. Instructions: "Motion continues while you think; expiry starts braking"; "after an uncertain result, observe rather than blindly retrying". |
| [makermods-robotics/metal-arm-harness](https://github.com/makermods-robotics/metal-arm-harness) | 15 / 2026-09 | Coding agent (Claude Code/Codex) issues `op tip X Y Z PITCH` / `op nudge` CLI calls. | Anecdotal | Teammate note: the first direct Messages-API loop was removed after 8 h in favour of a CLI + skill file. |
| [hesd10/astra-robot-sim2real](https://github.com/hesd10/astra-robot-sim2real) | 7 / 2026-09-24 | GPT-6 Astra pressing an elevator button on XLeRobot, sim and real. | 30 fixed-start sim trials: geometry/camera info cuts mean time by 57.4%; synchronized experience by 68.6%. 12 real trials. | Body knowledge + recorded experience as context. |
| [openroboto-ai/jev-robot-control](https://github.com/openroboto-ai/jev-robot-control) | 55 / 2026-09-19 | xArm7 in MuJoCo; per cycle: intent + X/Y/Z direction + gripper. | One seed each: Jev 1.13 placed in 113 cycles for $0.018825; GPT-6 Astra (low) placed in 106 cycles for $5.933624; GPT-4.1 mini hit the 160-cycle limit | Offline verifier and replay. |
| [FBddcz/embodied-jev](https://github.com/FBddcz/embodied-jev) | 251 / 2026-09-22 | Browser MuJoCo/Franka workbench: Jev, Claude native API, OpenAI-compatible APIs, local MiniCPM5-2B; skill-level or stepwise XYZ/gripper. | Demos | Candidate actions are previewed in a sim copy to intercept collisions and grasp loss. |
| [RobotKitAI/piper-astra-jev](https://github.com/RobotKitAI/piper-astra-jev) | 11 / 2026-09-19 | Real PiPER; GPT-6 Astra or Jev selecting skills over Grounding-DINO/SAM3 + depth state. | n=1 per run | See `sources/piper-astra-jev.md`. |

### 2.5 Coding-agent self-improvement (offline LLM, online code)

| Repo | ★ / push | Summary |
|---|---|---|
| [NVlabs/ENPIRE](https://github.com/NVlabs/ENPIRE) (read) | 244 / 2026-09-09 | A coding agent runs the loop reset → execute → verify → record → refine on a physical station. CaP skill scripts or actor/learner RL. The README prompt requires calibration preflight and "explicit motion authorization" ([arXiv 2606.19980](https://arxiv.org/abs/2606.19980)). |
| [NVlabs/ASPIRE](https://github.com/NVlabs/ASPIRE) (read) | 228 / 2026-09-01 | "Training is skill refinement"; the "model" is a repo of skills. Sim workflow packaged for Claude Code with Opus 4.6 1M; real-robot runs used Codex ([arXiv 2607.00272](https://arxiv.org/abs/2607.00272)). |
| [capgym/cap-x](https://github.com/capgym/cap-x) | 833 / 2026-05-28 | Coding-agent manipulation benchmark (ICML 2026). See `llm-planner-codegen-lineage.md`. |
| [EmbodiedSWE/EmbodiedSWE](https://github.com/EmbodiedSWE/EmbodiedSWE) | 111 / 2026-09-30 | Sim data generation with coding agents. See `sources/embodiedswe.md`. |
| [robocurve/inspect-robots](https://github.com/robocurve/inspect-robots) | 631 / 2026-09-30 | "Run any LLM/VLA on any arm/humanoid against any real/sim benchmark" (eval harness). See `sources/robocurve-gpt6-astra.md`. |

### 2.6 Desktop/social robots and the LeRobot/SO-101 ecosystem

| Repo | ★ / push | Summary |
|---|---|---|
| [pollen-robotics/reachy_mini_conversation_app](https://github.com/pollen-robotics/reachy_mini_conversation_app) (read) | 319 / 2026-10-01 | Realtime voice through the Hugging Face realtime backend. LLM tools: `dance`, `play_emotion`, `camera`, `move_head`, `head_tracking`, `remember`/`forget`, `robot_status`. **MCP "Tool Spaces"** (HF Spaces as tools). A layered motion queue blends speech-reactive wobble. SDK: [pollen-robotics/reachy_mini](https://github.com/pollen-robotics/reachy_mini) (1531★). |
| [jackccrawford/reachy-mini-mcp](https://github.com/jackccrawford/reachy-mini-mcp) | 29 / 2026-07-29 | 7 MCP tools (`speak`, `listen`, `snap`, `show`, `look`, `rest`, `discover`). Similar: agentculture/reachy-mini-mcp (32★), tomrikert/clawbody (39★, OpenClaw), NVIDIA-AI-IOT/reachy-mini-jetson-assistant (33★, fully local). |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | 27,897 / 2026-10-01 (v0.6.1, 2026-08-03) | No LLM-agent runtime. But the **v3.1 dataset format** carries a tool catalog in `meta/info.json["tools"]` (OpenAI function schemas; default `say`) and policies can emit tool-call atoms (`docs/source/tools.mdx`). **`lerobot-annotate`** uses a VLM (Qwen-VL via an OpenAI-compatible vLLM server) to write plans, subtasks, memory, interjections and VQA into language columns: the data path for training a head conditioned on LLM subtasks. |
| [phospho-app/phosphobot](https://github.com/phospho-app/phosphobot) | 394 / 2026-09-15 (main last commit 2025-12-22) | Teleop/record/train (ACT, SmolVLA, π0.5, GR00T) for SO-100/101, Piper, Go2. Policy layer only, no LLM. |
| [itsbharatj/so101_ros_mcp](https://github.com/itsbharatj/so101_ros_mcp) | 5 / 2026-07-23 | SO-101 + ROS 2 + MoveIt2 + "semantic MCP" on Jetson. **Fake mode by default**; real execution requires `allow_real_execution=true`. |
| [arcadeai-labs/safe-hands](https://github.com/arcadeai-labs/safe-hands) | 1 / 2026-09-22 | SO-101 over MCP governed by **Cedar policies** ("an explicit `forbid` always overrides a `permit`"). Notes that typical robot-MCP servers "authenticate nothing and authorize nothing". |
| [noah-wardlow/lerobot-mcp](https://github.com/noah-wardlow/lerobot-mcp) | 1 / 2026-09-25 | MCP over LeRobot CLI/datasets/policy pre- and post-processors. A workflow tool, not control. |

### 2.7 Lineage (2023-2024), kept for orientation

- [microsoft/PromptCraft-Robotics](https://github.com/microsoft/PromptCraft-Robotics) (2118★, last push 2024-01) and [microsoft/ChatGPT-Robot-Manipulation-Prompts](https://github.com/microsoft/ChatGPT-Robot-Manipulation-Prompts) (385★, 2023-11): the "ChatGPT for Robotics" API-prompting pattern.
- [huangwl18/VoxPoser](https://github.com/huangwl18/VoxPoser) (838★), [huangwl18/ReKep](https://github.com/huangwl18/ReKep) (986★), [moka-manipulation/moka](https://github.com/moka-manipulation/moka) (102★): constraint/keypoint generation. See the planner-codegen note.
- [NVIDIA-AI-IOT/remembr](https://github.com/NVIDIA-AI-IOT/remembr) (359★): LLM+VLM long-horizon spatio-temporal memory.
- Indices:
  - [GT-RIPL/Awesome-LLM-Robotics](https://github.com/GT-RIPL/Awesome-LLM-Robotics) (4476★)
  - [visitworld123/Awesome-Robot-Use-Agent](https://github.com/visitworld123/Awesome-Robot-Use-Agent)
  - [kairunwen/Awesome-Robot-Use-Agent](https://github.com/kairunwen/Awesome-Robot-Use-Agent) (139★; has an "Open Source" systems table and 61 social demos)

---

## 3. Recurring design patterns, with code pointers

1. **Lease-based motion commands.** The LLM never issues an open-ended velocity.
   - DrivingBench: `set_motion(..., duration_s)` with a 2 s native watchdog; "expiry starts braking".
   - dimOS Jev nav: cmd_vel zeroes when no decision arrives for `max(DEADMAN_MIN_S, DEADMAN_PERIODS/max_hz)`.
   - ros-mcp-server: `publish_for_durations`.
   - ros-skill: `--duration`.

   This is the minimum viable safety primitive when model latency is seconds.
2. **Non-blocking tool + independent success detector.** Safari's `run_instruction_until_done` is declared `NON_BLOCKING`. A separate ER model compares start, history and current frames. Thea's "Evaluation as Exit Codes" and PhyAgentOS's verifier/evidence split generalise this. **Implication for an Opus harness:** do not let the planner self-report success. Run a separate checker, either a cheap VLM or a learned success classifier.
3. **A frozen learned primitive as a retryable tool, with re-staging lifted to the LLM.** RPent `pi0_pick`, RoboCrew `create_vla_single_arm_manipulation`, Strands `run_policy`, quackd `manipulate`, EmbodiedSkills' persistent π0.5 backend. RPent's ablations attribute the gains to the planner fixing *where* the VLA starts (re-grounding, staging) rather than to the VLA itself ([arXiv 2607.08448](https://arxiv.org/abs/2607.08448)).
4. **Discrete, named action vocabulary with explained conventions.** Show-Harness drops from 20/20 to 1/20 when unit names and the image-direction convention are both removed. Interpreters own frames, gains, the Z floor and IK. This is the strongest controlled evidence in the corpus that **interface design dominates model choice** once the model is frontier-class.
5. **Capability/resource locking and single command bus.** dimOS `CapabilityRegistry`; Strands "Exclusive: the arm has a single command bus"; PhyAgentOS session-centred scheduling.
6. **Fresh-observation obligation / receipt grammar.**
   - OpenETA forces an observation after every world-changing action.
   - RoboICL formats history as `observation -> action -> receipt -> observation`.
   - ROSClaw returns an `ExecutionReceipt`.
7. **Governed self-improvement.** Zetta's held-out-seed gate and shadow replay, PhyAgentOS's "guarded Skill promotion", and OpenETA's "candidate → review → canary → holdout" are the antidote to agents overfitting their own memory files.
8. **Human-in-the-loop authorization as code.**
   - Strands motion grants.
   - Read-only modes: ros2_mcp, and so101_ros_mcp's fake mode.
   - safe-hands Cedar policies.
   - ENPIRE's "explicit motion authorization".
9. **Distillation path built in.** Show-Harness (GUMI + LoRA), EmbodiedSkills (Qwen3-VL LoRA on AgentLoop traces), RPent "flywheel" export, LeRobot `lerobot-annotate`. A harness should log every (observation, tool call, receipt) in a trainable schema from day one.

---

## 4. Ranking: most useful references for a new "Claude/Opus API backbone + light action head" harness

| Rank | Repo | Why |
|---|---|---|
| 1 | **RLinf/RPent** | The same architecture as the target: the Claude Agent SDK planner (`rpent/planner/claude_code.py`) drives a frozen VLA as a primitive plus analytic primitives and file memory. It has the largest controlled gains (+38.6 pp LIBERO-Pro), real dual-Franka support, budget/timeout knobs, and an episode flywheel for training the head. Read `robots/libero/tools.py` for the tool surface. |
| 2 | **showlab/Show-Harness** | The best evidence on the interface. Opus 5 is in the zero-shot top tier. The same vocabulary distils into 0.8-9B VLMs at 12-33 Hz, a concrete "light head" path. Plugin and naming ablations tell you what to build first (multi-view, subtask planning, proprioception). GUMI lets a computer-use agent generate data. |
| 3 | **Safari SDK agent framework** (google-deepmind/gemini-robotics-sdk) | Google's reference for the orchestrator / VLA / success-detector split: non-blocking instruction tools, timestamped frame history for success detection, event bus, eval binary (`safari_sdk/agent/framework/eval/`). Port the pattern; the models need Trusted-Tester access. |
| 4 | **dimensionalOS/dimos** | The most complete open runtime: modules/blueprints, MCP server+client, skills with capability locks, spatial memory, many bodies. Its `typesafe` agent shows how a 2 Hz typed decider sits under an LLM with deadman semantics. Use it as a runtime substrate or crib its skill/capability layer. |
| 5 | **strands-labs/robots** | The cleanest "agent tool wraps any LeRobot/GR00T policy" implementation, with sim/real parity, RTC timing contract, HITL motion grants and a mesh E-STOP. The best starting point if the head is a LeRobot policy on SO-101-class hardware. |
| 6 | **air-embodied-brain/Zetta-Embodiment** | The template for *evolving* the harness without touching the head: code critics plus recovery under a preregistered, held-out-seed promotion protocol. Its timescale separation (action-frequency governance vs. rollout-level LLM reasoning) matches the latency constraints of an API backbone. |
| 7 | **OpenETA / Thea / PhyAgentOS** | Contracts for verification and evidence: fresh-observation obligation; evaluation as exit codes and scene graph as context; execution/evidence/verdict separation. ~~OpenETA's 3-tool Codex plugin (`observe`, `mark_point`, `move_to`) at 70.8% Pass@1 on 130 LIBERO tasks is a strong minimal baseline.~~ [corrected: the 70.8% Pass@1 / 90.0% Pass@5 result (`gpt-5.6-sol`, medium effort) used the **six-tool** OpenETA-Light plugin. One of those tools, `check_task`, lets the agent query the simulator's ground-truth success checker during the episode. It is a strong minimal baseline, but its numbers are not comparable to harnesses that lack a success oracle.] |
| 8 | **Mosi-AI/RoboICL** | Context engineering for in-context demonstrations: receipt grammar, anchored memory, and a Jev gate that cuts frontier calls by 33-48%. Directly portable to Opus prompts. |
| 9 | **DrivingBench harness** | The smallest safe tool surface for a slow LLM controlling a real, dangerous machine. Copy its lease/expiry semantics and instruction text. |
| 10 | **robotmcp/ros-mcp-server, nasa-jpl/rosa, RobotecAI/rai** | Southbound integration and diagnostics for ROS robots: discovery of custom types, robot spec YAMLs, the `rai_bench` tool-calling benchmark. Wrap them behind your own governed tools; do not expose them raw to the planner. |

Also read: the teammate notes on quackd (executor contract, feasibility verdicts), innate-os PR #817 (bounded `act` steps with host re-validation) and metal-arm-harness (low-level safety floor work).

---

## 5. Promising vs. unpromising implementation choices

**Promising (with evidence):**
- **Frozen VLA as a retryable contact primitive.** The LLM does re-grounding, staging, transport and verification (RPent: ~~+38.6/+25.4 pp~~ [corrected: +38.6/+27.1 pp per arXiv 2607.08448v5]; Zetta: 90.8%/93.6%; both sim).
- **Discrete semantic action units with deterministic interpreters,** plus explicit image-direction conventions (Show-Harness ablation 20/20 vs 1/20). Multi-view (wrist + front) is the single most valuable plugin (−38 pp when removed).
- ~~**Distilling the frontier-driven interface into a 2B-class VLM**~~ [corrected: **fine-tuning a 2B-class VLM on demonstrations recorded in the same action vocabulary**. The HF data card lists 164 real episodes (Franka 101 + AgileX 63; 7,933 samples) and 230 sim episodes (13,753 samples), with 9 units of 2 cm each. It does not state that frontier-VLM rollouts were the teacher, so "distillation" is an analogy, not a documented frontier-to-small pipeline] (Show-Harness: 86% cross-task fine-tuned vs 89% zero-shot, at 39 ms/decision).
- **An independent success detector** (Safari ER model; Thea exit codes; PhyAgentOS verifier).
- **Leases and deadman timers** (DrivingBench, dimOS).
- **Typed fast deciders for 2-10 Hz loops.** Jev in dimOS nav; RoboICL gating. Evidence is still mostly single-seed (jev-robot-control: one seed per controller).

**Unpromising or risky:**
- **Raw ROS-topic MCP as the control path.** ros-mcp-server and similar are excellent for diagnosis (the industrial-gripper demo). For control they push timing, units, frames and safety into the prompt. safe-hands' critique ("authenticates nothing and authorizes nothing") applies to most robot-MCP servers in the sweep.
- **1 Hz natural-language fusion loops for manipulation** (OM1's design point is social/quadruped behaviour, [arXiv 2412.18588](https://arxiv.org/abs/2412.18588)).
- **Messaging-app-first architectures (OpenClaw/Telegram) as the core.** PlaiPin/rosclaw (625★) has stalled in "re-architecture" since 2026-03. AgenticROS/ROSClaw add governance but publish no success-rate evaluations.
- **Big-OS claims without released code or evals.** HoloAgent-1 is "expected" 2026-12; RoboOS 2.0's manual is "coming soon"; RoboOS has had no push since 2025-12.
- **n=1 demos.** Most real-hardware LLM demos in this sweep (Jev/Astra workbenches, Reachy MCPs, SO-101 MCPs) report single episodes. Treat them as interface ideas, not evidence.

---

## 6. Gaps, caveats, UNVERIFIED

- **No open-source framework found built on the OpenAI Agents SDK specifically for robots.** `gh search repos "openai agents robot"` and `gh search code "from agents import Agent robot arm"` returned nothing relevant. The AWS Strands-based `strands-labs/robots` fills that niche. **No Anthropic-published robotics repo** was found under the `anthropics` org.
- ~~**LeRobot/SO-101 + LLM MCP repos are all tiny** (≤5★, e.g. so101_ros_mcp, lerobot_lekiwi_mcp, safe-hands).~~ [corrected: the largest is [IliaLarchenko/robot_MCP](https://github.com/IliaLarchenko/robot_MCP) at 85★. It provides an SO-ARM100/101 MCP server plus a CLI agent for Claude/Gemini/GPT; its last push was 2025-08-12, so it is stale and broken by later LeRobot API changes, per its README. All other SO-101 MCP repos found are ≤5★.] I found no maintained, widely used "LeRobot MCP" for control.
- **Stars are a weak quality signal in 2026.** Dozens of "GPT-6 Astra" repos with 100-1000★ are prompt galleries or jailbreaks. Zetta (1269★) and PhyAgentOS (2639★) gained stars within weeks of creation.
- **Numbers are author-reported and unreplicated.** RPent's 741/800, Zetta's 90.8%/93.6%, OpenETA's 70.8% and Show-Harness's 86-96% all come from READMEs, arXiv abstracts or the project page. I did not rerun them. They use different LIBERO-Pro protocols; see `llm-planner-codegen-lineage.md` §4.3 before comparing.
- **Show-Harness GPU-hours:** ~~the page says only "a few"~~ [corrected: the arXiv abstract says "a few GPU-hours"; the repo README (line 79) is more specific: "less than a few H200 GPU-hours"]; exact training cost is UNVERIFIED.
- **Web search budget was exhausted early in this session.** Discovery relied on `gh search` plus the two Awesome-Robot-Use-Agent indices. Projects without GitHub presence or with poor naming (e.g. Waddle Labs, Manda) may be under-covered here. Waddle's blog was not opened (**UNVERIFIED**).
- **Licensing:** Zetta-Embodiment has no LICENSE file (pyproject points to a missing one). ABot-Claw, RoboClaw, EmbodiedSkills and EMERGE-Policy show no detected license. Check before reuse.

---

## Sources

GitHub repos (metadata via GitHub API on 2026-10-01; README/code read from shallow clones in `research/repos/`):
- https://github.com/dimensionalOS/dimos (files: `dimos/agents/mcp/mcp_client.py`, `dimos/agents/capabilities.py`, `dimos/agents/typesafe/constants.py`, `dimos/agents/system_prompt.py`, `AGENTS.md`)
- https://github.com/OpenMind/OM1 (`internal/fuser/fuser.go`, `config/unitree_go2_autonomy.json5`, `config/conversation.json5`)
- https://github.com/PhyAgentOS/PhyAgentOS-core
- https://github.com/ros-claw/rosclaw
- https://github.com/agenticros/agenticros
- https://github.com/PlaiPin/rosclaw
- https://github.com/automatika-robotics/embodied-agents ; https://github.com/automatika-robotics/emos
- https://github.com/FlagOpen/RoboOS
- https://github.com/HorizonRobotics/HoloAgent
- https://github.com/innate-inc/innate-os
- https://github.com/nasa-jpl/rosa (`src/rosa/rosa.py`, `src/rosa/tools/ros2.py`, `src/rosa/prompts.py`)
- https://github.com/robotmcp/ros-mcp-server (`ros_mcp/tools/*.py`, `robot_specifications/`)
- https://github.com/RobotecAI/rai (`src/rai_core/rai/tools/ros2/`, `src/rai_bench/`)
- https://github.com/wise-vision/ros2_mcp
- https://github.com/lpigeon/ros-skill ; https://github.com/lpigeon/unitree-go2-mcp-server
- https://github.com/omnilink-tech/omnisim ; https://github.com/omni-mcp/isaac-sim-mcp
- https://github.com/Auromix/ROS-LLM ; https://github.com/fujitatomoya/ros2ai ; https://github.com/mgonzs13/llama_ros
- https://github.com/RLinf/RPent (`rpent/planner/claude_code.py`, `robots/libero/tools.py`)
- https://github.com/air-embodied-brain/Zetta-Embodiment
- https://github.com/google-deepmind/gemini-robotics-sdk (`safari_sdk/agent/framework/flags.py`, `tools/run_instruction_until_done.py`, `tools/success_detection.py`)
- https://github.com/google-gemini/robotics-samples ; https://github.com/google-gemini/cookbook/blob/main/quickstarts/gemini-robotics-er.ipynb
- https://github.com/strands-labs/robots (`strands_robots/hardware_robot.py`, `strands_robots/_motion_grants.py`)
- https://github.com/Grigorij-Dudnik/RoboCrew
- https://github.com/ZJU4EmbodiedAI/EmbodiedSkills
- https://github.com/EIT-HAI/Thea
- https://github.com/OpenMOSS/OpenETA
- https://github.com/RoboClaw-Robotics/RoboClaw ; https://github.com/amap-cvlab/ABot-Claw
- https://github.com/rokbenko/quackd
- https://github.com/showlab/Show-Harness (`prompts/controller.txt`, `prompts/web_operator.txt`, `interpreters/README.md`, `gumi/README.md`)
- https://github.com/Mosi-AI/RoboICL
- https://github.com/aditya-ramabadran/drivingbench_harness_v1 (`shared/contracts.py`, `mcp/tools.py`)
- https://github.com/makermods-robotics/metal-arm-harness
- https://github.com/hesd10/astra-robot-sim2real
- https://github.com/openroboto-ai/jev-robot-control ; https://github.com/FBddcz/embodied-jev
- https://github.com/RobotKitAI/piper-astra-jev
- https://github.com/NVlabs/ENPIRE ; https://github.com/NVlabs/ASPIRE
- https://github.com/capgym/cap-x ; https://github.com/EmbodiedSWE/EmbodiedSWE ; https://github.com/robocurve/inspect-robots
- https://github.com/pollen-robotics/reachy_mini_conversation_app ; https://github.com/pollen-robotics/reachy_mini ; https://github.com/jackccrawford/reachy-mini-mcp
- https://github.com/huggingface/lerobot (`docs/source/tools.mdx`, `docs/source/annotation_pipeline.mdx`)
- https://github.com/phospho-app/phosphobot ; https://github.com/itsbharatj/so101_ros_mcp ; https://github.com/arcadeai-labs/safe-hands ; https://github.com/noah-wardlow/lerobot-mcp
- https://github.com/microsoft/PromptCraft-Robotics ; https://github.com/microsoft/ChatGPT-Robot-Manipulation-Prompts ; https://github.com/huangwl18/VoxPoser ; https://github.com/huangwl18/ReKep ; https://github.com/moka-manipulation/moka ; https://github.com/NVIDIA-AI-IOT/remembr
- https://github.com/GT-RIPL/Awesome-LLM-Robotics ; https://github.com/visitworld123/Awesome-Robot-Use-Agent ; https://github.com/kairunwen/Awesome-Robot-Use-Agent

Papers (abstracts fetched via the arXiv export API) and project pages:
- https://arxiv.org/abs/2607.08448 (Harness VLA / RPent)
- https://arxiv.org/abs/2608.16590 (Zetta)
- https://arxiv.org/abs/2609.10522 (Show-Harness) and https://showlab.github.io/Show-Harness/
- https://arxiv.org/abs/2609.34261 (RoboICL)
- https://arxiv.org/abs/2608.03924 (ETA / OpenETA)
- https://arxiv.org/abs/2608.11246 (Thea)
- https://arxiv.org/abs/2607.16636 (PhyAgentOS)
- https://arxiv.org/abs/2410.06472 (ROSA)
- https://arxiv.org/abs/2505.07532 (RAI)
- https://arxiv.org/abs/2412.18588 (OM1 / "A Paragraph is All It Takes")
- Cited from READMEs only, not opened: arXiv 2606.19980 (ENPIRE), 2607.00272 (ASPIRE), 2603.11558 (RoboClaw), 2505.03673 (RoboOS)

---

## Verification (fact-check pass)

Adversarial re-check done 2026-10-02 against primary sources: the GitHub REST API (`gh api`), the local clones in `research/repos/`, arXiv abstract and HTML pages, the Show-Harness project page and the Hugging Face cards. A claim counts as confirmed only if I saw it in one of these sources. Inline fixes in the body are marked `~~old~~ [corrected: ...]`.

### Confirmed (primary source seen)

1. **Star counts and push dates.** All 60+ repo metadata values match `gh api repos/<r>` on 2026-10-02 to within ±2★, e.g. dimos 4601, OM1 2939, RPent 1336, Zetta 1269, Show-Harness 497 and lerobot 27,897. Drift: PhyAgentOS 2641, innate-os 82, RoboICL 154. Creation dates support "appeared Jul-Sep 2026": RPent 2026-07-07, OpenETA 2026-07-25, Zetta 2026-08-18, Show-Harness 2026-09-07, RoboICL 2026-09-19.
2. **RPent code.** `rpent/planner/claude_code.py:66-78`: `class ClaudeCodePlanner` with defaults `model="sonnet"`, `allowed_tools="Bash Read Write Glob Grep"`, `timeout_s=600`, `max_budget_usd=10.0`, and `import claude_agent_sdk` at line 136.
   - `robots/libero/tools.py` defines `pi0_pick` (line 195, "Closed-loop Pi0.5 pick"), `move_to`, `rotate_wrist`, `move_pose`, `set_gripper`, `segment`, `view_env_state` and `back_project`.
   - README line 34: "Codex / GPT-6 Astra / low / reasoning: **92.63% Overall (741/800)** across all eight LIBERO-PRO suites".
   - README news: "non-reasoning mode, which reduces average execution time by ~40%", plus RLDX-1 and LingBot-VLA support. SO-101 and YAM are listed without a ✅.
3. **Harness VLA paper** ([arXiv 2607.08448](https://arxiv.org/abs/2607.08448), v5 of 24 Sep 2026). Confirmed: +38.6 pp LIBERO-Pro, 58.4% RoboTwin C2R, dual-Franka real demos, and frozen VLA as a "retryable contact-rich primitive". RoboCasa365 is corrected below.
4. **Zetta** ([arXiv 2608.16590](https://arxiv.org/abs/2608.16590)).
   - Abstract: "90.8% and 93.6%, with an 11.1x inference speedup".
   - README lines 30-36 and 46-47: "50 development rollouts (never use seeds 1..20)", Shadow Replay, paired Same-seed Gate, "Held-out seeds 1..20", and "only the environment actor may write simulator actions".
   - `THIRD_PARTY_NOTICES.md:3`: "RPent-derived agent framework code" (70 former `rpent/` files migrated).
   - No LICENSE file, although `pyproject.toml:12` points at one.
   - Model names in the repo: `gpt-5.6-sol` ×62 and `gpt-5.6-terra` ×28; `gpt-5.6-luna` appears twice.
5. **Show-Harness** (project page and [arXiv 2609.10522](https://arxiv.org/abs/2609.10522)).
   - "Five frontier VLMs under one harness split into two tiers: Gemini-3.1 Pro, GPT-5.6-sol and Opus 5 at 86–96%, GPT-5.6-luna and Gemini-3.6-flash at 72–78%."
   - Naming ablation: 20/20 (A) vs 1/20 (D).
   - Removing Multi-View Guidance costs 38 points; Subtask Planning 36, Proprioception 28, Failure Recovery 24, Action History 20.
   - Cross-task: 89% / 86% ZS/FT, best baseline 57%. FT is Qwen3.5-2B; ZS is Gemini-3.1 Pro.
   - Fine-tuned backbones run at "12 Hz up to 33 Hz"; "79 ms per decision" at 9B vs "39 ms for 2B".
   - Interpreter API in `interpreters/README.md:35-38`; Piper interpreter streams joints at 50 Hz.
6. **OpenETA README lines 43-45.** "With `gpt-5.6-sol` at medium reasoning effort, Codex + OpenETA reaches **70.8% Pass@1** and **90.0% Pass@5** across all 130 LIBERO tasks."
   - "35 verified tool contracts" (line 36).
   - Fresh-observation obligation (line 56).
   - "review, canary, and holdout validation" (line 63).
7. **RoboICL** ([arXiv 2609.34261](https://arxiv.org/abs/2609.34261)). Confirmed: 50.64 vs 33.68 Overall on 30 RoboDojo tasks; 14.45 → 63.33 → 78.89 at 0/1/3 shots on 3 real tasks; Jev-gated reuse cuts Astra calls by 33–48%. Caveat: the 33–48% was measured on **two development tasks** only.
8. **Safari SDK** (`google-deepmind/gemini-robotics-sdk` @41cbd35).
   - `flags.py:173-176`: `agent.model_name` defaults to `"gemini-live-2.5-flash-preview"`.
   - `config.py:218`: `sd_model_name = 'gemini-robotics-er-1.5-preview'`.
   - `config.py:251`: `sd_async_sd_interval_s = 0.2`.
   - `run_instruction_until_done.py:41`: `behavior=types.Behavior.NON_BLOCKING`. Its docstring example is "put the red dice in the green tray".
   - `embodiments/aloha.py` exists.
   - README:86: "Trusted Testers can access the Gemini Robotics On Device model from SDK v2.4.1."
9. **dimOS** (@cdfaef0).
   - `dimos/agents/typesafe/constants.py`: `DEFAULT_MODEL = "jev-latest"`, `NAV_MAX_HZ = 2.0`, `PUBLISH_HZ = 10.0`, `MIN_PROBABILITY = 0.5`, `DEADMAN_MIN_S = 1.0`, `DEADMAN_PERIODS = 2.5`.
   - `mcp_client.py:84`: `model: str = "gpt-5.6-luna"`.
   - `mcp_server.py:175`: "Cannot start '{name}': capability '{cap}' is held by '{holder}'".
   - Release v0.0.14 on 2026-09-19; 45 contributors.
10. **DrivingBench.**
    - `shared/contracts.py:9,15,16`: `NATIVE_WATCHDOG_S = 2.0`, `SPEED_RANGE_MPS = (0.5, 3.5)`, `DURATION_RANGE_S = (5.0, 60.0)`.
    - Lines 101-108 contain "Motion continues while you think; expiry starts braking" and "after an uncertain result, observe rather than blindly retrying".
11. **jev-robot-control README lines 19-23.**
    - Jev 1.13: Placed, 113 cycles, $0.018825.
    - GPT-6 Astra (low): Placed, 106 cycles, $5.933624.
    - GPT-4.1 mini: 160-cycle limit reached ($0.288512).
    - "one seed-0 trial per controller".
12. **OM1.**
    - `config/unitree_go2_autonomy.json5:3` sets `hertz: 1`.
    - `internal/fuser/fuser.go` contains "Current observations:", "MEMORY:", "Available actions:" and "What will you do next?".
    - Python branch "deprecated"; 123 contributors.
    - [arXiv 2412.18588](https://arxiv.org/abs/2412.18588): four LLMs, 1 Hz, "around 40 bits/s".
13. **LeRobot.** `docs/source/tools.mdx` confirms the v3.1 tool catalog at `meta/info.json["tools"]` with a default `say`. `annotation_pipeline.mdx` confirms `lerobot-annotate` with one shared Qwen-VL vLLM/OpenAI-compatible server. Both files are present at tag v0.6.1 (2026-08-03).
14. **Other confirmed claims:**
    - Releases: ros-mcp-server v3.1.2 (2026-09-23, 18 contributors); ROSA v1.0.10; RAI 2.12.4; PhyAgentOS v1.0.0 (2026-09-05); strands robots v0.5.2 (2026-09-17); rosclaw "v1.3.0 — Internal Alpha" (2026-09-17).
    - ROSA: `max_iterations: int = 100` (`rosa.py:94`).
    - ros2_mcp: `server/tool_safety.py` plus `tests/tool_safety_test.py` ("A new tool that nobody classified fails").
    - Strands: "Exclusive: the arm has a single command bus" (`hardware_robot.py:3372`).
    - ASPIRE: "Claude Code with Opus 4.6 1M ... real-robot agent experiments were conducted with Codex".
    - ENPIRE: "explicit motion authorization".
    - astra-robot-sim2real: 57.4% / 68.6% (30 trials) and 12 real trials.
    - metal-arm-harness: `50ec3bd` → `e179e38` is 8 h 15 min.
    - The `anthropics` org (113 repos) has no robotics repo.
    - All 14 cited arXiv IDs resolve to the stated titles.

### Corrections (5; fixed inline)

1. **RPent / Harness VLA RoboCasa365 gain: +25.4 pp → +27.1 pp.**
   - The v5 HTML/PDF abstract and §3.2 say "improves over the strongest relevant baselines by 38.6 and 27.1 percentage points": Codex (GPT-5.5) 57.1% vs RLDX-1 30.0% (Table 4: 57.2 overall, 250 rollouts, 5 held-out seeds per task).
   - The arXiv listing abstract still shows the stale 25.4.
   - Context: WorldDreamer, the best prior paper-reported row, is 35.3%, so vs the best prior method the gain is about +21.9 pp.
2. **OpenETA Codex plugin: "3 tools" → 6 tools, including a ground-truth success oracle.**
   - Branch `openeta-for-codex` (README title "OpenETA-Light"; the main README links `openeta-light`, which 404s).
   - Its README lists six tools: `observe`, `mark_point`, `move_to`, `report_issue`, `check_task` ("Query the native LIBERO success checker") and `finish_episode`.
   - The 70.8% Pass@1 was therefore obtained with in-episode access to the simulator's success predicate. That is a material comparability caveat; the 3-tool description comes from the arXiv abstract only.
3. **"LeRobot/SO-101 + LLM MCP repos all ≤5★" → false.**
   - [IliaLarchenko/robot_MCP](https://github.com/IliaLarchenko/robot_MCP) has 85★ (last push 2025-08-12). It is an SO-ARM100/101 MCP server plus a CLI agent; the README says "Claude is the best and GPT is not so good, Gemini is in between". It is stale per the note's own >6-month rule.
   - [AI-FanGe/RobotArm-MCP-P340](https://github.com/AI-FanGe/RobotArm-MCP-P340) has 37★ (another arm).
4. **Show-Harness "distillation".** The released LoRAs are fine-tuned on recorded demonstrations: [HF Show-Harness-Data](https://huggingface.co/datasets/showlab/Show-Harness-Data) has 164 real and 230 sim episodes, with 9 units of 2 cm each. Nothing documents frontier-model rollouts as the teacher. The "light head" path is real; the "distil the backbone" framing is an analogy.
5. **Show-Harness GPU-hours.** README:79 says "less than a few H200 GPU-hours", not just "a few".

### Unverifiable / not checked

- **"PhyAgentOS gained stars within weeks of creation."** The stargazer-timestamp API returned 404, so this is UNVERIFIED. For Zetta it is trivially true: created 2026-08-18, 1269★ by 2026-10-01.
- **Show-Harness exact GPU-hours, and the RPent ablation attribution** ("gains come from re-staging, not the VLA"). The v5 text supports the decomposition argument (§3.3 "Key Findings 1–3") but I did not audit the ablation tables.
- **Teammate-sourced claims** (quackd, innate-os PR #817, piper-astra-jev): spot-checked for consistency with `sources/*.md` and the repos only; they were not re-derived.
- **Waddle Labs and Manda** are still not opened (UNVERIFIED, as the note says).
- **No search was exhaustive.** "No OpenAI-Agents-SDK robot framework" still holds after 3 more searches: the closest hit, wso2-incubator/unitree-go2-realtime-agent (20★), uses the raw Realtime API, not the Agents SDK.

### Added details and missed entries (within scope)

**Model provenance for RPent (important for an Opus-backbone harness).** Paper Table 3 / Appendix H, LIBERO-Pro overall:

| Planner | LIBERO-Pro overall | RoboCasa365 |
|---|---|---|
| Claude Code (Opus-4.7) | **82.4%** | 48.8% |
| Codex (GPT-5.5) | 72.1% | 57.2% |
| Codex (GPT-6 Astra) | 92.63% | 59.20% (148/250) |

The repo default `model="sonnet"` is **not** the configuration behind the paper headline. Zero-shot (no memory) LIBERO-Pro Goal with CC: 31.0% Pos-S and 79.0% Task-T, vs CaP-X 25.6% / 16.8%.

**EmbodiedSkills has strong controlled evidence, which the note omitted.** From the README results table:

| Benchmark | EmbodiedSkills | Baseline |
|---|---|---|
| RoboTwin 2.0 (50 tasks × 100 episodes) | 86.20 | π0.5 82.74 |
| LIBERO | 97.40 | OpenPI 96.85 |
| RMBench M(n) | 12.5 | X-VLA 7.3 |

Ablation over 5,000 episodes:
- removing intermediate verification drops success to 48.2%;
- full instruction instead of semantic subtasks drops it to 34.4%;
- one action chunk per subtask drops it to 19.5%.

This is the strongest in-corpus evidence for the "independent verification" pattern (§3 item 2).

**Missed repos:**
- **[hello-robot/stretch_ai](https://github.com/hello-robot/stretch_ai)** (246★, push 2026-07-02, Hello Robot, Apache-2.0).
  - LLM agent for language-directed pick-and-place on Stretch: `python -m stretch.app.ai_pickup --use_llm --llm openai|qwen25|gemma`.
  - The LLM emits a list of robot API calls (`docs/llm_agent.md`). Defaults are Qwen2.5-3B-Instruct locally or GPT-4o-mini through the API.
  - Includes DynaMem dynamic spatial memory.
  - Maturity P/R. A production-grade mobile-manipulator reference that is missing from §2.1/§2.3.
- **[mbodiai/embodied-agents](https://github.com/mbodiai/embodied-agents)** (300★, push 2025-12-16). `mbodied` PyPI package with language/motor/sensory agents; v1.5 (2025-04-18) added Gemini backend and OpenAI tool calling. Stale per the 6-month rule.
- **[HaxxorCialtion/SimpleTool](https://github.com/HaxxorCialtion/SimpleTool)** (71★, ICML 2026 "RealtimeTool", [arXiv 2603.00030](https://arxiv.org/abs/2603.00030)).
  - Parallel multi-head decoding of function name and arguments.
  - Claims "A 4B-parameter LLM achieving 16 Hz end-to-end real-time function calling" with 3–6× speedup, and targets robotic arms.
  - Directly relevant to a "light learned head that emits tool calls" (author-reported, UNVERIFIED on robots).
- **[ajtudela/nav2_mcp_server](https://github.com/ajtudela/nav2_mcp_server)** (85★, push 2026-05-22). Typed Nav2 MCP tools (`navigate_to_pose`, `follow_waypoints`, `spin_robot`, `backup_robot`, `dock_robot`), with CI and codecov. It is a better-typed southbound adapter than raw-topic MCP servers.
- **ROS MCP servers, older/smaller:** [kakimochi/ros2-mcp-server](https://github.com/kakimochi/ros2-mcp-server) (83★, 2025-06) and [Yutarop/ros-mcp](https://github.com/Yutarop/ros-mcp) (36★, 2025-08).
- **[mm-demo-collection/MiniMax-Agent-VLA-Demo](https://github.com/mm-demo-collection/MiniMax-Agent-VLA-Demo)** (62★). MiniMax-M2.1 planner plus a Pi05 VLA executor plus MCP visual verification in LIBERO, using the Anthropic-compatible endpoint (`ANTHROPIC_API_KEY`). A small family-3 demo.
- **[RobotecAI/agentic-mobile-manipulator](https://github.com/RobotecAI/agentic-mobile-manipulator)** (48★, 2026-07). RAI-based, fully local LLM/VLM warehouse mobile-manipulator demo on AMD Ryzen AI (O3DE + ROS 2).
- **UR + LLM** (scope item; the note has no UR entry). All are small and mostly stale:
  - [sahilrajpurkar03/nlp-pnp-robotic-arm](https://github.com/sahilrajpurkar03/nlp-pnp-robotic-arm) "SPARC", 70★, 2026-07: ROS 2 + MoveIt pick-and-place.
  - [cakh/llm-ur-control](https://github.com/cakh/llm-ur-control): 20★, 2024-12.
  - [brandyda/LLM_control_UR5e](https://github.com/brandyda/LLM_control_UR5e): 21★.
  - [nonead/Nonead-Universal-Robots-MCP](https://github.com/nonead/Nonead-Universal-Robots-MCP): 8★.
  - For xArm, only dimOS and jev-robot-control (sim) are relevant; the other xArm+LLM repos are ≤2★.
- **[lykycy123/RoboJEV](https://github.com/lykycy123/RoboJEV)** (52★, 2026-09-27). "Two-stage JEV control of a Franka Panda in MuJoCo", another Jev workbench (from the kairunwen index).

### Reliability verdict

Medium-high. Every repo, file path, constant and quoted string I sampled (≈60 checks) exists exactly as stated, and no entry appears fabricated or mis-attributed. The errors are in benchmark framing:
- a stale abstract number (RPent RoboCasa);
- an under-described evaluation setup that hides a success oracle (OpenETA);
- one over-broad negative claim (SO-101 MCP star counts);
- loose "distillation" wording.

The note also omits EmbodiedSkills' numbers and the fact that the RPent headline used Opus-4.7, both of which matter for an Opus-backbone design.
