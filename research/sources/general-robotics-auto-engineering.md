# General Robotics: "Introducing Auto-Engineering for Robotics" (GRID)

*Deep-dive note, written 2026-10-01.*

**What I read**
- The full blog post (dated 2026-09-09), extracted from the raw HTML, with both inline figures.
- The 140 s launch video (Whisper transcript, frame by frame) and ~~the four shorter embedded videos~~ [corrected: four of the eight shorter embedded videos. The post embeds 4 inline clips (1225144013, 1225201378, 1225144062, 1225229184) plus a 4-clip slider below the text (1225220280 "Skill1" 102 s, 1225223221 "reactive_pour" 6 s, 1225144913 "UR5e Stirring" 40 s, 1225226836 "swirling-skill3" 30 s; titles and durations from Vimeo oEmbed). The slider clips were not read.]
- The press release, seven trade-press articles and the launch tweet. GeekWire returned 403, so I read it through a syndicated rewrite.
- Earlier company posts, the 2023 GRID tech report and the Oct-2025 "Agentic Architectures for Robotics" paper.
- Docs v2.2 (via `llms.txt`), and the `GenRobo` GitHub org (two repos cloned to `research/repos/genrobo-*`).
- Robot Report podcast ep. 263 with CTO Sai Vemprala, transcribed locally.

**Conventions**
- **[derived]** marks my computation.
- **[inference]** marks my interpretation.
- **UNVERIFIED** marks a claim I could not confirm from a primary source.

---

## 1. TL;DR

- **What is automated.** The *engineering lifecycle* of deploying a robot skill. GRID agents:
  - ingest a robot (URDF, control interfaces, system identification);
  - assemble a task-specific simulator;
  - choose and execute a skill-creation method (modular composition, sim-data BC+DAgger, teleop-data fine-tuning, or video-to-sim);
  - run preflight checks and hardware evaluation;
  - debug sim-to-real gaps;
  - persist what they learned.

  This is **"LLM as robotics engineer", not "LLM as policy"**. Nothing in the post or video puts an LLM in the control loop. The robot is moved by motion planners, a state-based BC policy, a fine-tuned visuomotor policy and a fine-tuned detector/tracker.
- **Which LLM: not disclosed (UNVERIFIED).** The blog, press release, video, docs, job posts and all coverage I found name no model, prompt, tool schema, token count or cost.
  - **Strongest hint.** On the Robot Report podcast (51:31) the CTO says: "we're seeing … a lot of advances in, for example, the latest GPT models, which are really good at 3D reasoning and structured kind of semantic thinking … we are seeing a lot of that help with GRID also". I confirmed the wording with Whisper large-v3, unprompted.
  - **Lineage.** The 2023 GRID report used **GPT-4** in an Actor/Critic pair. The Oct-2025 agentic paper "primarily used **GPT 4.1 and GPT 5**" and calls the integration "LLM-agnostic". That paper also ran an ablation with **Claude Code, Codex and GitHub Copilot** as code generators.
  - **Reading.** OpenAI models are probably at least part of the stack [inference].
- **Evidence.** It is one lab demo: ~~dual **Flexiv Rizon** arms plus a **UR5e** doing test-tube pick → handover → pour, then a moving beaker, stirring and swirling~~ [corrected: dual **Flexiv Rizon** arms doing test-tube pick → handover → pour; the skill was then transferred to a single **UR5e** as a one-arm pick-and-pour with no handover (video t≈60–75 s; graph node "UR5e modular pour"). Stirring ran on the **UR5e** (video t≈76–88 s; Vimeo clip titled "UR5e Stirring"). Swirling (t≈86–104 s) and the moving-beaker reactive pour (t≈104–119 s; graph phase "03 Flexiv · reactive") ran on the Flexiv].
  - The evidence is a 54-node "engineering event graph" and a handful of before/after numbers (§4).
  - **There are no success rates, trial counts or baselines** for any hardware result.
  - Wall-clock: about **4 h from a fresh Flexiv to the first working skill** (≈20 min ingestion, ≈10 min initial sim). Later skills took **10–15 min**.
- **Product.** GRID is a closed, enterprise, cloud-first platform: a CLI plus Python SDKs (`grid-nexus-client`, `grid-cortex-client`), hosted models ("Cortex"), and Isaac Sim/AirGen sessions.
  - There is **no public pricing** ("Connect with us"). It is sold as a subscription "provider of capabilities" to enterprises and through OEM partners (podcast).
  - **Auto-Engineering has no public API or CLI command** in docs v2.2.
  - No Auto-Engineering code is open-source.
- **Verdict for Ilia.** The most complete public articulation of the *offline engineering-agent* half of a robot harness (ingestion → sim-as-tests → method selection → preflight → evidence log → memory), but weak as evidence. Borrow the **decomposition, preflight checks and evidence/repair log**; ignore the speed numbers.

---

## 2. Company and lineage

| Item | Finding | Source |
|---|---|---|
| Company | General Robotics Technology Inc., Redmond WA. Founded 2023 as **Scaled Foundations**; renamed 2025-05-12 | post `general-purpose-intelligence-for-every-robot`; SiliconANGLE |
| Founders | **Ashish Kapoor** (CEO; 17 yrs at Microsoft, GM of Autonomous Systems & Robotics Research, created **AirSim**), **Sai Vemprala** (CTO; ex-MSR; PhD Texas A&M), **Shuhang Chen**, **Dinesh Narayanan** (commercialization) | RuntimeWire; podcast |
| Size and money | ~50 employees, "about a dozen" enterprise customers, "revenue in the millions"; ~$34 M raised. Latest round ≈$25 M (Apr 2026), led by Construct Capital with Khosla, Accenture Ventures, NVIDIA and Valo. All figures are from Kapoor via GeekWire | RuntimeWire, GeekWire rewrite |
| Customers and partners | ~~HTX (Singapore MHA) is named.~~ [corrected: HTX is **not** named in the press release. It is named by GeekWire (read via the commstrader rewrite) and RuntimeWire. The `intelligence-grid` post says only "organizations within the Singapore government".] Also FANUC (General Robotics is a FANUC ASI), Galaxea, Ghost Robotics, Accenture, Microsoft Pegasus, Mcity | press release; `intelligence-grid` post |
| Research lineage | ~~AirSim → **"ChatGPT for Robotics"** (Vemprala, Bonatti, Bucker, Kapoor; IEEE Access 2024) → **GRID tech report** (arXiv 2310.00887, 2023) → PACT/SMART pretraining → **"Agentic Architectures for Robotics"** (Sept 2025) → DreamControl (humanoid skills) → **Auto-Engineering** (Sept 2026)~~ [corrected chronology, from the company blog index: AirSim → PACT (2022-10-26) → **"ChatGPT for Robotics"** (2023-02-20; IEEE Access 12, 2024) → SMART (2023-02-27) → **GRID tech report** (arXiv 2310.00887; post 2023-10-18) → DreamControl (2025-09-18; arXiv 2509.14353, which the agentic paper cites as ref. [15]) → **"Agentic Architectures for Robotics"** (post 2025-09-30, PDF dated 2025-10-2) → **Auto-Engineering** (2026-09-09). PACT/SMART *precede* the GRID report, and DreamControl *precedes* the agentic paper.] | blog index; papers |

The constant thesis is **modular "AI skills" + LLM orchestration, not one end-to-end "robot brain"**. Vemprala (podcast): "rather than training a policy to do everything, GRID will try to compose all of these individual capabilities together".

---

## 3. Architecture as disclosed

### 3.1 The four harnesses

The labels below are verbatim from the launch-video slide (t≈46–54 s):

```
TASK + ROBOT: "Pour the test tube into the beaker."  Flexiv Rizon · Bimanual setup
        │
        ▼
┌──────────────── GRID agents  (LLM: UNDISCLOSED) ─────────────────────────────┐
│ "determines which skills are required, assembles the optimal recipe of       │
│  models and approaches ... always reasoning over what must be built, tested, │
│  fixed, or improved next" (press release)                                     │
└───┬───────────────┬──────────────────────┬──────────────────────┬────────────┘
    ▼               ▼                      ▼                      ▼
Robot Ingestion  World Experience     Skill Creation           Deployment & Evaluation
[URDF]           [Simulation]         [Models][Policies]       [Preflight][Rollouts]
[Control         [Physics]            [Controllers]            [Metrics]
 interfaces]     [Success metrics]    COMPOSE: Modular         "Diagnose. Revise. Retest.
[System                               TRAIN: Policy training    Return evidence to GRID."
 identification]                      DEMONSTRATE: Human teleop [Regression tests]
                                      TRANSFER: Video-to-sim    [Validation]
                                      "+ Extend with new approaches"
    └───────────────┴──────────┬───────────┴──────────────────────┘
                               ▼
  Compounding intelligence — "Every result informs the next iteration."
  [Robots] [World/Sim] [Recipes/Skills] [Evals]
  (press release: "a set of knowledge graphs ... while ensuring customer data and IP remain protected")

RUNTIME (what actually moves the robot)                                [inference from docs + post]
 cameras ─► Cortex-hosted models (GSAM2/SAM3 segmentation, GraspGen grasps,
            depth, distilled tracker) ─► motion planner │ BC policy (EE actions)
          ─► GRID robot API (moveToPose, grasp, ...) ─► edge runtime on robot (Nexus/Zenoh)
          ─► vendor controller.  Safety and "loss-of-connectivity" logic stays on board (podcast).
```

### 3.2 What each harness did in the demo

| Harness | Concrete actions in the demo (post and video) |
|---|---|
| **Robot Ingestion** | Captured "joints, grippers, cameras, workspace, control interfaces, and calibration" as artifacts used in both sim and real. Transferred the skill to a UR5e through "common robot abstractions". These artifacts feed **preflight checks** for "novel grippers, altered sensor mounting, calibration drift, or an unexpected control frequency". |
| **World Experience** | "Simulation itself an engineering variable". For pouring, the agent coupled a **DFSPH fluid solver in NVIDIA Warp with MuJoCo rigid-body dynamics**. Kapoor (Computerworld): the fluid part "was actually created by [the] agent itself by looking at the literature", a year of work done "in minutes". Also claimed: a PLC-conveyor cell, Gaussian-splat navigation worlds, cloth. |
| **Skill Creation: Modular** | Pour on the Flexiv: **object segmentation + grasp generation + collision-avoidant motion planning**. The skill is expressed in object-relative end-effector motion, "checked … in physics-based simulation for reachability and collisions before hardware evaluation". "GRID did not need to train anything new." The video overlay reads "No task-specific data". |
| **Skill Creation: Policy** | Moving beaker. "GRID generated expert demonstrations in simulation and trained a reactive policy using behavior cloning and DAgger, entirely from synthetic data". It "chose a state-based representation with end-effector actions" for cross-robot transfer, and was evaluated on both the UR5e and the Flexiv. |
| **Skill Creation: Demonstration** | Stirring. The agent judged primitives insufficient, "request[ed] data collected with **GELLO** arms", and fine-tuned "a visuomotor policy" via "GRID's teleoperation and fine-tuning recipe" (video). The policy is not named (UNVERIFIED); `finetune submit … --action-horizon` and hosted `pi05`/`rdt-1b` suggest a VLA [inference]. The post also claims failure-targeted re-collection. |
| **Skill Creation: Video-to-Sim** | Swirling a flask from **one iPhone video, no robot demos**: hand-pose estimation + 3D object segmentation → 3D motion → sim reconstruction → randomized synthetic demos → BC policy. Motions were "checked … against the observed behavior". Specific models are not named. |
| **Deployment & Evaluation** | System-ID preflight; segmentation-prompt tuning; distilling the slow segmentation pipeline into a fine-tuned real-time detector/tracker; a command-pacing fix; depth-model selection (numbers in §4). |

### 3.3 The engineering event graph (the only "trace" published)

The figure `grid_event_graph_1.webp` is titled "GRID · Auto Engineering a Pouring Skill · **54 engineering events**". Its footer reads: "54 events · 19 main handoffs · Select any marker for findings, repairs, and supporting evidence". That footer implies an interactive viewer inside GRID, which is not public.

The legend has ~~six~~ [corrected: seven] marker types:
- Finding (△)
- Task/experiment (•)
- Simulation/training (◇)
- Demonstrated workflow (○)
- Handoff/dependency
- Finding→response
- [corrected: missing seventh entry: "□ Project-reported setup". It is used on the "Ingest robot" node.]

| Phase (events) | Nodes (verbatim) |
|---|---|
| 01 Set up & validate in sim (8) | Ingest robot · Create sim scene · Localize objects · Generate grasps · Plan object-relative motion · Configure pour motion · Modular sim eval · Repair sim limits |
| 02 Flexiv · modular (11) | Build handover · Coordinate two arms · △Robot model mismatch → **Calibrate robot 143 → 5.7 mm** · Fix joint conversion · △Wrong object selected → Tune segmentation prompts · Reject empty grasps · Test motion gates · ○Flexiv modular pour |
| 03 Flexiv · reactive (19) | Define state inputs · Use Cartesian actions · Label detector data · Synthetic demos · Train detector · Train reactive policy · Reactive sim eval · △Segmentation too slow → Speed up tracking · Restore wrist range · Deploy reactive policy · ○Reactive workflow runs · Tilt / tracking gaps |
| 04 UR5e · reuse & adapt (16) | Reuse shared skill core · UR5e sim eval · △Speed setting has no effect → **Fix pacing 500 → 29.7 Hz** · △Inaccurate scene geometry → **Improve depth 28 → 1.2 mm RMS** → Expose collisions · ○UR5e modular pour |

The graph uses the rows "Hardware checks" and "Hardware execution". Stirring and swirling are **not** in the graph; they appear only in prose and video.

---

## 4. Results with numbers

| Claim | Number | Context and caveats |
|---|---|---|
| Robot-model vs controller mismatch | **143 mm / 6.4° → 5.7 mm / 0.68°** | Preflight sysid "corrected the model and frame iteratively". Flexiv modular phase. n=1 |
| Command pacing | **~500 Hz → 30 Hz target, verified 29.7 Hz** | Found because "reducing the robot's configured speed produced little change". UR5e phase per the graph. Became part of "the shared motion layer" |
| Depth model selection | Tabletop plane error **28.0 mm → 1.2 mm** (RMS per graph) | Multiple depth models were evaluated on one capture. Better geometry then "exposed collisions that the previous depth map had hidden" |
| Fresh robot → first skill | **≈4 h** (≈20 min ingestion, ≈10 min initial sim) | Flexiv. Includes skill creation, eval, deployment, preflight and refinement |
| Subsequent skills, same setup | **10–15 min** | "as little as". Reuses integration and fixes |
| Press-release KPIs | Onboarding **1 month → ≤2 h**; model ingestion **3 days → ≤20 min**; cross-form-factor skill transfer **3 days → ≤1.5 h**; new skill creation and deployment **≤2 days** | Vendor-reported. No baseline definition, no distribution |
| Sim OOD eval of "PORTABLE BC POLICY" (video t≈112 s) | 16 cases, each overlaid "tube 100% target 0% spill 0%" | [derived] The beaker x range is 0.151–0.189 m and y range ~~0.086–0.117 m~~ [corrected: 0.083–0.117 m. Case 01's beaker reads ≈(+0.156, +0.083, +0.768) and is partly hidden behind the GRID logo overlay; the ±≈2 cm conclusion still holds]: **±≈2 cm around nominal (0.170, 0.100)**. The tube start is fixed at ≈(0.170, 0.100, 0.843) m, already above the beaker, so the policy seems to cover only the final pour/track [inference]. "target 0%" indicates a pre-pour snapshot. Final success is not shown |

**Missing from the evidence:**
- hardware success rates, trials and seeds;
- any baseline (a human engineer, or an off-the-shelf coding agent);
- LLM calls, tokens, cost and model;
- how many of the 54 events needed humans. Computerworld: "Humans are involved in safety checks". In the 2025 paper, each G1 step was manually approved because "the G1 robot does not have an effective emergency stop".

---

## 5. What can be inspected: docs, SDK and repos

No Auto-Engineering code, prompts or tool schemas are public. The closest public artifacts are below.

### 5.1 Public GRID SDK (docs v2.2): the tool surface the harnesses likely drive [inference]

**CLI** (`/v2.2/cli/reference`). Each command maps onto a harness:

| Command | Harness |
|---|---|
| `robot add [--config <yaml>]` | Ingestion. ~~Ends with~~ [corrected: At its final step it *offers*, and lets you skip with Enter, (`/v2.2/deployment/camera-calibration`)] **fixed-camera extrinsic calibration** (fixed cameras only, not wrist cameras): 12×9 radon checkerboard, 16.5 mm squares, 15–20 hand-guided samples, consistency checks |
| `sim start` | Isaac Sim or AirGen cloud sessions |
| `device add/calibrate/teleop --dataset`, `dataset visualize` | Demonstration harness (leader arms, i.e. GELLO-style) |
| `finetune submit <dataset> --steps --batch-size --action-horizon` | Demonstration harness |
| `skill run [--dev\|--deploy]` | Deployment |

**Robot API.** `make_robot(name)` gives *direct dispatch*: "each call is one round trip". `make_robot(name, program=...)` *stages* a `@program` graph to run on the robot; use it "when a loop must not round-trip". In staged mode, "robot command validation is handled edge-side by `validate_robot_calls`" (`RemoteDeployment`). `MonitorClient` exposes `recent_errors()`, `recent_trace_refs()` and `list_graph_nodes()`. ~~`create_execution_observer()` records every Cortex model call when `GRID_EXECUTION_RECORDING_DIR` is set.~~ [corrected: `create_execution_observer(*, robot_name=None)` returns a `RuntimeObserver` only when `GRID_EXECUTION_RECORDING_DIR` is set, and `None` otherwise. It records a client's model calls only if passed explicitly as `CortexClient(observer=...)`. The docs say "the environment variable alone does not attach an observer to Cortex clients".] This is plausibly the raw material for "evidence" [inference].

**Arm primitives.** The Flexiv page lists `moveToPose`, `moveToDeltaPose`, `followJointTrajectory`, `grasp`, `validateGrasp`, `getEndEffectorForce` and others. The `moveToPose` signature is worth copying:

```python
moveToPose(pose: Pose, blocking=True, *, high_frequency=False, moving_time=2.0,
           accel_time=0.5, avoid_force=False, force_threshold=10.0)
```

- In planned mode it uses a trapezoidal profile "capped by the linear and angular speed and acceleration limits".
- A blocking move "waits until the TCP is within 1 mm and 5 mrad of the target and at rest".
- `avoid_force` stops when contact force rises more than `force_threshold` N above the pre-move baseline.
- Failures raise typed `ForceThresholdExceeded`, `RobotFault` or `RuntimeError`.
- With `high_frequency=True`, "the target is sent at the speed caps without waiting, for a caller streaming targets from a control loop". The 500 Hz-instead-of-30 Hz bug is the classic failure of this mode without a rate limiter. That is my guess, not their statement [inference].

**Cortex model zoo** (`/v2.2/models`). One `CortexClient().run(ModelType.X, ...)` call, served on Ray Serve, gives access to:

| Category | Models |
|---|---|
| Depth | da3metric, foundationstereo, fast-foundationstereo, metric3d, zoedepth |
| Detection | owlv2 |
| Segmentation | gsam2, sam2, **sam3, sam3-1**, eomt, oneformer |
| Grasping | contact-graspnet, graspgen, graspgenx |
| VLM | qwen_vl (Qwen3-VL), moondream, roborefer, locate_anything |
| Embeddings | c-radio, dinov2-base |
| VLA | **pi05, rdt-1b** |
| 3D reconstruction | lingbot-map |

The docstrings for model discovery say these methods are "Essential for LLM agents to discover what [models are available]".

### 5.2 GenRobo GitHub (public)

- `GenRobo/GRID-playground` (cloned to `research/repos/genrobo-GRID-playground`). Only sample AirGen/Isaac configs (e.g. `configs/isaac/ur5e_tabletop.json`) and notebooks from the 2024 "Open GRID" era.
- `GenRobo/isaac-sim-mcp` (cloned to `research/repos/genrobo-isaac-sim-mcp`, HEAD `0bb9c07`, 2026-03-10). ~~A fork of omni-mcp's MIT Isaac Sim MCP server.~~ [corrected: It is **not** a GitHub fork (`gh api repos/GenRobo/isaac-sim-mcp` → `fork: false`). It is derived from omni-mcp's MIT-licensed code (`LICENSE:3` "Copyright (c) 2025 omni-mcp"). `main` is still `0bb9c07`. The only other branch, `fix/launch-script-install-mode-detection` (`2c414f2`, PR #1), accounts for the repo's 2026-07-08 `pushed_at`.]
  - `isaac_mcp/server.py:221` creates `FastMCP("IsaacSimMCP")`.
  - It defines 20 `@mcp.tool`s in `server.py`, plus 2 Isaac Lab tools at `isaac_mcp/lab_tools.py:16,26`. Examples: `get_scene_info` `:271`, `set_joint_drive_params` `:769`, `simulation_control` `:968`, `capture_logs` `:998`, `execute_script` `:1132`, `generate_3d_from_text_or_image` `:1438`.
  - It has one `@mcp.prompt` `asset_creation_strategy` `:1260`, which says: "if execute script due to communication error, then retry 3 times at most".
  - The `execute_script` docstring is itself the prompt: "Execute arbitrary Python code in Isaac Sim. Before executing any code, first verify if get_scene_info() has been called … Always print the formatted code into chat to confirm before execution".
  - This shows the *style* GRID uses to expose simulators to agents: a few typed tools plus an arbitrary-code escape hatch. That matches the 2025 paper's "tool-invocation vs code-generation" modes. It is not the Auto-Engineering harness itself.
- Also `unreal-mcp`/`unreal-analyzer-mcp` (Unreal backs AirGen) and forks of Isaac-GR00T, lingbot-vla and dn-splatter. `GenRobo/lore`'s repo description leaks the **private monorepo layout**: "pinned by SHA in `GenRobo/GRID` `sim/engines/lore/pins.toml`" (EpicGames/lore is a VCS). This is consistent with the blog's "monorepo that integrates across 50+ OEM robots, hundreds of models".

### 5.3 The precursor paper: the best public description of GRID's agent layer

*"Agentic Architectures for Robotics: Design Principles and Model Abilities"*, General Robotics, 2025-10-02 (`genrobo.github.io/Agentic-Robotics/paper.pdf`). Auto-Engineering reads as this architecture plus the four harnesses. Key verbatim points:

- **Tool exposure.** "Each skill—whether a perception model, planning algorithm, or robot API—is wrapped into an MCP server that provides: 1. Tool definitions with typed parameters and return values, 2. Descriptive documentation for LLM comprehension, 3. Usage examples".
- **Two modes.** In *Tool Invocation Mode* the agent invokes "MCP-exposed tools directly … parsing their outputs to guide subsequent calls". In *Code Generation Mode* it "synthesizes Python programs that integrate tools with custom logic". The paper stresses: "Code generation is therefore not an add-on but a necessity".
- **Roles.** "an orchestrator agent maintains context … A planning agent produces structured plans with citations, a code generation agent translates plans into executable routines, and a critic agent validates outputs and proposes retries."
- **Structured outputs with provenance.** "All agent responses are constrained to typed schemas … each planned action must reference the documentation or examples from which it was derived."
- **Sim-in-the-loop skill creation** (App. B.2). The agent writes `skills/safe_navigation.py`, calls `launch_sim`, parses its own log statements ("'Collided': 'True'"), and edits thresholds and ROI until there are "no collisions for 60 seconds".
- **Memory.** *Observational* memory: VLM captions plus object embeddings in a vector DB. *Operational* memory: tool calls, code and traces, plus RAG over manuals (e.g. FAA Part 107 for drone missions).
- **Infrastructure numbers** (Seattle → us-west1, 640×480 JPEG):

  | Model | H100 inference + RTT |
  |---|---|
  | ZoeDepth | 57 ms backend / 136 ms RTT over WebSocket |
  | OWLv2 | 187 / 216 ms |
  | Moondream | 1.34 / 1.38 s |

  One-way control latency: LAN ≈1.2–2.6 ms; US-West 3.7–8.4 ms; US-East 36–75 ms. The paper recommends **Zenoh for control and WebRTC for video**.
- **Ablation (Table 3).** The task was writing a UR5e pick-and-place script.

  | Condition | Claude Code | Codex | Copilot |
  |---|---|---|---|
  | Full GRID (APIs + hosted skills) | 2 attempts | 1 attempt | 3 attempts |
  | APIs only | Fail | Fail | Fail |
  | Raw SDKs | Fail | Fail | Fail |

  Copilot succeeded only "after manual guidance". Without skills, Claude Code "attempted to use color thresholding". This is n=1 per cell, but it is the core GRID argument: **the skill library, not the LLM, is the bottleneck**.

---

## 6. Third-party reception

- **Coverage** is mostly press-release driven. The analytical pieces:
  - **RuntimeWire**: "These are controlled demonstrations reported by General Robotics, rather than independent benchmarks"; "The system still needed human demonstrations for stirring, and real hardware revealed perception problems that simulation had not eliminated."
  - **Lapaas Voice**: "A ten-minute rollout after the platform has learned the cell is not comparable with first-time integration"; "No public independent benchmark was identified."
  - **Computerworld**: Kapoor on cloud latency, "The robot needs to have two loops … inner control loop … at a millisecond level … on the edge". Also car makers' reluctance to put IP in the cloud. He describes deployment options as "commercial cloud … on-prem server … edge".
- **Launch tweet** (`x.com/genrobotics_ai/status/2097713940893679902`, 2026-09-09 15:49Z): "What if a robot could take a goal and determine how to achieve it? Today, we're introducing Auto-Engineering on GRID." At fetch time it had 268 likes and 13 replies. I found **no HN or Reddit discussion**; the HN Algolia search returned only a 2-point 2023 GRID-playground post.
- **Podcast** (Robot Report ep. 263, 2026-09-25, Vemprala, interview from 24:40; my local Whisper transcript). The four layers are described exactly as in the blog.
  - **Architecture.** He calls it "kind of like a system one, system two": heavy reasoning, sims, training and inference in the cloud; "lightweight clients" on the robot that "also run safety critical operations such as obstacle avoidance or in the loss of connectivity".
  - **Deployment data.** It "should be taken and used as interventions within simulation so you can actually go simulate what you're seeing as a long tail failure".
  - **Agent layer** (48:43): "we provide MCP servers that can parse through all these skills and tools that are in GRID, either user created or GR created".
  - **Auto-Engineering** (51:02–52:25): "we have given access to agents [to] each and every piece of GRID … if you give it a task today, it can assemble or suggest the right robots and the right models. It can build the simulations … of course the human still has to provide the intent … everything is a harness where the agent is learning from its outputs and refining itself for the next run".
  - **Business model** (52:48, 60:07). GRID is "a provider of capabilities … subscribed to by the users". Deployments are mostly on "private tenants" or on-prem, and customers keep their data and skills. Access is only via enterprise deals or OEM partners, "through agentic means or … a command line interface".

---

## 7. Assessment

### Strengths
1. **It is the right decomposition for the "engineer" half of a harness.** The four harnesses plus persistent memory map cleanly onto how real integration fails. Every bug they report is a classic one: frame/kinematics mismatch, control-rate pacing, prompt-sensitive open-vocabulary segmentation, a bad depth plane, and perception too slow for a reactive loop. That makes the Deployment & Evaluation harness credible even without statistics.
2. **Method selection is the interesting agentic decision.** The agent chose modular → synthetic BC+DAgger → teleop fine-tune → video-to-sim based on task physics. Choosing a state-based EE action space for transfer is a sound engineering choice.
3. **Repairs become reusable artifacts.** A pacing fix lands in the shared motion layer. A distilled tracker becomes a skill. Failed hypotheses are kept. This "compounding" is what coding agents get from a repo plus tests.
4. **There is a real platform underneath.** It has typed robot API exceptions, force-guarded moves, edge-side command validation in staged mode, execution recording and a large hosted model zoo.

### Weaknesses
1. **Zero reproducibility.**
   - No model, prompts, tool schemas, costs or code.
   - No hardware trial counts or success rates.
   - The sim "OOD" set spans about ±2 cm.
   - The KPI table has undefined baselines.
2. **Autonomy is unquantified.** Human involvement in safety checks, GELLO collection and possibly prompt steering is acknowledged but never counted.
3. **The demo is narrow.** One lab bench, two arm families with mature SDKs, short-horizon tabletop tasks. Nothing on industrial-cell constraints (PLCs, safety-rated I/O, cycle time), despite industry being the stated market.
4. **The hybrid sim claim is unvalidated.** "Agent wrote DFSPH from the literature in minutes" (Kapoor) has no fidelity comparison. Fluid sim-to-real is notoriously hard.
5. **Lock-in.** Value depends on a private monorepo (50+ robots, hundreds of models) and the Cortex cloud; their own ablation shows coding agents fail without that library.

### Novel vs repackaged
- **Repackaged.** Code-as-policies/LLM orchestration of perception + planning is the authors' own lineage (ChatGPT-for-Robotics 2023, GRID 2023, Agentic 2025). Also repackaged: BC+DAgger from scripted sim experts; GELLO teleop + fine-tune; video-to-sim (compare DexAgent, V2D, Real2Sim pipelines); LLM-built simulations (RoboGen, GenSim lineage).
- **Genuinely new as a public claim.**
  - The agent is responsible for **the whole deployment lifecycle on real hardware**, including **preflight system identification and low-level debugging** (pacing, depth-model choice, distillation of a missing real-time component).
  - Engineering knowledge persists across tasks and robots.
  - An **engineering event graph** (findings → responses → evidence) is offered as a product artifact.
- **Maturity.** The platform is production-grade. Auto-Engineering is an internal capability shown in one curated demo, without public access.

### Relationship to other sources in this study
- It sits with **EmbodiedSWE**, **DexAgent** and **llm-robotics-playground**. All of these use the LLM as an offline engineer or teacher whose output is code, data or policies. EmbodiedSWE supplies the benchmark numbers GRID lacks.
- It contrasts with **GPT-as-Policy**, **GPT-Policy**, **metal-arm-harness** and **quackd** (LLM in the loop).
- Manda's Gemini connector fits GRID's thesis: 0/4 emitting Cartesian deltas vs 6/6 emitting a point + phase for a geometric controller. The LLM decides *what/where*; a classical controller decides *how*.

### What Ilia's Opus-backbone harness should borrow
1. **Two tiers, explicitly.** Opus works at engineering and decision time: it writes and repairs skills, picks methods, reads evidence, and invokes at most about 1 Hz semantic tools. A deterministic controller or a **small learned head** runs at control rate. GRID's "distill slow segmentation into a fine-tuned tracker" is a ready template for the light learned head: Opus plus slow foundation models label the data, and a small model runs in the loop.
2. **A robot card plus preflight tools, run before every session.** Each check returns structured pass/fail and numbers to Opus:
   - (a) FK vs controller TCP-pose error, with a threshold such as <10 mm / <1°;
   - (b) measured command rate vs target (±5%);
   - (c) depth plane-fit RMS on the table (a few mm);
   - (d) camera extrinsic consistency;
   - (e) gripper open/close/empty-grasp check;
   - (f) a speed-scaling sanity test. This is exactly the check that exposed their 500 Hz bug.
3. **Semantic, safety-typed motion tools.** Copy the `moveToPose(... blocking, moving_time, avoid_force, force_threshold)` contract. Use arrival tolerances, and typed exceptions (`ForceThresholdExceeded`, `RobotFault`, "stopped short") as tool errors the LLM can reason about. In streaming mode, enforce rate limits **in the tool**, not in LLM-written code.
4. **Typed tool docs with usage examples and a code-generation escape hatch.** Use MCP-style schemas plus `execute_script` in *simulation only*. On hardware, send programs to an edge executor that validates every robot call, as in GRID's `validate_robot_calls`.
5. **An evidence log and a repair memory.** Log every finding → hypothesis → experiment → metric before/after → accepted/rejected, with trace IDs back to model calls (compare `create_execution_observer`), and retrieve it in later sessions. This is cheap to build and is what "compounding" actually is.
6. **Sim as unit tests for LLM-written skills.** Gate hardware on reachability, collision and task-metric checks in sim, and keep a regression suite per skill and robot.

### What to avoid
- Copying GRID's evaluation style. Report n, seeds, success, interventions and $/tokens from day one; EmbodiedSWE and Manda show how.
- Trying to replicate the breadth (50+ robots, hundreds of models). The ablation says the skill library dominates, so start with 1–2 robots and a curated 10–20-tool library.
- Putting control loops across WAN. Even GRID keeps the inner loop and safety on the edge. Their measured US-East one-way control latency was 36–75 ms.
- Labelling ±2 cm perturbations "OOD", or presenting "10–15 min" re-deployments as integration speed.

---

## Sources
- https://www.generalrobotics.company/post/introducing-auto-engineering-for-robotics (main post; figures `…/6aa128d7735034f9178a3df6_grid_event_graph_1.webp`, `…/6aa193268b4b3e8e256df0a6_image_36_1.webp`)
- https://vimeo.com/1225154363/2638b7883d (launch video, 140 s); embedded videos 1225144013, 1225201378, 1225144062, 1225229184
- https://x.com/genrobotics_ai/status/2097713940893679902
- https://www.roboticstomorrow.com/news/2026/09/09/general-robotics-grid-becomes-first-robot-intelligence-platform-to-auto-engineer-entire-development-and-deployment-lifecycle/27070/ (press release)
- https://runtimewire.com/article/general-robotics-auto-engineering-grid-robot-deployment
- https://www.computerworld.com/article/4221292/general-robotics-takes-a-new-approach-to-building-a-robotic-brain.html
- https://siliconangle.com/2026/09/09/general-robotics-grid-platform-engineers-itself-cutting-robot-setup-to-hours/
- https://lapaasvoice.com/grid-auto-engineering-robot-setup
- https://roboticsandautomationnews.com/2026/09/10/general-robotics-says-grid-can-automate-entire-robot-development-and-deployment-lifecycle/104729/
- https://www.geekwire.com/2026/general-robotics-led-by-microsoft-vets-says-its-ai-has-cut-robot-setup-from-a-month-to-hours/ (403; read via https://commstrader.com/technology/general-robotics-led-by-microsoft-vets-says-its-ai-has-cut-robot-setup-from-a-month-to-hours/)
- https://www.therobotreport.com/general-robotics-is-betting-on-modular-intelligence-not-one-robot-brain/ ; audio https://soundcloud.com/robot-report-podcast/modular-intelligence
- https://www.generalrobotics.company/post/agentic-robotics ; paper https://genrobo.github.io/Agentic-Robotics/paper.pdf
- https://arxiv.org/abs/2310.00887 (GRID tech report, 2023)
- https://www.generalrobotics.company/post/intelligence-grid ; https://www.generalrobotics.company/post/general-purpose-intelligence-for-every-robot ; https://www.generalrobotics.company/post/introducing-grid-enterprise ; https://www.generalrobotics.company/grid
- https://docs.generalrobotics.dev/llms.txt ; /v2.2/cli/reference ; /v2.2/python-api/overview ; /v2.2/python-api/flexiv-rizon/flexivrizon ; /v2.2/python-api/grid-nexus-client/remotedeployment ; /v2.2/deployment/camera-calibration ; /v2.2/models/overview
- https://github.com/GenRobo (GRID-playground, isaac-sim-mcp, unreal-mcp, lore)
- https://jobs.ashbyhq.com/generalrobotics (job posts; "Develop Auto-Engineering methods for scalable and robust deployment")
- https://github.com/AkihikoWatanabe/paper_notes/issues/6544

---

## Verification (fact-check pass)

*Adversarial re-check, 2026-10-02.*

**Sources I re-fetched or re-ran myself:**
- the post HTML and both figures;
- the launch video, re-sampled frame by frame;
- the press release, RuntimeWire, Computerworld, SiliconANGLE, Lapaas and the commstrader GeekWire rewrite (GeekWire itself returned 403 again);
- the Robot Report page and the SoundCloud audio, with my own faster-whisper `medium.en` / `large-v3` passes over 38:55–39:30, 43:20–44:05, 48:20–52:30, 52:30–54:05, 60:00–60:40 and 60:55–61:20;
- the agentic paper PDF (same bytes as the colleague's copy), and arXiv 2310.00887;
- the docs v2.2 pages, the `gh api` listing of the GenRobo org, the local clones, the Ashby job API, fxtwitter (tweet) and HN Algolia.

**Not reproduced independently.** I could not fetch the launch video myself (Vimeo returned 401 / a Cloudflare block). I used the colleague's download, `/tmp/gr/main.mp4` (140.416 s, 1920×1080), which matches Vimeo oEmbed (140 s, title "Introducing auto-engineering in GRID", uploaded 2026-09-09).

### Confirmed (seen in a primary source)

**Blog post** (dated "September 9, 2026"):
- 143 mm / 6.4° → 5.7 mm / 0.68°.
- "roughly 500 Hz rather than the intended 30 Hz … verified the repair at 29.7 Hz".
- Plane error "from 28.0 mm to 1.2 mm".
- "about four hours", "roughly 20 minutes for robot ingestion, 10 minutes for the initial simulation", "as little as 10–15 minutes".
- "monorepo that integrates across 50+ OEM robots, hundreds of models".
- DFSPH in NVIDIA Warp + MuJoCo.
- BC + DAgger "entirely from synthetic data".
- "state-based representation with end-effector actions".
- GELLO arms.
- "single phone video".
- Segmentation-prompt tuning.
- The distilled "lightweight detector/tracker".
- The "shared motion layer".
- PLC conveyor, Gaussian splat and cloth.

**Event graph** (`grid_event_graph_1.webp`, 2606×1409):
- Title "Auto Engineering a Pouring Skill", "54 ENGINEERING EVENTS".
- Phase counts 8 / 11 / 19 / 16 (sum 54).
- Footer "54 events · 19 main handoffs · Select any marker for findings, repairs, and supporting evidence".
- Every node label in the §3.3 table.
- Calibration sits in phase 02 (Flexiv modular). Pacing and depth sit in phase 04 (UR5e). The graph labels the depth result "28 → 1.2 mm RMS".
- Stirring and swirling are absent from the graph.

**Launch video:**
- The slide labels in §3.1 match the frame at t=50 s, including "Four robotics harnesses. One continuous engineering loop."
- The overlay "No task-specific data." appears at t≈26–41 s.
- The narration "We gave it a single iPhone video" is burned into the subtitles.
- The OOD grid at t≈110–115 s shows 16 cases labelled "CASE nn | OOD | PORTABLE BC POLICY", each "tube 100% target 0% spill 0%". The beaker x range is 0.151–0.189 m. The tube start is (0.169–0.171, 0.100–0.101, 0.842–0.843) m.

**Press release** (RoboticsTomorrow, 09/09/26):
- All four KPIs, verbatim: "one month to as little as two hours", "three days to as little as 20 minutes", "three days to as little as 1.5 hours", "as little as two days".
- The "knowledge graphs … customer data and IP remain protected" and "assembles the optimal recipe of models and approaches" quotes.
- FANUC ASI and Galaxea.

**Company facts:**
- About 50 employees, about a dozen customers, "revenue in the millions", nearly $34 M raised, and an April round of about $25 M led by Construct Capital. These come from RuntimeWire, which cites GeekWire/Kapoor; the commstrader rewrite has everything except the $25 M.
- Founders and Kapoor's 17 years at Microsoft as GM; AirSim; Vemprala's PhD from Texas A&M.
- Rename on 2025-05-12 ("Scaled Foundations is now General Robotics").
- Partners in the `intelligence-grid` post (2026-04-15): Ghost, Accenture, Pegasus, Mcity.
- Legal entity: "General Robotics Technology, Inc. a Delaware corporation" (`/legal/terms-of-use`).

**Computerworld** (Agam Shah, 2026-09-14). Confirmed quotes:
- "created by [the] agent itself by looking at the literature";
- "at least a year … done in minutes";
- "two loops";
- "inner control loop…operating at a millisecond level…on the edge";
- "commercial cloud … on-prem … edge".

"Humans are involved in safety checks" is the reporter's sentence, not a Kapoor quote.

**RuntimeWire and Lapaas.** The quotes in §6 are verbatim.

**Podcast** (ep. 263, 2026-09-25, interview from 24:40). Re-transcribed independently.
- `large-v3` at 51:31: "a lot of advances in, for example, the latest GPT models, which are really good at 3D reasoning and structured kind of semantic thinking … we are seeing a lot of that help with GRID also". `medium.en` heard "GPD"; `large-v3` heard "GPT".
- 51:07: "we have given access to agents, this each and every piece of grid".
- ~52:20: "everything is a harness where the agent is learning from its outputs".
- 48:50–49:00: the MCP servers quote. The colleague's 48:43 is about 10 s early, not material.
- 39:03: "system one, system two".
- 43:29: "interventions within simulation".
- 33:09: "rather than training a policy to do everything…" is Vemprala's line. The host repeats it at 61:00.
- 52:48: "provider of capabilities".
- 60:08: "agentic means or … command line interface".

**Agentic paper** (37 pp., PDF CreationDate 2025-10-02; blog post 2025-09-30):
- p.6 (lines ~392–393 of `pdftotext`): "primarily used GPT 4.1 and GPT 5 models as the agent … LLM-agnostic".
- The MCP 3-item list (§2.3).
- The Tool Invocation and Code Generation mode quotes.
- "not an add-on but a necessity".
- The orchestrator / planner / code-gen / critic roles.
- The "typed schemas … citations" quote.
- App. B.2: `skills/safe_navigation.py`, `launch_sim`, "'Collided': 'True'".
- "no collisions for 60 seconds" (§4.2.3).
- Two memory axes and FAA Part 107.
- Table 1 H100 WebSocket figures: ZoeDepth 57.3 / 135.8 ms, OWLv2 186.7 / 215.8 ms, Moondream 1337.0 / 1379.5 ms.
- Table 2 one-way latency: LAN 1.178–2.552 ms, US West 3.659–8.388 ms, US East 35.929–74.752 ms.
- "Zenoh can be a strong choice for control channels … WebRTC is well suited for high-resolution vision".
- Table 3: 2 / 1 / 3 for Case 1, Fail ×6 for the other cases.
- "Claude attempted to use color thresholding" (App. C.2.1).
- G1 "does not have an effective emergency stop mechanism".

**2023 GRID report:**
- "We use GPT-4 [42] as the LLM for reasoning and code generation".
- Actor/Critic roles.

**Docs v2.2:**
- The CLI commands as listed.
- 12×9 radon checkerboard, 16.5 mm, 15–20 samples.
- The `moveToPose` signature and defaults are verbatim.
- "within 1 mm and 5 mrad".
- "sent at the speed caps without waiting".
- `ForceThresholdExceeded` / `RobotFault` / `RuntimeError`.
- "each call is one round trip" / "when a loop must not round-trip" (`/v2.2/python-api/overview`).
- `validate_robot_calls` inside `prepare_program_nodes` (RemoteDeployment).
- `MonitorClient.recent_errors(limit=50)`, `recent_trace_refs`, `list_graph_nodes`.
- Ray Serve.
- "Essential for LLM agents to discover what…" (`CortexClient.available_models`).
- The full Cortex model list matches exactly.
- No Auto-Engineering command or page.

**Repos:**
- `server.py:221` `FastMCP("IsaacSimMCP")`.
- 20 active `@mcp.tool` decorators. `grep` counts 21 because `:1429` is commented out.
- `lab_tools.py:16,26`.
- All cited line numbers.
- The `asset_creation_strategy` "retry 3 times at most" text.
- The `execute_script` docstring.
- GRID-playground contents, including `configs/isaac/ur5e_tabletop.json`.
- The `GenRobo/lore` description, "pinned by SHA in GenRobo/GRID sim/engines/lore/pins.toml"; its parent EpicGames/lore is "an open source version control system".

**Other:**
- Tweet `2097713940893679902`: 2026-09-09 15:49:11Z, text verbatim, 268 likes, 13 replies.
- HN: only the 2-point 2023-10-17 GRID-playground story.
- Ashby: "Develop Auto-Engineering methods for scalable and robust deployment" appears in both Foundation-Model-Training postings (Redmond and Singapore). No posting names an LLM vendor.
- `/pricing` returns 404; `/grid` shows only "Connect with us".

### Corrections (9; also fixed inline)

1. **"the four shorter embedded videos".** There are **8** besides the hero video: 4 inline plus a 4-clip slider ("Skill1" 102 s, "reactive_pour" 6 s, "UR5e Stirring" 40 s, "swirling-skill3" 30 s), per Vimeo oEmbed. The slider clips were never read. Their content is UNVERIFIED beyond the thumbnails; "Skill1" shows "Flexiv | UR5e" tabs and a sim inset.
2. **"dual Flexiv plus a UR5e doing pick → handover → pour".**
   - The handover is dual-Flexiv only. The UR5e does a single-arm pour.
   - **Stirring was on the UR5e** (video t≈76–88 s, chip "Inference"; Vimeo title "UR5e Stirring").
   - Swirling and the reactive moving-beaker pour were on the Flexiv.
3. **"HTX … named" in the press release.** It is not in the press release. It comes from GeekWire (commstrader rewrite) and RuntimeWire.
4. **Lineage order.** PACT (2022-10-26) and SMART (2023-02-27) precede the GRID report (2023-10-18). DreamControl (2025-09-18) precedes the agentic paper (2025-09-30), which cites it. The note's "Sept 2025" (§2) and "2025-10-02" (§5.3) are both defensible: the post is dated 2025-09-30 and the PDF 2025-10-2.
5. **Event-graph legend.** It has **7** entries, not 6. The missing one is "□ Project-reported setup", used on "Ingest robot".
6. **OOD beaker y range.** 0.086–0.117 → **0.083–0.117 m**. Case 01 is partly hidden behind the logo overlay. ±≈2 cm still holds.
7. **`robot add` "ends with" calibration.** Calibration is *offered* at the final step and can be skipped. It covers fixed cameras only.
8. **`create_execution_observer()` "records every Cortex model call".** It returns an observer only if `GRID_EXECUTION_RECORDING_DIR` is set, and that observer must be passed as `CortexClient(observer=...)`. The env var alone attaches nothing.
9. **isaac-sim-mcp "a fork of omni-mcp".** It is not a GitHub fork (`fork:false`). It is MIT code derived from omni-mcp (`LICENSE:3`).

### Unverifiable / inference flagged

- **"This is n=1 per cell" (Table 3).** The paper never states a trial count. The cells are attempts-to-success for one task. Treat "n=1" as [inference].
- **"Copilot succeeded only after manual guidance".** The paper's wording is "Copilot resolved initialization issues after manual guidance".
- **Manda cross-reference (0/4 vs 6/6).** I did not check it against a primary source here. It matches `sources/manda-robotics.md`, which cites `connector.py:3-7`.
- **The $25 M round size.** It appears only in RuntimeWire's paraphrase of GeekWire. The original is still 403.
- **The stirring policy's identity, the video-to-sim models, the LLM behind Auto-Engineering, and the share of the 54 events that needed humans.** None are disclosed (UNVERIFIED, as the note says).
- **"'target 0%' indicates a pre-pour snapshot".** Plausible, but [inference].
- **Real-time playback.** The video carries no playback-speed annotation, so it cannot be determined whether the hardware clips run in real time.

### Missed details (added)

1. **Task interface = typed natural-language turns.** The video shows chat-style prompts being typed:
   - "Pick up a test tube, pass it between two robot arms, and pour into a beaker." (t≈18–22 s; the slide card reads "Pour the test tube into the beaker. / Flexiv Rizon / Bimanual setup")
   - "Can you run this on the UR5e instead?" (≈60 s)
   - "Can you stir the solution in the beaker?" (≈76–78 s)
   - "Can you swirl a flask like this?" with an attached phone clip (≈86–90 s)
   - "By the way, my beaker is moving" (≈106–110 s)

   The human supplies the goal and the constraints; the agent picks the method. The CTO adds that it "can receive some input from the user on what objects they are interested in or … what kinds of worlds" (51:25). Humans are therefore in the loop for scene and world specification, not only for intent and safety.
2. **Video order differs from the post:** UR5e transfer → stirring → swirling → moving beaker ("Finally, we told it the beaker could move"). The only overlay chips are "No task-specific data." and "Inference".
3. **The post never uses the words "LLM" or "language model".** It says only "agent(s)" (9 times) and "foundation model" once. The model class, prompts and tool schemas are absent even at the vocabulary level.
4. **Strongest autonomy claims in the post, unquantified:**
   - for the modular skill: "GRID did not need to train anything new, and built the entire skill autonomously";
   - "When a task demands something GRID cannot yet do, the agent builds that capability and adds it back to the platform";
   - the harnesses were built "over the last few months".
5. **Explicit method menu in the post.** "composing existing models and controllers, generating synthetic demonstrations and training policies, collecting human demonstrations, **fine-tuning VLAs**, or learning from human video". This supports the note's "probably a VLA" inference for stirring, though the post itself calls that policy a "visuomotor policy".
6. **Provenance claims.**
   - Modular skill: "retaining the component configurations and test results so the deployed behavior could be traced to what had been verified".
   - Video-to-sim: "Each generated demonstration remained traceable to the source video".
   - The repair memory covers "what failed, how GRID tested it, which approaches were rejected, and which repairs survived evaluation".
7. **Training-signal framing.** The post motivates harnesses by coding harnesses, where verifiable feedback is turned "into learning signals that improve the models themselves". It does **not** claim GRID trains its agent model on robot feedback.
8. **Graph structure.**
   - There are only three bands: "Robot ingestion / Simulation" (ingestion and world experience are merged), "Skill Creation Harness" and "Deployment & Evaluation Harness".
   - The sub-rows are: Robot & scene, Modular composition, Data & training, Simulation evaluation, Model & control, Perception adaptation, Hardware execution, Hardware checks.
   - Several markers have no caption, so the "54 events" cannot be fully enumerated from the figure.
9. **Docs details relevant to the pacing bug.**
   - `setJointAngles(high_frequency=True)` is documented for "a caller streaming targets from a ~1-100 Hz control loop (e.g. a Vision-Language-Action model)". "The robot's motion generator interpolates between targets at the configured speed caps".
   - `avoid_force` together with `high_frequency` raises `ValueError`, so there is no force guard in streaming mode.
   - `moveToPose` also raises `ValueError` / `TypeError` on bad arguments.
10. **The public product lacks the demo's simulator.** Docs v2.2 and `llms-full.txt` mention no MuJoCo, Warp or DFSPH. The only public simulators are Isaac Sim and AirGen. `agent.yaml` (Isaac session config) lists teleoperation agents including "Gello devices", so GELLO is a first-class device.
11. **Same number, different metric.** The press release's "Model Ingestion … 20 minutes" refers to AI models. The post's "20 minutes for robot ingestion" refers to robots. Do not conflate them. Likewise "Robot Onboarding … two hours" in the press release versus "about four hours" fresh-robot-to-first-skill in the post.
12. **Data-sharing tension.**
   - The press release says the knowledge graphs feed "back into the platform, raising the baseline of future deployments for everything that comes after it".
   - The CTO says "we don't use any of their data ourselves" and describes most deployments as "private tenants" (53:28–53:42).
   - What, if anything, transfers across customers is unspecified.
13. **Agentic-paper extras.**
   - Table 3 has a fourth column, "GRID Interface" = 1 attempt for Case 1.
   - In Case 3, output degraded to "low-level joint commands — syntactically valid but behaviorally unusable".
   - "GPT-5 outperformed GPT-4.1" on chess reasoning (§4.2.4).
   - Prompting must be "schema-constrained, grounded in retrieved documentation, and tuned for actionability".
   - Table 1 also covers T4 and L4 GPUs and REST vs WebSocket. REST adds about 15–40 ms over WebSocket on H100.
14. **Robot ecosystem clip** (1225201378, "Robot Ecosystem on GRID"). It shows UR5e, Flexiv Rizon, FANUC CRX, Galaxea R1 Pro, Unitree Go2, Ghost Vision 60, Clearpath Jackal, Clearpath Husky, Freefly Astro, ModalAI Starling 2, Unitree G1 and DEEP Robotics DR02. The 2026-04-15 `intelligence-grid` post said "more than 40 robots"; the 2026-09 post says "50+ OEM robots".
15. **"sim-creation" clip** (1225144062, 15 s). It shows a lab-pour sim, a bimanual cloth (shirt) task, a warehouse conveyor with an arm palletizing, and a humanoid in a photoreal office scan. This is visual support for the post's PLC, splat and cloth claims; no fidelity numbers are given.
16. **Misc.**
   - Press-release investors also include E14 and Shorooq.
   - Tweet: 57 reposts and 54,323 views at fetch time.
   - Kapoor (Computerworld): "Having one layer that looks after 50 to 100 robots is way more important than 50 or 100… robots that operate completely independently".
   - Two more GenRobo repos, DreamControl (225★) and DreamControl-v2. GRID-playground has 346★.

**Net.** Every headline number in the note (143→5.7 mm, 500→29.7 Hz, 28→1.2 mm, 4 h / 20 / 10 min, 10–15 min, the KPIs, the paper's latency tables and ablation, and the CTO "GPT" quote) checks out against primary sources. The corrections are about attribution, chronology, robot assignment and API nuance.
