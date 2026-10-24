# Deep research prompt: coding-agent / self-improving harnesses for robot manipulation (RoboDojo focus)

> Paste everything below the line into a deep-research tool. Written 2026-10-24.

---

## Who is asking and why

I run **controlr**, a research harness where a frontier vision-LLM, called over an API, *is* the
robot controller: there is no VLA in the action path. Today it works turn by turn. The model gets
camera frames, a robot manual and the task, replies with low-level actions (`MOVE ee_delta dx dy dz`,
`GRIP`), those go through a safety envelope, and the next frame comes back. We plug into
**RoboDojo** (HKU MMLab; 42 manipulation tasks on a dual-arm ARX X5 in Isaac Sim 5.1, scored by its
own XPolicyLab eval client and public leaderboard). On `general_pickup` we get roughly 9/20. **Our goal is to top the RoboDojo-Sim
leaderboard by any means.** LLM, VLA, world-action model (WAM), video model, digital twin,
classical pose estimation and planning are all allowed, in any combination. **The one hard
requirement is an LLM/VLM agent loop on top** that decides what happens next. VLAs, WAMs,
planners and pose estimators may serve as tools under it. The current leaders are **coding
agents** (Codex CLI / Claude Code with GPT-6 Astra) that write Python against a robot API, keep
reusable skills, tools and lessons, and improve themselves between episodes. Earlier,
"PhysicalRSI" (an agent driving a VLA) held #1.

I want **every project that could raise a RoboDojo score** and that I don't know yet. From each
one I want the concrete mechanisms that move the success rate.

## Seed projects (already known, use them to calibrate "similar")

| Repo | What it is |
|---|---|
| github.com/HexaFuture/HexaShell | Thin harness: a coding agent drives RoboDojo through one CLI `robo`, 3 RGB cameras + calibration, a one-page manual. After a failure a second agent writes or fixes *tools*, which are unit-tested Python sub-commands. It triangulates instead of using depth. 46.9% in RoboDojo's official evaluation. |
| github.com/DAGroup-PKU/PyRUA-Lean | "Robot-use agent": the robot is a Python object (`robo`), and one LLM call writes a cell that perceives, moves, grasps, checks and retries. Fewer tokens than tool calling. #1 on RoboDojo (50.35% official). Also a 4B coder model; "PyRUA-RSI" announced. arXiv 2610.01939. |
| github.com/Einsia/EmbodiedEvo | A coding agent attached to a persistent RoboDojo session. It edits and executes control code, and builds `skills/` and `lessons/` in a Playground phase; these are frozen for Test, where every scene gets a fresh agent. arXiv 2610.18867. |
| github.com/OpenSenseNova/SenseNova-RoboRSI | Shared runtime plus recursive self-improvement of the *harness*, with model weights frozen. Candidate changes are retested before they are kept. Then a 27B model is trained on validated experience. Reports 54.17% SR on RoboDojo vs a 28.97 GPT-6 Astra baseline; also covers LIBERO, BEHAVIOR-1K and DexJoCo. |
| github.com/robot-agent-exploration/rax | RAX: coding agents (Codex, Claude Code, DeepSeek, Kimi) learn robot tasks by *online exploration*. Branches for RoboDojo, BEHAVIOR-1K, HumanoidArena, DexJoCo, real Franka and RB-Y1; full session logs on HF. arXiv 2610.22496. |
| github.com/RLinf/RPent | Recursive physical-agent framework: an agentic planner with memory steers frozen VLAs as primitives. Covers LIBERO, LIBERO-PRO, RoboCasa and RoboTwin, plus real Franka. arXiv 2607.08448 ("Harness VLA"). |
| github.com/capgym/cap-x | CaP-X (ICML 2026): Code-as-Policy gym and benchmark across abstraction tiers. CaP-Agent0 adds visual differencing, auto-synthesized skill libraries and ensembled reasoning; CaP-RL uses GRPO. |
| github.com/SAIL-Research-Lab/Physical-RSI | Curated, evidence-audited map of "physical recursive self-improvement" papers (EmbodiedRSI, PhysEvo, RoboHarn-Evo, Embodied Turing Machines, RSI-Anything…). **Mine its paper list.** |
| developers.googleblog.com/gemini-4-argon-for-robotics | Gemini 4 Argon: physical-reasoning model. It reconstructs calibrated MuJoCo twins from photos or video and rehearses in sim before acting. It uses `deproject(u,v,z)` with grid and reticle overlays, acts as a value function and progress detector, and generates sim data that lifted a VLA from 56% to 75%. |
| research.nvidia.com/labs/srl/projects/astranomy | Astranomy (NVIDIA SRL): GPT-6 Astra as a policy on RoboLab-120. Agentic 74.8% vs Direct 66.3% vs Astra supervising the Cosmos WAM 66.5%. Ablations: absolute EEF, geometry in the prompt, history. Model gap: Astra 72.7% vs Opus 5.5 25.3%. |
| github.com/MiaoDX/roboharness | Visual approval/evidence harness for unattended agent runs in MuJoCo: proof packs, phase manifests, metric alarms. This is the *dev-loop/verification* side, not control. |

Also already known (skip unless there is major new material since September 2026): agent-as-policy,
GPT-as-Policy, EmbodiedSWE, DexAgent, Jev-as-Policy, piper-astra-jev, RoboClaw, ABot-Claw,
PhyAgentOS, EMERGE-Policy, RoboICL, HoloAgent, NVlabs ENPIRE/ASPIRE, Voyager, CoPa, RoboOS, RAI,
dimos, XPolicyLab's own LLM adapters (`GPT_6_Astra_Direct_EEF`).

## What to find

Projects (repo, paper, tech report or serious blog post with code) from **2026-06 onward**, with
the emphasis on September–October 2026. Each must either have an LLM/VLM agent in the loop or
supply a component such an agent could call: a VLA/WAM, a pose estimator, a twin builder or a
grasp planner. The families are:

1. **Coding-agent-as-controller.** The agent writes and executes code against a robot API, CLI,
   REPL or notebook (Codex CLI, Claude Code, OpenHands, Aider, custom).
2. **Self-improving / RSI harnesses.** Skills, tools, lessons or memory are written from failures,
   verified, then frozen or inherited; a second "repair" or "critic" agent; harness evolution with
   fixed weights.
3. **Online exploration / test-time learning.** Exploration budgets, playground vs test splits,
   in-context experience packages.
4. **Perception without depth or privileged state.** Triangulation from calibrated RGB, moving a
   wrist camera to get a second view, visual differencing, virtual cameras.
5. **Token- and latency-lean agent loops.** Multi-step code cells instead of per-step tool calls,
   KV/prompt caching, small fine-tuned coder models distilled from agent traces.
7. **VLA / WAM as a tool under an agent.** This covers agents that call π0.5, GR00T, DreamZero,
   Cosmos-Policy, LingBot or similar for sub-skills, and agents that accept, edit or override
   proposals. Include **every RoboDojo-fine-tuned VLA/WAM checkpoint that is publicly released**,
   with its per-dimension scores, as tool candidates.
8. **Real-to-sim twins and model-based 6-D pose.** Photo or video → simulation scene; render-and-
   compare pose estimation from known meshes (FoundationPose-style); rehearsal or MPC in the twin
   before acting. Note how each handles lighting and background randomisation, occlusion and
   stacking.
9. **Episodic memory for manipulation.** Video captioning or annotation that an agent can query
   ("what happened in the last 30 s", "which order were the objects placed"). These matter for
   RoboDojo's Memory dimension.
10. **Bimanual and precision skills.** Handover, two-arm lifts, insertion, plugging, screwing,
    laptop closing, orientation-matched placement, grasp choice at the centre of mass.
11. **Entries on RoboDojo and its neighbours.** Anything on the RoboDojo-Sim leaderboard
   (robodojo-benchmark.com/leaderboard), PRs or adapters in github.com/XPolicyLab/XPolicyLab, and
   agentic entries on RoboTwin 2.0, BEHAVIOR-1K, LIBERO-PRO, RoboCasa365, DexJoCo, HumanoidArena,
   ManiSkill. Also the ECCV 2026 SafeWorldModels challenge on the same 42 tasks: the winners and
  their methods.

## Where to look (do all of these)

- The RoboDojo leaderboard: every entry above about 25%, with its method name, link and whether
  the evaluation was official or self-reported. Also open and closed PRs and issues in XPolicyLab
  and RoboDojo-Benchmark/RoboDojo that add policies or adapters.
- The paper list in Physical-RSI. Follow each paper to its code.
- Citations *of* and references *in* arXiv 2610.01939 (PyRUA-Lean), 2610.18867 (EmbodiedEvo),
  2610.22496 (RAX), 2607.08448 (RPent/Harness VLA), 2603.22435 (CaP-X), and SenseNova's "Has GPT-6
  Achieved Embodied Intelligence?" report.
- GitHub: topics `robot-use-agent`, `recursive-self-improvement`, `robodojo`, `code-as-policy`,
  `embodied-agent`, `self-evolving-agent`. Also forks and stargazers of the seed repos (new labs
  often fork first), and repos that import `robodojo` or XPolicyLab.
- arXiv cs.RO / cs.AI / cs.LG, 2026-08 to now, with queries such as: "coding agent" robot
  manipulation; "robot-use agent"; "self-improving" embodied harness; "skill library" LLM
  manipulation RoboDojo; "Codex" OR "Claude Code" robot; "test-time exploration" manipulation
  agent; "code as policy" 2026; "harness" VLM manipulation leaderboard.
- Hugging Face: models and datasets from these groups (robot agent session logs, coder models
  trained on robot traces, ICL experience packages).
- Lab and company blogs and X/Twitter threads (HKU MMLab, SenseTime/SenseNova, PKU, Harvard/NVIDIA
  GEAR, RLinf, OpenAI/Anthropic robotics demos), and Chinese-language posts (Zhihu, WeChat
  articles), where many RoboDojo entries are announced first.

## What to extract per project

For each project, give a block like this:

- **Name, link(s), group, first-release date, stars, and last push.**
- **Control interface:** what the model emits (code cell, CLI command, tool call, numeric
  action), how often it is called, and what one call can do.
- **Observation:** cameras and resolution, whether depth or privileged state is used (be strict
  and say *exactly* what ground truth, if any, the agent gets), calibration, and active
  perception.
- **Primitives/API:** the list of robot functions exposed. Are motion planning and IK (e.g.
  cuRobo) inside or outside the agent?
- **Memory and self-improvement:** what persists across episodes (skills, tools, lessons, code,
  weights), how a change is verified before it is kept, and whether train and test scenes are
  separated.
- **Task-specific content:** are there per-task recipes, prompts or tools? (This matters a lot
  for comparing leaderboard numbers.)
- **Models, cost per episode, latency, tokens.**
- **Results:** benchmark, n, official vs self-reported, held-out or hidden seeds, and ablations
  that isolate what helped.
- **Transferable idea for controlr:** the 1–3 mechanisms we should steal, and what each would
  cost to build.
- **Red flags:** privileged information, task-specific engineering, cherry-picked seeds, no code.

## Output

1. **A ranked table** of the new projects (not the seeds), most relevant to "general LLM agent
   beats VLAs on RoboDojo" first.
2. **A mechanism matrix:** rows = mechanisms (code-cell control, CLI tools, repair agent,
   skills/lessons, triangulation, wrist-camera re-view, visual diff, exploration phase, ensembles,
   distilled coder model, …); columns = projects, seeds included; cells = used / ablated (+Δ) /
   not used.
3. **Mechanisms with ablation evidence:** a short list of the ones whose effect on success rate is
   backed by an ablation, with the numbers.
4. **The RoboDojo leaderboard as of today:** **every** entry (not only agentic ones) with its
   per-dimension scores (Generalization, Precision, Long-Horizon, Memory, Open). For each dimension
   name the best method, since a hybrid could pick the best per dimension. Note what the protocol
   allows: known object meshes, multiple models, exploration on public layouts.
5. **Gaps:** ideas nobody has tried yet that a harness like ours could test cheaply.

Rules: cite a link for every claim, and separate "paper says" from "code shows". Mark anything
you could not verify. Prefer primary sources. Do not pad the list with general LLM-robotics
surveys or ROS/MCP wrappers that have no evaluation. Include pure VLA/WAM work only if it has
RoboDojo (or RoboTwin) numbers or a released checkpoint usable as a tool.
