# LLM-as-planner, Code-as-Policy and spatial-constraint generation: the 2022–2026 lineage

*Landscape sweep for the "frontier LLM API as backbone + light action head" harness. Compiled 2026-10-01. Every 2025–2026 claim below comes from a primary source I opened (arXiv PDF/abstract, official docs or blog, or a GitHub repo). Effect sizes are quoted with trial counts where the paper gives them. "(computed)" marks numbers I derived from a paper's table. "UNVERIFIED" marks claims I could not confirm in a primary source.*

Working files: PDFs and extracted text are in `research/tmp_planner/{pdf,raw}/`. Repos cloned for this note are `research/repos/{cap-x,agent-as-policy,Pigey}`.

---

## 0. TL;DR for the harness design

1. **The field has converged on the shape the user is planning.** By mid-2026 the strongest zero-shot systems pair a frontier LLM, called through an API or coding-agent CLI, with a tool interface that is typed and verifiable. Contact-rich motion goes to a frozen learned policy (π0.5) or to analytic primitives (IK/TAMP). Two examples:
   - **Pigey** (Claude Opus 4.7 + π0.5-DROID + TiPToP TAMP): 97.3% over 30 real Franka tasks × 5 trials. The same frozen VLA prompted directly gets 16.7%.
   - **Harness VLA** (frozen π0.5 exposed as a `vla_act` primitive): LIBERO-Pro rises from 50.0% (raw VLA) to 82.4% with Claude Code/Opus 4.7 and 92.6% with Codex/GPT-6 Astra.
2. **Pure code-as-policy over low-level APIs is still far below human-written code in single-shot mode.** On CaP-Bench S4, Gemini 3 Pro scores 32.3%, Opus 4.5 23.8% and GPT-5.2 22.0%, against 88.5% for human experts. The gap closes mainly through test-time compute: multi-turn execution feedback, text-based visual differencing, skill libraries and ensembles. Those took Gemini 3 Pro from roughly 24% (S3) to 68%.
3. **The LLM should not be in the fast loop.** Costs and latencies:
   - Agent-as-Policy: 9–51 min and USD 4–24 per real task.
   - ALRM: ~~33–83 s per task for planning alone.~~ [corrected: 33.44 s (CaP) → 82.60 s (TaP) is Claude-4.1-Opus only; across the 10 LLMs mean per-task latency spans 24.89 s (Falcon-H1-7B, CaP) to 161.73 s (DeepSeek-V3.1, TaP), e.g. GPT-5 145.59 s CaP / 113.88 s TaP — arXiv 2601.19510 Table, p.4]
   - DualManip: geometric adaptation runs about 46× faster than agentic replanning.

   Every winning design keeps the LLM event-driven (plan, verify, recover). Execution runs at 10–20 Hz in an optimizer, tracker or VLA.
4. **Recurring wins, with measured effects:**

   | Technique | Effect |
   |---|---|
   | Code instead of NL for spatial reasoning | 35% → 98% |
   | Constraint + solver instead of direct numeric poses | 37% → 63% |
   | Coarse-to-fine part grounding | 46% → 63% |
   | Closed-loop VLM re-check | +16–17 pp |
   | Sensor-verified grasp between pick and place | 80% → 100% |
   | Two in-context examples | 74% → 94% |
   | Text visual-diff instead of raw images in the coder's context | raw images hurt |
   | Persistent skill library | 4% → 31% on unseen long-horizon tasks; +20.6 pp |

   Details and sources are in §5.
5. **What did not work.**
   - Single-shot open-loop programs.
   - Huge hand-written prompt banks: VoxPoser's 85 examples fail when cut to 3.
   - Pipelines compiled in sim and then deployed blind: GaP 0/25 and ASPIRE 3/25 real, against 23/25 for a runtime agent.
   - Safety constraints stated only in the prompt: collision avoidance 59%, versus 87.5% with harness-level route checking.
   - Raw RGB pasted into a coder's context.

---

## 1. Taxonomy: what the LLM emits, and what executes it

| Output representation | Executor | Exemplars |
|---|---|---|
| Choice among pre-trained language skills, scored by value/affordance | Learned skill policies (BC/RL) | SayCan, Inner Monologue, Text2Motion, Hi Robot, Agentic Robot |
| Python policy code over a perception/control API | API calls → IK/motion planner/primitives | Code as Policies, ProgPrompt, ChatGPT-for-Robotics, MALMM, CaP-X, ALRM (CaP mode), ASPIRE, RATs |
| Code that builds 3D value/cost maps | Model-based planner (MPC over voxel maps) | VoxPoser |
| Code that defines keypoint/geometric cost functions | Constrained optimizer (SLSQP) + IK, re-solved at ~10 Hz | ReKep, GeoManip, CoPa (vector/surface constraints), OmniManip, Code-as-Monitor (as monitors) |
| Choice among marks or points drawn on the image | Grasp sampler, IK, waypoint interpolation | PIVOT, MOKA, CoPa, Manipulate-Anything, RoboPoint/KALIE (fine-tuned point VLMs), Gemini Robotics-ER pointing |
| Agentic tool calls in a loop (typed JSON) | TAMP, VLA rollouts, analytic primitives | Gemini-ER orchestration, Pigey, Harness VLA, RoboOS, RACAS, ALRM (TaP mode), EMERGE-Policy, RoboHarness |
| Runtime coding agent with a workspace and a CLI bridge | IK (Mink/PyRoKi), controller-enforced limits | Agent-as-Policy, CaP-Agent0, Project Fetch |
| GUI actions on a 3D interface (MCP tools) | Waypoint controller | VIA, Show-Harness (semantic action units), World Action Agent |

---

## 2. Era 1 (2022–2023): language planners and code-as-policy

| System (org, date) | LLM output → executor; perception; loop | Key quantitative results | Code | Relevance |
|---|---|---|---|---|
| **SayCan** (Google, Apr 2022) | PaLM scores candidate skill strings; each score is multiplied by the learned value function ("affordance") of that skill. Pre-trained BC/RL skills on an Everyday Robots mobile manipulator. Open-loop. | 101 instructions in 7 families. Mock kitchen: **84% planning / 74% execution**. Real kitchen: 81% / 60%. FLAN backbone: 70% / 61%. | `google-research/saycan` notebook | Origin of "LLM picks, learned skill grounds feasibility". Requires enumerating skills. |
| **Inner Monologue** (Google, Jul 2022) | Same planner, with textual feedback (success detector, scene description, human) fed back each step. Closed-loop. | Kitchen with adversarial disturbances: SayCan 12.5% / 0% / 0% (manipulation / mobile / drawers). IM with success feedback: 25% / 25% / 44.4%. Without disturbances: IM (object + success) 75 / 75 / 100 vs SayCan 50 / 50 / 83.3. | none | First evidence that closed-loop text feedback is the main lever for recovery. |
| **Code as Policies** (Google, Sep 2022) | Few-shot Codex writes Python over perception APIs (MDETR/ViLD) and control primitives. Hierarchical code-gen defines undefined functions recursively. UR5e, xArm, mobile robot. | Sim tabletop, 50 trials per task, unseen attributes and instructions: long-horizon **80.0%** vs CLIPort 0.0% and NL-planner 64%. Spatial-geometric: 62.0%. Code vs NL for spatial reasoning (Table IV): vanilla NL 35%, CoT 58%, **code 98%**. HumanEval 39.8% with hierarchical code-gen. | `google-research/code_as_policies` (notebooks) | Canonical CaP. Its high-level primitives are what CaP-X later shows inflate success (§4). |
| **ProgPrompt** (NVIDIA/USC, Sep 2022) | Pythonic prompt: imported action functions plus an object list plus example programs, with assertion-based recovery. Real Franka uses `grab_and_putin(obj1,obj2)` over ViLD + Contact-GraspNet + MPPI. | VirtualHome: SR 0.40, Exec 0.90, GCR 0.72. Zero-shot planner baseline: 0.00 SR. Real robot: qualitative only, no assertions. | `NVlabs/progprompt-vh` | Typed API stubs in the prompt constrain outputs. This is the ancestor of today's tool schemas. |
| **ChatGPT for Robotics** (Microsoft, Feb 2023) | Four-step recipe: (1) define a high-level function library, (2) prompt with allowed functions and constraints, (3) human in the loop checks the code in sim, (4) deploy. Drones, arms, AirSim. | Mostly qualitative. | `microsoft/PromptCraft-Robotics` (2.1k★) | Explicit "API design is the work" principle. |
| **Text2Motion** (Stanford, Mar 2023) | LLM proposes skill sequences. Q-functions of learned skills plus geometric-feasibility search over the whole sequence. | **82% vs 13%** for prior language planners on geometric-dependency tasks. | not found | Feasibility checking across skill chains, not just per skill. |
| **VoxPoser** (Stanford, Jul 2023) | LLM writes code that queries an open-vocabulary detector and composes 3D affordance/avoidance/rotation/velocity/gripper **value maps**. MPC plans end-effector waypoints and replans. | Real, 5 tasks × 10 trials: **88% static / 70% with disturbances**, vs LLM+primitives (CaP variant) 24% / 0%. Sim, unseen instructions and attributes: object interaction 65.0% vs 17.5%. | `huangwl18/VoxPoser` (MIT) | Dense spatial output. CoPa later found it relies on **85 hand-crafted prompt examples** and "almost complete failure" when cut to 3 per module. |
| **Voyager** (NVIDIA, May 2023; Minecraft) | GPT-4 writes JavaScript skills. Automatic curriculum, a growing **code skill library**, and self-verification. | 3.3× more unique items, tech-tree milestones up to 15.3× faster. | `MineDojo/Voyager` (7.2k★) | Template for the persistent skill libraries now used in robotics (CaP-Agent0, ASPIRE, RATs). |
| **REFLECT** (Columbia/Stanford, Jun 2023) | Hierarchical multisensory summary (vision, audio, state) → LLM explains the failure → corrected plan. RoboFail dataset. | Sim correction-planning success **~80%**. Removing audio drops explanation/localization by ~20% on execution failures. | `real-stanford/reflect` | Failure reasoning as a separate module. |

---

## 3. Era 2 (2024–mid-2025): visual prompting, spatial constraints, failure models

| System (org, date) | Output → executor; perception; loop | Key results | Code | Relevance |
|---|---|---|---|---|
| **PIVOT** (Google DeepMind, Feb 2024) | Candidate actions/points are drawn as numbered arrows on the image. The VLM picks the best; the sampling distribution is refit and the loop repeats (CEM-like). Parallel calls are ensembled. | Real navigation (4 tasks, 25% granularity, so small N), mean success (computed): no-iter/no-parallel 37.5%, 3 iterations 50%, 3 parallel 81%, both 75%. "Far from perfect" (authors). | HF Space demo | Iterative visual optimization and ensembling. Cheap to add to any API harness. |
| **MOKA** (Berkeley, Mar 2024) | GPT-4V picks from marks: grasp/function/target keypoints plus waypoints. A grasp sampler snaps to the nearest analytic grasp. Two-stage, coarse object then fine point. | 4 tasks × 2 subtasks × 10 trials, mean (computed): CaP 72.5%, VoxPoser 61.3%, **MOKA zero-shot 73.8% → in-context (2 examples) 93.8%**. Distilling 50 MOKA successes per task into Octo: 87.5%. | `moka-manipulation/moka` (MIT) | Marks plus point affordances. Distillation into a light policy is the "light action head" data path. |
| **CoPa** (Tsinghua/Shanghai Qi Zhi, Mar 2024) | GPT-4V with Set-of-Mark: coarse-to-fine part grounding for grasp, then VLM-generated **part-level geometric constraints** (vectors/surfaces) solved for post-grasp SE(3) poses. Motion planning between poses. Owl-ViT + SAM. | 10 tasks × 10 trials: **63%** vs VoxPoser 18%. Ablations: w/o foundation model 11%, **w/o coarse-to-fine 46%**, **w/o constraints (VLM outputs numeric pose) 37%**. | ~~prompts published as PDFs; no code found~~ [corrected: official code exists — `HaoxuHuang/copa` ("Official implementation of CoPa", 111★, created 2024-10-12; dirs `constraint_solver/`, `som_gpt4v/`, `graspnet/`, `real_world/`)] | The cleanest ablation for "constraints + solver beat direct numbers" and for zooming. |
| **ReKep** (Stanford, Sep 2024) | DINOv2 keypoint proposals (SAM masks, k-means) are overlaid as marks. GPT-4o writes Python **constraint functions** `f(end_effector, keypoints) → cost`, with sub-goal and path constraints per stage. SLSQP solves; re-solve ~10 Hz; keypoints tracked at 20 Hz. | 7 tasks × 10 trials: **auto 44.3%**, human-annotated constraints 68.6%, VoxPoser 10.0%. Under disturbances: 26.7% / 46.7% / 6.7%. Error breakdown: **point tracker is the largest error source**, then keypoint proposal and VLM; the optimizer contributes little. | `huangwl18/ReKep` (986★; prompt in `vlm_query/prompt_template.txt`) | Strongest "LLM writes the objective, a solver executes" design. The VLM-to-annotation gap (24 pp) shows constraint authoring is the bottleneck. |
| **Manipulate-Anything** (UW/AI2, Jun 2024) | Multi-view VLM selection (views tiled and numbered) → grasp/action generation → **sub-task verifier** → re-plan. | All 14 RLBench tasks covered (Scaling-up 10, VoxPoser 9, CaP 7, even with privileged state). Real, 7 tasks zero-shot: 38.6%. BC on its data matches human data on many tasks. | `Robot-MA/manipulate-anything` | Verifier plus view selection. A data engine for training light policies. |
| **RoboPoint** (UW/NVIDIA, Jun 2024) | Fine-tuned VLM outputs image points for "where" (synthetic data only). | +21.8% point accuracy vs GPT-4o and PIVOT. +30.5% downstream. Real robot: +39.5% over GPT-4V. | `wentaoyuan/RoboPoint` (Apache-2.0) | A small, specialized point model beats the frontier API on precise "where". |
| **KALIE** (Berkeley, Sep 2024) | Fine-tunes an open VLM for keypoint affordances from **50 human-labelled images** plus diffusion-inpainting augmentation. | 5 tasks × 15 trials: VoxPoser 12/75 (16%), MOKA 32/75 (43%), **KALIE 64/75 (85%)**. | not found | Same message as RoboPoint: a cheap, specialized point head. |
| **MALMM** (Inria/CTU, Nov 2024) | Planner agent + Coder agent + **Supervisor** agent (AutoGen). Coder outputs 3D waypoints; an RLBench motion planner executes. Observation after each step. | 9 RLBench tasks, GPT-4-Turbo: CaP 0.09, VoxPoser 0.17, single agent 0.50, **MALMM 0.81** (LLaMA-3.3-70B: 0.70). Ablations: removing step feedback costs 12/24 pp; planner/coder split adds 16/12 pp; supervisor adds 20/16 pp. | ~~project page claims code (not found on GitHub)~~ [corrected: code is on GitHub at `malmm1/MALMM` ("Repo for MALMM: Multi-Agent Large Language Models for Zero-Shot Robotics Manipulation", 6★)] | Role separation and per-step feedback, with numbers. |
| **Code-as-Monitor** (Dec 2024, CVPR'25) | VLM writes monitoring code over "constraint elements" (points, lines, surfaces) for reactive and proactive failure detection in real time. | +28.7% success and −31.8% execution time under severe disturbances vs baselines (3 simulators + real). | project page https://zhoues.github.io/Code-as-Monitor/ | Generated code used as a fast runtime check, not as the policy. Directly reusable. |
| **AHA** (NVIDIA/UW, Oct 2024) | Fine-tuned failure-detection VLM. FailGen procedurally perturbs sim demos. | +10.3% over GPT-4o in-context learning. Plugged into 3 LLM/VLM pipelines: **+21.4% task success** vs GPT-4 feedback. | project page | Specialized critic. |
| **OmniManip** (PKU/AgiBot, Jan 2025, CVPR'25) | GPT-4o picks **interaction points and directions in the object's canonical frame**. Dual closed loop: render the candidate interaction and let the VLM check/resample (RRC), plus 6D pose tracking (GenPose++) during execution. | 12 tasks × 10 trials. Rigid: VoxPoser 15.0%, CoPa 30.0%, ReKep 45.0%, **OmniManip 68.3% closed-loop vs 51.7% open-loop**. Articulated: 16.7 / 26.7 / – / **61.7 vs 45.0**. Viewpoint sweep (Recycle battery): ReKep 0/10 at 25°, 7/10 at 90°; OmniManip 7–8/10 throughout. | project page only | "Render and self-check before acting" is worth about +17 pp. |
| **SoFar** (Feb 2025) | GPT-4o agent plus PointSO, which predicts **semantic orientation** ("plug-in direction") from point clouds. 6-DoF goal → planner. | Zero-shot **48.7% Open6DOR**, **74.9% SIMPLER-Env**. 60 real tasks × 3 repeats. | `qizekun/SoFar` | Fills the orientation gap that keypoint-only constraints leave. |
| **HAMSTER** (NVIDIA, Feb 2025) | Fine-tuned high-level VLM draws a **2D end-effector path**. A low-level 3D policy (RVT-2/3DDA) follows it. | Real robot: +20% absolute (50% relative) over OpenVLA across 7 generalization axes. | project page | Shows a VLM-drawn intermediate representation working with a light 3D policy. |
| **Hi Robot** (Physical Intelligence, Feb 2025) | Fine-tuned 3B VLM high level emits language subtasks to the π0 low level. Handles user interjections. | **GPT-4o API used as the high level (same π0 low level)** scored >40% lower instruction accuracy (mis-identified objects, skipped subtasks). | none | A 2025 cautionary datapoint: frontier APIs without affordance grounding lose to an aligned small planner. Pigey (2026) reverses this using verified tools (§4). |
| **A0** (Apr 2025) | Diffusion model predicts contact point plus post-contact trajectory (embodiment-agnostic). An action-execution module maps to robots. | Franka **62.5%** (next best Molmo 43.75%). Kinova 53.75% (ReKep 33.75%). | not found | A "light affordance head" alternative to API pointing. |
| **RoboOS** (BAAI, May 2025) | RoboBrain MLLM as "brain". **Cerebellum skill library** (VLA-based and expert tools). Redis shared memory for multi-robot. Edge-cloud. | Qualitative multi-embodiment demos. | `FlagOpen/RoboOS` (627★) | An OS-style orchestration reference. |
| **RoboFAC** (SJTU, May 2025) | Lightweight failure-analysis VLM. 9,440 failure trajectories, 78,623 QA pairs. | +34.1% failure-analysis accuracy vs GPT-4o. +29.1% relative as a VLA supervisor on 4 real tasks, with lower latency than GPT-4o. | `MINT-SJTU/RoboFAC` | A local critic avoids API latency. |
| **Agentic Robot** (May 2025) | GPT-4o planner → OpenVLA executor → **Qwen2.5-VL-3B LoRA verifier** on a sliding window of third-person and wrist frames ("Standardized Action Procedure"). | LIBERO average **79.6%**. +24% over OpenVLA on Bowl-Drawer. | `Agentic-Robot/agentic-robot` (MIT) | Planner/executor/verifier triad with a cheap learned verifier. |
| **EmbodiedBench** (ICML'25) | Benchmark: 1,128 tasks, high-level vs low-level. | MLLMs do well on high-level planning, poorly on low level. Best **EB-Manipulation 28.9% (GPT-4o)**. ~~Errors: planning 55%, reasoning 41%, perception 4%.~~ [corrected: 55/41/4 is the GPT-4o error breakdown for the *high-level* EB-ALFRED. For the low-level EB-Manipulation the paper reports planning errors 44% (inaccurate gripper poses) and perception errors 33% (wrong recognition 22%) — arXiv 2502.09560 §5 error analysis] | `EmbodiedBench/EmbodiedBench` | Evidence that raw MLLM-to-low-level control is the weak link. |

**Spatial-constraint successors (2025–2026), summarized.**

- **GeoManip** (Jan 2025): symbolic geometric constraints plus a solver. Beats ReKep in OmniGibson.
- **SEAM** (CVPR'26, arXiv 2511.19315): a "vocabulary + grammar" intermediate representation with retrieval-augmented part segmentation.
- **UniManip** (Feb 2026): bi-level agentic operational graph. +22.5 pp over VLA and +25 pp over hierarchical baselines.
- **DualManip** (Sep 2026): infrequent semantic VLM path plus a fast geometric path. Geometric adaptation is about 46× faster than agentic re-verification and replanning.
- **ZeroDex** (Jun 2026): multi-view triangulation of VLM keypoints for dexterous hands.

The common direction: richer geometric primitives, better 3D lifting of 2D VLM outputs, and keeping the VLM out of the fast loop. There is no shared benchmark. Each paper uses about 6–12 real tasks × 10 trials, so cross-paper numbers are not comparable.

---

## 4. Era 3 (late 2025–2026): agentic coding harnesses and frontier models as runtime policies

### 4.1 Vendor APIs built for this role

- **Gemini Robotics-ER (Google DeepMind).**
  - Versions: ER 1.5 preview (Sep 2025), then 1.6, then **ER 2 (30 Jul 2026)**.
  - ER 2 model IDs: `gemini-robotics-er-2-preview` (built on Gemini 3.5 Flash) and `gemini-robotics-er-2-streaming-preview` (Live API, bidirectional streaming). 131,072 input / 65,536 output tokens.
  - Pointing contract: JSON `[{"point": [y, x], "label": ...}]`, coordinates normalized to 0–1000. Sample prompt: "Point to no more than 10 items in the image."
  - Thinking control: `thinking_level` ("use medium for a good balance between latency and performance").
  - Function calling works with user-defined robot APIs. The official orchestration sample declares `move(x, y, high)` and `setGripperState(opened)`, sets `max_steps = 15`, and returns `{"status": "success"}` as `function_result`.
  - ER 2 positioning: a "high-level brain" that "hands off motor execution to any given lower level VLA" or to user functions. Reported **moment-finding 91.3% (0.96 s mean absolute distance)** and **progress classification 57.4% (5 buckets)**. Demonstrated orchestrating Spot navigation and arm APIs (`google-gemini/robotics-samples/live-api`).
  - ~~Pigey and TiPToP use Gemini-ER as the perception and grounding module, not as the planner.~~ [corrected: TiPToP uses Gemini Robotics-ER 1.5 only for one grounding call (labels + boxes + symbolic goal). Pigey's real-robot runs use Gemini-ER for detection labels, but its LIBERO-PRO reasoner sweep (Table 15, arXiv 2607.21725) also runs "Gemini Rob-ER 1.6" *as the planner/reasoner*, scoring 48.0% mean]
- **Anthropic.** No robotics-specific model.
  - **Project Fetch phase 2** (18 Jun 2026): Claude Opus 4.7 in Claude Code at max effort, working autonomously, was about **20× faster** than the fastest human team. The four tasks every team completed took 9 min 35 s versus 181 min (team with Claude) and 361 min (team without). It **failed** fully autonomous ball retrieval: "poorly controlled." ~~The tasks "did not involve low-level control."~~ [corrected: exact wording is "none of the tasks in these experiments implicate the more challenging, low-level elements of robotic control". Also, the researcher still entered the prompt, approved commands and approved moving to the next task; 3 trials per task]
  - Claude models appear as backbones in Pigey (Opus 4.7 by default), Harness VLA (Opus 4.7), ASPIRE (Opus 4.6), AGP (Opus 5, Fable 5.1) and VIA (Fable 5).

### 4.2 Systems

| System (org, date) | Architecture: LLM role → tools/executor | Results (trials) | Code | Relevance |
|---|---|---|---|---|
| **ALRM** (TII, Jan 2026) | ReAct loop with two modes: **Code-as-Policy** (one program) vs **Tool-as-Policy** (iterative tool calls). ROS/MoveIt/Gazebo, WX250s. | 56-task benchmark, 10 LLMs. Claude-4.1-Opus best: **TaP 93.5% / CaP 92.6%**, latency **33.4 s → 82.6 s** with TaP. GPT-5: CaP 90.7 > TaP 85.2. Gemini-2.5-Pro gains 13.9 pp with TaP. Falcon-H1-7B CaP 84.3%. Caveat: ~~scored in a lightweight mock simulator with Gazebo-provided grasp poses, so this is a high-level benchmark.~~ [corrected: evaluation runs in a lightweight environment whose movement/perception APIs "return placeholder poses"; a move counts as successful "as long as the parameters follow the correct format". Success is scored by an LLM-as-judge panel (GPT-4.1, Claude-Sonnet-4, Gemini-2.5-Flash, averaged 0/1/2 scores) against ground truth. Gazebo/WX250s was used only to validate the ground-truth code/tool calls. No physics is executed during scoring, so this is a plan-correctness benchmark] | `tiiuae.github.io/ALRM` | Direct CaP vs tool-call comparison, with latency. |
| **CaP-X** (NVIDIA/Berkeley/Stanford/CMU, Mar 2026, ICML'26) | CaP-Gym: a Gymnasium REPL over Robosuite, LIBERO-PRO, BEHAVIOR (187 tasks) and a real Franka/AgiBot G1. **Tiers:** S1 privileged high-level; S2 high-level with real perception; S3 low-level with examples; S4 low-level without examples; M1 stdout/stderr; M2 raw RGB; M3/M4 Visual Differencing Module. Low-level APIs include `segment_sam3_text_prompt`, `point_prompt_molmo`, `plan_grasp` (Contact-GraspNet), `solve_ik` (PyRoKi) and `move_to_joints` (`capx/integrations/franka/control_reduced.py`). | 7 core tasks × 100 trials per tier. **S4 Pass@1: Gemini 3 Pro 32.3%, Opus 4.5 23.8%, GPT-5.2 22.0%, human 88.5%.** Closed-model average by abstraction: **S4 18.2 → S3 21.4 → S2 36.5 → S1 56.9**. M1 > S2 for most models, **M2 (raw images) < M1**, M3 (VDM) best. CaP-Agent0 ablation (Gemini 3 Pro, average): **M4 55 → +skill library 59 → +9 queries 1 model 66 → +3 models (GPT-5.2/Opus 4.5/Gemini 3 Pro) 68**; matches or beats human code on 4/7 tasks. BEHAVIOR pick radio: task success 24% (S3) → 56% (Agent0) vs 36% human. **CaP-RL** (GRPO on Qwen2.5-Coder-7B): Cube Lift 25 → 80% in sim; real Franka (N=25) 24 → 84% lift, 12 → 76% stack. LIBERO-PRO six-split average 18.2%. | `capgym/cap-x` (MIT, 833★) | The key benchmark for "how much is the LLM vs the API". Shows RLVR on a 7B coder transfers sim→real because the code-level action space is shared. |
| **TiPToP** (MIT, Mar 2026) | Not an agent: Gemini-ER 1.5 (one call: labels + boxes) → depth, segmentation, grasps → **cuTAMP** → joint impedance control. | Beats π0.5-DROID (350 h of demos) on average. Multi-step: wins 6/7 scenes, color cubes 9/10 vs 0/10. Open-loop: 80% on simple pick-place in Pigey's test. | `tiptop-robot/tiptop` (MIT) | Strong analytic backend to wrap as tools. |
| **RACAS** (Mar 2026) | Monitors + Controller + Memory Curator, talking only in natural language. Needs just a robot description, an action list and a task. GPT-4.1/mini. | Solved all tasks on a wheeled robot, a novel multi-jointed limb and an underwater ROV. | ~~–~~ [corrected: code + prompts at `github.com/janprz11/robot-agnostic-control` (stated in arXiv 2603.05621 §I)] | Cross-embodiment from a text description alone. |
| **RoboClaw** (Mar 2026) | VLM controller orchestrates policies. **Entangled Action Pairs** (forward + inverse reset) make data collection self-resetting. | +25% long-horizon success; human time −53.7%. | `RoboClaw-Robotics/RoboClaw` | The agent runs the data flywheel for a light policy. |
| **ENPIRE** (NVIDIA, Jun 2026) | Coding agents (Codex/GPT-5.5 xhigh, Claude Code/Opus 4.7 High, Kimi K2.6) **train** real policies: automatic reset and verification, rollout fleet, agents edit training code. | 99% success on pin-box, zip-tie and tool-use. Gym-PushT 95% in ~2 h (Claude Code and Codex; Kimi took 2×). Pin insertion: 1 → 8 agents cuts time to near-perfect from >1.5 h to ~40 min. | `NVlabs/ENPIRE` | An automation path for training the user's light action head. |
| **ASPIRE** (NVIDIA et al., Jun 2026) | Code-as-policy plus **evolutionary search** over programs. Fine-grained execution traces drive repair; a **skill library** distills validated fixes. Claude Code/Opus 4.6 in sim, Codex/GPT-5.5 for real. | +77 pp LIBERO-Pro (perturbed), +72 pp Robosuite handover, +32 pp BEHAVIOR-1K. **LIBERO-Pro Long zero-shot 31% vs 4%**. Six-split LIBERO-Pro average 72.0%. Real robot: only 3/25 in AGP's comparison. | ~~not found~~ [corrected: `NVlabs/ASPIRE` (Apache-2.0, created 2026-07-08, 228★; cloned at `research/repos/NVlabs_ASPIRE`). README: sim workflow packaged for "Claude Code with Opus 4.6 1M"; real-robot runs used Codex (paper: GPT-5.5 reasoning-xhigh)] | Skill compounding works in sim. Sim-to-real is the weak point. |
| **GaP** (Berkeley, Jul 2026) | Multi-agent coding harness emits **computation graphs** over MORSL (51 skills, Skill.md conventions). Graphs are rehearsed in a generated sim. | Beats baselines on 8 "variational automation" benchmarks. 0/25 in AGP's real comparison. | `graph-robots.github.io/gap` | Industrial-style interpretable graphs. |
| **Harness VLA** (RLinf, Jul 2026) | Planner (Codex or Claude Code) issues JSON commands from **a fixed library of 6 analytic primitives + `vla_act`** (frozen π0.5 run in short bursts with `prompt`, `max_chunks`, `stop` predicate). Task-specific memory (traces from one reference seed) plus global memory (rules, failure models). | LIBERO-Pro, 8 cells × 100 trials (few-shot memory): **raw πRLinf 50.0 → GPT-5.5 72.1 → Opus 4.7 82.4 → GPT-6 Astra (low effort) 92.63**. RoboCasa365 59.2% (Astra). RoboTwin C2R 58.4%. | `RLinf/RPent` (Apache-2.0, 1.3k★) | **Closest published match to "LLM backbone + light action head".** |
| **VIA** (Stanford, Jul 2026) | Claude Code or Codex drives a browser **3D UI via MCP tools** (`screenshot`, `gripper_teleport_via_click`, `gripper_translate`, `gripper_rotate`, `camera_orbit_via_key`, `execute_waypoint`, `end_episode`). No code generation, no perception primitives. | Overall 60% (Codex-5.5) to **88% (Claude Code + Fable)**. **96.7%** on 3 LIBERO-Goal tasks, 100% on a 7-block rainbow. A text waypoint demo lifts CC-Opus 77 → 100%. "A few dollars per successful episode." | – | A computer-use-style interface works, and performance scales with the model. |
| **Pigey** (Galanti, Shah, Dao; Jul 2026) | Claude Opus 4.7 through the raw Messages API (`real/agent.ts`, 301-line system prompt `agent-system.md`). Tools: `Perceive`, `Pick(label)`, `DropAbove(label)`, `VLARollout(subgoal)`, `LookAway/LookBack`, `Release`, `Done`. Labels must be exact Gemini-ER detections. Typed failures. **`is_grasped` sensor check overrides an optimistic backend.** Escalation: TAMP ↔ π0.5. | **DROID Franka, 30 tasks × 5 trials: π0.5 16.7%, TiPToP 48.7%, Pigey 97.3%.** On reasoning probes the VLA alone gets 4.6% vs 96.9%. **LIBERO-PRO, 6 suites: π0.5 12.8, CaP-Agent0 18.2, Pigey 44.3–53.3 across 9 reasoners** (GPT-5.5 low/med/high, Gemini-ER 1.6, Gemini 3.5 Flash, 3.1 Pro, Claude Haiku 4.5, Sonnet 4.6, Opus 4.7). "The reasoner sets the magnitude of the gain, not its sign." | `lianegalanti/Pigey` | **Most directly reusable blueprint for the user's harness.** |
| **RATs / Playful Agentic Robot Learning** (Berkeley, Jun 2026) | Self-directed **play** builds a code skill library before any tasks arrive. | +20.6 pp over CaP-Agent0 on LIBERO-PRO and +17.0 on MolmoSpaces. Retrieved skills improve RoboSuite by 8.9 and real robot by 8.8 pts. Six-split LIBERO-Pro 43.8%. | ~~–~~ [corrected: `Playful-RATs/RATs` ("Implementation of paper 'Playful Agentic Robot Learning'", Apache-2.0, created 2026-06-15)] | Offline skill pre-population. |
| **Agent as Policy (AGP)** (Notre Dame/UCSD, Sep 2026) | A general agent (Codex or Claude Code) **is** the runtime policy. It runs a workspace, writes scripts and calls a CLI bridge (`python3 robot_client.py . move_ee '{json}'`; commands `frames`, `deproject`, `move_ee`, `move_delta`, `move_joints`, `gripper`). Mink IK. The controller enforces **≤0.03 m/s, workspace clamp, ≤0.25 m per move**, and returns `IK_FAILED`, `JOINT_LIMIT`, `SETTLE_MISS`. Budget: 400 commands / 120 min. I2RT YAM arm, D405 wrist, BRIO overhead. | **GPT-6 Astra:** 4-pair assembly 8/10 (37 min, $16.6); pyramid 10/10 ($11.7); dice 10/10 ($21.1); towel sequential 5/5 (51 min, $24.1); towel simultaneous 3/5. **2-pair assembly by model:** Astra 5/5 (9.2 min, $4.47); GPT-5.6 Sol 5/5 ($3.94); Terra 1/5; Luna 0/5; **Claude Opus 5 5/5 (22.2 min, 12.4M tokens, $9.75)**; **Claude Fable 5.1 3/5 (27.7 min, $11.30)**. **Baselines, 25 trials each over 3 tasks:** GaP 0/25, ASPIRE 3/25, AGP 23/25. | `agent-as-policy-2026/agent-as-policy` (Apache-2.0) | Shows maximum flexibility (throwing, deformables) at a high time and cost per task. The bridge design is a template. |
| **Show-Harness** (NUS, Sep 2026) | **Discrete semantic action units** with deterministic interpreters. Gemini 3.1 Pro zero-shot, or a fine-tuned **Qwen3.5-2B** after a few GPU-hours. | Beats agentic and VLA baselines (10 trials per task). On LIBERO-Pro, WAA's rerun got 6.7%. | ~~–~~ [corrected: `showlab/Show-Harness` (Apache-2.0, created 2026-09-07, ~497★; cloned at `research/repos/showlab_Show-Harness`); paper links code, model and dataset from showlab.github.io/Show-Harness] | Small-model distillation of the interface. |
| **World Action Agent** (Sep 2026) | Gemini 3.7 Flash multi-agent. Auto-selected **contact views**, **action rehearsal** with planning feedback before execution, in-view correction. Skills evolved from videos and teaching (GPT-5.5 curates). | LIBERO-Pro six splits × 60 episodes: **zero-shot 28.9%, seed skills 43.3%, evolved skills 75.6%**. Fine-tuning Qwen3.5-9B on harness traces: OOD success **1.7% → 43.3%**. | – | Rehearse-before-act plus trace distillation into a small model. |
| **SafeHarness** (USC/UCF/UCSB, Sep 2026) | Built on Harness VLA. Adds obstacle-aware **route planning** (boxes + candidate waypoint routes, verify and replan before executing) and contact-side selection. GPT-5.5 / GPT-6 Astra. | SafeLIBERO: π0.5 57.8 TSR / 17.1 CAR. **Model-only 6.0 / 50.0 → +skills 31.0 / 59.0 → SafeHarness 71.9 / 87.5**, beating AEGIS by ~~+6.5 / +27.0~~ [corrected: the paper's "6.5% and 27.0%" are *relative* gains (71.9/67.5 = 1.065; 87.5/68.9 = 1.27). In absolute terms SafeHarness (GPT-6) beats AEGIS (67.5 TSR / 68.9 CAR) by +4.4 pp TSR and +18.6 pp CAR — arXiv 2609.20822 Table 1. The model-only/+skills/SafeHarness ablation (Table 2) covers 32 SafeLIBERO scenes] | – | **Prompt-stated constraints do not yield constraint-respecting behavior. Enforce them in the harness.** |
| **EMERGE-Policy** (Aug 2026), **RoboHarness** (Jul 2026), **HyCodePolicy** (Aug 2025), **Embodied Tool Protocol** (May 2026) | EMERGE: main agent plus isolated-context sub-agents (perception, monitoring, verification, memory) and Branch-Stack recovery. RoboHarness: VLAs, world-action models, RL and TAMP wrapped as skills, with a Memory Bridge for handoffs. HyCodePolicy: code plus VLM checkpoint monitoring and repair. ETP: 100+ tools. | ETP: tools help cognition and perception (+31% EB-ALFRED, +36% EB-Navigation) but **gains are limited for execution-type capabilities**. RoboHarness: 135 real trials. | EMERGE `EMERGE-Policy/EMERGE-Policy` | Orchestration variants. ETP sets expectations on what tools can fix. |

### 4.3 LIBERO-Pro: the de facto 2026 scoreboard (protocols differ, so read with care)

| Method | Backbone | Protocol | Score |
|---|---|---|---|
| π0.5 (direct) | – | 6 suites | 12.8% |
| CaP-Agent0 | Gemini 3 Pro (+ensemble) | 6 suites | 18.2% |
| World Action Agent, zero-shot | Gemini 3.7 Flash | 6 × 60 episodes | 28.9% |
| RATs (play skills) | – | 6 suites | 43.8% |
| Pigey, zero-shot, no memory | 9 reasoners, best Opus-class | 6 suites, 10 per task | 44.3–53.3% |
| ASPIRE | Opus 4.6 | 6 suites (from WAA table) | 72.0% |
| World Action Agent, evolved skills (LIBERO-90) | Gemini 3.7 Flash | 6 × 60 episodes | 75.6% |
| Harness VLA, few-shot memory from 1 seed per task | Opus 4.7 / GPT-6 Astra | 8 cells (incl. LIBERO-10) × 100 | 82.4% / 92.6% |

The ordering tracks how much **task-specific memory** is allowed: none, then cross-task skills, then per-task seed traces. Anyone citing these must state the memory regime. LIBERO-PRO itself (Oct 2025) showed VLAs that score >90% on LIBERO collapse to about 0% under perturbation, which is why agent harnesses look strong here.

---

## 5. Recurring design lessons with measured effect sizes

| # | Lesson | Evidence (effect size, N) |
|---|---|---|
| 1 | **Use code, not prose, for metric and spatial reasoning** | CaP Table IV: NL 35% → CoT 58% → code 98%. |
| 2 | **The LLM authors the objective; a solver or optimizer produces the numbers** | CoPa: constraints 63% vs direct numeric pose 37% (10 × 10). ReKep: the optimizer is a minor error source. |
| 3 | **Coarse-to-fine grounding (zoom) and iterative visual prompting** | CoPa: 63% vs 46% without coarse-to-fine (Hammer: 0% without). PIVOT navigation: 37.5% → 50% (iterations) → 81% (3 parallel calls); N is small. |
| 4 | **Close the loop at the plan level with verification** | Inner Monologue under disturbance: 12.5 → 25–33%. MALMM: per-step feedback worth 12–24 pp. OmniManip: closed vs open loop +16.6 / +16.7 pp. Manipulate-Anything: a verifier allows recovery. CaP-X: M1 (stdout/stderr) > S2 for most models. |
| 5 | **Verify with deterministic sensors, not only the VLM** | Pigey: a gripper-width `is_grasped` check between pick and place takes simple pick-place from 80% (TiPToP open-loop) to 100%. Optimistic backends are overridden. |
| 6 | **Don't paste raw images into the coder's context; translate them to text** | CaP-X: M2 (raw RGB per turn) < M1 (text only); M3 (VLM visual-diff text) best for all model families. |
| 7 | **Few-shot examples matter a lot, and hand-written prompt banks are brittle** | MOKA: 74% → 94% with 2 in-context examples. VIA: text waypoint demo 77% → 100%. VoxPoser: near-total failure when 85 examples are cut to 3 (per CoPa). CaP-X: S3 (with examples) vs S4, +3.2 pp closed-model average. |
| 8 | **High-level human primitives inflate success and cap expressivity** | CaP-X closed-model average: S4 18.2 → S1 56.9. Harness VLA deliberately keeps a **small fixed** library (6 + `vla_act`) and learns its operating range via memory. |
| 9 | **Ensembles and parallel sampling** | CaP-Agent0: +7 pp from 9 queries to one model, +2 pp more from a 3-model ensemble. PIVOT: parallel calls gave the largest gain. |
| 10 | **Persistent skill and experience libraries** | CaP-Agent0: +4 pp (9 synthesized helpers). ASPIRE: LIBERO-Pro Long 31% vs 4%. RATs: +20.6 pp. WAA: 28.9 → 43.3 → 75.6. AGP: reused notes shorten repeated trials, and experience transfers to weaker models. |
| 11 | **A frozen learned policy as a contact-rich primitive, staged by the planner** | Harness VLA: 50.0 → 82.4 / 92.6. Pigey: 16.7 → 97.3 on real DROID; VLA-only reasoning probes 4.6% → 96.9%. |
| 12 | **Role separation (planner, coder, supervisor, verifier)** | MALMM: split +16 / +12 pp, supervisor +20 / +16 pp. Agentic Robot: a 3B LoRA verifier is enough. |
| 13 | **Specialized small models beat frontier APIs on narrow perception** | RoboPoint: +21.8% point accuracy over GPT-4o. KALIE: 85% vs MOKA (GPT-4V) 43%. RoboFAC: +34.1% over GPT-4o. AHA: +10.3%. These fit as tools under the LLM. |
| 14 | **Model choice dominates within one harness** | AGP: GPT-5.6 Luna 0/5 to Astra 5/5. CaP-Bench S4 ranges 4.0–32.3%. Pigey's 9 reasoners span 44.3–53.3 (the sign of the gain holds for all). Harness VLA: GPT-5.5 72.1 vs Opus 4.7 82.4 vs Astra 92.6. |
| 15 | **Safety belongs in the harness, not the prompt** | SafeHarness: skills-only collision avoidance 59% → harness route verification 87.5%. ~~AGP and Project Fetch use controller-level clamps and an e-stop.~~ [corrected: AGP does (`agp/README_interface_real.md`: ≤0.03 m/s, workspace clamp, ≤0.25 m per move, "A human supervisor holds an emergency stop"). Neither Project Fetch article (phase 1, 12 Nov 2025; phase 2, 18 Jun 2026) mentions an e-stop, speed limits or clamps; in phase 1 the organizer had to "grab hold of the robot and power it off"] |
| 16 | **Runtime adaptation beats sim-compiled pipelines on real hardware** | AGP real comparison: GaP 0/25, ASPIRE 3/25, AGP 23/25. |
| 17 | **RL on the code-writer transfers sim→real** because the action space is code over shared tools | CaP-RL: real cube lift 24 → 84%, stack 12 → 76% (N=25). |
| 18 | **Error budgets sit in perception and tracking, not reasoning, once the loop is closed** | ReKep: point tracker is the largest error source. OmniManip: viewpoint sensitivity. EmbodiedBench: low-level drops 40–70% without vision. ETP: tools barely help execution-type capabilities. |

**Perception and control stack that keeps recurring:**
- Detection/segmentation: Owl-ViT, Grounding-DINO, then SAM/SAM2/SAM3.
- Keypoints: DINOv2 features.
- Points: Molmo / Molmo 2, Gemini-ER.
- Grasps: Contact-GraspNet, M2T2.
- 6D pose: GenPose++, FoundationPose.
- Stereo depth: FoundationStereo.
- Motion: cuRobo/cuTAMP, PyRoKi, Mink IK.
- Learned contact primitive: π0.5 (OpenPI).

---

## 6. Promising vs unpromising implementations for an "LLM API + light action head" harness

**Promising, with the strongest evidence**

1. **An event-driven agent loop over typed, verifiable tools.**
   - Tools: Perceive / Pick / Place / VLARollout / Release / Done, or Harness VLA's JSON contract.
   - Tool results must include typed failures and sensor readouts.
   - The LLM acts at subgoal granularity, every few seconds.
   - Blueprints: Pigey (`real/agent.ts`, `agent-system.md`) and Harness VLA (`RLinf/RPent`).
2. **A light learned head as one tool among several**, used only for contact-rich local phases with a stop predicate and a chunk budget (`vla_act(prompt, max_chunks, stop)`). The LLM re-stages and retries it. Analytic IK or TAMP covers free-space motion.
3. **Spatial grounding through points and constraints**, not raw numbers:
   - Gemini-ER-style point JSON, or a small fine-tuned point model (RoboPoint/KALIE class).
   - ReKep/CoPa-style cost functions solved by an optimizer.
   - Rendered or rehearsed proposals checked before execution (OmniManip RRC, WAA action rehearsal).
4. **Text-based visual state diffs and deterministic success checks** between steps (CaP-X VDM, Pigey `is_grasped`, Code-as-Monitor-style generated monitors).
5. **Persistent memory and skill libraries** of validated traces and helper code (CaP-Agent0 SkillLibrary in `capx/skills/library.py`, ASPIRE, RATs, Harness VLA task/global memory). State the memory regime in every evaluation.
6. **Controller-owned safety envelope**: velocity caps, workspace clamps, max step length, an e-stop, and refusal codes, as in the AGP bridge README. Add harness-level route verification (SafeHarness).
7. **Distill the harness into smaller models**: WAA trace fine-tuning gives 1.7 → 43.3% OOD; Show-Harness fine-tunes Qwen3.5-2B; CaP-RL uses a 7B coder with GRPO; MOKA distills into Octo. These reduce the API dependence and cost over time.

**Unpromising, or demonstrated to underperform**

- Single-shot, open-loop program generation (CaP 2022 style, S2/S4 tiers).
- LLM-emitted metric poses without a solver.
- Very large hand-crafted few-shot prompt banks.
- Feeding raw image streams into a coding model's context.
- Expecting prompt-level constraints to produce safe behavior.
- Sim-rehearsed static pipelines deployed blind on real hardware.
- Putting the API LLM inside a >1 Hz control loop. Latencies are tens of seconds per decision, and runtime-agent tasks cost $4–24 and take 9–51 min.
- Using a frontier API as the high-level planner without grounding its vocabulary in detected objects and robot affordances. Hi Robot's GPT-4o baseline lost >40% instruction accuracy. Pigey fixes this by forcing labels to come from detector output.

---

## 7. Open problems and gaps the harness could target

- **Latency and cost.** Tasks take 9–51 min and $4–24 (AGP). ER 2 claims 4× speed and streaming tool calls, but nobody reports per-decision latency against success under a fixed budget. A harness that amortizes reasoning, e.g. by planning while the arm moves (~~AGP's non-blocking controller~~ [corrected/UNVERIFIED: no non-blocking controller found in the AGP paper or repo; AGP motion commands return after the move with the achieved pose. Its fast-motion mechanism is a buffered `run_program` (agent-authored joint/gripper knots interpolated at 50 Hz, J4 up to 180°/s; `agp/README_interface_programs.md`). That pre-authors a trajectory; it does not plan concurrently]; ER 2 "avoid stop-and-think"), is under-explored.
- **Benchmark fragmentation.** LIBERO-Pro protocols differ in splits, episodes and memory regime. Real-world evaluations are 5–10 trials per task on bespoke tasks. No standard real benchmark for API-backbone harnesses exists yet. The user's RoboDojo source (teammate) may partly fill this.
- **Contact-rich and dynamic tasks.** Pure-agent systems do throwing and deformables slowly (AGP towel: 51 min, 3/5 simultaneous fold). Project Fetch could not close the physical loop.
- **Safety evaluation.** SafeLIBERO is the only collision-scored benchmark found for coding agents. Typical success-only scoring hides collisions.
- **Distillation path.** How much of the agent's behavior a 2–9B model or a light action head can absorb is shown only in sim (WAA, Show-Harness, CaP-RL).

**Cross-references to the teammate deep-dives.** GPT-6 Astra is the strongest backbone in AGP, Harness VLA and SafeHarness. SafeHarness cites the GPT-as-Policy report as its [38]: Astra "completes a zero-shot pick-and-place suite on a Franka arm almost perfectly," and paired with frozen π0.5 handles bimanual tasks. The RoboDojo/GPT-as-Policy hybrid (π0.5 with GPT corrections on 14.4% of steps) belongs to the same "LLM supervises a light policy" family as Pigey and Harness VLA. YAM arms (AGP) match the user's YAM-based sources.

---

## Sources

**Era 1–2 (arXiv, all opened)**
- SayCan https://arxiv.org/abs/2204.01691 · code https://github.com/google-research/google-research/tree/master/saycan
- Inner Monologue https://arxiv.org/abs/2207.05608
- Code as Policies https://arxiv.org/abs/2209.07753 · https://github.com/google-research/google-research/tree/master/code_as_policies
- ProgPrompt https://arxiv.org/abs/2209.11302 · https://github.com/NVlabs/progprompt-vh
- ChatGPT for Robotics https://arxiv.org/abs/2306.17582 · https://github.com/microsoft/PromptCraft-Robotics
- Text2Motion https://arxiv.org/abs/2303.12153
- VoxPoser https://arxiv.org/abs/2307.05973 · https://github.com/huangwl18/VoxPoser
- Voyager https://arxiv.org/abs/2305.16291 · https://github.com/MineDojo/Voyager
- REFLECT https://arxiv.org/abs/2306.15724 · https://github.com/real-stanford/reflect
- PIVOT https://arxiv.org/abs/2402.07872
- MOKA https://arxiv.org/abs/2403.03174 · https://github.com/moka-manipulation/moka
- CoPa https://arxiv.org/abs/2403.08248 · https://copa-2024.github.io/
- ReKep https://arxiv.org/abs/2409.01652 · https://github.com/huangwl18/ReKep
- Manipulate-Anything https://arxiv.org/abs/2406.18915 · https://github.com/Robot-MA/manipulate-anything
- RoboPoint https://arxiv.org/abs/2406.10721 · https://github.com/wentaoyuan/RoboPoint
- KALIE https://arxiv.org/abs/2409.14066
- MALMM https://arxiv.org/abs/2411.17636
- Code-as-Monitor https://arxiv.org/abs/2412.04455
- AHA https://arxiv.org/abs/2410.00371
- LRLL https://arxiv.org/abs/2406.18746
- RoboCodeX https://arxiv.org/abs/2402.16117

**2025 (opened)**
- OmniManip https://arxiv.org/abs/2501.03841
- GeoManip https://arxiv.org/abs/2501.09783
- SoFar https://arxiv.org/abs/2502.13143 · https://github.com/qizekun/SoFar
- HAMSTER https://arxiv.org/abs/2502.05485
- Hi Robot https://arxiv.org/abs/2502.19417
- EmbodiedBench https://arxiv.org/abs/2502.09560
- Gemini Robotics https://arxiv.org/abs/2503.20020
- A0 https://arxiv.org/abs/2504.12636
- RoboOS https://arxiv.org/abs/2505.03673 · https://github.com/FlagOpen/RoboOS
- RoboFAC https://arxiv.org/abs/2505.12224 · https://github.com/MINT-SJTU/RoboFAC
- Agentic Robot https://arxiv.org/abs/2505.23450 · https://github.com/Agentic-Robot/agentic-robot
- HyCodePolicy https://arxiv.org/abs/2508.02629
- Gemini Robotics 1.5 https://arxiv.org/abs/2510.03342
- LIBERO-PRO https://arxiv.org/abs/2510.03827
- SEAM https://arxiv.org/abs/2511.19315

**2026 (opened)**
- ALRM https://arxiv.org/abs/2601.19510
- UniManip https://arxiv.org/abs/2602.13086
- RACAS https://arxiv.org/abs/2603.05621
- TiPToP https://arxiv.org/abs/2603.09971 · https://github.com/tiptop-robot/tiptop
- RoboClaw https://arxiv.org/abs/2603.11558 · https://github.com/RoboClaw-Robotics/RoboClaw
- CaP-X https://arxiv.org/abs/2603.22435 · https://capgym.github.io · https://github.com/capgym/cap-x · chart data https://capgym.github.io/data/model_data.json
- Code as Agent Harness (survey) https://arxiv.org/abs/2605.18747
- Embodied Tool Protocol https://arxiv.org/abs/2605.26637
- Playful Agentic Robot Learning (RATs) https://arxiv.org/abs/2606.19419
- ENPIRE https://arxiv.org/abs/2606.19980 · https://github.com/NVlabs/ENPIRE
- ZeroDex https://arxiv.org/abs/2606.19340
- ASPIRE https://arxiv.org/abs/2607.00272
- GaP https://arxiv.org/abs/2607.05369
- Harness VLA https://arxiv.org/abs/2607.08448 · https://github.com/RLinf/RPent
- VIA https://arxiv.org/abs/2607.11119
- RoboHarness https://arxiv.org/abs/2607.18060
- Pigey https://arxiv.org/abs/2607.21725 · https://github.com/lianegalanti/Pigey
- EMERGE-Policy https://arxiv.org/abs/2608.29896 · https://github.com/EMERGE-Policy/EMERGE-Policy
- Show-Harness https://arxiv.org/abs/2609.10522
- Agent as Policy https://arxiv.org/abs/2609.12541 · https://github.com/agent-as-policy-2026/agent-as-policy
- SafeHarness https://arxiv.org/abs/2609.20822
- World Action Agent https://arxiv.org/abs/2609.29964
- DualManip https://arxiv.org/abs/2609.31112

**Vendor docs and blogs (opened)**
- Gemini Robotics-ER overview https://ai.google.dev/gemini-api/docs/robotics-overview
- Robot task orchestration https://ai.google.dev/gemini-api/docs/robotics-orchestration
- Introducing Gemini Robotics ER 2 (30 Jul 2026) https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/
- Gemini Robotics 2 https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/
- Samples https://github.com/google-gemini/robotics-samples
- Anthropic Project Fetch phase 2 (18 Jun 2026) https://www.anthropic.com/research/project-fetch-phase-two
- Project Fetch phase 1 https://www.anthropic.com/research/project-fetch-robot-dog

**Not verified / unreachable**
- technobezz ER 2 article (HTTP 403). Its ER 2 numbers were instead confirmed on blog.google.
- ~~MALMM, KALIE, CoPa, OmniManip, A0 and AHA code repositories were not found on GitHub.~~ [corrected: MALMM (`malmm1/MALMM`) and CoPa (`HaoxuHuang/copa`) do have GitHub code. KALIE, OmniManip, A0 and AHA were still not found in this fact-check] Project pages exist.
- Text2Motion code (`agiachris/text2motion`) returned 404.

---

## Verification (fact-check pass)

*Adversarial fact-check performed 2026-10-01/02. I opened every source independently: I re-downloaded the 2026 arXiv PDFs into `research/tmp_fc_planner/pdf/`, extracted them myself with `pdftotext -layout` into `tmp_fc_planner/txt/`, pulled arXiv API metadata (`tmp_fc_planner/meta2026.txt`), queried the GitHub API for repos, stars and licenses, read code in `research/repos/`, and fetched the vendor pages. A claim counts as "confirmed" only if I saw it in a primary source. Wrong claims in the body are struck through with an inline `[corrected: …]`.*

### A. Confirmed claims (primary source seen)

**Pigey** (arXiv 2607.21725: Galanti, Shah, Dao; submitted 2026-07-23)
- Real DROID, 30 tasks × 5 trials, Table 3: π0.5-DROID 16.7%, TiPToP 48.7%, Pigey 97.3%.
- Reasoning-limited probes: raw VLA 4.6% vs Pigey 96.9% (§5.2).
- LIBERO-PRO, Table 2: π0.5 12.8%, CaP-Agent0 18.2%, Pigey 53.3%.
- Nine-reasoner sweep (Table 15): 44.3% (GPT-5.5 low) to 53.3% (Opus 4.7). The quote "The reasoner sets the magnitude of the gain, not its sign" is verbatim.
- Simple pick-place: TiPToP 80% vs Pigey 100% (4 tasks × 5 trials).
- Repo: `real/agent.ts` (1017 lines) calls `https://api.anthropic.com/v1/messages` directly. The model is `process.env.CLAUDE_MODEL ?? 'claude-opus-4-7'` with `max_tokens: 4096`. `real/agent-system.md` is exactly 301 lines.
- The tool list `Perceive / Pick / DropAbove / VLARollout / Release / LookAway / LookBack / Done` matches lines 270–294. `Pick` returns `is_grasped`; line 566 overrides `success` when `is_grasped === false`.

**Harness VLA** (arXiv 2607.08448)
- Six analytic primitives plus one VLA primitive (§2.3).
- JSON `vla_act` takes `prompt`, `max_chunks` and `stop` (e.g. `"stop": "object_lifted"`).
- LIBERO-Pro Table 3, 8 cells × 100 trials: πRLinf 50.0, Codex 72.1, Claude Code 82.4. The planners are GPT-5.5 and Opus-4.7 (Appendix line "All three planners, GPT-5.5, Opus-4.7, and GPT-6 Astra").
- Appendix H, Astra at low reasoning effort: 92.63% (741/800) on LIBERO-Pro and 59.20% (148/250) on RoboCasa365.
- RoboTwin C2R: 58.4% (Claude Code) vs 50.4% for the frozen LingBot-VLA.
- Repo `RLinf/RPent`: Apache-2.0, 1,336★.

**CaP-X** (arXiv 2603.22435)
- Venue confirmed: the PDF header reads "Proceedings of the 43rd International Conference on Machine Learning, Seoul … PMLR 306, 2026", i.e. ICML 2026.
- 187 tasks (7 Robosuite + 130 LIBERO-PRO + 50 BEHAVIOR). 7 core tasks × 100 trials per tier. Human reference 88.5%.
- S4 per-model numbers come from `capgym.github.io/data/model_data.json`: Gemini 3 Pro 32.33, Opus 4.5 23.83, GPT-5.2 22.0, lowest Qwen3-235B 4.0. The mean over the 7 closed models is 18.19 (computed).
- CaP-Agent0 ablation (Fig. 8): M4 55, +SL 59, +1M 66, +3M 68. Matches or beats human code on 4/7 tasks.
- BEHAVIOR radio task success: 24% (S3) → 56% (Agent0) vs 36% (human). 25 trials per task.
- CaP-RL (Table 4): sim Cube Lift 25 → 80% (N=100); real lift 24 → 84% and stack 12 → 76% (N=25).
- LIBERO-PRO six-split mean 18.17% (computed from Table 2).
- Repo `capgym/cap-x` (MIT, 833★): `capx/integrations/franka/control_reduced.py` exposes `segment_sam3_text_prompt`, `point_prompt_molmo`, `plan_grasp` (docstring "Contact-GraspNet"), `solve_ik` (PyRoKi) and `move_to_joints`. `capx/skills/library.py` defines `class SkillLibrary`.

**Agent as Policy** (arXiv 2609.12541: Notre Dame / UCSD / SDSU)
- Every number in the note's row matches Tables 1–3:
  - 4-pair assembly 8/10, 37.2 min, $16.62. Pyramid 10/10, $11.69. Dice 10/10, $21.07. Towel sequential 5/5, 50.8 min, $24.14. Towel simultaneous 3/5.
  - 2-pair assembly: Astra high effort 5/5, 9.2 min, $4.47. Sol 5/5, $3.94. Terra 1/5. Luna 0/5. Opus 5 5/5, 22.2 min, 12.40 M tokens, $9.75. Fable 5.1 3/5, 27.7 min, $11.30.
  - Baselines: GaP 0/25, ASPIRE 3/25, AGP 23/25.
- Repo `agent-as-policy-2026/agent-as-policy` (Apache-2.0). `agp/README_interface_real.md` confirms:
  - Speed caps: ≤0.03 m/s Cartesian, ≤10°/s tool rotation, ≤20°/s joints.
  - Workspace clamp, and a single move may not travel more than 0.25 m.
  - Error codes `IK_FAILED`, `JOINT_LIMIT`, `SETTLE_MISS`. Commands `frames`, `deproject`, `move_ee`, `move_delta`, `move_joints`, `gripper`.
  - Hardware: D405 wrist camera, BRIO overhead camera, Mink IK (`hardware-bridge/src/agp_yam_bridge/kinematics.py`).
  - Budget: "at most 400 counted commands and 120 minutes" (60 min for throwing).

**ALRM** (arXiv 2601.19510)
- Claude-4.1-Opus: TaP 93.5% / CaP 92.6%, latency 33.44 → 82.60 s.
- GPT-5: CaP 90.7 > TaP 85.2. Gemini-2.5-Pro gains 13.9 with TaP. Falcon-H1-7B CaP 84.3%.

**Gemini Robotics-ER 2**
- Blog dated "Jul 30, 2026": 91.3% accuracy / 0.96 s MAD on moment finding; 57.4% on 5-bucket progress classification; "4x the execution speed".
- Positioning quotes: "high-level brain"; "hands off motor execution to any given lower level vision-language-action (VLA) model"; Spot demo.
- Docs, model IDs: `gemini-robotics-er-2-preview` ("Builds on Gemini 3.5 Flash") and `gemini-robotics-er-2-streaming-preview` (Live API). Limits: 131,072 input / 65,536 output tokens.
- Docs, pointing: format `[{"point": [y, x], "label": …}]` normalized to 0–1000; sample prompt "Point to no more than 10 items in the image."
- Docs, thinking: "use medium for a good balance between latency and performance."
- Orchestration doc: `move(x, y, high)`, `setGripperState(opened)`, `max_steps = 15`, `{"status": "success"}`.
- Repo: `google-gemini/robotics-samples/live-api/{agent,spot,tinybot}` exists (committed 2026-07-29).

**Project Fetch phase 2** (18 Jun 2026)
- Opus 4.7 in Claude Code, adaptive thinking at max effort, about 20× faster than the fastest human team.
- Times: 9 min 35 s vs 181 min (Team Claude) vs 361 min (Team Claude-less).
- The beach-ball efforts were "poorly controlled and … not successful."

**Other 2026 papers (abstracts or tables seen)**
- **SafeHarness** (2609.20822, USC/UCF/UCSB): π0.5 57.8 / 17.1. GPT-6 model-only 6.0 / 50.0 → +skills 31.0 / 59.0 → SafeHarness 71.9 / 87.5. Its ref [38] is the "GPT 6 Astra as an embodied policy" study, and the "almost perfectly" quote is verbatim.
- **WAA** (2609.29964), Gemini 3.7 Flash, 60 episodes per split:
  - LIBERO-Pro: zero-shot 28.9 / seed skills 43.3 / evolved skills 75.6. ASPIRE 72.0. Show-Harness rerun 6.7.
  - Qwen3.5-9B fine-tune: 1.7 → 43.3% OOD. Skills are curated by GPT-5.5.
- **VIA** (2607.11119): overall success from 60% (Codex-5.5) to 88% (CC-Fable); 96.7% on 3 LIBERO-Goal tasks; 100% on the 7-block rainbow; text waypoint demo takes CC-Opus 77 → 100%. MCP tool names verified (Table 1).
- **ASPIRE** (2607.00272): "up to 77 points" on LIBERO-Pro, "up to 72 points" on handover, up to 32% on BEHAVIOR-1K. LIBERO-Pro Long 31% vs 4%. Opus 4.6 1M in sim; Codex GPT-5.5 reasoning-xhigh on the real robot.
- **TiPToP** (2603.09971): Gemini Robotics-ER 1.5 called once; cuTAMP; joint impedance control. Multi-step scenes 6/7 won; color cubes 9/10 vs 0/10.
- **ENPIRE** (2606.19980): 99%. Gym-PushT 95% in about 2 h for Claude Code (Opus 4.7 High) and Codex (GPT-5.5 xhigh); Kimi K2.6 took twice as long. Pin insertion with 1 → 8 agents: >1.5 h → ~40 min.
- **RoboClaw**: +25% success, −53.7% human time.
- **RATs**: +20.6 / +17.0 / +8.9 / +8.8.
- **ETP**: +31% EB-ALFRED, +36% EB-Navigation; gains "limited for execution-type capabilities".
- **RoboHarness**: 135 real trials.
- **DualManip** (2609.31112, 2026-09-25): ~46× faster geometric adaptation.
- **UniManip**: +22.5 / +25.0.
- **ZeroDex**: Jun 2026.
- **SEAM**: listed in CVF open access, CVPR 2026.
- **RACAS**: GPT-4.1 / GPT-4.1-mini.
- **Show-Harness** (NUS Show Lab): Gemini-3.1 Pro (medium thinking) zero-shot; Qwen3.5-2B fine-tuned; 10 trials per task.
- **GaP**: MORSL with 51 skills, Skill.md conventions.
- **LIBERO-PRO**: >90% → 0.0% collapse (abstract).

**2022–2025 entries spot-checked against the PDFs (all confirmed)**
- **SayCan**: 84/74, 81/60, FLAN 70/61; 101 instructions.
- **Inner Monologue**: 12.5/0/0 → 25/25/44.4.
- **CaP**: Table IV NL 35, CoT 58, code 98. Long-horizon 80.0 vs CLIPort 0.0 vs NL planner 64.0. Spatial-geometric 62.0. HumanEval 39.8.
- **VoxPoser**: 88/70 vs 24/0; 65.0 vs 17.5.
- **CoPa**: 63/18/11/46/37; VoxPoser's "85 hand-crafted examples" → "almost complete failure" with 3.
- **ReKep**: 44.3/68.6/10.0 and 26.7/46.7/6.7; point tracker is the largest error source.
- **KALIE**: 12/75, 32/75, 64/75.
- **MALMM**: 0.09/0.17/0.50/0.81.
- **OmniManip**: 68.3/51.7 and 61.7/45.0.
- **A0**: 62.50 / 43.75 / 53.75 / 33.75.
- **Agentic Robot**: 79.6%, +24%, Qwen2.5-VL-3B verifier.
- **Hi Robot**: PaliGemma-3B; ">40% higher instruction accuracy than GPT-4o".
- **Others**: Code-as-Monitor 28.7 / 31.8 (CVPR 2025 per arXiv comment). AHA 10.3 / 21.4. RoboFAC 34.1 / 29.1, 9,440 trajectories / 78,623 QA. RoboPoint 21.8 / 30.5 / 39.5. SoFar 48.7 / 74.9. HAMSTER 20% / 50%. Text2Motion 82 vs 13. REFLECT ~80%. EmbodiedBench 1,128 tasks, GPT-4o EB-Man 28.9.
- **Star counts** (GitHub API, 2026-10-01): RoboOS 627, PromptCraft 2,118, Voyager 7,236, ReKep 986.

### B. Corrections (14). Each is also fixed inline above.

1. **SafeHarness vs AEGIS "+6.5 / +27.0"** → these are *relative* gains. The absolute margins are **+4.4 pp TSR (71.9 vs 67.5) and +18.6 pp CAR (87.5 vs 68.9)** (2609.20822 Table 1).
2. **EmbodiedBench "Errors: planning 55%, reasoning 41%, perception 4%"** → that breakdown is for the high-level **EB-ALFRED**. For the low-level **EB-Manipulation** the paper gives planning 44% and perception 33% (2502.09560 §5).
3. **ALRM caveat "mock simulator with Gazebo-provided grasp poses"** → the evaluation APIs "return placeholder poses". Success is decided by an **LLM-as-judge panel (GPT-4.1, Claude-Sonnet-4, Gemini-2.5-Flash)** scoring the emitted code or tool calls. Gazebo was used only to validate the ground truth (2601.19510 §IV).
4. **ALRM "33–83 s per task"** → that range is Claude-4.1-Opus only. Across models the latency spans **24.89–161.73 s** (GPT-5 CaP 145.59 s).
5. **"Pigey and TiPToP use Gemini-ER … not as the planner"** → Pigey's LIBERO-PRO sweep runs **Gemini Robotics-ER 1.6 as the reasoner (48.0%)** (Table 15).
6. **Project Fetch quote "did not involve low-level control"** → the actual text is "none of the tasks in these experiments implicate the more challenging, low-level elements of robotic control." A human also approved commands and task transitions.
7. **"AGP and Project Fetch use controller-level clamps and an e-stop"** → true for AGP only. Neither Project Fetch post mentions an e-stop or clamps; in phase 1 the organizer physically grabbed the robot and powered it off.
8. **"AGP's non-blocking controller"** → not found in the paper or repo (UNVERIFIED, likely wrong). AGP moves are blocking and return the achieved pose. Fast motion uses a buffered `run_program` (50 Hz interpolated joint and gripper knots).
9. **ASPIRE code "not found"** → `NVlabs/ASPIRE` (Apache-2.0, created 2026-07-08, 228★).
10. **Show-Harness code "–"** → `showlab/Show-Harness` (Apache-2.0, ~497★).
11. **RATs code "–"** → `Playful-RATs/RATs` (Apache-2.0).
12. **RACAS code "–"** → `github.com/janprz11/robot-agnostic-control` (cited in the paper).
13. **MALMM code "not found on GitHub"** → `malmm1/MALMM`.
14. **CoPa "no code found"** → `HaoxuHuang/copa` (official implementation, 111★).

**Minor nuances (not counted as corrections)**
- **AGP cost range.** "USD 4–24 per real task" covers only the Astra and Sol means; Terra's single success cost $1.98.
- **Pigey 80 → 100%.** The comparison is N = 20 per arm (4 tasks × 5 trials). The paper attributes it to the closed verify-and-retry loop, not to `is_grasped` alone.
- **Pigey tool set.** The LIBERO-PRO interface is a different set: `Perceive, Grasp, Place, VLARollout, VerifyCandidate, GoHome, Release`.
- **CaP-X "Gemini 3 Pro from ~24% (S3) to 68%".** The 68 needs the 3-model ensemble (GPT-5.2 + Opus 4.5 + Gemini 3 Pro); Gemini alone reaches 66.
- **GaP code.** There is a code repo, not just a page: `graph-robots/graph-as-policy` (Apache-2.0, 154★).

### C. Unverifiable or not independently re-derived (treat with care)

- **CaP-X closed-model averages S3 21.4 / S2 36.5 / S1 56.9.** These are read off Fig. 3 bars; I could not find them as text or JSON. Only S4 = 18.19 was reproduced. `model_data.json` lists 6 tasks while the paper says 7.
- **MOKA per-task means (72.5 / 61.3 / 73.8 → 93.8, Octo 87.5).** Table I does not survive text extraction. Not re-derived.
- **PIVOT navigation means** (computed) and **OmniManip viewpoint sweep** (ReKep 0/10 at 25°). Not re-checked.
- **ReKep "re-solve ~10 Hz, tracking 20 Hz".** Not re-checked.
- **Units.** UniManip "+22.5 / +25.0" and ETP "+31% / +36%" are stated in the abstracts as "%". Whether they are pp or relative is not specified.
- **Anthropic Project Fetch robot model** is not named in either post.

### D. Important entries the note MISSED (in scope)

**D1. Claude Plays Robotics** — Anthropic Frontier Red Team, 9 Jul 2026, https://www.anthropic.com/research/claude-plays-robotics (Berman, Ilie, Deng, Freeman). This is the most directly relevant Anthropic datapoint for an Opus-backbone harness, and the note omits it.
- **Setup.** Claude runs through the Anthropic Agent SDK with built-in tools disabled, so only a robot action server is exposed. Compared models: Opus 4 through 4.7, Mythos Preview, Sonnet 4.6, GPT-5.4/5.1, Gemini 3.1 Pro / 2.5 Pro, Kimi K2.6, Qwen 3.6+.
- **Embodiments.** All MuJoCo except one: Franka on LIBERO-style kitchens, Unitree G1 and Go2, plus one real Go2.
- **Control modes:**
  - direct 7-D end-effector commands;
  - code `controller(obs) -> action`;
  - RL (the model writes the reward and a policy capped at 200k parameters, trained with PPO);
  - VLA supervision, where the model accepts, edits or replaces **MolmoAct** actions.
- **Results:**
  - Direct LIBERO full-task success is only 0–5.5%.
  - A "cursor" query tool lifts Mythos Preview from 6% to 32% on a 10-task subset.
  - With VLA supervision, "every tested model still performs substantially worse than MolmoAct does on its own". Mythos over-overrides the VLA.
  - No model stood the G1 up even once.
- **Latency.** "Real-time control would require roughly 83 Hz; current non-reasoning inference runs at ~0.2-0.4 Hz". Opus 4.6/4.7 reasoning turns took 15–60 s, with tails of 60–180 s. The simulator was paused between calls.
- **Code.** Promised at `github.com/safety-research/embody`; not released at publication.

**D2. CodeActionBench** (arXiv 2609.33807, 27 Sep 2026; repo cloned at `research/repos/CodeActionBench`). An agentic Code-as-Policy benchmark.
- **Design.** 25 tasks. No specialist perception or grasp modules and no privileged state. A shared robot API gives RGB, calibrated geometry and bounded motion. A hidden physical-outcome verifier scores results.
- **Scale.** 9 configurations × 75 attempts = 675 attempts; 184/675 succeed overall.
- **Results:**

  | Configuration | Success (attempts) | Tasks solved at least once |
  |---|---|---|
  | GPT-6 Astra (Codex CLI) | 73.3% (55/75) | 22/25 |
  | Claude Opus 5 (reference harness) | 49.3% | 19/25 |
  | Claude Opus 5 (Claude Code) | 45.3% | 14/25 |
  | Gemini 3.6 Flash | 20.0% | – |
  | GPT-5.6 Sol | 16.0% | – |
  | Qwen 3.8 Max | 14.7% | – |
  | Kimi K3 | 12.0% | – |
  | Claude Sonnet 5 | 2.7% | – |

- **Cost.** Astra's total cost for 75 attempts was $252.87.
- **Failure modes.** Spatial alignment, object retention, and completion judgment.

**D3. Zetta** (arXiv 2608.16590, 17 Aug 2026). A closed-loop harness that keeps the LLM out of the online loop.
- **Design.** Evolutionary agents run only offline. They evolve code-based runtime critics and recovery skills around a frozen VLA: π0.5 for LIBERO-Pro, GR00T N1.5 for RoboCasa.
- **Results.** The headline "90.8% LIBERO-Pro" is the **Goal suite only** (T 92.5 / S 89.0). Over Goal + LIBERO-10 (T/S) the average is 71.13% vs 32.00% for π0.5. RoboCasa reaches 93.56% on 18 Atomic-Seen tasks vs 73.56% for GR00T.
- **Protocol.** Held-out seeds 1–20.
- **Speed.** 11.1× lower latency than RPent / Harness VLA (392 s per episode for RPent).
- **Placement.** It belongs in §4.3's scoreboard, with its memory and evolution regime stated.

**D4. Guava** (arXiv 2606.18363, 16 Jun 2026).
- **Design.** "Distilling Frontier VLMs into a Compact Agent through a Robotic Manipulation Harness".
- **Findings.** A design-space study names three ingredients: iterative perception-reasoning-action loops, semantic action abstractions, and multimodal observations.
- **Distillation.** Into a 4B open model using fewer than 2K sim trajectories, reported "comparable to frontier proprietary models".
- **Relevance.** This is the distillation path the user needs.

**D5. Thea, "Towards the Harness of Embodied Agents"** (arXiv 2608.11246; repo `research/repos/EIT-HAI_Thea`).
- **Design.** Robot capabilities are exposed as tools.
- **"Scene Graph as Context"** gives the agent a persistent symbolic world state.
- **"Evaluation as Exit Codes"** detects termination, judges success, and diagnoses failure causes.
- **Decision model.** GPT-5.5 at high reasoning effort.

**D6. OpenETA / "ETA: A New Agentic Paradigm for Embodied Tasks"** (arXiv 2608.03924).
- **Loop.** A Planner makes one Tool call per step through an Interface and a World.
- **Memory.** Auditable memory and replayable trajectories.
- **Codex plugin.** Exposes only `observe`, `mark_point` and `move_to`.

**D7. EmbodiedSWE** (arXiv 2609.27308).
- **Benchmark.** EmbodiedSWE-Bench: coding agents on contact-rich, deformable and long-horizon tasks, with up to 30 min of interaction.
- **Data generation.** EmbodiedSWE-Gen expands one verified agent solution into diverse VLA training demonstrations.
- **Result.** A VLA trained only on agent-generated sim demonstrations completes a long-horizon real task. This is a data path for a light action head.

**D8. HoloAgent-0** (arXiv 2606.23565, Horizon Robotics).
- **Design.** An Embodied AgentOS compiles language into skill graphs, schedules resources, monitors execution, and re-plans or asks for clarification.
- **Memory.** Adds 3D spatial memory.

**D9. EmbodiedSkills** (ZJU; `research/repos/ZJU4EmbodiedAI_EmbodiedSkills`; numbers are README-only, no paper located).
- **Loop.** A VLM AgentLoop with six stages: observation, planning, preflight, bounded execution, verification, recovery. Uses a π0.5 backend.
- **Results.** RoboTwin 2.0 is 86.20% vs 82.74% for π0.5, over 50 tasks × 100 episodes.
- **Ablations:**
  - removing intermediate verification drops success to 48.2%;
  - full-instruction instead of semantic subtasks gives 34.4%;
  - one action chunk per subtask gives 19.5%.
- **Relevance.** This is the strongest verification ablation in the space.

**D10. Gemini Robotics 1.5 agentic system** (arXiv 2510.03342; cited in Sources but not discussed).
- **Setup.** GR-ER 1.5 orchestrates the GR 1.5 VLA as a tool.
- **Result.** Against Gemini 2.5 Flash as orchestrator, total long-horizon failures fall from 44.5% to 22%: planning failures 25.5% → 9%, success-detection failures 6% → 4%. Progress roughly doubles on complex tasks.
- **Relevance.** This is the vendor's own evidence that an embodied-reasoning-tuned orchestrator beats a general model.

**D11. Frontier-model-as-policy evaluations on RoboDojo** (borderline scope, but they bracket what the harness adds).
- **"An Unexpected Robot Policy: Early Evaluations of GPT-6 Astra on RoboDojo"** (arXiv 2609.24170; `research/repos/RoboProbe`). Over 42 tasks and 2,100 trials, Astra achieves 22.48% success (Score 28.97). GPT-5.5 gets 0.88% and DeepSeek-Flash 1.92%. One-shot demonstrations bring no aggregate benefit.
- **RoboICL** (arXiv 2609.34261; `research/repos/Mosi-AI_RoboICL`):
  - 30-task Overall score 50.64 vs 33.68 for the strongest baseline.
  - Real tasks go 14.45 → 63.33 → 78.89 progress at 0, 1 and 3 shots.
  - Jev-gated action reuse cuts Astra calls by 33–48%.

**D12. Missing classic lineage entries** (titles and dates verified on arXiv; numbers not re-checked here):
- Socratic Models (2204.00598, Apr 2022).
- LLM+P (2304.11477): LLM → PDDL → classical planner, the ancestor of TiPToP's goal grounding.
- KnowNo / "Robots That Ask For Help" (2307.01928): conformal uncertainty for LLM planners. Relevant to when the harness should ask a human.
- Instruct2Act (2305.11176).
- Language to Rewards (2306.08647) and Eureka (2310.12931): LLM-written reward code, the precursor of ENPIRE and Claude Plays Robotics' RL mode.
- RT-Trajectory (2311.01977): trajectory sketches as the intermediate representation, the precursor of HAMSTER.
- RoboCodeX (2402.16117) and LRLL (2406.18746): both are listed in Sources but never discussed in the body.
- MolmoAct (2508.07917): the VLA that Claude supervises in D1.

**D13. Additional details missed inside existing entries**
- **Pigey has a second, context-isolated Claude verifier.** `runVerifier()` in `real/agent.ts` (lines ~342–405) uses `max_tokens: 200` and returns strict JSON `{"ok","reason"}` from task + action log + scene geometry, without the agent's reasoning. It runs after every DropAbove / VLARollout / Release and at Done. It **fails open** ("letting through") on API or parse errors. That is a safety-relevant design choice to avoid copying blindly.
- **Gemini Robotics-ER 1.6 preview "will be shut down at the end of August"** (ai.google.dev robotics overview). Pipelines pinned to ER 1.5/1.6, such as TiPToP and the Pigey sweep, need migration to `gemini-robotics-er-2-preview`. The docs' own code sample sets `thinking_level: "high"` even though it recommends "medium".
- **Harness VLA backbone ranking reverses on RoboCasa365.** Codex/GPT-5.5 scores 57.2 vs 48.8 for Claude Code/Opus 4.7 (Table 4).
- **Harness VLA memory regime matters.** Zero-shot (no task memory) on LIBERO-Pro Goal falls to 31.0% (position swap) and 79.0% (task redirect), against 87.0 / 87.0 with memory (Table 5).
- **VIA's "CC-Opus" is Opus 4.8.** All VIA agents ran at xhigh effort. Per-agent overall: CC-Opus 70%, CC-Fable 88%, Codex-5.5 60%, Codex-5.6-Sol 62%.
- **ENPIRE's fleet is eight bimanual YAM stations.** This is directly relevant to the user's YAM-based sources.
- **TiPToP overall result.** 98/165 vs 55/165 for π0.5-DROID (task progress 74.6 vs 52.4) over 28 scenes. π0.5-DROID is better on simple pick-place (27/40 vs 22/40). TiPToP executes open-loop.
- **CaP-RL trained with the privileged S1 state APIs.** The sim→real numbers are not from low-level, perception-grounded tiers.

Sources added in this pass:
- https://www.anthropic.com/research/claude-plays-robotics
- https://www.anthropic.com/research/project-fetch-robot-dog
- https://arxiv.org/abs/2609.33807
- https://arxiv.org/abs/2608.16590
- https://arxiv.org/abs/2606.18363
- https://arxiv.org/abs/2608.11246
- https://arxiv.org/abs/2608.03924
- https://arxiv.org/abs/2609.27308
- https://arxiv.org/abs/2606.23565
- https://arxiv.org/abs/2609.24170
- https://arxiv.org/abs/2609.34261
- https://github.com/google-gemini/robotics-samples/tree/main/live-api
- https://github.com/NVlabs/ASPIRE
- https://github.com/showlab/Show-Harness
- https://github.com/Playful-RATs/RATs
- https://github.com/graph-robots/graph-as-policy
- https://github.com/malmm1/MALMM
- https://github.com/HaoxuHuang/copa
- https://openaccess.thecvf.com/content/CVPR2026/papers/Tang_Rethinking_Intermediate_Representation_for_VLM-based_Robot_Manipulation_CVPR_2026_paper.pdf
