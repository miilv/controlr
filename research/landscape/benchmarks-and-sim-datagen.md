# Benchmarks, real-robot eval methodology, and LLM-driven sim/data generation

**Purpose.** This is a landscape note for a harness that uses a frontier LLM API (e.g. Claude Opus) as the backbone, possibly with a light learned action head.

**Coverage.** Written 2026-10-01. Every 2025–2026 claim below links to a page or arXiv HTML that I opened on that date.

**Conventions.**
- **[derived]** marks my own computation.
- **UNVERIFIED** marks a claim I saw only in a search snippet.
- "AB" means A/B comparison.

**What other notes cover.** Teammates deep-dive RoboDojo, EmbodiedSWE, GRID auto-engineering and the Robocurve/GPT-6 Astra reports in `research/sources/*.md`. Here I give context lines only for those.

---

## 0. TL;DR

1. **No public leaderboard measures "frontier API model as robot policy" in a way you could trust for a Claude-backbone decision.** There are three relevant 2026 benchmarks:
   - **CodeActionBench** (2609.33807): 9 model+harness configurations, 675 attempts in RoboTwin/SAPIEN sim.
   - **CaP-Bench** (2603.22435): 12 models, 100 trials per tier.
   - **RoboDojo**: agent rows on a VLA leaderboard.

   Real-robot comparisons are tiny. Examples: StationeryBench 100 trials per model, unblinded; ClapboardBench 11 runs; AGP 5–10 trials per configuration.
2. **Claude-specific signal (Opus 5).**
   - CodeActionBench: Opus 5 is the best model under the shared reference harness, at 37/75 = 49.3%. GPT-6 Astra under Codex CLI scores 55/75 = 73.3%.
   - Running Opus 5 under Claude Code instead gives 34/75 (45.3%) and coverage drops from 19 to 14 tasks. ~~The harness matters about as much as the model.~~ [corrected: the data do not show this. Within Opus 5 the harness changes success by 4.0 pp (37/75 vs 34/75, Fisher p=0.74; the paper says the two configurations "differ by only 4.0 percentage points in success rate but by five tasks in coverage", 2609.33807 §4.1). The Astra–Opus gap is 24 pp (p=0.004), and that comparison confounds model with harness (Codex CLI vs reference). §4.2 below already says the harness difference is not established.]
   - Gemini Robotics ER 2's own chart: Opus 5 gets 67.2 on ERQA (ER 2: 78.5) and **37.1%** on 5-way progress classification (ER 2: 57.4%).
3. **Completion judgment is a first-class failure mode.** In CodeActionBench, when Opus 5 (reference harness) failed, it claimed success 13/38 times. GPT-6 Astra overclaimed 3/20.
   - Across all 9 configurations, 70.3% of failed attempts end with no claim at all.
   - Design implication: never let the LLM's `done()` be the success signal. Add an independent verifier or critic.
4. **Learned-policy benchmarks are saturated or brittle.**
   - LIBERO: VLAs score above 90% but fall to **0.0%** under LIBERO-PRO perturbations. LIBERO-Plus drops them from 95% to below 30%.
   - Long-horizon and compositional suites remain open. RoboCasa365 composite-unseen is ≤4.4% for every VLA. The BEHAVIOR 2025 winner had 12.4% full success and 0.26 Q-score.

   These are the gaps an LLM planner over a skill head could plausibly fill.
5. **Real-robot statistics.** Best practice is:
   - blind, randomized, matched-initial-condition AB;
   - 50 real and 200 sim rollouts per task per policy per condition (TRI LBM);
   - Beta posteriors plus Bonferroni-corrected sequential tests, summarized with Compact Letter Display (CLD).

   Distributed alternative: RoboArena, with double-blind pairwise trials and a task-aware Bradley-Terry ranking.

   Scale check [derived]: with 20 trials per arm you can only detect about a 40-point success-rate gap at 80% power.
6. **Synthetic data does train action heads.**
   - MolmoBot: 1.7–1.8M sim trajectories gave **79.2% zero-shot real** pick-and-place over 120 trials, versus 39.2% for π0.5.
   - Sim+real co-training: +38% on average.
   - RoboTwin 2.0: +367% relative with 10 real demos.
   - EmbodiedSWE: a coding agent's sim program, expanded to 500 demos, lifted π0.5-DROID from 0/10 to 2/10 real.

   The productive pattern is **LLM as engineer/teacher offline** (task code, verifiers, scene and reward generation, data multiplication). It is not the LLM in the loop at control rate.
7. **For Ilia's harness.** Use the LLM for planning, tool selection and recovery. Train a light head (skill-conditioned or target-conditioned servo) plus a **learned progress/success critic** on LLM-generated, replay-verified sim data. Report results with a pre-registered, blinded, interleaved AB protocol.

---

## 1. What each benchmark family actually measures for an API-backbone harness

| Level | What is tested | Examples | Harness relevance |
|---|---|---|---|
| A. Embodied reasoning QA | Perception and spatial/temporal reasoning only; no control | ERQA, ERQA-Plus, RoboProcessBench, Embodied-BenchForge OE-Track, ER 2 metrics | Model selection for the planner and critic |
| B. LLM agent in sim with abstracted actions | Planning, tool use, code, recovery | EmbodiedBench, VLABench (VLM/workflow mode), ALRM, CaP-Bench, CodeActionBench, EmbodiedSWE-Bench, RoboDojo `AGENT` rows | Closest to the harness itself; results depend on the API surface and time model |
| C. Learned-policy sim suites | Visuomotor generalization | LIBERO/-Plus/-PRO, SimplerEnv, RoboCasa365, BEHAVIOR-1K, ManiSkill3, RoboTwin 2.0, Colosseum | Where a light action head is measured; also data sources |
| D. Real robot | Everything, including latency and safety | RoboArena, StationeryBench, ClapboardBench, AGP, AgiBot World Challenge, RoboDojo-Real | The only ground truth; very low n |

**The time model matters for Level B.** RoboDojo counts control steps, not wall-clock time, and pauses the sim while the LLM thinks (see the teammate `robodojo.md` note). CodeActionBench charges a tool-call budget plus a simulated-time budget. StationeryBench gives the agent a 900-step cap and a 20-LLM-call budget, against 1,200 steps for the VLA.

A harness that is fast in "sim steps" can still be unusable in wall-clock time. Some examples:
- AGP: mean ~~9–50 min~~ [corrected: 9.2–50.8 min; per-configuration means over successful trials only (2609.12541 Tables 1–2). Towel folding (sequential) is 50.8 min. Throwing is 17.71 min per whole session] per real trial.
- EmbodiedSWE: median ~~39–103 min~~ [corrected: 39–119 min. 2609.27308 Table 8 "Median min. to solve": Opus 5 103, Opus 4.8 98, Fable 5.1 73, Sol 58, Terra 119, Astra 39] to solve.
- CodeActionBench: the best configuration has a median of 477 s.

---

## 2. Benchmarks of API models as robot agents (Level A/B/D, 2025–2026)

| Benchmark | Org, date | Setup | Key numbers (exact) | Code | Relevance |
|---|---|---|---|---|---|
| **CodeActionBench** | NTU, 2026-09-27 (2609.33807) | 25 RoboTwin 2.0 tasks, ALOHA-AgileX dual arm in SAPIEN; RGB plus calibration only (no depth or object pose); hidden verifier; fixed seed; 3 attempts per task | Astra/Codex 55/75 (73.3%, 22/25 tasks covered). Opus 5 reference harness 37/75 (49.3%, 19/25). Opus 5 / Claude Code 34/75 (45.3%, 14/25). Gemini 3.6 Flash 15/75. GPT-5.6 Sol 12/75. Qwen 3.8 Max 11. Grok 4.6 9. Kimi K3 9. Claude Sonnet 5 2/75 | `lyhkk/CodeActionBench` (MIT; data "coming soon") | Best controlled comparison; tool surface worth copying (below) |
| **CaP-X / CaP-Bench** | Berkeley/NVIDIA et al., 2026-03-23, ICML 2026 (2603.22435) | 7 Robosuite tasks × 8 tiers (S1–S4 single-turn, M1–M4 multi-turn); 100 trials per tier; 12 models up to Opus 4.5 / GPT-5.2 / Gemini-3-Pro; CaP-Gym ships 187 tasks | Raw RGB interleaved per turn (M2) *reduced* success vs text-only feedback (M1). A "visual differencing" module turning images into text (M3) helped consistently. CaP-RL: GRPO on Qwen2.5-Coder-7B raised sim success from 25/4/30% to 80/44/93% (Lift/Stack/Wipe, n=100) and real Franka from 24/12% to 84/76% (n=25) | `capgym/cap-x` | Evidence for image→text grounding and for RLVR of a small coder |
| **EmbodiedBench** | UIUC et al., ICML 2025 (2502.09560) | 1,128 tasks in 4 environments. EB-Manipulation is VLMBench/CoppeliaSim Franka with discretized 7-D action (100 position bins, 120 rotation bins), YOLO boxes plus pose hints, ≤15 steps, temperature 0, 500×500 images | Best EB-Manipulation average: GPT-4o 28.9. Claude-3.7-Sonnet 28.5, Claude-3.5-Sonnet 25.4. High-level EB-ALFRED: Claude-3.7-Sonnet 67.7 | `EmbodiedBench/EmbodiedBench` (CVPR 2026 FMEA challenge on EB-ALFRED and EB-Navigation) | Low-level control via discretized tokens is weak even with heavy scaffolding |
| **ALRM** | TII, 2026-01-27 (2601.19510) | 56 tasks, Gazebo + ROS + MoveIt; Code-as-Policy vs Tool-as-Policy | Claude-4.1-Opus best closed model: TaP 93.5%, CaP 92.6%. Falcon-H1-7B 84.3% CaP | tiiuae.github.io/ALRM | High-level, MoveIt-backed; ceiling-ish |
| **VLABench** | OpenMOSS, 2024-12-24 (2412.18194) | MuJoCo, 100 task categories, 2,000+ objects; VLMs act through a skill DSL; Progress Score | VLM workflows and VLAs both struggle. VoxPoser reaches Progress Score 30–40 only on basic tasks | `OpenMOSS/VLABench` | DSL-skill API is a template for an LLM→skill-head interface |
| **RoboDojo (context)** | HKU MMLab, 2026-07 (2607.04434) | 42 Isaac Sim tasks, 50 episodes each; 18 real tasks | Agent rows (RoboDojo-run RoboProbe harness): GPT-6-Astra 28.97 / 22.48% SR (#7 of 51). GPT-5.5 1.13 / 0.88%. #1 is PhysicalRSI "Agent + VLA" at 36.27 / 31.38%. Real-world Astra testing was stopped for safety | see `sources/robodojo.md` | Only leaderboard ranking agents next to VLAs; not like-for-like |
| **StationeryBench** | Robocurve, 2026-09-10 | 5 bimanual YAM tasks, 20 trials per task per model; human-graded 0/25/50/75/100 | GPT-6 Astra 7/100 completions (mean progress 46) vs MolmoAct2 0/100 (mean 12). Grader not blinded; different rigs; different horizons | `robocurve/stationerybench` (MIT) | Wilson intervals reported; [derived] Astra 95% CI 3.4–13.7% |
| **ClapboardBench** | Robocurve, 2026-07-31 | One task (clapperboard), one motion per LLM call, 224×224 frames | `claude-opus-5` 2/6 pass (incl. "opus-early"); `gemini-robotics-er-2-preview` 0/5 (1 error) | full wire traces in repo | Rich transcripts; [derived] Fisher p=0.45, i.e. no evidence either way |
| **AGP (Agent as Policy)** | Notre Dame / UCSD / SDSU, 2026-09-11 (2609.12541) | Codex or Claude Code drives a real robot through a documented interface | GPT-6 Astra ≥80% in 7/8 configs. Two-pair assembly, 5 trials each: Opus 5 5/5 (22.2 min, $9.75 mean), Fable 5.1 3/5, Astra 5/5 at every effort level (≈9–10 min, ~$4–5) | `agent-as-policy-2026/agent-as-policy` | Real-world cost/latency data; n=5 |
| **ERQA** | Google DeepMind, 2025-03 | 400 multiple-choice VQA | Feb 2025: Gemini 2.0 Pro Exp 48.3, GPT-4o 47.0, Claude 3.5 Sonnet 35.5. ER 2 page: ER 2 78.5, Gemini 3.6 Flash 73.0, ER 1.6 72.5, Opus 5 67.2, GPT 5.6 Sol 43.2 | `embodiedreasoning/ERQA` | Planner perception proxy; vendor-reported |
| **ER 2 metrics** | Google DeepMind, 2026-07-30 | Vendor chart | Video success detection: Opus 5 81.0, ER 2 82.4. Progress (5-way): Opus 5 37.1, ER 2 57.4. Safety-instruction following: Opus 5 95.9, ER 2 97.9 | API `gemini-robotics-er-2-preview` | Opus is weak at progress estimation, so use a critic |
| **RoboProcessBench** | 2026-06 (2606.13040) | ~58k QA over 260 tasks, 12 process families | Primitive-local progress (T5): best zero-shot ≈34%, close to the random baseline. SFT'd Qwen2.5-VL-7B jumps (e.g. T10 92.5 vs ≤38.4 for every zero-shot model). Closed models tested only up to Claude-Sonnet-4.6 / GPT-5.4-mini | project page | **A small fine-tuned critic beats frontier zero-shot at progress tracking** |
| **Embodied-BenchForge** | 2026-09-11 (2609.13082) | Agentic benchmark construction; OE-Track (6 benchmarks) and IE-Track (220 executable tasks) | OE mean: GPT-5.5 57.67, Claude Opus 4.7 56.61, humans 85.89. IE success rate: GPT-5.5 83.18, Opus 4.7 77.73 | — | Shows LLM agents can *build* benchmarks with verify/repair |

**Interface lessons from these benchmarks.**
- **CodeActionBench tool surface** (`src/codeaction/interface/registry.py:15-37`):
  - Exposed tools: `capture_head`, `capture_wrist`, `capture_evidence_views`, `capture_motion_pair`, `triangulate_correspondence`, `scale_from_gripper`, `grasp_quat_candidates`, `check_tcp_pose_reachability`, `preview_tcp_pose`, `probe_contact_along`, `move_delta`, `reach_tcp`, `set_gripper`, `done`, plus `run_code`. [corrected: this is a subset. `D0_TOOLS` (`registry.py:15-29`) lists 30 tools. It also includes `get_world_frame`, `get_embodiment`, `get_camera_info`, `get_arm_pose`, `get_gripper_state`, `get_robot_state`, `project`, `ray`, `plane_intersect`, `scale_from_object_size`, `compare_tcp_poses`, `draw_marks`, `move_both_delta`, `reach_both_tcp`, `camera_aim_pose` and `get_grasp_contact`. `run_code` is not in `D0_TOOLS`. It is the program-submission channel; the module docstring says the registry is "consumed by MCP and run_code".]
  - A `DENY` set hides `get_object_pose`, `get_depth`, `get_segmentation` and `get_point_cloud` from the agent. [corrected: the full `DENY` set (`registry.py:35-38`) has 7 names. The other three are `get_scene_objects`, `actor_center` and `get_workspace`.]
  - The design separates geometric primitives the agent may use from privileged ground truth only the verifier sees. That is a good template for a Claude tool schema.
- **CaP-Bench.** Abstraction level dominates. High-level human-written primitives raise success but "limit expressivity". Multi-turn execution feedback and image→text visual differencing recover most of the gap.

---

## 3. Learned-policy benchmarks (where a light head would be measured)

| Benchmark | Facts | Status in 2026 | Use for Ilia |
|---|---|---|---|
| **LIBERO** (2306.03310) | 4 suites, 130 tasks, teleop demos | Saturated: VLAs >90% | Smoke test only |
| **LIBERO-Plus** (2510.13626) | 10,030 test tasks over 7 perturbation dimensions (camera, robot init, language, light, background, noise, layout) | Modest perturbations drop VLAs from 95% to <30%; models "tend to ignore language instructions" | Robustness check for a head |
| **LIBERO-PRO** (2510.03827) | Perturbs objects, initial states, instructions, environments | >90% → **0.0%**. OpenVLA and π0 collapse past 0.2 units of object displacement | Coding agents do better here: RHO 45.0% vs π0.5 12.83% (2606.16458); ASPIRE ~~+77%~~ [corrected: "up to 77 points" on LIBERO-Pro perturbation suites, measured against prior *coding agents* (CaP-X, Fu et al. 2026), not against VLAs (2607.00272 §1)] (2607.00272) |
| **SimplerEnv** (2405.05941) | Real-to-sim for Google Robot and WidowX/Bridge; >1,500 eval episodes; introduces the **MMRV** metric | Standard proxy; MMRV reused by AutoEval and RoboArena | Use MMRV when validating any sim proxy |
| **RoboCasa365** (2603.04356, ICLR 2026) | 365 tasks, 2,500 kitchens, 612 h human + 1,615 h MimicGen data; composite tasks drafted from **LLM blueprints** (60 activities) | Multi-task (DP / π0 / π0.5 / GR00T N1.5):<br>• Atomic 15.7 / 36.3 / 39.6 / 43.0<br>• Composite-Seen 0.2 / 5.2 / 7.1 / 9.6<br>• Composite-Unseen 1.25 / 0.7 / 1.2 / 4.4 | Composites are where an LLM planner over a skill head should win |
| **BEHAVIOR-1K Challenge 2025** | 50 tasks, 10k demos (1,200+ h); standard track plus privileged-info track; 18 teams | Held-out: Robot Learning Collective 0.124 success / 0.2599 Q; NVIDIA Comet 0.114 / 0.2514 (both π0.5-based). Best privileged-track team 0.052 / 0.0947 | Long-horizon gap is large |
| **BEHAVIOR Challenge 2026** | 100 tasks, 7 scenes, 20,000 demos (1,950 h); one RGB+D+proprio track; baselines π0.5 and GR00T N1.7; deadline 2026-10-16 | Page lists "LLM-assisted policies" as in scope; no privileged track this year | Candidate venue for a planner+head entry |
| **ManiSkill3** (2410.00425) | GPU-parallel SAPIEN, up to 30,000+ FPS, 12 domains | Active (`mani-skill/ManiSkill`) | Cheap data and RL for a head; FAEA uses it |
| **RoboTwin 2.0** (2506.18088) | 50 dual-arm tasks, 5 embodiments, 731 objects; MLLM codegen with sim-in-the-loop refinement | Codegen success 47.4% (R1.0 vanilla) → 71.3% (R2.0 + multimodal feedback), using DeepSeek-V3 plus moonshot-v1-32k-vision. VLA + 10 real demos: +367% relative; sim-only +228% | Substrate for CodeActionBench and RoboDojo |
| **Colosseum** (2402.08191) | 20 RLBench tasks × 14 perturbation axes | 30–50% drop per axis, ≥75% combined; sim↔real R̄²=0.614 | Perturbation design reference |
| **AgiBot World Challenge** | IROS 2025: 431 teams; Manipulation and World Model tracks. ICRA 2026: 526 teams; Reasoning-to-Action track with real G2 finals | R2A 2026 winner PrismBot (vivo) 43.47 pts | Industry-scale; no API-agent tracks |
| **RoboArena** (2506.18123, CoRL 2025) | DROID platform; distributed double-blind pairwise evaluation | Official board, 2026-10-01: DreamZero 1735 (SD 42.6, 190 AB evals), pi05_droid 1608 (745), pi0_fast_droid 1582 (941). **No LLM-agent entries.** Runs "through Dec 2026" | A Claude harness could in principle be submitted as a DROID policy server; latency would be the issue |

---

## 4. Real-robot evaluation methodology

### 4.1 Trial counts actually used

| Study | Trials | Statistics |
|---|---|---|
| TRI LBM (2507.05331) | 50 real rollouts per task per policy per condition; 200 sim; 1,800 real + 47k sim total | Beta(1,1) posterior violins. Pairwise sequential test (Lai 1988) with Bonferroni correction for 95% global confidence. CLD letters. Welch t-test on task-completion rubrics |
| RoboArena | 612 pairwise AB comparisons across 7 universities; "oracle" ranking from 4,284 rollouts; official board requires 100+ AB evaluations per policy | Bradley-Terry extended with latent task buckets and policy-task offsets, fitted by EM; beats Elo, plain BT and a 17-task × 44-episode standardized evaluation on Pearson and MMRV |
| MolmoBot (2603.16861) | 4 environments × 10 tasks × 3 trials = 120 per policy | Point estimates |
| StationeryBench | 20 per task, 100 per model | Wilson 95% CIs (plotted) |
| CodeActionBench | 3 attempts × 25 fixed seeds = 75 | Coverage plus success; no CIs |
| CaP-RL real | 25 per task | Point estimates |
| AGP | 5–10 per configuration | Min/max ranges |
| RoboDojo-Real | 10 trials per task, 3 blind raters | See teammate note |

### 4.2 Required n [derived]

Two-sided α=0.05, power 0.8, two independent arms. The minimum detectable improvement is:

| Trials per arm | From a 50% baseline | From a 20% baseline |
|---|---|---|
| 10 | +47.5 pts | +59 pts |
| 20 | +39 pts | +42.5 pts |
| 50 | +27 pts | +26 pts |
| 100 | +19.5 pts | +18 pts |
| 200 | +14 pts | +12.5 pts |

Worked examples:
- CodeActionBench Opus 5, reference harness vs Claude Code: 37/75 vs 34/75, Fisher p=0.74. The harness difference is not established.
- Astra vs Opus 5 (reference): 55/75 vs 37/75, p=0.004. This is real, but attempts are clustered by task and seed, so effective n is smaller than 75.
- Most "X beats Y" claims at n≤10 are noise. Kress-Gazit et al. (2409.09491) show 13/20 vs 14/20 is indistinguishable under a uniform prior.

### 4.3 Blinding, interleaving, resets

**Blinding and ordering.**
- TRI: evaluators blind to the policy, policy order randomized, initial conditions matched "within human error".
- RoboArena: double-blind, back-to-back AB on the same scene.
- StationeryBench, by contrast, discloses an unblinded grader and different rigs per model.

**Sequential testing.** STEP (2503.10966) gives a sequential test with Type-I control. It saves up to 32% of trials and avoids the p-hacking you get from "run more trials until significant". Running Barnard's test repeatedly inflates the false-positive rate.

**Reset variance.**
- TRI uses hardware displays to reproduce initial conditions.
- AutoEval (2503.24278) replaces humans with a learned success classifier, a learned reset policy and safety checks. It closely matches human evaluation (Pearson and MMRV).
- Note that OpenVLA's original evaluation needed >2,500 rollouts and >100 human-hours.

**Proxies.**
- SimplerEnv (MMRV).
- Veo world simulator (2512.10675): validated against 1,600+ real evaluations of 8 Gemini Robotics checkpoints.
- RoboWorld (2607.01060): Pearson 0.989 and Spearman 0.970 vs the RoboArena ranking for 8 policies.
- RobotArena ∞ (2510.23571): real video → sim, with VLM plus crowd pairwise scoring.
- REALM (2512.19562) and WorldEval (2505.19017).

All of these were validated on **VLA** policies. None is validated for slow, tool-calling LLM agents.

### 4.4 LLM-agent-specific pitfalls

1. **Harness confound.** "Stop Comparing LLM Agents Without Disclosing the Harness" (2605.23950) argues that harness variance can exceed model variance and even reverse rankings. CodeActionBench's Opus/Claude Code numbers are an instance (coverage 19 vs 14).
2. **No seeds.** API sampling is nondeterministic, and CodeActionBench sets no sampling seed. Repeat attempts per fixed scene and report coverage (≥1/3) and consistency (3/3) separately.
3. **Self-report bias.** Hide the verifier verdict from the agent and score claims separately: done-rate, overclaim, underclaim, absent.
4. **Pin versions and effort.** Record exact IDs and effort, e.g. `claude-opus-5` with adaptive thinking at High and a 128,000-token output limit (CodeActionBench Table 8).
5. **Account for wall-clock, cost and safety.** RoboDojo halted real Astra trials after hardware damage. SafeHarness (2609.20822) shows coding agents violate prompt-stated obstacle constraints "in most cases". Their obstacle-aware route harness reaches 71.9% success with 87.5% collision avoidance.

### 4.5 Recommended protocol for the Claude harness

- **Pre-register:** tasks, initial-condition sheet (photo overlay), rubric (0/25/50/75/100) and success predicate.
- **Interleave** harness variants per initial condition (ABBA order), with a grader blind to the variant (video-only grading).
- **Trial count:** ≥50 trials per arm per task for headline claims. Use STEP to stop early.
- **Report:** Wilson or Beta intervals plus CLD; tokens, $ and wall-clock per trial; claim calibration; intervention and e-stop counts.
- **Proxy validation:** check that the sim proxy (RoboTwin/ManiSkill clone) ranks harness variants the same as real, using MMRV, before trusting it.

---

## 5. LLM-driven sim and data generation

| System | Date / org | Mechanism | Key result | Code |
|---|---|---|---|---|
| **GenSim** (2310.01361) | 2023, L. Wang et al. | GPT-4 writes simulation task code (expands an existing CLIPort-style benchmark), goal-directed or exploratory | Benchmark grown 10× to 100+ tasks; real long-horizon transfer +25% | yes |
| **GenSim2** (2410.03645) | 2024 | Multimodal/reasoning LLMs write articulated tasks; planning and RL solvers produce demos; PPT policy | Up to 100 articulated tasks / 200 objects; sim+real co-training +20% over real-only on 8 real tasks | yes |
| **RoboGen** (2311.01455) | 2023, CMU / UMass / MIT-IBM | GPT-4 propose→generate scene→decompose→choose RL / motion planning / trajectory optimization | "Endless stream" of skills; mostly sim | yes |
| **Gen2Sim** (2310.18308) | 2023, Katara, Xian, Fragkiadaki | Image→3D assets, LLM physics params, task decomposition and reward code | Long-horizon RL where non-decomposed reward fails | yes |
| **Eureka / DrEureka / Eurekaverse** (2310.12931 / 2406.01967 / 2411.01775) | NVIDIA/UPenn | LLM evolves reward code, then domain-randomization ranges, then environment curricula | Eureka beats humans on 83% of 29 environments (+52% normalized); DrEureka transfers quadruped yoga-ball walking | yes |
| **Text2Reward** (2309.11489) | 2023 | Dense reward code from text | ≥ expert rewards on 13/17 manipulation tasks | yes |
| **Holodeck** (2312.09067) | 2023, AI2 | GPT-4 spatial constraints → layout optimization over Objaverse | Scene generation for navigation | yes |
| **MimicGen / DexMimicGen** (2310.17596 / 2410.24185) | NVIDIA | Segment-wise transform-and-replay of human demos (no LLM) | 50K demos from ~200 human demos; 21K from 60; GR00T N1 generated 780k trajectories (≈6,500 h) in 11 h | `NVlabs/mimicgen` |
| **RoboVerse / MetaSim** (2504.18904) | 2025 | Simulator-agnostic config; multi-sim, cross-embodiment retargeting | ~500k trajectories, 276 task categories, ~5.5k assets | yes |
| **Genesis** | Dec 2024 → `Genesis-Embodied-AI/genesis-world` v1.4.3 (2026-09-30) | Multi-solver physics in Python; generative layer partially released | No canonical paper; GitHub only | Apache-2.0 |
| **RoboTwin 2.0 codegen** | 2025 | DeepSeek-V3 writes expert code; a VLM observer verifies | 71.3% average codegen success | yes |
| **AnyTask** (2512.17853) | RAI Institute, 2025-12 | LLM-orchestrated task/scene generation; ViPR (TAMP + VLM refine), ViPR-Eureka, ViPR-RL | Behavior cloning on generated data: 44% average real success | site |
| **SAGE** (2602.10116) | NVIDIA, 2026-02 | Agentic scene generation with generator/critic loop; SAGE-10k | Policies "exhibit clear scaling trends" | site |
| **EmbodiedGen V2** (2607.07459) | Horizon Robotics, 2026-07 | Sim-ready assets, task worlds, "Vibe Coding" agent interface | 83.3% of task worlds usable without manual edits; RL sim 9.7%→79.8%; real 21.7%→75.0% | `HorizonRobotics/EmbodiedGen` |
| **V-CAGE** (2601.15164) | 2026-01 | Collision-aware scene instantiation; VLM rejection sampling after each subtask for "silent failures" | Better downstream success than unverified data | — |
| **ARSTAG** (2609.24563) | B. Li, Y. Zhang, C. Liu, 2026-09 | One RGB image + instruction → agentic Real2Sim scene, demos, randomization | π0.5 trained on its data: 74.6% average real success on 7 tasks | site |
| **BLAZER** (2510.08572) / **LLM Trainer** (2509.20070) | 2025 | LLM planner generates sim demos, then finetunes the LLM / LLM-annotated keypose retargeting from 1 demo | Sensor-based transfer; Franka hardware | — |
| **FAEA** (2601.20334) | 2026-01 | Unmodified Claude Agent SDK, privileged state | LIBERO 84.9%, ManiSkill3 85.7%, MetaWorld 96%; proposed as a trajectory generator | `robiemusketeer/faea-sim` |
| **CaP-RL / RHO / ASPIRE / ENPIRE** | 2026 | RLVR on a coder / search over multi-file "repositories-as-policies" / evolving skill library / coding agents run real-robot reset-train-evaluate loops | RHO: LIBERO-PRO 45.0%, Robosuite 70.0%. ENPIRE: 99% on pin-box, zip-tie, tool use | cap-x; others |
| **EmbodiedSWE-Gen (context)** | ByteDance Seed et al., 2026-09 (2609.27308) | One verified agent program expanded into replay-verified demos | π0.5-DROID + 500 sim demos: 2/10 real vs 0/10 baseline (teammate note) | Apache-2.0 |
| **GRID auto-engineering (context)** | General Robotics, 2026-09-09 | Agent ingests the robot, builds a custom sim (Warp DFSPH + MuJoCo), picks a skill route (composition / BC+DAgger / teleop fine-tune / video-to-sim) | ~4 h to first skill; sysID error 143 mm → 5.7 mm; no success rates | closed |

**Pure-sim data scale for comparison.**
- **MolmoBot** (2603.16861): 1.7–1.8M procedurally generated trajectories, about 660 successful episodes per A100-hour.
  - Zero-shot real: MolmoBot 79.2% vs π0.5 39.2%.
  - MolmoBot-Pi0 (π0 architecture, MolmoBot data) 46.7%, so the data is responsible for much of the gain.
  - **MolmoBot-SPOC**, a *lightweight* edge policy with quantile-binned parallel decoding, reaches 36.6% on a real subset.
- **InternData-A1**: 630k trajectories / 7,433 h, per MolmoBot's comparison table.
- **Sim-and-real co-training** (2503.24361): +38% on average.
- **SIM1** (2604.08544): a deformable-object physics-aligned engine claiming parity at a 1:15 synthetic:real ratio.

---

## 6. What role can this data play for a light action head?

**Evidence-backed roles.**

1. **Skill-conditioned head.** Train small policies for a fixed menu (`grasp(obj)`, `place(target)`, `insert`, `wipe`) on MimicGen or procedurally generated data. The LLM then calls them as tools, as in VLABench's DSL or Gemini ER 2's "VLA as a tool".
   - Evidence that this works: RoboCasa365 atomic tasks reach 43% while composites stay below 10%. That split matches "LLM sequences skills; head executes atomics".
2. **Target-conditioned servo head.** The LLM emits a 3D target or keypoint, and the head closes the last centimetres from wrist RGB.
   - Train it on planner trajectories with injected noise and corrections (EmbodiedSWE's phase-aware noise; DART-style).
   - This addresses CodeActionBench's dominant failures: "spatial alignment, object retention", where failed tasks contain successfully completed motions.
3. **Learned critic head** (success, progress, contact). This is the highest-ROI item.
   - Frontier models are weak at progress: Opus 5 scores 37.1% on 5-way progress; RoboProcessBench T5 is near chance zero-shot.
   - They are miscalibrated on completion (CodeActionBench overclaims).
   - ~~SFT on generated traces lifts a 7B VLM dramatically (RoboProcessBench).~~ [corrected: the dramatic lifts are on recognition families, e.g. T10 current-primitive 33.1→92.5 and T11 33.1→96.5. On primitive-local *progress* (T5), the family that matters for a progress critic, SFT-Qwen2.5-VL-7B reaches only 45.4%. Its base model scores 32.2%, the best zero-shot model 34.4% and random 33.3%. On temporal ordering (T8), SFT does not help: 17.0 vs 17.9 base (2606.13040 Table 3). Treat a learned progress critic as an open problem, not a solved one.]
   - Sim data gives free, exact labels for success, progress and contact.
4. **Teacher-student loop.** The LLM writes privileged-state programs in sim, which are verified, multiplied and distilled into the head.
   - Evidence: EmbodiedSWE, CaP-RL sim→real (Franka 84%/76%), FAEA, ARSTAG 74.6% real.
   - This keeps the expensive API off the control loop and turns API spend into reusable data.

**Caveats.**
- LIBERO-PRO and LIBERO-Plus show heads trained on narrow data memorize layouts. Train with RoboTwin-style 5-axis randomization and evaluate on perturbation suites.
- Small real-demo top-ups matter: RoboTwin 2.0 gets +367% with 10 real demos, versus +228% sim-only.
- UNVERIFIED (search snippet only): 2606.24448 argues generated *videos* are a mismatched supervision source for low-level control. [fact-check: now verified from the arXiv abstract. "Supervise What Survives: Geometry-Guided VLA Adaptation from Synthetic Robot Videos" (Chen et al., 2026-06-23) states "deriving low-level control from generated visuals is a mismatched abstraction". It proposes GRA, which uses generated-video geometry (2D end-effector waypoints) to supervise only the vision backbone and leaves control to real demos.]

---

## 7. Promising vs unpromising

**Promising**
- Tool surfaces with geometric helpers and privileged-state denial (CodeActionBench), plus image→text visual differencing (CaP-Bench M3).
- Hidden-verifier evaluation that scores claim calibration.
- LLM-as-engineer pipelines: RoboTwin 2.0 codegen with VLM verification, V-CAGE rejection sampling, EmbodiedSWE-Gen replay verification.
- Procedural-scale sim data for small heads (MolmoBot, including the SPOC variant).
- Learned success and progress critics trained on sim labels.
- RoboArena-style blinded pairwise evaluation, and TRI-style CLD statistics.

**Unpromising**
- Raw discretized action tokens from an API model (EmbodiedBench EB-Manipulation ≤28.9%).
- Raw image interleaving into coding turns (CaP-Bench M2 below M1).
- Trusting the agent's `done`.
- Prompt-only safety (SafeHarness).
- Single-digit-n real comparisons and unblinded grading.
- Headline LIBERO numbers.
- World-model evaluation proxies applied to agents without validation.

---

## 8. Open gaps / UNVERIFIED

**UNVERIFIED items.**
- The full CaP-Bench per-model numbers: they are in figures only, which I did not extract.
- RoboArena's per-policy hidden ranks 6–8 on the official board.
- Whether the AgiBot 2026 R2A winners used LLM planners (not stated).
- Sim2Real-VLA (ICLR 2026): search-only.

**Real gaps.**
- No benchmark validates sim-step-budgeted agent results against real wall-clock behavior.
- No public real-robot leaderboard admits API agents.
- Opus 5 has never been evaluated under RoboArena's protocol.

---

## Sources

Benchmarks and agents
- CodeActionBench: https://arxiv.org/html/2609.33807 · https://github.com/lyhkk/CodeActionBench
- CaP-X: https://arxiv.org/html/2603.22435 · https://github.com/capgym/cap-x
- EmbodiedBench: https://arxiv.org/html/2502.09560 · https://embodiedbench.github.io/ · https://embodiedbench.github.io/challenge.html · https://github.com/EmbodiedBench/EmbodiedBench
- ALRM: https://arxiv.org/html/2601.19510
- VLABench: https://arxiv.org/html/2412.18194
- RoboDojo: https://arxiv.org/abs/2607.04434 · https://robodojo-benchmark.com/leaderboard
- StationeryBench: https://openai.robocurve.org/stationerybench/ · https://github.com/robocurve/stationerybench
- ClapboardBench: https://github.com/robocurve/clapboardbench
- AGP: https://arxiv.org/html/2609.12541
- SafeHarness: https://arxiv.org/abs/2609.20822
- ERQA: https://github.com/embodiedreasoning/ERQA · https://arxiv.org/abs/2503.20020 · https://arxiv.org/abs/2510.03342
- Gemini Robotics ER 2: https://deepmind.google/models/gemini-robotics/embodied-reasoning/ · https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/
- RoboProcessBench: https://arxiv.org/html/2606.13040
- Embodied-BenchForge: https://arxiv.org/html/2609.13082
- ERQA-Plus: https://arxiv.org/abs/2606.17639
- Embodied Arena: https://arxiv.org/abs/2509.15273
- Harness-disclosure position paper: https://arxiv.org/abs/2605.23950
- LIBERO: https://arxiv.org/abs/2306.03310
- LIBERO-Plus: https://arxiv.org/html/2510.13626
- LIBERO-PRO: https://arxiv.org/html/2510.03827
- SimplerEnv: https://arxiv.org/html/2405.05941
- RoboCasa: https://arxiv.org/abs/2406.02523
- RoboCasa365: https://arxiv.org/html/2603.04356
- BEHAVIOR: https://arxiv.org/abs/2403.09227 · https://behavior.stanford.edu/challenge/index.html · https://behavior.stanford.edu/challenge/archive/2025/index.html · https://behavior.stanford.edu/challenge/archive/2025/leaderboard.html
- BEHAVIOR 2025 winners: https://arxiv.org/abs/2512.06951 · https://arxiv.org/abs/2512.10071
- ManiSkill3: https://arxiv.org/abs/2410.00425
- RoboTwin 2.0: https://arxiv.org/html/2506.18088
- Colosseum: https://arxiv.org/abs/2402.08191
- AgiBot World Challenge: https://roboticsandautomationnews.com/2025/11/11/winners-of-agibot-world-challenge-at-iros-2025/96498/ · https://www.therobotreport.com/agibot-holds-world-challenge-2026-see-how-ai-models-perform-real-tasks/
- RoboArena: https://arxiv.org/html/2506.18123 · https://robo-arena.github.io/leaderboard
- RHO: https://arxiv.org/abs/2606.16458
- ASPIRE: https://arxiv.org/abs/2607.00272
- ENPIRE: https://arxiv.org/abs/2606.19980

Evaluation methodology
- TRI LBM: https://arxiv.org/html/2507.05331
- Kress-Gazit et al.: https://arxiv.org/html/2409.09491
- STEP: https://arxiv.org/html/2503.10966
- AutoEval: https://arxiv.org/html/2503.24278
- Veo world simulator: https://arxiv.org/abs/2512.10675
- RoboWorld: https://arxiv.org/abs/2607.01060
- RobotArena ∞: https://arxiv.org/abs/2510.23571
- REALM: https://arxiv.org/abs/2512.19562
- WorldEval: https://arxiv.org/abs/2505.19017

Sim and data generation
- GenSim: https://arxiv.org/abs/2310.01361
- GenSim2: https://arxiv.org/html/2410.03645
- RoboGen: https://arxiv.org/html/2311.01455
- Gen2Sim: https://arxiv.org/abs/2310.18308
- Eureka: https://arxiv.org/abs/2310.12931
- DrEureka: https://arxiv.org/abs/2406.01967
- Eurekaverse: https://arxiv.org/abs/2411.01775
- Text2Reward: https://arxiv.org/abs/2309.11489
- Holodeck: https://arxiv.org/abs/2312.09067
- MimicGen: https://arxiv.org/html/2310.17596
- DexMimicGen: https://arxiv.org/abs/2410.24185
- GR00T N1: https://arxiv.org/html/2503.14734
- RoboVerse: https://arxiv.org/html/2504.18904
- Genesis: https://github.com/Genesis-Embodied-AI/Genesis
- Sim-and-real co-training: https://arxiv.org/html/2503.24361
- AnyTask: https://arxiv.org/abs/2512.17853
- SAGE: https://arxiv.org/abs/2602.10116
- EmbodiedGen V2: https://arxiv.org/abs/2607.07459
- Generative 3D worlds for VLA RL: https://arxiv.org/abs/2603.18532
- V-CAGE: https://arxiv.org/abs/2601.15164
- ARSTAG: https://arxiv.org/abs/2609.24563
- BLAZER: https://arxiv.org/abs/2510.08572
- LLM Trainer: https://arxiv.org/abs/2509.20070
- FAEA: https://arxiv.org/abs/2601.20334
- MolmoBot: https://arxiv.org/html/2603.16861v2
- InternVLA-A1: https://arxiv.org/abs/2601.02456
- SIM1: https://arxiv.org/abs/2604.08544
- Sim-to-real world-action model: https://arxiv.org/abs/2606.31101
- EmbodiedSWE: https://arxiv.org/abs/2609.27308
- GRID auto-engineering: https://www.generalrobotics.company/post/introducing-auto-engineering-for-robotics

Teammate notes: `research/sources/robodojo.md`, `embodiedswe.md`, `general-robotics-auto-engineering.md`, `robocurve-gpt6-astra.md`.

---

## Verification (fact-check pass)

**Pass details.** Adversarial pass, 2026-10-02.
- Primary sources were re-fetched independently:
  - arXiv HTML and abstract pages;
  - the RoboDojo JS bundle;
  - the RoboArena public API (`https://roboarena-api-domain-name.online/api/leaderboard`, `last_updated` 2026-10-02T00:06Z);
  - the Google blog chart images;
  - local repos under `research/repos/`.
- I did not reuse the author's `tmp_bench/` captures, with one exception: I could not render the RoboDojo page myself because the headless browser crashed. The entry count and type tags were re-derived from the site's own JS chunk instead.

**Overall.** The note is accurate on almost every number. I found no fabricated entries and no mis-attributed papers. The errors are mostly over-interpretation in the TL;DR and §6, plus one wrong range.

### Confirmed (seen in a primary source)

**CodeActionBench** (2609.33807; submitted 2026-09-27; NTU ×4 plus one independent researcher).
- Table 10 success counts: Astra/Codex 55 (coverage 22); Opus 5 Ref 37 (19); Opus 5/Claude Code 34 (14); Gemini 3.6 Flash 15; GPT-5.6 Sol 12; Qwen 3.8 Max 11; Grok 4.6 9; Kimi K3 9; Sonnet 5 2.
- Overclaims are 15.0% / 34.2% / 26.8% of failed attempts, i.e. 3/20 Astra, 13/38 Opus Ref, 11/41 Opus CC.
- "Across all nine configurations, 345/491 failed attempts (70.3%) end without a claim" (App. G.2).
- Astra median wall time 477.2 s (§4.2, Table 11).
- Table 8 settings: `claude-opus-5`, reasoning High, 128,000 output tokens, adaptive thinking (§C.1).
- "Model sampling seeds are not fixed" (A.1). Fixed scene seed, domain randomization disabled.
- Repo: MIT; "Full data (coming soon)"; HEAD f5cf933 (2026-09-30).

**CaP-X** (2603.22435; v1 2026-03-23; NVIDIA, UC Berkeley, Stanford, CMU).
- ICML 2026 status is confirmed by the icml.cc poster page (virtual/2026/poster/66369) and the repo tagline.
- 12 models; 7 core tasks at 100 trials per tier; 187 tasks (7 Robosuite + 130 LIBERO-PRO + 50 BEHAVIOR).
- M2 below M1; M3 (VDM) consistently helps.
- CaP-RL Table 4: sim 25/4/30 → 80/44/93 (N=100); Franka 24/12 → 84/76 (N=25).

**AGP** (2609.12541; 2026-09-11; Notre Dame / UCSD / SDSU).
- ≥80% in 7/8 configurations.
- Table 2:
  - Opus 5: 5/5, 22.2 min, $9.75, 12.40M tokens.
  - Fable 5.1: 3/5, 27.7 min, $11.30.
  - Astra: 5/5 at low, medium and high effort; 9.2–9.9 min; $4.09–4.79.
- Code at `github.com/agent-as-policy-2026/agent-as-policy`.

**Gemini Robotics ER 2** (blog, Jul 30, 2026; read from the chart images).
- ERQA: ER 2 78.5, Gemini 3.6 Flash 73.0, ER 1.6 72.5, Opus 5 67.2, GPT 5.6 Sol 43.2.
- Video success detection: Opus 5 81.0 vs ER 2 82.4.
- 5-way progress classification: Opus 5 37.1 vs ER 2 57.4.
- Safety-instruction following: Opus 5 95.9 vs ER 2 97.9.

**RoboDojo** (2607.04434: 42 sim tasks, Isaac Sim, 18 real tasks).
- Site bundle: 51 ranked models.
- GPT-6-Astra is type `agent`, rank 7, 28.97 / 22.48%, added 2026/09/16. Same entry: GPT-5.5 1.13 / 0.88%.
- #1 is PhysicalRSI, type `agent-vla`, 36.27 / 31.38% ("HKU MMLab & KAI", added 2026/09/28).
- The Astra eval page says real testing "was halted for safety … including incidents that damaged hardware". Astra ran through the "RoboProbe L3 harness".

**StationeryBench** (report dated September 10, 2026).
- Astra 7/100 completions vs MolmoAct2 0/100; mean progress 46 vs 12.
- Limitations section: "Grading was operator-judged with the model known"; "The two models were not always run on the same rig".
- Specs: 20-LLM-call budget, 900-step cap vs 1,200; Wilson 95% CIs plotted; repo MIT.

**ClapboardBench** (repo).
- `2026-07-31/runs/README.md`: opus-1 and opus-early pass, so 2/6 for `claude-opus-5`.
- Gemini: 4 fail plus 1 error, so 0/5. The model string in the EvalLogs is `gemini-robotics-er-2-preview`.
- 224×224 frames; one motion per LLM call.

**MolmoBot** (2603.16861v2).
- 79.2% vs π0.5 39.2% over 4 environments × 10 tasks × 3 trials = 120.
- MolmoBot-Pi0 46.7%.
- ~660 successful episodes per GPU-hour.
- InternData-A1 630k / 7,433 h in its comparison table.
- Abstract says 1.8M trajectories; body and Table 1 say 1.7M. Both are covered by "1.7–1.8M".

**RoboCasa365** (2603.04356, ICLR 2026).
- Atomic 15.7/36.3/39.6/43.0; Composite-Seen 0.2/5.2/7.1/9.6; Composite-Unseen 1.25/0.7/1.2/4.4.
- 612 h human + 1,615 h MimicGen; LLM blueprints over 60 activities.

**BEHAVIOR.**
- 2025 leaderboard, held-out: RLC 0.1240 / 0.2599; Comet 0.1140 / 0.2514; best privileged team 0.0520 / 0.0947; 18 teams; 50 tasks, 10,000 demos (1200+ h).
- 2026 page: 100 tasks; 7 scenes; 20,000 demos / 1,950 h; one RGB + depth + proprio track; π0.5 and GR00T N1.7 baselines; deadline 10/16/2026; "LLM-assisted policies" in scope.

**RoboArena.**
- API on 2026-10-02:
  - dreaming_zebra (DreamZero) 1735, std 42.6, 190 evals;
  - pi05_droid 1608, 30.7, 745;
  - pi0_fast_droid 1582, 29.6, 941.
- No LLM-agent entries. "running live through Dec 2026" is in the site bundle.
- Paper: 7 universities, 612 pairwise comparisons, 4,284-rollout oracle, baseline of 17 tasks × 44 episodes.

**TRI LBM** (2507.05331).
- 50 real rollouts per task per policy per condition; 200 sim.
- Blind evaluation; uniform-Beta posteriors.
- Lai (1988) sequential test; Bonferroni correction to a global 95% level; CLD; Welch t-test for task completion.
- Project page: "1,800 real-world evaluation rollouts and over 47,000 simulation rollouts".

**LIBERO robustness suites.**
- LIBERO-PRO: >90% → 0.0%; OpenVLA and π0 collapse beyond 0.2 units of displacement.
- LIBERO-Plus: 10,030 tasks; 7 dimensions; 95% → <30%; "almost all models ignore the language instructions".

**RoboProcessBench** (2606.13040).
- ~58k QA, 260 tasks, 12 families.
- T5 best zero-shot 34.4% vs 33.3% random.
- T10: SFT-Qwen 92.5 vs best zero-shot 38.4 (Gemini-3.1-Flash).
- Closed models go up to Claude-Sonnet-4.6 and GPT-5.4-mini.

**Embodied-BenchForge** (2609.13082).
- OE mean: GPT-5.5 57.67, Opus 4.7 56.61, humans 85.89.
- IE success rate: GPT-5.5 83.18, Opus 4.7 77.73.
- The "220 executable tasks" figure was not re-checked.

**ALRM** (2601.19510): Claude-4.1-Opus TaP 93.5 / CaP 92.6; Falcon-H1-7B 84.3 under CaP, tied with DeepSeek-V3.1.

**EmbodiedBench** (2502.09560).
- 1,128 instances. EB-Manipulation: GPT-4o 28.9, Claude-3.7-Sonnet 28.5, Claude-3.5-Sonnet 25.4. EB-ALFRED: Claude-3.7-Sonnet 67.7.
- 100/120 bins; 15-step cap.
- CVPR 2026 FMEA challenge on EB-ALFRED and EB-Navigation.

**RoboTwin 2.0** (2506.18088).
- 47.4 → 63.9 (R1.0) and 62.1 → 71.3 (R2.0) with multimodal feedback.
- Models: DeepSeek-V3 plus `moonshot-v1-32k-vision-preview` (App.).
- +367% with 10 real demos; +228% zero-shot; 731 objects; 5 embodiments.

**ERQA** (Gemini Robotics report 2503.20020, Table 1): Gemini 2.0 Pro Exp 48.3, GPT-4o 47.0, Claude 3.5 Sonnet 35.5.

**Other papers.**
- SafeHarness (2609.20822): 71.9% success, 87.5% collision avoidance; the agent "collides with the obstacle in most cases".
- RHO (2606.16458): 45.0 vs π0.5 12.83; Robosuite 70.0.
- STEP (2503.10966): up to 32% fewer trials; repeated Barnard's test violates Type-I error control.
- Kress-Gazit (2409.09491): 13/20 vs 14/20 under a uniform prior.
- AutoEval (2503.24278): >2,500 rollouts and >100 h for OpenVLA.
- Veo (2512.10675): 1600+ real evaluations, 8 checkpoints.
- RoboWorld (2607.01060): r=0.989, ρ=0.970, 8 open-source RoboArena policies.
- Harness paper (2605.23950): title as cited; "including cases of model ranking reversal".

**Sim/data-generation papers.**
- EmbodiedGen V2 (2607.07459): 83.3%; 9.7→79.8; 21.7→75.0.
- ARSTAG (2609.24563): Li, Zhang, Liu; 2026-09-21; π0.5 74.6% over 7 tasks.
- FAEA (2601.20334): Claude Agent SDK; 84.9 / 85.7 / 96; repo exists.
- AnyTask (2512.17853): 44%.
- SAGE (2602.10116): "clear scaling trends".
- V-CAGE (2601.15164): VLM rejection sampling against "silent failures".
- SIM1 (2604.08544): 1:15.
- Sim-and-real co-training (2503.24361): +38%.

**Older references.**
- MimicGen: 50K demos from ~200.
- DexMimicGen: 21K from 60.
- GR00T N1: 780k trajectories ≈ 6,500 h generated in 11 h.
- Eureka: 83% of tasks, +52%.
- Text2Reward: 13/17.
- Colosseum: 30–50% per axis, ≥75% combined, R̄²=0.614.
- GenSim: 10× to 100+ tasks, +25%.
- GenSim2: 100 tasks / 200 objects, +20%.
- ManiSkill3: 30,000+ FPS.
- VLABench: VoxPoser PS 30–40 on basic tasks.

**Genesis.** `Genesis-Embodied-AI/genesis-world` v1.4.3 was published 2026-09-30T16:32Z; Apache-2.0.

**AgiBot ICRA 2026** (The Robot Report, 2026-06-07): 526 teams; PrismBot (vivo) 43.47 points. The article attributes the score to the R2A track only by context.

**Derived statistics.** All recomputed:
- Fisher p: 0.744 (37/75 vs 34/75), 0.0042 (55/75 vs 37/75), 0.4545 (2/6 vs 0/5).
- Wilson interval for 7/100: 3.43–13.75%.
- The power table matches an arcsine approximation within ±0.5 pt. Exact values: n=20 from a 20% baseline is +42.0, not +42.5; n=200 is +13.8 / +12.2.

### Corrections (also fixed inline)

1. **TL;DR item 2.** "The harness matters about as much as the model" is not supported.
   - The Opus harness effect is 4.0 pp (p=0.74).
   - The Astra–Opus gap is 24 pp, and it confounds model with harness.
   - Source: 2609.33807 §4.1 and Table 10.
2. **§1, EmbodiedSWE time-to-solve.** 39–103 min → **39–119 min**.
   - Source: 2609.27308 Table 8. Terra = 119, Astra = 39.
   - The teammate note `sources/embodiedswe.md` already carries this fix.
3. **§1, AGP trial duration.** "9–50 min" → **9.2–50.8 min**.
   - These are means over successful trials. Throwing is reported as 17.71 min per session.
4. **§3, LIBERO-PRO row.** ASPIRE "+77%" → **"up to 77 points" vs prior coding agents (CaP-X)** on LIBERO-Pro perturbation suites.
   - It is not a gain over VLAs.
   - Source: 2607.00272 abstract and §1.
5. **§2 interface lessons, CodeActionBench tool list.** The list is a subset of the 30 `D0_TOOLS` (`src/codeaction/interface/registry.py:15-29`).
   - `run_code` is not in that list; it is the program-submission channel.
   - `DENY` (`:35-38`) has 7 names. `get_scene_objects`, `actor_center` and `get_workspace` were omitted.
6. **§6 item 3, RoboProcessBench SFT.** "Lifts a 7B VLM dramatically" is true only for recognition families (T10 92.5, T11 96.5).
   - On primitive-local progress (T5), SFT reaches **45.4%**, against 34.4% best zero-shot and 33.3% random.
   - Temporal ordering (T8) is unchanged.
   - The §2 row's bold claim "A small fine-tuned critic beats frontier zero-shot at progress tracking" is technically true (45.4 vs 34.4) but only by ~11 pt, at well under 50% accuracy.

Minor issues, not counted as corrections:
- The ERQA row labels the old scores "Feb 2025". They come from the Gemini Robotics tech report (arXiv 2503.20020, March 2025), Table 1, without CoT. With CoT, Claude 3.5 Sonnet scores 45.8 (Table 2).
- The RoboTwin "47.4 → 71.3" figure mixes a framework-version change with the feedback ablation. The abstract states "a 10.9% gain in code generation success rate".
- ALRM is internally inconsistent: the abstract says 56 tasks, while §3 says "3 environments × 3 tasks × 6 instructions = 54 tasks".
- CaP-X: the paper says CaP-Gym ships 187 tasks, but the repo README (pushed 2026-05-28) says "39 tasks across Robosuite, LIBERO-PRO, and BEHAVIOR". Check the repo before planning to use all 187.
- The Sources list cites "InternVLA-A1: 2601.02456", but the body's 630k / 7,433 h figure is for the *InternData-A1* dataset, taken from MolmoBot's table.

### Unverifiable or not re-checked

- **GRID auto-engineering** (~4 h to first skill; 143 mm → 5.7 mm). Consistent with the teammate note `sources/general-robotics-auto-engineering.md` (blog 2026-09-09). I did not re-fetch it.
- **RoboVerse** (~500k trajectories, 276 categories, ~5.5k assets). These figures are not in the abstract, and I did not check the body.
- **ManiSkill3 "12 domains".** Not re-checked.
- **Embodied-BenchForge IE "220 tasks".** Not re-checked.
- **AgiBot ICRA 2026 "real G2 finals".** Not re-checked.
- **API model ID `gemini-robotics-er-2-preview`.** Seen only as the model string in ClapboardBench EvalLogs, not in Google API docs.

### Added missed details (within scope)

**1. LIBERO-Agent** (arXiv 2609.39507, submitted 2026-09-30). This is the closest "frontier agent as direct policy" benchmark to Ilia's design, and the note omits it.
- Agents choose which observations to inspect and issue native action commands. It is neither code-as-policy nor a skill API.
- 200 integrated tasks. The primary suite has 30: 10 perception, 10 short-horizon, 10 long-horizon.
- Each task gets 3 rollouts with the same initial state and seed, a 1,800 s wall-clock limit per rollout, and a fixed action-submission budget.
- Vendor harnesses at high effort; Claude Code 2.1.139 for the Claude models.
- Performance Score / time / tokens per episode:

  | Agent | Score | Time | Tokens |
  |---|---|---|---|
  | GPT-6 Astra | 45.0 | 11.5 min | 1.38M |
  | Claude Opus 5 | 19.3 | 22.0 min | 5.49M |
  | Claude Fable 5.1 | 15.8 | 22.4 min | 3.01M |
  | GPT-5.6 Sol | 10.8 | — | — |
  | Qwen3.8-Max | 10.5 | — | — |
  | Kimi K3 | 5.5 | — | — |
  | DeepSeek-V4.1-Flash | 4.5 | — | — |

- Opus 5 breakdown: perception judgment 100%; perception task success 80%; easy short-horizon 60%; hard short-horizon **0%**; long-horizon stage completion 16.7% (easy) and **0%** (hard).
- So the paper's "correct identification does not ensure reliable physical execution" applies directly to Opus.

**2. DrivingBench** (arXiv 2609.38948, 2026-09-30).
- A real Toyota Corolla with 3 tools (camera frames, steering, velocity) on a parking-lot cone course.
- The car keeps moving while the model thinks, so latency is part of the task. This is the only benchmark in scope with a wall-clock time model.
- Models: Astra, Fable 5.1, Sol and Grok 4.6, in vendor harnesses (Codex, Claude Code, Cursor), with up to 3 attempts.
- Only Astra finished, on its 2nd attempt. No other attempt passed 50% of the course. Opus 5 was not tested.

**3. Inspect Robots** (`robocurve/inspect-robots`, MIT, alpha; local clone HEAD 2026-09-30). This is the eval framework under StationeryBench and ClapboardBench.
- "Define a robotics benchmark once, then run any policy (LLM agent, VLA) against any compatible embodiment", with auditable logs (grader scores, LLM transcript, config) and Rerun output.
- It is described as "Inspect AI for robotics".
- It is the most directly reusable piece of eval infrastructure for a Claude harness. StationeryBench used v0.58.0 with `inspect-robots-yam` 0.36.

**4. RAI Bench** (`RobotecAI/rai`, `src/rai_bench/`). The note omits it.
- **Manipulation O3DE benchmark** (`rai_bench/manipulation_o3de`). Score = (correctly_placed_now − correctly_placed_initially) / initially_incorrect. Scenarios come at easy, medium and other levels.
- **Tool Calling Agent benchmark** (`tool_calling_agent`). It is simulator-free and uses tool mocks plus validators. This is a cheap CI-style check for a Claude tool schema.
- **VLM benchmark.**
- RHO (2606.16458) reports results on "RAI's O3DE benchmark" with an LLM in the control loop.

**5. SafeLIBERO.** SafeHarness's numbers (71.9% / 87.5%) are on SafeLIBERO with GPT-6 as the agent. It is a ready-made obstacle-constraint benchmark for testing prompt-only vs harness-enforced safety.

**6. More numbers from the ER 2 blog charts** (Opus 5 vs ER 2).
- Image success detection: 83.6 vs 87.7.
- Instrument reading: 53.0 vs 65.7.
- Human proximity (1 m): **77.1** vs 93.0, with GPT 5.6 Sol at 83.4.
- In the progress chart, Opus 5 is the *lowest* of five models: Sol 46.2, Gemini 3.6 Flash 43.9, ER 1.6 42.7.
- ER 2 as a tool orchestrator, "Physical agent performance":

  | Control mode | ER 1.6 | ER 2 |
  |---|---|---|
  | Real VLA | 48.6% | 60.0% |
  | Sim VLA | 37.4% | 42.9% |
  | Human tele-op | 63.6% | 74.0% |

- Moment-finding: 91.3%, with 0.96 s mean absolute distance.
- Google also announces a new safety benchmark for "a foundation model's ability to act as a safe VLA orchestrator".

**7. CodeActionBench details relevant to harness design.**
- `done` called in 72/75 (Astra), 63/75 (Opus Ref) and 54/75 (Opus CC) attempts.
- Correct claim rate 90.7%, 66.7% and 56.0% respectively.
- Sonnet 5 leaves 70/75 attempts without a claim; 69 of them hit a budget limit.
- The top-4 configurations send 85.6–92.6% of charged calls through `run_code`, against 15.6–35.2% for the rest.
- Opus probes contact in 71/75 attempts (Astra 6/75), while Astra triangulates in 75/75.
- Astra costs 15.1% less than Opus Ref, with a 60.0% shorter median wall time. Gemini 3.6 Flash is cheapest at $52.00 total.

**8. AGP baselines and process data.**
- On two-pair, four-pair and pyramid, AGP succeeds in 23/25 real trials. ASPIRE succeeds in 3/25 and Graph-as-Policy in 0/25. Their sim-refined programs failed to transfer.
- Counting preparation, ASPIRE takes ≈7× AGP's time and 4.2× its cost.
- Experience transfer from Astra lifts GPT-5.6 Terra from 1/5 to 4/5.
- In a 48.6-min four-pair run, 68.2% of time is planning/reasoning and 23.5% is robot motion.
- Server-side safety envelope: radial 0.12–0.65 m, max Cartesian step 0.25 m, 0.03 m/s translation.

**9. Context that weakens two "promising" items.**
- **MolmoBot-SPOC.** Its 36.6% is on a 30-trial single-camera "Pick Kitchen" subset where zero-shot π0.5 scores **63.3%**, so the light head is below the VLA baseline there (2603.16861 Table 7).
- **ClapboardBench.** The runs were "selected from a larger internal 24-hour capture window", so selection bias is possible.
- **StationeryBench.** In 47/100 MolmoAct2 trials the arms never left the start pose.

**10. RoboArena ranking (fills the §8 gap).** Policies with ≥100 A/B evals in the public API as of 2026-10-02. The site counts "official policies with 100+ A/B evals".

| Rank | Policy | Score | Evals |
|---|---|---|---|
| 1 | DreamZero | 1735 | 190 |
| 2 | pi05_droid | 1608 | 745 |
| 3 | pi0_fast_droid | 1582 | 941 |
| 4 | paligemma_vq_droid | 1546 | 952 |
| 5 | paligemma_diffusion_droid | 1533 | 949 |
| 6 | paligemma_fast_specialist_droid | 1531 | 1057 |
| 7 | paligemma_fast_droid | 1516 | 1062 |
| 8 | pi0_droid | 1461 | 1120 |
| 9 | paligemma_binning_droid | 786 | 629 |

- Higher raw scores from low-n entries are excluded: j2-vla 1788 (n=25), Apricot 1661 (n=63).

**11. RoboDojo-Real for Astra.** No official 18-task campaign was run. Only "12-task, 33-retained-trial" diagnostic material exists (RoboDojo Astra report page).

**12. Cross-model embodied QA table** (UNVERIFIED by me; from the sibling note `frontier-models-embodied-2026.md`). PhysBrain 1.5 (arXiv 2609.14973), Table 4, reports overall scores on 28 benchmarks: Astra 73.3, Gemini 3.6 Flash 73.0, Opus 5 67.9.
