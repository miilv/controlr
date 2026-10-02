# Frontier models as direct zero-shot / in-context low-level policies

*Landscape sweep for an "LLM API backbone + light action head" robot harness. Written 2026-10-01.*

**Scope.** This note covers systems where a general LLM or VLM, with no robot-action training, emits the motion itself. That means joint targets, end-effector (EEF) poses or deltas, waypoints, or fine-grained step commands. The model may work zero-shot or from demonstrations placed in its context.

**Excluded, used only as contrast:** VLAs, skill-selector planners, and offline controller-writing agents.

**Teammate deep-dives.** The user-listed sources have their own notes in `research/sources/*.md`. I cite them here but do not repeat them: Robocurve, GPT-as-Policy, GPT-Policy, RoboDojo, EmbodiedSWE, DexAgent, piper-astra-jev, metal-arm-harness, quackd, innate-os #817, so101-painting, General Robotics and Manda.

**Conventions.**
- **[derived]** marks my own arithmetic.
- **UNVERIFIED** marks claims I could not check against a primary source.
- Every 2025–2026 claim links to a page I opened.

---

## 0. Bottom line

1. **The interface matters as much as the model.**
   - Anthropic found that direct torque or EEF control "mostly fail[s]". The same models succeed through code, VLA supervision or gait policies ([Claude plays robotics](https://www.anthropic.com/research/claude-plays-robotics)).
   - Changing only the harness moves results 2–3×:
     - URAI: 18% → 53% ([2609.39018](https://arxiv.org/abs/2609.39018));
     - RoboDawn 35.7% vs RoboProbe 22.5% for the same Astra model on RoboDojo, though RoboDawn ran 5 runs and RoboProbe 50 episodes per task ([2609.22966](https://arxiv.org/abs/2609.22966));
     - Show-Harness 89% vs 13–57% for its VLA, VLA-agent and code-as-policy baselines ([2609.10522](https://arxiv.org/abs/2609.10522)).
2. **The 2026 working pattern:**
   - the LLM emits a *bounded Cartesian target, a delta chunk, or a discrete semantic step*;
   - deterministic code validates it, runs IK, and interpolates at 10–25 Hz;
   - the model sees the outcome one call later.

   Raw joint or torque streaming works only on toys: pendulum, hopper, ~~a ~2 s Go2 balance~~ [corrected: the "nearly two full seconds" Go2 balance (Opus 4.6/4.7, Mythos) was achieved with *programmatic* control (a Python torque-force controller), not per-step LLM streaming; under direct control Opus 4.6 "can keep the robot balanced but cannot successfully stand it up" — anthropic.com/research/claude-plays-robotics, "Programmatic locomotion control"]. No model stood up a G1 humanoid, and Astra's dense locomotion references went 0/5 ([Galbot](https://arxiv.org/abs/2609.38537)).
3. **Decision rate is ~100× too slow for reactive control.**
   - Anthropic: legged control needs ~83 Hz; non-reasoning API inference runs at ~0.2–0.4 Hz.
   - Measured per-decision latency is ~~5–60 s~~ [corrected: ≈2–60 s, tails to 180 s — Anthropic reports 2–8 s for non-reasoning text-only turns and 15–60 s (tails 60–180 s) for Opus 4.6/4.7 at high reasoning; innate-os #817 `act` calls 3.2–8.5 s]. Astra at medium effort: median 5.3 s per call. Astra at xhigh: ~29–47 s per call [derived].
   - Almost every simulation result **pauses physics** while the model thinks.
4. **Works:** semantic and open-vocabulary pick-and-place (19/20 block-into-bowl on real YAM arms, [Robocurve](https://openai.robocurve.org/gpt-6-astra/)), navigation, games, long-horizon arrangement.

   **Fails:** sub-cm precision (puzzle 2/20, 8 mm T-block 10–40%), contact-rich and dynamic tasks (pouring, in-hand rotation, locomotion), large rotations, tight bimanual coordination. RoboDojo halted Astra's real-robot testing after "incidents that damaged hardware".
5. **Generation scaling is real but step-like.**

   | Series | Progression |
   |---|---|
   | Robocurve bowl | Fable 5 1/20 → Fable 5.1 8/20 → Astra 19/20 |
   | RoboDojo, same harness | GPT-5.5 0.88% → Astra 22.48% SR |
   | RoboDawn | GPT-5.6-Luna 14.4% → Sol 43.2% → Astra 73.6% |
   | Robocurve real-robot progress | Opus 5 19.9% → Opus 5.5 36.0%, vs Astra 36.7% |

6. **In-context demonstrations help only when structured.**
   - **Helped:**
     - RoboICL, with receipts: +20–27 points; real robot 14.5 → 63.3 → 78.9 at 0/1/3 shots;
     - RoboDawn: 53.2 → 73.6%;
     - Gemini-ER: 53 → 65%.
   - **Hurt:** RoboDojo, where naive demos lowered success from 22.9% to 17.9% (image) and 12.9% (text).
   - **Most robust effect:** online self-correction from the model's own history.
7. **Light-action-head routes exist:**
   - **Distill harness traces into a 2–9B VLM using the same interface:**
     - Guava-4B 87.1% vs its GPT-5.4 teacher at 90.4%;
     - Show-Harness FT-2B 86% vs zero-shot 89%;
     - WAA-9B 1.7% → 43.3%.
   - **VLA hybrid:** π0.5 + Astra 48% vs direct 26%.
   - **Calibrated gate:** Jev cuts Astra calls by 33–48%.

---

## 1. Taxonomy of action paths

This refines RoboDojo's RoboProbe "what occupies the action path" taxonomy ([report](https://robodojo-benchmark.com/report/gpt-6-astra-eval)).

| LLM emits | Executed by | Examples | Decision rate |
|---|---|---|---|
| Joint or torque targets per step | PD loop, sim paused | Prompt2Walk; Anthropic "direct"; Robocurve Opus 5 stack-blocks | 10 Hz nominal (paused); ~0.2–0.4 Hz real |
| Dense trajectory (list or code), one shot | Position controller, open loop | Kwon; KAT; RoboPrompt; Gemini-ER ICL; SAIL | 1 call per episode |
| Absolute EEF pose(s) per call | IK + interpolation at 10–25 Hz | Inspect Robots `move_to`; RoboProbe `move_eef`; GPT-Policy; GPT-as-Policy Direct | 0.03–0.2 Hz |
| Delta-EEF chunk H×D + receipt | Validate → IK → admissible prefix | RoboICL | ~0.02–0.03 Hz |
| Discrete semantic steps | Embodiment interpreter | Show-Harness; RoboDawn; EmbodiedBench | 30–240 commands per episode |
| GUI / 3D-widget operations | Controller to waypoint | VIA | ~~28–217~~ [corrected: 21–217 per-task mean MCP tool calls per successful episode; column-overall means 40–84 — VIA Table 3] tool calls per success |
| Agent-written tools per phase | Tool code | URAI; FAEA | fewer, coarser decisions |
| Edit or approve VLA actions | VLA | GPT-as-Policy Hybrid; Anthropic VLA supervision; RPent | VLA rate |

---

## 2. Lineage, 2023–2025

| # | Work | Approach | Key numbers | Code | Relevance |
|---|---|---|---|---|---|
| 1 | **LLMs as General Pattern Machines** (Stanford/GDM, Jul 2023) [2307.04721](https://arxiv.org/abs/2307.04721) | Integer-tokenized state/action sequences. Covers completion (sweeping at ~3 Hz, 7-D EEF in 0–100 bins), return-conditioned improvement (CartPole, Marker-in-Cup) and "clicker" rewards at 2 s per step | text-davinci-003 extrapolates motions and discovers CartPole oscillation within 100 episodes. "difficult to deploy today … due to latency, context size limitations, and compute costs" | project page | Origin of LLM-as-sequence-model control; single-token integer design |
| 2 | **Prompt a Robot to Walk** (Berkeley/Tsinghua, Sep 2023) [2309.09969](https://arxiv.org/abs/2309.09969) | GPT-4 (temperature 0) outputs A1 joint targets at a nominal 10 Hz with the sim paused; a 200 Hz PD tracks them. Prompt is a description plus up to 50 steps of history, seeded by an RL policy | Success ≤0.6 (5 × 10 s trials). ~3.1–7.3k input and 38–62 output tokens per step. Only GPT-4 worked (GPT-3.5, davinci-003, Llama 2, Vicuna and Alpaca failed). ~US$2,000 in API cost. Integers in [-300, 300] are single tokens | prompt2walk.github.io | Ceiling on LLM-as-feedback-controller. RL-bootstrapped, so not truly zero-shot |
| 3 | **LMs as Zero-Shot Trajectory Generators** (Imperial, RA-L 2024) [2310.11604](https://arxiv.org/abs/2310.11604) | One task-agnostic GPT-4 prompt plus `detect_object` (LangSAM → 3D boxes). Emits Python that generates dense 4-D EEF poses and `open/close_gripper`, run open loop on a Sawyer, with XMem-based failure detection and replanning | **57.3%** over 30 real tasks × 5 (Code-as-Policies: 22.0%). Model comparison, 5 tasks × 5: GPT-4 76%, Claude 3 Opus 44%, Gemini 1.0 Pro 16%, Claude 2 8%, Llama 2-70B 0%. **Code 60% vs raw numeric lists 10%** on shapes. 48.3% of failures are gripper-pose errors | prompts + code on project page | Emit programs that generate trajectories, not number lists; use explicit gripper functions |
| 4 | **PIVOT** (GDM, Feb 2024) [2402.07872](https://arxiv.org/abs/2402.07872) | GPT-4V selects among action arrows drawn on the image; the candidate distribution is refitted and iterated | Pick coke can: reach 50% → 100%, grasp 0% → 67% (no iteration vs 3 iterations × 3 parallel calls). Scales with Gemini size | HF demo | Visual-servoing template without numeric output |
| 5 | **KAT** (Imperial, RSS 2024) [2403.19578](https://arxiv.org/abs/2403.19578) | DINO keypoints → text; EEF pose as a 3-point triplet → text. GPT-4 Turbo maps them to an action-token trajectory from 10 demos, open loop | 9 real tasks × 10 trials: **0.68** vs Diffusion Policy 0.10. ICL stops improving beyond ~20 demos. GPT-2 < GPT-3.5 Turbo < GPT-4 Turbo | project page | Representation is the lever; context ceiling is low |
| 6 | **RoboPrompt** (Berkeley, ICRA 2025) [2410.12782](https://arxiv.org/abs/2410.12782) | Keyframes plus object poses → text, 10 ICL examples. The LLM outputs the keyframe action sequence | RLBench 16 tasks × 25: **51.8%** (KAT and VoxPoser 21.0%; RVT-2 81.4%). Real Franka 6 × 10: 50–90%. Llama3-8B 28.3, GPT-4o-mini 44.8, GPT-4 Turbo 51.8, GPT-4o 56.3 | [roboprompt](https://github.com/davidyyd/roboprompt) | Compact keyframe format; robust to ~1.7 cm / 4.6° pose noise |
| 7 | **ICRT** (Berkeley, Aug 2024) [2408.15980](https://arxiv.org/abs/2408.15980) | *Robot-trained contrast case.* A 12-layer, 768-d Llama2-architecture transformer trained on DROID + multi-task data, prompted with teleop trajectories, with KV cache | **39.6 Hz** closed loop; a LoRA-tuned Llama2-7B runs at 10.7 Hz and does worse | icrt.dev | What a trained in-context action head looks like |
| 8 | **Gemini Robotics-ER control** (GDM, Mar 2025) [2503.20020](https://arxiv.org/abs/2503.20020) | Zero-shot: iterative code over a robot API (`move_gripper`, `detect_object`, `get_grasp_pose`). ICL: KAT-style pose trajectories plus language reasoning from 10 demos | ALOHA sim (50 trials per task): 2.0 Flash 27% → **ER 53%** zero-shot; ICL 51% → **65%**. Real: zero-shot 25% (dress fold 0%) → ICL 65% (dress fold 56%) | ER via API | Embodied-reasoning post-training ≈ 2× zero-shot control |
| 9 | **EmbodiedBench EB-Manipulation** (ICML 2025) [2502.09560](https://arxiv.org/abs/2502.09560) | MLLM emits a binned 7-D action (100 position bins, 120 rotation bins) with detection boxes and ICL; 15-step cap | GPT-4o 28.9%, Claude-3.7 28.5%, Claude-3.5 25.4%, Gemini-2.0-flash 16.7%, GPT-4o-mini 4.8%. Text-only GPT-4o 16.2% | embodiedbench.github.io | Early-2025 baseline: low-level manipulation is the weakest MLLM skill |
| 10 | **VLA-0** (NVIDIA, Oct 2025) [2510.13054](https://arxiv.org/abs/2510.13054) | *Fine-tuned.* Qwen2.5-VL-3B prints actions as integers 0–1000; no new tokens, no head | LIBERO **94.7%**. Real SO-100: +12.5 points over SmolVLA. **4 Hz** on an RTX 5090. ~32 h on 8×A100 | vla0.github.io | "Actions as text" works once fine-tuned. A head-free option for a distilled student |
| 11 | **Robust Instant Policy** (AIST, IROS 2025) [2506.15157](https://arxiv.org/abs/2506.15157) | KAT-style ICL; fuses several LLM trajectory samples with Student-t regression to reject hallucinations | "at least 26%" success gain over IL baselines | project site | Cheap self-consistency for action samples |

**Boundary cases (planner-level, covered elsewhere):**
- Natural Language as Policies ([2403.13801](https://arxiv.org/abs/2403.13801));
- MALMM ([2411.17636](https://arxiv.org/abs/2411.17636));
- MOKA ([2403.03174](https://arxiv.org/abs/2403.03174));
- MoMa-LLM ([2403.08605](https://arxiv.org/abs/2403.08605)): scene-graph planning over object-centric actions, "zero-shot" only at the planning level.

---

## 3. The 2026 frontier era

### 3.1 Primary studies and benchmarks

**12. FAEA** (Jan 2026) — [2601.20334](https://arxiv.org/abs/2601.20334), [code](https://github.com/robiemusketeer/faea-sim)
- **Setup.** An unmodified Claude Agent SDK runs `claude-opus-4-5-20251101`. It scripts and debugs **with privileged state and Cartesian control**, retrying until success.
- **Success:** LIBERO 84.9% (88.2% with one round of coaching), ManiSkill3 85.7%, MetaWorld 96%.
- **Cost:** $0.80–$4.08 per LIBERO task.
- **Relevance.** Trajectory search and a data generator, not a runtime policy.

**13. SAIL** (Sakana/UTokyo, IROS 2026) — [2603.08269](https://arxiv.org/abs/2603.08269), [project](https://pub.sakana.ai/sail/)
- **Setup.** `gemini-robotics-er-1.5-preview` acts as both generator and scorer. MCTS runs over whole trajectories, using success-archive retrieval and step-level progress feedback, evaluated in simulation or Real2Sim.
- **Results:** 25% (1 rollout) → 73% (45 nodes); real 5/6.
- **Limitation.** Execution is open loop.

**14. Anthropic, "Claude plays robotics"** (2026-07-09) — [page](https://www.anthropic.com/research/claude-plays-robotics). Code is planned at `safety-research/embody` (UNVERIFIED).
- **Models.** Opus 4 → 4.7 and Mythos Preview, against GPT-5.4/5.1, Gemini 3.1/2.5 Pro, Kimi K2.6 and Qwen 3.6+.
- **Interfaces compared (MuJoCo plus a real Go2):** direct torque or 7-D EEF, Python controller, RL supervision, MolmoAct VLA supervision.
- **Results:**
  - Embody composite: **0.115 (Opus 4) → 0.389 (Mythos)**.
  - LIBERO direct full success: **0–5.5%**. Newer models reach, touch and grasp more.
  - A cursor tool lifts Mythos from 6% to 32%.
  - No model stood up the G1.
  - "On the classic control tasks, newer models regressed when given a higher reasoning budget."
- **Latency.** Text-only 2–8 s; with images 5–15 s; Opus 4.6/4.7 at high reasoning 15–60 s (tails 60–180 s).
- **VLA supervision.** Every model scores below the VLA alone, ~~but the best supervisors recover most of the gap~~ [corrected: the page says every model "still performs substantially worse than MolmoAct does on its own"; Mythos Preview underperforms Opus 4.5/4.6 because it "overrides the VLA more often than is warranted". On three novel LIBERO-goal-scene tasks where MolmoAct alone scores 0, Opus 4.5, Opus 4.6 and Gemini 3.1 outperform MolmoAct alone]. Only Mythos solves a significant share of tasks the VLA fails.
- **Relevance.** For an Opus backbone: give abstractions, perception aids and learned low-level controllers, not torques.

**15. VIA** (Stanford, 2026-07-13) — [2607.11119](https://arxiv.org/abs/2607.11119). Simulation only; code UNVERIFIED.
- **Setup.** Claude Code or Codex drives a browser 3D view through MCP tools (`gripper_teleport_via_click`, `gripper_translate`, `gripper_rotate`, `execute_waypoint`). No privileged state; xhigh effort; 10 seeds per task; 1 h cap.
- **Success (minimal prompt):**

  | Agent | Success |
  |---|---|
  | CC-Opus 4.8 | 70% |
  | CC-Fable 5 | **88%** |
  | Codex GPT-5.5 | 60% |
  | Codex GPT-5.6-Sol | 62% |

- A text waypoint "demo" lifts CC-Opus from 77% to 100% on the LIBERO tasks.
- T-block (~8 mm tolerance) stays at 10–40%.
- **Cost:** $4.1–$15.1 per *successful* episode.

**16. Robocurve real-arm series** (Inspect Robots, I2RT YAM bimanual, 224×224 frames, 10 Hz control, model-known human grading)
- **Opus 5 test-time scaling** (2026-08-19) — [page](https://anthropic.robocurve.org/stack-blocks/)
  - **Setup.** `claude-opus-5` emits **raw joint targets** with no IK or planner, one move per call. Limits: 100 calls, 16k output tokens per call, 10% speed cap, n=5 per condition.
  - **Weighted score:** low effort 46 → high 76; GPT-5.6 Sol (high) 60.
  - **Full stacks:** high 1/5, medium 1/5, low 0/5, Sol 0/5 [derived from the run table]. Runs took 3.8–30.7 min.
- **Fable 5 → 5.1** (2026-09-03) — [page](https://anthropic.robocurve.org/fable-5.1/)
  - Bowl 1/20 → 8/20; puzzle 0/20 → 2/20.
- **GPT-6 Astra** (2026-09-04) — [page](https://openai.robocurve.org/gpt-6-astra/)
  - Bowl **19/20**, 2.5 min and ~$0.94 per trial. Fable 5.1: 8/20, 6.8 min, $2.12.
  - Puzzle 2/20 for both, stalling "at the same final step".
  - The runs were not interleaved; see the teammate note.
- **ClapboardBench** (2026-07-31) — [repo](https://github.com/robocurve/clapboardbench)
  - `claude-opus-5` 1/5 (plus one earlier pass) vs `gemini-robotics-er-2-preview` 0/5.
  - Runs were selected from a 24 h capture window.
- **RoboDojo-RC Tier 1** (2026-09-23) — [report](https://robocurve.org/opus-5-5-robodojo-rc-tier-1/), [coverage](https://runtimewire.com/article/robocurve-opus-55-robotics-benchmark)
  - **Setup.** Six real tasks × 20 trials. The tasks are the six easiest *for Astra* out of 18. EEF poses, medium effort, 40 calls, 25% speed cap.
  - **Results:**

    | Model | Mean progress | Full completions | Cost per trial | Output tokens |
    |---|---|---|---|---|
    | Astra | 36.7% | 5/120 | $1.14 | 3,829 |
    | Opus 5.5 | 36.0% | 1/120 | $0.90 | 14,496 |
    | Opus 5 | 19.9% | 2/120 | $1.76 | — [added: 23,308] |

  - Stand Up Bottles: ≤5% progress for every model.
  - Prices used: Opus 5.5 $4/$20 per M tokens; Astra $10/$50.

**17. RoboDojo / RoboProbe "An Unexpected Robot Policy"** (~~HKU MMLab~~ [corrected: RoboProbe and RoboDojo teams; listed affiliations are HKU, Tsinghua, UC Berkeley, Princeton, MIT and PKU, and "MMLab" is not stated — arXiv 2609.24170 author block], 2026-09-16) — [report](https://robodojo-benchmark.com/report/gpt-6-astra-eval), [arXiv 2609.24170](https://arxiv.org/abs/2609.24170), [code](https://github.com/RoboProbe/RoboProbe)
- **Harness ("Inspect EEF").** `move_eef` takes world-frame grasp-point targets. Code bounds each target, converts it to a joint trajectory, and plays it open loop at 25 Hz. No object names or oracle poses.
- **Simulation**, 42 tasks × 50 episodes, one seed ~~:~~ [corrected: this holds for Astra and GPT-5.5 only; DeepSeek-Flash ran 1 seed × 10 episodes per task — 2609.24170 abstract and "†" note]:

  | Model | SR | Score |
  |---|---|---|
  | **Astra** | **22.48%** | 28.97 (above every public VLA) |
  | GPT-5.5 | 0.88% | — |
  | DeepSeek-Flash | 1.92% | — |

- **Real robot.** Testing was halted for safety; retained diagnostic trials gave 1/33 full successes.
- **Demonstration ICL:** zero-shot 22.9% vs image demo 17.9% vs text demo 12.9% (340 matched episodes).
- **Perturbations:** negated xyz recovered 4/8, mirrored vision 6/8.
- **Joint-position campaigns** on a Franka and a humanoid: "task semantics hold, grasp affordance and closed-loop timing do not".
- **Caveat.** Astra's prompt includes reward-ladder "recipes"; see the teammate note.

**18. Galbot, "Systematically Exploring … GPT-6 Astra as Embodied Policies"** (2026-09-29) — [2609.38537](https://arxiv.org/abs/2609.38537). Extends GPT-as-Policy.
- **RoboDojo, paired:** Hybrid **24/50** vs Direct **13/50**. In Hybrid, π0.5 proposes a 50×14 chunk and Astra executes a prefix or substitutes bounded EEF corrections.
- **In-hand rotation:** Astra at-goal 0.51% (cylinder) and 4.40% (cuboid), vs RL 76.9% and 63.5%. Translation + rotation: 0/5 vs RL 4/5.
- **Locomotion** (Astra replacing PASSAGE's motion generator on a G1): 0/5. The final attempt used 250 calls averaging **39.86 s** each, vs ~0.08 s for PASSAGE.
- **Navigation:** RxR 92%, HM3D 82%.
- **Tokens:** Hybrid 624.8M vs Direct 1,132.3M per 50 instances.

**19. GPT-Policy** (Morphi/SII, 2026-09-16) — [2609.19138](https://arxiv.org/abs/2609.19138), [repo](https://github.com/cheng-haha/GPT-Policy); teammate deep-dive.
- **Setup.** `gpt-6-astra` emits one JSON tool call per decision (absolute TCP poses, gripper). A constrained controller (IK + Ruckig) executes and reports.
- **Results** (n=3 per condition):
  - human video: towel 0/3 → 2/3;
  - video + action: uncapping 0/3 → 3/3, plug 0/3 → 2/3.
- **Single towel runs:**

  | Model | Progress | Time | Tokens |
  |---|---|---|---|
  | Astra (human video) | 100% | 15.9 min | 4.96M |
  | Fable 5.1 | 30% | — | — |
  | Kimi K3 | 20% | — | — |
  | Astra, no context | 55% | — | 12.05M |

**20. Show-Harness** (NUS, 2026-09-09) — [2609.10522](https://arxiv.org/abs/2609.10522), [code (Apache-2.0)](https://github.com/showlab/Show-Harness)
- **Action units:** `MV_FWD/…/MV_DOWN`, `ROTATE_CW/CCW`, `GRASP`, `RELEASE`. Steps are 2 cm (target visible in the wrist camera) or 4 cm.
- **Context:** multi-view images, proprioception, the last 5 actions; 50-step cap.
- **Real Franka/AgileX, 10 tasks × 10:**

  | System | Success |
  |---|---|
  | Zero-shot Gemini-3.1 Pro | **89%** |
  | Fine-tuned Qwen3.5-2B (LoRA, 164 real episodes, <2 h on one H200) | 86% |
  | RATS | 57% |
  | π0.5 | 39% |

- **Backbones:** Gemini-3.1 Pro > GPT-5.6-sol > Opus 5 > GPT-5.6-luna > Gemini-3.6 Flash.
- **Effort:** more thinking effort mainly reduces steps (3.4× wall-clock for GPT-5.6-sol). Over 98% of outputs are valid units.
- **Bottleneck:** errors concentrate on grasp and place precision. Target boxes help; a 1 cm step unlocks finer tasks without retraining.
- **Relevance.** The most head-friendly design: the same tokens work for a frontier model zero-shot and for a 2B student.

**21. RoboDawn** (Sep 2026) — [2609.22966](https://arxiv.org/abs/2609.22966)
- **Commands:** discrete `move`/`rotate`/`point`/`gripper`/`home`/`wait`/`done` with magnitudes, clipped to 20 cm / 90°. Supported by grid localization and a command primer.
- **RoboTwin C2R:** Astra 53.2% → **73.6%** one-shot, vs π0.5 46.0%.
- **Model scaling:** GPT-5.6-Luna 14.4 → GPT-5.6-Sol 43.2 → Gemini-3.8-Flash 62.2 → Astra 73.6.
- **RoboDojo:** 35.67% → 47.17%. Raising the command budget from 60 to 240 lifts one-shot SR from 31.2% to 47.2%.
- **Latency:** **9.74 s** of inference per 3.4 commands, vs π0.5 at 101 ms.
- **Real robot:** basket 9/10, stacking 5/10, cloth 0/10. [added: these runs used **Gemini 3.8 Flash zero-shot**, not Astra. Basket and stacking were on a Franka, cloth on a Piper — 2609.22966 §4.2, Table 5]
- **Stated limits:** rotation and near-contact precision.

**22. RoboICL** (Samsung/SJTU, 2026-09-28) — [2609.34261](https://arxiv.org/abs/2609.34261), [code (MIT)](https://github.com/Mosi-AI/RoboICL)
- **Interface.** The Responses API with a single `act` tool: an H×14 matrix of per-step Δp, Δr and absolute gripper target, H = 5–25 at 25 Hz. Code validates the matrix, runs IK, and executes the admissible prefix.
- **Context.** One 1920×480 triptych per step. Demonstrations and the model's own history share one observation–action–receipt–observation grammar, with 25 anchored memory slots.
- **RoboDojo 30 tasks:** Overall **50.64** vs 33.68 for the best baseline; +20–27 points over zero-shot Astra in every category.
- **Real FR3** (3 tasks × 5): 14.45 → 63.33 → 78.89 at 0/1/3 shots. Peg-in-hole 2/5. A larger unseen towel scores 40.
- **Cost.** 2.6–5.6M tokens per episode; 92.4% cache hits over 1,810 calls; ~29–47 s per call [derived].
- **Jev gate** (0.86 s per gate call): Astra calls 181 → 121 and 116 → 60. Success went 3/5 → 2/5 on one task and 4/5 → 5/5 on the other.

**23. URAI** (2026-09-30) — [2609.39018](https://arxiv.org/abs/2609.39018)
- **Native fingertip control** on 5 RoboDojo tasks × 5:

  | Agent | Native success |
  |---|---|
  | DeepSeek-V4-Flash | 8% |
  | Astra | 16% (13.8 min, 6.9k output tokens) |
  | Fable 5.1 | 16% |
  | **Opus 5.5** | **32%** (17.4 min, 32.3k output tokens) |

- **With agent-written phase tools** (model in the loop between calls): 48–60%. Opus 5.5 + URAI reaches 60%. Episodes run 1.3–1.5× faster.
- **Program written in advance:** only 24%.

**24. Guava** (Jun/Sep 2026) — [2606.18363](https://arxiv.org/abs/2606.18363), [code](https://github.com/hdacnw/guava-release)
- **Distillation.** 2,268 simulated GPT-5.4 trajectories through a semantic harness fine-tune Qwen3.5-4B.
- **Results:** 22.2% → **87.1%** (teacher 90.4%); unseen tasks 90.8%; real Franka transfer reported.
- **Relevance.** Direct evidence for "frontier teacher → light head".

**25. World Action Agent** (2026-09-24) — [2609.29964](https://arxiv.org/abs/2609.29964) (abstract)
- Editable action "rehearsal" and in-view offset correction.
- LIBERO-Pro 75.6%. Qwen3.5-9B fine-tuned on traces: 1.7% → 43.3% out-of-domain.

**26. Negative control** (Aug 2026) — [2608.06154](https://arxiv.org/abs/2608.06154) (abstract)
- ~~Nine direct-action VLMs, 32,874 calls, in driving simulators.~~ [corrected: the 32,874 scored calls cover nine direct-action models, six structured local VLMs and an exploratory VLM–MPC hierarchy, over two embodiments and three simulation environments — 2608.06154 abstract]
- "a constant-SLOW policy outperforms a scripted geometric controller"; several models are image-invariant.
- **Lesson.** Run blind-image and mirror ablations before trusting direct-control wins.

### 3.2 Vendor context

- **OpenAI GPT-6 Astra** (2026-09-03) — [Wayback](https://web.archive.org/web/20260903193913/https://openai.com/index/gpt-6-astra/); the live page returns a 403.
  - The announcement does **not mention robots**. It claims computer use (OSWorld 2.0 72.6%).
  - API: `gpt-6-astra`, $10/$50 per M tokens. Fast mode is "up to 2.5x the speed" at 2× price.
- **Gemini Robotics ER 2** (2026-07-30) — [blog](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)
  - A "high-level brain" that "hands off motor execution" to a VLA. Live API streaming for latency.
  - Not positioned as a direct policy.
- **Anthropic Model Hardware Standard** (2026-08-27) — [announcement](https://www.anthropic.com/news/model-hardware-standard-research-preview)
  - Driver primitives (`read`/`write`/discovery) via MCP, CLI or code. Safety limits live in the driver.
  - "When the agent needs to … operate devices faster than its online reasoning would allow, it can chain together driver commands … in code files."
  - LeRobot support is coming. A SO-ARM101 brick demo (4.1 mm accuracy) appears only in secondary coverage: **UNVERIFIED**.

### 3.3 Community demos (n=1 unless stated)

- [Kaifeng Zhang](https://x.com/kaiwynd/status/2098823484474348008): Astra learns keyboard typing on a real arm in "40 minutes".
- [Axel](https://x.com/ax_pey/status/2098216469012283681): video-only ICL for mobile manipulation; Astra "chooses when to use end-effector or joint space".
- [Lucas Cassiano](https://x.com/lucascassiano/status/2097830777438486557): an unseen embodiment, "ZERO VLA".
- [ENPIRE](https://github.com/NVlabs/ENPIRE) ([X](https://x.com/TongheZhang01/status/2097801107602911243)): agentic real-world policy self-improvement.
- [Jiafei Duan](https://x.com/DJiafei/status/2096601096705995155): an Astra harness in the MolmoAct2 sim.
- [OpenRoboto](https://github.com/openroboto-ai/jev-robot-control/blob/main/docs/RESULTS.md): one MuJoCo trial each — Jev 1.13 $0.019 / 182 s vs Astra (low effort) $5.93 / 707 s, both placed; GPT-4.1 mini failed.
- **User-listed sources** (teammate notes):
  - **innate-os #817:** Astra imitates a single episode via `joint_step` ≤0.15 rad and `ee_delta` ≤4 cm at ~~~3.4–8.5 s per call~~ [corrected: 3.2–8.5 s per `act` call, median ≈4.3 s, over 45 steps, plus 19.6 s of planning calls — corrected values in `research/sources/innate-os-pr817.md`; caps in `ros2_ws/src/brain/brain_client/innate/imitation_actions.py:15-19`].
  - **metal-arm-harness:** the first commit was a raw-joint `claude-opus-5` Messages loop, replaced within ~8 h by IK primitives plus a coding agent.
  - **piper-astra-jev:** a skill selector, not a direct policy.

Curated indexes: [Awesome-Astra-Embodied-AI](https://github.com/zjwzcx/Awesome-Astra-Embodied-AI) and [awesome-jev](https://github.com/Frank-ZY-Dou/awesome-jev).

---

## 4. Cross-cutting quantification

### 4.1 Latency and control rate

| System | Model | Per-decision latency | Motion per decision |
|---|---|---|---|
| Prompt2Walk | GPT-4 | n/r (paused) | 0.1 s |
| Anthropic | Opus 4.6/4.7 high | 15–60 s (tails 180 s) | 1 step; legged needs ~83 Hz |
| Robocurve bowl | Astra medium | median 5.3 s, p90 11.5 s (teammate) | ~11 s total motion in a 153 s trial |
| Robocurve bowl | Fable 5.1 | median 12.4 s | same |
| Galbot locomotion | Astra | 39.86 s | 0.5 s reference |
| RoboICL | Astra xhigh | ~29–47 s [derived] | 0.2–1.0 s chunk |
| RoboDawn | Seed-2.1-Pro | 9.74 s per 3.4 commands | 2.09 s |
| Jev gate | jev-1.13 | 0.86 s | reuses 5 actions |
| ICRT (trained head) | 12-layer transformer | 25 ms (39.6 Hz) | 1 step |
| VLA-0 (fine-tuned 3B) | Qwen2.5-VL-3B | 250 ms (4 Hz) | chunk |

Every credible direct-LLM system either pauses the simulator or executes ~~0.2–2 s~~ [corrected: 0.2 s to ~10 s. RoboDojo's official RoboProbe controller plans joint paths "lasting up to 10 s" per target (RoboICL 2609.34261 §4). URAI tool calls run multi-phase motions of 29–72 s each (2609.39018 Table 2)] open-loop chunks between decisions. No frontier API supports reactive control above ~1 Hz.

### 4.2 Cost

| Setting | Cost |
|---|---|
| Robocurve Tier 1 (real) | Opus 5.5 $0.90, Astra $1.14, Opus 5 $1.76 per trial |
| VIA (sim) | $4.1–$15.1 per successful episode |
| RoboICL | ≈~~$9–13~~ [corrected: $7–13 (re-derived from Table 4: 2.6–5.6M tokens per episode, 89–94% cache hits, 45–85k output). Assumes $1/M cache reads, $10/M uncached input and $50/M output; the cheapest condition (Classify Objects, 0-shot) comes to $7.3–9.3 depending on its output share. Cache-write charges are excluded] per episode at Astra list prices with cached reads [derived] |
| GPT-as-Policy | ~$17 (Hybrid) to ~$29 (Direct) per episode (teammate) |
| FAEA | $0.51–$5.60 per task |
| Prompt2Walk | ~$2,000 in total |

### 4.3 Generation scaling (same harness within each row)

| Series | Progression |
|---|---|
| Kwon 2024 | Llama 2 0 → Claude 2 8 → Gemini 1.0 Pro 16 → Claude 3 Opus 44 → GPT-4 76% |
| RoboPrompt | Llama3-8B 28.3 → GPT-4o-mini 44.8 → GPT-4 Turbo 51.8 → GPT-4o 56.3 |
| Gemini-ER | 2.0 Flash 27 → ER 53% |
| EB-Manipulation | GPT-4o-mini 4.8 → Gemini-2.0-flash 16.7 → Claude-3.5 25.4 → GPT-4o 28.9 |
| Anthropic Embody | Opus 4: 0.115 → Mythos: 0.389 |
| VIA | Opus 4.8 70 → Fable 5 88%; GPT-5.5 60 → GPT-5.6-Sol 62% |
| Robocurve bowl | Fable 5 1/20 → 5.1 8/20 → Astra 19/20 |
| Robocurve Tier 1 | Opus 5 19.9 → Opus 5.5 36.0 ≈ Astra 36.7% |
| RoboDojo L3 | GPT-5.5 0.88 → Astra 22.48% |
| RoboDawn | Luna 14.4 → Sol 43.2 → Gemini-3.8-Flash 62.2 → Astra 73.6 |
| URAI native | DeepSeek 8 → Astra 16 = Fable 5.1 16 → Opus 5.5 32 |

Rankings flip by harness. Astra dominates Robocurve's bowl task and RoboDojo, but **Opus 5.5 doubles Astra under URAI native control** and matches it on Tier 1. Precision-gated tasks barely move across generations. Effort scaling is real for Opus 5 on real arms (46 → 76 weighted score, n=5), but flat or negative elsewhere.

### 4.4 What works vs fails

| Works (≥50% in some credible setting) | Fails |
|---|---|
| Open-vocabulary pick/place/sort/stack (19/20; 89%; 9/10) | Sub-cm insertion and alignment (2/20; 10–40%) |
| Long-horizon arrangement (VIA rainbow ~~80–100%~~ [corrected: 50–100% across agents — CC-Opus 80, CC-Fable 100, Codex-5.5 60, Codex-5.6-Sol 50; 80–100% holds only for the Claude Code agents — VIA Table 2], sim) | Pouring and dynamics (stream lag; bottles ≤5%) |
| Turn-based tasks (tic-tac-toe 3/3; 98.75 progress) | In-hand dexterity (0.51% vs RL 76.9%) |
| Navigation via primitives (92% / 82%) | Locomotion (0/5; G1 never stands) |
| I/O perturbation recovery (4/8, 6/8) | Large rotations and cloth (0/10) |
| Structured-demo ICL (RoboICL, RoboDawn, Gemini-ER) | Physical safety (RoboDojo hardware damage) |

---

## 5. Promising vs unpromising patterns for an Opus-backbone harness

**Promising:**
1. **Bounded Cartesian actions with deterministic grounding.** Absolute waypoints, delta chunks or discrete steps with magnitudes, plus:
   - clamps: per-tick deltas, workspace boxes, floor checks;
   - IK and densification (Ruckig in GPT-Policy and metal-arm-harness);
   - an **execution receipt**: commanded vs measured, contact and grasp state.

   All the top systems share this structure.
2. **Fewer, coarser decisions.** URAI phase tools (18 → 53%), RoboICL with H up to 25, and MHS's advice to "chain together driver commands … in code files". Each call costs 5–40 s, so verify via proprioception rather than another LLM call.
3. **Structured, cache-friendly memory.** One grammar for demonstrations and the robot's own history, anchored slots and triptychs give 92–98% cache hits. Naive demo-appending loses 5–10 points.
4. **Perception aids:** a cursor tool (6 → 32%), grid localization (~+15 points in RoboDawn), target boxes (Show-Harness).
5. **Learned priors where precision or dynamics bind.** The π0.5 hybrid (48 vs 26%), VLA supervision (net-positive on VLA-failed tasks for the best models), whole-body trackers for humanoids.
6. **Distill to a light head through the same interface.** Guava-4B, Show-Harness FT-2B and WAA-9B: fewer than 2.3k simulated trajectories, or 164 real episodes. This fixes latency while the frontier model stays teacher and fallback.
7. **Gating and test-time compute.** A Jev-style gate (−33–48% calls), MCTS with a twin (SAIL 25 → 73%), and effort as a dial on hard steps.

**Unpromising:**
- **Torques or joint targets at control rate for anything dynamic.** G1 0 stand-ups; LIBERO direct 0–5.5%; locomotion 0/5; Prompt2Walk needed an RL bootstrap and $2k.
- **Long raw numeric trajectory lists.** Kwon: 10% vs 60% for code on shaped paths. Use code or short integer bins (single-token ints, 0–1000).
- **Open-loop one-shot trajectories without verification.** KAT and RoboPrompt plateau beyond ~20 demos.
- **Text-only decision models emitting numbers.** Jev's own documentation says it "struggles with tasks that require numeric precision".
- **Treating paused-physics simulation, progress scores, model-known grading or easiest-task selection as deployment evidence.**

**Gaps worth building:**
- No published system pairs a frontier LLM with a *small learned residual or action head that turns sparse LLM waypoints into closed-loop contact-rich motion* on a fixed arm. The nearest are VLA hybrids and humanoid trackers.
- No head-to-head of Opus 5.5 vs Astra under identical delta-chunk + receipt interfaces on real precision tasks.
- No study of streaming or fast-mode APIs for closed-loop control.

---

## 6. UNVERIFIED / open

- Release of `safety-research/embody`.
- Code for VIA, URAI, WAA ~~and RoboDawn (robodawn.top not opened)~~ [corrected: RoboDawn code is public at https://github.com/Hugo-AGI/RoboDawn (MIT, Tsinghua + Tencent Hunyuan), linked from robodawn.top. WAA's `ZYH-Lightyear/world-action-agent` is a project page only (index.html, no code). No VIA or URAI code repo found as of 2026-10-02].
- LeRobot × MHS demo numbers.
- Whether Robocurve token counts include hidden reasoning.
- Show-Harness zero-shot per-step latency.

---

## Sources

**Lineage**
- https://arxiv.org/abs/2307.04721 — Pattern Machines
- https://arxiv.org/abs/2309.09969 — Prompt a Robot to Walk
- https://arxiv.org/abs/2310.11604 — Zero-Shot Trajectory Generators
- https://arxiv.org/abs/2402.07872 — PIVOT
- https://arxiv.org/abs/2403.19578 — KAT
- https://arxiv.org/abs/2410.12782 — RoboPrompt; https://github.com/davidyyd/roboprompt
- https://arxiv.org/abs/2408.15980 — ICRT
- https://arxiv.org/abs/2503.20020 — Gemini Robotics
- https://arxiv.org/abs/2510.03342 — Gemini Robotics 1.5
- https://arxiv.org/abs/2502.09560 — EmbodiedBench
- https://arxiv.org/abs/2510.13054 — VLA-0
- https://arxiv.org/abs/2506.15157 — Robust Instant Policy
- https://arxiv.org/abs/2403.13801 ; https://arxiv.org/abs/2411.17636 ; https://arxiv.org/abs/2403.03174 ; https://arxiv.org/abs/2403.08605

**2026**
- https://arxiv.org/abs/2601.20334 ; https://github.com/robiemusketeer/faea-sim
- https://arxiv.org/abs/2603.08269 ; https://pub.sakana.ai/sail/
- https://arxiv.org/abs/2606.18363 ; https://github.com/hdacnw/guava-release
- https://www.anthropic.com/research/claude-plays-robotics
- https://arxiv.org/abs/2607.11119
- https://anthropic.robocurve.org/stack-blocks/
- https://anthropic.robocurve.org/fable-5.1/
- https://openai.robocurve.org/gpt-6-astra/
- https://robocurve.org/opus-5-5-robodojo-rc-tier-1/
- https://runtimewire.com/article/robocurve-opus-55-robotics-benchmark
- https://github.com/robocurve/clapboardbench
- https://robodojo-benchmark.com/report/gpt-6-astra-eval ; https://arxiv.org/abs/2609.24170 ; https://github.com/RoboProbe/RoboProbe
- https://arxiv.org/abs/2609.38537
- https://arxiv.org/abs/2609.19138 ; https://github.com/cheng-haha/GPT-Policy
- https://arxiv.org/abs/2609.10522 ; https://github.com/showlab/Show-Harness
- https://arxiv.org/abs/2609.22966
- https://arxiv.org/abs/2609.34261 ; https://github.com/Mosi-AI/RoboICL
- https://arxiv.org/abs/2609.39018
- https://arxiv.org/abs/2609.29964
- https://arxiv.org/abs/2608.06154

**Vendor**
- https://web.archive.org/web/20260903193913/https://openai.com/index/gpt-6-astra/
- https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/
- https://www.anthropic.com/news/model-hardware-standard-research-preview

**Community**
- https://github.com/zjwzcx/Awesome-Astra-Embodied-AI ; https://github.com/Frank-ZY-Dou/awesome-jev
- https://github.com/openroboto-ai/jev-robot-control/blob/main/docs/RESULTS.md
- https://github.com/NVlabs/ENPIRE ; https://github.com/RLinf/RPent
- https://x.com/kaiwynd/status/2098823484474348008 ; https://x.com/ax_pey/status/2098216469012283681 ; https://x.com/lucascassiano/status/2097830777438486557 ; https://x.com/TongheZhang01/status/2097801107602911243 ; https://x.com/DJiafei/status/2096601096705995155 ; https://x.com/chooi_jeq/status/2102847198690210031

**Teammate deep-dives** (`research/sources/`): robocurve-gpt6-astra, gpt-as-policy, gpt-policy-in-context, robodojo, metal-arm-harness, innate-os-pr817, piper-astra-jev, quackd, embodiedswe, dexagent, so101-painting, llm-robotics-playground, general-robotics-auto-engineering, manda-robotics.

---

## Verification (fact-check pass)

*Adversarial check, 2026-10-02.* Every item below was re-opened in a primary source.
- **arXiv:** full text via `arxiv.org/html/<id>v1`, or the PDF via `pdftotext`.
- **Vendor and Robocurve pages:** fetched with curl. The RoboDojo SPA was checked through its JS bundle `assets/Gpt6AstraEvalPage-OGTNXw1J.js`.
- **X posts:** checked via `api.fxtwitter.com`.
- **Repos:** `research/repos/` and the `gh` API.

Any line not listed under "Confirmed" was not re-checked.

### Summary

- **Reliability: high.** No fabricated or mis-attributed entries were found.
  - All 13 cited 2026 arXiv IDs resolve to papers with the stated titles and dates.
  - All cited repos exist.
  - Nearly every headline number matched to the decimal.
- **12 inline corrections.** Most are scope, attribution or range errors, not invented numbers.
  - The two that matter most for design conclusions:
    1. The ~2 s Go2 balance came from programmatic control, not raw streaming.
    2. "Best VLA supervisors recover most of the gap" is not supported.
- **Missed caveats.** Several important caveats were missing; see "Added missed details".

### Confirmed claims (checked in a primary source)

**Anthropic, "Claude plays robotics"** (Jul 9, 2026)
- Composite: 0.115 (Opus 4) → 0.389 (Mythos Preview), from the figure alt-text.
- LIBERO direct full success 0–5.5%. Cursor tool lifts Mythos from 6% to 32% on a 10-task subset.
- Real-time legged control needs "roughly 83 Hz", vs "~0.2-0.4 Hz" non-reasoning inference.
- No model stood the G1 up.
- Quote: "newer models regressed when given a higher reasoning budget".
- Latency: 2–8 s, 5–15 s with images, 15–60 s with tails of 60–180 s.
- Model list: 12 models via `claude_agent_sdk` and OpenRouter.
- Code "once released, will be in github.com/safety-research/embody". `gh api repos/safety-research/embody` still returns 404 (2026-10-02).

**URAI** (2609.39018, "Make Code as Policy Great Again", 30 Sep 2026)
- Native vs URAI success: DeepSeek-V4-Flash 8 → 48, Astra 16 → 52, Fable 5.1 16 → 52, Opus 5.5 32 → 60. Overall 18.0 → 53.0.
- Native cost:
  - Astra 13.8 min and 6.9k output tokens;
  - Opus 5.5 17.4 min and 32.3k output tokens.
- "Program" (written in advance) condition: 24% for both Astra and Opus 5.5.
- Three agents run 1.3–1.5× faster.
- Protocol: 5 tasks × 5 seeds. Exact McNemar p = 7.9×10⁻⁸.

**RoboDawn** (2609.22966, 19 Sep 2026)
- RoboTwin C2R: 53.2% → 73.6% (π0.5: 46.0%).
- RoboDojo: 35.67% → 47.17%, "average results over 5 runs".
- Command budget 60 → 240: one-shot 31.2% → 47.2%.
- Model scaling: Luna 14.4 / Sol 43.2 / Gemini-3.8-Flash 62.2 / Astra 73.6.
- Latency: 9.74 s per 3.4 commands vs π0.5 101 ms. Seed-2.1-Pro was used for the latency measurement.
- Command clipping: 20 cm / 90°.
- Grid localization ablation: 47.0 → 32.4 without grids (Gemini 3.8 Flash, zero-shot), i.e. ≈+15 points as claimed.

**RoboProbe** (2609.24170 and the report page, Sep 16, 2026)
- 22.48% SR, Score 28.97 (472/2100), ranked above all 40 public policies. GPT-5.5: 0.88%. DeepSeek-Flash: 1.92%.
- Paths resampled to 25 Hz.
- Real-robot material: 1/33 full successes. Testing was halted after "incidents that damaged hardware".
- Demonstration ICL: 78/340 = 22.9% (zero-shot), 61/340 = 17.9% (image), 44/340 = 12.9% (text).
- Perturbations: negated axes 4/8, left–right mirror 6/8.
- Quote "task semantics hold, grasp affordance and closed-loop timing do not" (in the JS bundle).
- Task-recipe prompts confirmed ("TASK RECIPE:" in the appendix template).

**Galbot** (2609.38537, 29 Sep 2026)
- RoboDojo: Hybrid 24/50 (48%) vs Direct 13/50 (26%).
- Hybrid mechanics: π0.5 proposal is 50×14; Astra accepts a 1–15-step prefix.
- In-hand rotation at-goal: Astra 0.51% / 4.40% vs RL 76.90% / 63.50%.
- Translation + rotation: Astra 0/5 vs RL 4/5.
- Locomotion: final 30 s run used 250 calls at a mean of 39.86 s each (PASSAGE ≈0.08 s); physics paused.
- Navigation: RxR 46/50 = 92%, HM3D 41/50 = 82%.
- Tokens: 624.8M vs 1,132.3M.

**RoboICL** (2609.34261, 28 Sep 2026)
- Overall 50.64 vs 33.68 for the best baseline.
- Gains over RoboProbe zero-shot Astra by category (recomputed from category means):
  - Open +20.39;
  - Memory +26.96;
  - Precision +25.88;
  - Long-Horizon +22.66.
- Real FR3: 14.45 → 63.33 → 78.89 at 0/1/3 shots. Peg-in-hole 2/5. Unseen large towel scores 40.
- Cache: 92.4% hit rate over 1,810 requests (113.15M of 122.51M input tokens cached).
- Jev gate: Astra calls 181 → 121 and 116 → 60; success 3/5 → 2/5 and 4/5 → 5/5; 0.86 s per gate call.
- Per-call latency, derived from Table 4 (API minutes ÷ calls): 29.3–47.0 s. Matches the note's derivation.
- Code: a single `act` function tool (`roboicl/policy/astra_policy.py:204`); default `reasoning_effort` is "xhigh" (`astra_policy.py:115`).

**Show-Harness** (2609.10522, Table 2 from the PDF)
- Cross-task averages: ZS 89.0, FT 86.0, RATS 57.0, CaP-X 44.0, H-VLA 50.0, π0.5 39.0, GR00T 35.0, G-VLA 13.0. This confirms "13–57%".
- Training: 164 real episodes (7.8K steps); Qwen3.5-2B LoRA in <2 h on one H200.
- Steps 2/4 cm; 1 cm step gives ZS 60 → 82% and FT 40 → 65%.
- Thinking effort: 3.4× wall-clock for GPT-5.6-sol. Over 98% of outputs are valid units.

**VIA** (2607.11119, Stanford, 13 Jul 2026)
- Minimal-prompt success 70 / 88 / 60 / 62%. T-block 10–40%.
- Cost per success: $4.1–$15.1 (overall means).
- xhigh effort, 10 seeds, 1 h cap.
- With a detailed prompt: CC-Opus 77 → 100% on the LIBERO-Goal tasks.

**FAEA** (2601.20334)
- Model: `claude-opus-4-5-20251101`.
- Success: LIBERO 84.9% (88.2% with coaching), ManiSkill3 85.7%, MetaWorld 96%.
- Cost: LIBERO $0.80–$4.08 per task; $0.51–$5.60 across benchmarks. Both of the note's figures are right; they have different scope.

**SAIL** (2603.08269)
- Model: `gemini-robotics-er-1.5-preview` for policy and scoring.
- 25% → 73% at 45 nodes; real 5/6 (SO-101 BlockIntoBowl).
- Venue: "Accepted to IROS 2026" (pub.sakana.ai/sail).

**Guava** (2606.18363 **v3**)
- 2,268 trajectories (714 of them perturbation branches).
- 22.2 → 87.1% vs GPT-5.4 at 90.4%; unseen tasks 90.8%.

**WAA** (2609.29964)
- LIBERO-Pro 75.6%. Qwen3.5-9B: 1.7 → 43.3% OOD.

**GPT-Policy** (2609.19138; Morphi Robot + SII)
- Human video, towel: 0/3 → 2/3.
- Uncapping 0/3 → 2/3 → 3/3; plug 0/3 → 0/3 → 2/3.
- Towel runs: Astra 100% at 15.85 min and 4.956M tokens; Fable 5.1 30%; Kimi K3 20%; Astra without context 55% at 12.047M tokens.
- Controller: IK + Ruckig.

**Robocurve**
- **stack-blocks** (Aug 19):
  - Weighted means: low 46, medium 64, high 76; Sol 60.
  - Only score-100 runs are full stacks: high 1/5, medium 1/5, low 0/5, Sol 0/5. Run times 3.8–30.7 min.
  - Limits: 100 calls, 16,000 output tokens per call, 10% speed cap, raw joint targets.
- **fable-5.1** (Sep 3):
  - Bowl 1/20 → 8/20; puzzle 0/20 → 2/20.
- **gpt-6-astra** (Sep 4):
  - Astra: 19/20, $0.94, 2.5 min.
  - Fable 5.1: 8/20, $2.12, 6.8 min.
  - Puzzle 2/20 for both; "stalls at the same final step".
  - Not interleaved.
- **Tier 1** (Sep 23):
  - 36.7 / 36.0 / 19.9% mean progress; 5 / 1 / 2 of 120 completions; $1.14 / $0.90 / $1.76 per trial.
  - Output tokens 3,829 / 14,496.
  - Six easiest tasks for Astra out of 18.
  - EEF poses, medium effort, 40 calls, 25% speed cap, 10 Hz recorded control, 224×224 frames.
  - Prices: Opus 5.5 $4/$20, Astra $10/$50.
  - Stand Up Bottles: 2.5 / 0.5 / 5.0%.
- **ClapboardBench** (`research/repos/clapboardbench/2026-07-31/runs/README.md`):
  - opus-1 pass; opus-2..5 fail; opus-early pass.
  - gemini-1..5: 4 fail + 1 error.
  - "selected from a larger internal 24-hour capture window".

**Vendor pages**
- **GPT-6 Astra** (Wayback 20260903193913):
  - Zero occurrences of "robot".
  - OSWorld 2.0: 72.6%.
  - Pricing $10/$50. Fast mode "up to 2.5x the speed … at 2x the Standard price".
- **Gemini Robotics ER 2** (Jul 30, 2026): "high-level brain"; "hands off motor execution" to a VLA; Live API bidirectional streaming.
- **MHS** (Aug 27, 2026):
  - read/write primitives; MCP, CLI or code files.
  - Quoted "chain together driver commands … in code files" sentence is verbatim.
  - Hugging Face is "adding MHS support in LeRobot".

**Lineage (spot-checked)**
- **Kwon:** 57.3% vs CaP 22.0%; 76 / 44 / 16 / 8 / 0 by model; code 60% vs numbers 10%; 48.3%.
- **Prompt2Walk:** 10 Hz paused, 200 Hz PD, temperature 0, "roughly costed $2,000".
- **KAT:** 0.68 vs Diffusion Policy 0.1.
- **RoboPrompt:** 51.8 vs 21.0 vs 81.4; 28.3 / 44.8 / 51.8 / 56.3; 4.61°.
- **Gemini Robotics ER:** 27 → 53 (zero-shot), 51 → 65 (ICL) in sim with 50 trials; real 25 → 65, dress 0 → 56 (9 trials).
- **EmbodiedBench:** GPT-4o 28.9%, 100/120 bins, 15-step cap.
- **VLA-0:** 94.7%, 4 Hz on a 5090, +12.5 vs SmolVLA, ~32 h on 8×A100.
- **ICRT:** 39.6 Hz vs 10.7 Hz.
- **Pattern Machines:** latency quote.
- **RIP:** "at least 26%".

**Other**
- Jev docs: "It struggles with tasks that require numeric precision" (docs.typesafe.ai/model-jaggedness/jev-1.13).
- OpenRoboto `docs/RESULTS.md:7-9`: Jev 1.13 $0.018825 / 181.8 s; Astra low $5.93 / 707.3 s; GPT-4.1 mini hit the 160-cycle limit.
- All six X posts resolve, with matching text and dates (Sep 6–23, 2026).

### Corrections (claim → correct value, evidence)

1. **§0.2 "~2 s Go2 balance" as an example of raw joint/torque streaming.**
   - Correct: the ≈2 s balance (Opus 4.6/4.7, Mythos) used **programmatic** control, a Python torque-force controller.
   - Under direct control, Opus 4.6 "can keep the robot balanced but cannot successfully stand it up".
   - Evidence: claude-plays-robotics, "Programmatic locomotion control".
2. **§0.3 "Measured per-decision latency is 5–60 s".**
   - Correct: ≈2–60 s with tails to 180 s. Anthropic non-reasoning text-only is 2–8 s; innate-os `act` calls are 3.2–8.5 s.
3. **§1 VIA "28–217 tool calls per success".**
   - Correct: per-task means 21–217; column-overall means 40–84 (VIA Table 3).
4. **§3.1 #14 "best supervisors recover most of the gap".**
   - Correct: every model is "substantially worse than MolmoAct does on its own". Mythos underperforms Opus 4.5/4.6 by over-overriding.
   - Only on the three novel tasks (MolmoAct alone = 0) do Opus 4.5, 4.6 and Gemini 3.1 beat the VLA.
5. **§3.1 #17 "HKU MMLab".**
   - Correct: RoboProbe and RoboDojo teams; listed affiliations are HKU, Tsinghua, UC Berkeley, Princeton, MIT and PKU. "MMLab" is not stated.
6. **§3.1 #17 "42 tasks × 50 episodes, one seed" applied to all three models.**
   - Correct: DeepSeek-Flash ran 1 seed × 10 episodes per task.
7. **§3.1 #26 "Nine direct-action VLMs, 32,874 calls".**
   - Correct: the 32,874 calls span nine direct-action models, six local VLMs and a VLM–MPC hierarchy, over two embodiments and three simulators (2608.06154 abstract).
8. **§3.3 innate-os "~3.4–8.5 s per call".**
   - Correct: 3.2–8.5 s, median ≈4.3 s.
   - The teammate note itself was corrected; the "3.85/5.05" readings were the UI's "S" suffix misread as a 5.
9. **§4.1 "executes 0.2–2 s open-loop chunks".**
   - Correct: 0.2 s up to ~10 s. RoboProbe's official controller plans paths "of at most 10 s" (RoboICL §4).
   - URAI tool calls run 29–72 s each.
10. **§4.4 "VIA rainbow 80–100%".**
    - Correct: 50–100% across agents. Codex-5.6-Sol scored 50 and Codex-5.5 60.
11. **§4.2 RoboICL "≈$9–13 per episode" [derived].**
    - Re-derived: ≈$7–13. The cheapest condition comes to $7.3–9.3 depending on its output share.
    - Assumptions: $1/M cache read (Robocurve's stated Astra cache-read rate); cache-write charges excluded. Soft correction.
12. **§6 "Code for … RoboDawn" listed as unverified.**
    - RoboDawn code is public: https://github.com/Hugo-AGI/RoboDawn (MIT, created 2026-09-22), linked from robodawn.top.
    - WAA's `ZYH-Lightyear/world-action-agent` is a static project page with no code.

### Unverifiable or not re-checked

- **Teammate-derived numbers:**
  - Robocurve per-call latency: Astra median 5.3 s / p90 11.5 s; Fable 5.1 12.4 s. These are derived from wire captures in `sources/robocurve-gpt6-astra.md:185-188` and are internally consistent, but not independently recomputed.
  - GPT-as-Policy ~$17 / ~$29 per episode.
  - metal-arm-harness "~8 h" replacement.
  - piper-astra-jev classification.
- **MHS SO-ARM101 brick demo (4.1 mm):** still UNVERIFIED; the primary page does not contain it.
- **Show-Harness backbone order.** The note's strict order (Gemini-3.1 Pro > GPT-5.6-sol > Opus 5 > Luna > Flash) matches the figure's label order. The page text only states two tiers: {Gemini-3.1 Pro, GPT-5.6-sol, Opus 5} at 86–96% and {GPT-5.6-luna, Gemini-3.6-flash} at 72–78%. Treat within-tier ordering as noisy.
- **EmbodiedBench per-model EB-Manipulation numbers:** Claude-3.7 28.5, Claude-3.5 25.4, Gemini-2.0-flash 16.7, GPT-4o-mini 4.8, text-only GPT-4o 16.2. The main table is an image in the PDF; only GPT-4o 28.9% was confirmed (abstract).
- **Not re-opened:**
  - PIVOT (50 → 100 reach, 0 → 67 grasp);
  - Pattern Machines detail (3 Hz, 0–100 bins);
  - Prompt2Walk "≤0.6", "3.1–7.3k tokens" and the "[-300, 300] single tokens" claim. The PDF shows history token counts of 348–7,298.
- **Effort used in RoboICL main runs:** the repo default and the Jev study use xhigh; `bounded_policy.py:331` defaults to "medium". The "xhigh" label on the 29–47 s latency is plausible but not stated for the main tables.

### Added missed details (within scope)

1. **Galbot hybrid result does not generalize.**
   - On **RoboLab**, Direct succeeds 49/50 (98%), Hybrid 46/50 (92%) and π0.5 18/50 (36%, DROID weights, zero-shot). Direct beats Hybrid there.
   - The RoboDojo 10-task subset was chosen by stratifying *published π0.5 success rates* (six tasks from the lowest interval). So "48 vs 26%" is measured where π0.5 is weak and task-finetuned.
   - Hybrid control steps: 85.6% follow π0.5, 14.4% are Astra-generated (2609.38537 §3.2–3.3, Fig. 2).
2. **Agent as Policy (AGP), 2609.12541 (11 Sep 2026)** — a missing boundary entry with direct Opus data.
   - A coding agent (Codex / Claude Code) is the runtime policy on a real YAM arm. Interface: `move_ee`, `move_joints`, `gripper` and local programs.
   - Two-pair assembly, n=5 per model:

     | Model | Success | Time | Tokens | Cost |
     |---|---|---|---|---|
     | Astra (low/med/high effort) | 5/5 each | 9.2–9.9 min | — | $4.09–4.79 |
     | GPT-5.6 Sol | 5/5 | 14.2 min | — | $3.94 |
     | GPT-5.6 Terra | 1/5 | — | — | — |
     | GPT-5.6 Luna | 0/5 | — | — | — |
     | Claude Opus 5 | 5/5 | 22.2 min | 12.4M | $9.75 |
     | Claude Fable 5.1 | 3/5 | 27.7 min | — | $11.30 |

   - AGP reaches ≥80% in 7 of 8 task configurations.
3. **URAI details missing from the note:**
   - Tools are authored by a **Fable 5.1 programming agent** and frozen before evaluation. The 18 → 53% gain is therefore a tool-authoring plus interface change, not a harness-only change.
   - Real AgileX results with GPT-6 Astra + URAI:
     - block-into-bowl 3/3 at 61.3 s, 6.3 calls and 443 output tokens per trial (vs Robocurve's 2.5 min direct EEF);
     - tic-tac-toe 3/3 in 5.0 min vs GPT-Policy's 13.6 min;
     - bottle cap 3/3 in 5.1 min vs 17.9 min.
   - Native control: 21.0–25.6 calls of 19–20 s each; URAI: 3.5–10.0 calls of 29–72 s.
4. **Robocurve caveats:**
   - **Astra page:**
     - The bowl comparison ran on **different rigs**: Fable on rig-3, Astra on rig-1.
     - Astra trials ran two days later.
     - Anthropic requests were sent **without prompt caching**.
     - All models were priced at $10/$50.
     - Fable 5 exhausted its 20-call budget in 18/20 bowl trials.
   - **Tier 1:**
     - Astra's $1.14 is a **cache-adjusted estimate** (re-priced assuming successful caching); Opus costs use recorded caching.
     - Opus 5 is priced at $5/$25 and averaged 23,308 output tokens.
     - Step cap 900.
     - Per task, Opus 5.5 led Pack & Pour (24 vs 17%) and Cap Pen; Astra led Stack Bowls and Stand Up Bottles.
   - **Stack-blocks:**
     - Opus 5 ran in **fast mode**, priced at $10/$50; Sol at $5/$30.
     - Collision guardrail was off.
     - inspect-robots 0.53.0.
5. **RoboProbe details:**
   - The perturbation probe uses only the 8 `general_pickup` layouts Astra already solves.
   - Other perturbation results: top–bottom flip 8/8, no head camera 6/8, right-wrist only 3/8, 10 cm per-move jitter 3/8.
   - Real diagnostic sample: 12 tasks, 33 trials on ARX X5/Piper/Piper X, n=1–4 per task. SR 3.03%, score 6.97, vs official 18-task Pi-05 at 12.8% SR.
   - Astra also wrote a Shadow-hand piano controller emitting 45-D joint commands every 0.05 s. This is code, not per-step LLM control.
6. **RoboDawn details:**
   - Real-robot runs used **Gemini 3.8 Flash** zero-shot (added inline).
   - Shot scaling (Gemini-3.8-Flash, RoboTwin) at 0/1/2/4/8 shots: 47.0 / 62.2 / 63.6 / 65.4 / 62.7. More demos stop helping at 8.
   - Ablations: w/o reasoning 34.8, w/o primer 44.0.
   - The 34-task setting (excluding 8 open tasks) gives 33.96 / 43.33%.
   - RoboDawn's own table lists RoboProbe-Astra as 22.58% SR vs 22.48% in the RoboProbe paper, a citation discrepancy in RoboDawn.
7. **Show-Harness light-head latency:**
   - Fine-tuned 0.8–9B models run at 12–33 Hz (39 ms per decision at 2B, 79 ms at 9B). π0.5 and GR00T run at 20/24 Hz. Frontier ZS runs at "seconds per step" (project page).
   - Sim-to-real: FT 13/20 vs both VLAs 0/20.
   - Cross-embodiment: ZS 93% / FT 87% vs best baseline 52%.
   - This partly answers §6's "Show-Harness zero-shot per-step latency".
8. **RPent (RLinf):**
   - Its arXiv link is **Harness VLA** (2607.08448), where a frozen VLA is a retryable contact-rich *primitive*. This fits the skill/primitive row better than "edit or approve VLA actions".
   - Its README reports Codex / GPT-6 Astra (low reasoning) at **92.63% (741/800) on LIBERO-PRO**. That conflicts with WAA's "state-of-the-art 75.6%" claim. The protocols differ (RPent uses memory batches with separately frozen suites), so do not compare them directly.
9. **CodeActionBench (2609.33807)** — boundary, agentic code-as-policy.
   - 25 tasks, 9 configurations, 675 attempts; success 2.7–73.3%.
   - Best: GPT-6 Astra + Codex CLI, solving 22 of 25 tasks at least once in three attempts.
10. **OpenETA (2608.03924):** its Codex plugin exposes only `observe`, `mark_point` and `move_to`, a minimal direct-EEF interface.
11. **Thea (2608.11246):** a harness with "Scene Graph as Context" and "Evaluation as Exit Codes". Both are boundary entries for interface design.
12. **Guava version drift.**
    - v1 (Jun 16) reported Guava-Agent-4B at 75.6% vs GPT-5.4 at 70.2% using "<2K trajectories"; real ID 86% / OOD 92%.
    - v3 reports 87.1 / 90.4 / 22.2 with 2,268 trajectories.
    - Recovery/perturbation trajectories raise 79.3 → 87.1% (unseen tasks 77.1 → 90.8%). Cite the version.
13. **FAEA:**
    - Costs are pay-per-use equivalents; experiments actually used subscription access.
    - Sub-cm tasks fail: PegInsertion 0%, PlugCharger 0–60%.
    - LIBERO coaching *hurt* ManiSkill: 85.7 → 81.4%, with cost $150 → $220.
14. **GPT-6 Astra announcement:**
    - The OSWorld 2.0 figure is "v2026.08.08, offline set, partial score", at ~40 min per task vs Sol 65.7% at ~75 min.
    - The announcement is available on Amazon Bedrock as well as the API.
15. **MHS:**
    - The primary page includes a **LeRobot-based arm demo** (UW lab): Claude Code coordinates a plate handoff with a liquid handler via MHS, and the instruments "never collided" across repeated tests.
    - Raspberry Pi is also an early adopter.
    - The 4.1 mm SO-ARM101 brick figure remains unsourced.
16. **Anthropic page extras:**
    - Gemini 3.1 and GPT-5.4 build "similarly strong" programmatic Go2 controllers but "lag far behind when controlling the motors directly".
    - Direct-control physics loops would run at 10–125 Hz.
    - LIBERO-40 direct runs took 6–12 h per model and condition.
    - Most cells used 35 trials; some Mythos/Opus 4.7 reruns used 50.
