# VLAs, dual-system architectures and light action heads (2024 – Oct 2026)

*Landscape note for an "frontier-LLM-API backbone + light learned action head" robot harness. Written 2026-10-01. Every 2025–2026 claim below was checked against the linked page or paper (arXiv API metadata plus the arXiv HTML full text, company pages, GitHub READMEs). Items I could not open are marked **UNVERIFIED**. Teammates cover the user's own sources (Robocurve/GPT-6 Astra, GPT-as-Policy, piper-astra-jev and others) in `research/sources/*.md`. I cite them here only for context.*

---

## 0. TL;DR

1. ~~**Every production "VLA" is now a dual system.**~~ [corrected: **Most, but not every, production VLA is a dual system.** GEN-0/GEN-1 are trained "without depending on System1-System2 architectures" (generalistai.com/blog/nov-04-2025-GEN-0), DreamZero is a monolithic WAM, and Figure's Helix 2.5 (2026-09-17) "was itself pretrained from random initialization entirely on Index, unlike Helix 02, which started from a pretrained vision-language model" (figure.ai/news/helix-2-5-zero-shot-30-home-generalization).] A 2–7B VLM, running at roughly 1–10 Hz, conditions a small or medium action expert that produces 16–50-step chunks at 20–200 Hz (1 kHz on Figure's S0).
   - Examples: Helix (S2 7B @7–9 Hz, S1 80M @200 Hz), GR00T N1–N1.7 (2–3B; VLM + flow-matching DiT), π0/π0.5/π0.6/π0.7 (VLM + 300M→860M flow "action expert"), Gemini Robotics (cloud backbone under 160 ms + on-robot decoder, 50 Hz effective).
   - In nearly all of them the S2→S1 interface is a **learned latent or KV-cache/hidden-state** interface trained jointly with the VLM (end-to-end, or with KI-style stop-gradient). Either way it needs the backbone's internals, so a closed API (Claude, GPT, Gemini) cannot sit in that slot.
2. **The interface a closed API *can* drive is language subtasks, plus visual prompts (points, traces, masks, subgoal images).** The strongest steerable low-level models are being built precisely for this:
   - π0.7 is conditioned on language subtasks, metadata, control-modality labels and world-model subgoal images.
   - Gemini Robotics 1.5/2 have an ER orchestrator that calls the VLA as a tool.
   - GR00T N1.7 adds task- and subtask-level reasoning.
3. **Evidence on off-the-shelf frontier models as the high level is mixed but improving fast.**
   - 2025: zero-shot GPT-4o or GPT-4 as the high-level policy was the *worst* condition in Hi Robot and π0.5. A fine-tuned 3B high level beat GPT-4o by ~~more than 40 points of~~ [corrected: "over 40% higher" (paper wording, Fig. 5 caption; it does not say whether this is absolute points or relative)] instruction accuracy.
   - Gemini 2.5 Flash as orchestrator had 2× the failure rate of the fine-tuned GR-ER 1.5 (44.5% vs 22%).
   - 2026: Claude Opus 4.6 orchestrating π0.5 + SAM3 + grasp primitives raised success from 12.6%→41.8% in simulation and from 14.3%→42.9% on a real Franka (VoLo).
   - Anthropic's own study: LLM supervision of MolmoAct helps weak policies, but on in-distribution LIBERO every model was still *worse* than MolmoAct alone. Older models "destroy most of the policy's value".
4. **Closed VLMs are poor zero-shot producers of 2D gripper paths.** HAMSTER (human rank 3.4–3.6 for GPT-4o vs 1.3–1.6 for fine-tuned VILA) ~~and PEEK both say so~~ [corrected: says so with its own measurement; PEEK only restates it by citing prior work ("prior work has found that even the best closed-source models struggle…", 2509.18282 §III) and runs no closed-model comparison]. Every trace or mask interface that works fine-tunes a 3–13B "pointer" VLM. The exception is Gemini Robotics-ER (`gemini-robotics-er-2-preview`), which is trained to emit `[y, x]` points normalized to 0–1000, plus boxes and trajectories.
5. **"Light head" is real, but LIBERO cannot tell you which head to pick.**
   - A 0.54M-parameter policy reaches 95.1% on LIBERO (MINERVA). TurboVLA (0.2B) reaches 97.6%, VLA-Adapter (0.5B backbone) 97.3%.
   - Under perturbation, LIBERO models collapse (LIBERO-PRO: to 0%; MINERVA: to 46–56% on LIBERO-Plus).
   - What matters for a head: action chunking, vision capacity, steerability and robustness. Flow matching vs L1 makes no detectable difference at small scale (MINERVA).
6. **Latency is the binding constraint for an API backbone.**
   - Cloud VLM calls take 1–5 s (VoLo), or ~~2–8 s without reasoning and 15–60 s at high reasoning (Anthropic)~~ [corrected: per Anthropic, about 2–8 s for *text-only* turns without reasoning, "5–15 seconds" with one or two images, and 15–60 s (tails 60–180 s) for Opus 4.6/4.7 at high reasoning; image turns are the relevant case for a robot harness].
   - So the API must sit at the 0.1–0.5 Hz "orchestrator / subgoal / verifier" layer. A local head at ≥10–50 Hz must absorb the delay with chunking and RTC-style delay-conditioned training.
   - Helix trains S1 with a deliberate S1/S2 temporal offset, and π0.7 trains with 0–12-step simulated delays. Both are the template.
7. **2026 frontier:** π0.7 (5B, steerable, distils RL specialists), GR00T N1.7 (3B, Apache-2.0, relative EEF, 20k h human video), Gemini Robotics 2 / ER 2 (ER 2 on the public API, built on Gemini 3.5 Flash), Helix 02 (S0/S1/S2), MolmoAct2 (fully open), GEN-1 (99% on 3 tasks with ~1 h robot data/task), world-action models (DreamZero 14B @7 Hz; GR00T N2 **UNVERIFIED**).

---

## 1. Where the "backbone" ends and the "head" begins: five integration patterns

| # | Pattern | What crosses the interface | Trained jointly? | Usable with closed API backbone? | Exemplars |
|---|---|---|---|---|---|
| A | Monolithic VLA | Nothing (one network) | yes | no | OpenVLA, VLA-0, π0-FAST |
| B | VLM + action expert, shared attention/KV | VLM KV-cache / hidden states | yes (often KI stop-grad) | no | π0/π0.5/π0.7, SmolVLA, MolmoAct2, GR00T |
| C | Dual system with a learned latent | 1 latent vector or token buffer, async | yes, end-to-end | no (needs gradients and hidden states) | Helix, LCB, ThinkAct, DuoCore-FS, FiS-VLA |
| D | Hierarchy with an explicit, human-readable interface | language subtask, 2D trace/points, mask, goal image, coarse action | separately trainable | **yes** | Hi Robot, π0.5-HL, GR 1.5 agent, RT-H, RT-Trajectory, HAMSTER, PEEK, LoHo-Manip, π0.7 (subgoal images) |
| E | Big model proposes and a small head refines (residual) | a reference action chunk plus features | head-only training | **partially** (the API emits the coarse action; the local head refines) | RoboDual, Hume, RL Token (RLT), GPT-as-Policy "hybrid" (inverted) |

For an API backbone the realistic design space is **D + E**, with an optional distilled open-weights "proxy S2" if you later want pattern C.

---

## 2. Dual-system / hierarchical systems

### 2.1 Industrial foundation models

| System (org, date) | High level (S2) | Low level (S1 / head) | Interface | Rates / latency | Key numbers | Code |
|---|---|---|---|---|---|---|
| **Helix** (Figure, 2025-02-20) | 7B open-weight VLM, 7–9 Hz | 80M cross-attention enc-dec transformer, conv vision backbone pretrained in sim | **single continuous latent vector**, concatenated to S1 tokens; gradients flow S1→S2 | 200 Hz, 35-DoF; S1 and S2 on separate embedded GPUs; train-time S1/S2 offset equals deploy latency | ~500 h teleop; no success rates published | closed |
| **Helix 02** (Figure, 2026-01-27) | S2 emits "a sequence of semantic latents" | S1 transformer → full-body joint targets @200 Hz; **S0** 10M net @1 kHz trained on >1,000 h of retargeted human motion with RL in >200k parallel sim envs | latent → joint targets → torques | 1 kHz / 200 Hz / slow | 61 loco-manipulation actions in a 4-min dishwasher task; S0 "replaces 109,504 lines of hand-engineered C++" | closed |
| **GR00T N1** (NVIDIA, 2025-03-18) | Eagle-2 VLM (1.34B), **12th-layer** embeddings, 10 Hz on L40 | flow-matching DiT, embodiment-specific encoders and decoders | cross-attention to mid-layer tokens | 16-action chunk in 63.9 ms (L40, bf16); S1 at "120 Hz" | 2.2B total; latent actions (VQ-VAE) for human video | Apache-2.0 |
| **GR00T N1.5 / N1.6** (2025) | N1.5: **frozen** Eagle 2.5. N1.6: Cosmos-Reason-2B variant, top 4 layers unfrozen | N1.6: 32-layer DiT (vs 16 in N1.5); state-relative chunks | same | N1.6 E2E 37 ms (5090), 38 ms (H100), 44 ms (4090), 105 ms (Thor); 4 denoise steps | N1.5 language following 93.3% vs 46.6% (N1) on GR-1 | open |
| **GR00T N1.7** (NVIDIA, EA 2026-04-17, now GA) | Cosmos-Reason2-2B (Qwen3-VL), task- and subtask-level reasoning | flow-matching DiT (README: 32→16 diffusion layers; HF blog says "32-layer DiT"), horizon 16→40, state/action dims 29→132 | **relative-EEF action space shared by human and robot data** | ONNX/TensorRT; Jetson Thor/Orin | 3B; 20,854 h EgoScale human video ("1k→20k h more than doubles task completion"); LIBERO 97.0% (per MolmoAct2 table) | Apache-2.0 (backbone gated on HF) |
| GR00T N2 / "Isaac GR00T 2" | world-action model based on DreamZero; ">2× success on new tasks"; "end of 2026" | — | — | — | **UNVERIFIED**: NVIDIA newsroom returned 403 | — |
| **π0** (Physical Intelligence, 2024-10-31) | PaliGemma-3B | 300M flow-matching action expert (shared attention) | KV prefix | 50-step chunks up to 50 Hz; 10 Euler steps; 73 ms on-board (4090) | 10k h, 7 robot configurations, 68 tasks | openpi |
| **Hi Robot** (PI, 2025-02-26) | PaliGemma-3B fine-tuned on real plus **synthetic** prompt/interjection data | π0 | **language command** ("pick up the cup") | high-level decode 47 + 13.2 ms (4090) | beats GPT-4o high level by more than 40% instruction accuracy (20 trials/task/method, blind evaluator) | — |
| **π0.5** (2025-04-22) | same model does high-level subtask inference | flow expert (post-training); FAST in pretraining | text subtask then actions | — | ~400 h mobile manipulation; 97.6% of phase-1 examples are *not* mobile-manipulation data; zero-shot **GPT-4 as high level = worst ablation** | openpi |
| **π\*0.6 / Recap** (2025-11-18) | Gemma 3 4B | 860M expert; text "Advantage: positive/negative" conditioning; 670M value function | — | — | more than 2× throughput and about half the failures on the hardest tasks | — |
| **MEM** (2026-03-03) | π0.6 + video encoder (short-term) + **text summaries** (long-term) | — | — | — | tasks up to 15 min | — |
| **RL Token** (2026-04-24) | frozen π0.6 exposes an "RL token" | **2-layer MLP (256 hidden) actor and critic**; 3-layer/512 for the screw task; input = RL token + **reference action chunk**; BC-anchored TD3 | — | 50 Hz, 10×14 = 140-D chunk | up to 3× faster on the critical phase; minutes to hours of practice | unofficial port (`yknxh/rlt-openpi`) |
| **π0.7** (2026-04-16) | Gemma3-4B (+400M vision encoder, MEM-style history) | 860M flow expert, KI, training-time RTC (0–12-step delays = 240 ms @50 Hz) | **language subtask + episode metadata (speed/quality) + control-modality label + subgoal image** from a BAGEL-based world model (14B) | 38 ms minimal on H100, 127 ms worst case; subgoal images 1.25 s on 4×H100 (TP4, 8-bit) | ~5B; matches RL-specialist π\*0.6 on laundry, espresso and boxes; zero-shot UR5e laundry matches expert teleoperators' first try | closed |
| **Gemini Robotics** (GDM, 2025-03-12) | distilled GR-ER backbone **in the cloud**, under 160 ms | **local action decoder on the robot** | — | ~250 ms end-to-end, 50 Hz effective via chunks | 100 demos for new short tasks | closed |
| **Gemini Robotics 1.5 + ER 1.5** (2025-10) | GR-ER 1.5 orchestrator (tool use, success detection) | GR 1.5 "Thinking VLA" (interleaves natural-language thoughts and actions), Motion Transfer | **natural-language instruction; VLA exposed as a tool** | — | failure rate 22% (ER orchestrator) vs 44.5% (Gemini 2.5 Flash orchestrator); planning errors 9% vs 25.5% | ER via API |
| **Gemini Robotics 2 / ER 2 / On-Device 2** (2026-07-30) | ER 2 (`gemini-robotics-er-2-preview`, `-streaming-preview` via Live API; built on Gemini 3.5 Flash) | GR2 VLA (whole body, 22-DoF SharpaWave hand); On-Device 2 | ER 2 "hands off motor execution to any given lower level VLA" registered as tools | streaming endpoint for low-latency agents | On-Device 2: new bi-arm embodiments in "a few hours" with fewer than 200 examples (SO101, Dexmate, Trossen); ER 2: progress classification 57.4%, moment finding 91.3% | ER 2 public; VLAs early access |
| **GEN-0** (Generalist, 2025-11-04) | none: "without depending on System1-System2" | "Harmonic Reasoning": asynchronous, continuous-time sensing and acting token streams | — | — | 270k h; **ossification below about 7B** | closed |
| **GEN-1** (2026-04-02) | — | — | — | — | 99% vs 64% (GEN-0) vs 19% (scratch) on 3 tasks; about **1 h robot data per task**; 2.8× faster box folding | early access |
| **LBM** (TRI, 2025-07-07; Science Robotics 2026) | — | DiT (8 blocks, 768 dim), 16×20-D chunk, run at 10 Hz, execute 8 | — | — | 1,695 h; 1,800 blind real rollouts + 47k sim; **3–5× less data** for new tasks | — |
| **GO-1 / GO-2** (AgiBot, 2025-03 / 2026-04-09) | GO-1: InternVL2.5-2B + **latent planner** over VQ latent actions; GO-2: "Action-CoT" plus async dual system | diffusion action expert; GO-2 "residual refinement" | latent action tokens / action intents | — | GO-1: more than 60% on complex tasks, +32% over RDT; GO-2 (company claim): LIBERO 98.5%, sim-only→real 82.9% | GO-1 open |
| **MolmoAct / MolmoAct2** (Ai2, 2025-08 / 2026-05-04) | Molmo(2)-ER; depth tokens + **editable 2D trace** (v1); adaptive depth (Think) | v2: flow expert via **per-layer KV conditioning** | — | YAM 30-step chunk @30 Hz | v1 SimplerEnv VM 70.5%; v2 LIBERO 97.2% / Think 98.1%; DROID 87.1% vs π0.5-DROID 45.2% (15 trials/cell); 720 h BimanualYAM | fully open |
| **DreamZero** (NVIDIA, 2026-02-17) | none: 14B Wan2.1 video DiT jointly denoises video and action | — | — | **7 Hz** closed loop; ~~~0.6 s/inference on GB200, ~3 s on H100, at least 2 GPUs~~ [corrected: two different figures are mixed here. Paper: 7 Hz "using 2 GB200s", 150 ms/chunk only with DreamZero-Flash (38× over a 5.7 s baseline). Released GitHub code (README): "~0.6s on GB200 and ~3s on H100", minimum 2 GPUs] | more than 2× generalization vs SOTA VLAs; #1 on RoboArena / MolmoSpaces (Feb 2026) | Apache-2.0 |

### 2.2 Academic hierarchical / dual-system models (interface-centric)

| Model (date) | High level → interface → low level | Key quantitative result | Relevance |
|---|---|---|---|
| **RT-H** (2024-03) | RT-2-style VLM predicts **language motions** ("move arm forward") → actions | +15% over RT-2 on 8 tasks; language-motion corrections reach near-perfect success | The motion-language interface is what Show-Harness (2026) feeds a frontier VLM |
| **LCB** (2024-05) | LLaVA emits an `<ACT>` token; its last-layer embedding is the latent goal for 3D Diffuser Actor | CALVIN average length 1.78 vs 1.42 (3DDA); beats GPT-4V + language-only interface | needs hidden states, so not API-compatible |
| **RoboDual** (2024-10) | OpenVLA (discretized actions + latents) → **20M** DiT specialist | +26.7% real, CALVIN 3.27→3.52 after 1 h training; 15 Hz vs 3.9 Hz; works with 5% of demos | the "small head refines a big model's coarse action" template |
| **HAMSTER** (2025-02, ICLR'25) | VILA-1.5-13B fine-tuned to emit a **2D EEF path** drawn on the image → RVT-2 / 3D-DA | +20% absolute (50% relative) over OpenVLA across 7 generalization axes; zero-shot GPT-4o paths ranked worst | the 2D-path interface works, but a fine-tuned pointer is needed |
| **DexVLA** (2025-02) | Qwen2-VL + **1B ScaleDP** diffusion expert, embodiment curriculum | 60 Hz on one A6000; 100 h pretraining | plug-in expert pretrained separately from the VLM |
| **OneTwoVLA** (2025-05) | one model switches `[BOR]` (reason) / `[BOA]` (act) | +30% vs flat, +24% vs dual-system on long-horizon planning | argues against S1/S2 mismatch |
| **Fast-in-Slow** (2025-06) | S1 embedded in the LLaMA2-7B S2 (partial parameter sharing), 1:4 rate | 117.7 Hz with chunk 8 on a 4090; +8% sim, +11% real | — |
| **ThinkAct** (2025-07, NVIDIA) | Qwen2.5-VL-7B (GRPO with visual trajectory rewards) → **visual plan latent** → DiT | LIBERO 84.4%; reasons every 75 actions (25/50/75/100 → 84.0/84.6/84.4/83.7%); 17% slower than OpenVLA | how often to re-plan |
| **PEEK** (2025-09) | VILA-1.5-3B emits `TRAJECTORY: [(x,y)…] MASK: [(x,y)…]`, overlaid on the image every H steps → any RGB policy (ACT 90M, π0 3.5B, 3DDA) | 41.4× real gain for a sim-trained 3D policy; 2–3.5× for VLA and small policies | **policy-agnostic overlay**: the cleanest "API-like" interface |
| **DuoCore-FS** (Astribot, 2025-12) | 3B VLM writes a **latent buffer** at 1–3 Hz → diffusion decoder at 25–30 Hz | ~~~3× faster than synchronous~~ [corrected: 30 Hz whole-body chunk generation with a 3B VLM, "approximately three times as fast as prior VLA models with comparable model sizes"; against synchronous fast-slow baselines the paper reports only qualitatively better success and responsiveness] | async latent buffer |
| **LoHo-Manip** (2026-04) | task-manager VLM predicts remaining subtasks + **2D keypoint trace** each step → trace-conditioned VLA | gains on long-horizon and OOD tasks (sim + Franka) | receding-horizon "remaining plan" gives implicit recovery |
| **Libra-VLA** (2026-04) | discrete macro-direction tokens → continuous refiner, asynchronous | LIBERO 97.2%; inverted-U in decomposition granularity | explicit coarse actions instead of latents |
| **Goal2Skill** (2026-04) | VLM agent with memory, verification and reflection → diffusion VLA | RMBench 32.4% vs 9.8% | — |
| **Fast Plans, Faithful Actions** (2026-09) | π0.5-style waypoint hierarchy | **erasing waypoint endpoints barely changed success**: the executor ignored the plan. Fix: Block-AR (57→8 VLM passes, 8.7× lower latency) + goal modulation with anti-shortcut training → LIBERO-Long 91.0→96.2% | **heads learn to ignore the conditioning unless trained against shortcuts** |

---

## 3. The crux: can an off-the-shelf frontier model be the high level?

| Study | High level (API) | Low level | Result |
|---|---|---|---|
| Hi Robot (2025-02) | GPT-4o, prompt lists about 50 allowed skills: "Every 2 seconds, you can issue one instruction from a provided list" | π0 | Fine-tuned 3B high level beat GPT-4o by more than 40% instruction accuracy. GPT-4o "loses context once physical interaction begins" ("pick up bermuda triangle") |
| π0.5 (2025-04) | GPT-4 picks from the most common labels | π0.5 | "zero-shot GPT-4 ablation attains the worst performance" |
| HAMSTER (2025-02) | GPT-4o zero-shot 2D paths (direct, or via Code-as-Policies + OWLv2) | RVT-2 / 3DDA | human rank 3.40/3.63 (GPT-4o) vs 1.59/1.28 (fine-tuned VILA) |
| Gemini Robotics 1.5 (2025-10) | Gemini 2.5 Flash vs GR-ER 1.5 orchestrator | GR 1.5 VLA | total failure 44.5% vs 22% (planning 25.5% vs 9%, success detection 6% vs 4%, action 13% vs 9%) |
| Hi-VLA systematic study (Google, 2026-06) | Gemini 2.5 Lite / Flash / Pro (± thinking) | Gemini Robotics On-Device (3B) | **Thinking matters more than VLM size** (Flash-Lite with thinking: 74.4/58.2/75.2 vs Pro 70.1/53.1/74.4 on short/long/reasoning tasks). Larger, *steerable* VLAs matter; narrow fine-tuning destroys instruction following. A 4–8 s VLA execution horizon is a good tradeoff; bounding-box scene descriptions help |
| VoLo (NVIDIA, 2026-06) | **Claude Opus 4.6** (+ SAM3, Molmo2, GraspGen, primitives); VLA interruptible mid-rollout | π0.5 (DROID Franka setup) | Sim (RoboVoLo): 12.57% (π0.5 alone) → **41.80%**. Swapping the orchestrator: Sonnet 4.6 36.3, GPT-5.5 35.5, GPT-5-mini 34.7, Gemini 2.5 Flash 32.0, Qwen3-VL-8B 20.0. Real Franka (168 rollouts, 42/system): 14.3% → 42.9%, **but "No-VLA" (VLM + primitives) scored 45.2%**. Cloud call latency 1–5 s. Completion-monitor errors dominate |
| Anthropic "Claude plays robotics" (2026-07-09) | Opus 4 → 4.7, Mythos Preview, GPT-5.4, Gemini 3.1 Pro | MolmoAct supervision (accept, modify or replace each 7-D action) | Direct control on LIBERO-40: 0–5.5% success. With supervision success rises for every model, but all are "substantially worse than MolmoAct on its own" in-distribution. On 3 novel tasks Opus 4.5/4.6 and Gemini 3.1 beat MolmoAct alone. A gripper-cam "cursor" tool lifted Mythos from 6% to 32%. ~~Turns take 2–8 s (15–60 s at high reasoning)~~ [corrected: text-only turns took about 2–8 s without reasoning, 5–15 s with one or two images, 15–60 s at high reasoning; the simulator was paused between calls, so all numbers are a latency-free upper bound] |
| Show-Harness (2026-09) | Gemini-3.1 Pro (zero-shot) choosing discrete **semantic action units** (2 or 4 cm steps, gripper) | deterministic interpreter | Situated planning task 85% (ZS) vs 10% (fine-tuned Qwen3.5-2B) vs 0% (π0.5). Fine-tuned model reaches 70% when given Gemini subtasks |
| EmbodiedSkills (2026-09) | Qwen3-VL skill selection with pre-/post-condition checks | π0.5 | ~~86.2% RoboTwin, 97.4% LIBERO, but 12.5% on memory-dependent RMBench~~ [corrected: these three numbers measure the *task-adapted π0.5 executors*, not the orchestrator. The paper says they "establish the execution performance of the task-adapted low-level VLA policies" (86.20% vs 82.74% π0.5 reference on 50 RoboTwin 2.0 tasks; 97.40% vs 96.85% OpenPI on LIBERO; 12.5% on 4 RMBench memory tasks). Agent-loop ablations are reported separately] |
| Semantic-handoff study (2026-07) | VLM verifier (advance / retry / replan) | π0.5 skills (BEHAVIOR-1K) | skills reach 77–100% from clean states, yet composition "frequently stalls" from chained states |

**Reading.** In 2025, frontier models were weak *situated* planners and poor 2D pointers. By mid-2026 the best (Opus 4.6/4.7, Mythos, GPT-5.5, Gemini 3.x) are good orchestrators and failure detectors over a capable VLA, and they still mostly fail at completion monitoring. Gains concentrate where the VLA alone fails: novel tasks, semantics, memory and recovery. On in-distribution skills the LLM mostly gets in the way.

The teammates' notes point the same way:
- GPT-6 Astra hybrid over π0.5: 48% vs 26% direct on RoboDojo, where the LLM authored only 14.4% of steps (`sources/gpt-as-policy.md`).
- Robocurve: pure-LLM control of bimanual YAM through open-loop absolute-EEF waypoint chunks played at 10 Hz gave Astra 19/20 on block-into-bowl but 2/20 on puzzle-into-groove (`sources/robocurve-gpt6-astra.md`).

---

## 4. Interfaces a frontier API can emit, and the heads that consume them

| Interface the API emits | Head that consumes it | Training data for the head | Evidence | API-readiness (2026) |
|---|---|---|---|---|
| **Language subtask** ("open the fridge") | any language-conditioned VLA: π0.5/π0.7, GR00T N1.7, SmolVLA, MolmoAct2, Gemini Robotics; or a skill library | segment demos and label subtasks (VLM auto-labelling: LeRobot v0.6 `lerobot-annotate`, Helix hindsight labels) | Hi Robot, π0.5, GR 1.5, VoLo, Hi-VLA | **High.** The risk is VLA steerability: fine-tuned VLAs ignore rephrasings (Hi-VLA) |
| **Language motions / semantic action units** ("move left 2 cm") | RT-H policy or a deterministic interpreter | automatic labels from proprioception | RT-H, Show-Harness | High, but slow (one LLM call per micro-step unless chunked) |
| **2D EEF path / trace** drawn on the image | RT-Trajectory (RT-1), HAMSTER (RVT-2/3DDA), LoHo-Manip, TraceVLA, MolmoAct | hindsight projection of EEF trajectories (no manual labels) | RT-Trajectory 67% (2.5D) vs RT-1 16.7% and RT-2 11.1% on 7 unseen skills; HAMSTER +20% | **Medium.** Zero-shot GPT-4o paths are poor; Gemini ER emits trajectories natively; Claude/GPT need a tool (cursor, SoM marks) or a small fine-tuned pointer |
| **Task-relevant points or masks** | PEEK (mask out distractors), ARRO-style masking; any RGB policy | auto-labelled from 20+ OXE datasets | PEEK 2–41× | **Medium-high.** SAM3 / Grounding-DINO can turn API text into masks locally (as in VoLo and piper-astra-jev) |
| **Goal / subgoal image** | goal-conditioned policies (SuSIE, Octo goal images, π0.7) | hindsight future frames | SuSIE (CALVIN SOTA in 2023); π0.7 uses a 14B BAGEL world model at 1.25 s per subgoal | **Low** for closed APIs unless an image-generation API is used; latency in seconds |
| **3D waypoints / EEF targets** (absolute or relative) | IK / impedance plus an optional residual head; RoboDual-style specialist | none for IK; small demos for a residual | Robocurve YAM (19/20 bowl with Astra); GPT-as-Policy direct 26% | High for coarse moves; precision and contact need a learned head |
| **Reference action chunk** from a VLA, edited by the LLM | VLA executes and the LLM edits (Anthropic, GPT-as-Policy hybrid); or a small head refines (RLT) | RL / online data | RLT up to 3× speed on precise phases | High (supervision mode) |
| **Episode metadata / strategy** ("fast, high quality") | π0.7-style metadata-conditioned VLA | metadata labels on all data, including failures | π0.7 | High, if you train the head with it |
| **Learned latent** (Helix, LCB, ThinkAct, GO-1 latent actions) | joint-trained S1 | end-to-end | strong | **None** for a closed API (no gradients or hidden states). Possible only through a distilled open "proxy S2" |

**Anti-shortcut caveat.** Fast-Plans-Faithful-Actions found that the executor ignored the waypoint plan. You need ablation tests that erase or perturb the conditioning, and anti-shortcut training (dropout of other cues, goal modulation). RT-Trajectory's "prompt engineering" appendix and PEEK's re-query every H steps are practical protocols.

---

## 5. Action representations and small heads

### 5.1 Representation choices (verified points)

- **Chunking is the single most important choice.**
  - ACT: 80M parameters, chunk k, temporal ensembling, 50 Hz. ~~10–20 min of demos per task (50 episodes)~~ [corrected: "only 10 minutes or 50 demonstration trajectories" (2304.13705 §1)] gives 80–90% success. Inference takes 10 ms on a 2080 Ti.
  - MINERVA: "only action-chunk length and ~~vision capacity~~ [corrected: vision allocation] consistently exceed a ±1-point training-seed band".
- **Generative head:** Diffusion Policy (+46.9% over prior SOTA on 12 tasks); flow matching (π0: 10 Euler steps, GR00T: 4, π0.7: 5); L1 regression (OFT: LIBERO 97.1% at **26× throughput** over OpenVLA). MINERVA found "no detectable advantage" of flow over L1 (L1 3.8× faster); multimodality matters more on real multi-strategy teleop data than on LIBERO.
- **Tokenized actions** (only if the head is autoregressive or shares an LLM vocabulary): per-dimension binning (RT-2/OpenVLA) fails at high frequency; FAST (DCT + BPE, ~30 tokens per arm-chunk), FAST+ and MolmoAct2 OpenFAST (universal tokenizers), VQ-BeT (5× faster than DP), VQ-VLA (+30% on long-horizon real tasks), BEAST (B-spline, fixed length, smooth).
- **Actions as plain text integers.** VLA-0 asks Qwen2.5-VL-3B to ~~"Output a single sequence of … integers (0 - 1000 each)… Provide only space-separated numbers."~~ [corrected: the verbatim template is "Output a single sequence of H\times D integers (0 - B each), representing the H timesteps sequentially. Provide only space-separated numbers. Nothing else." B is a resolution hyperparameter; B=1000 was "sufficient" on LIBERO and 250 degraded success (2510.13054 §III)]. It reaches LIBERO 94.7% without robot pretraining, and +12.5 points over SmolVLA on real SO-100. But it ran at **4 Hz** on a 5090, and needs ensembling and masked-action augmentation.
  - Implication: an API LLM *can* emit valid action text, but fine-tuning is what makes it good, and API latency rules out closed-loop low-level use.
- **Action space:** GR00T N1.6 moved to state-relative chunks and N1.7 to **relative EEF** shared by robots and humans ("a key factor" for cross-embodiment); π0.7 found EE vs joint "does not show clear advantage" and used joints; MolmoAct2 uses absolute joint pose. For an LLM-facing interface, EEF deltas/targets are the natural language; the head maps them to joints.

### 5.2 Small / light policies (candidate heads)

| Head | Params | Speed | LIBERO avg | Data / notes |
|---|---|---|---|---|
| MINERVA (2026-09) | **0.54M** | 5–9 ms per chunk on a laptop, replans every step | 95.1% (2,000 rollouts) | LIBERO-Plus 46–56%; instruction conditioning mostly selects memorized tasks |
| ACT (2023) | 80M | 10 ms | — | 50 demos per task |
| TurboVLA (2026-07) | 0.2B, no LLM | 31.2 ms, 0.9 GiB (4090) | 97.6% | 0.4B: RoboTwin 2.0 88.06% |
| SmolVLA (2025-06) | 0.45B (0.24B, 2.25B variants) | async client/server inference | 87.3% | 22.9k community episodes; SO100 real 78.3% vs ACT 48.3% |
| VLA-Adapter (2025-09) | 0.5B backbone (Qwen2.5-0.5B) | 219.2 Hz, 36.5 ms | 97.3% (Pro 98.5%) | 8 h on one consumer GPU; no robot pretraining |
| TinyVLA (2024-09) | 0.4–1.3B | — | — | diffusion decoder; bimanual UR5 94.0% (1.3B, 10 trials) |
| Evo-1 (2025-11) | 0.77B | RTC in LeRobot | 94.8% | — |
| GR00T N1.6/1.7 | 3B | ~~37–44 ms on a desktop GPU; 105 ms on Thor~~ [corrected: 37–44 ms / 105 ms (Thor) are **N1.6** torch.compile numbers (`n1d6` README). N1.7 official (`getting_started/hardware_recommendation.md`, 4 denoising steps, 1 camera): H100 11.7 Hz eager / 35.9 Hz TensorRT; RTX Pro 6000 12.8/35.9 Hz; L40 7.8/26.0 Hz; AGX Thor 8.9/12.4 Hz; Orin 2.9/6.6 Hz] | 97.0% (N1.7; also official repo `examples/LIBERO/README.md`: 97.65/97.5/98.45/94.35) | `NEW_EMBODIMENT` fine-tuning: one GPU, 2,000 steps, batch 32 (example) |
| π0 / π0.5 | 3.3B | 73 ms; 46 ms prefill | 94.2 / 96.9% | openpi |
| OpenVLA-OFT | 7B | 71.4 Hz throughput; 25 Hz real ALOHA | 97.1% | parallel decoding + L1 + FiLM |
| OpenVLA | 7B | 4.2 Hz | 76.5% | baseline |

The LIBERO numbers come from each paper or from the MolmoAct2, VLA-0 and VLA-Adapter comparison tables. LIBERO is saturated and memorization-dominated: LIBERO-PRO reports collapse "to 0.0%" under perturbations of object, initial state, instruction and environment. Use LIBERO-Plus/PRO, RoboArena, MolmoSpaces, RoboDojo (teammate note) or blinded real A/B (the TRI LBM protocol: blind, randomized, matched initial conditions, Wilson CIs) to choose heads.

### 5.3 Real-time execution of chunked heads under a slow backbone

- **Synchronous chunking** pauses at chunk boundaries. **Temporal ensembling** (ACT) averages modes, which RTC shows is harmful on multimodal tasks.
- **RTC** (PI, 2025-06; works on any flow/diffusion VLA without retraining) freezes the actions guaranteed to execute and inpaints the rest with a soft mask; it is "uniquely robust to inference delay" (lights a match). **Knowledge insulation** (stop-gradient from the expert into the VLM, FAST-token supervision of the VLM) is the companion training recipe used by π0.6/π0.7. **Training-time RTC** conditions on a simulated delayed prefix and has zero inference cost. ~~Both are now in LeRobot v0.6 rollouts~~ [corrected: test-time RTC has been in LeRobot since 2025-11-19 (PR #1698, Pi0/SmolVLA/Pi0.5) and v0.6.0 (2026-07-07) wires it into `lerobot-rollout`. Training-time RTC landed on `main` only on 2026-08-21 (PR #4056, "feat(pi05): add optional training-time RTC", π0.5 only, `--rtc.mode=trained`). It is not in the v0.6.0 or v0.6.1 tags (`docs/source/rtc.mdx` at those tags has no mention)], and π0.7 uses delays of 0–12 steps.
- **Async S1/S2 with a train-time offset** (Helix), latent buffers (DuoCore-FS: S2 at 1–3 Hz, S1 at 25–30 Hz), and SmolVLA's async policy server are the standard engineering patterns.
- For an API backbone, the delay distribution is seconds and heavy-tailed. Treat the API output as a slowly refreshed condition. Train the head with dropout of the condition and staleness augmentation. Never block actuation on the API.

---

## 6. What looks promising vs. unpromising for an "LLM-API backbone + light head" harness

### Promising (with evidence)

1. **API as orchestrator, success detector and recovery planner over a steerable language-conditioned executor.** This is pattern D, as in GR 1.5/ER 2, VoLo and Hi-VLA.
   - Use an 8 s-class execution horizon.
   - Make success detection a first-class tool. Completion monitoring is the dominant error even for Opus 4.6.
   - Use bounding-box or SAM3 scene descriptions instead of raw pixels only.
   - Use cross-episode memory.
2. **A policy-agnostic visual overlay interface**: 2D trace + task mask drawn on the head's input image (PEEK, HAMSTER, LoHo-Manip, RT-Trajectory).
   - Labels come for free via hindsight projection.
   - The head stays small (ACT/DP-class) and the interface is inspectable.
   - Use Gemini-ER-style pointing, a local fine-tuned 3B pointer, or a cursor tool for Claude rather than raw zero-shot coordinates.
3. **"Propose, then refine."** The API emits coarse EEF waypoints or a choice among VLA proposals. A small local head (RoboDual-style 20M DiT, or an RLT-style MLP residual anchored to the proposal) closes the last centimetres at 30–50 Hz. RLT shows a tiny MLP head can add up to 3× speed on contact-rich phases.
4. **Steerability via prompt fields the API can fill**: π0.7 metadata (speed/quality), subtask text, "Advantage: positive" (Recap). Train the head with these fields from day one so the LLM has knobs.
5. **Distil the API into the head.** Hi Robot's synthetic-interjection data and π0.7's "language coaching → fine-tuned high-level policy" show that LLM-generated plans and corrections become training data. LeRobot v0.6 now stores subtasks, plans, memory and corrections in datasets.

### Unpromising or high-risk

- **Learned-latent S2→S1 interfaces** (Helix, LCB, ThinkAct, FiS, GR00T) with a closed API in S2. They are impossible without gradients or hidden states. Use them only with an open proxy VLM.
- **The API producing low-level actions in the loop.** LIBERO-40 direct control is 0–5.5% (Anthropic). Turns of 2–60 s cannot close a contact loop. VLA-0 shows text actions work only after fine-tuning, and at 4 Hz.
- **Trusting LIBERO / SimplerEnv deltas** to choose a head (MINERVA, LIBERO-PRO).
- **Narrowly fine-tuning a big VLA as executor.** It loses steerability, and the hierarchy then underperforms (Hi-VLA). Keep instruction diversity in fine-tuning data.
- **Waypoint conditioning without anti-shortcut checks.** The executor may ignore it (Fast-Plans).
- **World-action models as the local head.** Today they are 14B, 7 Hz, need at least 2 GPUs (DreamZero) and add 3–4× latency (FastWAM per NVIDIA blog). Promising as a *planner or verifier* (π0.7 subgoals, World Action Planner), not as a light head.
- **Skill-composition handoff.** Clean-state skill success (77–100%) does not compose. Budget for chained-state data and post-condition checks (EmbodiedSkills, BEHAVIOR handoff study).

---

## 7. Gaps and UNVERIFIED items

- **UNVERIFIED:** GR00T N2 / "Isaac GR00T 2" (DreamZero-based WAM, ">2×", end-2026; NVIDIA newsroom returned 403 to curl, WebFetch and headless browser); Gemini Robotics 2 per-task rates (e.g., Apollo pick-from-shelf 76.3%, third-party only; blog text has no numbers); conflicting MolmoAct2 latency figures in press (180/790 ms vs 450/1,300 ms); GEN-1.5; X-VLA's 98.1% LIBERO (third-party table).
- **Company claims only:** GO-2 numbers (via The Robot Report; no paper or weights found). Not published: Helix's S2 VLM identity and success rates; GEN-0/GEN-1 parameter counts and Hz.
- **Open niche:** no published work pairs Claude Opus 5.x / Fable or GPT-6 Astra as orchestrator with a trained light head on real hardware with ≥100 blinded trials. Closest: VoLo (Opus 4.6, 42 real trials per system) and the Anthropic study (simulation).

---

## Sources

**2026 primary sources**
- π0.7 blog (rendered): https://www.pi.website/blog/pi07 ; paper: https://arxiv.org/abs/2604.15483 (HTML v2)
- MEM: https://arxiv.org/abs/2603.03596 ; RL Token: https://arxiv.org/abs/2604.23073
- Helix 02: https://www.figure.ai/news/helix-02
- GR00T N1.7: https://huggingface.co/blog/nvidia/gr00t-n1-7 ; https://github.com/NVIDIA/Isaac-GR00T (README main, `n1d6`, `n1d5` branches)
- Gemini Robotics 2: https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/ ; ER 2: https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/ ; API docs: https://ai.google.dev/gemini-api/docs/robotics-overview
- GEN-1: https://generalistai.com/blog/gen-1
- MolmoAct2: https://arxiv.org/abs/2605.02881
- DreamZero: https://arxiv.org/abs/2602.15922 ; https://github.com/dreamzero0/dreamzero ; NVIDIA WAM blog: https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/
- AgiBot GO-2: https://www.therobotreport.com/agibot-releases-go-2-foundation-model-embodied-ai/
- LeRobot v0.6.0: https://huggingface.co/blog/lerobot-release-v060
- Anthropic, How Claude performs on robotics tasks: https://www.anthropic.com/research/claude-plays-robotics
- VoLo: https://arxiv.org/abs/2606.07723 ; Hi-VLA study: https://arxiv.org/abs/2606.10267 ; Fast Plans, Faithful Actions: https://arxiv.org/abs/2609.30833 ; LoHo-Manip: https://arxiv.org/abs/2604.21924 ; Libra-VLA: https://arxiv.org/abs/2604.24921 ; Goal2Skill: https://arxiv.org/abs/2604.13942 ; Show-Harness: https://arxiv.org/abs/2609.10522 ; EmbodiedSkills: https://arxiv.org/abs/2609.01281 ; Semantic handoff: https://arxiv.org/abs/2607.06256 ; τ0-VLA: https://arxiv.org/abs/2608.16885 ; LongAct/HoloMind: https://arxiv.org/abs/2605.14504
- TurboVLA: https://arxiv.org/abs/2607.27205 ; MINERVA: https://arxiv.org/abs/2609.03715 ; PredVLA: https://arxiv.org/abs/2608.26673 ; VLANeXt: https://arxiv.org/abs/2602.18532

**2025 primary sources**
- Hi Robot: https://arxiv.org/abs/2502.19417 ; π0.5: https://arxiv.org/abs/2504.16054 ; FAST: https://arxiv.org/abs/2501.09747 ; KI: https://arxiv.org/abs/2505.23705 ; RTC: https://arxiv.org/abs/2506.07339 ; training-time RTC: https://arxiv.org/abs/2512.05964 ; π\*0.6/Recap: https://arxiv.org/abs/2511.14759
- GR00T N1: https://arxiv.org/abs/2503.14734
- Gemini Robotics: https://arxiv.org/abs/2503.20020 ; Gemini Robotics 1.5: https://arxiv.org/abs/2510.03342
- GEN-0: https://generalistai.com/blog/nov-04-2025-GEN-0
- TRI LBM: https://arxiv.org/abs/2507.05331 ; project page: https://toyotaresearchinstitute.github.io/lbm1/
- AgiBot World / GO-1: https://arxiv.org/abs/2503.06669
- HAMSTER: https://arxiv.org/abs/2502.05485 ; DexVLA: https://arxiv.org/abs/2502.05855 ; OneTwoVLA: https://arxiv.org/abs/2505.11917 ; Hume: https://arxiv.org/abs/2505.21432 ; Fast-in-Slow: https://arxiv.org/abs/2506.01953 ; ThinkAct: https://arxiv.org/abs/2507.16815 ; MolmoAct: https://arxiv.org/abs/2508.07917 ; Galaxea G0: https://arxiv.org/abs/2509.00576 ; PEEK: https://arxiv.org/abs/2509.18282 ; DuoCore-FS: https://arxiv.org/abs/2512.20188 ; GR-3: https://arxiv.org/abs/2507.15493 ; InternVLA-M1: https://arxiv.org/abs/2510.13778 ; X-VLA: https://arxiv.org/abs/2510.10274
- SmolVLA: https://arxiv.org/abs/2506.01844 ; VLA-Adapter: https://arxiv.org/abs/2509.09372 ; OpenVLA-OFT: https://arxiv.org/abs/2502.19645 ; VLA-0: https://arxiv.org/abs/2510.13054 ; Evo-1: https://arxiv.org/abs/2511.04555 ; LIBERO-PRO: https://arxiv.org/abs/2510.03827 ; UniVLA: https://arxiv.org/abs/2505.06111 ; VQ-VLA: https://arxiv.org/abs/2507.01016 ; BEAST: https://arxiv.org/abs/2506.06072 ; CoT-VLA: https://arxiv.org/abs/2503.22020

**Earlier lineage**
- Helix: https://www.figure.ai/news/helix
- π0: https://arxiv.org/abs/2410.24164 ; OpenVLA: https://arxiv.org/abs/2406.09246 ; TinyVLA: https://arxiv.org/abs/2409.12514 ; RoboDual: https://arxiv.org/abs/2410.08001 ; RDT-1B: https://arxiv.org/abs/2410.07864 ; Octo: https://arxiv.org/abs/2405.12213
- RT-H: https://arxiv.org/abs/2403.01823 ; LCB: https://arxiv.org/abs/2405.04798 ; RT-Trajectory: https://arxiv.org/abs/2311.01977 ; TraceVLA: https://arxiv.org/abs/2412.10345 ; RoboPoint: https://arxiv.org/abs/2406.10721 ; ATM: https://arxiv.org/abs/2401.00025 ; SuSIE: https://arxiv.org/abs/2310.10639
- ACT: https://arxiv.org/abs/2304.13705 ; Diffusion Policy: https://arxiv.org/abs/2303.04137 ; VQ-BeT: https://arxiv.org/abs/2403.03181

**Teammate context:** `research/sources/gpt-as-policy.md`, `research/sources/robocurve-gpt6-astra.md`, `research/sources/piper-astra-jev.md`.

---

## Verification (fact-check pass)

*Adversarial re-check done 2026-10-01/02 against primary sources only. I re-downloaded every source myself (arXiv HTML full text, company pages, GitHub READMEs and repo files). Line references are to my own text conversions in `research/tmp_fc_vla/*.txt` (the HTML is in the same directory), or to repo files. I cloned two new repos: `research/repos/NVIDIA_Isaac-GR00T` (main @ 51d4c89, 2026-08-20) and `research/repos/huggingface_lerobot` (main @ e0d5021, 2026-09-29). Overall verdict: the note is accurate on almost every number I checked. I found no fabricated entries: all 38 cited arXiv IDs resolve to papers with the described titles and dates. The errors are attribution and precision slips plus one stale claim about LeRobot. Twelve corrections are applied inline above as ~~struck~~ + [corrected: …].*

### A. Confirmed claims (seen in the primary source)

1. **VoLo (2606.07723, NVIDIA, 2026-06-05).** Everything here matches the paper (`2606.07723.txt:508,535-592,597,2100-2210`):
   - Orchestrator: Claude Opus 4.6, with π0.5, SAM3, Molmo2 and GraspGen as tools. The VLA and primitives run at 15 Hz; the VLM monitors at 0.2 Hz.
   - Sim: π0.5 alone 12.57% → full system 41.80%. Other orchestrators: Sonnet 4.6 36.34, GPT-5.5 35.52, GPT-5-mini 34.70, Gemini-2.5-Flash 31.97, Qwen3-VL-8B 19.95.
   - Real Franka FR3: 14 tasks × 3 trials = 42 per system, 168 total.
     - π0.5: 14.3% [95% Wilson CI 6.7–27.8]
     - No-VLA: 45.2% [31.2–60.1]
     - Only-VLA: 40.5%
     - Full: 42.9% [29.1–57.8]
   - Cloud VLM latency "∼1–5 s". Completion-monitor errors are ">67% of total events".
2. **Anthropic "Claude plays robotics" / "How Claude performs on robotics tasks" (2026-07-09, Frontier Red Team).** Matches the page (`anthropic_robotics.txt:11,22,23,90,95,108,157-164,176-178,191-192`):
   - Direct manipulation success is "from 0 to 5.5%".
   - Every model is "substantially worse than MolmoAct does on its own" on LIBERO-40, and "earlier models destroy most of the policy's value".
   - On the 3 novel tasks, Opus 4.5, Opus 4.6 and Gemini 3.1 beat MolmoAct alone.
   - The cursor tool lifted Mythos Preview from 6% to 32% on the 10-task subset.
   - Models tested: Opus 4 → 4.7, Mythos Preview, GPT-5.4, GPT-5.1 and Gemini 3.1 Pro Preview.
   - Trial counts: LIBERO-40 = 40 tasks × 5 seeds = 200 trials; tool ablations = 50 trials.
3. **π0.7 (2604.15483, 2026-04-16).** Matches (`2604.15483.txt:115-117,135,149,169-170,277,776-779`):
   - ~5B total: Gemma3-4B backbone (including a 400M vision encoder) plus an 860M flow expert with 50-step chunks.
   - Training-time RTC uses delays of 0–12 steps (240 ms at 50 Hz).
   - Latency: 38 ms minimal (H100, 3 cameras, 5 denoising steps) and 127 ms worst case.
   - Subgoal world model: BAGEL 14B, 1.25 s per image (4×H100, TP4, 8-bit, 25 steps).
   - "EE control does not show clear advantage" → they used joint control.
   - Zero-shot UR5e shirt folding: π0.7 80% success vs 80.6% for 10 expert teleoperators.
4. **Gemini Robotics 2 / ER 2 / On-Device 2 (2026-07-30).**
   - DeepMind blog (`gr2_blog.txt:170,194,203,187`): 22-DoF SharpaWave hand; On-Device 2 adapts "with just a few hours of adaptation time, typically with less than 200 examples" (Dexmate, SO101, Trossen); VLA and On-Device models are available to early-access partners.
   - Google blog (`er2_blog.txt:244,260,263`): ER 2 "hands off motor execution to any given lower level vision-language-action (VLA) model"; 57.4% progress classification; 91.3% moment finding.
   - API docs (`gr_api.txt:203-215,253-254`): `gemini-robotics-er-2-preview` "Builds on Gemini 3.5 Flash"; `gemini-robotics-er-2-streaming-preview` serves the Live API; points come as `[y, x]` normalized to 0-1000.
5. **Helix (2025-02-20) and Helix 02 (2026-01-27).**
   - Helix (`helix.txt:26-52,63`): 7B open-weight S2 at 7–9 Hz; 80M S1 at 200 Hz; 35-DoF; ~500 h; gradients flow S1→S2; a train-time S1/S2 offset that matches deployment latency.
   - Helix 02 (`helix02.txt:15,26,30-32,45,55`): S0 is a 10M network at 1 kHz trained on >1,000 h of retargeted human motion in >200,000 parallel environments; it "replaces 109,504 lines of hand-engineered C++"; 61 actions in a 4-minute task.
6. **GR00T N1 / N1.5 / N1.6 / N1.7.**
   - N1 (2503.14734): 2.2B total with 1.34B VLM; 12th-layer features; 63.9 ms per 16-action chunk (L40, bf16); 10 Hz S2 and 120 Hz S1.
   - N1.5 README: VLM frozen; language following 93.3% vs 46.6%.
   - N1.6 README: 32- vs 16-layer DiT; top 4 VLM layers unfrozen; state-relative chunks; 37/38/44/105 ms.
   - N1.7: HF blog dated 2026-04-17 (Early Access); repo README now says GA; Cosmos-Reason2-2B (Qwen3-VL); EgoScale 20,854 h; "1k to 20k hours more than doubles average task completion"; Apache-2.0 with a gated backbone.
   - N1.7 code changes in README lines 117–119: `select_layer` 16→12; state/action dims 29→132; `action_horizon` 16→40; "changes from `32` to `16` diffusion layers". This confirms the note's flag that the HF blog's "32-layer DiT" contradicts the README.
7. **GEN-0 (2025-11-04) and GEN-1 (2026-04-02).**
   - GEN-0 (`gen0.txt:30,32,34`): 270,000 h; a "phase transition at 7B" (since scaled to 10B+); "without depending on System1-System2 architectures".
   - GEN-1 (`gen1.txt:35,49,69-78,92,104`): 99% vs 64% (GEN-0) vs 19% (from scratch) averaged over 3 tasks; ~1 h of robot data per task; box folding in 12.1 s, 2.8× faster than ~34 s; early access.
8. **MolmoAct2 (2605.02881, 2026-05-04)** (`2605.02881.txt:190,491-501,531,995-1023,1085-1136`):
   - 720 h BimanualYAM; per-layer KV conditioning, detached from the VLM during post-training; 30-step chunks at 30 Hz with absolute joint pose.
   - LIBERO 97.2% (Think 98.1%).
   - DROID 87.1% vs π0.5-DROID 45.2% (15 trials per cell).
   - The same table lists GR00T N1.7 LIBERO at 97.0%.
9. **DreamZero (2602.15922, 2026-02-17).** 14B, Wan2.1-I2V-14B, 7 Hz, Apache-2.0. "#1 on both MolmoSpaces and RoboArena" is from the README news item dated 02/27. See correction C5 for the latency figures.
10. **2026 hierarchy and benchmark papers.**
    - Hi-VLA (2606.10267, Google DeepMind, `:697-709`): Flash-Lite with thinking 74.44/58.21/75.20 vs Pro with thinking 70.10/53.06/74.39; recommended horizon "4-8 seconds"; bounding-box observations help.
    - Show-Harness (2609.10522, `:287,289,337-338`): Gemini-3.1 Pro; 2 cm/4 cm steps; 85% vs 10% vs 0%; 70% with Gemini subtasks.
    - Fast Plans, Faithful Actions (2609.30833, `:64-69,88`): 57→8 passes; 8.7×; 1094→125 ms; 91.0→96.2% on LIBERO-Long.
    - MINERVA (2609.03715, `:49-56`): 0.54M parameters, 95.1% over 2,000 rollouts, 46–56% on LIBERO-Plus, 5–9 ms on a CPU, regression 3.8× faster.
    - TurboVLA (2607.27205, `:61,69`): 0.2B, 97.6%, 31.2 ms, 0.9 GiB; the 0.4B variant gets 88.06% on RoboTwin 2.0.
    - RL Token (2604.23073, `:51,262,566`): 2-layer MLP with 256 hidden (3-layer/512 for the screw task); TD3; C=10 × 14-D = 140-D chunks at 50 Hz; up to 3× speed-up.
    - LoHo-Manip, Libra-VLA (97.2%), Goal2Skill (32.4% vs 9.8%) and the semantic-handoff study (77–100%, "stall") are also confirmed.
11. **2025 numbers.**
    - Gemini Robotics 1.5: failure 44.5% vs 22%; planning 25.5/9; success detection 6/4; action 13/9 (2510.03342 Table 1).
    - Gemini Robotics: cloud backbone "under 160ms", ~250 ms end-to-end, 50 Hz effective.
    - Hi Robot: 20 trials per task per method, blind evaluator; GPT-4o prompt "Every 2 seconds, you can issue one instruction from a provided list" with ~47 listed skills; "pick up bermuda triangle"; 47 + 13.2 ms on a 4090.
    - π0.5: 97.6%; ~400 h; "the zero-shot GPT-4 ablation attains the worst performance".
    - HAMSTER Table 6: ranks 3.40/3.63 vs 1.59/1.28.
    - PEEK: 41.4×; 2–3.5×; VILA-1.5-3B; output format `TRAJECTORY: [...] MASK: [...]`.
    - OpenVLA-OFT: 97.1%, 26×.
    - SmolVLA: 0.45B, 87.3%, 78.3 vs 48.3.
    - VLA-0: 94.7%, 4 Hz on a 5090, +12.5 points over SmolVLA.
    - VLA-Adapter: 97.3%, 219.2 Hz.
    - Evo-1: 0.77B, 94.8%.
    - LIBERO-PRO: "collapses to 0.0%".
    - TRI LBM: ~1,695 h, 1,800 real trials, 3–5× less data, 8 DiT blocks of width 768, 16×20-D chunks, 10 Hz with 8 steps executed.
    - ThinkAct: 84.4%; 25/50/75/100 → 84.0/84.6/84.4/83.7; 17% slower.
    - FiS-VLA: 117.7 Hz; +8% / +11%.
    - OneTwoVLA: +30% / +24%.
    - RT-Trajectory: 67% vs 16.7% / 11.1%.
    - π0: 300M expert; 73 ms; 10k h; 7 configurations; 68 tasks.
    - ACT: ~80M parameters; 0.01 s inference on a 2080 Ti.
12. **Other sources.**
    - GO-2 (The Robot Report, 2026-04-09): LIBERO 98.5% and Genie Sim 3.0 sim→real 82.9%. These are company claims only.
    - The NVIDIA WAM blog (2026-06-15) supports "3–4×": 590–800 ms per chunk for WAM modes vs ~190 ms for π0.5 (per Fast-WAM).
    - LeRobot v0.6.0 blog (2026-07-07) supports `lerobot-annotate` and datasets that store "timestamped subtasks, plans, memory, corrections".
    - Teammate numbers (GPT-as-Policy 24/50 vs 13/50 with 14.4% LLM-authored steps; Robocurve 19/20 and 2/20) match `sources/gpt-as-policy.md:15-19` and `sources/robocurve-gpt6-astra.md:9-10`.

### B. Corrections (all also applied inline)

| # | Claim in note | Correct value | Evidence |
|---|---|---|---|
| C1 | "Every production VLA is now a dual system" | Most are, not all. GEN-0/1 explicitly have no S1/S2. Figure's newest model, Helix 2.5 (2026-09-17), "was itself pretrained from random initialization entirely on Index, unlike Helix 02, which started from a pretrained vision-language model" | `gen0.txt:32`; `helix25.txt:34`; https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization |
| C2 | Anthropic turns "2–8 s without reasoning" (TL;DR §6 and §3 table) | 2–8 s applies to *text-only* turns. With one or two images, turns took "5–15 seconds". The simulator was paused between LLM calls, so results are latency-free upper bounds. Go2 real-time control "would require roughly 83 Hz; current non-reasoning inference runs at ~0.2-0.4 Hz" | `anthropic_robotics.txt:28,42,191-192` |
| C3 | EmbodiedSkills "86.2% RoboTwin, 97.4% LIBERO, 12.5% RMBench" listed as an orchestration result | These are the *task-adapted π0.5 executor* scores (vs 82.74% / 96.85% references), not gains from the Qwen3-VL agent loop | `2609.01281.txt:100-111,133-134,630-645` |
| C4 | Test-time and training-time RTC are "both now in LeRobot v0.6 rollouts" | Test-time RTC has been in LeRobot since 2025-11-19 (PR #1698). Training-time RTC landed on `main` 2026-08-21 (PR #4056), π0.5 only, and is **not** in the v0.6.0 or v0.6.1 tags | `gh api repos/huggingface/lerobot/commits?path=docs/source/rtc.mdx`; `repos/huggingface_lerobot/docs/source/pi05.mdx:178-200` |
| C5 | DreamZero "7 Hz; ~0.6 s/inference on GB200, ~3 s on H100" | Two different measurements are conflated. The paper gets 7 Hz on 2×GB200 with DreamZero-Flash at 150 ms per chunk (38× from 5.7 s; system-only speed-ups are 9.6× on H100 and 16.6× on GB200). The 0.6 s / 3 s figures are the released code's numbers (README) | `2602.15922.txt:184-187,381`; `dreamzero_readme.md:22,142` |
| C6 | DuoCore-FS "~3× faster than synchronous" | The paper says "30 Hz … approximately three times as fast as prior VLA models with comparable model sizes" | abstract of 2512.20188 |
| C7 | GR00T N1.6/1.7 head row "37–44 ms; 105 ms Thor" | Those are N1.6 numbers. N1.7 official: H100 11.7 Hz eager / 35.9 Hz TRT; Thor 8.9/12.4 Hz; Orin 2.9/6.6 Hz (4 denoising steps, 1 camera) | `repos/NVIDIA_Isaac-GR00T/getting_started/hardware_recommendation.md:13-25` |
| C8 | Hi Robot "more than 40 points" | The paper says "over 40% higher instruction accuracy" and does not say whether that is absolute or relative | `2502.19417.txt:158` |
| C9 | "HAMSTER … and PEEK both say so" | Only HAMSTER measured it. PEEK cites prior work | `2509.18282.txt:99` |
| C10 | MINERVA quote "vision capacity" | The wording is "vision allocation" | `2609.03715.txt:49` |
| C11 | VLA-0 quoted prompt "(0 - 1000 each)" | The template is "(0 - B each)"; B=1000 was used on LIBERO | `2510.13054.txt:255,289` |
| C12 | ACT "10–20 min of demos (50 episodes)" | "only 10 minutes or 50 demonstration trajectories" | `act.txt:96` |

Minor, not applied inline:
- The RL Token PI blog is dated 2026-03-19; 2026-04-24 is the arXiv date.
- The Anthropic study also tested GPT-5.1.
- In Hi-VLA, Gemini 2.5 Pro was evaluated only with thinking on.
- The note gives Helix 02 as 2026-01-27 (correct). Figure has since posted "Helix 02 Living Room Tidy" (2026-03-09) and "Bedroom Tidy" (2026-05-08); the note does not cover these.

### C. Unverifiable (still UNVERIFIED)

- **GR00T N2 / "Isaac GR00T 2".** The NVIDIA newsroom still returns HTTP 403 to curl, WebFetch and agent-browser (re-tried 2026-10-02). Secondary sources only (an X post by @TheHumanoidHub and search snippets) say it was previewed in the GTC 2026 keynote, is DreamZero-based, gives ">2×" on new tasks, is "#1 MolmoSpaces/RoboArena", and is due by the end of the year. No release was found in the Isaac-GR00T repo (main lists N1.7 GA only).
- Gemini Robotics 2 per-task success rates: the blog has bar charts only.
- GO-2 numbers: company claims via The Robot Report, with no paper or weights.
- TRI LBM "Science Robotics 2026": not re-checked.
- π0.7 weight availability ("closed"): no open-weights release is listed on the PI blog index (the latest post is π0.7, 2026-04-16), but I found no explicit statement.
- X-VLA 98.1%: not re-checked.
- MolmoAct2 press latency figures: not re-checked.

### D. Important missed details (in scope)

1. **Figure Helix 2.5 (2026-09-17) and the Index dataset (2026-08-25).** This is the most important omission.
   - Helix 2.5 is pretrained from random init entirely on Index, a crowd-sourced human-video dataset collected through a phone app: 264k downloads, 44k WAU, "30 minutes of video uploads every second", $15M paid to creators, and "over $1B the next 12 months on data and compute".
   - Three whole-body behaviors (living-room tidy, towel folding, bed making) were tested **zero-shot in 30 unseen Bay Area homes**. Under blind evaluation with no partial credit, success was 9% from scratch vs 56% with Index pretraining, using the same task data and architecture.
   - It used half the task-specific data of a Helix 02 behavior.
   - A human-to-humanoid scaling law over 8× data forecast the largest run's loss "to four decimal places", with error 0.54% of the range.
   - Implication: frontier humanoid stacks are moving away from VLM-initialized S2. An LLM-API harness should not assume the low-level model shares a VLM's language interface. https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization ; https://www.figure.ai/news/introducing-index
2. **Gemini Robotics-ER 1.6 (April 2026)** sat between ER 1.5 and ER 2. `gemini-robotics-er-1.6-preview` "will be shut down at the end of August" (`gr_api.txt:212-216`).
   - ER 2 also reports moment finding with a "0.96s mean absolute distance" at "4x the execution speed" (`er2_blog.txt:263`).
   - ER 2 adds multi-robot orchestration.
   - New benchmark ASIMOV-Agentic tests whether the orchestrator refuses unsafe VLA tool calls (`gr2_blog.txt:208`).
   - API samples set `thinking_level: "high"` (`gr_api.txt:268`).
3. **GEN-0/GEN-1 base models are trained "without any robot data"**: "data from low-cost wearable devices on humans doing millions of activities" (`gen1.txt:45`). GEN-0 post-training used 5.6 h (1%) of task data. GEN-0 has been scaled to 10B+.
4. **Helix S1 predicts a synthetic "percentage task completion" action** to decide its own termination (`helix.txt:41`). This is a built-in completion signal, which addresses VoLo's dominant failure (completion monitoring).
5. **Anthropic study methodology** (see C2): it is a paused-simulator upper bound. Extra reasoning "made no major difference for any of the Claude-family models" on manipulation (`anthropic_robotics.txt:124`). Opus 4.6/4.7 do slightly worse with text descriptions in place of images (`:111`).
6. **VoLo real-robot CIs overlap completely**: π0.5 [6.7, 27.8] vs full [29.1, 57.8] vs No-VLA [31.2, 60.1]. The authors state that ranking the ablations "requires a larger real-robot study".
7. **MINERVA's 0.54M model has no language encoder.** It replaces language with a 40-way task ID, so it is not a language-steerable head (`2609.03715.txt:199`). On LIBERO it also found ACT-style temporal ensembling *beat* RTC soft-inpainting and BID (`:118`), which qualifies §5.3's "temporal ensembling is harmful". CPU latency is 8.9 ms vs SmolVLA 1.0 s and π0.5 12.8 s.
   - **PredVLA (2608.26673, cited but not discussed)** is the language-conditioned sub-million alternative: 0.68M parameters, a predictive-coding recurrent policy with no robot pretraining, 86.9% on the 3 short LIBERO suites and 75.4% over all 4.
8. **RL Token design details relevant to a "light head"** (`2604.23073.txt:166,566,267-268`):
   - The reference chunk is masked 50% of the time during training to stop the actor copying it.
   - The critic is a TD3-style twin-Q.
   - The base VLA was fine-tuned on 1–10 h of teleop per task, then RL ran 400–1000 episodes (≈15 min–5 h of robot data).
   - Replacing the RL token with a ResNet-10 encoder cut throughput by 50%.
9. **LeRobot v0.6.0 (2026-07-07)** (`lerobot06.txt:98,144-164,180,206-217`):
   - Ships Robometer-4B, a general reward model (Qwen3-VL-4B) that "scores task progress and success from raw video plus a language instruction". This is a drop-in local success detector for an API orchestrator.
   - Also ships TOPReward; world-model policies (VLA-JEPA, FastWAM, LingBot-VA); and MolmoAct2, EO-1 and EVO1 ports.
   - The `lerobot-rollout` DAgger strategy turns deployment into correction collection.
   - GR00T N1.7 replaces N1.5 in LeRobot.
10. **GO-2 also claims 86.6% zero-shot on LIBERO-Plus** (company claim). That is far above MINERVA's 46–56%, so it is worth independent replication before trusting it.
11. **DreamZero on RoboArena**: Elo 1750 vs π0.5 1622 in the April 2026 snapshot (NVIDIA WAM blog). DreamZero-DROID was trained only on DROID.
12. **MEM uses an off-the-shelf LLM to generate the summarization training targets** for long-term text memory. "Naive" language memory works significantly worse because of train/inference mismatch (`2603.03596` §III, lines 142-143,185). This is a direct recipe for distilling an API LLM's memory into the robot model.
13. **τ0-VLA (2608.16885, cited but not discussed)**: a hierarchical model that runs world-model-guided test-time search over candidate subtasks, trained on 40,115 h. It is the closest published analogue to "spend more API compute on hard high-level decisions".
14. **Show-Harness extras**: with the same Gemini-generated subtasks, π0.5 stays at 5% (vs 70% for the fine-tuned Qwen3.5-2B). The FT model trains in under 2 h on one H200. Data: 164 real episodes / 7.8k decision steps.
15. **Hi-VLA real-robot check** (ALOHA, 5 trials): best hierarchy 12/15 fruits, naive hierarchy 9/15, flat VLA 3/15 (`2606.10267.txt:214`).

Sources newly opened in this pass:
- https://www.figure.ai/news
- https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization
- https://www.figure.ai/news/introducing-index
- https://ai.google.dev/gemini-api/docs/robotics-overview
- https://github.com/NVIDIA/Isaac-GR00T (main and the `n1d5` / `n1d6` READMEs)
- https://github.com/huggingface/lerobot (main; tags v0.6.0 and v0.6.1)
- https://github.com/dreamzero0/dreamzero (README)
- https://www.pi.website/blog (rendered)
- https://www.therobotreport.com/agibot-releases-go-2-foundation-model-embodied-ai/
- https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/
- arXiv 2608.16885, 2608.26673, 2603.03596, 2512.20188
