# Gap 3: Which conditioning makes a light action head follow LLM targets, and does a trained head beat a frozen VLA primitive?

Date: 2026-10-02. Every number below comes from the paper text or HTML I opened (local copies in `research/tmp_vla`, `tmp_fc_vla`, `tmp_planner/txt`, `tmp_gap3`) or from my own pilot run (`research/gaps/gap-3-sim/`). Values I could not find in text are marked "n/s" (not stated).

## Bottom line

1. **No published system tests what §3.5 proposes** (LLM-pointer-conditioned 20–100M head, erasure/shift tests, vs. a frozen VLA on ≤2 mm insertion). Closest: GAE (frozen learned expert fed by VLM 3D waypoints), Fast Plans (erasure/sensitivity tests), RL Token (MLP residual on a frozen VLA).
2. **Heads follow a condition only if nothing else carries the same information and they are trained against two shortcuts** — redundant cue (Fast Plans: erasure costs 0.0 pp) and label leak (naive injection → SR 15%; π0.7 rations subgoal images to 25%). Working fixes: condition noise (GAE scale 0.10; NGM σ_c = 0.7), coarse-stage gating, removing the competing channel (RT-Trajectory). See §2.
3. **Condition dropout alone is not an anti-shortcut measure.** My pilot shows this. With a redundant scene cue, 20% condition dropout makes erasure tests pass trivially (Δ ≈ 0 pp). Yet when the condition and the scene cue disagree, 96–100% of episodes reach *neither* target. Only dropping the redundant cue (50%) produced following under conflict, and even then only 44–84%.
4. **A metric keypoint trained on exact hindsight labels turns the head into an open-loop integrator.** In the pilot it scored 100% with perfect points but only 59–60% at σ = 2 mm and 17–20% at 5 mm. That is no better than IK-to-point (44% / 8%). A tracked selection mask was invariant to pointer noise (89–94% at σ up to 10 mm).
5. **Trained head vs frozen VLA: real-robot evidence does not favour training first.** VoLo real: analytic primitives 19/42 vs π0.5 primitive 18/42; Pigey, Harness VLA and Pointing-VLA win by *staging/guiding* a frozen VLA. Learned components win in sim or on top of a generalist (GAE vs fine-tuned π0.5 on RoboTwin long-horizon 0.60 vs 0.50, and far more depth-noise-tolerant than IK; RoboDual +28 pp over OpenVLA). In my toy, analytic association + servo (B2) and analytic staging + frozen primitive (B3) beat every learned head.

## 1. Literature table (requested systems plus 3 additions)

| System | Conditioning (how injected) | Head / size | Demos | Success (trials) | Conditioning-use test | Noise / latency robustness |
|---|---|---|---|---|---|---|
| HAMSTER 2502.05485 | 2D EEF path from fine-tuned VILA-1.5-13B, drawn on RGB or as 3 extra channels; produced **once at t=0** | RVT-2, 3D-DA (n/s) | 320 teleop (220 + 50 + 50) | Colosseum: 3D-DA 0.18±0.10 → 0.43±0.05 (0.36 with 50% of data). Real, by task type: RVT2 0.28/0.13/0.17 → 0.79/0.50/0.47. Novel camera, 10 trials: overlay 0.73, concat 0.98, OpenVLA 0.23 | No erasure test. Failure attribution: RVT2 72% of failures from **not following the path**, 3D-DA 10% | Not re-planned; "cannot dynamically adjust" listed as a failure mode |
| PEEK 2509.18282 | Path + mask (8%-of-image squares; everything else blacked out); VILA-1.5-3B re-queried every H = 25–32 steps; training labels refreshed every 30–32 steps | ACT, π0, 3DDA | sim + BRIDGE | Cube stacking, 3DDA: none 33.5, path 52.8, mask 65.6, both 73.6 (±3–4). Real: 535 evals, 5 trials per task; 41.4× for sim-trained 3DDA | Path/mask ablation only | VLM 4–6 s per query on an RTX 3090. Staleness up to H steps is built into training |
| RT-Trajectory 2311.01977 | 2D/2.5D sketch concatenated as input channels (zero-initialised); **language/FiLM removed**, so the sketch is the only task specification | RT-1 | ~73K | Unseen skills: 2.5D 67% vs RT-1 17%, RT-2 11%. GPT-4 CaP sketches: **IK planner 83/71 vs RT-Traj 89/60** (pick / open drawer) | None needed (only one task channel). Changing the sketch reproducibly changes behaviour | Video-derived sketches: IK 42/25 vs RT-Traj 100/75 |
| TraceVLA 2412.10345 | **Not a target.** Overlay of the robot's own *past* CoTracker tracks (M = 5, window 6) | OpenVLA 7B / Phi-3 4B | 150K + 120 | SimplerEnv +10%; WidowX 3.5× | Fine-tuning without traces: +1.1% | Trace dropout for tracker failure (probability n/s); +0.03 s/step |
| MOKA→Octo 2403.03174 | GPT-4V keypoints execute scripted motions. Distilled Octo gets **language only** | Octo-Base 93M | 50 successful MOKA rollouts per task | Over 8 subtasks (10 trials each): zero-shot 0.74 → distilled 0.875 | n/a (no condition) | n/a |
| RoboDual 2410.08001 | DiT cross-attends to OpenVLA's discretised actions + latents; condition drop 0.1; shifted window | **16.2M** (+7.5M per extra sensor) | full / 5% | Real (15 per cell): 70.0 vs OpenVLA 41.7, DP 40.0, ACT 21.7. 5 demos: 73.3 vs ACT 0. CALVIN: removing action conditioning costs 0.8 avg length | Source ablation | Trained with generalist delay offset τ ∈ [0, k_g]; 15 Hz vs 3.9 Hz |
| RL Token 2604.23073 | Frozen π0.6 RL token + reference chunk (10×14), **reference masked 50%** | 2-layer MLP, 256 units | 400–1000 RL episodes | Up to 3× faster on critical phases (50 episodes per task) | Without pass-through: same final result, slower, more failures | Runs at 50 Hz |
| LoHo-Manip 2604.21924 | Manager VLM's 2D keypoint trace rendered into the executor's image | π0.5 / StarVLA (fine-tuned) | 100 real | LIBERO 97.5. Real results in a figure only | None | Manager runs every 100 executor steps (2 Hz vs 10 Hz) |
| Fast Plans 2609.30833 | Waypoints (joint target, gripper, duration) as suffix tokens; NGM adds a gated AdaRMS route with g(t) = clip((t−0.2)/0.3, 0, 1), endpoint noise σ_c = 0.7, null 0.15 | 300M flow expert (46–50M LoRA) | 1,693 LIBERO / 120 real | LIBERO-Long 91.0 → 96.2. Real: 48 / 46 / 51 of 60 | **Suffix tokens:** sensitivity 0.6%, erasure Δ 0.0 pp. **Naive injection:** 348% sensitivity, SR 15, visual retention 27%. **NGM:** 42.6%, retention 82%, erasure −7.4 pp | Plan latency 1094 → 125 ms. One training run per variant |
| MolmoAct 2508.07917 | Its own trace, user-replaceable (5 sketched points) | 7B | 100 | Trace steering 0.75 vs language 33 pts lower (15 trials) | Steering only | n/s |
| π0.7 2604.15483 | Subtask text + world-model subgoal image + metadata. Subgoals in **25%** of batches ("inverse dynamics", "trains significantly faster"); text dropped 30%, metadata 15% | 860M expert | large | Subgoal gains in figures only | Component dropout + CFG | Subgoal refresh every Δ = 4 s, asynchronous; trained with **generated** subgoals and 0–4 s-ahead frames |
| Embodied-R1 / R1.5 2508.13998 / 2606.11324 | Points/traces fed to CuRobo or "unified motion logic"; **no learned head** in the zero-shot results | 3B VLM | 0 task demos | R1: 87.5% on 8 XArm tasks. R1.5 on RoboTwin 2.0 (7 tasks): **65.0 zero-shot vs π0.5 fine-tuned on 400 demos/task 72.9**, π0 15.0, RDT 12.0. Real: 6 trials per task | n/a | n/a |
| + GAE 2510.03896 | VLM sparse 3D waypoints + 512-point uncoloured cloud | Frozen conv diffusion expert (150k trajectories) | 0 per task | RoboTwin short/mid/long: 0.82/0.77/0.60 vs π0.5 0.74/0.62/0.50 | — | **Training goal-noise scale** 0 → 0.71/0.67/0.43; 0.10 → 0.82/0.77/0.60; 0.50 → 0.68/0.68/0.41. **Depth noise** 0/1/10/50 cm: GAE 0.57/0.49/0.33/0.21 vs IK 0.42/0.30/0.11/0.04 (20 runs × 7 tasks) |
| + 3D HAMSTER 2606.31329 | World-frame 3D waypoints appended to the point cloud with modality embeddings | 3DFA | — | Colosseum no-variation: none 53.8, **2D 49.5 (worse than none)**, 3D 62.9. Real: 3D 80/68/62 vs 2D 60/45/46 | None / 2D / 3D | 3D-trajectory accuracy, 5 cm on both endpoints: **Sonnet-4.6 0.7%**, GPT-5.2 2.7%, Gemini-3.0-Pro 16.2% |
| + Pointing-VLA 2608.23138 | Typed point/heatmap → robot-frame cue for π0.5 (exact injection n/s) | π0.5 | — | PiPER: 79/150 → 121/150; grasp failures 47 → 16 | — | — |

**Frozen-primitive systems (the gap's caution).**
- **VoLo 2606.07723**, real (3 trials × 14 tasks): π0.5 6/42; No-VLA (SAM3 + GraspGen + IK) **19/42**; Only-VLA 17/42; full 18/42 (CIs overlap). Sim is the opposite: 17.76 / 34.97 / 41.80.
- **Pigey 2607.21725** (5 × 30 tasks): π0.5 direct 16.7%, TiPToP 48.7%, Pigey 97.3%; on plain pick-place open-loop TiPToP (80%) is *below* direct π0.5 (95%).
- **Harness VLA 2607.08448**: analytic primitives do non-contact motion and *stage* the frozen VLA ("it learns where the VLA should begin acting") — conditioning by initial state, not tokens.

## 2. Mechanisms the literature supports

- **Redundant cue**: if image + language already specify the target, BC under-uses the condition (Fast Plans 0.6% sensitivity, Δ 0.0 pp). RT-Trajectory removes language; HAMSTER's 72% RVT-2 adherence failures are the same symptom.
- **Label leak**: a state-relative target equals the integral of the imitated actions, so the head stops using vision (naive injection: SR 15%, retention 27%; π0.7 rations subgoal images for this reason).
- **Fixes**: condition noise sized like demo displacement (GAE 0.10 best, 0.50 hurts; NGM 0.7 normalised); a coarse-stage gate; null dropout 0.1–0.5 (RoboDual 0.1, NGM 0.15, RLT 50%) as a robustness add-on; training on test-like conditions (π0.7 generated subgoals, PEEK every-H relabelling, RoboDual delay offsets).
- **Measurement**: erasure Δ + sensitivity + retained visual response (any one alone misleads, per Fast Plans), plus **counterfactual shift** (condition points at a distractor) — the test that matters when the LLM decides *which* object.

## 3. CPU pilot (new evidence; a toy proxy, not the requested study)

**Why a toy.** No GPU, disk full (0–230 MB free); torch 2.14.1+cpu loaded from the uv cache. The requested YAM/RoboTwin + ACT-80M + π0.5 study was infeasible here. Code/results: `gaps/gap-3-sim/{pilot.py,results.json,summary.csv}`.

**Setup.** Peg-in-hole, **2 mm clearance**, 15 mm depth, 20 Hz; 3 identical fixtures (≥6 cm apart, tops at 2–6 cm so a 2D point lacks depth); unordered wrist detections of all holes with σ = 0.5 mm + 3% of distance; contact blocks motion and reports force. 200 demos/seed from a privileged expert with DART-style clean-label noise (EmbodiedSWE `noise.md`: executed = label + N(0, 2 mm) far, N(0, 0.3 mm) near; expert 100%). Head: memoryless slot-attention MLP (41,945 params), L1 on 8-step chunks, executes 4. Conditionings: phase only; 3D keypoint (gripper frame); 2D keypoint; tracked selection mask seeded from the LLM point; keypoint + mask. Regimes: **R0** exact hindsight; **R1** + 20% condition dropout; **R2** R1 + test-like noise σ ~ U(0, 5 mm) and staleness ≤ 2.5 cm; **R3** R2 + 50% redundant-cue dropout. Scenes: **S1** only the condition identifies the target; **S2** the target also carries a "colour" flag (instruction + image already identify it). 2 seeds × 200 paired episodes = **400/cell**; staleness s = target moved 0.5 cm/s × s (2/5/10 s → 1/2.5/5 cm). Baselines (same perception): B1 IK to the LLM point; B2 associate point → tracked detection, EMA, wrist servo, contact retry (n = 200); B3 analytic transport to point + 3 cm, then a frozen *condition-free* learned primitive (Harness/Pigey pattern).

| S1 model | σ0 | σ2mm | σ5mm | σ10mm | stale 1cm | 2.5cm | 5cm | erased | follows shifted target |
|---|---|---|---|---|---|---|---|---|---|
| B1 IK→point | 100 | 44 | 8 | 2 | 0 | 0 | 0 | 0 | 100 |
| **B2 assoc + servo** | 100 | 100 | 100 | 100 | 100 | 100 | 83 | 32 | 100 |
| **B3 staged frozen primitive** | 98 | 98 | 94 | 88 | 92 | 86 | 63 | 25 | 98 |
| phase only | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 22 |
| kp3d R0 | 100 | 59 | 20 | 5 | 3 | 0 | 0 | 0 | 100 |
| kp3d R1 | 100 | 61 | 21 | 6 | 2 | 0 | 0 | 23 | 100 |
| kp3d R2 | 91 | 90 | 88 | 72 | 86 | 54 | 1 | 24 | 90 |
| kp2d R2 | 85 | 86 | 86 | 80 | 84 | 55 | 2 | 26 | 87 |
| mask R0 | 90 | 90 | 90 | 90 | 90 | 90 | 76 | 0 | 90 |
| mask R1 | 94 | 94 | 94 | 94 | 94 | 94 | 78 | 25 | 92 |
| kp3d+mask R0 | 100 | 60 | 18 | 4 | 1 | 0 | 0 | 0 | 100 |
| kp3d+mask R2 | 90 | 88 | 86 | 71 | 87 | 58 | 0 | 22 | 88 |

Wilson 95% CIs at n = 400 are about ±3–5 pp: kp3d R0 at σ = 5 mm is 20 [17–24]; mask R1 is 94 [91–96].

**S2 (redundant cue), shift test:** share of episodes reaching the LLM's target / the cue's target / neither.

| Regime | Shift result | Erasure Δ |
|---|---|---|
| kp3d R0 | 100 / 0 / 0 | **R0 erasure Δ +100 pp** |
| kp3d R1 | 3.5 / 0 / 96.5 | **R1 erasure Δ +11 pp** |
| kp3d R2 | 0 / 0.2 / 99.8 | **R2 erasure Δ ≈ 0 pp** |
| kp3d R3 | 60.8 / 1.2 / 38.0 | — |
| mask R3 | 73.2 / 4.2 / 22.5 | — |
| kp3d+mask R3 | **84.0** / 0 / 16.0 | — |

A phase-only head with the cue ("language-only VLA" analogue) scores 90% but never follows a redirected target (0%).

**What the pilot shows** (within its limits):
- (a) **Clean hindsight keypoints build a metric shortcut.** Adding a mask does not fix it (kp3d+mask R0 behaves like kp3d R0).
- (b) **Test-like condition noise is what fixes metric fragility.** R2 at σ = 10 mm scores 72% vs 5% for R0. The cost is −9 pp at σ = 0.
- (c) **A tracked selection mask is the most noise-robust conditioning.** At 5 cm staleness (≈ half the object spacing) keypoint heads collapse to ≤2% and mask heads fall to 75–78% via wrong associations.
- (d) **Erasure Δ ≈ 0 does not mean the condition is unused.** The R2 heads still use it, and break on conflict.
- (e) **Redundant-cue dropout is needed for following.** Still only 44–84%, because training never contained a conflict. Counterfactual relabelling, where the demo goes where the condition says, is the untested next step (CAST-style).
- (f) **In the toy, both analytic baselines beat every learned head.** The toy has:
  - Gaussian detections;
  - no occlusion in the last centimetre;
  - no compliance or jamming;
  - a memoryless head, whereas B2 filters.

  A learned head therefore has to *earn* its place on exactly those missing effects.

## 4. Implications for HARNESS_DESIGN §3.5

1. **Split the condition into "which" and "where".**
   - Pass *which* as a tracked SAM3 mask or object id, refreshed every tick.
   - Let the head's own wrist perception supply *where*.
   - If a 3D keypoint is added, express it relative to the tracked object.
   - Train it with noise matched to the **measured** pointer error (Gap 2) and the spec-age distribution (Gap 1). Do not train it on exact hindsight points.
2. **Correct the design text.**
   - "Drop the target condition 10–30%" buys graceful degradation, not following.
   - Add **cue dropout** and **conflict/shift evaluations**.
   - Report success alongside erasure Δ, shift-follow rate and visual retention (the Fast Plans protocol).
3. **Make re-grounding mandatory before contact** if spec age × object speed exceeds about half the inter-object spacing (≈3 cm here) or the head's capture radius. With the design's 5–40 s spec interval this is the common case.
4. **Build order:**
   - B2-style association + servo and a staged frozen π0.5 / MolmoAct2-YAM come first; this is what VoLo, Pigey and Harness VLA show.
   - Train a head only for skills where both fail on the 2 mm suite. Candidates: GAE-style geometry experts, or RLT-style residuals over a frozen VLA.
5. **Corrections to existing notes.**
   - `landscape/vla-and-action-heads.md` lists TraceVLA as a 2D-path/trace *target* consumer. Its traces are the robot's past motion.
   - The distilled MOKA→Octo policy has no keypoint input.
   - `inspect-robots-yam/src/inspect_robots_yam/assets/yam_collision.xml` is collision-only: visual geoms stripped and no actuators. A dynamics study must use Menagerie `i2rt_yam/yam.xml` (commit 71f066ad).

## 5. Requested full study (still to run)

Menagerie YAM (MuJoCo) or RoboTwin 2.0; peg 2 mm, disc-in-groove, grasp centring; 200 DART demos/task; 5 conditionings × {0, 20%} dropout + R2/R3; ACT-80M heads; baselines π0.5 (openpi, same data), B2, B3; ≥100 episodes/cell under σ 2/5/10 mm, 0/2/5/10 s staleness, erasure, shift. Scale (UNVERIFIED estimate): ~30–40 ACT trainings at ~5 h each (2080 Ti) ≈ 150 GPU-h, 3 π0.5 fine-tunes, ~35k episodes. Add pixels with last-cm occlusion, a short-history head, and counterfactual relabelling.

## Sources

- Fast Plans, Faithful Actions — https://arxiv.org/abs/2609.30833 (Tables 3–5, §4.3)
- HAMSTER — https://arxiv.org/abs/2502.05485 (Tables 1, 2, 7; App. C.1, E)
- 3D HAMSTER — https://arxiv.org/abs/2606.31329 (Tables II–IV)
- PEEK — https://arxiv.org/abs/2509.18282 (Table I, App. A-C, Tables III–IV)
- RT-Trajectory — https://arxiv.org/abs/2311.01977 (Tables 1, 4)
- TraceVLA — https://arxiv.org/abs/2412.10345 (§3.2, §4.3, §5)
- MOKA — https://arxiv.org/abs/2403.03174 (Table I, §V-A); Octo-Base 93M — https://huggingface.co/rail-berkeley/octo-base
- RoboDual — https://arxiv.org/abs/2410.08001 (Tables 3–5, §3.2)
- RL Token — https://arxiv.org/abs/2604.23073 (§V, Fig. 7–8)
- LoHo-Manip — https://arxiv.org/abs/2604.21924 (§3.4, App. C.3)
- MolmoAct — https://arxiv.org/abs/2508.07917 (§5.6, App. D.7)
- π0.7 — https://arxiv.org/abs/2604.15483 (§V–VII, App. A)
- Embodied-R1 — https://arxiv.org/abs/2508.13998 (§3.4); Embodied-R1.5 — https://arxiv.org/abs/2606.11324 (Tables 12–13)
- GAE — https://arxiv.org/abs/2510.03896 (Tables 2, 6, 7, 16, 19)
- Pointing-VLA — https://arxiv.org/abs/2608.23138 (Tables 2, 6)
- VoLo — https://arxiv.org/abs/2606.07723 (Tables 2–3, App. G)
- Pigey — https://arxiv.org/abs/2607.21725 (Tables 3, 13; App. C, H)
- Harness VLA / RPent — https://arxiv.org/abs/2607.08448 (Key Findings 1–3)
- ACT — https://arxiv.org/abs/2304.13705 (80M, 50 demos, ~5 h on 2080 Ti)
- EmbodiedSWE noise channel — `repos/EmbodiedSWE/data_engine/agent/prompts/noise.md`
- inspect-robots-yam — `repos/inspect-robots-yam/src/inspect_robots_yam/assets/yam_collision.xml`, `scripts/gen_collision_model.py`
- Pilot — `research/gaps/gap-3-sim/{pilot.py,results.json,summary.csv,run.log}`
