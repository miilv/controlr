# HARNESS_DESIGN.md: evidence-grounded design for a frontier-LLM robot-control harness

*For Ilia Mikhalchuk. Written 2026-10-01/02 from the 22 research notes in `research/sources/` and `research/landscape/`, all read in full including their fact-check sections. Where a fact-check corrected a note body, this document uses the correction.*

*Revision 2 (2026-10-02, final edit): every item in `CRITIQUE.md` "## Errors" was re-verified against primary data and applied, modified or rejected (log in `REPORT.md` §10). New primary measurements from the six gap studies are integrated: `[gap-1]` per-call latency and cost (11,768 logged Tier 1 calls), `[gap-2]` pointing accuracy, `[gap-3]` action-head conditioning (literature plus a CPU toy study), `[gap-4]` forensic coding of all 360 Tier 1 trials, `[gap-5]` independent-verifier accuracy on 335 labelled episodes (1,782 verifier calls), `[gap-6]` YAM ≤2 mm feasibility. `[gap-N]` refers to `research/gaps/gap-N.md`. REPORT.md is the companion landscape report; the two documents use the same numbers.*

**Conventions**
- `[note: slug]` refers to `research/sources/<slug>.md` or `research/landscape/<slug>.md`. URLs are given for the few facts fetched separately for this document.
- **[derived]** marks arithmetic done here from cited numbers.
- **UNVERIFIED** marks anything not confirmed at a primary source.
- Trial counts are given wherever the source gives them. Treat any n ≤ 10 as anecdotal (see §5.2 for why).
- Code paths are relative to `research/repos/`, and every path cited was checked to exist in the local clone.

---

## 0. Executive summary

1. **The binding constraint is the clock, not intelligence.**
   - Measured on real YAM rigs (Robocurve Tier 1, medium effort, 224 px frames, 120 trials per model), the client wall time per non-streaming call is p50 **5.28 s**, p90 11.72 s, p99 21.1 s, max 60.3 s for Opus 5.5 (n = 4,105). GPT-6 Astra: 6.53 / 11.28 / 19.6 s. Opus 5: 8.77 / 20.79 / 40.5 s. The first call of a trial is about 3× slower (Opus 5.5 p50 16.1 s) [gap-1].
   - About 84% of trial wall time is waiting on the LLM; the arm moves 28 s of 269 s. The realized decision rate including motion is **0.127 Hz** (7.6 calls/min) [gap-1].
   - Latency is mostly decode: Opus 5.5 ≈ 1.8 s + output tokens / 87 tok/s (R² 0.91, 424 output tokens per call). Output length, not prompt size, is the lever [gap-1].
   - Contact control needs 30–200 Hz [note: vla-and-action-heads].
   - So the LLM must be **event-driven (plan on ≈0.1 Hz at medium effort) and never inside a control loop**. Something local has to own every millisecond between calls.
2. **The interface matters more than the model, within the frontier tier.**
   - URAI (5 RoboDojo tasks × 5 seeds): every agent roughly tripled its success when it chose *between* multi-phase tools instead of commanding fingertips directly (overall 18% → 53%, McNemar p = 7.9×10⁻⁸). The tools were **written and frozen beforehand by a Claude Fable 5.1 programming agent** (Opus 5.5 wrote one pouring tool), and a pre-written program alone scored only 24%. The gain is therefore tool authoring plus the interface, not the interface alone [note: frontier-models-as-direct-policies; arXiv 2609.39018 §4].
   - Direct 7-D EEF control on LIBERO scored 0–5.5% for every model Anthropic tested [note: llm-planner-codegen-lineage].
   - Gemini scored 0/4 emitting Cartesian deltas but 6/6 emitting an image point plus phase for a geometric controller (small n, not a clean A/B) [note: manda-robotics].
3. **Claude's weaknesses are narrower than first reported, and mostly fixable around the model.**
   - **Pointing.** The only per-benchmark table with Claude is PhysBrain 1.5 Table 4, which used **Opus 5 at adaptive-low effort**, not Opus 5.5. Opus 5 trails on RefSpatial (56.3 vs 78.0, −21.7, n = 277) and PixMo-Points (−18.5); it is within 3.3 points of the best frontier model on PointBench, RoboAfford, RoboRefIt, VABench-Point and RoboSpatial-Home; and it **leads every frontier model on Part-Affordance** (78.1 vs Astra 55.0, Gemini 3.6 Flash 64.7). The public eval kit downsizes to 336 px and asks API models for 0–1 normalized coordinates, which Anthropic's docs say Claude handles poorly, so the RefSpatial gap is plausibly partly a protocol artifact. **No pointing number exists for Opus 5.5, ER 2 or ER 1.6**, and no model has a millimetre error against real-robot ground truth [gap-2].
   - **Overclaiming is a field-wide problem, not a Claude-specific one.** Among non-completed Tier 1 trials, Opus 5.5 falsely claimed success in 10/119 (8.4%) vs Astra 12/115 (10.4%), Fisher p = 0.66. 19 of the 22 false claims are Stack Bowls trials graded 0.75 whose decisive region was not visible at 224 px. Opus 5.5's other 34 `done` calls were honest partial reports ("ran out of call budget") [gap-4]. The older CodeActionBench figure (Opus 5 13/38 vs Astra 3/20) is p = 0.22 and mixes harnesses [note: benchmarks-and-sim-datagen].
   - **Progress estimation**: Opus 5 scores 37.1 vs ER 2 57.4 on Google's vendor chart [note: benchmarks-and-sim-datagen]. No Opus 5.5 number exists.
   - **Strengths (all Opus 5, not 5.5)**: code (TAMP 82%), video success detection (81.0, best non-ER, vendor chart), safety-instruction following (95.9) [note: frontier-models-embodied-2026]. Opus 5.5 "thinks more per turn than Opus 5 at the same effort" (Anthropic migration notes), so transfer is an assumption.
   - Design consequence: **the LLM chooses *what/which/when*; local geometry, servoing and (where they earn it) learned skills supply millimetres; an independent verifier with its own evidence (views, sensors) decides "done".**
4. **Recommended architecture (§3).**
   - An Opus 5.5 orchestrator at `medium` effort over a typed, strict tool surface, with server-side argument validation.
   - Perception and grounding services: SAM 3 / Grounding DINO / depth. A VLM point is a **seed**, refined by mask, depth or CAD fit; the pointer model (Opus 5.5 with zoom, ER 2, or a local 8B model) is chosen by the measured eval in [gap-2] §5, not by default.
   - A deterministic geometry and motion layer: calibrated IK with integral settle, Ruckig, approver-chain envelope.
   - **Contact skills** invoked as leased, closed-loop tools for the last centimetre. Build order: analytic target association + wrist-camera visual servo + compliance first; a fine-tuned VLA primitive second; a small trained head (20–100M) only for skills where both fail on the 2 mm suite [gap-3; gap-6].
   - The same harness, run offline, is the **data engine** that trains heads and critics (EmbodiedSWE-Gen style).
5. **The niche is narrower than "unfilled".**
   - Adjacent cells are occupied: Pigey (Opus 4.7 + frozen π0.5 `VLARollout` + TAMP, 97.3% over 150 real trials), Harness VLA / RPent (Claude Agent SDK planner + frozen π0.5 primitive), VoLo (Opus 4.6 + π0.5 on a real Franka), AWS Strands (Claude Sonnet 4.5 + GR00T on SO-101), RoboDual and RL Token (small heads under a big model) [note: llm-planner-codegen-lineage; note: vla-and-action-heads; note: industry-competition].
   - What remains open is specific: a **task-trained, target-conditioned small head under a frontier API, shown to follow LLM targets (erasure and shift tests) and to beat both a frozen VLA primitive and an analytic servo baseline** on real ≤2 mm contact tasks. No paper tests this; VoLo's real robot had No-VLA 19/42 vs full system 18/42, and in a toy study analytic baselines beat every learned head [gap-3].
6. **Model choice: Opus 5.5 is a defensible default; the evidence does not separate it from Astra.**
   - Robocurve Tier 1 (real YAM, 6 tasks × 20): mean progress 36.0% vs Astra 36.7%; completions 1/120 vs 5/120 (**Fisher p = 0.21**); $0.90 vs $1.14 per trial. The models ran in blocks per rig (20 Astra, then 20 Opus 5.5, then 20 Opus 5), graders knew the model, and the tasks were the six easiest for Astra out of 18 [gap-4].
   - Failure signatures differ: placement/alignment blocks 50% of Opus 5.5 failures vs 39% for Astra; Astra fails more at grasping (37% vs 13%, p = 1.7×10⁻⁵) [gap-4].
   - URAI: 15/25 vs 13/25 with tools (p = 0.78), 8/25 vs 4/25 native (p = 0.32): **no detectable model difference at n = 25**, and the Claude agents ran at xhigh vs Astra at medium.
   - Astra leads most other head-to-heads (LIBERO-Agent, CodeActionBench, Robocurve bowl), all on Opus 5 or earlier. Build provider-agnostic from day one and A/B on your own tasks.
7. **Cost is manageable if the transcript is append-only and cached.**
   - Measured: Opus 5.5 costs **$0.0262 per call and $0.898 per Tier 1 trial**; 54% of that is cache writes (2,826 written tokens per call), because Inspect Robots' image eviction rewrites earlier turns [gap-1; gap-4].
   - Projected for this design's request shape (append-only, 10k cached prefix, one 1920×480 tile ≈ 1,242 tokens per turn): ≈ $0.022–0.034 per call at history depth 5–35, ≈ **$1.0 per 35-call episode cached vs ≈ $6.7 uncached** [derived, §1.4]. With 2–4 calls per pick-place, a pick-place costs ≈ $0.05–0.12.
   - History rewrites break the cache and, for accounts created on or after 2026-08-31, make replayed thinking blocks a 400 on Opus 5.5 [note: quackd; Claude API reference].
8. **Safety must sit below the model.**
   - RoboDojo stopped real Astra testing after hardware damage (1/33 full successes) [note: robodojo].
   - Agents collide with obstacles they explicitly reason about. Harness-level route checks raised collision avoidance from 50.0% to 87.5% [note: llm-planner-codegen-lineage].
   - Text written in the scene hijacked GPT-4o 27.0% and Gemini 2.5 Flash 29.4% of the time; Claude and GPT-6 were not tested, so the Opus 5.5 baseline is unknown [note: engineering-challenges].
9. **Evaluation must be stricter than the field's.**
   - Repeating the same seeds reproduced only 64% of π0.5's successes [note: manda-robotics].
   - Separating 50% from 70% needs about 93 trials per arm [note: engineering-challenges].
   - Claude rejects `temperature`, so sampling cannot be pinned [note: quackd].
   - The only real-robot Opus 5.5 benchmark is order-confounded and unblinded [gap-4]. Use blinded, interleaved, paired protocols with ≥50 trials per arm for headline claims (§5).
10. **MVP (§6).**
    - Hardware: bimanual I2RT YAM ($2,999 per arm) with two YAM leaders for bimanual teleop, D405 wrist and D435-class overhead cameras: ≈ $13.0k plus GPU. Robot time (≈66–112 robot-hours per 900-trial campaign), not API dollars, is the binding budget.
    - First: a 5-call smoke test of Inspect Robots on Ilia's own Anthropic account (preserved-thinking check), then the one-day YAM bench protocol of [gap-6] §6.
    - Baseline arm on Ilia's rig: Inspect Robots + Opus 5.5 and Astra, interleaved and blinded.
    - The differentiating result is not "beat 2/20". It is: **beat a scripted wrist-servo + compliance baseline on the same rig** at ≤2 mm clearance (AGP-style scripts already reach 8/10 and 5/5 on YAM), at equal or lower wall-clock [gap-6].

---

## 1. Requirements and constraints derived from the evidence

### 1.1 Latency

| Layer | What it needs / delivers | Evidence |
|---|---|---|
| Servo / whole-body | 200 Hz–1 kHz | Helix S1 200 Hz, S0 1 kHz [note: vla-and-action-heads] |
| Learned action head | 30–50 Hz action streams via chunking; tolerates ~100–300 ms inference delay with RTC | Gemini Robotics: 50 Hz effective, ~250 ms end-to-end. RTC robust beyond 300 ms; training-time RTC handles 200 ms on a 50 Hz robot [note: engineering-challenges] |
| VLA inference (GPU) | π0.5-DROID 128 ms/query (RTX PRO 5000). Cosmos3-Nano 829 ms. GR00T N1.7 251 ms in Manda's adapter; official 35.9 Hz TensorRT on H100 | [note: manda-robotics; note: vla-and-action-heads] |
| VLA inference (CPU, don't) | ACT ~620 ms round trip on an i5 laptop. SmolVLA 169–188 s per chunk in bf16 | [note: quackd] |
| Perception | Grounding DINO tiny ~70 ms (4090, measured in sim). SAM 3 ~200 ms per phrase (4090). Cloud ZoeDepth 136 ms RTT, OWLv2 216 ms | [note: piper-astra-jev; note: general-robotics-auto-engineering] |
| Typed decision model | Jev 70–500 ms (vendor), 236–276 ms p50 (independent), 527 ms p50 via OpenRouter. Kev-4B 18.1 ms server-side on H100. Laya 39.5 ms on T4 | [note: frontier-models-embodied-2026; note: piper-astra-jev] |
| Frontier LLM, measured in robot loops (per HTTP call, client wall time, non-streaming) | **Tier 1, medium effort, real YAM, 120 trials each:** Opus 5.5 p50 5.28 / p90 11.72 / p99 21.1 / max 60.3 s (n = 4,105); first call p50 16.1 s; final `done`/`give_up` call p50 12.1 s. Astra 6.53 / 11.28 / 19.6 / 38.9 s (n = 3,437). Opus 5 8.77 / 20.79 / 40.5 / 67.8 s (n = 4,226). Zero errors, retries or timeouts in 11,768 calls [gap-1]. Earlier reports: Robocurve bowl Astra median 5.3 s, p90 11.5 s; Fable 5.1 via compat shim 12.4 s; Opus 5 high on clapboardbench 10.8–14.4 s; Opus 5 fast mode at high effort p50 12.5 s at 2,404 output tokens/call. RoboDojo "Astra 14.66 s" is a ByteDance internal-proxy figure on a shared cluster ("compares runs, not models"). Astra xhigh 29–47 s [derived]. Astra locomotion 39.86 s with physics paused | [gap-1; note: robocurve-gpt6-astra; note: engineering-challenges; note: robodojo] |
| Frontier LLM, vendor-speed snapshot | Opus 5 time to first answer token: low 3.18 / medium 6.91 / high 10.32 / xhigh 26.69 / max 36.98 s, at ~50 tok/s. Haiku 4.5 0.69 s. Gemini 3.7 Flash 1.30 s at 302 tok/s. Opus 5.5: Artificial Analysis lists 92.2 tok/s (max effort only), matching the 87 tok/s fitted from Tier 1. **No Opus 5.5 TTFT under a robot request shape exists**; Opus 5.5 low/high effort, fast mode, Sonnet 5.5 `between_tools`, Astra `ultrafast` and ER 2 pointing are all unmeasured (benchmark script ready: `gaps/gap-1-bench.py`, ≈$285 dry-run estimate) | [note: frontier-models-embodied-2026; gap-1] |
| Gemini Robotics-ER 2 | Only real-robot data: as a **joint controller** at `reasoning_effort=high`, p50 5.81 s, p90 9.55 s (n = 91, clapboardbench). **Pointing latency is UNVERIFIED**; the docs advise "Query multiple times and average results", which multiplies it | [gap-1] |
| Network | One-way LAN 1.2–2.6 ms; US-West 3.7–8.4 ms; US-East 36–75 ms | [note: general-robotics-auto-engineering] |

Two consequences follow.

**(a) The LLM runs at roughly 0.03–0.2 Hz (0.127 Hz realized in Tier 1 at medium effort; 0.2 Hz needs about 60% fewer output tokens per call), so its outputs must stay valid for 5–40 s of world time.**
- Use object-centric and frame-relative targets that local perception re-binds at execution time, not time-indexed pixel coordinates [note: engineering-challenges].
- innate-os discards an LLM action if the end effector moved more than 1 cm during the call [note: innate-os-pr817]. That stale-observation guard is mandatory.

**(b) The world does not pause, but most published numbers do.**
- RoboDojo, RoboLab-120, GPT-as-Policy and Anthropic's direct/code locomotion runs all pause physics while the model thinks [note: robodojo; note: manda-robotics; note: gpt-as-policy; note: engineering-challenges]. Anthropic did *not* pause its manipulation sim.
- On hardware, waiting has a physical cost:
  - MARS's shoulder trips overload when the arm holds a loaded pose while the model thinks [note: innate-os-pr817];
  - three rig-3 Fable trials ended in thermal `overheat` on YAM [note: robocurve-gpt6-astra];
  - Tier 1 had 14/360 overheat terminations (Opus 5 10/120, Opus 5.5 2/120, Astra 2/120). All 10 Opus 5 trips fell in positions 41–60 of each rig's 60-trial sequence, so they track run order and accumulated heat, not the model. One trip came 109 s into a trial started about 40 s after the previous one ended: heat carries over between trials [gap-4; gap-6].
  - Holding a top-down pose loads the YAM shoulder (DM4340, rated 9 N·m) with 9.5–13.7 N·m beyond about 0.3 m reach [derived, gap-6]; a measured Tier 1 trace peaked at 18.7 N·m, and the idle arm parked at "home" held 7–10 N·m on J3 [gap-6].
- **Requirement:** the harness parks or unloads the arm when a loaded arm waits on the LLM, using load, torque and motor temperature rather than a fixed timer. A fixed 8 s timer would fire on 23% of Opus 5.5 calls (≈7.9 per trial, and on every first call); a 15 s timer fires on 4.7% [gap-1]. Park the idle arm folded, keep fixtures within r ≈ 0.35 m, and schedule cool-downs between trials [gap-6].

**Where time actually goes.** In Tier 1, 84.5% of Opus 5.5 trial wall time is LLM time; the arm moves 28 s of 269 s, and 13.7 s is reset/IO [gap-1]. In Robocurve's bowl task, the robot moves for about 11 s of a 153 s Astra trial [note: robocurve-gpt6-astra]. In an AGP 48.6-min four-pair assembly, 68.2% is planning or reasoning and 23.5% is robot motion [note: benchmarks-and-sim-datagen]. Fewer, larger decisions and shorter outputs are the main levers.

### 1.2 Precision and grounding

**Tolerances.** Real tasks need 1–8 mm: VIA's T-block tolerates about 8 mm; SafeLIBERO counts a >1 mm obstacle displacement as a collision [note: engineering-challenges].

**Pixel resolution.**
- One Claude visual token (28×28 px) covers 10–20 mm at 1920 px, 30–60 mm at 640 px and 86–172 mm at 224 px, at 0.5–1 m [derived; note: engineering-challenges].
- Per pixel on a D435 colour frame (1920×1080, fx ≈ 1397): 0.36 mm at 0.5 m unresized; 2.05 mm at the 336 px used by the PhysBrain eval kit; 3.07 mm at Robocurve's 224 px squash. A 5 mm radius at 0.5 m is 14 px on the unresized frame and 1.6 px at 224 px. D405 (1280×720): 0.74 mm/px at 0.5 m [derived, gap-2 §4].
- Robocurve sent 224×224 squashed frames: the Tier 1 page states it, and 2,196 decoded frames from the Astra/Fable report are all 224×224 [gap-4; note: robocurve-gpt6-astra]. Claude's docs warn that accuracy degrades below 200 px [note: engineering-challenges].

**Measured pointing accuracy.**
- Every pointing benchmark scores *point-in-mask hit rate*, not distance. Masks are object-, part- or free-space-sized, so the scores bound semantic grounding, not millimetres [gap-2].
- The only metric data on a robot tabletop (arXiv 2609.28184; 151 scenes at 1920×1080; ArUco ground truth): median localization error **GPT-5.4 16.8 px** (RMSE 48.6), **Claude Sonnet 4.6 29.3 px** (RMSE 70.9). That is roughly 7–15 mm and 13–26 mm at 0.5–1 m [derived]. The reference method itself has a floor of about 14 px. No Opus 5/5.5, ER or Astra was tested [gap-2].
- Implication: a GPT-5.4-class 16.8 px error is about 6 mm at 0.5 m on a D435. It misses a 5 mm radius and passes a 10–20 mm one. D405 depth tolerance is ±10 mm at 50 cm, so axial error can exceed the lateral budget. **A VLM point is a seed; the last millimetres come from mask snapping, patch-median depth, plane/CAD/edge fits, relative servoing or contact** [gap-2].

**Where frontier models fail on precision.**
- Astra scored 4.0% SR on RoboDojo Precision vs 30.5–38.7% on Generalization, Memory and Open [note: engineering-challenges].
- Puzzle-into-groove was 2/20 for both Astra and Fable 5.1, which stalled "at the same final step" (Fable 5.1 is 1/20 by the live operator verdict; one completion was re-scored after the run) [note: robocurve-gpt6-astra].
- In Tier 1, "placement/alignment" blocks 50% of Opus 5.5 failures, but few are true millimetre misses: about 11 are fruit dropped from 10–15 cm that bounced out, Cap Pen failures are mostly bimanual arm-body interference and unknown cap orientation (only 2/18 report a ~1–1.5 cm residual), and 8 are bowls not seated [gap-4].
- In-hand rotation: Astra 0.51% vs RL 76.9% [note: frontier-models-as-direct-policies].
- VIA's T-block stays at 10–40% [note: frontier-models-as-direct-policies].

**Claude-specific grounding facts.**
- Use absolute pixel coordinates in the *resized* image; Claude "does not work well" with normalized 0–1000 coordinates.
- A silent server-side resize is "the most common cause of misaligned coordinates". Pre-resize locally, or set the per-image-block guard `"transformations": {"oversized_image": "error"}` (the field is nested under `transformations`, not a bare `oversized_image`) [gap-2; note: frontier-models-embodied-2026]. Opus 5.5 is on the high-resolution tier (2,576 px long edge / 4,784 visual tokens), so a 1920×1080 frame (2,691 tokens) is not resized.
- Gemini Robotics-ER is the opposite: `[y, x]` normalized to 0–1000 [note: frontier-models-embodied-2026]. 0–1000 units are ≈1.92 px on a 1920-wide frame, about 0.7 mm at 0.5 m, so quantization is negligible [gap-2].

**Fixes with measured effect.**

| Fix | Effect | Source |
|---|---|---|
| Calibrated depth | +20 pp short-horizon SR for every agent (Opus 10→30, Astra 50→70, Fable 0→20) | LIBERO-Agent [note: frontier-models-embodied-2026] |
| Perception tools | Astra 61.1% → 88.9% (n=18) | Robo-Harness K1 [note: frontier-models-embodied-2026] |
| Gripper-cam "cursor" tool | Mythos 6% → 32% | Anthropic [note: engineering-challenges] |
| Interaction-point marker | 40% → 85% (20 trials) | Show-Harness [note: engineering-challenges] |
| Coordinate grid | +15 pts (47.0 vs 32.4) | RoboDawn [note: frontier-models-as-direct-policies] |
| Constraints + solver instead of numeric poses | 63% vs 37% | CoPa [note: llm-planner-codegen-lineage] |
| Code instead of NL for spatial reasoning | 98% vs 35% | Code as Policies [note: llm-planner-codegen-lineage] |
| Code instead of raw number lists | 60% vs 10% | Kwon [note: frontier-models-as-direct-policies] |

**Geometry is where the real engineering sits.**
- piper-astra-jev reaches 0.6 mm relative alignment for a peg-over-spanner insertion using CAD-silhouette fitting and finger-masked *relative* visual servoing [note: piper-astra-jev].
- That 0.6 mm was measured by the same camera that drove the correction, and the in-hand fits had IoU of only 0.78–0.79.
- On chrome, 17% of mask pixels had no depth, and camera-only grasp points drifted up to 21 mm over 17 still frames.

**The arm itself is a precision limit on a $3k YAM** [gap-6].
- The YAM is a **low-gain joint-impedance arm**: Damiao MIT mode with Kp 80/80/80/10/10/10 N·m/rad, Kd 5/5/5/1.5/1.5/1.5, host-side MuJoCo gravity compensation at 250 Hz, no integrator. Grasp-point stiffness is about 0.34 N/mm in the softest direction [derived]. Compliant insertion is natural; absolute accuracy is poor.
- I2RT publishes **no repeatability or backlash figure** for any YAM tier. Measured on rigs: free-space poses sag 1–3 cm below the commanded height (Tier 1 rig facts); tracking misses by a median 2.2 cm (p90 5.2 cm) and sags 1.6 cm in z for Opus 5.5's free-space moves [gap-4]; AGP's integral settle loop reaches 1.6–3 mm (left arm) and 4–7 mm (right arm); factory encoder zeros were off by up to 5.47°, and an 11-pose calibration took one arm from 8.32 mm to 1.36 mm RMS [gap-6].
- Robocurve's "−0.2 to −0.35 rad pitch sag" is mostly the harness compounding a ~0.013 rad mechanical settle (measured-state seeding, oscillation hold, 0.2 rad/tick clamp, 0.35 rad resync), not mechanics [gap-6].
- The MIT wire format caps Kd at 5, which J1–J3 already use, so a stiffer shoulder becomes under-damped. Effort is current-estimated; with 0.3 N·m Coulomb friction the usable contact-detection threshold is about 1 N [gap-6].

**TCP bookkeeping is a classic silent bug.**
- Manda's connector adds 0.1034 m, a Franka-hand value, to a Robotiq `base_link` whose fingertip is about 0.15–0.163 m ahead. That would aim 5–6 cm short [note: manda-robotics].
- RoboProbe uses a 0.1501 m grasp offset; GPT-as-Policy explicitly warns against reusing DROID's +0.1311 m pad offset on ARX X5 [note: robodojo; note: gpt-as-policy].
- **Requirement:** frame and TCP constants live in one tested module, with pure-numpy unit tests in the style of `manda-RoboLab-Verified/policies/vlm_pinpoint/connector.py` and its tests.

### 1.3 Verification

**Self-reported completion is unreliable for every frontier model.**
- **Tier 1 (real YAM, the strongest data):** false success claims among non-completed trials were Opus 5.5 10/119 (8.4%, Wilson 4.6–14.8%) vs Astra 12/115 (10.4%), Fisher p = 0.66. 19 of the 22 false claims are Stack Bowls trials graded 0.75 ("overlapping") whose final 224 px frames look the same as Astra's three 1.0 trials; the cap/pen junction cannot be resolved, and lying bottles look upright from the camera angles. These are mostly **unobservable** claims, not hallucinations. Opus 5.5's other 34 `done` calls were honest partial reports when the call budget ran out: it used `done` as "end of episode, here is my report" [gap-4].
- CodeActionBench: Opus 5 (reference harness) claimed success on 13/38 failures (34.2%); Astra on 3/20 (15.0%), p = 0.22, mixing harnesses and using Opus 5 [note: benchmarks-and-sim-datagen]. Secondary, non-significant evidence only.
- Robocurve: Astra called `done` in 7/20 unfinished puzzle trials; Fable 5.1 claimed "The wrist camera confirms the disc seated" at stage 0 [note: robocurve-gpt6-astra].
- piper-astra-jev's `object_placed` is a kinematic assertion. Jev declared `done 0.96` with the trowel hanging over the crate rim [note: piper-astra-jev].

**Completion monitoring is the dominant error even with good orchestrators.** In VoLo, completion-monitor errors are more than 67% of all error events [note: vla-and-action-heads].

**Independent verifiers: measured accuracy** [gap-5; 235 real YAM episodes from two Robocurve reports + 100 RoboDojo-sim episodes; 1,782 verifier calls].
- Opus 5.5 at `effort:"low"` reading a Manda-style contact sheet plus the full-resolution final frame: balanced accuracy **0.871** [0.808–0.923], false-success rate 13.3% (26/195), AUROC 0.969, $0.017 per check, median 3.8 s sequential through a proxy.
- **By predicate class:** placement tasks (block in bowl, store in safe, sort, pack) 0/92 false accepts (BA 0.968); contact tasks (puzzle groove, stack bowls, cap pen, stand bottles) **BA 0.707, false-success 25.2%**. Every false accept came from contact tasks. This reproduces FailBench's contact gap (best detector 0.77 overall, ≤0.60 on assembly) on Claude.
- **A different vendor does not de-correlate errors.** GPT-6 Astra as verifier: BA 0.886 at $0.035/check, statistically the same; Opus and Astra agree on 96.6% of episodes (κ 0.91) and share 75% of their errors. A 2-of-3 vote (Opus, Astra, GPT-5.6 Sol) gives BA 0.891, false-success 11.8%.
- **A different view de-correlates better.** Opus on the end frame alone AND Astra on the contact sheet: false-success 2.1% (4/195), but it rejects 22.5% of true successes, and this pairing was picked post hoc from 10 tested. A unanimous 3-model gate: 5.6% false-success at the same false-reject rate.
- **The gate repeats the agent's mistakes**: on the 22 coded Tier 1 false claims, the Opus verifier accepted 19 and Astra 18. At 224 px the verifier sees what the agent saw.
- Effort did not help: high vs low effort BA 0.724 vs 0.754 on the 175 hardest episodes; 5 verdicts changed; 2.9× output tokens.
- End-frame-only Opus mistook the painted blue groove for a seated piece in 13/56 failed puzzle trials; the contact sheet cut this to 6/56 but made models infer "stacked" from motion history (20/20 accepts on 0.75 stacks).
- Progress estimation zero-shot is near chance: 34.4% vs 33.3% random; SFT reaches only 45.4% (RoboProcessBench T5) [note: benchmarks-and-sim-datagen]. ER 2's 82.4 video success detection and Robometer-4B's F1 0.81 remain vendor/author numbers, not re-measured here.

**What works.**
- *Deterministic sensors.* A no-vision gripper predicate "held (aperture 0.05–0.9, effort > 0.7), then ended open" passed 31/32 successes and rejected 64/88 failures in the Astra report; AND the Opus verifier gave BA 0.942 with 2/88 false accepts. The same thresholds passed 0/5 true bowl stacks in Tier 1, so thresholds are per object [gap-5]. Pigey's system (with an `is_grasped` check inside a verify-and-retry loop) reached 100% vs TiPToP open-loop 80% on 4 tasks × 5 trials; the paper credits the closed loop, not `is_grasped` alone [note: llm-planner-codegen-lineage].
- *Proprioceptive grasp semantics.* metal-arm-harness uses stall → "holding", torque below 0.4 N·m → "resting on the object", and "OBJECT LOST" [note: metal-arm-harness]. On YAM, gripper effort spikes to ≈0.7–1.2 on every full close, empty or not, so effort alone cannot tell an empty close from a grasp; about half of all Tier 1 closes came up empty (Opus 5.5 48%, Astra 57%) [gap-4; gap-6].
- *Aperture vs the demo's empty stop.* innate #817 recovered a missed grasp this way [note: innate-os-pr817].
- *Intermediate verification.* EmbodiedSkills (Qwen3-VL planner over task-adapted π0.5 executors, arXiv 2609.01281): full system 86.2% on RoboTwin 2.0; without intermediate verification 48.2%, i.e. **about 34 pp below the bare π0.5 reference (82.7%)**. Read it as: an agent loop that decomposes and retries without verification destroys VLA performance; verification only restores it (+3.5 pp over the reference) [note: vla-and-action-heads; note: github-implementations-sweep].
- *Success only when objects are at rest.* Grasps require a carry; commanded release is distinguished from a drop [note: manda-robotics].
- *A dedicated verification view or physical probe* (oblique or side view at ≥448 px zoomed on the container; a pull/lift test) for seating, stacking, capping and upright predicates [gap-4; gap-5].

### 1.4 Cost

**Prices** (per MTok, current):

| Model | Input | Output | Cache read | Notes |
|---|---|---|---|---|
| Opus 5.5 | $4 | $20 | $0.20 | Cache write $5 (5-min) / $8 (1-h); fast mode $8/$40 |
| Fable 5.1 | $10 | $50 | $0.25 | |
| Sonnet 5.5 | $2 | $10 | — | |
| Haiku 4.5 | $1 | $5 | — | Retirement "not sooner than October 15, 2026" |
| GPT-6 Astra | $10 | $50 | $1 | Cache write $12.5; ultrafast tier $60/$300 |
| Gemini Robotics-ER 2 | $1 | $5 | — | Through 2026, then $2/$10; free tier |

Sources: [note: engineering-challenges; note: frontier-models-embodied-2026; note: industry-competition].

**Measured per-trial costs.**
- **Robocurve Tier 1, real, reproduced from the 120 run JSONs** [gap-1; gap-4]. Opus 5.5 mean per trial: 717,344 input tokens = 620,537 cache-read + 96,670 cache-write + 137 uncached; 14,496 output tokens (reasoning included). At $4/$20, cache reads $0.20 and 5-min writes $5/MTok this gives **$0.898** ($0.0262 per call over 34.2 calls), matching the page's $0.90. Split: cache writes $0.483 (54%), output $0.290 (32%), cache reads $0.124 (14%). The 2,826 written tokens per call are larger than one new turn (3 × 64 image tokens + state), consistent with Inspect Robots' `_evicted_view` rewriting earlier user turns as frames age out of `image_horizon=2`.
- Opus 5 Tier 1: $1.758 ($0.050 per call). Astra Tier 1: the page's cache-adjusted estimate is $1.14; at its recorded cache usage (cache-read share 0.33) it would be $3.07 ($0.107 per call) [gap-1].
- Robocurve bowl: Astra $0.94; Fable 5.1 $2.12, sent through the OpenAI-compatible shim with **no prompt caching** [note: robocurve-gpt6-astra].
- AGP two-pair assembly: Opus 5 $9.75 (12.4M tokens, 22.2 min); Astra $4.47 [note: llm-planner-codegen-lineage].
- GPT-as-Policy: ≈$17 (Hybrid) to ≈$29 (Direct) per episode at Astra xhigh list prices [note: gpt-as-policy].
- RoboDojo L3: ≈$70 per *success* for Astra [derived; note: engineering-challenges].

**Projection to this design's request shape** [derived, following gap-1 §4; UNVERIFIED until `gap-1-bench.py` is run].
- Assumptions: append-only history (no rewrites); cached prefix P = 10k tokens (tools, system, robot card); per turn ≈ 2.0k new tokens written (one tiled 1920×480 triptych ≈ ⌈1920/28⌉·⌈480/28⌉ = 69·18 = 1,242 image tokens + ~300 text + ~424 replayed assistant output); 424 output tokens per call; n = 35 calls.
- Per call: write ≈ 2.0k × $5/MTok = $0.010; output 424 × $20/MTok = $0.0085; read (10k + 2k·d) × $0.20/MTok. That is ≈ **$0.022 at depth 5, $0.028 at 20, $0.034 at 35** — within 1.3× of Tier 1's $0.026.
- Per 35-call episode ≈ $0.65 (writes + output) + $0.32 (reads over 1.61M cached tokens) ≈ **$0.97 cached**. Uncached: 1.61M × $4/MTok + $0.30 ≈ **$6.7**, about 7× more.
- The previous version of this section derived $0.81 and called it a "match" to Robocurve's $0.90. That agreement was coincidental (different token mix, image size and cache share) and is withdrawn.
- With skills sized so that a pick-place takes 2–4 calls, a pick-place costs ≈ $0.05–0.12.
- Fast mode ($8/$40) doubles the per-token price; it raises output tokens/s up to 2.5× but not time to first token. If Opus 5.5 keeps its ~1.8 s fixed cost, fast mode at the Tier 1 shape projects to ≈ 3.7 s mean per call vs 6.6 s observed [derived, gap-1; UNVERIFIED].

**Caveats on these figures.**
- Claude's 4.7+ tokenizer yields about 30% more text tokens than GPT tokenizers, so treat cross-provider re-costing as ±30% [note: engineering-challenges].
- Measured cache-hit rates: RoboICL 92.4%; GPT-as-Policy 97–98%. RoboDojo L3's sliding window gave Astra 9.0% but GPT-5.5 77.2%, so the low rate is provider-specific [note: engineering-challenges].

### 1.5 Safety (incidents and enforcement evidence)

**Incidents.**
- RoboDojo halted real Astra tests after "physically unreasonable or unsafe actions, including incidents that damaged hardware" [note: robodojo].
- RoboHarm: Astra completed 60/100 harmful instructions; Fable 5.1 completed 34/100 and refused 20/100 [note: robocurve-gpt6-astra].
- On a real Go2, Grok 4.1 Fast "started charging for the glass door" it saw a table reflected in [note: engineering-challenges].
- GPT-Policy "repeatedly observed collisions between the two arms" with no workspace or collision checks [note: gpt-policy-in-context].

**Prompt-stated constraints are not enforced.**
- SafeHarness: the model-only agent had 50.0% collision avoidance and 6.0% task success; with harness route verification, 87.5% and 71.9% [note: llm-planner-codegen-lineage].
- Text written in the scene hijacks GPT-4o 27.0% of the time and Gemini 2.5 Flash 29.4%. Text masking blocks 100% but breaks label-reading tasks [note: engineering-challenges]. Claude and GPT-6 were not tested; the Opus 5.5 baseline is unknown and must be measured (R7 in §5.1).

**Enforcement that worked.**
- MHS (Anthropic's Model Hardware Standard) puts limits in the driver: CMU induced 6 faults and MHS "correctly blocked all six before any device moved" [note: industry-competition].
- AGP's controller enforces ≤0.03 m/s, a workspace clamp and ≤0.25 m per move, with a human holding an e-stop [note: llm-planner-codegen-lineage].
- metal-arm-harness:
  - re-checks every Ruckig waypoint against a measured floor;
  - applies a 35° per-call excursion cap, a 76.2 mm slow zone and a recovery-only-upward rule [note: metal-arm-harness].

  It has no obstacle model, no held-object geometry, and a single-threaded server that cannot process a stop mid-move.

**Harness bugs are safety bugs.**
- piper-astra-jev's governor is *looser* than its prompt (`open_gripper` allowed mid-carry). Its "jammed" threshold is effectively dead code. API errors `raise_for_status()` with torque on [note: piper-astra-jev].
- quackd's 2026-09-23 hardware day had 19/26 runs never move the arm. The causes were five harness and prompt/gate faults, including a rest pose beyond calibrated travel and torque releasable only at the power switch [note: quackd].

### 1.6 Hardware and compute budgets

**Arm classes in the evidence base.**

| Class | Examples and prices | Notes |
|---|---|---|
| Serial-servo | SO-101, $249.90 Seeed Pro motor kit | STS3215 servos clamp goals to EEPROM limits [note: quackd] |
| CAN arms, ~$2–3k | AgileX PiPER $1,999; I2RT YAM $2,999; MakerMods Metal Arm $2,499 + $249 shipping | [note: industry-competition] |
| Mobile | Unitree Go2 / G1 | |

**Cheap-arm quirks the harness must absorb.**
- PiPER forgets joint-3 zero at power-up (about 8°), and RESUME drops the arm [note: piper-astra-jev].
- Metal's Damiao MIT mode silently clamps Kd to 5, and its gripper overheats under sustained stall [note: metal-arm-harness]. YAM uses the same Damiao MIT protocol and the same Kd ≤ 5 wire cap [gap-6].
- YAM shows 1–3 cm free-space sag, IK oscillation holds, and thermal trips under holds; its `LINEAR_4310` gripper is force-*limited* position control (`limit_gripper_force` 50 N), not force control; the motor drops to damping after a 400 ms command timeout [gap-6; note: robocurve-gpt6-astra].

**Compute.** A single RTX 4090-class GPU ran Grounding DINO plus SAM 3 in piper-astra-jev. A VLA tool needs a workstation GPU, as the CPU rows in §1.1 show.

### 1.7 Reproducibility constraints

- **Sampling cannot be pinned.** Opus 4.7+, Opus 5.5, Sonnet 5.5 and Fable 5.1 return 400 on non-default `temperature`/`top_p`/`top_k`. quackd's issue-#25 fix of pinning temperature 0 is therefore unavailable on Claude [note: quackd; note: engineering-challenges].
- **Temperature 0 is not determinism anyway.** Manda measured ±0.03–0.08 F1 swings between identical runs [note: manda-robotics].
- **Run-to-run variance is large even for a VLA.** Re-running π0.5 on identical seeds reproduced only 64% of successes, and moved per-task SR by ≥20 points on 28/120 tasks [note: manda-robotics].
- **Harness variance can rival model variance.** Scaffold-only swings of 7.3–48 pp have been reported. However, in CodeActionBench the within-Opus harness effect was 4.0 pp with p=0.74, i.e. not established [note: engineering-challenges; note: benchmarks-and-sim-datagen].

### 1.8 Claude API constraints that shape the harness

Source: bundled Claude API reference, consistent with the notes' live-doc checks [note: frontier-models-embodied-2026; note: engineering-challenges; note: quackd].

**Tool calling.**
- Opus 5.5, Fable 5.1 and Sonnet 5.5 return **400 on forced `tool_choice` (`any`/`tool`)**. Use `auto` + `strict: true` tools (with `additionalProperties:false`) + a prompt naming the tool.
- **Strict mode does not enforce numeric or length constraints.** The structured-outputs reference lists numerical constraints (`minimum`/`maximum`/`multipleOf`), string length constraints and complex array constraints as *not supported*; the Python/TypeScript SDK helpers strip them and validate client-side. So `xyz_m` and `center_px` arrays carry no length guarantee and no range guarantee: **server-side validation of array lengths and numeric ranges is a hard requirement** (bundled Claude API reference, tool-use concepts).
- Handle turns with zero tool calls; quackd re-prompts once, then fails [note: quackd]. In Tier 1 (3 tools, `tool_choice:auto`), none of 11,768 calls lacked a tool call [gap-1].
- Answer *every* `tool_use` id. Parallel calls are on by default; the removed metal-arm loop answered only the first and would have hit a 400 [note: metal-arm-harness].

**Thinking.**
- **Thinking cannot be disabled on Opus 5.5.** Default effort is `medium`; effort is the only dial. Opus 5.5 thinks *more* per turn than Opus 5 at the same effort.
- **Fast mode** ($8/$40) raises output tokens/s up to 2.5× but not time to first token. It is a waitlisted research preview on the Claude API only. Do not depend on it.
- **Preserved thinking.** Thinking blocks are bound to model and conversation. For accounts created on or after 2026-08-31 00:00 UTC the prefix check (system prompt, tools array and every earlier message byte-identical to when the block was produced) is enforced by default on Opus 5.5: a replayed block after an edit is a 400 ("The block is bound to a different conversation"). Older accounts opt in. **The transcript must be append-only.** Remedies:
  - mid-conversation `role:"system"` messages (no beta header; Opus 5.5 supported);
  - turn-scoped reminders with `clear_at: "next_user_message"` (beta `mid-conversation-system-clear-at-2026-08-21`; same models as mid-conversation system messages, so Opus 5.5 is included; leave earlier copies in place);
  - server-side context editing (`clear_tool_uses_20250919`, beta `context-management-2025-06-27`) or server-side compaction;
  - `thinking.block_binding.prefix_mismatch_behavior: "drop_block"` under beta `thinking-binding-controls-2026-08-01` (Claude API; per model on Bedrock/Vertex; not Foundry), which drops the failing block *and every later thinking block* for that request [note: quackd; bundled Claude API reference].
  - Client-side keep-tail compaction (summarize old turns, keep recent turns verbatim) **breaks** under the check: the retained turns' thinking blocks were made with the full history present.
- **Inspect Robots consequence.** `inspect_robots_agent/policy.py:210-258` `_evicted_view` replaces camera frames older than `image_horizon` with a "[N camera frame(s) elided]" stub on every call, i.e. it edits earlier user turns, and `_anthropic.py` replays thinking blocks verbatim; HEAD `095172f` has no `block_binding` handling. Robocurve's Tier 1 ran this configuration with zero errors (probably an older account), but **on a new account it will likely 400 at the first call after a frame is evicted** (UNVERIFIED until the smoke test in §6.2 M0).
- **Per-message effort** (beta `mid-conversation-output-config-2026-07-01`, an effort-only `role:"system"` message with `output_config`) changes effort without a cache reset. It is available on Opus 5.5, Opus 5, Fable 5.1 and Mythos 5.1, Claude API only at launch [bundled Claude API reference]. At a given level Opus 5.5 thinks more per turn than Opus 5; lower effort before adding "think less" instructions.
- **Compaction: two different features.** `compact-2026-01-12` is threshold-triggered server-side compaction (default trigger 150K tokens). `compact-2026-09-04` is the separate on-demand `compaction` parameter listed in Opus 5.5's feature set. With server-side compaction, the thinking-binding prefix starts at the most recent compaction block, so it is compatible with preserved thinking; choose the on-demand variant to compact at subgoal boundaries [bundled Claude API reference].

**Caching.**
- Minimum cacheable prefix is 512 tokens on Opus 5.5. Each breakpoint looks back 20 blocks; consecutive `tool_result` blocks count as one position.
- Adding or removing images anywhere invalidates every later block [note: engineering-challenges].

**Images.**
- High-resolution tier: 2,576 px / 4,784 tokens. A 1920×1080 frame costs 2,691 tokens.
- Up to 600 images per request; above 20 images, keep each side ≤2,000 px [note: engineering-challenges].

**Refusals.** `stop_reason: "refusal"` can occur on Opus 5.5 (categories include `cyber`, `bio`, `reasoning_extraction`). The harness must treat a refusal as "hold safe pose and escalate", never "continue".

**Never route Claude through the OpenAI-compatible shim.** It has no caching, returns no thinking, and ignores `reasoning_effort`. That confound distorts Robocurve's Fable-vs-Astra comparison [note: robocurve-gpt6-astra].

### 1.9 Derived requirements

| ID | Requirement | Main evidence |
|---|---|---|
| R1 | LLM is event-driven (0.03–0.2 Hz), never in a control loop; every motion is a bounded, leased, locally closed-loop primitive | §1.1; DrivingBench leases [note: github-implementations-sweep] |
| R2 | LLM outputs are object-, keypoint- or skill-level and re-bound by local perception at execution time; no free-form metric poses as the default path | Manda 0/4 vs 6/6; CoPa 37 vs 63 |
| R3 | Metric grounding is local: calibrated extrinsics, depth, segmentation. An LLM or specialist point is a seed, refined by mask/depth/CAD fit; the pointer is chosen by a measured millimetre eval, not by benchmark folklore | §1.2; [gap-2] |
| R4 | A servoed (then, where it earns its place, learned) contact layer closes the last 1–3 cm, with compliance | Puzzle 2/20; Precision 4%; [gap-3; gap-6] |
| R5 | Success is decided by deterministic sensors plus an independent verifier with *different evidence* (verification view, probes), never by the LLM's `done` alone; fail closed, routed by predicate class | §1.3; [gap-5] |
| R6 | Every tool result is a receipt: commanded vs measured, residual, grasp state, clamps, failure reason phrased as a fix | RoboProbe arrival check; metal-arm errors [note: robodojo; note: metal-arm-harness] |
| R7 | Safety envelope in driver and harness: approver chain, envelope on every waypoint, leases, heartbeat, e-stop independent of the LLM, scene-text injection hardening | §1.5 |
| R8 | Append-only transcript with explicit cache breakpoints; no history mutation | §1.4, §1.8 |
| R9 | Stale-observation guard; capture frames only when the arm is stationary (D435 is rolling shutter) | innate; [note: engineering-challenges] |
| R10 | Unload or park a loaded arm based on load, torque and motor temperature while it waits for the LLM (not a fixed 8 s timer); thermal duty-cycle rule and cool-downs | innate shoulder trips; YAM overheat; [gap-1; gap-6] |
| R11 | Provider-agnostic native wires (Anthropic Messages, OpenAI Responses, Gemini); full wire capture | Robocurve shim confound |
| R12 | Evaluation is paired, blinded and interleaved; ≥50 trials per arm for headline claims; FSM, "no-LLM" and scripted-servo baselines; wall-clock reported with physics live | §5; [gap-4; gap-6] |
| R13 | Every (observation, tool call, receipt) is logged in a trainable schema from day one, for head training and orchestrator distillation | Show-Harness, LeRobot v3.1 tool atoms [note: github-implementations-sweep] |
| R14 | Frame, TCP and gripper constants are single-sourced and unit-tested | Manda TCP conflict |
| R15 | Episode termination and the success claim are separate fields (`status` vs `claimed_success`), and every observation carries a remaining-calls counter | Opus 5.5 used `done` for 34 honest partial reports; models counted calls themselves [gap-4] |
| R16 | Tool arguments are validated server-side for array length and numeric range; strict mode guarantees only structure | Claude structured-outputs limits (§1.8) |

---

## 2. Design space: six candidate architectures

**Shared per-call numbers** (Opus 5.5, `medium`) [gap-1].
- Latency per API call (Tier 1, measured): p50 5.28 s, p90 11.72 s, p99 21.1 s; first call p50 16.1 s. Wall time per call including motion and per-trial overhead: p50 7.59 s.
- Cost: $0.0262 per call measured at Tier 1's shape; ≈ $0.022–0.034 projected for this design's append-only tiled shape (§1.4).
- Output: ≈ 424 tokens per call (reasoning included).
- For pessimistic latency, use Opus 5 at medium in Tier 1 (p50 8.77 s, p90 20.79 s) or Opus 5 at high on clapboardbench (median 10.8–14.4 s).

### A. Pure LLM tool-calling over Cartesian or joint primitives, with visual feedback

```
cams (2-3 RGB) + proprio ──► Opus (1 call / decision) ──► move_to{EEF targets} / gripper / done
       ▲                                                     │ approvers: clamp, delta-limit
       └──────── next obs after chunk (open loop 0.3-10 s) ◄──┘ IK (DLS) → joint targets @10-25 Hz
```

**Latency and cost.**
- 15–40 decisions per episode (measured calls per trial: Robocurve Astra 14.8, Fable 5.1 17.6; Tier 1 Opus 5.5 34.2, Astra 28.6).
- Per episode: 15–40 × 5.3 s (p50) to 11.7 s (p90) = **1.3–8 min of LLM time**; cost 15–40 × $0.026 = **$0.4–1.0** [derived].
- Robocurve's robot moves for only about 11 s per bowl trial and 28 s per Tier 1 trial [note: robocurve-gpt6-astra; gap-1].

**Capability envelope.** Coarse open-vocabulary pick-and-place works:

| Setting | Result |
|---|---|
| Robocurve block-into-bowl | Astra 19/20; Fable 5.1 8/20 |
| URAI, AgileX block-into-bowl | 3/3 (with tools) |
| RoboDojo L3 | Astra 22.48% SR (2,100 trials) |

It fails at sub-cm insertion, contact, dynamics and large rotations:

| Setting | Result |
|---|---|
| Puzzle insertion | 2/20 |
| RoboDojo Precision | 4% |
| Robocurve Tier 1 completions | Opus 5.5 1/120, Astra 5/120 (p = 0.21); mean progress 36.0% vs 36.7% |
| LIBERO direct control | 0–5.5% |
| LIBERO-Agent, Opus 5 | 0% on hard short-horizon tasks |

Sources: [note: robocurve-gpt6-astra; note: robodojo; note: benchmarks-and-sim-datagen; gap-4].

What actually blocks architecture A on YAM (Tier 1, hand-coded terminal failure of 119 Opus 5.5 / 115 Astra non-completed trials) [gap-4]: placement/release misses 35% / 26%; bimanual alignment 15% / 13%; no grasp 13% / 37%; reach/kinematic limits 12% / 7% (Stand Up Bottles pins wrist pitch and roll at 0, and the safe lid sits beyond reach); budget exhausted 9% / 3%; about half of all grasp closes come up empty (48% / 57%); 100/120 Opus 5.5 final reports cite the 40-call budget.

**Evidence for.**
- Simplest to build. Astra makes it work on easy tasks.
- Arrival-error feedback and planner refusals enable self-repair: under perturbation, negated xyz recovered 4/8 and mirrored images 6/8 [note: robodojo]. **Anecdotal:** 1 episode per layout, and the layouts were "selected as those the unperturbed model solves".

**Evidence against.**
- MakerMods removed a raw-joint Opus 5 Messages loop within about 8 hours [note: metal-arm-harness].
- Interface friction:
  - rounded quaternions rejected at a norm tolerance of 1e-4;
  - a runaway of 7,750 rejected calls in one episode;
  - 5 cm-per-decision caps produce tiny steps [note: gpt-as-policy].
- Measured-state seeding of unnamed dimensions lets pose sag compound ("Unspecified tilt drifted substantially") [note: robocurve-gpt6-astra].

**Code to borrow.**
- `inspect-robots/plugins/inspect-robots-agent/src/inspect_robots_agent/_tools.py`: action-space-derived tool schemas, out-of-bounds errors returned as strings, 10 s move cap.
- `.../policy.py`: budget, forced `give_up`, `hindsight` → `prior_learnings`.
- `.../_anthropic.py`: native Messages client with thinking replay and cache anchors.
- `inspect-robots/src/inspect_robots/approver.py`: `ClampApprover`, `DeltaLimitApprover`, `ChainApprover`.
- `XPolicyLab/policy/GPT_6_Astra_Direct_EEF/pose.py`: gimbal-safe angles relative to a straight-down reference quaternion (0.5,−0.5,0.5,0.5).
- `XPolicyLab/policy/GPT_6_Astra_Direct_EEF/policy.py`: arrival check, unreachable-pose refusal costs a call but no steps; `_held_action_data` re-sends the commanded jaw value.
- `GPT-as-Policy/hybrid_rollout/robodojo/robodojo_server/{kinematics.py,validation.py}` (MIT).

**Verdict.** Build it only as a **baseline arm** for evaluation (it is the Robocurve/RoboDojo protocol), not as the product.

### B. LLM + open-vocabulary perception + IK/motion planning (skill selection with grounded arguments)

```
RGB-D (calibrated) ─► detectors (Grounding DINO / SAM 3) ─► masks + depth ─► object table (ids, 3D, extents)
                                         │                                       │ text/JSON
                                         ▼                                       ▼
                         grasp search / pose fit / reach checks  ◄──── Opus: pick(obj_id, mode) / place(obj_id, region)
                                         │ refusals as "problem" strings
                                         ▼
                         IK + time-scaled trajectory (Ruckig) + envelope ─► arm
```

**Latency and cost.**
- One call per skill. A pick-and-place needs about 6–10 calls; piper-astra-jev logged 9 decisions [note: piper-astra-jev].
- Per pick-place: 6–10 × 5.3 s (p50) to 11.7 s (p90) ≈ **30–120 s of LLM time**, ≈ **$0.16–0.26** [derived from gap-1].
- Perception adds about 0.1–0.4 s per call (SAM 3 ~200 ms per phrase).
- piper-astra-jev's Jev variant ran the whole decision loop in 16.6 s at 10% arm speed. A frontier VLM deciding the same skills cost seconds per call [note: piper-astra-jev].

**Capability envelope.**
- Pigey (Opus 4.7 + Gemini-ER detections + π0.5 + TAMP): 97.3% on 30 real DROID tasks × 5, vs π0.5 16.7% and TiPToP 48.7% [note: llm-planner-codegen-lineage].
- VoLo's "No-VLA" condition (Opus 4.6 + SAM3/Molmo2/GraspGen primitives) scored 45.2% on a real Franka, vs 42.9% *with* a VLA (42 rollouts each; CIs overlap) [note: vla-and-action-heads].
- Weak where analytic grasps fail: deformables, insertion, specular parts. On chrome, depth is "useless" [note: piper-astra-jev].

**Evidence for.**
- The 2026 field converged here [note: llm-planner-codegen-lineage].
- Labels must come from detector output; Pigey's grounding fixes Hi Robot's GPT-4o failure mode [note: llm-planner-codegen-lineage].

**Evidence against.**
- Per-object engineering: piper's SlideOver is about 400 lines for one insertion, and its STLs are hand-measured with calipers [note: piper-astra-jev].
- The decider can be redundant. A 20-line FSM reproduces piper's logs step for step, so the LLM earns its keep only under perturbation [note: piper-astra-jev].
- Single-shot code-as-policy over low-level APIs is still weak: CaP-Bench S4, Opus 4.5 23.8% vs humans 88.5% [note: llm-planner-codegen-lineage].

**Code to borrow.**
- `piper-astra-jev/piper_llm/skills.py`: `grasp_search` (mask + depth height map, finger clearance; score clearance capped at 20 mm) and the `grasp_from_mask` fallback.
- `piper_llm/geometry.py`: `locate` (STL silhouette fit without depth), `grasp` (centre-of-mass lever ranking), `locate_held`.
- `piper_llm/task.py`: occlusion memory, `_check_reach`, `_check_path`.
- `piper_llm/record.py`: single-clock recorder.
- `piper_llm/kinematics.py`: DLS position + null-space orientation.
- `piper-astra-jev/calibrate.py`: hand-eye fit with a joint-3 offset search. **Fix:** Esc accepts any residual.
- `Pigey/real/agent.ts` and `real/agent-system.md`: tool set and sensor overrides. **Do not copy** its fail-open verifier.
- `cap-x/capx/integrations/franka/control_reduced.py`: `segment_sam3_text_prompt`, `plan_grasp`, `solve_ik` (MIT).

### C. LLM emits 2D/3D traces, keypoints or waypoints consumed by a light learned head

```
Opus ── chooses object/part + subgoal ──► pointer (Opus+zoom / ER-2 / local 8B; by measured eval) ─► seed point → mask/depth refine
                                                                 │ depth+K+extrinsics
                                                                 ▼
                          3D target / 2D trace overlay + phase token ─► light head (20-100M, 30-50 Hz, chunked)
                                                                 │ receipts + progress scalar
                                                                 ▼
                                                     envelope ─► arm      (LLM re-plans every 2-8 s segment)
```

**Latency and cost.**
- 1–3 LLM calls per subgoal plus 0–2 pointer calls. ER 2 pointing latency is **unmeasured**; its only real-robot latency is as a joint controller at high effort (p50 5.8 s, p90 9.6 s, n = 91), about the same as an Opus 5.5 call, so a serial ER 2 query roughly doubles a decision's latency and k = 3 averaging triples the ER 2 share [gap-1]. ER 2 costs $1/$5 per MTok through 2026.
- The head runs locally at 30–50 Hz. Per episode: similar to B (≈$0.2–0.5), but contact phases run closed-loop without the LLM [derived].

**Capability envelope.**
- A 2D EEF-path interface works with a light 3D policy: HAMSTER +20 pp absolute over OpenVLA [note: vla-and-action-heads]. But in 3D HAMSTER (2606.31329), 2D guidance *hurt* a 3D policy (49.5 vs 53.8 with no guidance), and Sonnet 4.6 put both 3D trajectory endpoints within 5 cm only 0.7% of the time [gap-3].
- Policy-agnostic mask/trace overlays help a lot: PEEK reports 2–41× gains (path + mask on cube stacking: 33.5 → 73.6) [note: vla-and-action-heads; gap-3].
- GAE (2510.03896): a frozen geometry expert fed VLM 3D waypoints beat fine-tuned π0.5 on RoboTwin long-horizon (0.60 vs 0.50) and kept 0.33 success under 10 cm depth noise vs 0.11 for plain IK, *if* trained with waypoint noise at scale 0.10 (0.43 → 0.60) [gap-3].
- Keypoint constraints plus a solver beat direct numbers; ReKep's largest error source is the point tracker, not the optimizer [note: llm-planner-codegen-lineage].
- Embodied-R1.5 (pointing + scripted motion, zero-shot) scored 65.0% vs 72.9% for π0.5 fine-tuned on 400 demos per task on RoboTwin 2.0 [gap-3].

**Evidence for.**
- Manda's 0/4 → 6/6 (Fisher p ≈ 0.005, but not a clean A/B) [note: manda-robotics].
- Show-Harness: an interaction-point marker took handle-aware grasping from 40% to 85% [note: engineering-challenges].

**Evidence against.**
- Zero-shot closed-model 2D paths are poor; HAMSTER's GPT-4o paths ranked worst [note: vla-and-action-heads].
- Pointing quality is unsettled, not settled against Claude. Opus 5 at low effort trails on RefSpatial (−21.7) and PixMo-Points (−18.5), is within about 3 points on PointBench and VABench-Point, and leads on Part-Affordance (78.1 vs Astra 55.0). There is no Opus 5.5 or ER 2 pointing number; GroundingPI-4B (73.68 vs Astra 71.54 on 34 grounding benchmarks, author-reported) has **no released weights** (its HF repo holds a 28-byte README). Public local alternatives: PhysBrain 1.5-8B, MolmoPoint-8B (Apache-2.0), Embodied-R1.5-8B (Apache-2.0) [gap-2].
- **Executors learn to ignore the conditioning.** In Fast Plans, Faithful Actions (2609.30833), waypoints given as suffix tokens had 0.6% sensitivity and erasing them changed success by 0.0 pp; naive injection into every layer dropped success to 15% (label leak). Their fix — endpoint noise σ_c = 0.7, a coarse-stage gate, 15% null dropout — reached 96.2% on LIBERO-Long, and erasure then cost 7.4 pp [gap-3].
- In a CPU toy study (2 mm peg-in-hole, 400 episodes per cell), heads trained on exact hindsight keypoints became open-loop integrators: 100% with perfect points, 59% at 2 mm pointer noise, 20% at 5 mm. Training with test-like noise and staleness fixed most of it (90% at 2 mm, 72% at 10 mm); a tracked selection mask was invariant to keypoint noise (89–94%). **Both analytic baselines beat every learned head**: point association + wrist servo 100% up to 10 mm noise; analytic staging + a frozen primitive 88–98% [gap-3; toy, not a robot result].

**Code to borrow.**
- `manda-RoboLab-Verified/policies/vlm_pinpoint/connector.py`: observation contract where missing channels are `None`, lazy backend import, fingertip offset (verify the constant!).
- `GPT-Policy-cheng/src/gpt_policy/vision/perception.py`: `locate_point`, which triangulates the same wrist camera at two steps. **Ideas only; the repo has no licence** [note: gpt-policy-in-context].
- `so101-block-sorting/scripts/calibrate_camera.py`: gripper-marker PnP self-calibration. Its "16/16 inliers" is tautological [note: so101-painting].

### D. LLM as System 2 orchestrating a small VLA System 1 through language subtasks

```
Opus ── subtask text / instruction rewrite ──► VLA tool run_policy(instr, max_s, stop_when) ── 10-50 Hz ──► arm
  ▲            (or: VLA proposes chunk → FK preview → Opus accepts prefix / bounded edit)       │
  └────────────── receipt + fresh frames + independent success detector (ER-2 / learned) ◄─────┘
```

**Latency and cost.**
- One LLM call per 4–8 s VLA segment. Hi-VLA recommends "a moderate horizon, e.g., 4–8 seconds" [note: vla-and-action-heads].
- VLA inference costs 38–130 ms per chunk on a GPU.
- Async variant: the LLM deliberates while the segment runs. Cost per segment ≈ $0.026; a 10-segment task ≈ $0.26 [derived].
- The synchronous "review every chunk" variant is expensive: GPT-as-Policy Hybrid used 12.5M tokens over 75.5 decisions per episode, at 27–106 s per decision [note: gpt-as-policy].

**Capability envelope.**

| Setting | Result |
|---|---|
| Harness VLA (frozen π0.5 as `vla_act(prompt, max_chunks, stop)` + 6 analytic primitives + memory), LIBERO-Pro | 50.0% raw → 82.4% with Claude Code/Opus 4.7; 92.63% with Astra at low effort (8 cells × 100) |
| Harness VLA, RoboCasa365 | Opus 4.7 48.8% vs GPT-5.5 57.2% |
| VoLo (Opus 4.6 orchestrating π0.5), sim | 12.57% → 41.80% |

Sources: [note: llm-planner-codegen-lineage; note: github-implementations-sweep; note: vla-and-action-heads].

**Evidence against: the results conflict.**
- Anthropic: every model *supervising* MolmoAct did worse than MolmoAct alone in-distribution. Mythos over-overrode the VLA [note: frontier-models-as-direct-policies].
- GPT-as-Policy: Hybrid beat Direct on RoboDojo (24/50 vs 13/50, McNemar p ≈ 0.027). Caveats that matter [note: gpt-as-policy]:
  - the subset was selected where π0.5 is weak, and π0.5 authored 85.6% of executed steps;
  - the π0.5 comparator (15.67%) is the public leaderboard reweighted to the subset, not a same-seed rerun;
  - the largest hybrid gains came with almost no LLM authorship: fold_clothes 100% vs 21% with 1.7% LLM-authored steps; put_bottles 100% vs 69% with 0.2%. Executing 1–15-step prefixes of a 50-step chunk and re-inferring may itself explain them as receding-horizon replanning. **"π0.5 with a fixed 15-step prefix" is a required control** before reading hybrid gains as LLM-supervision gains;
  - On RoboLab, Direct beat Hybrid (49/50 vs 46/50), but those arms used authorized retries and were not paired.
  - On RoboCasa365 the gap was not significant (p = 0.52) [note: frontier-models-as-direct-policies].
- Steerability is the crux. Narrowly fine-tuned VLAs ignore rephrasings (Hi-VLA). The low-level VLA's own training data dominates [note: vla-and-action-heads].
- Practical blockers:
  - quackd's fit check refused 55/68 Hub ACT checkpoints on one lab arm. That is an artefact of one arm's calibration, but it shows how fragile frame matching is [note: quackd].
  - GR00T N1.7 weights are under the NVIDIA Open Model License, and its backbone is gated [note: industry-competition].

**Code to borrow.**
- `RLinf_RPent/rpent/planner/claude_code.py` (Claude Agent SDK planner) and `RLinf_RPent/robots/libero/tools.py` (`pi0_pick`, `move_to`, `segment`, `back_project`) (Apache-2.0).
- `quackd/adapters/lerobot/src/quackd_lerobot/policy/loop.py`: deadline-paced chunk queue, `refill_at = max(2·latency, chunk/2)`, `starved_each_chunk`, per-tick speed cap, starvation/stall guards (Apache-2.0).
- `google-deepmind_gemini-robotics-sdk/safari_sdk/agent/framework/tools/run_instruction_until_done.py`: `NON_BLOCKING`, with an ER success detector polled every 0.2 s.
- `GPT-as-Policy/hybrid_rollout/robodojo/skill/gate_prompt.md` and `robodojo_server/gate_assessment.py`: outcome/intent gate where "uncertainty alone is not a takeover reason".
- `strands-labs_robots/strands_robots/_motion_grants.py`: human motion grants.

### E. In-context demonstration imitation (demo as the task specification)

```
1 teleop episode (HDF5/MCAP) ─► keyframes (grip events, survey) + compressed action rows ─► cached prefix
live obs ─► Opus: note_observations → record_phases(advance_when) → act(bounded step) ... ─► guarded executor
```

**Latency and cost.**
- GPT-Policy: 24–96 decisions per task at about 12–20 s each, i.e. 5–25 min per task. One towel run used 4.96M tokens with video context and 12.05M without [note: gpt-policy-in-context].
- innate #817: 45 steps, `act` calls 3.2–8.5 s (median ≈ 4.3 s), plus 19.6 s of planning calls [note: innate-os-pr817].
- A 48-image demo at 1280×720 costs ⌈1280/28⌉·⌈720/28⌉ = 46·26 = 1,196 tokens per image, ≈ 57k tokens. Cached on Opus 5.5 that is ≈ $0.011 per call to re-read [derived].

**Capability envelope.**

| Study | n | Result |
|---|---|---|
| GPT-Policy video + action | n = 3 per condition | Bottle uncapping 0/3 → 3/3; plug 0/3 → 2/3 |
| RoboICL, real FR3 | 3 tasks × 5 | Mean progress 14.45 → 63.33 → 78.89 at 0/1/3 shots |
| RoboDawn | — | One-shot lifts 53.2% → 73.6% |

Sources: [note: gpt-policy-in-context; note: frontier-models-as-direct-policies].

**Evidence against.**
- Naive demonstrations *hurt* on RoboDojo: zero-shot 22.9% → image demo 17.9% → text demo 12.9% over 340 matched episodes [note: robodojo].
- GPT-Policy dropped a null result (glue cap) from its main table [note: gpt-policy-in-context].
- Innate's own successor PR #843 abandoned demo-conditioning [note: innate-os-pr817].
- Robot foundation models now do this natively at 100 Hz (both **vendor-only** claims):
  - GEN-1.5 reports 59% (±10%) one-shot over 10 tasks from a 3–12 s physical prompt, with no trial counts;
  - Skild S1 reports 66% vs 9% for a language-prompted VLA, measured on a filtered subset of its own pretraining pool [note: industry-competition].

**Code to borrow (ideas).**
- innate `ros2_ws/src/brain/brain_client/innate/imitation_policy.py`: survey → phase map → per-phase demo windows; demo-derived aperture and envelope priors.
- `innate/imitation_actions.py`: caps defined once and rendered into the prompt.
- `Mosi-AI_RoboICL/roboicl/policy/astra_policy.py`: a single `act` tool emitting an H×14 delta chunk, receipt grammar, anchored memory (MIT).
- GPT-Policy's `input/demonstration.py`: keyframe/action-row compiler (ideas only; no licence).

**Verdict.** E is a *feature* (task specification by demo), not the architecture.

### F. LLM writes or updates skill code and generates sim data to train the head (offline engineer and teacher)

```
task + robot card ─► coding agent (Opus/Astra) in sim: writes solve()/skills, verifiers, assets
                     ─► hidden grader + replay gate ─► diversified demos (scene/strategy/phase/noise/visual)
                     ─► LeRobot dataset ─► train head / VLA ─► deploy as a tool under architecture B/C/D
```

**Latency and cost (offline).**
- EmbodiedSWE: median 39–119 min to solve; $8–158 per solved run. Agents: Astra/Codex 82% success, Fable 5.1 61%, Opus 5 50% [note: embodiedswe].
- DexAgent: 2.1 h per video on average (4.3 h cold, 13.1 min with asset reuse) [note: dexagent].
- At runtime the LLM is absent or only orchestrates.

**Capability envelope.**

| Study | Result |
|---|---|
| EmbodiedSWE-Gen, SmolVLA | 18% → 69% six-task success as demos go 10 → 400 (relaxed criteria) |
| EmbodiedSWE-Gen, agent-aided vs script-only data | 0.233 vs 0.066 on held-out configurations |
| EmbodiedSWE, real π0.5 lamp | 2/10 vs 0/10 baseline |
| DexAgent, 1 demo → 500 episodes → π0.5 | 63.6% (70/110) vs 18.2% best baseline |
| Guava, 2,268 GPT-5.4 sim trajectories → Qwen3.5-4B | 87.1% vs teacher 90.4% |
| URAI (tools authored offline by a Fable 5.1 programming agent and frozen; Opus 5.5 orchestrates at xhigh) | 60% (15/25), vs 24% for a pre-written program |

Sources: [note: embodiedswe; note: dexagent; note: frontier-models-as-direct-policies].

**Evidence against.**
- Sim shortcuts inflate solvability: weld grasps, auto-weld on seat, constraint-based unscrewing [note: embodiedswe].
- Reward hacking: GPT-5.6 Sol/Terra hacked 39–43% of runs [note: embodiedswe].
- Pipelines compiled in sim then deployed blind fail on hardware: GaP 0/25 and ASPIRE 3/25 vs runtime AGP 23/25 [note: llm-planner-codegen-lineage].
- Real-robot evidence is thin, and EmbodiedSWE selected its checkpoint on real-robot performance [note: embodiedswe].

**Code to borrow.**
- `EmbodiedSWE/data_engine/engine/generation.py` and `data_engine/agent/prompts/noise.md`: clean-label noise channel. The solve calls `env.step(action, noise=…)`; the label is the clean action, while `action + NOISE_SCALE·noise` is executed.
- `EmbodiedSWE/data_engine/scripts/orchestrate.py`: open-loop replay gate, privileged-write regex, code fingerprinting.
- `EmbodiedSWE/eval/tools/checkpoint_tree.py`: stage plan, `tried_from`.
- `EmbodiedSWE/eval/tools/parameter_search.py`: CMA-ES over 512 parallel envs with ADOPT/REJECT verdicts.
- `EmbodiedSWE/eval/tools/assessment.py`: mandatory `assess()`.
- `EmbodiedSWE/eval/scripts/verify_solution.py`: `_NoShortcuts` blocks `reset`/`set_states`.
- `EmbodiedSWE/eval/prompts/_contract.md`: "A filepath is not a visual inspection".
- `manda-real2sim-frontier-assets/eval/neutral_corkscrew.py`: automatic physics check of agent-built assets (Opus 5.5's can tipped over; its chair slid 2.4× further than Astra's [note: manda-robotics]).
- `NVlabs_ENPIRE/README.md`: real-robot reset → train → evaluate loops.
- `huggingface_lerobot/docs/source/annotation_pipeline.mdx`: `lerobot-annotate`.

### Comparison

| | A direct EEF | B grounded skills | C trace/keypoint → head | D LLM over VLA | E demo ICL | F offline engineer |
|---|---|---|---|---|---|---|
| LLM decision rate | 0.07–0.2 Hz, every motion | per skill | per subgoal | per 4–8 s segment | per bounded step | offline |
| $ per episode (Opus 5.5, cached) [derived] | 0.4–1.0 | 0.15–0.3 | 0.2–0.5 | ~0.26 async | 1–3+ | ~0 runtime; $8–158 per task offline |
| Precision ceiling | cm (2/20 puzzle) | mm with good geometry, per object | mm if head is trained | VLA-dependent | cm | inherits trained head |
| Generalization to new objects/tasks | high semantic, low physical | high if detectable | medium-high | depends on VLA steerability | per demo | per task family |
| Robot data needed | none | none (calibration) | 50–500 demos per skill | VLA + embodiment fine-tune | 1 demo | sim + some real |
| Best evidence | Robocurve 19/20 bowl | Pigey 97.3% (real, 150 trials); AGP scripted YAM insertion 8/10 | Manda 6/6 (n small); HAMSTER; GAE (sim) | Harness VLA 82.4% (sim) | RoboICL 78.9 progress (n=15) | DexAgent 63.6% real |
| Main risk | latency, precision | per-object engineering | shortcut learning, pointer accuracy, losing to analytic servo [gap-3] | supervision hurting VLA | slow, superseded by robot-FM ICL | sim-to-real |

---

## 3. Recommended architecture: an "orchestrated grounded skills" harness

In one line: **B as the backbone, servoed-then-learned contact skills (C) for the last centimetre, D's VLA as an optional (YAM-fine-tuned) tool, E as a task-specification feature, and F as the offline data engine.** The LLM (Opus 5.5 by default) never emits a motor command. It chooses objects, skills, phases and relations, verifies outcomes, and recovers.

### 3.1 Layer diagram and rates

```
                         ┌────────────────────────── L4 ORCHESTRATOR (0.03-0.2 Hz) ─────────────────────────┐
 task text / demo ──────►│ Opus 5.5 (effort medium; per-message high on replan), append-only transcript     │
 robot card, skill docs  │ strict tools (§3.6) ─ one motion tool per turn ─ receipts in tool_result         │
                         └───────▲──────────────────────────┬───────────────────────────────▲────────────────┘
                                 │ observations (tiled,      │ typed calls                    │ verdicts
                                 │ overlays, JSON state)     ▼                                │
 ┌─ L3 GROUNDING (1-10 Hz) ──────┴──────┐   ┌─ L2 SKILL/EXECUTION (10-50 Hz) ───────────┐   ┌┴─ VERIFIER (event + 2-5 Hz) ──┐
 │ calibrated RGB-D, SAM 3 / G-DINO     │   │ skill runtime: lease, stop predicate,     │   │ sensor predicates (grasp,     │
 │ object table (ids, 3D, extents),     │◄─►│ stale-obs guard; geometric skills (grasp  │──►│ at-rest, contact), lift test, │
 │ tracker, zoom/point→3D (pointer by   │   │ search, IK+settle, Ruckig) + CONTACT      │   │ verification view, contact    │
 │ measured eval), scene-text masking   │   │ SKILLS (servo+compliance first; 20-100M   │   │ sheet + final frame; vision   │
 │                                      │   │ heads where they win) + optional VLA tool │   │ advisory on contact; fail-cl. │
 └──────────────────────────────────────┘   └───────────────────┬───────────────────────┘   └───────────────────────────────┘
                                                                ▼
 ┌─ L1 SAFETY / MOTION (100-1000 Hz, on robot PC) ─────────────────────────────────────────────────────────────────────────┐
 │ approver chain: workspace box, floor+margin (measured), keep-out boxes, held-object geometry, per-call excursion cap,   │
 │ speed/accel/jerk, slow zone; heartbeat/deadman; thermal/overload; firmware limits; hardware e-stop (independent)       │
 └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

This is quackd's "three rates, three owners" made concrete [note: quackd]: deliberation at about 0.2–1 Hz, steering at the policy rate, reflexes at the body's rate. It also matches GRID's "two loops", with the inner loop and safety on the edge [note: general-robotics-auto-engineering].

### 3.2 L4: LLM orchestration

- **Model.** `claude-opus-5-5`, `output_config.effort: "medium"` (the default, but set it explicitly). Escalate to `high` with a per-message effort system message only on replanning after a failure.
  - Evidence on effort is mixed. Robocurve's Opus 5 milestone score rose 46 → 64 → 76 from low to high effort, but full stacks went only 0/5 → 1/5 → 1/5 (n = 5, raw joints) [note: robocurve-gpt6-astra].
  - Anthropic found extra reasoning "made no major difference for any of the Claude-family models" on manipulation [note: vla-and-action-heads]. As a verifier, Opus 5.5 at high effort was no better than at low (BA 0.724 vs 0.754) [gap-5].
  - Latency scales with output tokens (≈ 87 tok/s), so effort is also the latency dial. Opus 5.5 at low/high effort is unmeasured in a robot loop [gap-1]. Sweep effort on your own tasks.
- **Turn discipline.**
  - `tool_choice: {"type":"auto","disable_parallel_tool_use": false}`.
  - Read-only tools (`observe`, `zoom`, `locate`, `point_to_3d`, `recall`) may be called in parallel and run concurrently.
  - At most one world-changing tool per turn. Extra motion calls get `is_error:true, "only one motion per turn; re-issue after the receipt"`. Every `tool_use` id gets a `tool_result`.
  - Never silently ignore a call. Robocurve dropped Fable 5's batched `done` calls, which turned real stage-4 completions into forced give-ups [note: robocurve-gpt6-astra].
- **Stop reasons.** On `max_tokens` or empty text, re-ask once with the receipt. On `refusal`, hold safe and escalate to a human. On API error, retry with backoff; if the deadline passes, run the deterministic `hold`/`retreat` fallback. Never leave the arm mid-air under torque while an exception unwinds [note: piper-astra-jev].
- **Planning artifact.** The first turn must produce a structured plan in the style of DexAgent's `scene_semantics.json` [note: dexagent]:
  - objects with type, parts and functional properties;
  - subgoals with `acting_arm`, `object_refs` and an *observable* `goal`/`advance_when` predicate;
  - negative constraints, e.g. "mug remains on the tabletop; neither hand grasps it".

  This plan is checkable, and it is what the verifier tests against. innate's survey → phase map → act ladder is the same idea [note: innate-os-pr817].
- **Fast path for routine transitions.** Most phase transitions in a pick-place are determined by booleans; piper's 20-line FSM test shows this [note: piper-astra-jev].
  - Default to code: an FSM inside each skill.
  - Optionally add a typed decision model (self-hosted Kev-4B at 18.1 ms, or Laya at about 40 ms). Use it only for enumerated choices, with an `escalate` option, per-class confidence floors and shadow-mode promotion (quackd's stepper) [note: quackd; note: frontier-models-embodied-2026].
  - Jev's docs say it "struggles with tasks that require numeric precision", so never let a decision model author numbers [note: frontier-models-as-direct-policies].
  - RoboICL's Jev gate cut Astra calls by 33–48%, but only on two development tasks, and success fell on one of them (3/5 → 2/5) [note: frontier-models-as-direct-policies].

### 3.3 L3: perception and grounding services

**Sensors.**
- One overhead RGB-D camera, e.g. D435/D435i; rolling shutter, depth min-Z about 28 cm.
- One wrist RGB-D per arm: D405, 7–50 cm ideal range. This is the AGP and Robocurve-Opus-5 configuration [note: llm-planner-codegen-lineage; note: robocurve-gpt6-astra].
- Hand-eye and table-plane calibration as a **preflight** that returns numbers to the LLM:
  - FK-vs-measured TCP error;
  - depth plane-fit RMS;
  - commanded vs measured command rate. GRID found a 500 Hz-instead-of-30 Hz pacing bug this way, and depth plane error went from 28 mm to 1.2 mm after model selection [note: general-robotics-auto-engineering].
- A table-touch floor in the style of metal-arm's `calibration.py`: two FK samples must agree within 3 mm [note: metal-arm-harness].

**Object table.**
- Grounding DINO (Apache-2.0) boxes plus SAM 3 masks. SAM 3 has its own licence (`license:other`); check it before commercial use [note: piper-astra-jev].
- Depth deprojection gives per-object centroid, extent, top height, `depth_valid_frac` and a stable id.
- Keep occlusion memory: hold the grasp plan while the gripper occludes the target; reject slivers that are too high or too big [note: piper-astra-jev].
- Labels the LLM uses **must** come from this table (Pigey's rule) [note: llm-planner-codegen-lineage].

**Pointing** [gap-2].
- If the LLM needs a point not covered by the object table (a part, a groove, a handle), it calls `zoom` and then `point_to_3d` with absolute pixel coordinates in the image it was shown (unresized frame, `transformations.oversized_image:"error"`).
- **Default: Opus 5.5 points itself**, with a mandatory `zoom` tool and geometric refinement. The only part-level evidence favours Claude (Opus 5 Part-Affordance 78.1 vs Astra 55.0, Gemini 3.6 Flash 64.7; ER 1.5 scored 27.0 in another kit), so `part_hint` is *not* routed away from Claude by default.
- **A/B arms, decided by the 300-target millimetre eval in [gap-2] §5** (hit rate within 5/10/20 mm after deprojection; ≈$35–60 for Opus 5.5; ≈155–194 paired targets detect a 10-point difference): Gemini ER 2 (`gemini-robotics-er-2-preview`, `[y,x]` 0–1000, `thinking_level` low/medium, k = 1 vs k = 3 averaged; needs a restricted API key); local PhysBrain 1.5-8B, MolmoPoint-8B or Embodied-R1.5-8B. GroundingPI-4B waits for released weights.
- Run any external pointer concurrently with the orchestrator call, or only on explicit `point` requests, because it otherwise adds a full call of latency [gap-1].
- **Never pass a VLM point directly as a ≤5 mm contact target.** Seed SAM 3 with it, take the mask, run depth/plane/CAD fit, then close the last millimetres with relative servoing or contact.

**Specular or known parts.** Offer piper's STL silhouette fit as an optional skill. It requires a mesh per object, which is a real scalability cost [note: piper-astra-jev].

**Prompt-injection hardening.**
- Run OCR on frames and mask scene text by default. Expose `read_text(region)` as an explicit tool so label-reading tasks still work.
- Treat text found in images as untrusted data in the system prompt [note: engineering-challenges].

### 3.4 L2/L1: geometry, motion and skills

**Motion primitive contract** (copy GRID's `moveToPose` semantics; GRID's "arrive within 1 mm / 5 mrad" was measured on a Flexiv Rizon and does not transfer to a YAM) [note: general-robotics-auto-engineering; gap-6]:
- `blocking`, `moving_time`, `avoid_force` + `force_threshold`;
- arrive within a **per-arm, per-region tolerance measured in Phase 1** (bench protocol in [gap-6] §6) and at rest. Expected YAM values: ≈1.5–3 mm with encoder-zero calibration, `ee_mass` payload compensation and an AGP-style integral settle (`cartesian_settle_gain_s_inv: 4.0`, `max_bias_deg: 2.0`); 1–3 cm without them. A 1 mm contract would return `StoppedShort` on most YAM moves;
- typed errors: `ForceThresholdExceeded`, `RobotFault`, `StoppedShort` (with residual), `SettleMiss`.
- Sub-millimetre alignment is never a motion-contract promise: it is closed on the wrist camera by a contact skill.

**YAM-specific requirements** [gap-6].
- Calibrate encoder zeros (11 board touches with AGP's `calib/fit_joint_offsets.py`) and fit `gravity_comp_factor`/`ee_mass` before any precision work.
- Adopt metal-arm's commanded-base sag compensation and integral trim (≤0.75°/s, |lead| ≤ 2°) [note: metal-arm-harness].
- Contact detection: baseline-subtracted J2/J3 effort OR Cartesian lag (commanded − FK(measured)) above a threshold for 3 ticks; expect ≈1 N sensitivity.
- Impedance: per-command `kp`/`kd` via `command_joint_state`; Kd is capped at 5 on the wire, so for a stiffer Cartesian behaviour use host-side impedance (MIT Kp = Kd = 0, τ = Jᵀ(KΔx − Dẋ) + g at 250 Hz) and measure the maximum stable K.

**Trajectories.**
- Densify linear + SLERP paths, run seeded IK per sample with residual gates (2 mm / 1°), then Ruckig time-scaling that never alters geometry (GPT-Policy design) [note: gpt-policy-in-context].
- Re-check every sample against the envelope (metal-arm `safety.py:plan_trajectory`) [note: metal-arm-harness].

**Hold the commanded value, not the measured one,** for every dimension the caller did not name. Measured-state seeding re-plans gravity sag into the goal and creeps the arm toward the table (metal-arm `control.py` `base()`). Re-send the last commanded jaw (RoboProbe `_held_action_data`) [note: metal-arm-harness; note: robodojo].

**Known bugs to avoid.**
- **Chunking origin bug** (metal-arm): legs are capped from the *measured* pose but planned from the last *command*, so a 34° leg can exceed the 35° cap. Plan and cap from the same origin [note: metal-arm-harness].
- **"Clearance" ambiguity:** report TCP height and lowest-link clearance separately; a "down" nudge raised metal-arm's reported clearance from 100 to 111 mm.
- **Held objects:** include them in floor and collision checks. Metal-arm does not, and a held part hangs 4.5–5 cm below the tips [note: metal-arm-harness].

**Atomic contact primitives.** Never let model latency fall inside a contact phase. so101-painting's atomic `trajectory(targets, times)` stroke with a `contact_frames` postcondition is the pattern [note: so101-painting].
- Caveat: the fact-check showed most dwell came from settle frames and easing, not planning latency.
- Its simulator steps physics while idle, so client latency becomes paint dwell.

**Grasp verification.** Combine stall + torque + aperture checks (metal-arm, innate) with a short lift test, as Astra learned to do itself in Robocurve [note: robocurve-gpt6-astra].

### 3.5 Contact skills: servo first, light learned head where it earns its place

**Role.** A small library of closed-loop skills that execute the last 1–3 cm, where the evidence says LLM + IK fails:
- grasp centring under wrist-camera parallax (Fable's hindsight: the block is 3–5 cm ahead of where it appears); about half of all Tier 1 closes were empty [gap-4];
- descend-until-contact;
- insertion with compliance;
- regrasp after slip;
- place with release verification: lower into the container and open at contact. Dropping fruit from 10–15 cm caused about 11 of Opus 5.5's 42 Tier 1 placement failures [note: robocurve-gpt6-astra; gap-4].

Each skill is a tool with a **lease** (`max_s`), a stop predicate, and a receipt.

**Expected coverage, measured against Tier 1** [gap-4]: these five skills *touch* 66% of Opus 5.5's non-completed trials but would *directly fix* only about 16% (bounce-outs and unseated stacks). About 29% need things outside the skill set: reach-aware planning and scene layout (Stand Up Bottles with pitch/roll pinned at 0; a safe lid beyond reach), articulated push/pull and pour skills, collision-aware bimanual pose planning (Cap Pen: arm-body interference, unknown cap orientation), and stall handling. Fold approach–descend–close–lift–verify into one call: 100/120 Opus 5.5 trials cited the call budget. For Astra, `center_grasp`/`regrasp` matter most (no-grasp 37%).

**Build order** [gap-3; gap-6].
1. **Analytic association + servo (B2).** Associate the LLM's point with a tracked detection (mask/id), filter it (EMA), servo the wrist camera to ≤0.5 mm image error at a 3 cm standoff, descend at ≤5 mm/s, stop on the contact detector, spiral ±1 mm (0.3 mm pitch), release. This is what every cheap-arm ≤3 mm success in the evidence base does (relative visual alignment + slow descent + passive compliance, never F/T): AGP-style scripted programs reached **8/10** on four 3D-printed pairs at 0.73–2.83 mm radial clearance and **5/5** (Opus 5) on two pairs on a YAM, without learning; piper-astra-jev reached 0.6 mm relative alignment with CAD-silhouette fitting (n = 1) [gap-6; note: piper-astra-jev]. In the toy study this baseline scored 100% up to 10 mm pointer noise and 83% at 5 cm staleness [gap-3].
2. **Staged frozen VLA primitive (B3).** The planner stages a VLA from a good start pose (the Pigey / Harness VLA / VoLo pattern). On YAM this means **fine-tuning first**: MolmoAct2-BimanualYAM scored 0/100 completions in StationeryBench (mean progress 12; in 47/100 trials the arms never left the start pose) and 0/4 on three unseen tasks on Manda's simulated YAM rig; π0.5 has no YAM checkpoint (DROID-Franka, ALOHA). Budget YAM demonstrations for either [note: benchmarks-and-sim-datagen; note: manda-robotics].
3. **A trained small head (20–100M)** only for skills where 1 and 2 both fail on the 2 mm suite, e.g. residual alignment and seating under last-centimetre occlusion, compliance and jamming — exactly the effects the toy study did not model.

**What a learned head conditions on (if built).** Split the condition into *which* and *where* [gap-3]:
1. *Which*: a **tracked SAM 3 mask or object id**, refreshed every tick (PEEK/ARRO-style overlay or extra channel). In the toy study a tracked mask was invariant to keypoint noise (89–94% at σ up to 10 mm) and degraded gracefully at 5 cm staleness (75–78%) where keypoint heads collapsed to ≤2%.
2. *Where*: the head's own wrist perception. If a 3D keypoint is added, express it relative to the tracked object, and **train it with noise matched to the measured pointer error** ([gap-2] eval) **and the measured spec-age distribution** ([gap-1]); never on exact hindsight points (exact-label heads: 100% → 59% at 2 mm noise → 20% at 5 mm).
3. Wrist RGB at 128–224 px, plus a crop of the overhead view around the target.
4. Proprioception, including the contact signals the YAM actually provides: Cartesian lag (commanded − FK(measured)), J2/J3 effort deltas, gripper effort [gap-6]. "Rising effort while position stops tracking indicates contact" [note: robocurve-gpt6-astra].
5. A **skill/phase token** from a closed vocabulary (`center_grasp`, `insert`, `place_release`, …), optionally with an episode-metadata field such as speed/quality (π0.7 style) [note: vla-and-action-heads].
6. The age of the target spec.

**What it outputs.**
- A 16–50-step chunk of **relative EEF deltas** at 30–50 Hz, capped at ≤0.3 mm/step in the final insertion phase (ENPIRE's YAM pin-insertion setting) [gap-6]. GR00T N1.7 found relative EEF "a key factor" for cross-embodiment; π0.7 saw no clear advantage of EE over joints [note: vla-and-action-heads].
- A **progress/completion scalar** and a failure flag. Helix S1 predicts its own task completion; this feeds the verifier [note: vla-and-action-heads].

**Size.** 20–100M parameters:
- ACT is 80M, runs at 10 ms on a 2080 Ti, and reaches 80–90% from 50 demos;
- RoboDual's specialist is 16.2M (+7.5M per extra sensor);
- an RLT-style residual is a 2-layer, 256-unit MLP on a reference chunk [note: vla-and-action-heads; gap-3].

Flow matching vs L1 regression showed "no detectable advantage" at small scale (MINERVA). Chunk length and vision allocation mattered most.

**Training recipe.**
- Behaviour cloning with action chunking, on the *final 3 cm only*, starting from the scripted servo's hand-off pose [gap-6].
- **No training-time RTC for a small local head.** ACT runs in about 10 ms, so there is no chunk-boundary delay to absorb; LeRobot's TT-RTC (PR #4056) is π0.5-only and not in the v0.6.x tags. The real staleness is a target spec seconds old, far outside RTC's ≤300 ms regime. Use target re-binding every tick plus spec-age/staleness augmentation instead. Use RTC only if the "head" is a remote GPU VLA [note: vla-and-action-heads; note: engineering-challenges].
- **Anti-shortcut measures** (condition dropout alone is *not* one) [gap-3]:
  - condition noise sized like the demo displacement (GAE: scale 0.10 best, 0.50 hurts; Fast Plans: σ_c = 0.7 normalized), plus a coarse-stage gate;
  - **redundant-cue dropout** (50%): drop the scene/language cue that already identifies the target, so the head must read the LLM condition. In the toy study, 20% condition dropout made erasure tests pass trivially (Δ ≈ 0 pp) while 96–100% of episodes reached *neither* target when condition and scene disagreed; dropping the redundant cue gave 44–84% following;
  - counterfactual relabelling (demos that go where the condition says, even against the scene cue) — untested, next step;
  - null dropout 0.1–0.5 only as a robustness add-on (RoboDual 0.1, Fast Plans 0.15, RLT 50% reference masking).
- **Evaluation must report**: success, erasure Δ, sensitivity, retained visual response (the Fast Plans protocol) **and a counterfactual shift test** (the condition points at a distractor). Any one alone misleads.

**Data sources, in order of cost.**
1. **Scripted plus noise in sim.** Opus writes the scripted controller for each skill against privileged sim state; the EmbodiedSWE-Gen clean-label noise channel and open-loop replay gate multiply it into hundreds of episodes.
   - Label with **commanded joint/EEF targets at about 15 Hz**: achieved joint positions at 60 Hz gave 0% for π0.5 in EmbodiedSWE [note: embodiedswe].
   - Randomize perception (masks, depth noise) as well as physics.
   - Avoid welded or teleported grasps in the asset (EmbodiedSWE's weld list).
2. **Teleop** with a leader arm: 50–100 demos of the final 3 cm per skill (ACT's 50-demo regime). Add a few real demos even if sim data dominates; RoboTwin 2.0 reports +367% with 10 real demos vs +228% sim-only [note: benchmarks-and-sim-datagen]. Behaviour cloning alone on insertion is mixed: ACT battery-slot insertion 96% from 50 demos, ACT velcro 20% from 100, ManiSkill peg (3 mm clearance, 100 demos) 38% for state-based Diffusion Policy and **0% for every RGB policy** [gap-6].
3. **On-robot correction.** Harness rollouts as DAgger: every time the orchestrator, verifier or a human corrects a skill, log the (state, correction) pair; LeRobot's `lerobot-rollout` has a DAgger strategy; v3.1 datasets carry tool-call atoms and language columns [note: vla-and-action-heads; note: github-implementations-sweep]. Or ENPIRE-style RLPD with BC regularization (50 consecutive YAM pin insertions from one wrist camera with ≤0.3 mm steps, ≤8 retries allowed). Budget 1–5 h of robot time per skill (RL Token: 15 min–5 h) [gap-6].
4. **Offline RL / RLT residual** on the critical phase for speed; up to 3× faster in RL Token [note: vla-and-action-heads].

**Fallback for an untrained skill.** "IK + wrist visual servo + compliance" (build-order step 1), not a VLA: neither π0.5 (no YAM checkpoint) nor MolmoAct2-BimanualYAM (0/100 completions in StationeryBench) is a working zero-shot fallback on YAM. Treat both as fine-tuning bases that need YAM demonstrations [note: benchmarks-and-sim-datagen; note: manda-robotics].

### 3.6 Tool and function schema sketch (Anthropic Messages format)

**Shared rules for all tools.**
- `"strict": true` and `"additionalProperties": false`.
- Numeric bounds and array lengths are written in the description *and* enforced server-side (R16). Strict mode does **not** enforce `minimum`/`maximum`/`multipleOf`, string lengths or complex array constraints (verified in the Claude structured-outputs reference); out-of-range or wrong-length arguments come back as `is_error:true` with an instruction-phrased reason.
- Units are in field names. Angles are relative to a "straight-down grasp" reference (RoboProbe `pose.py`). Never require 3×3 rotation matrices; `pilot.py` demanded them orthonormal to 1e-3 [note: llm-robotics-playground].

```json
[
 {"name": "observe",
  "description": "Read-only. Capture a synchronized observation with the arm at rest. Returns one tiled image (views left→right as requested) with overlays, plus a JSON state block. Call after every world-changing action before judging its outcome.",
  "strict": true,
  "input_schema": {"type": "object", "additionalProperties": false, "required": ["views", "overlays"],
    "properties": {
      "views":    {"type": "array", "items": {"enum": ["overhead", "wrist_left", "wrist_right"]}},
      "overlays": {"type": "array", "items": {"enum": ["object_ids", "tcp_projection", "grasp_axis", "table_grid_5cm", "workspace_bounds"]}}}}},

 {"name": "zoom",
  "description": "Read-only. Return a native-resolution crop of the last observation. Coordinates are ABSOLUTE pixels in the image you were shown.",
  "strict": true,
  "input_schema": {"type": "object", "additionalProperties": false, "required": ["view", "center_px", "half_size_px"],
    "properties": {"view": {"enum": ["overhead", "wrist_left", "wrist_right"]},
                   "center_px": {"type": "array", "items": {"type": "integer"}},
                   "half_size_px": {"type": "integer", "description": "32-256"}}}},

 {"name": "locate",
  "description": "Read-only. Open-vocabulary detection + segmentation + depth. Returns objects with ids you MUST use in later calls: {id, label, score, centroid_m:[x,y,z] (base frame), extent_m, top_z_m, depth_valid_frac, visible_in:[views]}.",
  "strict": true,
  "input_schema": {"type": "object", "additionalProperties": false, "required": ["queries"],
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}}}},

 {"name": "pick",
  "description": "Grasp object_id. The skill plans approach, centres with the wrist camera, closes, and lift-tests. Receipt reports grasp state from aperture/effort, not from images.",
  "strict": true,
  "input_schema": {"type": "object", "additionalProperties": false, "required": ["object_id", "arm", "grasp_mode", "part_hint", "note"],
    "properties": {"object_id": {"type": "string"},
                   "arm": {"enum": ["left", "right", "auto"]},
                   "grasp_mode": {"enum": ["top", "side", "auto"]},
                   "part_hint": {"type": ["string", "null"], "description": "e.g. 'handle'; resolved by the part pointer"},
                   "note": {"type": "string", "description": "1-2 sentences: what you see and why"}}}},

 {"name": "place",
  "description": "Place the held object relative to a target object/region. Clears rims before lateral moves; releases; verifies the object is at rest.",
  "strict": true,
  "input_schema": {"type": "object", "additionalProperties": false, "required": ["target_id", "relation", "note"],
    "properties": {"target_id": {"type": "string"},
                   "relation": {"enum": ["in", "on", "left_of", "right_of", "in_front_of", "behind"]},
                   "note": {"type": "string"}}}},

 {"name": "move_tcp",
  "description": "Bounded Cartesian move of the grasp point (between fingertips). Target is in base frame or relative to an object. Max 0.25 m and 35 deg per call; speed is capped by the controller. Unnamed dimensions hold their last COMMANDED value. Gripper is never changed by this tool.",
  "strict": true,
  "input_schema": {"type": "object", "additionalProperties": false, "required": ["arm", "frame", "xyz_m", "yaw_deg", "pitch_deg", "note"],
    "properties": {"arm": {"enum": ["left", "right"]},
                   "frame": {"type": "string", "description": "'base' or 'object:<id>'"},
                   "xyz_m": {"type": "array", "items": {"type": "number"}},
                   "yaw_deg": {"type": ["number", "null"]},
                   "pitch_deg": {"type": ["number", "null"]},
                   "note": {"type": "string"}}}},

 {"name": "run_skill",
  "description": "Run a closed-loop learned/servoed contact skill under a lease. Returns when stop_when holds, the lease expires, or a guard trips.",
  "strict": true,
  "input_schema": {"type": "object", "additionalProperties": false, "required": ["skill", "target_id", "max_s", "note"],
    "properties": {"skill": {"enum": ["center_grasp", "descend_until_contact", "insert", "place_release", "regrasp"]},
                   "target_id": {"type": "string"},
                   "max_s": {"type": "number", "description": "1-20"},
                   "note": {"type": "string"}}}},

 {"name": "run_policy",
  "description": "Hand control to a language-conditioned VLA for one bounded segment. 'ok' means only that the segment ran, never that the subtask is done.",
  "strict": true,
  "input_schema": {"type": "object", "additionalProperties": false, "required": ["instruction", "max_s"],
    "properties": {"instruction": {"type": "string", "description": "one short subtask, <=200 chars"},
                   "max_s": {"type": "number", "description": "2-10"}}}},

 {"name": "verify",
  "description": "Run the independent verifier on a predicate. Returns {verdict: pass|fail|uncertain, evidence: [...]} from sensors and a separate vision model. Required before finish(success).",
  "strict": true,
  "input_schema": {"type": "object", "additionalProperties": false, "required": ["predicate", "args"],
    "properties": {"predicate": {"enum": ["holding", "gripper_empty", "object_in", "object_on", "object_at_rest", "seated", "stacked", "upright", "subgoal_done"]},
                   "args": {"type": "object", "additionalProperties": false, "required": ["object_id", "target_id", "subgoal_index"],
                            "properties": {"object_id": {"type": ["string", "null"]},
                                           "target_id": {"type": ["string", "null"]},
                                           "subgoal_index": {"type": ["integer", "null"]}}}}}},

 {"name": "finish",
  "description": "End the episode. Termination and the success claim are separate: status says why you stop; claimed_success says whether you believe the goal is met. claimed_success=true is REJECTED unless the last verify(subgoal_done) for every subgoal passed. Use status=budget_exhausted or partial to report honest partial progress.",
  "strict": true,
  "input_schema": {"type": "object", "additionalProperties": false, "required": ["status", "claimed_success", "summary", "hindsight"],
    "properties": {"status": {"enum": ["goal_reached", "partial", "budget_exhausted", "give_up"]},
                   "claimed_success": {"type": "boolean"},
                   "summary": {"type": "string"},
                   "hindsight": {"type": "string", "description": "Concrete, transferable facts about this rig/task you wish you had known at the start"}}}}
]
```

The `hindsight` wording follows Robocurve's `_HINDSIGHT_DESCRIPTION`. The `status`/`claimed_success` split follows [gap-4]: with only `done`/`give_up`, Opus 5.5 ended 34 Tier 1 trials with `done` to report partial progress honestly, which a naive scorer reads as overclaiming. Other tools: `point_to_3d`, `gripper`, `hold`, `read_text`, `recall`/`remember`.

**Receipt returned in every motion `tool_result`.** Never return "executing over N steps"; that message was synthesized before playback in Robocurve [note: robocurve-gpt6-astra].

```json
{"status": "reached | stopped_short | rejected | aborted | lease_expired",
 "reason": "e.g. 'pose needs joint2=-0.70 past the -0.50 floor; approach from the front instead'",
 "commanded": {"tcp_xyz_m": [0.381, -0.370, 0.076]},
 "measured":  {"tcp_xyz_m": [0.379, -0.364, 0.081], "residual_mm": 8.0, "residual_deg": 2.1, "tolerance_mm": 3.0},
 "grasp": {"state": "holding | empty | slipped | resting_on_object", "aperture_mm": 34.1, "effort_nm": 1.03},
 "contacts": {"unexpected": false},
 "approver": {"clamped_steps": 0, "rules": []},
 "skill": {"progress": 0.92, "frames_used": 140},
 "elapsed_s": 3.4,
 "obs_ref": "obs_042",
 "budget": {"llm_calls_left": 17},
 "next_obligation": "observe"}
```

This combines OpenETA's "fresh-observation obligation", RoboICL's receipt grammar and RoboProbe's arrival check [note: github-implementations-sweep; note: robodojo].

**Rejections** return `is_error:true` with an instruction-phrased reason, in the style of metal-arm's "aim higher", "lift first" and "open, descend lower, close again".

**Repeat handling.**
- After 3 consecutive rejections of any kind, append a `stuck_hint` and force an `observe` (innate).
- After 5, hold and escalate.
- This prevents GPT-as-Policy's 7,750-call runaway [note: gpt-as-policy].

### 3.7 Observation design

**Tiling.** One tiled image per turn: overhead | wrist_left | wrist_right, each 640×480, so 1920×480 ≈ 1,242 tokens. RoboICL's triptych gave 92.4% cache hits.
- A single image block sits more easily within the 20-block cache lookback [note: engineering-challenges].
- Pre-resize locally and set `"transformations": {"oversized_image": "error"}` on each image block, so the coordinates the model sees are exactly yours [gap-2].
- **Verification view.** Before `finish(claimed_success=true)` and at contact-skill ends, capture a dedicated view at ≥448 px: an oblique or side shot zoomed on the container, groove or junction. At 224 px the decisive region of 19 of 22 Tier 1 false claims was not observable [gap-4; gap-5].
- **Budget counter.** Every observation states remaining LLM calls and steps; Tier 1 models counted calls themselves and 100/120 Opus 5.5 final texts cite the budget [gap-4].

**Overlays** (drawn on the image, with listed meanings in the system prompt):
- projected TCP and grasp axis on the overhead and wrist views. Every Astra trial spent calls re-deriving the frame mapping;
- a 5 cm table grid (RoboDawn: +15 pts);
- object-id marks from the object table (Set-of-Mark);
- workspace bounds.

Show-Harness's naming ablation (20/20 with named units and an explained image-direction convention vs 1/20 with neither) says **state the image↔robot axis conventions explicitly** [note: github-implementations-sweep].

**Numbers as text, not pixels.** Joint and TCP state rounded to 4 decimals (innate #737), object table, aperture/effort, last receipt, budget left. CaP-X: raw RGB per turn (M2) did *worse* than text feedback (M1); visual differencing as text (M3) did best [note: llm-planner-codegen-lineage].

**Depth.** Never send a depth image to the LLM. Robocurve never used its rendered-depth option [note: robocurve-gpt6-astra]. Send metric results via `locate` and `point_to_3d`.

**Capture timing.** Capture when the arm is stationary (after settle) and stamp frames with capture time. Frames 5.6–12.5 s old confused quackd's pilot [note: quackd].

**Mandatory looking.** Count image reads per world-changing action and nudge if zero. Opus 5 read 6 images in a multi-hour syringe run and reported "57% draw" on empty rollouts [note: embodiedswe].

**Ablation in evaluation.** Run blind-image and mirrored-image checks. Several models are "image-invariant" in 2608.06154; that was drone and car sims with small models, so it may not transfer, but the test is cheap [note: engineering-challenges].

### 3.8 Control-loop timing: async reasoning vs fast control

```
time (s)   0          5          10         15         20         25
L4 Opus    [plan: locate→pick(o3)]           [verify+place(o3,bin)]           [verify→finish]
            ~5-16 s (1st call slow) ▲receipt  ~5-12 s (p50-p90)    ▲receipt
L2 skills            |pick(o3): approach→center_grasp(head 30 Hz)→lift-test|  |place: clear rim→release|
L3 grounding  ── tracker/masks 5-10 Hz (re-binds targets every tick) ────────────────────────────────────
Verifier                                     [grasp:holding ✓ + lift test]    [object_in ✓: sensors + Opus low-effort check]
L1          ═══ 200 Hz-1 kHz servo · envelope · heartbeat 0.5 s · lease watchdog ═══════════════════════
```

**Leases, not open-ended commands.**
- Every skill or VLA call has `max_s`; expiry means a controlled stop.
- DrivingBench's contract: "Motion continues while you think; expiry starts braking", with a 2 s native watchdog [note: github-implementations-sweep].
- quackd's heartbeat is 0.5 s: one miss sends `stop` [note: quackd].

**Speculative next call.** Once a skill is ≥80% through its expected duration, the harness can issue the next LLM call with the *predicted* receipt. It commits only if the actual receipt matches.
- TypeGo reports −50% per-step delay and −73% time-to-first-action.
- Speculative Actions reports up to 20% latency cut.
- Restrict this to idempotent or reversible steps [note: engineering-challenges].
- Slow calls cannot be predicted from the previous call (lag-1 latency autocorrelation 0.18 for Opus 5.5), but output length explains 91% of between-trial latency variance. **Bounding output length** (effort, per-message effort for routine steps, terse tool schemas) is the more reliable lever [gap-1].

**Fewer, larger decisions.** URAI's multi-phase tools ran 1.3–1.5× faster with 1.5–1.7× fewer output tokens [note: engineering-challenges]. The skill library should make a pick-and-place 2–4 LLM calls, not 15–40. At Tier 1 steady state, 2–4 calls cost ≈10–21 s of LLM time at p50 and ≈20–40 s at p90 per pick-place, the dominant term [gap-1].

**First call.** Every Opus 5.5 Tier 1 trial's first call exceeded 8 s (p50 16.1 s, cold cache plus initial planning). Issue the planning call while the arm is at rest, and pre-warm the cached prefix with a `max_tokens: 0` request [gap-1].

**Safe waiting.** When a loaded arm waits on the LLM, the harness lowers or parks it based on load, torque and motor temperature (R10), not on a fixed 8 s timer, which would fire on 23% of calls. Arms that stop moving must still be servo-held. Metal-arm's resume rules show how to re-adopt a powered hold safely [note: metal-arm-harness; gap-1; gap-6].

**Head-side delay handling.** Never block actuation on the API. A small local head (ACT-class, ≈10 ms) needs no RTC; it handles seconds-old target specs by re-binding the target every tick and by staleness augmentation in training (§3.5). Use RTC/TT-RTC only for a remote GPU VLA tool [note: vla-and-action-heads].

### 3.9 Memory and context management

**Prefix layout and caching.**
1. `tools` (deterministic order).
2. System prompt: role, rules, robot card with frame conventions, TCP definition, limits, the "mistakes already made" list.
3. Episode header: task, plan schema, calibration numbers, demo block if any.
4. Append-only turns.

- Put `cache_control` breakpoints after (2) with a 1-h TTL, after (3), and on the latest turn.
- Verify caching with `usage.cache_read_input_tokens`. The cache-diagnostics beta (`cache-diagnosis-2026-04-07`) reports where a prefix diverged.

**Never delete or rewrite earlier turns.**
- Image eviction by mutation, as in Inspect Robots `_evicted_view` and RoboProbe's stub rewrite, breaks the cache: in Tier 1 it made cache writes 54% of Opus 5.5's per-trial cost [gap-1].
- On Opus 5.5 it also invalidates preserved thinking; for accounts created on or after 2026-08-31 a replayed thinking block after such an edit is a 400 (§1.8) [note: robocurve-gpt6-astra; note: robodojo].
- Bound the context instead with server-side **compaction** at subgoal boundaries, or **context editing** (`clear_tool_uses_20250919` with `clear_at_least`, so each invalidation is worth its rewrite).
- Compaction headers resolved: `compact-2026-01-12` is threshold compaction; `compact-2026-09-04` is the on-demand `compaction` parameter in Opus 5.5's feature set (not on Bedrock). Use the on-demand variant at subgoal boundaries; server-side compaction keeps later thinking blocks valid because the binding prefix restarts at the compaction block. Do not use client-side keep-tail compaction.
- Anthropic found that truncating context "did not significantly hurt any model except Opus 4.6", provided turn 1 is kept; "models rely much more on the recent past" [note: engineering-challenges].

**Decision-time hints go in mid-conversation `role:"system"` messages,** not static rules.
- Cross-model evidence only: quackd issue #25 moved correct `feasible` verdicts from 5/30 to 27/30 (p = 1.14e-8) with one observation-time sentence, while rewording the static prompt did nothing (4/30 → 3/30). That was **Qwen3-32B-AWQ in `microduck:sim2d` with temperature pinned to 0**; transfer to Opus 5.5 is unmeasured [note: quackd]. Plan an n ≥ 30 test on Opus 5.5.
- Anthropic's own guidance is consistent: deliver per-turn reminders as turn-scoped system messages rather than editing history, and pair a long system prompt with a one-line reminder near the end (Claude API migration notes). That placement moves behaviour *more* than system-prompt wording is a plausible reading, not a measured Anthropic claim (UNVERIFIED).
- Turn-scoped messages with `clear_at: "next_user_message"` (beta `mid-conversation-system-clear-at-2026-08-21`) are available on the same models as mid-conversation system messages, **including Opus 5.5** (Claude API, Claude Platform on AWS, Bedrock, Vertex; not Foundry). Leave earlier copies in place.

**Cross-episode memory.**
- `hindsight` → `prior_learnings.md`, injected with the caveat "may be wrong or stale; the current observation always wins" (Robocurve's wording) [note: robocurve-gpt6-astra].
- A versioned "mistakes already made" document (metal-arm `docs/OPERATING.md` §5) [note: metal-arm-harness].
- A skill and verifier library with DexAgent's rules: fork, never overwrite; admit a skill only after N parameter variants pass [note: dexagent].
- **State the memory regime in every evaluation.** LIBERO-Pro scores range from 44–53% with no memory (Pigey) to 82–93% with per-task seed traces (Harness VLA) [note: llm-planner-codegen-lineage].

### 3.10 Verification and recovery

**Three tiers; `finish(claimed_success=true)` requires tiers 1 and 2 to pass. The harness fails closed. The routing is by predicate class, because vision is reliable for placement and weak for seating, stacking, capping and upright [gap-5].**
1. **Deterministic, every step:**
   - grasp state from aperture, stall and effort, plus a short lift test (on YAM, gripper effort alone cannot tell an empty close from a grasp) [gap-6];
   - per-object "held, then released" gripper predicate, with thresholds calibrated per object (the same thresholds passed 31/32 bowl/puzzle successes but 0/5 true bowl stacks) [gap-5];
   - arrival residual against the measured tolerance;
   - motion settled;
   - object at rest for ≥0.3 s (llm-robotics-playground dwell predicates);
   - containment capped at the rim;
   - contact/effort stall at insertion depth for `insert` and `seated`;
   - wrong-object check against the plan (Manda's success semantics) [note: llm-robotics-playground; note: manda-robotics].
2. **Independent model check at subgoal ends.** A separate context with no access to the agent's reasoning (Pigey's `runVerifier` idea, but fail *closed*; Pigey's `real/agent.ts:347` fails open on API/parse errors).
   - Input: a Manda-style **timestamped contact sheet** (20 frames, 5 columns, burned-in times, last tile stamped FINAL) **plus the full-resolution final frame**, plus the verification view. Prompt rules: "Judge failure by what you can see. Do not infer success from the task description." and "An object that is near, touching, or resting on the edge of its target is not 'in' or 'on' it" [gap-5].
   - Add (untested fix): "Judge the end state from the FINAL tile and final frame only; earlier tiles only help locate objects", to stop process inference ("the other bowls are no longer visible, so they are nested").
   - **Placement predicates** (`object_in`, `object_on`, containment): one Opus 5.5 check at `effort:"low"` (BA 0.968, 0/92 false accepts, $0.017) AND the gripper predicate.
   - **Contact predicates** (`seated`, `stacked`, `upright`, capping): vision is advisory (BA ≈ 0.71, 25% false accepts). Require a geometric or force signal; otherwise return `uncertain` and request the verification view.
   - A second vendor on the same pixels does not de-correlate errors (Opus/Astra κ 0.91). If you add a second check, change the *evidence* (end frame vs contact sheet, a different camera, a probe), not just the model. Do not tune with `p_success` thresholds: at 95% true-success retention every verifier still passed 15–26% of failures.
   - Gemini ER 2 (video success detection 82.4, vendor chart) and LeRobot's Robometer-4B (F1 0.81, author-reported) are candidates **not yet measured on this corpus**; `gaps/gap-5-verify.py` needs only a new `CONDS` entry [gap-5].
3. **The LLM's own judgment.** It must cite which images it inspected. "A filepath is not a visual inspection" [note: embodiedswe].

**What this achieves, measured** [gap-5]: an Opus 5.5 agent falsely claims success on 8.4% of failed Tier 1-like trials; with one Opus 5.5 verifier this stays at 5.9% (7/119); with the post-hoc "Opus end-frame AND Astra" pairing it falls to 0.8% but rejects 4 of 6 true successes. Every verifier rejected Opus 5.5's only real Tier 1 completion. Fail-closed gating on contact tasks trades overclaim directly against false rejects unless the evidence improves.

**Recovery policy, encoded in the harness rather than the prompt.**
- Typed failure → retry with a changed approach. "If the same approach fails twice, change the approach" (innate prompt; EmbodiedSWE `tried_from`).
- Refuse exact repeats of a target that already failed (innate).
- Return to a verified stage-entry state, a real-world "checkpoint".
- Escalate effort for one turn.
- After K failures, ask a human with a structured question.

**A learned critic is the highest-leverage research item** (sim labels for progress and contact are free), but treat it as open. Supervised fine-tuning on progress estimation reached only 45.4% [note: benchmarks-and-sim-datagen].

### 3.11 Safety envelope

Defence in depth, with every layer independent of the LLM:

| Layer | Mechanism | Borrow from |
|---|---|---|
| Hardware | Physical e-stop on robot power; firmware torque/current limits; separate motor-temperature watchdog. Note: PiPER never sends RESUME after e-stop, because "the arm drops" | piper `arm.py` [note: piper-astra-jev] |
| Driver | Device manifest of what can be measured, what can be adjusted and which limits are enforced (MHS-style); limits enforced in the driver | MHS [note: industry-competition] |
| Motion approver | Workspace box; measured floor + 10 mm margin; recovery-only-upward; slow zone near the table; keep-out boxes from depth occupancy; held-object geometry; per-call excursion cap; speed/accel/jerk caps; per-tick delta limit vs the last *approved* action; FK-vs-measured self-check (2 mm / 0.01 rad) | `metal-arm-harness/src/metal_arm_harness/safety.py`; `inspect-robots/src/inspect_robots/approver.py`; `GPT-as-Policy/.../kinematics.py` `check()` |
| Execution contract | Fixed gate order: abort (stop always allowed) → allowlist → params → confirm → budgets → preconditions → dry-run → execute racing abort; capability locks (one owner per arm); single command bus | `quackd/quackd/safety.py` (`_run_verb`); `dimensionalOS_dimos/dimos/agents/capabilities.py` |
| Liveness | Leases with expiry braking; 0.5 s heartbeat; stale-observation discard; thermal and overload park | DrivingBench `shared/contracts.py`; quackd; innate runtime |
| Semantic | Scene-text masking; human confirmation for novel or high-energy actions (motion grants); refusal → safe hold; a RoboHarm-style eval in CI | `strands-labs_robots/strands_robots/_motion_grants.py`; RoboHarm [note: robocurve-gpt6-astra] |
| Process | Unarmed dry-run mode on the *same* code path. Metal-arm's unarmed mode skips settle and later legs; fix that. Fail-closed tool classification (a new tool must be classified or tests fail). Full audit log | metal-arm; `wise-vision_ros2_mcp/server/tool_safety.py` |

**Also.**
- Make the governor and the prompt the **same** table. piper's governor allowed `open_gripper` mid-carry while its prompt forbade it [note: piper-astra-jev].
- Use ISO 10218-2:2025 Annex M force limits (140 N quasi-static for hands) as the design bound for any human-proximate operation [note: engineering-challenges].
- Never give the model Python execution inside the robot or simulator process. Codegen goes to a separate sandbox that calls the same governed tools (llm-robotics-playground's own warning; GRID's `validate_robot_calls`) [note: llm-robotics-playground; note: general-robotics-auto-engineering].

### 3.12 Provider abstraction (Claude, GPT, Gemini)

```python
class Turn(Protocol):            # canonical, provider-neutral
    role: Literal["system","user","assistant","tool"]
    blocks: list[Text | Image(png_bytes, w, h) | ToolCall(id, name, args) | ToolResult(id, json, is_error) | Opaque(provider, payload)]

class Backend(Protocol):
    caps: Caps   # forced_tool_choice, temperature, effort_levels, coord_convention ('abs_px'|'norm1000_yx'),
                 # max_images, image_token_fn, cache_mode ('explicit'|'auto'), binds_thinking, refusal_semantics
    def step(self, prefix_id: str, turns: list[Turn], tools: list[ToolSpec], effort: str) -> Reply  # Reply has usage, latency, stop_reason
```

**Adapters.**
- **Anthropic Messages.** Native; `cache_control` breakpoints; replay `thinking` blocks unchanged; per-message effort; `strict` tools; `auto` tool choice; refusal handling; server-side fallbacks (beta `server-side-fallback-2026-07-01`) optional.
- **OpenAI Responses.** `store:false`, `include:["reasoning.encrypted_content"]`, replay output items by `call_id` (`inspect-robots/.../_responses.py`). Astra rejects `none` effort; tool calling requires Responses [note: frontier-models-embodied-2026].
- **Gemini.** `interactions` with `thinking_level`. ER 2 returns `[y,x]` normalized 0–1000: convert to canonical absolute pixels. The streaming endpoint lacks caching and structured outputs.

**Keep `Opaque` blocks per provider.** Reasoning state is not portable: Opus 5.5 thinking blocks are read only by Fable 5.1 / Mythos 5.1, so a fallback model runs without them.

**Record per call:** model ID, effort, beta headers, harness SHA, prompt hash, full request/response, latency and usage. Robocurve's wire capture and RoboProbe's trace schema are the templates.

**Starting point.** Inspect Robots' agent plugin already has native Messages, Responses and chat clients, so it is the fastest start [note: robocurve-gpt6-astra].

---

## 4. Model choice

### 4.1 Evidence by model (closed-loop or agentic robotics)

| Evidence (n) | Opus 5.5 | Opus 5 | Fable 5.1 | GPT-6 Astra | Others |
|---|---|---|---|---|---|
| Robocurve RoboDojo-RC Tier 1, real YAM (6 tasks × 20, tasks picked easiest-for-Astra; models run in blocks, not interleaved; graders knew the model) | **36.0% progress, 1/120 (p = 0.21 vs Astra), $0.90, 14.5k out tok, 269 s; API p50 5.28 s; false-success claims 10/119; failures led by placement/alignment (50%)** | 19.9%, 2/120, $1.76; p50 8.77 s; 10 overheats, all in each rig's last block | — | 36.7%, 5/120, $1.14, 3.8k tok, 251 s; p50 6.53 s; false-success 12/115; failures led by no-grasp (37%) | — |
| Robocurve bowl / puzzle, real (20 each; different rigs for bowl; Fable via compat shim; not interleaved) | — | — | 8/20, 2/20 (1/20 by live verdict) | 19/20, 2/20 | Fable 5: 1/20, 0/20 |
| URAI native → with tools written by a Fable 5.1 agent (5 × 5; Claude at xhigh, Astra at medium) | 32% → 60% (best point estimate; vs Astra p = 0.32 native, 0.78 with tools) | — | 16 → 52% | 16 → 52% | DS-V4-Flash 8 → 48 |
| Independent verifier on 235 real YAM episodes (low effort) [gap-5] | BA 0.871, $0.017/check; contact tasks BA 0.707 | — | — | BA 0.886, $0.035/check | GPT-5.6 Sol 0.869 |
| CodeActionBench (75 attempts) | — | 49.3% (ref) / 45.3% (Claude Code) | — | 73.3% | Gemini 3.6 Flash 20%, Sonnet 5 2.7% |
| LIBERO-Agent score (30 × 3) | — | 19.3 | 15.8 | 45.0 | Sol 10.8 |
| AGP two-pair assembly, real (5) | — | 5/5, 22.2 min, $9.75 | 3/5, 27.7 min | 5/5, 9.2 min, $4.47 | Sol 5/5, Luna 0/5 |
| EmbodiedSWE, offline coding (28 tasks) | — | 50% (hack 7%) | 61% (4%) | 82% (0%) | Sol 18% (43% hacks) |
| VIA, sim (xhigh) | — | — | — | — | CC-Fable 5 88%; CC-Opus 4.8 70% |
| Show-Harness zero-shot tier | — | in the 86–96% top tier | — | — | Gemini 3.1 Pro, GPT-5.6-sol also in top tier |
| Generalized TAMP, coding | — | 82% | — | 95% | Sol 56%; planners 47% |
| Orchestrating a VLA | — | — | — | Harness VLA 92.6% | Opus 4.7 82.4%; VoLo best orchestrator Opus 4.6 41.8 vs GPT-5.5 35.5 |
| ER 2 chart (vendor; no Opus 5.5 row) | — | Opus 5: video success detection 81.0; progress 37.1 (lowest); safety instructions 95.9 | — | not compared | ER 2: 82.4 / 57.4 / 97.9 |
| PhysBrain 1.5 Table 4 grounding (low effort; 336 px, normalized coords; no Opus 5.5 row) | — | Opus 5: RefSpatial 56.3; PixMo 56.7; PointBench 68.6; Part-Affordance **78.1**; VSI 21.3; overall 67.9 | — | 78.0; 75.2; 71.9; 55.0; 59.8; 73.3 | Gemini 3.6 Flash 76.4; 74.8; 69.6; 64.7; 51.8; 73.0 |
| Real2Sim coding (n = 1) | cheapest ($5.70–5.98), wrong dynamics | — | only working gear coupling | best visuals, most expensive | — |

Sources: [note: robocurve-gpt6-astra; note: frontier-models-as-direct-policies; note: benchmarks-and-sim-datagen; note: llm-planner-codegen-lineage; note: embodiedswe; note: frontier-models-embodied-2026; note: manda-robotics].

**Reading.**
- Astra leads direct-policy and code-as-policy head-to-heads against **Opus 5 and earlier**, often by 2–3×. Every Claude row except Tier 1, URAI and the verifier study is Opus 5 or older; Opus 5.5 "thinks more per turn than Opus 5 at the same effort", so transfer is an assumption.
- Opus 5.5 is the only Claude with real-robot evidence at Astra's level of *progress* (Tier 1), and in URAI it has the highest point estimate. Neither study separates the two models: Tier 1 completions 1/120 vs 5/120 (p = 0.21, order-confounded, unblinded), URAI p = 0.32–0.78 [gap-4].
- The Tier 1 failure signatures differ more than the totals: Opus 5.5 gets further and then fails at placement/alignment (50% vs 39%), Astra fails earlier at grasping (37% vs 13%, p = 1.7×10⁻⁵). Few of Opus 5.5's placement failures are millimetre misses; they are release strategy (drops from height), bimanual interference, unknown cap orientation, unseated bowls and scene layout (pinned wrist pitch/roll; lid out of reach) [gap-4]. "Last centimetre" holds only in that narrower sense.
- Rankings flip with the harness [note: frontier-models-as-direct-policies], so these tables cannot settle the choice for *your* interface.
- Gemini 3.x Flash is about 6× faster in output and roughly matches Astra at grounding, but is weak at agentic manipulation (20% on CodeActionBench).

### 4.2 Recommendation and routing

| Role | Default | Alternative to A/B | Why |
|---|---|---|---|
| Orchestrator, planner, recovery | **Opus 5.5 @ medium**, per-message escalation to high | **GPT-6 Astra @ medium/low** (same tools, native Responses wire) | Cheapest Opus ($4/$20, $0.20 cache reads); lowest median per-call latency of the three Tier 1 models (p50 5.28 s vs Astra 6.53 s; p90 about equal); Tier 1 progress parity; URAI point estimate highest but not significant. Astra is the strongest general robot model on third-party benches, so it must be the comparison arm |
| Pointing, part grounding | **Opus 5.5 with zoom + geometric refinement** (default) | ER 2 (k = 1/3), PhysBrain 1.5-8B, MolmoPoint-8B, Embodied-R1.5-8B; choose by the [gap-2] §5 millimetre eval | No Opus 5.5 or ER 2 pointing data exists; part-level evidence favours Claude; RefSpatial gap is protocol-confounded; GroundingPI weights unreleased [gap-2] |
| Progress classification | Learned critic on sim labels (open problem) | ER 2 (vendor 57.4 vs Opus 5 37.1) | Zero-shot progress is near chance; SFT reaches 45.4% [note: benchmarks-and-sim-datagen] |
| Independent verifier | Sensors + per-object gripper predicate + **Opus 5.5 at low effort on contact sheet + final frame + verification view**, routed by predicate class | ER 2 video success detection or local Robometer-4B, *after* measuring them on the [gap-5] corpus | Measured BA 0.871 at $0.017; a second vendor shares 75% of errors, so diversify the evidence, not the vendor [gap-5] |
| Routine discrete transitions | FSM in code; optionally self-hosted Kev-4B / Laya with `escalate` | Sonnet 5.5 with `thinking:{type:"between_tools"}` at low (latency UNMEASURED; with `between_tools`, per-message effort changes return 400) | 18–40 ms vs seconds; never numbers |
| Offline engineer (skills, sim, data) | Opus 5.5 or Fable 5.1 in Claude Code; Astra in Codex | — | EmbodiedSWE: Astra 82% > Fable 61% > Opus 5 50%. Fable is the most token-frugal (87 requests) |

Two caveats:
- Haiku 4.5 may retire from October 15, 2026, so do not build a cheap monitoring tier on it [note: industry-competition].
- Fast mode and Astra Ultrafast ($60/$300; "up to 8× faster token generation" per NVIDIA; no independent latency measurement exists) are levers to *measure*, not to depend on. The ready-to-run benchmark `gaps/gap-1-bench.py` covers both; run first the ≈$50 subset (Opus 5.5 low/medium/per-message-low/fast, Sonnet 5.5 `between_tools` low, ER 2 point low k = 1) [note: frontier-models-embodied-2026; note: industry-competition; gap-1].

### 4.3 Staying model-agnostic

- **One tool schema, three wires (§3.12).** Never pass control through an OpenAI-compatible shim.
- **Keep the prompt free of provider idioms.** Absolute pixels internally, converted per model. No reliance on forced tool choice or `temperature`.
- **Regression gates per model.** Run a sim smoke suite and a scripted-model end-to-end test with `httpx.MockTransport`, as in metal-arm's removed API loop [note: metal-arm-harness]. Re-sweep effort on every new model; Opus 5.5 behaves differently from Opus 5 at the same effort.
- **A model registry** that records per-model capability flags (quackd keeps 117 hand-curated IDs with `forced_tools` and `binds_thinking`) [note: quackd].

### 4.4 Caching and token efficiency

1. **Append-only plus breakpoints** (§3.9). Expected 90%+ cache reads; RoboICL measured 92.4%.
2. **One tiled image per turn at 640 px per view** (≈1,242 tokens). Use `zoom` for detail instead of sending 1920 px frames (2,691 tokens each).
3. **Round floats to 3–4 decimals** and strip telemetry the model does not use. innate #817 sent full doubles; its sibling PR #737 cut total estimated cost by 43% partly through 4-decimal rounding and cache breakpoints [note: innate-os-pr817].
4. **Fewer calls.** Skills sized so that a pick-and-place is 2–4 calls. That alone is a 5–10× token reduction vs architecture A [derived].
5. **Effort.** `medium` by default; `low` for verify-only turns, via per-message effort.
6. **Never re-send the demo or plan block outside the cached prefix.**
7. **Budget accounting per episode.** Abort if spend exceeds 3× the median; RPent has `max_budget_usd` [note: github-implementations-sweep].

---

## 5. Evaluation plan

### 5.1 Task suite

**Tier S: simulation.** Physics is paused in all of these, so report wall-clock separately.

| Suite | Why | Baselines (as of 2026-10-01) |
|---|---|---|
| RoboLab-120 (Isaac Sim 6.0 port) with a `vlm_pinpoint`-style connector; **report both upstream scoring and Manda's RoboLab-Verified scoring** | Matched seeds, 15 Hz DROID, scoring fixes | Leaderboard: FLUX 3 Action 42.9%; Phoenix (TAMP+FM) 34.4%; **VoLo (agent) 28.2%**; Manda re-runs Cosmos3 35.1%, π0.5 27.5%. All of these use **upstream scoring**; the Verified patches (at-rest success, carry-required grasps, rim-capped containment) are validated on ~25/120 tasks and the full patched re-run is "planned, not yet scheduled". Scoring the harness on Verified against upstream-scored baselines biases against the harness, so re-run π0.5 under Verified scoring too (cheap: ~49.7 episodes per GPU-hour) [note: manda-robotics] |
| RoboDojo (42 tasks) via `XPolicyLab/.../deploy.py` `eval_one_episode` | Only board ranking agents next to VLAs | Astra L3 22.48% SR / 28.97 Score (1 seed, with recipes and 2 override cells). PhysicalRSI (agent+VLA) 36.27 / 31.38%. RoboICL 50.64 on 30 tasks [note: robodojo] |
| LIBERO-PRO / LIBERO-Plus | Perturbation robustness; memory regime | Pigey 44.3–53.3% (no memory); Harness VLA 82.4 / 92.6% (memory) [note: llm-planner-codegen-lineage] |
| SafeLIBERO | Collision-scored | SafeHarness 71.9% TSR / 87.5% collision avoidance [note: benchmarks-and-sim-datagen] |
| Own MuJoCo twin of the real cell | CI regression, head training, ablations | — |

**Tier R: real robot.** Physics is live; wall-clock counts.

| Task | Purpose | External baseline |
|---|---|---|
| R1 Block into bowl (YAM, Robocurve string verbatim) | Reproduce the known baseline | Astra 19/20, Fable 5.1 8/20 (external, non-interleaved, different rigs) [note: robocurve-gpt6-astra]; **re-run on own rig** |
| R2 Puzzle disc into groove ("Use your right arm only.") | A precision task with a public baseline | 2/20 Astra, 2/20 Fable 5.1 (1/20 by live verdict), external; **the real bar is a scripted wrist-servo + compliance baseline on the same rig** [gap-6] |
| R3 RoboDojo-RC Tier-1-like tasks (stack bowls, pack & pour, cap pen, stand up bottles) | Opus 5.5 vs Astra baseline | 36.0 vs 36.7% progress; 1 vs 5 /120 completions (external, blocked, unblinded). Unpin wrist pitch/roll for bottles and keep lids reachable, or report those tasks as embodiment-limited [gap-4] |
| R4 Insertion ≤2 mm clearance (Ø10 peg into Ø12 chamfered and Ø11 plain holes) | Contact-skill value | Scripted baseline per [gap-6] §6 step 7 (20 trials); external: AGP YAM scripts 8/10 (0.73–2.83 mm radial) and 5/5 (Opus 5, two pairs); ENPIRE RL 50 consecutive pin insertions (≤8 retries); piper SlideOver (n=1); GPT-Policy plug 2/3 |
| R5 Long-horizon sort (6–10 objects) | Planning + memory | — |
| R6 Perturbation: move the object mid-task; occluder; distractor | Where the LLM should earn its keep (piper lacks this) | — |
| R7 Safety: obstacle in the path; adversarial text in scene; harmful instruction | Envelope + injection + refusal | RoboHarm-style; SafeLIBERO. Scene-text injection baseline for Opus 5.5 is **unknown** (27–29% measured only on GPT-4o / Gemini 2.5 Flash) |

### 5.2 Protocol

**Design.**
- **Pre-register** tasks, initial-condition sheets (photo overlay resets; RoboDojo restores 5 objects in about 14 s), rubric (0/25/50/75/100 stages) and success predicates [note: robodojo; note: benchmarks-and-sim-datagen].
- **Interleave** arms per initial condition in ABBA order on the same rig and the same day. Robocurve's Astra and Fable runs were two days apart on different rigs [note: robocurve-gpt6-astra]; in Tier 1, every rig ran 20 Astra, then 20 Opus 5.5, then 20 Opus 5 trials, so all 10 Opus 5 overheats fell in the last block [gap-4]. Interleaving also spreads motor heat across arms; log motor temperature per trial.
- **Blind grading:** the grader sees video only, with the model hidden. Use 3 raters for disputed trials, as RoboDojo-Real does [note: robodojo].
- **Don't revise scores after the run.** Robocurve's in-harness "y" verdicts totalled 31 against 32 published completions [note: robocurve-gpt6-astra].

**Sample size.**

| Trials per arm | Minimum detectable improvement (from a 50% baseline; two-sided α=0.05, power 0.8) |
|---|---|
| 20 | ≈ 39 pts |
| 50 | ≈ 27 pts |
| 100 | ≈ 19.5 pts |

Source: [note: benchmarks-and-sim-datagen].
- Use **≥50 per arm per task** for headline claims and 20 for development. Paired designs use McNemar.
- **STEP sequential testing** saves up to 32% of trials without p-hacking [note: benchmarks-and-sim-datagen].
- Because Claude sampling cannot be pinned, every LLM-gate measurement needs n≥30 (quackd #25).

**Intervals.** Report Wilson CIs. Some reference widths: 19/20 → 76–99%; 2/6 → 10–70% [note: engineering-challenges].

**Integrity.**
- Freeze the evaluator before the first trial. so101-painting's metric was finalized after 2 of 4 runs [note: so101-painting].
- Hide the verifier verdict from the agent during benchmarking, and never give it a success oracle. OpenETA's 70.8% used `check_task`, which queries LIBERO's own success checker [note: github-implementations-sweep].

### 5.3 Metrics (per trial, logged automatically)

- **Outcomes:** success (binary), stage score, time to success (wall-clock), number of LLM calls, motion fraction of wall time.
- **Cost:** input / cached / output tokens; $ at list price, recorded with the cache state.
- **Calibration:** claim rates (done-rate, overclaim, underclaim, absent), as in CodeActionBench [note: benchmarks-and-sim-datagen]. Count an overclaim only from `claimed_success=true`, never from termination type; record whether the decisive region was observable [gap-4].
- **Latency:** per-call p50/p90/p99, first-call latency, output tokens per call, park events [gap-1].
- **Interventions:** human interventions, e-stops, approver clamps, envelope rejections, refusals.
- **Head metrics:** skill success, progress-scalar calibration, condition-ablation sensitivity.
- **Health:** thermal events.

### 5.4 Comparing against RoboDojo / Robocurve-style baselines

**Robocurve.**
- **First, a 5-call smoke test on Ilia's own Anthropic account.** Inspect Robots' `_evicted_view` (`policy.py:210-258`) rewrites earlier user turns once frames age past `image_horizon`, and `_anthropic.py` replays thinking blocks verbatim; HEAD `095172f` has no `block_binding` handling. Robocurve's runs (zero errors in 11,768 calls) most likely came from an account created before 2026-08-31. On a newer account the call after the first eviction is expected to return 400 (§1.8).
- Then run the baseline with `policy=agent`, `wire=messages` for Opus, `effort=medium`, the same 20/40-call budgets and 25% speed cap, and **one** of two disclosed departures from the published configuration:
  - set `image_horizon` large enough to keep all frames (224 px frames are about 64 tokens each, so this is cheap and also removes the eviction-driven cache writes that were 54% of Tier 1 cost); or
  - send `thinking.block_binding.prefix_mismatch_behavior:"drop_block"` with beta `thinking-binding-controls-2026-08-01` (drops every thinking block after the first mismatch, so reasoning continuity changes).
- Run Opus 5.5 and Astra on Ilia's rig, interleaved and blinded, ≥50 trials per arm per task for headline comparisons; n = 20 results are "development".
- Then swap in this harness as a new `policy` plugin. Same tasks, same rig, interleaved [note: robocurve-gpt6-astra].
- Inspect Robots is MIT and has a YAM embodiment (`inspect-robots-yam`) [note: robocurve-gpt6-astra]. Its bundled `yam_collision.xml` is collision-only (no visual geoms, no actuators); dynamics studies need MuJoCo Menagerie `i2rt_yam/yam.xml` [gap-3].

**RoboDojo.**
- Implement `eval_one_episode(TASK_ENV, model_client)`.
- Disclose: no task recipes (or report both with and without), 3 seeds, call budget, render-patch status.
- Note that the LLM adapter runs in-process and reads `task_name` from the env, a grey zone the board does not discuss [note: robodojo].

**RoboLab.** Report wall-clock and per-decision latency next to VLA query latency (π0.5 128 ms), because the sim hides LLM latency [note: manda-robotics].

### 5.5 Required ablations (each is cheap and decisive)

1. **FSM-only vs orchestrator.** Same skills, no LLM. Detects a redundant decider (piper) [note: piper-astra-jev].
2. **Learned head vs analytic servo vs frozen (YAM-fine-tuned) VLA primitive vs IK-only.** Measures the head's value on R2 and R4. In the toy study both analytic baselines beat every learned head [gap-3]; VoLo real: No-VLA 19/42 vs full 18/42.
3. **No-VLA vs VLA tool.** VoLo's No-VLA matched the full system on real hardware (overlapping CIs) [note: vla-and-action-heads].
4. **Direct-EEF baseline (architecture A) vs grounded skills.** Same model.
5. **Verifier on vs off, and verifier evidence variants** (contact sheet, final frame, verification view, sensor predicate). EmbodiedSkills: 86.2% with verification vs 48.2% without, against an 82.7% bare-π0.5 reference [note: github-implementations-sweep; gap-5].
6. **Blind-image and mirrored-image runs.**
7. **Model swap** (Opus 5.5 / Astra / Fable 5.1) at fixed harness, and **effort sweep** (low / medium / high), with latency per call.
8. **Head condition tests:** erasure Δ, sensitivity, retained visual response **and a counterfactual shift test** (condition points at a distractor). Success must follow the condition [gap-3].
9. **If a VLA-proposal hybrid is used: π0.5 with a fixed 15-step prefix and no LLM** as the control, to separate receding-horizon replanning from LLM supervision [note: gpt-as-policy].
10. **Pointer A/B** on the 300-target millimetre set: Opus 5.5 (plain, zoom, Set-of-Mark, 5 cm grid) vs ER 2 (k = 1, 3) vs local 8B pointers [gap-2].

---

## 6. MVP roadmap

### 6.1 Hardware recommendation (prices as published; check before ordering)

| Item | Qty | Unit price | Source / rationale |
|---|---|---|---|
| I2RT YAM 6-DoF arm (standard) | 2 (bimanual) | $2,999 | [note: industry-competition]. Robocurve rigs, AGP, ENPIRE's 8-station fleet, MolmoAct2-BimanualYAM checkpoint, Manda's sim YAM rig, `inspect-robots-yam` (MIT) → direct baselines. Specs: 750 mm reach, 2 kg nominal payload, 95 mm gripper, CAN, MuJoCo model [note: robocurve-gpt6-astra] |
| YAM Leader arm | 2 (bimanual leader-follower teleop) | $2,999 | For teleop data collection in Phase 3 [note: industry-competition]. One leader suffices only for single-arm demos |
| RealSense D405 (wrist) | 2 | $325 (official store; tariff surcharge since Feb 3, 2026) | https://store.realsenseai.com/buy-realsense-depth-camera-d405.html; search summary of the store page |
| RealSense D435i (overhead; D435 $375) | 1 | $399 (out of stock at time of search) | https://store.realsenseai.com/buy-realsense-depth-camera-d435i.html; D435: https://store.realsenseai.com/buy-realsense-depth-camera-d435.html |
| Workstation GPU, RTX 4090-class or better | 1 | price UNVERIFIED | piper-astra-jev ran DINO + SAM 3 on a 4090; EmbodiedSWE ran its sim on one 4090 [note: piper-astra-jev; note: embodiedswe] |
| SO-101 Pro kit (cheap second embodiment, LeRobot experiments) | 1–2 | $249.90 (+ $29.90 printed parts) | [note: industry-competition] |
| Bench-protocol tooling (dial indicator 0.01 mm, conical-seat plate, 0.25/0.5/1.0 kg weights, 0.1 N force gauge, AprilTag board, Ø10 pegs with Ø12 chamfered and Ø11 plain holes) | 1 set | price UNVERIFIED (small) | One-day YAM characterization, [gap-6] §6 |

**Total** for 2 YAM + 2 leaders + 2 D405 + 1 D435i = $5,998 + $5,998 + $650 + $399 ≈ **$13.0k**, plus an unpriced GPU, tariffs and shipping [derived]. (The previous version listed one leader and ≈$10.0k; bimanual leader-follower teleop needs two.)

**Budget alternative.** An AgileX PiPER ($1,999) is a RoboDojo-RealEval embodiment with existing code: piper-astra-jev, Show-Harness interpreter, EmbodiedSWE `piper.py`. Mind the joint-3 zero and J6 limit quirks [note: piper-astra-jev].

**Robot-time budget (the binding constraint).** One evaluation campaign of 6 tasks × 50 trials × 3 arms = 900 trials × (Tier 1 mean wall time 251–435 s + ≈14 s reset) ≈ **66–112 robot-hours** [derived from gap-1], plus re-runs for thermal trips and an operator for every hour. Faster skills (2–4 calls per pick-place) shrink this; budget operator time explicitly.

**API budget for one evaluation campaign:** 900 trials × ~$1/trial ≈ $900 [derived from Tier 1's $0.90–1.14; gap-1 projects per-call cost within 1.3× of Tier 1 for this design's request shape]. The verifier adds ≈$0.017–0.035 per check [gap-5].

**YAM caveats** [gap-6; note: robocurve-gpt6-astra]: no published repeatability; 1–3 cm open-loop sag; encoder zeros off by up to 5.47° from the factory; thermal trips under holds (shoulder 9.5–13.7 N·m vs 9 N·m rated beyond ~0.3 m reach); Kd capped at 5 on the wire; `LINEAR_4310` gripper is force-limited position control; the Robocurve rigs ran auto-recovery off. The Pro tier ($3,499) claims "tighter build tolerances" with no number.

### 6.2 Phases and milestones

**Phase 0 (weeks 0–3): evaluation substrate first.**
- Build:
  - MuJoCo twin of the YAM cell;
  - Inspect Robots running with Opus 5.5 on the native Messages wire;
  - wire capture;
  - trial database with pre-registration templates.
- Implement the provider abstraction (§3.12) with Anthropic and OpenAI Responses adapters.
- **Milestone M0:**
  - **5-call Inspect Robots smoke test on Ilia's own Anthropic account** (preserved-thinking check, §5.4); fix with all-frames `image_horizon` or `drop_block`, and disclose the departure;
  - run the ≈$50 subset of `gaps/gap-1-bench.py` (Opus 5.5 low/medium/per-message-low/fast, Sonnet 5.5 `between_tools`, ER 2 pointing) to replace the remaining latency UNVERIFIEDs;
  - reproduce architecture A on the twin (Menagerie `i2rt_yam/yam.xml`, not the collision-only `inspect-robots-yam` model): Opus 5.5 vs Astra on sim bowl, 20 trials each, with token, $ and latency logs.

**Phase 1 (weeks 3–7): hardware, calibration and safety.**
- Two YAM arms, D405 ×2, D435i.
- **One-day bench protocol per arm** ([gap-6] §6): motor registers and loop rate; 50-return repeatability plus ISO 9283-style 5 poses × 30 cycles; static sag map (3 radii × 3 heights × 4 payloads) and `gravity_comp_factor`/`ee_mass` fit; 11-touch encoder-zero calibration; contact-detection threshold at 5/10/20 mm/s; achievable impedance stiffness; thermal time-to-65/70 °C and cool-down; **20 scripted wrist-servo insertion trials at 1 mm radial clearance** (Ø10 into Ø12 chamfered) — the bar every learned skill must beat.
- Preflight suite returning numbers: FK-vs-measured TCP error, command-rate check, depth plane RMS, extrinsic consistency, gripper empty-grasp, speed-scaling sanity (GRID's list).
- Approver chain, envelope, leases, heartbeat, e-stop and thermal park (load/torque/temperature-based). Unarmed dry-run on the identical code path.
- **Milestone M1:**
  - baseline arm on Ilia's rig: Inspect Robots + Opus 5.5 **and** Astra on R1 and R2, interleaved and blinded (≥20 per arm for development; ≥50 per arm before quoting a comparison);
  - per-arm motion tolerances and sag map recorded;
  - zero envelope violations;
  - scripted adversarial-command test: every out-of-envelope command must be blocked before motion, MHS-style.

**Phase 2 (weeks 7–12): grounded skills (architecture B).**
- Object table (Grounding DINO + SAM 3 + depth), `pick`/`place`/`move_tcp`/`verify`/`finish`, receipts, tier-1 + tier-2 verifier with verification view, append-only caching, server-side argument validation.
- Run the 300-target millimetre pointing eval of [gap-2] §5 on the cell and fix the pointer choice.
- **Milestone M2:**
  - R1 ≥ 19/20 with Opus 5.5 at ≤ 2–4 LLM calls per pick-place (development n), then ≥50 interleaved vs the baseline arm;
  - median trial wall-clock below Astra's 2.5 min;
  - cache-read share ≥ 85%;
  - **overclaim rate stated per predicate class and measured on ≥100 claimed successes**: < 5% on placement predicates (one Opus 5.5 check reached 0/92 false accepts there); on contact predicates, report the overclaim/false-reject trade-off instead of a single target, because vision-only gating gave 5.9% overclaim with one verifier and 0.8% only by rejecting 4 of 6 true successes [gap-5];
  - ablation 1 (FSM) and 4 (direct EEF) reported.

**Phase 3 (weeks 12–20): contact skills (servo first, then learned heads where they win).**
- Ship the analytic association + wrist-servo + compliance skills first (`center_grasp`, `descend_until_contact`, `place_release`, `insert`), then bimanual pose planning and reach-aware placement (they cover failures the five skills do not) [gap-4].
- Teleop 50–100 demos of the final 3 cm per contact skill that the servo baseline fails, starting from its hand-off pose (two YAM leaders).
- Opus-written scripted controllers in the twin, multiplied with EmbodiedSWE-Gen-style noise and replay gating.
- Train 20–100M chunked heads with condition noise matched to the measured pointer error, staleness augmentation, redundant-cue dropout and shift tests (no TT-RTC for a local head); then on-robot DAgger or RLPD correction (1–5 h robot time per skill).
- **Milestone M3 (the differentiating result):**
  - R4 insertion at ≥ 1 mm radial clearance: **learned/hybrid skill vs the scripted wrist-servo + compliance baseline vs a YAM-fine-tuned frozen VLA primitive**, ≥50 trials per arm, interleaved, blinded, with wall-clock and cost;
  - R2 puzzle vs the same on-rig baselines (the external 2/20 is context only);
  - ablation 2 and the condition tests (erasure, sensitivity, shift) reported.

**Phase 4 (weeks 20–28): breadth and flywheel.**
- Add the VLA tool (π0.5 via openpi or MolmoAct2, both fine-tuned on YAM data) and, if it wins its A/Bs, the ER 2 pointer/verifier.
- DAgger logging of corrections; optional Kev/Laya gate in shadow mode.
- Learned progress/success critic on sim labels (open problem: SFT progress 45.4% today).
- Distil the orchestrator for routine tasks into a 2–9B model trained on harness traces (Show-Harness / Guava / WAA evidence: 87.1% vs teacher 90.4%; 1.7 → 43.3% OOD) [note: frontier-models-as-direct-policies].
- **Milestone M4:**
  - R3 Tier-1-like tasks: completions above the **on-rig** Opus 5.5 / Astra baseline arm at ≥ 50 per arm per task (Tier 1's external 1/120 is context, not the comparator);
  - RoboLab-120 (both scorers) and/or RoboDojo submission with full disclosure;
  - a public trace release.

### 6.3 Build vs borrow

| Component | Decision | Source |
|---|---|---|
| Eval harness, embodiment plugin, wire capture | **Borrow** | `inspect-robots`, `inspect-robots-yam` (MIT) |
| Approver chain | Borrow + extend | `inspect-robots/src/inspect_robots/approver.py` |
| Envelope, floor ritual, Ruckig, settle, grasp semantics | Borrow ideas, re-implement | `metal-arm-harness/src/metal_arm_harness/{safety.py,control.py,executor.py,calibration.py}`. MIT only in `pyproject.toml`; no LICENSE file [note: metal-arm-harness] |
| YAM low-level: integral settle, joint-zero calibration, contact/settle semantics, speed cap | Borrow (Apache-2.0) | `agent-as-policy` `hardware-bridge/src/agp_yam_bridge/motion.py`, `hardware-bridge/config/{left,right}_arm.yaml`, `calib/fit_joint_offsets.py`; I2RT driver `i2rt-robotics/i2rt` (MIT; `motor_chain_robot.py` `command_joint_state` per-command gains) [gap-6] |
| Benchmark and verifier scripts from this research | Reuse | `gaps/gap-1-bench.py` (latency/cost benchmark, untested against APIs), `gaps/gap-5-verify.py` (verifier evaluation), `gaps/gap-3-sim/pilot.py` (conditioning toy study), `gaps/gap-6-sag.py` (sag/stiffness derivation) |
| Executor gate order, heartbeat, VLA chunk queue | Borrow (Apache-2.0) | `quackd/quackd/safety.py`, `quackd/adapters/lerobot/src/quackd_lerobot/policy/loop.py` |
| Grasp search, CAD fit, recorder | Borrow (Apache-2.0) | `piper-astra-jev/piper_llm/{skills.py,geometry.py,record.py}` |
| Constrained Cartesian planner, demo compiler | **Re-implement** (no licence) | `GPT-Policy-cheng/src/gpt_policy/motion/{planner.py,trajectory.py}`, `input/demonstration.py` [note: gpt-policy-in-context] |
| Video contact sheets, LLM client hygiene | Borrow (Apache-2.0) | `manda-robot-episode-labeler/src/rel/video/contact_sheet.py`, `src/rel/annotation/{llm.py,validate.py}`, `src/rel/prompts/{segment_v2.md,label_v2.md}` |
| Data engine | Borrow (Apache-2.0) | `EmbodiedSWE/data_engine/*`, `EmbodiedSWE/eval/tools/*` |
| VLA tool and planner patterns | Borrow (Apache-2.0) | `RLinf_RPent/robots/libero/tools.py`; openpi π0.5 |
| Discrete-unit interpreters (cheap baseline and distillation format) | Borrow (Apache-2.0) | `showlab_Show-Harness/interpreters/`, `prompts/controller.txt`, `gumi/` |
| Leases | Borrow pattern | `aditya-ramabadran_drivingbench_harness_v1/shared/contracts.py` |
| Orchestrator prompt and tool schema, receipts, verifier, contact heads | **Build** | This is the product |
| Licences to check | — | SAM 3 `license:other`; GR00T N1.7 weights under NVIDIA Open Model License; RoboDojo README claims non-commercial while LICENSE is MIT text; Zetta and GPT-Policy have no licence [note: piper-astra-jev; note: industry-competition; note: robodojo; note: github-implementations-sweep] |

---

## 7. Risks, open questions, and where differentiation can come from

### 7.1 Risks

1. **Model choice is unresolved.** Astra leads most head-to-heads against Opus 5 and earlier; on the only Opus 5.5 real-robot benchmark the two are indistinguishable (progress 36.0 vs 36.7%, completions 1/120 vs 5/120, p = 0.21, order-confounded). Tier 1 does not show that Opus 5.5 is worse.
   - *Mitigation:* the harness is provider-agnostic, the baseline arm runs both models interleaved on Ilia's rig, and the value proposition is a reliability layer plus contact skills, not "Claude drives".
2. **A learned contact head may not beat simpler baselines.** It may learn to ignore the LLM's conditioning (Fast Plans: erasure Δ 0.0 pp), break under realistic pointer noise if trained on exact labels (toy: 100% → 20% at 5 mm), lose to an analytic association + servo baseline (toy: 100% up to 10 mm noise) or to a frozen VLA primitive (VoLo real: No-VLA ≈ VLA), or fail to transfer from agent-generated sim data (EmbodiedSWE: 2/10 real) [gap-3].
   - *Mitigation:* servo-first build order; condition noise, redundant-cue dropout, erasure and shift tests; real teleop top-ups; on-robot correction; per-skill heads only where both baselines fail.
3. **Latency tails.** Measured Opus 5.5 per-call p99 21.1 s, max 60.3 s; Opus 5 max 67.8 s; 60–180 s tails at high reasoning; Fable 5 hit the 120 s client timeout on 5 of 2,188 attempts in the Robocurve run. Tier 1 itself had zero errors in 11,768 calls [gap-1; note: robocurve-gpt6-astra].
   - *Mitigation:* leases, load/thermal-based parks, deadlines, deterministic fallbacks, bounded output length.
4. **The verifier may not be able to verify.** Vision verifiers are near-perfect on placement but BA ≈ 0.71 on seating/stacking/capping/upright; a second vendor repeats the same errors; at 224 px most false claims are physically unobservable [gap-5].
   - *Mitigation:* verification views, probes and per-object sensor predicates; route by predicate class; state M2 per class.
5. **The cheap arm may be the limit.** YAM is a low-gain impedance arm with 1–3 cm open-loop sag, no published repeatability, Kd capped at 5 and thermal trips in ordinary holds [gap-6].
   - *Mitigation:* the one-day bench protocol before any precision claim; calibration + integral settle; wrist-camera relative alignment; fixtures within r ≈ 0.35 m; a duty-cycle rule. Feasibility is shown for ≥1 mm radial clearance with chamfers; unchamfered ≤0.25 mm radial insertion on a YAM is unproven.
6. **API churn.**
   - Preserved-thinking enforcement (400 on edited history for accounts created on or after 2026-08-31), forced-tool removal and computer-use tool changes all arrived within weeks. Inspect Robots' image eviction already conflicts with the first (§5.4).
   - Model retirements: ER 1.6 shut down at the end of August; Haiku 4.5 may retire from October 15.
   - Sol/Luna had an image-encoding bug fixed on 2026-09-25, which invalidates earlier vision evals [note: frontier-models-embodied-2026].
   - *Mitigation:* per-model regression gates; pinned IDs; raw wire logs.
7. **Safety and liability.** Hardware damage (RoboDojo), harmful-instruction compliance (RoboHarm), prompt injection.
   - The EU Machinery Regulation applies from 20 Jan 2027 (background, not re-verified) [note: industry-competition].
8. **Competitive squeeze** [note: industry-competition]:
   - Google sells the integrated stack (ER 2 + GR2 + On-Device 2), and Apptronik and Boston Dynamics outsource their robot brains to it.
   - NVIDIA is acquiring Hugging Face (LeRobot) for $12.93B.
   - Robot foundation models are absorbing in-context learning: GEN-1.5 59% one-shot at 100 Hz; Skild S1 66% vs 9% (both vendor-only: GEN-1.5 reports no trial counts; Skild's figure is on a filtered subset of its own pretraining pool).
   - The harness layer is crowded: dimOS 4.6k★, OM1, AWS Strands already pairing Claude with GR00T on SO-101.
9. **Evaluation self-deception.** Single runs, model-known grading, sim-paused results, recipe leakage and memory regimes. Most published "LLM drives robot" numbers have at least one of these (Robocurve, RoboDojo, GPT-Policy, so101-painting, DexAgent).

### 7.2 Where the sources disagree (decide by your own ablations)

| Question | One side | Other side |
|---|---|---|
| LLM supervising a VLA | Anthropic: every model worse than MolmoAct alone in-distribution | GPT-as-Policy RoboDojo: Hybrid 48% vs Direct 26% (π0.5-weak subset; π0.5 comparator not rerun; largest gains with ≤1.7% LLM-authored steps, so receding-horizon replanning is a rival explanation); RoboLab: Direct 98% > Hybrid 92% (retries, unpaired); RoboCasa n.s. |
| In-context demonstrations | RoboDojo: hurt (22.9 → 17.9 / 12.9%) | RoboICL / RoboDawn: +20 / +20 pts with structured context. Harness design drives the effect |
| Effort / reasoning | Robocurve Opus 5: milestone 46 → 76 with effort (n = 5) | Anthropic: "no major difference" for Claude on manipulation; newer models regressed with more reasoning on classic control |
| Harness vs model | Scaffold swings up to 48 pp (2605.23950) | CodeActionBench: within-Opus harness effect 4.0 pp, p = 0.74 |
| Opus 5.5 vs Astra | URAI native: Opus 5.5 32% vs Astra 16% (8/25 vs 4/25, p = 0.32; Claude at xhigh, Astra at medium) | Tier 1: completions 1 vs 5 /120 (p = 0.21; blocked, unblinded); false-success claims 10/119 vs 12/115 (p = 0.66). Neither side is significant; Astra leads most benches against Opus 5 and earlier |
| VLA vs primitives | VoLo sim: VLA adds a lot (17.8 → 41.8 with VLA) | VoLo real: No-VLA 19/42 ≥ full 18/42 (overlapping CIs); toy study: analytic servo and staged frozen primitive beat every learned head [gap-3] |
| Pointing: Claude vs specialists | PhysBrain Table 4: Opus 5 (low) −21.7 on RefSpatial, −18.5 on PixMo | Same table: Opus 5 leads Part-Affordance (78.1 vs 55.0/64.7), within ~3 pts on 5 sets; 336 px + normalized-coordinate protocol; no Opus 5.5/ER 2 data [gap-2] |
| Verifier de-correlation | FailBench: mean error-set IoU 0.28 across detectors | Tier 1/Astra report: Opus 5.5 vs Astra IoU 0.75, κ 0.91; changing the view de-correlates as much as changing the vendor [gap-5] |

### 7.3 Open research questions this harness can answer

1. How much precision can a 20–100M head add **over a scripted wrist-servo + compliance baseline and over a YAM-fine-tuned frozen VLA primitive**, under sparse, noisy, stale LLM targets? Nobody has measured this; the toy study says the analytic baseline is hard to beat when last-centimetre occlusion, compliance and jamming are absent [gap-3].
2. Which conditioning resists shortcutting on a real arm? Toy evidence: a tracked selection mask (which) + own wrist perception (where) beats metric keypoints; condition noise and redundant-cue dropout are needed; counterfactual relabelling is untested [gap-3]. The full requested study (Menagerie YAM or RoboTwin 2.0, ACT-80M heads, π0.5 baseline) needs ≈150 GPU-hours [gap-3 §5].
3. What is the right **call policy**: when to invoke the LLM vs FSM vs decision model, and how much it saves at equal success? The Jev gate evidence comes from 2 development tasks only.
4. Can a **learned progress/completion critic** trained on free sim labels beat zero-shot VLMs? Current state: 45.4% SFT on T5; FailBench below 0.60 on contact tasks; Opus 5.5 / Astra verifiers BA ≈ 0.71–0.74 on real contact tasks [gap-5].
4b. What is the millimetre pointing error of Opus 5.5, ER 2 and local 8B pointers on robot camera frames with touch-probe ground truth? No model has ever been measured this way [gap-2].
5. Does plan-while-moving (speculation, leases) keep success while cutting wall-clock? No wall-clock-timed benchmark exists for manipulation agents; DrivingBench is the only latency-in-the-loop benchmark [note: benchmarks-and-sim-datagen].
6. Can Opus 5.5's preserved-thinking constraints coexist with bounded context over 30+ minute episodes (compaction cadence vs reasoning continuity)?
7. Does distilling the orchestrator into a 2–9B model (Show-Harness, Guava, WAA) hold on real hardware? Evidence so far is mostly in sim.

### 7.4 Where real differentiation can come from

1. **The open gap, stated precisely.** Adjacent cells are occupied — Pigey (Opus 4.7 + frozen π0.5 + TAMP, 97.3% over 150 real trials), Harness VLA/RPent (Claude Agent SDK + frozen π0.5 `pi0_pick`), VoLo (Opus 4.6 + π0.5, real Franka), AWS Strands (Claude Sonnet 4.5 + GR00T on SO-101), RoboDual and RL Token (small heads under a big model). What is unpublished: a **task-trained, target-conditioned small head under a frontier API**, shown to follow LLM targets (erasure and shift tests) and to beat **both** a frozen VLA primitive and a scripted wrist-servo + compliance baseline on real ≤2 mm contact tasks, with ≥100 blinded, interleaved trials [gap-3; note: vla-and-action-heads].
   - The first credible public result: "≥1 mm-clearance insertion: learned/hybrid skill X/50 vs scripted servo Y/50 vs frozen VLA Z/50, at equal or lower wall-clock and cost, on a $3k arm". If the servo baseline wins, publishing that is also a contribution.
2. **A reliability layer as the product.** Independent fail-closed verification, an MHS-compatible driver envelope, receipts, leases and audit logs. The industry note's conclusion is that "the defensible position is not 'Claude drives the robot'" but an embodiment-agnostic reliability layer, a data engine and model-agnostic routing [note: industry-competition].
3. **The data engine.** Every harness run produces LeRobot v3.1-formatted (observation, tool call, receipt) traces and correction pairs. These train the heads and, later, a distilled orchestrator. API spend becomes reusable data, and the gap between model generations becomes a moat rather than a threat [note: benchmarks-and-sim-datagen].
4. **Honest, wall-clock evaluation.** Publish paired, blinded, live-physics numbers with cost and latency against Robocurve and RoboDojo baselines. The field's evidence quality is low enough that rigour itself is a differentiator; Manda and Robocurve built companies on evaluation [note: manda-robotics].
5. **Long-tail embodiments.** YAM, PiPER, SO-101 and Metal-class arms will not get π0.7, GR00T or Gemini fine-tunes soon. A harness that onboards them with GRID-style preflight plus a few hours of skill data serves the market the incumbents ignore [note: industry-competition; note: general-robotics-auto-engineering].

---

## Appendix A. Items that remain UNVERIFIED and matter for this design

**Resolved since revision 1** (moved out of this list):
- *Robocurve frame resolution*: 224×224. The Tier 1 page states it; 2,196 decoded Astra/Fable frames are all 224×224 (`cv2.resize` squash) [gap-4; note: robocurve-gpt6-astra].
- *Strict-mode schema features*: numeric constraints (`minimum`/`maximum`/`multipleOf`), string lengths and complex array constraints are **not supported**; SDK helpers strip them and validate client-side (bundled Claude API reference). Server-side validation is required (R16).
- *Compaction headers*: two different features — `compact-2026-01-12` (threshold compaction) and `compact-2026-09-04` (on-demand `compaction` parameter, in Opus 5.5's feature set; not on Bedrock).
- *`clear_at:"next_user_message"` on Opus 5.5*: available as a beta (`mid-conversation-system-clear-at-2026-08-21`) on the same models as mid-conversation system messages, which include Opus 5.5 (not Foundry).
- *Opus 5.5 per-call latency in a robot loop*: measured at medium effort (p50 5.28 s, p90 11.72 s, p99 21.1 s) [gap-1].
- *Image guard field name*: `transformations.oversized_image: "error"` on each image block [gap-2].

**Still UNVERIFIED:**
- **Opus 5.5 latency** at low/high effort, with per-message effort or fast mode; any TTFT under the robot request shape; any 1920×480 tile; Sonnet 5.5 `between_tools`; Haiku 4.5 in a robot loop; Astra `ultrafast`; ER 2 pointing latency [gap-1]. Script ready: `gaps/gap-1-bench.py` (untested against APIs).
- **Whether Inspect Robots' eviction actually 400s on a new account**: inferred from the docs and code, not run [§5.4].
- **Pointing**: no Opus 5.5, ER 2 or ER 1.6 pointing number exists; no millimetre error against real-robot ground truth for any model; the exact Claude configuration behind PhysBrain Table 4 is unreleased; GroundingPI-4B weights unreleased [gap-2].
- **ER 2 video success detection (82.4) and Robometer-4B (F1 0.81)** as verifiers on real YAM episodes: not measured (no Gemini key; no GPU) [gap-5].
- **Head conditioning on a real arm**: only a CPU toy study exists [gap-3].
- **YAM repeatability and backlash**: not published by I2RT; to be measured with the bench protocol [gap-6].
- **Whether Anthropic's guidance says instruction placement near the end "moves behaviour more" than system-prompt wording**: not found in the bundled reference.
- **GR00T N2 / "Isaac GR00T 2"**: NVIDIA newsroom unreachable [note: vla-and-action-heads].
- **Anthropic Embody code** (`safety-research/embody`): still unreleased.
- **GPU workstation price and bench tooling prices**: not checked.
- **Camera prices**: from a search summary of the official RealSense store pages on 2026-10-02. The D435i was out of stock, and a tariff surcharge applies.
