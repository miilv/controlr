# REPORT: Frontier-LLM robot control — state of the field, evidence, competition and direction

*For Ilia Mikhalchuk. Final edit 2026-10-02. Companion to `HARNESS_DESIGN.md`, which holds the concrete architecture; both documents use the same numbers. Scope: a new robot-control harness with a frontier LLM API (Claude Opus 5.5 by default) as the backbone, possibly with a light learned action head.*

**Conventions.**
- `[note: slug]` = `research/sources/<slug>.md` or `research/landscape/<slug>.md`.
- `[gap-N]` = `research/gaps/gap-N.md`, the new primary measurements made for this report.
- **[derived]** = arithmetic done here from cited numbers. **UNVERIFIED** = not confirmed at a primary source.
- Trial counts are given wherever known; treat n ≤ 10 as anecdotal.
- "Opus 5" and "Opus 5.5" are different models and are never merged. Most published Claude robotics numbers are Opus 5 or older.

---

## 0. How this was produced / file map

**Process.**
1. **Fourteen deep-dives** on the user-supplied sources (`sources/`). Each involved code reading and artifact checks, then a separate fact-check pass that corrected the note body inline.
2. **Eight landscape sweeps** (`landscape/`), each with its own fact-check pass. Topics: models; VLAs and action heads; the planner/codegen lineage; direct-policy studies; benchmarks and data generation; industry; engineering challenges; GitHub implementations.
3. **A design synthesis** (`HARNESS_DESIGN.md`) written from all 22 notes.
4. **A senior critique** (`CRITIQUE.md`): 24 errors and 6 research gaps.
5. **Six gap studies** (`gaps/`) that produced new primary data:
   - all 360 public Robocurve Tier 1 trials mined and hand-coded;
   - 11,768 logged API calls analysed;
   - 1,782 verifier calls run on 335 labelled episodes;
   - a pointing-accuracy audit;
   - a YAM characterization from driver code and rig logs;
   - a CPU toy study of action-head conditioning.
6. **This final edit.** Every critique item was verified (some rejected; see §10), gap findings were integrated into both documents, and numbers were reconciled.

**What could not be done.** The environment had no OpenAI or Google API keys and no GPU, the root disk was at ~100%, and no spending was authorized on Ilia's Anthropic account. Four studies were therefore *designed and scripted* but not run (details in the gap files):
- the 14-configuration latency benchmark;
- the 300-target millimetre pointing eval;
- the ER 2 and Robometer verifier tests;
- the full head-conditioning study.

**File map.**

| Path | What it is |
|---|---|
| `REPORT.md` | This report: landscape, challenges, competition, promising vs not, corrections log |
| `HARNESS_DESIGN.md` | Evidence-grounded architecture, tool schemas, evaluation plan, MVP roadmap (revision 2) |
| `CRITIQUE.md` | Senior review: 24 errors, 6 gaps. The verdict on each error is in §10 here |
| `README.md` | One-page index of the folder, read order, cloned repos grouped by relevance |
| `sources/manda-robotics.md` | Manda Robotics: evaluation company; VLM-pointing connector; 6,000-episode VLA study |
| `sources/llm-robotics-playground.md` | dimentary/llm-robotics-playground: MuJoCo controllers authored by Astra in Codex |
| `sources/piper-astra-jev.md` | RobotKitAI piper-astra-jev: Astra vs Jev as skill selector on a real PiPER |
| `sources/embodiedswe.md` | EmbodiedSWE: coding agents as offline robot engineers and data teachers |
| `sources/gpt-as-policy.md` | GPT-as-Policy (Galbot): Astra direct vs π0.5 hybrid on RoboDojo |
| `sources/robocurve-gpt6-astra.md` | Robocurve Astra/Fable report on real YAM, the Inspect Robots harness, and the Tier 1 addendum |
| `sources/robodojo.md` | RoboDojo benchmark and protocol; RoboProbe LLM harness |
| `sources/dexagent.md` | DexAgent: agentic human-video → sim → π0.5 pipeline |
| `sources/metal-arm-harness.md` | MakerMods metal-arm-harness: coding-agent CLI as policy; envelope engineering |
| `sources/so101-painting.md` | HF Bucket so101-painting: Astra as offline engineer for SO-101 painting in MuJoCo |
| `sources/quackd.md` | rokbenko/quackd: executor contract, VLA segments, decision models |
| `sources/general-robotics-auto-engineering.md` | General Robotics GRID "Auto-Engineering" |
| `sources/gpt-policy-in-context.md` | GPT-Policy (cheng-haha): in-context robot learning with VLM agents on real arms |
| `sources/innate-os-pr817.md` | innate-os PR #817: one-demo imitation with Astra on the MARS robot |
| `landscape/frontier-models-embodied-2026.md` | Model landscape: Claude, GPT-6, Gemini ER 2, decision models, open embodied VLMs |
| `landscape/vla-and-action-heads.md` | VLAs, dual systems, light heads, real-time chunking |
| `landscape/llm-planner-codegen-lineage.md` | LLM-as-planner and code-as-policy lineage, 2022–2026 |
| `landscape/frontier-models-as-direct-policies.md` | Frontier models emitting motion directly |
| `landscape/benchmarks-and-sim-datagen.md` | Benchmarks, real-robot statistics, LLM-driven sim and data generation |
| `landscape/industry-competition.md` | Companies, funding, harness-layer competitors, hardware prices |
| `landscape/engineering-challenges.md` | Quantitative engineering constraints: latency, grounding, caching, safety, variance |
| `landscape/github-implementations-sweep.md` | About 45 real open-source LLM-robot codebases, ranked |
| `gaps/gap-1.md`, `gap-1-latency-table.csv`, `gap-1-tier1-per-call.csv.gz`, `gap-1-bench.py` | Per-call latency and cost from 11,768 Tier 1 calls; the unexecuted 14-configuration benchmark script |
| `gaps/gap-2.md` | Pointing and grounding accuracy: published numbers, protocol audit, millimetre eval protocol |
| `gaps/gap-3.md` (+ `gaps/gap-3-sim/`) | Head conditioning: literature table and CPU toy study (400 episodes per cell) |
| `gaps/gap-4.md`, `gap-4-trials.csv` | Forensic coding of all 360 Tier 1 trials |
| `gaps/gap-5.md`, `gap-5-verifier-results.csv`, `gap-5-verify.py` | Independent-verifier accuracy on 335 labelled episodes |
| `gaps/gap-6.md`, `gap-6-sag.py` | YAM ≤2 mm feasibility, characterization, one-day bench protocol |
| `repos/` | About 85 cloned repositories (grouped in README.md) |
| `tmp_*` | (deleted 2026-10-02 to free disk — scratch paper-text extractions; re-downloadable from the cited arXiv/URLs) |

---

## 1. Bottom line

1. **The clock, not intelligence, is the binding constraint.**
   - On real YAM arms, Opus 5.5 at medium effort takes p50 5.28 s, p90 11.72 s and p99 21.1 s per call [gap-1].
   - About 84% of a trial is spent waiting on the model, and the realized decision rate is 0.127 Hz. Contact control needs 30–200 Hz.
   - So the LLM must be event-driven and never sit inside a control loop.
   - Latency is mostly decode time, ≈1.8 s + output tokens ÷ 87 tok/s. Output length is the lever.
2. **The field has converged on one shape.** The strongest 2026 systems put a frontier LLM over a *typed, verifiable tool surface*. Contact-rich motion is delegated to analytic primitives (IK, TAMP, servoing) or to a *frozen* learned policy [note: llm-planner-codegen-lineage]:
   - Pigey (Opus 4.7 + π0.5 + TAMP): 97.3% over 150 real trials;
   - Harness VLA / RPent (Claude Agent SDK + frozen π0.5): LIBERO-Pro 50.0% → 82.4% with Opus 4.7;
   - VoLo and AGP follow the same pattern.

   Direct low-level control by frontier models fails: LIBERO 0–5.5% for every model Anthropic tested.
3. **Interface beats model within the frontier tier** [note: frontier-models-as-direct-policies; note: manda-robotics; note: github-implementations-sweep].
   - URAI: every agent roughly tripled its success when choosing between pre-written multi-phase tools (18% → 53% overall). A Fable 5.1 programming agent wrote and froze those tools.
   - Manda: the same Gemini model scored 0/4 emitting Cartesian deltas vs 6/6 emitting a point plus phase.
   - Show-Harness: 20/20 vs 1/20 from naming and axis conventions alone.
4. **The evidence does not decide between Opus 5.5 and GPT-6 Astra.**
   - The only real-robot Opus 5.5 benchmark is Robocurve Tier 1 (6 tasks × 20 trials). Progress was 36.0% vs 36.7% and completions 1/120 vs 5/120 (Fisher p = 0.21). The design ran models in blocks, and graders knew the model [gap-4].
   - Astra leads most other head-to-heads, but those were against Opus 5 or older.
   - Opus 5.5 is cheaper ($4/$20, cache reads $0.20) and had the lowest median latency of the three Tier 1 models.
   - Stay provider-agnostic.
5. **Claude's "known weaknesses" were overstated.**
   - *Pointing.* Only Opus 5 at low effort has numbers, on a protocol hostile to Claude. It trails on RefSpatial (−21.7) and PixMo (−18.5), is within ~3 points on five sets, and *leads* on Part-Affordance (78.1 vs Astra 55.0). No Opus 5.5 or ER 2 pointing number exists [gap-2].
   - *Overclaiming.* Opus 5.5 made 10/119 false success claims vs Astra 12/115 (p = 0.66), and most were physically unobservable at 224 px [gap-4].
6. **Millimetres come from geometry, not the LLM.**
   - No model has a measured millimetre pointing error against real-robot ground truth.
   - The best tabletop data is GPT-5.4 at 16.8 px and Sonnet 4.6 at 29.3 px median at 1920×1080, about 6 mm at 0.5 m for the better model [gap-2].
   - Every cheap-arm ≤3 mm success closes the last millimetres with relative wrist-camera alignment, slow descent and compliance. AGP's LLM-written scripts did 8/10 four-pair insertions at 0.73–2.83 mm radial clearance on a YAM, with no learning [gap-6].
7. **Verification is the weakest link, and "another model" does not fix it** [gap-5].
   - An Opus 5.5 verifier reaches balanced accuracy 0.871 overall, but only 0.707 on contact tasks (25% false accepts).
   - Astra as verifier shares 75% of its errors (κ 0.91).
   - Different *evidence* de-correlates errors: a verification view, physical probes, per-object gripper predicates. A different vendor does not.
8. **A light learned head has not yet been shown to beat simpler options.**
   - No paper tests a frontier-API-conditioned 20–100M head against a frozen VLA primitive on ≤2 mm insertion.
   - On a real robot, VoLo's No-VLA system scored 19/42 vs 18/42 for the full system.
   - In the toy study, the analytic servo scored 100% up to 10 mm of pointer noise, while learned heads trained on exact keypoints fell to 59% at 2 mm.
   - So build the servo baseline first, and train heads only where it fails [gap-3].
9. **Competition is structural** [note: industry-competition].
   - Google sells the integrated System-2-over-API stack. Gemini Robotics-ER 2 costs $1/$5 per MTok with a free tier and orchestrates Google's own VLAs. Apptronik and Boston Dynamics outsource their robot brains to it.
   - NVIDIA is buying Hugging Face, home of LeRobot, for $12.93B.
   - Robot foundation models are absorbing in-context learning: GEN-1.5, Skild S1 and π0.7 (all vendor-reported).
   - The open harness layer is crowded: dimOS, OM1, AWS Strands, RPent.

   The defensible position is a reliability layer, a data engine and model-agnostic routing for long-tail embodiments, not "Claude drives the robot".
10. **Evaluation quality in the field is low enough that rigour is itself a differentiator** [note: engineering-challenges; note: manda-robotics].
    - Typical published numbers are single runs, model-known grading, paused-physics simulation, recipe leakage, or n ≤ 20.
    - Separating 50% from 70% needs ~93 trials per arm.
    - Re-running π0.5 on identical seeds reproduced only 64% of its successes.

---

## 2. The 14 user sources

| Source (note slug) | What it is | LLM role | Headline result (n) | Verdict |
|---|---|---|---|---|
| Manda Robotics (`manda-robotics`) | ~2-person robot-evaluation company; RoboLab-Verified fork; Gemini video labeler; Real2Sim study | Gemini as pointer (`vlm_pinpoint`, private) and annotator; agents build sim assets | Same Gemini 0/4 with Cartesian deltas vs 6/6 with point + phase → geometric controller (4–6 trials). 6,000-episode VLA study: π0.5 27.5%. An identical-seed rerun reproduced 64% of successes | Best "how to evaluate" reference, and the clearest case for "the LLM picks where, a controller does how". Its control code is private |
| llm-robotics-playground (`llm-robotics-playground`) | MuJoCo demos: writing, drawing, headphone untangling, spider | GPT-6 Astra in Codex as **offline code author**. Runtime controllers read privileged state and make no model calls | Single tuned episodes; the headphone run is stitched from 19 runs; model settings unknown | Not LLM control. Borrow the "trusted firmware / untrusted pilot" API, the dwell predicates and the integrity checks |
| piper-astra-jev (`piper-astra-jev`) | 8 single runs on a real AgileX PiPER | Skill **selector** from a menu (Astra at high effort vs the Jev decision model); never emits poses | Cube-into-tray: Astra 1 min 11 s vs Jev ~27 s. Jev + SAM 3 run: 9 decisions, 16.6 s loop. 0.6 mm relative alignment for a peg (n = 1) | A 20-line FSM reproduces the logs, so the decider is redundant. The real value is the geometry toolkit: mask+depth grasp search, CAD-silhouette fit, relative servoing |
| EmbodiedSWE (`embodiedswe`) | Benchmark of coding agents writing `solve(env)` in Isaac Lab, plus a data engine (arXiv 2609.27308) | Offline engineer and teacher (Claude Code / Codex) | Success over 28 tasks, 1 run each: Astra 82%, Fable 5.1 61%, Opus 5 50%. GPT-5.6 Sol/Terra reward-hacked 39–43% of runs. π0.5 fine-tuned on 500 agent demos: lamp task 2/10 real vs 0/10 | Strongest evidence for the LLM as offline engineer. Borrow the clean-label noise channel, replay gate and anti-hack stack. Real-robot evidence is thin |
| GPT-as-Policy (`gpt-as-policy`) | Galbot simulation study on a bimanual ARX X5 in RoboDojo (arXiv 2609.38537) | Astra at xhigh, via Codex app-server, either as a direct EEF policy or gating/editing π0.5 chunks | Hybrid 24/50 vs Direct 13/50 (McNemar p ≈ 0.027). RoboLab: Direct 49/50 vs Hybrid 46/50. 18–106 s per decision; ≈$17–29 per episode at list price | Good proposal-review contract. But the π0.5 comparator was not rerun, the biggest gains came with ≤1.7% LLM-authored steps, and physics was paused |
| Robocurve Astra report (`robocurve-gpt6-astra`) | Real bimanual YAM arms; Inspect Robots harness (MIT); full wire transcripts | Direct EEF `move_to` per turn, 224 px frames | Bowl: Astra 19/20 vs Fable 5.1 8/20 (different rigs, compat shim, not interleaved). Puzzle: 2/20 each (Fable 1/20 by live verdict). Sibling Tier 1 report: Opus 5.5 36.0% vs Astra 36.7% progress, 1/120 vs 5/120 completions | The best public real-robot data, and the evaluation harness to borrow. Its comparisons are confounded |
| RoboDojo (`robodojo`) | HKU benchmark: 42 sim tasks, 18 real; RoboProbe LLM harness | Astra as direct EEF policy (L3) | Astra #7 of 51 (22.48% SR) on 1 seed, with recipes and 2 per-task override cells worth ~4.3 Score points. Real testing stopped after hardware damage (1/33). One-shot demos hurt (22.9% → 17.9% / 12.9%) | The most careful public L3 baseline harness, but the leaderboard number is not like-for-like with VLAs. Precision tasks: 4% SR |
| DexAgent (`dexagent`) | Stanford/Columbia agentic pipeline: human video → sim → π0.5 (arXiv 2609.35318) | Offline tool-making agent with a self-evolving library of skills and verifiers | π0.5 trained on 500 agent-generated episodes: 63.6% (70/110) vs 18.2% best baseline. Unharnessed Astra baseline: 16.4% | Promising data-engine pattern. No code, baseline rows copied from other papers, no self-ablations |
| metal-arm-harness (`metal-arm-harness`) | MakerMods Metal arm safety and control layer | A coding agent (Claude Code / Codex) drives a CLI. A raw-joint Opus 5 Messages loop was **removed after ~8 h** | Anecdotal bench: 3 blocks into cups, with retries and human corrections. Ruckig cut elbow 2–6 Hz tracking error from 0.712° to 0.040° | Borrow the envelope engineering: measured floor, per-waypoint checks, excursion caps, sag compensation, LLM-readable rejections. No perception, no evaluation |
| so101-painting (`so101-painting`) | HF Bucket: SO-101 brush painting in MuJoCo | Astra as offline engineer and parameter tuner between runs | Score 83.37 → 96.26 over 4 runs, on a metric finalized after 2 of them; deterministic sim, n = 1 | A clean, inspectable "frontier model as outer-loop engineer" example. Not evidence of control |
| quackd (`quackd`) | Large open harness: 7 bodies, 11 vendors, MCP, VLA segments, decision models | Skill selector, one verb per turn; optional decision-model "stepper" | Real SO-101: gestures only. On 2026-09-23, 19/26 runs never moved the arm (five harness/prompt/gate faults). Issue #25: one decision-time sentence moved correct verdicts 5/30 → 27/30 (Qwen3-32B, temperature 0). No Claude run on record | The best executor-contract and VLA-segment code (Apache-2.0). Evidence is thin |
| General Robotics GRID (`general-robotics-auto-engineering`) | Closed enterprise platform; "Auto-Engineering" announcement | LLM as robotics **engineer**: ingest, sim, method selection, preflight, debugging. Model undisclosed (GPT-family hints) | One demo. Sysid error 143 mm / 6.4° → 5.7 mm / 0.68°. Found a command-pacing bug (~500 Hz instead of 30 Hz; verified 29.7 Hz after fix). Depth plane error 28 → 1.2 mm. ~4 h to first skill. No success rates | The right decomposition for the engineer half (preflight, evidence log, memory). Weak as evidence |
| GPT-Policy (`gpt-policy-in-context`) | Real-robot in-context learning study on ARX X5 / YAM (arXiv 2609.19138) | Astra via Codex app-server, one JSON tool call per decision (absolute TCP poses) | n = 3 per condition. Human video: towel 0/3 → 2/3. Video + action rows: uncapping 0/3 → 3/3, plug 0/3 → 2/3. 12–20 s per decision, 5–25 min per task. A null result was moved out of the main table | Clean context taxonomy and demo compiler. Slow, tiny n, and **no licence** (ideas only) |
| innate-os PR #817 (`innate-os-pr817`) | Open, unmerged PR: one-demo imitation on the MARS robot | Astra: survey → phase map → one bounded, host-validated `act` per observation | n = 1 public success: 45 steps, `act` median ≈4.3 s. The authors' successor PR #843 dropped demo-conditioning (sim: Gemini 3.6 Flash 4/5, Astra 1/2, Sonnet 5 1/8) | Good host-side guards: stale-observation discard, grip latch, demo-derived priors. Superseded by its own authors |

**Pattern across the 14.**
- Seven use the LLM *offline*, as engineer, teacher or author: EmbodiedSWE, DexAgent, so101-painting, llm-robotics-playground, GRID, and partly metal-arm and Manda's Real2Sim.
- Seven put it *in the loop*. In every in-loop case:
  - the robot spends 60–90% of wall time waiting;
  - success collapses on precision and contact;
  - the published evidence is n ≤ 20 or unblinded.
- The two sources with the best data, Robocurve and RoboDojo, are evaluation organizations, not controller builders.

---

## 3. Landscape

### 3.1 Frontier models as robot brains (as of 2026-10-01)

| Model | Price per MTok (in / out / cache read) | Robot-relevant API facts | Robot evidence |
|---|---|---|---|
| Claude Opus 5.5 (`claude-opus-5-5`, 2026-09-22) | $4 / $20 / $0.20; fast mode $8/$40 | Thinking cannot be disabled; default effort `medium`. Forced `tool_choice` returns 400. Thinking blocks are bound to model and conversation (new accounts: 400 on edited history). Per-message effort beta; mid-conversation system messages; `clear_at` beta. Images up to 2,576 px / 4,784 tokens | Tier 1 real YAM: 36.0% progress, 1/120, $0.90, p50 5.28 s [gap-1; gap-4]. URAI 32% → 60%. Verifier BA 0.871 [gap-5] |
| Claude Opus 5 (legacy) | $5 / $25 / $0.50 | Adaptive thinking by default | LIBERO-Agent 19.3 vs Astra 45.0. CodeActionBench 49.3%. TAMP coding 82%. AGP two-pair assembly 5/5. ER 2 chart: video success 81.0, progress 37.1. PhysBrain pointing at low effort |
| Claude Fable 5.1 | $10 / $50 / $0.25 | Default effort `high`; same preserved-thinking rules | Robocurve bowl 8/20, puzzle 2/20 (via compat shim). EmbodiedSWE 61%. URAI 16 → 52% |
| Claude Sonnet 5.5 (2026-09-28) | $2 / $10 | `thinking:{type:"between_tools"}` skips up-front thinking (the lowest setting); default effort `high` | None in a robot loop |
| GPT-6 Astra (`gpt-6-astra`, API 2026-09-03) | $10 / $50 / $1; `ultrafast` tier at 6× price | Tool calling requires Responses; effort `low…max`, `none` rejected; no Realtime | Robocurve bowl 19/20. RoboDojo 22.48%. LIBERO-Agent 45.0. CodeActionBench 73.3%. AGP four-pair YAM assembly 8/10. EmbodiedSWE 82%. RoboHarm: completed 60/100 harmful requests |
| Gemini Robotics-ER 2 (2026-07-30) | $1 / $5 through 2026, then $2/$10; free tier | Native `[y,x]` 0–1000 points, boxes, trajectories, video success and progress; Live streaming twin; `thinking_level` | Vendor chart: video success 82.4, progress 57.4, ERQA 78.5. Clapboardbench joint control 0/5. No published pointing number |
| Decision models (Jev 1.13, Kev, Laya) | Jev $0.042/MTok input; Kev and Laya self-hosted (Apache-2.0) | Text-only typed `choice/noul/score` questions → probabilities over `/v1/systemone` | Jev 70–500 ms (vendor), 236–276 ms p50 (independent). Kev-4B 18.1 ms server-side. RoboICL's Jev gate cut Astra calls 33–48% on 2 dev tasks. Jev "struggles with numeric precision" |

Sources: [note: frontier-models-embodied-2026; note: industry-competition; gap-1; gap-4; gap-5].

**Head-to-heads.**
- Astra leads clearly in direct and code-as-policy control:
  - LIBERO-Agent 45.0 vs Opus 5 19.3;
  - CodeActionBench 73.3% vs 49.3%;
  - Robocurve bowl 19/20 vs Fable 8/20;
  - DrivingBench: only Astra finished.
- Claude is closest on code (TAMP 82% vs 95%), video success detection and safety-instruction following. All of these are Opus 5 numbers, the last two from a vendor chart.
- Under URAI's tool interface, Opus 5.5 had the highest point estimate (60%) but no significant lead (p = 0.78).
- Rankings flip with the harness, so these tables cannot pick the model for a new interface [note: frontier-models-as-direct-policies].

**Open embodied VLMs** usable as local perception and pointing services [gap-2]:
- PhysBrain 1.5-8B (best open model overall in PhysBrain's table, 72.5);
- Embodied-R1.5-8B (Apache-2.0);
- MolmoPoint-8B (Apache-2.0);
- RoboBrain 2.5;
- Qwen3-VL.

GroundingPI-4B claims 73.68 vs Astra 71.54 over 34 grounding benchmarks but has **no released weights**.

### 3.2 What the LLM emits: the architecture taxonomy

| Architecture | LLM emits | Best evidence | Ceiling and main risk |
|---|---|---|---|
| A. Direct Cartesian/joint tool calls with visual feedback | EEF targets or deltas per turn | Robocurve bowl 19/20 (Astra); RoboDojo 22.48% SR | Centimetre-level: puzzle 2/20, Precision 4%, LIBERO direct 0–5.5%; latency. Use only as a baseline |
| B. LLM + open-vocabulary perception + IK/planning (skill selection with grounded arguments) | Object ids, skills, relations | Pigey 97.3% (real, 150 trials); VoLo No-VLA 45.2% real; AGP scripted YAM insertion 8/10 | Millimetres with good geometry, but per-object engineering |
| C. LLM points, traces or keypoints → light head | Points, masks, traces, phase | Manda 6/6 (small n); HAMSTER +20 pp; PEEK 2–41×; GAE (sim) | Pointer accuracy; heads ignore conditioning; may lose to an analytic servo [gap-3] |
| D. LLM as System 2 over a VLA | Language subtasks; accept/edit VLA proposals | Harness VLA LIBERO-Pro 82.4% (Opus 4.7) / 92.6% (Astra); VoLo sim 12.6 → 41.8% | Supervision can hurt in-distribution (Anthropic); VLA steerability |
| E. In-context demonstration imitation | Bounded steps conditioned on a demo | RoboICL 14.5 → 78.9 progress (n = 15); GPT-Policy video + action 0/3 → 3/3 | Naive demos hurt on RoboDojo (22.9 → 12.9%); slow; robot FMs now do one-shot natively |
| F. LLM writes skills, sims and data offline (engineer and teacher) | Code, assets, verifiers, demos | DexAgent 63.6% real; EmbodiedSWE-Gen 0.233 vs 0.066 held-out; ENPIRE 99% pin-box | Sim shortcuts, reward hacking (Sol/Terra 39–43%), sim-to-real |

**Recommended combination** (HARNESS_DESIGN §3):
- **B** as the backbone;
- servo-first contact skills (**C**);
- a VLA as an optional, fine-tuned tool (**D**);
- demonstrations as task specification (**E**);
- **F** as the offline data engine.

### 3.3 The planner / code-as-policy lineage and the 2026 convergence

**Era 1 (2022–23).** SayCan, Inner Monologue, Code-as-Policies (code 98% vs natural language 35% on spatial reasoning), VoxPoser (85 prompt examples; fails at 3).

**Era 2 (2024–mid-2025): visual prompting and constraints.**
- CoPa: constraints + solver 63% vs direct numeric poses 37%.
- MOKA: 74% → 94% with two in-context examples.
- ReKep: the largest error source is the point tracker, not the optimizer.

**Era 3 (late 2025–2026): agentic coding harnesses and frontier models at runtime.**
- **CaP-X / CaP-Bench.**
  - Single-shot low-level code is far below humans. Tier S4: Gemini 3 Pro 32.3%, Opus 4.5 23.8%, humans 88.5%.
  - Multi-turn feedback, text visual-differencing, skill libraries and ensembles close most of the gap. Raw RGB in the coder's context *hurts*.
  - CaP-RL on a 7B coder transfers sim → real: cube lift 24 → 84%.
- **Pigey.**
  - Raw Messages API with Opus 4.7. Tools: `Perceive / Pick / DropAbove / VLARollout / Release / Done`.
  - Labels must come from Gemini-ER detections; failures are typed; it escalates between TAMP and π0.5.
  - 97.3% vs π0.5 16.7% and TiPToP 48.7% on 30 real DROID tasks × 5.
- **Harness VLA / RPent** (RLinf, Apache-2.0).
  - A frozen π0.5 exposed as `vla_act(prompt, max_chunks, stop)`, plus 6 analytic primitives and file memory.
  - LIBERO-Pro 50.0 → 82.4% with Claude Code / Opus 4.7, and 92.6% with Astra.
  - The gains come from staging *where* the VLA starts.
- **AGP (Agent as Policy).**
  - Codex or Claude Code is the runtime policy, over a CLI bridge on a YAM. Controller caps: ≤0.03 m/s, workspace clamp, ≤0.25 m per move.
  - Four-pair assembly: 8/10 (Astra, 37 min, $16.6). Two-pair assembly: Opus 5 5/5 (22.2 min, $9.75).
  - On real hardware, GaP scored 0/25 and ASPIRE 3/25 vs AGP 23/25. Pipelines compiled in sim and deployed blind fail.
- **SafeHarness.** Prompt-stated constraints are not enforced. Collision avoidance went from 50.0% (model only) to 87.5% with harness route verification.

**The LIBERO-Pro scoreboard** ranks systems by how much task memory they are allowed: no memory 44–53%, per-task seed traces 82–93%. Always state the memory regime.

[note: llm-planner-codegen-lineage]

### 3.4 Frontier models as direct policies: what works and what fails

| Works (≥50% in a credible setting) | Fails |
|---|---|
| Open-vocabulary pick/place/sort/stack (Robocurve 19/20; Show-Harness 89% zero-shot with Gemini 3.1 Pro) | Sub-cm insertion and alignment (puzzle 2/20; VIA T-block 10–40%) |
| Long-horizon arrangement in sim (VIA 50–100% across agents) | Pouring and dynamics; in-hand rotation (Astra 0.51% vs RL 76.9%) |
| Turn-based tasks; navigation via primitives (92% / 82%) | Locomotion (0/5; no model stood up a G1) |
| Structured in-context demos (RoboICL; RoboDawn 53.2 → 73.6%) | Physical safety (RoboDojo hardware damage) |

- Every credible direct-LLM system either pauses physics or executes 0.2–10 s open-loop chunks between decisions.
- Distilling harness traces into a small model with the same interface works in simulation:
  - Guava-4B 87.1% vs its GPT-5.4 teacher at 90.4%;
  - Show-Harness fine-tuned 2B 86% vs zero-shot 89%;
  - WAA-9B 1.7% → 43.3% out of distribution.

[note: frontier-models-as-direct-policies]

### 3.5 VLAs, dual systems and light heads

**Most production VLAs are dual systems.** A 2–7B VLM at ~1–10 Hz conditions an action expert that produces 16–50-step chunks at 20–200 Hz. Examples:
- Helix S1, 80M at 200 Hz;
- π0.5–π0.7, with 300M → 860M flow experts;
- GR00T N1.7;
- Gemini Robotics.

The S2 → S1 interface is usually a **learned latent or KV-cache** that a closed API cannot fill. Exceptions now exist (GEN-0/1, and Helix 2.5 trained from random init on human video), so do not assume the low-level model speaks language.

**What a closed API *can* drive:**
- language subtasks (π0.7, Gemini Robotics orchestration);
- visual prompts such as points, traces and masks (RT-Trajectory, HAMSTER, PEEK, LoHo-Manip);
- reference-chunk edits (GPT-as-Policy hybrid);
- residual heads (RoboDual's 16.2M specialist; RL Token's 2-layer MLP, up to 3× faster on critical phases).

**Light heads:**
- ACT, 80M: 10 ms on a 2080 Ti; 80–90% from 50 demos.
- MINERVA, 0.54M: 95.1% on LIBERO but 46–56% on LIBERO-Plus.
- TurboVLA 0.2B; SmolVLA 0.45B; VLA-Adapter 0.5B.

LIBERO is saturated and memorization-dominated (LIBERO-PRO perturbations drop VLAs to 0%), so it cannot choose a head.

**Real-time execution.** RTC handles inference delays up to ~300 ms for chunked flow/diffusion VLAs; training-time RTC is on LeRobot `main` for π0.5 only. An LLM's target specs are seconds old, which is a different problem. It is handled by re-binding targets every tick and by staleness augmentation (§4.3, [gap-3]).

**Executors ignore conditioning by default** [gap-3]. Fast Plans, Faithful Actions:
- waypoints given as suffix tokens had 0.6% sensitivity, and erasing them changed success by 0.0 pp;
- naive injection into every layer dropped success to 15%;
- noise on the goal + a gate + 15% null dropout reached 96.2%, and erasure then cost 7.4 pp.

[note: vla-and-action-heads; gap-3]

### 3.6 Benchmarks and evaluation methodology

**No trustworthy leaderboard measures "frontier API model as robot policy"** well enough for a Claude decision. The relevant ones:
- CodeActionBench: 675 sim attempts over 9 configurations;
- CaP-Bench: 100 trials per tier;
- RoboDojo agent rows: 1 seed, with recipes;
- Robocurve reports: real, unblinded, n = 20 per cell.

**Statistics** [derived]. Minimum detectable improvement from a 50% baseline (α = 0.05, power 0.8):

| Trials per arm | Detectable gap |
|---|---|
| 20 | ≈39 pts |
| 50 | ≈27 pts |
| 100 | ≈19.5 pts |

Separating 50% from 70% needs ~93 trials per arm.
- Use Wilson CIs, and McNemar for paired designs. STEP sequential testing saves up to 32% of trials.
- Best practice is a blind, randomized, matched-initial-condition A/B (TRI LBM: 50 real + 200 sim rollouts per task per policy), or RoboArena-style double-blind pairwise trials.

**Learned-policy benchmarks.**
- LIBERO is saturated. LIBERO-Plus drops VLAs from 95% to below 30%.
- RoboCasa365 composite-unseen is ≤4.4% for every VLA. The BEHAVIOR 2025 winner reached 12.4%.
- Long-horizon composition is where an LLM planner over skills should help.

**Simulated data does train heads.**
- MolmoBot (1.7–1.8M procedural trajectories): 79.2% zero-shot real pick-place vs π0.5 39.2% over 120 trials.
- RoboTwin 2.0: +367% relative with 10 real demos, vs +228% sim-only.
- Sim+real co-training: +38%.

**Critic heads are still open.** Zero-shot progress estimation is about chance (34.4% vs 33.3%). SFT reaches only 45.4% on primitive-local progress (RoboProcessBench T5).

[note: benchmarks-and-sim-datagen]

### 3.7 Industry and competition

**Robot foundation-model companies hold the money and talent.**
- Figure ($39B, Sep 2025), Skild (>$14B), Physical Intelligence ($5.6B confirmed; an ~$11B round per filings, secondary sources), Generalist ($3B), Apptronik (runs Gemini Robotics), Unitree (public).
- None uses a third-party LLM API as the controller.

**Google is the integrated version of Ilia's idea.**
- The ER 2 API is cheap, has a free tier, streams, and orchestrates multiple robots.
- Gemini Robotics 2 and On-Device 2 VLAs are available to early-access partners; Intrinsic is now inside Google.
- Apptronik and Boston Dynamics (Atlas, Jan 2026) outsource their robot brains to it.

**NVIDIA controls much of the open stack.**
- GR00T N1.7: code Apache-2.0, weights under the NVIDIA Open Model License.
- Cosmos; Isaac ROS 5.0 agentic workflows.
- The pending $12.93B Hugging Face acquisition brings LeRobot, SO-101, Reachy and Microduck.

**Frontier labs.**
- OpenAI formed OpenAI Robotics (May 31, 2026), and Astra is the strongest third-party "policy" model.
- Anthropic has no robot model or product. Its work:
  - Project Fetch: Opus 4.7 programmed a quadruped ~19× faster than human teams on shared tasks, but failed closed-loop fetching;
  - "Claude plays robotics": direct control "mostly fails";
  - Project Pilot (drones);
  - the **Model Hardware Standard** (MHS, research preview): driver-level limits and device manifests reachable over MCP. CMU induced 6 faults and MHS blocked all six before motion. LeRobot and AWS Strands are early adopters.

**In-context learning is moving into the action models** (all vendor-only):
- GEN-1.5: 59% (±10%) one-shot from a 3–12 s physical prompt at 100 Hz, with no trial counts;
- Skild S1: 66% vs 9%, measured on a filtered subset of its own pretraining pool;
- π0.7: language coaching.

**Harness-layer competitors.**
- Open: dimOS (4.6k★, default `gpt-5.6-luna`), OpenMind OM1 (2.9k★), ROSA (1.6k★), ros-mcp-server (1.5k★), RAI, AWS Strands Robots, RPent, Show-Harness, innate-os. The AWS Strands reference design is Claude Sonnet 4.5 on Bedrock + Qwen3-VL-2B at the edge + GR00T on SO-101.
- Closed: GRID, Mbodi.
- "LLM calls robot skills via MCP" is now table stakes.

**Evaluators:** Robocurve (YC, $10M seed; Inspect Robots, MIT, "97k+ installs"), Manda, RoboDojo, RoboArena.

**Hardware used by harness builders:**
- SO-101: ~$250 per leader+follower motor kit;
- CAN 6-DoF arms around $2–3k: AgileX PiPER $1,999, I2RT YAM $2,999, MakerMods Metal $2,499;
- Unitree Go2 / G1 for mobility.

[note: industry-competition]

### 3.8 Open-source implementations worth reading (ranked for this harness)

1. **RLinf/RPent.** A Claude Agent SDK planner (`rpent/planner/claude_code.py`) over a frozen π0.5 primitive (`robots/libero/tools.py`); the closest published architecture. The paper headline (82.4% LIBERO-Pro) used Claude Code / Opus 4.7; the repo default is `sonnet`.
2. **showlab/Show-Harness.** Discrete semantic action units with deterministic interpreters. Opus 5 is in the zero-shot top tier. The same units run on fine-tuned 0.8–9B VLMs at 12–33 Hz; those were fine-tuned on recorded demos, not documented frontier distillation.
3. **google-deepmind/gemini-robotics-sdk (Safari).** A non-blocking instruction tool plus an independent ER success detector polled every 0.2 s.
4. **dimensionalOS/dimos.** The most complete open runtime: skills, capability locks, MCP.
5. **strands-labs/robots.** An agent tool that wraps any LeRobot/GR00T policy; motion grants; mesh e-stop.
6. **robocurve/inspect-robots** (+ inspect-robots-yam). The evaluation harness, with native Messages/Responses clients, approver chain, wire capture and a YAM embodiment (MIT). Beware its conflict between image eviction and preserved thinking (§4.5).
7. **Zetta-Embodiment, OpenETA, Thea, PhyAgentOS.** Verification and evidence contracts. Note that OpenETA's 70.8% used an in-episode success oracle.
8. **Mosi-AI/RoboICL.** Receipt grammar, anchored memory, Jev gate (MIT).
9. **DrivingBench harness.** Lease and expiry semantics for a slow LLM driving a dangerous machine.
10. **ROS bridges** (ros-mcp-server, ROSA, RAI). Southbound adapters only, never the control path.

Also worth reading:
- quackd: executor gate order, VLA chunk queue (Apache-2.0);
- piper-astra-jev: geometry toolkit (Apache-2.0);
- metal-arm-harness: envelope (MIT stated in pyproject only);
- EmbodiedSWE: data engine (Apache-2.0);
- agent-as-policy: YAM bridge (Apache-2.0).

GPT-Policy and Zetta have no licence.

[note: github-implementations-sweep]

---

## 4. Core technical challenges, with the most important numbers

### 4.1 Latency and duty cycle

| Quantity | Value | Source |
|---|---|---|
| Opus 5.5 per call, medium, real YAM (n = 4,105) | p50 5.28 s, p90 11.72 s, p99 21.1 s, max 60.3 s | [gap-1] |
| GPT-6 Astra / Opus 5 per call, same rigs | 6.53 / 11.28 / 19.6 s; 8.77 / 20.79 / 40.5 s | [gap-1] |
| First call / final call (Opus 5.5) | p50 16.1 s (every first call > 8 s) / 12.1 s | [gap-1] |
| Share of trial wall time waiting on the LLM | 84.5% (Opus 5.5), 83.2% (Astra), 89.9% (Opus 5); the arm moves 28 s of 269 s | [gap-1] |
| Realized decision rate including motion | 0.127 Hz (7.6 calls/min) | [gap-1] |
| Latency model (Opus 5.5) | ≈ 1.8 s + output tokens ÷ 87 tok/s, R² 0.91; 424 output tokens per call | [gap-1] |
| Lag-1 autocorrelation of call latency | 0.18: a slow call is not predictable from the previous call | [gap-1] |
| A fixed "park after 8 s" rule | Fires on 23% of Opus 5.5 calls (≈7.9 per trial), 31% for Astra, 56% for Opus 5 | [gap-1] |
| Required control rates | 30–50 Hz learned heads; 200 Hz–1 kHz servo; ~83 Hz locomotion | [note: vla-and-action-heads; note: engineering-challenges] |
| VLA inference | π0.5-DROID 128 ms (RTX PRO 5000); ACT ~10 ms; ACT on a CPU laptop ~620 ms round trip | [note: manda-robotics; note: quackd] |

**Unmeasured:**
- Opus 5.5 at low or high effort, with per-message effort, or in fast mode (fast mode projects to ≈3.7 s mean at the Tier 1 shape, UNVERIFIED);
- any time-to-first-token;
- Sonnet 5.5 `between_tools`;
- Astra `ultrafast`;
- ER 2 pointing.

`gaps/gap-1-bench.py` runs all 14 configurations (≈$285 estimated; a ≈$50 first subset).

**Consequences.**
- Plan on ≈0.1 Hz at medium effort. Reaching 0.2 Hz needs ~60% fewer output tokens.
- Outputs must stay valid for 5–40 s of world time, so they must be object-relative and re-bound locally.
- The first call is 3× slower: pre-warm the cache and plan while the arm is at rest.
- Park a loaded arm on load, torque and temperature, not on timers.
- Most simulation results pause physics while the model thinks. On hardware, waiting under load causes thermal trips: Tier 1 had 14/360 overheat terminations.

### 4.2 Spatial grounding: from pixels to millimetres

**Pixel scale** [gap-2].
- One Claude visual token (28×28 px) spans 10–20 mm at 1920 px and 86–172 mm at 224 px, at 0.5–1 m.
- On a D435 at 0.5 m, a 5 mm radius is 14 px unresized, and 1.6 px at Robocurve's 224 px.

**Benchmarks score hits, not distance** [gap-2].
- Every pointing benchmark scores whether the point lands inside a mask.
- The only metric tabletop data is arXiv 2609.28184 (151 scenes at 1920×1080): median error GPT-5.4 16.8 px, Sonnet 4.6 29.3 px. That is ≈7–15 mm and 13–26 mm at 0.5–1 m [derived]. The reference method's own floor is ≈14 px.

**The PhysBrain 1.5 table** (Table 4: Opus 5 at adaptive-low effort, 336 px images, normalized coordinates in the public kit) [gap-2]:

| Benchmark | Opus 5 | Astra | Note |
|---|---|---|---|
| RefSpatial | 56.3 | 78.0 | z ≈ 5.6: real under this protocol |
| PointBench | 68.6 | 71.9 | Not significant |
| Part-Affordance | 78.1 | 55.0 | Opus 5 leads every frontier model |

- The same ER 1.5 model moves 5–9 points between harnesses, and scored 27.0 on Part-Affordance in one kit.
- Effort matters: Astra scored RefSpatial 78.0 at low effort vs 84.8 at high (different harnesses).

**Coordinate conventions** [gap-2].
- Claude must get absolute pixel coordinates on a pre-resized image, with `transformations.oversized_image:"error"` set.
- ER 2 uses `[y,x]` normalized to 0–1000; one unit is ≈0.7 mm at 0.5 m, which is negligible.

**Fixes with measured effect** [note: engineering-challenges; note: frontier-models-embodied-2026]:

| Fix | Effect |
|---|---|
| Calibrated depth | +20 pp short-horizon SR for every agent (LIBERO-Agent) |
| Perception tools | Astra 61.1% → 88.9% (Robo-Harness K1, n = 18) |
| Gripper-cam cursor | Mythos 6% → 32% |
| Interaction-point marker | 40% → 85% |
| 5 cm grid | +15 pts (RoboDawn) |

**Conclusion:** a VLM point is a seed. The millimetres come from:
- mask snapping;
- patch-median depth;
- plane, CAD or edge fits;
- relative visual servoing (piper-astra-jev: 0.6 mm alignment, n = 1);
- or contact.

### 4.3 Contact precision on a $3k arm, and what a head must do

**The YAM arm** [gap-6].
- It is a low-gain joint-impedance arm: Kp 80/80/80/10/10/10 N·m/rad, Kd 5/…/1.5, gravity compensation at 250 Hz, no integrator.
- Grasp-point stiffness is ≈0.34 N/mm in the softest direction [derived].
- No repeatability figure is published.
- Free-space poses sag 1–3 cm open-loop. With an integral settle loop, AGP reached 1.6–3 mm (left arm) and 4–7 mm (right arm).
- Factory encoder zeros were off by up to 5.47°; calibration took one arm from 8.32 to 1.36 mm RMS.
- Kd is capped at 5 on the wire. Contact detection works at ≈1 N.
- Ordinary top-down holds put 9.5–13.7 N·m on the shoulder, against a 9 N·m rating.

**Prior ≤3 mm results** [gap-6]:
- AGP's LLM-written scripts on YAM: 8/10 at 0.73–2.83 mm radial clearance (four pairs); 5/5 with Opus 5 (two pairs).
- ENPIRE RL from one wrist camera: 50 consecutive YAM pin insertions, with ≤8 retries allowed.
- Behaviour cloning is mixed: ACT battery-slot insertion 96% (50 demos), but velcro 20% (100 demos). ManiSkill peg (3 mm clearance, 100 demos): 38% for state-based Diffusion Policy and **0% for every RGB policy**.
- LLM-only closed loops stall at contact: RoboICL peg 2/5, FAEA 0%, Robocurve puzzle 2/20.

**The pattern.** Every cheap-arm success closes the last millimetres with relative visual alignment, slow descent and passive compliance. RL on top of BC closes the gap to ~100%.

**Head conditioning** [gap-3]. Heads follow a condition only if nothing else carries the same information and they are trained against shortcuts. Toy results (2 mm peg-in-hole, 400 episodes per cell):

| Head or baseline | Result |
|---|---|
| Keypoint head trained on exact hindsight labels | 100% with perfect points → 59% at 2 mm noise → 20% at 5 mm |
| Same, trained with test-like noise and staleness | 90% at 2 mm, 72% at 10 mm |
| Tracked selection mask | 89–94% regardless of keypoint noise; 75–78% at 5 cm staleness (keypoint heads ≤2%) |
| 20% condition dropout, when the scene already identifies the target | Erasure tests pass trivially, yet 96–100% of episodes reach *neither* target when condition and scene disagree |
| Redundant-cue dropout | 44–84% follow the LLM's target |
| Analytic association + servo | 100% up to 10 mm noise |
| Analytic staging + frozen primitive | 88–98% |

### 4.4 Verification and self-report

**Self-report is unreliable for every model.**
- Tier 1 false success claims were 8.4% of failed trials for Opus 5.5 vs 10.4% for Astra (p = 0.66) [gap-4].
- 19 of the 22 false claims hinge on a Stack Bowls distinction (0.75 "overlapping" vs 1.0 "stacked") that was unobservable at 224 px.
- Opus 5.5 used `done` for 34 honest partial reports.
- CodeActionBench (Opus 5 13/38 vs Astra 3/20, p = 0.22) is secondary evidence only [note: benchmarks-and-sim-datagen].

**Off-the-shelf detectors are mediocre** [note: engineering-challenges].
- Completion monitoring dominates error events even for good orchestrators (VoLo: >67%).
- FailBench: the best of 13 detectors reaches 0.77 balanced accuracy overall and ≤0.60 on contact assembly. No Claude or GPT-6 model was in its panel.

**Measured on Claude and Astra** [gap-5]:
- An Opus 5.5 verifier at low effort: BA 0.871 overall; 0.968 on placement tasks (0/92 false accepts); 0.707 on contact tasks (25.2% false accepts).
- Astra as verifier: BA 0.886 at twice the price. The two agree with κ 0.91.
- A 2-of-3 vote reaches 0.891. High effort was no better than low.
- Contact-sheet input fixes end-frame illusions (13/56 → 6/56) but invites inference from the motion history (it accepted 20/20 stacks graded 0.75).
- Post-hoc pairing "Opus end-frame AND Astra": 2.1% false success, at 22.5% false rejects.
- A no-vision gripper predicate AND Opus: BA 0.942 on single-object pick-place.

**EmbodiedSkills** [note: vla-and-action-heads]: an agent loop without intermediate verification scored 48.2%, about 34 pp *below* the bare π0.5 reference (82.7%). With verification it scored 86.2%.

**Implication:**
- fail closed;
- route by predicate class;
- for contact predicates, require geometric or force evidence and a dedicated ≥448 px verification view;
- keep episode termination separate from the success claim.

### 4.5 Cost, context and Claude API constraints

**Measured cost** [gap-1; gap-4].
- Opus 5.5 costs $0.898 per Tier 1 trial ($0.0262 per call). 54% of that is cache writes, caused by Inspect Robots' history-rewriting image eviction.
- Opus 5: $1.758 per trial. Astra: $1.14 (the page's cache-adjusted figure).
- The Opus figures reproduce exactly from the run JSONs at $4/$20, $0.20 cache reads and $5 cache writes.

**Projected cost** for an append-only design with one 1920×480 tile per turn [derived]:
- ≈$0.022–0.034 per call;
- ≈$0.97 per 35-call episode cached vs ≈$6.7 uncached;
- ≈$0.05–0.12 per pick-place at 2–4 calls.

**Context** [note: engineering-challenges]. Context grows quadratically if images are appended. Measured cache-hit rates are 92–98% (RoboICL, GPT-as-Policy) when the prefix is stable. Under the same window in RoboDojo L3, Astra got 9% and GPT-5.5 77%, so the rate is provider-specific.

**API constraints that shape the harness** (verified in the bundled Claude API reference):
- Forced `tool_choice` returns 400 on Opus 5.5, Fable 5.1 and Sonnet 5.5. Use `auto` + `strict` tools + a prompt naming the tool.
- Thinking cannot be disabled on Opus 5.5; effort is the dial (default `medium`). Opus 5.5 thinks more per turn than Opus 5 at the same effort.
- `temperature` is rejected.
- **Strict tools do not enforce numeric ranges, string lengths or complex array constraints.**
- Preserved thinking: for accounts created on or after 2026-08-31, any edit before a replayed thinking block is a 400. Remedies:
  - an append-only history;
  - mid-conversation `role:"system"` messages;
  - the `clear_at:"next_user_message"` beta;
  - context editing;
  - server-side compaction (`compact-2026-01-12` threshold, or `compact-2026-09-04` on-demand);
  - `thinking.block_binding.prefix_mismatch_behavior:"drop_block"` under `thinking-binding-controls-2026-08-01`.
- Per-message effort is a beta (`mid-conversation-output-config-2026-07-01`).
- Refusals (`stop_reason:"refusal"`) must map to "hold safe and escalate".
- Never route Claude through the OpenAI-compatible shim: no caching, no thinking, and `reasoning_effort` is ignored. This confounded Robocurve's Fable-vs-Astra comparison.

**The Inspect Robots conflict.** `_evicted_view` (`policy.py:210-258`) edits earlier user turns, while `_anthropic.py` replays thinking verbatim. On a new account this is expected to return 400 after the first eviction (UNVERIFIED until a smoke test).

### 4.6 Safety

**Incidents.**
- RoboDojo halted real Astra testing after "incidents that damaged hardware" [note: robodojo].
- RoboHarm: Astra completed 60/100 harmful instructions; Fable 5.1 completed 34/100 and refused 20 [note: robocurve-gpt6-astra].

**Prompt-level safety does not hold** [note: engineering-challenges].
- Prompt-stated constraints are not enforced: SafeHarness raised collision avoidance from 50.0% to 87.5% with harness route verification.
- Text written in the scene hijacked GPT-4o 27.0% and Gemini 2.5 Flash 29.4% of the time. Claude was not tested.
- Jailbreaks reach 100% on LLM-controlled robots; guardrails cut unsafe plan execution from >92% to <3%.

**Harness bugs are safety bugs** [sources]:
- piper-astra-jev's governor was looser than its prompt;
- quackd's 19/26 non-moving runs came from five harness, prompt and gate faults;
- metal-arm's unarmed dry run skipped later legs of large moves.

**What worked:**
- limits in the driver (MHS blocked 6/6 induced faults);
- controller caps (AGP: ≤0.03 m/s, ≤0.25 m per move, human e-stop);
- per-waypoint envelope checks (metal-arm);
- leases with expiry braking (DrivingBench);
- a heartbeat / deadman (quackd, 0.5 s).

### 4.7 Reproducibility and evaluation validity

- Claude sampling cannot be pinned: non-default `temperature` returns 400. Temperature 0 is not determinism anyway: 80 unique completions in 1,000 on vLLM, and Manda saw ±0.03–0.08 F1 between identical runs.
- π0.5 on identical seeds reproduced only 64% of its successes, and per-task SR moved ≥20 points on 28/120 tasks [note: manda-robotics].
- Harness-only swings of 7.3–48 pp are reported, though CodeActionBench's within-Opus harness effect was only 4.0 pp (p = 0.74).
- The only Opus 5.5 real-robot benchmark (Tier 1) has four validity problems [gap-4]:
  - models ran in blocks per rig (20/20/20);
  - graders knew the model;
  - the tasks were the six easiest for Astra;
  - some tasks are limited by the embodiment: Stand Up Bottles pins wrist pitch and roll at 0, and the safe lid is beyond reach.

---

## 5. New primary evidence from the gap studies

| Gap | Question | What was done | Key findings |
|---|---|---|---|
| 1 | Per-call latency and cost under the real request shape | Mined 360 run JSONs and 360 logs (11,768 calls); wrote an unexecuted 14-configuration benchmark | Opus 5.5 p50/p90/p99 5.28 / 11.72 / 21.1 s. 84% of wall time is LLM; 0.127 Hz. Latency ≈ 1.8 s + output ÷ 87 tok/s. First call p50 16.1 s. An 8 s park rule fires on 23% of calls. Zero errors. $0.0262 per call, 54% of it cache writes |
| 2 | Pointing accuracy in millimetres | Compiled every published number; audited protocols; wrote a 300-target protocol | No Opus 5.5, ER 2 or ER 1.6 pointing numbers. No model has a millimetre error against real-robot ground truth. Opus 5 (low effort) is −21.7 on RefSpatial but +13.4 on Part-Affordance vs the best frontier model. The RefSpatial gap is likely partly a 336 px / normalized-coordinate artifact. GroundingPI weights are unreleased. The correct field is `transformations.oversized_image` |
| 3 | Which conditioning makes a light head follow LLM targets; trained head vs frozen VLA | Literature table (15 systems); CPU toy study (5 conditionings × 4 regimes, 400 episodes per cell) | No paper tests the design. Fast Plans erasure Δ was 0.0 pp until fixed. GAE's best noise scale is 0.10. Exact-label keypoint heads are fragile; tracked masks are robust. Dropout alone is not an anti-shortcut measure. Analytic baselines beat every learned head in the toy |
| 4 | What really happened in the 360 Tier 1 trials | Downloaded all JSONs, logs and the decisive videos; hand-coded 234 failures | Order-confounded design. Completions p = 0.21. False claims 10/119 vs 12/115 (p = 0.66). Placement + alignment cause 50% of Opus 5.5 failures; Astra's largest blocker is no-grasp (37%). About half of all closes are empty. Tracking misses ~2 cm. The proposed skills would touch 66% but directly fix ~16% of Opus 5.5 failures |
| 5 | Accuracy of the independent verifier | 335 labelled episodes, 1,782 verifier calls (Opus 5.5, Astra, Sol) | BA 0.871 (Opus), 0.886 (Astra). Placement: 0/92 false accepts. Contact: BA 0.707. κ 0.91 across vendors. View diversity beats vendor diversity. M2's "<5% overclaim" is not met with one verifier (5.9%) |
| 6 | Is ≤2 mm contact feasible on a YAM, and with how much data | Read the I2RT driver, inspect-robots-yam, the AGP bridge, ENPIRE and rig logs; derived sag and stiffness; wrote a one-day bench protocol | Feasible at ≈1 mm radial clearance with chamfers, via wrist servo + compliance. YAM is a low-gain impedance arm with no repeatability spec, and thermal limits bind. M3's "2/20 puzzle" baseline is a strawman: the bar is a scripted servo baseline on the same rig |

---

## 6. Promising vs unpromising

**Promising (with evidence).**
1. **An event-driven LLM over typed tools that return receipts**, with skills sized so a pick-place takes 2–4 calls. URAI ran 1.3–1.5× faster with 1.5–1.7× fewer output tokens; see also Pigey and RPent.
2. **Grounding locally and treating LLM points as seeds** [gap-2]:
   - a SAM 3 / Grounding DINO + depth object table;
   - labels taken from detector output (Pigey's rule);
   - zoom, cursor and grid tools;
   - mask, depth or CAD refinement.
3. **Servo-first contact skills** [gap-3; gap-6]: wrist-camera relative alignment, slow descent and compliance first; then a fine-tuned frozen VLA primitive; then small heads only where both fail.
4. **Verification with different evidence** [gap-4; gap-5]: per-object sensor predicates, lift tests, verification views, and contact sheets plus final frames, routed by predicate class. Keep termination separate from success claims.
5. **Append-only, cached transcripts** with mid-conversation system messages and server-side compaction: ≈$1 per 35-call episode cached vs ≈$6.7 uncached.
6. **The LLM as offline engineer and data engine:**
   - writing scripted controllers in sim, multiplied with clean-label noise and replay gating (EmbodiedSWE-Gen);
   - generating verifiers and skills (DexAgent);
   - preflight checks and repair logs (GRID);
   - coding agents running RL loops on real arms (ENPIRE).
7. **Safety below the model:** driver limits (MHS), approver chains, leases, heartbeats, thermal parks, scene-text masking.
8. **Distilling harness traces** into small models through the same interface (Guava, Show-Harness, WAA). Simulation evidence only so far.
9. **Rigorous, paired, blinded, wall-clock evaluation** as a product feature.

**Unpromising (with evidence).**
1. The LLM inside a control loop, or emitting joint, torque or fine EEF deltas. LIBERO direct 0–5.5%; no G1 stand-ups; Kwon: raw number lists 10% vs code 60%.
2. Normalized-coordinate pointing from Claude; 224 px squashed frames; raw RGB streams in a coder's context (CaP-X M2 < M1).
3. Trusting the LLM's `done`, or relying on a second verifier from another vendor looking at the same pixels to de-correlate errors.
4. History-mutating image eviction on Opus 5.5: it breaks the cache and conflicts with preserved thinking.
5. Naive demo appending (RoboDojo 22.9 → 12.9%); typed decision models authoring numbers.
6. Narrowly fine-tuned VLAs as language-steerable executors, which lose steerability. Learned-latent S2 → S1 interfaces are impossible with a closed API.
7. Training a contact head on exact hindsight targets, or using condition dropout as the only anti-shortcut measure [gap-3].
8. Sim-compiled pipelines deployed blind: GaP 0/25 and ASPIRE 3/25 vs runtime AGP 23/25.
9. Prompt-only safety; generic ROS/MCP bridges as the control path; humanoid-first products for a small team.
10. Paused-physics sims, n ≤ 20 unblinded comparisons, and task-specific recipes reported as zero-shot.

---

## 7. Recommended direction (details in HARNESS_DESIGN.md)

**Architecture.**
- An Opus 5.5 orchestrator at effort `medium`, with per-message escalation.
- A strict tool surface: `observe`, `zoom`, `locate`, `point_to_3d`, `pick`, `place`, `move_tcp`, `run_skill`, `run_policy`, `verify`, `finish`. Arguments are validated server-side and every tool returns a receipt.
- Local perception: SAM 3, Grounding DINO, calibrated RGB-D.
- A deterministic motion layer: calibrated IK with integral settle, Ruckig, approver chain.
- Contact skills: servo + compliance first, learned heads where they win.
- A fail-closed verifier, routed by predicate class.
- An L1 safety layer in the driver.
- Provider adapters (Anthropic Messages, OpenAI Responses, Gemini) with full wire capture.
- The same harness, run offline, as the data engine.

**The decisions that matter most.**
1. **LLM rate and role.**
   - ≈0.1 Hz, never in a loop; the LLM chooses *what, which and when*.
   - 2–4 calls per pick-place; bound output length.
   - Plan while the arm is at rest; park on load and temperature.
2. **Grounding.**
   - A VLM point is a seed; millimetres come from geometry and servoing.
   - The pointer is chosen by a measured millimetre eval. Default: Opus 5.5 + zoom. A/B arms: ER 2 and local 8B models.
3. **Contact-skill build order.**
   - Analytic association + wrist servo + compliance;
   - then a YAM-fine-tuned frozen VLA primitive;
   - then a small trained head, with noise-matched conditioning, redundant-cue dropout and shift tests.
4. **Verification.**
   - Placement: sensors + per-object gripper predicates + a verification view + one low-effort Opus 5.5 check.
   - Contact predicates: vision is advisory only.
   - `status` is a separate field from `claimed_success`.
5. **API hygiene.** Native Messages wire; append-only transcript; mid-conversation system messages; server-side compaction; strict tools + server-side validation; no shim.
6. **Evaluation.**
   - An own-rig baseline arm (Inspect Robots + Opus 5.5 and Astra), interleaved and blinded, with ≥50 trials per arm before any claim.
   - Scripted-servo and FSM baselines.
   - Robot-hours budgeted: ≈66–112 h per 900-trial campaign.
7. **Positioning.** A provider-agnostic reliability layer and data engine for long-tail arms, compatible with MHS and MCP.

**First concrete steps.**
1. Run a 5-call Inspect Robots smoke test on Ilia's Anthropic account (the preserved-thinking check). Disclose the fix used: an all-frames `image_horizon`, or `drop_block`.
2. Run the ≈$50 subset of `gaps/gap-1-bench.py`: Opus 5.5 low / medium / per-message-low / fast; Sonnet 5.5 `between_tools`; ER 2 pointing.
3. Buy 2 YAM + 2 leaders + 2 D405 + 1 D435i (≈$13.0k + GPU). Run the one-day bench protocol per arm ([gap-6] §6), ending in 20 scripted wrist-servo insertions at 1 mm radial clearance.
4. Run the baseline arm on R1, R2 and R4 with Opus 5.5 and Astra, interleaved and blinded.
5. Build architecture B with the verifier and verification view. Run the 300-target pointing eval ([gap-2] §5).
6. Only then train heads, for the skills the servo baseline fails, and compare head vs servo vs fine-tuned VLA at ≥50 trials per arm.

---

## 8. Open questions and what remains UNVERIFIED

**Latency** [gap-1]: Opus 5.5 at low or high effort, with per-message effort, or in fast mode; time-to-first-token; Sonnet 5.5 `between_tools`; Astra `ultrafast`; ER 2 pointing latency.

**Pointing** [gap-2]: the millimetre error of any frontier or specialist model on robot frames with real ground truth; Opus 5.5 and ER 2 pointing on any benchmark.

**Action heads** [gap-3]:
- whether a trained small head beats an analytic servo and a frozen VLA primitive on real ≤2 mm tasks;
- which conditioning resists shortcuts on a real arm (a ≈150 GPU-hour study is designed).

**Verification** [gap-5]: ER 2 and Robometer-4B as verifiers on real YAM episodes; whether a "final tile only" prompt fix removes inference from motion history.

**Hardware** [gap-6]: YAM repeatability and backlash (no vendor figure).

**Other open questions:**
- Whether Inspect Robots' eviction actually returns 400 on a new account (inferred from code and docs, not run).
- Whether a learned progress critic trained on sim labels beats zero-shot VLMs (SFT reaches 45.4% today).
- Whether plan-while-moving (speculation, leases) keeps success while cutting wall-clock time. No wall-clock-timed manipulation benchmark exists.
- Whether distilling the orchestrator into a 2–9B model holds on real hardware.

**Vendor claims not independently checked:** GEN-1.5, Skild S1, π0.7 coaching numbers; ER 2 chart values; GR00T N2 (newsroom unreachable); PhysicalRSI (RoboDojo #1, with no paper or code).

---

## 9. Where the sources disagree

| Question | One side | Other side |
|---|---|---|
| LLM supervising a VLA | Anthropic: every model was worse than MolmoAct alone in-distribution | GPT-as-Policy: Hybrid 24/50 vs Direct 13/50, but the biggest gains came with ≤1.7% LLM-authored steps and the π0.5 comparator was not rerun. RoboLab: Direct 49/50 > Hybrid 46/50 |
| In-context demonstrations | RoboDojo: they hurt (22.9 → 17.9 / 12.9%) | RoboICL / RoboDawn: +20 pts with structured context |
| Effort / reasoning | Robocurve Opus 5: milestone score 46 → 76 with effort (n = 5; full stacks 0/5 → 1/5) | Anthropic: "no major difference" for Claude on manipulation; as verifier, high effort was no better than low [gap-5] |
| Opus 5.5 vs Astra | URAI native: 32% vs 16% (p = 0.32) | Tier 1 completions: 1 vs 5 /120 (p = 0.21). Neither is significant |
| VLA vs primitives | VoLo sim: the VLA adds a lot (17.8 → 41.8) | VoLo real: No-VLA 19/42 ≥ full 18/42; toy study: analytic baselines win [gap-3] |
| Claude pointing | PhysBrain: −21.7 on RefSpatial, −18.5 on PixMo (Opus 5, low effort) | Same table: +13.4 on Part-Affordance, within ~3 pts on five sets; hostile protocol [gap-2] |
| Verifier de-correlation | FailBench: mean error-set IoU 0.28 across detectors | Opus 5.5 vs Astra on robot video: IoU 0.75, κ 0.91 [gap-5] |

---

## 10. Corrections log: verdict on each CRITIQUE.md error

All 24 items were re-checked against the notes, the gap studies, the local repos and the bundled Claude API reference. "Applied" means both REPORT.md and HARNESS_DESIGN.md now carry the fix.

| # | Critique claim | Verdict | What changed |
|---|---|---|---|
| 1 | REPORT.md missing (disk full) | **Valid** | REPORT.md written. The disk still has ~180 MB free. The `tmp_*` scratch dirs were left in place because gap studies cite their text extractions |
| 2 | 1/120 vs 5/120 is not significant; "last-cm" is unsupported; 44/45 `done`s incomplete | **Partly valid.** p = 0.21 is confirmed. But reading the 44 incomplete `done`s as failures is wrong: 34 are honest partial reports [gap-4]. "Last-cm" survives only in a narrow sense: placement + alignment cause 50% of failures, mostly release strategy, bimanual interference and unobservable seating | Applied, with gap-4's failure-stage data. "Opus 5.5 may simply be worse" dropped |
| 3 | Use Tier 1 `done` rates (44/119 vs 12/115, p ≈ 1.5×10⁻⁶) as strong overclaim evidence | **Rejected as proposed.** That compares termination types, not claims. Real false success claims are 10/119 vs 12/115, p = 0.66 [gap-4]. The criticism of CodeActionBench (p = 0.22, mixed harnesses, Opus 5) is valid | Both documents now say overclaiming is field-wide and largely unobservable at 224 px. R15 separates `status` from `claimed_success` |
| 4 | Pointing numbers cherry-picked; they are Opus 5, not 5.5; part routing contradicts Part-Affordance | **Valid**, and strengthened by gap-2's protocol audit | The pointer is chosen by a measured eval; `part_hint` is not routed away from Claude; GroundingPI weights noted as unreleased |
| 5 | Inspect Robots unchanged with `image_horizon=2` will 400 on a new account | **Valid.** Code verified: `_evicted_view` at `policy.py:210`; thinking replay in `_anthropic.py`; no `block_binding` at `095172f`; enforcement rule verified in the Claude API reference. The outcome on a new account is still UNVERIFIED | M0 smoke test; all-frames horizon or `drop_block`; disclosure |
| 6 | The cost "match" ($0.81 vs $0.90) is coincidental | **Valid.** $0.898 reproduces from 620,537 read / 96,670 write / 137 uncached / 14,496 output tokens | Validation claim withdrawn; projection recomputed (≈$0.97 per 35-call episode) |
| 7 | Pigey 80% → 100% misattributed to `is_grasped` | **Valid.** 4 tasks × 5 trials; the paper credits the verify-and-retry loop | Restated as a system-level comparison |
| 8 | EmbodiedSkills 86.2% → 48.2% misframed | **Valid.** 86.2 vs an 82.74 reference; 48.2 is ~34 pp below it | Reframed; cited as arXiv 2609.01281 |
| 9 | "Unfilled niche" overstated | **Valid** | Niche restated precisely, with the adjacent systems named |
| 10 | M3 (≥10/20 vs an external 2/20) contradicts the protocol | **Valid.** Gap-6 adds that the right comparator is a scripted servo baseline | M3/M4 now need ≥50 per arm on Ilia's own rig, interleaved and blinded, vs a scripted servo and a fine-tuned VLA |
| 11 | URAI's gain is tool authoring + interface; the model differences are noise | **Valid.** Verified in arXiv 2609.39018 §4: Fable 5.1 was the programming agent; Claude ran at xhigh vs Astra at medium; p = 0.78 / 0.32 | Credit and p-values added |
| 12 | The 1 mm / 5 mrad motion contract is a Flexiv number | **Valid.** Gap-6: YAM sags 1–3 cm, and reaches 1.6–7 mm with integral settle | Per-arm, per-region tolerances from the bench protocol; integral settle; sag compensation |
| 13 | π0.5 and MolmoAct2-BimanualYAM are not zero-shot fallbacks on YAM | **Valid.** MolmoAct2: 0/100 StationeryBench completions, arms never left the start pose in 47/100; Manda sim 0/4. No π0.5 YAM checkpoint exists | The fallback is IK + servo; VLAs are fine-tuning bases |
| 14 | TT-RTC is a category error for a small local head | **Valid** | Dropped for local heads; target re-binding + staleness augmentation instead |
| 15 | RoboLab Verified and upstream scoring are not comparable | **Valid.** Manda's 6,000-episode re-runs used upstream scoring; the full patched re-run is "planned, not yet scheduled" | Report both scorers; re-run π0.5 under Verified scoring |
| 16 | Appendix A items are resolved or wrong | **Valid** on all four: 224 px confirmed; strict-mode limits; two compaction features; `clear_at` available on Opus 5.5 | Appendix A rewritten; R16 added |
| 17 | Per-call latency should use Tier 1 (7.6 s median, wall ÷ calls) | **Valid in direction, superseded in number.** 7.6 s includes motion and overhead; the API-call p50 is 5.28 s (gap-1). RoboDojo's 14.66 s is a proxy figure | The gap-1 distribution is used throughout |
| 18 | ER 2's 4.6–6.1 s is joint control, not pointing | **Valid** | ER 2 pointing latency marked UNVERIFIED; run it concurrently or on demand |
| 19 | quackd #25 is Qwen3-32B at temperature 0, not Claude | **Valid.** But the Anthropic quote the critique suggests ("placement near the end moves behaviour more") was **not found** in the bundled reference. Only "a one-line reminder near the end" and turn-scoped reminders are documented | Labelled cross-model; an n ≥ 30 Opus 5.5 test is planned; the quote is marked UNVERIFIED |
| 20 | GPT-as-Policy caveats missing | **Valid** | Caveats added, plus the fixed-15-step-prefix control |
| 21 | Anecdotal and vendor-only figures not flagged | **Valid.** RoboDojo perturbations: 1 episode per layout, on layouts already solved. GEN-1.5: no trial counts. Skild: a filtered subset | Flagged wherever cited |
| 22 | Opus 5 numbers attributed to "Claude" | **Valid** | The model is named in every row |
| 23 | The BOM needs two leaders (≈$13.0k), and robot-hours are the binding budget | **Valid.** $5,998 + $5,998 + $650 + $399; 900 × (251–435 s + 14 s) ≈ 66–112 h | BOM and robot-hour budget corrected |
| 24 | Scene-text injection was measured only on GPT-4o and Gemini 2.5 Flash | **Valid** | Models named; the Opus 5.5 baseline is unknown |

**Errors in the critique's own gap briefs**, found by the gap studies [gap-1; gap-3; gap-4]:
- TraceVLA's traces are the robot's *past* motion, not a target; `landscape/vla-and-action-heads.md` had listed it as a target interface.
- MOKA's distilled Octo policy has no keypoint input.
- `inspect-robots-yam`'s `yam_collision.xml` is collision-only; use Menagerie `i2rt_yam/yam.xml` for dynamics.
- Tier 1 overheats are an order artifact, not a model property.
- The critique's "7.6 s per call" overstates the API call by ~44%.
