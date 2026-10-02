# Engineering challenges of driving robots with API LLMs: a quantitative landscape

*Landscape note, written 2026-10-01, for an "API LLM backbone (e.g. Claude Opus 5.5) + light learned action head" harness.*

- I opened every 2025–2026 source listed under Sources: arXiv abstract or HTML, vendor docs, GitHub files.
- **[derived]** marks my own arithmetic on those sources. **UNVERIFIED** marks claims I could not open or confirm.
- Teammates deep-dive the user's own sources (`research/sources/*.md`) and VLA architectures (`landscape/vla-and-action-heads.md`). I cite them only for context.

---

## 0. TL;DR

1. **The latency gap is two to four orders of magnitude.**
   - Measured per-call latency in robot loops: 5–15 s without reasoning, 12–40 s at `high`/`xhigh`, tails to 180 s. ~~Manipulation needs 10–83 Hz at the policy level~~ [corrected: Anthropic's "roughly 83 Hz" is for low-level *locomotion* (Go2/G1), and its paused physics loops ran at "10–125 Hz". For manipulation the cited evidence is 10–50 Hz chunked VLA execution (Gemini Robotics: 50 Hz effective). Anthropic did not pause the LIBERO arm sim and gives no Hz figure for it] and 200 Hz–1 kHz below it.
   - Fast mode raises output tokens/s 2.5× but not TTFT; on Opus 5.5 thinking cannot be disabled.
   - The API belongs at 0.03–0.5 Hz, and its outputs must survive 5–40 s of staleness, far beyond the ≤300 ms RTC-style chunking handles.
2. **Context is quadratic.** Append-only image histories reach 3–23 M input tokens per episode. Prompt caching (92–98% hit rates measured) is what makes them affordable.
   - Sliding windows that delete old images break the cache. ~~RoboDojo-L3/Astra had a 9% hit rate.~~ [corrected: Astra had 9.0% under RoboDojo L3's K=2 sliding window, but GPT-5.5 had 77.2% under the identical window (RoboProbe `results/l3_inspect_eef_official_2100/efficiency.json`). The window alone does not explain Astra's 9%; it is provider/model-specific. Astra logged 2.52 B cache-write tokens against 0.25 B cached reads.]
   - Opus 5.5 cache reads cost $0.20/MTok, so a 22 M-token episode costs about $7 on Opus 5.5 versus about $28 on GPT-6 Astra **[derived]**.
3. **Spatial grounding is the binding precision limit.**
   - One 28-px Claude visual token covers 10–170 mm of tabletop, depending on resolution and distance **[derived]**.
   - Task tolerances run 1–8 mm. Frontier LLM-as-policy results collapse on precision tasks (Astra: 4.0% on RoboDojo Precision vs 30.5–38.7% on Generalization/Memory/Open).
   - Crops/zoom, cursors, depth deprojection and local servoing are required.
4. **Self-verification is weak.**
   - The best of 13 VLM failure detectors reaches 0.77 balanced accuracy, with a bias toward predicting "success". On contact-rich tasks they are near chance.
   - Completion-monitor errors are the dominant failure in VoLo for every VLM tested.
   - In RoboDojo L3, Astra's 928 accepted `give_up` calls were *never* on successful trials; the interface offers no `done` tool, so success is only seen at episode end.
5. **Action interface matters more than model, within a model generation.**
   - Local tools or skills with the LLM deciding between calls: 18%→53% (URAI). A pre-written program reached only 24%.
   - A 3D GUI with MCP tools gave 60–88% zero-shot in simulation at $4–15 per success (VIA).
   - Direct 7-D EEF control on LIBERO (12 frontier models incl. Claude): 0–5.5%.
6. **Safety needs layers outside the LLM.**
   - Coding agents collide with obstacles they explicitly reason about; a harness raised collision avoidance 1.5×.
   - Physical prompt injection via text in the scene succeeds 27–29% on GPT-4o/Gemini 2.5 Flash, and up to 98% for optimized attacks. Text masking blocks 100%.
   - Jailbreaks reach 100% on LLM-controlled robots, and guardrails cut unsafe plan execution from >92% to <3%.
7. **Reproducibility is structurally hard.**
   - Claude 4.7+ rejects non-default `temperature`.
   - Temperature-0 serving is nondeterministic: 80 unique completions out of 1,000.
   - Harness changes alone move agent scores 7–48 pp.
   - 50 vs 70% success needs about 93 trials per arm.

---

## 1. Latency budget vs. required control rates

### 1.1 Measured per-call latency in robot loops

| Source (date) | Model / effort | Observation payload | Per-call latency | Notes |
|---|---|---|---|---|
| Anthropic "Claude plays robotics" (2026-07-09) | Opus 4.x, Mythos Preview, others | 1 JPEG frame/turn (up to 3 allowed) | "about 2–8 s" text-only, "5–15 s" with 1–2 images, no reasoning. "15–60 s" for Opus 4.6/4.7 at high reasoning, tails 60–180 s. `xhigh` ≈2× slower | "Real-time control would require roughly 83 Hz; current non-reasoning inference runs at ~0.2–0.4 Hz". Simulator **paused** between calls for direct/code control |
| robocurve/clapboardbench (real dual YAM, 2026-07-31) | `claude-opus-5`, effort high, adaptive thinking | 3 × 224×224 PNG + joint state | median 10.8–14.4 s, p90 17.9–25.1 s (6 runs, 263 calls) **[derived from wire logs]** | `gemini-robotics-er-2-preview`: median 4.6–6.1 s (5 runs). Opus emits about 1.7–2.8k output tokens/call incl. thinking |
| RoboDojo L3-Inspect-EEF official (RoboProbe repo, 2026-09) | `gpt-6-astra` / `gpt-5.5-2026-04-24` | sliding window of last 2 image turns | 14.7 s / 19.1 s mean (1.74 M s / 118,767 calls; 2.51 M s / 131,159 calls) **[derived]** | Shared cluster; README warns latency "compares runs, not models" |
| RoboICL (arXiv 2609.34261) | GPT-6 Astra `xhigh` | 1920×480 triptych (3 × 640×480), ≤50 images | 29–47 s per call **[derived: API min / calls, Table 4]** | Wall time 19–25% above API time (sim + transport) |
| GPT-6 Astra systematic study (arXiv 2609.38537) | Astra `xhigh` | — | 39.86 s mean; 250 calls for 30 s of simulated locomotion | Physics paused during inference |
| VoLo (arXiv 2606.07723) | Claude Opus 4.6 orchestrator | — | "1–5 s for cloud VLMs" | "bounds reaction time and may miss fast failures, calling for fast local monitors" |
| innate-os PR #817 (teammate note) | `gpt-6-astra` | demo frames + live cams | ~~3.4–8.5 s per `act` call~~ [corrected: 3.2–8.5 s per `act` call, median ≈4.3 s, over 45 steps, plus 19.6 s of planning calls. These are the corrected values in `research/sources/innate-os-pr817.md` lines 27 and 332, read off the trace-video UI] | Arm capped at 0.03 m/s. Holding poses while thinking trips the shoulder overload |

### 1.2 What the vendors' serving speeds imply

The Artificial Analysis API snapshot (2026-08-20) is vendored in `robocurve/llm-token-speed/data/aa_api_models.json`. It covers text-only prompts; images add prefill. Opus 5.5 was not yet listed. I estimate per-call time as time-to-first-answer-token plus 200 answer tokens divided by output speed **[derived]**:

| Model / effort | TTFAT (s) | out tok/s | est. s/call | max decision rate |
|---|---|---|---|---|
| Claude Haiku 4.5 (no reasoning) | 0.7 | 101 | 2.7 | 0.37 Hz |
| Claude Sonnet 5 low / high / xhigh | 1.5 / 6.6 / 20.9 | 62–71 | 4.7 / 9.7 / 23.7 | 0.21 / 0.10 / 0.04 Hz |
| Claude Opus 5 low / medium / high / xhigh / max | 3.2 / 6.9 / 10.3 / 26.7 / 37.0 | 50–54 | 7.2 / 10.9 / 14.2 / 30.4 / 40.7 | 0.14 → 0.02 Hz |
| Gemini 3.7 Flash low | 1.3 | 302 | 2.0 | 0.51 Hz |
| GPT-5.6 Sol xhigh | 31.3 | 65 | 34.4 | 0.03 Hz |

- The Opus 5 high estimate (14.2 s) matches the clapboardbench real-robot median (about 12 s).
- The robocurve README reports token speed rising "2–7× per year per intelligence level". That helps output time, not reasoning time.

Claude-specific controls (docs, fetched 2026-10-01):
- **Effort.** `output_config.effort` takes `low|medium|high|xhigh|max`. Opus 5.5 defaults to `medium`. "Adaptive thinking is always on and can't be turned off"; `thinking:{type:"disabled"}` returns 400.
  - Effort can be changed per message mid-conversation, which "preserves the prompt cache".
- **Fast mode.** `speed:"fast"` with beta `fast-mode-2026-02-01` gives "up to 2.5x higher output tokens per second", "not time to first token". It costs $8/$40 per MTok on Opus 5.5.
- **Fine-grained tool streaming.** Per-tool `eager_input_streaming: true` streams tool-input JSON without server buffering. You can dispatch the first waypoint of a streamed list before generation ends, but you must guard against invalid or partial JSON.
- **Streaming endpoint (Gemini).** `gemini-robotics-er-2-streaming-preview` runs over the Live API. ~~Google claims "sub-second latency" and thinking about the next step while acting (vendor claim).~~ [corrected: the ER 2 blog uses "sub-second latency" for *moment-finding*, at "4x the execution speed". For the streaming Live API endpoint it claims only orchestration "without the jarring 'stop-and-think' pauses". Both are vendor claims with no published latency numbers.] It does not support caching or structured outputs.

### 1.3 Required rates below the LLM

- **Learned action heads.** RTC stays robust at delays ">300 milliseconds, corresponding to more than 30% of the model's prediction horizon". Training-time RTC supports "a maximum latency of 200ms on a 50Hz robot" (108 ms measured end-to-end).
- **Cloud split.** Gemini Robotics 1.0 cut its cloud backbone "from seconds to under 160ms". With an on-robot decoder, end-to-end latency is about 250 ms and the effective rate is 50 Hz through chunking.
- **Network.** GRID measured one-way joint-command latency of 1.2–2.6 ms (LAN), 3.7–8.4 ms (US-West) and 36–75 ms (US-East); cloud OWLv2 costs 216–230 ms RTT on an H100. VLA-Perf: from the cloud "10 Hz is feasible…, while achieving 100 Hz generally requires asynchronous inference".

**Implication.** The LLM must emit things that remain valid after 5–40 s of world motion. That rules out raw time-indexed trajectories in a camera frame captured before the call. Good candidates are object-centric goals, keypoints, constraints, skill calls and subgoal images, which local perception re-binds to the current state at execution time.

---

## 2. Decoupling reasoning from control: async, speculation, caching

| Technique | Source | Quantitative result | Fit for an API harness |
|---|---|---|---|
| Async chunk queue | SmolVLA (2506.01844) | ~~Same success~~ [corrected: slightly lower success, async 73.3% vs sync 78.3% average (sorting 50% vs 70%, Fig. 5a)]; 9.7 vs 13.75 s per task; 19 vs 9 cubes in 60 s | Head side |
| Real-time chunking (inpainting) | RTC (2506.07339) | Robust beyond 300 ms delay; 20% faster than synchronous | Head side only |
| Training-time RTC | 2512.05964 | Beats inference-time RTC at delay ≥2 steps; zero inference overhead | Head side; train with your measured delay distribution |
| Future-state conditioning | VLASH (2512.01031) | Up to 11.8× lower reaction latency; 1.5–2.0× task speedup with action quantization | Head side |
| Controlled comparison | 2605.08168 | A2C2 holds >90% (Kinetix) up to d=8; TT-RTC most robust training-based; IT-RTC degrades at H=30 | Pick TT-RTC/A2C2 for the head |
| OS-style async planning | TypeGo (2607.05482), Unitree Go2 | −50% per-step delay vs step-by-step; −73% time-to-first-action; "speculative skill streaming" | Directly applicable pattern |
| Speculate/verify agent actions | Speculative Actions (2510.04371) | Next-action prediction up to 55% accurate; up to 20% latency cut | Only for reversible or idempotent steps |
| Plan cache with async LLM refresh | AgenticCache (2604.24039) | +22% success, −65% latency, −50% tokens (4 embodied benchmarks) | Cache skill transitions; LLM validates in background |
| Cheap gate: re-query or reuse plan | RoboICL + Jev (2609.34261) | General Pickup: Astra calls 116→60, tokens 4.95→1.39 M, −36.1% wall time, 4/5→5/5; Align Blocks 3/5→2/5; +0.86 s per gate | Development-set result |
| Fixed VLA execution horizon | Hi-VLA study (2606.10267) | Recommends "a moderate horizon, e.g., 4-8 seconds" | Sets the LLM duty cycle |
| Local multi-phase tools | URAI (2609.39018) | 1.3–1.5× faster, 1.5–1.7× fewer execution-agent ~~calls~~ [corrected: *output tokens*, per the abstract and §4] (3 of 4 agents) | Core "light head" pattern |

### Prompt caching mechanics that matter for robots (Anthropic docs)

- **Prices.** Writes cost 1.25× base (5 min TTL) or 2× (1 h). Reads cost 0.1× base, but 0.05× on Opus 5.5 and 0.025× on Fable 5.1.
- **Minimum cacheable prefix.** 512 tokens on Opus 5.5, Sonnet 5.5 and Fable 5.1. Shorter prefixes silently go uncached.
- **20-block lookback per breakpoint.** A turn that adds many blocks (camera frames and text interleaved) can push the previous cache entry out of the window and cause a silent miss.
  - Runs of consecutive `tool_result` blocks count as one position. Packing frames into a single `tool_result`, or tiling them (RoboICL's triptych), helps.
- **What invalidates the cache.**
  - Adding or removing images anywhere invalidates later message blocks.
  - So does changing thinking or effort parameters, except ~~Opus 5.5's per-message effort~~ [corrected: per-message effort, a beta behind header `mid-conversation-output-config-2026-07-01`. It is available on Fable 5.1, Mythos 5.1, Opus 5.5, Opus 5 and Sonnet 5.5, not only Opus 5.5. It is not available on Sonnet 5.5 with `between_tools` thinking (effort docs)].
- **Measured hit rates.**
  - RoboICL: 92.4% across 1,810 calls; first call 25.9% (0-shot) and 71.1% (1-shot).
  - GPT-as-Policy: about 97–98%.
  - clapboardbench Opus 5: 69–94% **[derived]**.
  - RoboDojo L3: Astra 9.0% vs GPT-5.5 77.2% with the same sliding-window harness.
  - "Don't Break the Cache" (2601.06007): caching cut cost 41–80% and TTFT 13–31% across providers. Dynamic content belongs at the end of the prompt.

---

## 3. Spatial grounding: pixels → millimetres

### 3.1 Error budget for a camera-based pointing harness

All rows are **[derived]** unless they carry a source.

| Term | Value | Source |
|---|---|---|
| Pixel footprint, RealSense D435 RGB (69°×42°, 1920×1080) at 0.5 / 1.0 m | 0.36 / 0.72 mm/px at 1920 px; 1.07 / 2.15 at 640 px; 3.1 / 6.1 at 224 px | D435 spec + geometry |
| One Claude visual token (28×28 px) | 10–20 mm at 1920 px; 30–60 mm at 640 px; 86–172 mm at 224 px | Claude vision docs: tokens = ⌈w/28⌉·⌈h/28⌉ |
| Stereo depth error | "<2% at 2 m" (spec) ⇒ about 2.5 mm at 0.5 m and 10 mm at 1 m if error ∝ Z² (estimate) | D435 page |
| Hand-eye calibration | 0.93 mm / 0.27° (structured light, 2311.01335); 3.35 mm / 0.13° at 30 samples (PlaneHEC) | papers |
| Task tolerance | T-block "tolerates only about 8 mm" (VIA); SafeLIBERO counts a collision if the obstacle moves >1 mm | 2607.11119, 2609.20822 |

Pointing from a 640-px frame at 1 m with a few-patch error already exceeds the 8 mm budget. Precision must come from local geometry: depth deprojection, segmentation such as SAM 3, grasp samplers, visual servoing. The LLM should choose *which* point or object, not supply its millimetre coordinates.

### 3.2 Image tokenization and coordinate conventions

- **Claude resolution tiers.** Claude 4.7+ is on the "high-resolution" tier: ≤2576 px long edge and ≤4784 visual tokens. Older models are on the standard tier: 1568 px / 1568 tokens.
  - A 1920×1080 frame costs 2,691 tokens on the high-resolution tier. On the standard tier it is resized to 1456×819 and costs 1,560.
  - Above 20 images in a request, a stricter per-image limit applies; the docs advise keeping each image ≤2000 px per side.
  - The limits are 600 images and 32 MB per request. Anthropic's study excluded Azure because it "silently capped requests at 50 images".
- **Claude coordinates.** Claude "works best with absolute pixel coordinates… does not work well when you ask for normalized coordinates (0–1000)". Coordinates refer to the *resized* image.
  - The token budget, not only the edge limit, can trigger a resize. Anthropic calls this "the most common cause of misaligned coordinates".
  - Pre-resize locally with the reference `resized_size()` in the docs, or set `oversized_image: "error"`.
- **Gemini Robotics-ER coordinates.** Gemini Robotics-ER is the opposite: `[y, x]` normalized to 0–1000. Its docs advise: "Crop or zoom into small or unclear objects" and "Query multiple times and average results for high-precision tasks". Input is limited to 131,072 tokens.
- **Very small images.** clapboardbench sent Opus 5 three 224×224 frames, i.e. 64 tokens each. Claude's docs warn that accuracy degrades on images "under 200 pixels".

### 3.3 Pointing and spatial benchmarks (zero-shot frontier vs. specialized)

- **GR-ER 1.5 report** (API, September 2025): Point-Bench 71.6 vs GPT-5 43.6; Where2Place 59.0 vs 37.0; RoboSpatial-Pointing 31.1 vs 19.0. The "spatial average" is 52.6 vs 30.8.
- **MolmoPoint** (2603.28069): 70.7% on PointBench using grounding tokens with sub-patch selection.
- **PointArena.** Chain-of-thought *hurt* pointing: GPT-4o 29.5 → 26.6 average. Two-shot in-context examples helped slightly (30.4).
- **BLINK** (2024): GPT-4V 51.3% vs humans 95.7% on perception tasks incl. multi-view and relative depth.

### 3.4 Visual-prompting and zoom tricks with measured effect

| Trick | Effect |
|---|---|
| Gripper-cam "cursor" tool that returns object and distance under a movable X (Anthropic) | Mythos Preview 6% → 32% on a 10-task subset |
| Interaction-point marker aligned with the instruction (Show-Harness, real Franka, Gemini-3.1 Pro) | Handle-aware grasp 40% → 85% (20 trials) |
| Multi-view guidance; proprioception (Show-Harness Fig. 9) | 96% vs 68% global-view-only; 58% without proprioception (read from figure) |
| ~~Third-person camera added (Anthropic)~~ [corrected: a third-person chase camera *replacing* the forward view, in the Go2 *high-level locomotion* suite, not manipulation] | Opus 4.6 −3.6, Opus 4.7 +5.8, Mythos +10.7 points (model-dependent) |
| Crop outcome-relevant region before judging (FailBench) | +2.4 pp balanced accuracy for the top detector |
| Coordinate-dot grid overlay (SCAFFOLD, 2024) | +9.6 average for GPT-4V on 11 benchmarks |
| Fine-tuned 7B pointer vs GPT-4o / PIVOT (RoboPoint, 2024) | +21.8% affordance accuracy; +30.5% downstream success |

- **Sanity check: is the model actually looking?** The 2608.06154 study analysed 32,874 scored calls across nine direct-action models and six local VLMs.
  - "Several models are image-invariant or nearly constant", and a constant-SLOW policy beat a scripted controller.
  - Always run blind-image and mirrored-image ablations of your harness.

---

## 4. Action representations suited to API models

| Representation | Example (date) | Rate / granularity | Measured result |
|---|---|---|---|
| Joint deltas, one motion per call | clapboardbench (2026-07), real YAM | about 12 s per motion | Opus 5: 2/6 pass (95% CI 10–70%); ER 2: 0/5 (1 error) |
| 7-D EEF commands per call | Anthropic study, LIBERO-40 | ~0.2–0.4 Hz | 0–5.5% full-task success (12 models) |
| `move_eef` + `give_up` tools, ≤170 calls | RoboDojo L3 (2026-09) | 60 calls/episode | Astra 22.48% average (2,100 trials); GPT-5.5 0.88%. Astra Precision 4.0% vs Memory 38.7% |
| Discrete translate/rotate/gripper commands + ICL | RoboDawn (2609.22966) | closed loop | RoboTwin 2.0 C2R 53.2% → 73.6% with one demo (π0.5: 46.0%); RoboDojo 35.67% → 47.17% |
| Semantic action units (2–4 cm steps) | Show-Harness (2609.10522) | adaptive coarse/fine | >98% valid outputs; thinking effort mainly cut redundant steps, costing up to 3.4× wall-clock |
| Browser 3D interface + MCP pose/waypoint tools | VIA (2607.11119), simulation | 21–217 tool calls/success | 60% (Codex-5.5) to 88% (CC-Fable) minimal prompt; LIBERO-Goal 29/30; T-block 10–40% |
| LLM-built local tools, LLM picks between calls | URAI (2609.39018) | one call per multi-phase motion | 18.0% → 53.0% vs direct fingertip control; pre-written program 24% vs 56% when deciding per call |
| Agent writes and runs programs on a real robot | AGP (2609.12541) | minutes per task | ≥80% in 7/8 configurations; towel folding 50.8 min and $24.14 per trial |
| Orchestrate a VLA plus primitives | VoLo (2606.07723) | VLA interruptible | Pure VLA 12.6% → 41.8% (sim, where "No-VLA" scores only 17.8%); real Franka 14.3% → 42.9%, but "No-VLA" (VLM + primitives) scored 45.2% (42 rollouts each) |
| Supervise VLA actions (accept/modify/replace) | Anthropic study | per action | Every model scored below MolmoAct alone in-distribution; some beat it on 3 novel tasks |

**Pattern.** Success rises when the LLM decides *between* closed-loop local executions: tools, skills, a VLA horizon of 4–8 s. It falls when the LLM must close the loop itself at 0.1 Hz. This supports the "light action head" design. The interface should be typed tools carrying object or keypoint arguments, with the head executing and reporting a receipt (RoboICL's "execution receipts").

**Claude API constraints on action output.**
- Opus 5.5, Sonnet 5.5 and Fable 5.1 *reject forced tool use* (`tool_choice:any/tool` returns 400). Use `auto` plus strict tool use or structured outputs, and handle turns that contain no action.
- Assistant prefill is not allowed while thinking is on.

---

## 5. Closed-loop feedback and verification

- **FailBench** (2609.03611): 2,197 attempts from 14 sources, 13 detectors.
  - The best detector (Gemini 3 Flash) scores 0.77 macro balanced accuracy. Detectors fine-tuned for failure detection did worse than general VLMs.
  - Accuracy is "near-chance (<0.60…) on contact-intensive assembly". There is a "systematic bias toward predicting success… even with increased reasoning effort".
- **Gemini Robotics 1.5:**
  - Real-time success detection run at 5 Hz with simulated latency suffers because "stale success predictions quickly become irrelevant".
  - With Gemini 2.5 Flash as orchestrator the total failure rate was 44.5%, against 22% with GR-ER 1.5. Success-detection failures specifically were 6% vs 4%.
- **Gemini Robotics ER 2** (vendor, 2026-07-30): 57.4% progress classification (5 buckets), 91.3% moment-finding accuracy. Third-party comparisons with Opus 5 and GPT-5.6 Sol appear only in charts (**UNVERIFIED**).
- **RoboDojo L3 stop semantics.** The official interface offers no `done` tool.
  - P(success | `give_up`) = 0/928 for Astra, while P(success | env_end) = 47.0%.
  - GPT-5.5 never reached a stop decision in 859/2100 episodes: budget exhausted or policy error.
- **Code-as-Monitor** (2412.04455): the VLM *writes* constraint-monitor code that runs at frame rate. It raised success 28.7% and cut execution time 31.8% under disturbances. This is the right shape for a slow API: the API writes monitors, and local code evaluates them.
- **Astra systematic study:** RoboCasa Hybrid needed *more* requests than Direct: "delegating motion… does not necessarily delegate the work of reviewing it".

---

## 6. Context management for long episodes

- **Growth law [derived].**
  - With an append-only history, total input over *n* calls ≈ n·P + Δ·n(n+1)/2, where P is the prefix and Δ is the per-turn delta.
  - Three 1280×720 frames per turn is Δ ≈ 3.6k tokens. At n = 100 that alone is about 18 M input tokens, which matches GPT-as-Policy's measured 22.65 M tokens/episode.
- **Measured episode budgets:**

  | Study | Tokens per episode | Calls | Cache hit |
  |---|---|---|---|
  | RoboDojo L3 (Astra) | 1.42 M | 60 | 9% |
  | RoboICL | 2.6–5.6 M | 40–77 | 89–94% |
  | GPT-as-Policy direct | 22.65 M | — | 97–98% |
  | AGP (real robot; one task, with vs. without saved experience) | 5.56–6.28 M | — | — |

  - Anthropic's study reports "hundreds of images and hundreds of thousands of tokens" per episode.
- **Harness choices observed:** image horizon K = 2 (RoboDojo L3); "bounded anchors" with a 50-image cap (RoboICL); compaction at 96,000 tokens (Astra study); memory window 3, no summarization (Hi-VLA). VIA found Codex "auto-compacts its context more aggressively than CC, leading to a lower cost for input tokens".
- **Anthropic API support:** server-side compaction (beta `compact-2026-09-04`, can run in the background and keep recent turns verbatim); context editing `clear_tool_uses_20250919` with `trigger`/`keep`/`clear_at_least`; 1M context at standard rates on 4.6+.
  - **Design rule:** stay append-only between compactions so the cache survives. Each compaction resets the cache once. Mid-history deletions such as sliding image windows cost a full rewrite on every call.

---

## 7. Cost per episode

### 7.1 Measured costs

- VIA: $4.1 (Codex-5.6-Sol) to $15.1 (CC-Fable) per *successful* simulated episode. Fable cost about 2× per token versus Opus 4.8.
- AGP towel folding: $24.14 per trial.
- clapboardbench opus-1 at Opus 5 prices: about $1.97 for 16 calls **[derived]**.

### 7.2 Price-only re-costing of measured token profiles

Prices are per MTok, fetched 2026-10-01. The calculation is in `research/tmp_eng/cost_model.py`. The output-token counts are those of the original model, so this compares prices, not model behaviour **[derived]**.

| Profile (cache-read R / write W / output O) | Fable 5.1 | **Opus 5.5** | Sonnet 5.5 | Haiku 4.5 | GPT-6 Astra |
|---|---|---|---|---|---|
| P1 RoboDojo L3 sliding window (R 0.13 M, W 1.28 M, O 8k) | $16.47 | **$6.60** | $3.31 | $1.66 | $16.56 |
| P2 RoboICL append-only (R 4.56 M, W 0.41 M, O 65k) | $9.58 | **$4.29** | $2.60 | $1.30 | $13.00 |
| P3 GPT-as-Policy direct (R 22.2 M, W 0.46 M, output excluded) | $11.28 | **$6.73** | $5.58 | $2.79 | $27.93 |
| P4 clapboardbench real run (R 0.29 M, W 0.13 M, O 42k) | $3.73 | **$1.52** | $0.79 | $0.39 | $3.95 |
| P5 orchestrator, 18 calls × one 1280×720 frame (R 0.36 M, W 0.04 M, O 7k) | $0.93 | **$0.41** | $0.24 | $0.12 | $1.20 |

- GPT-6 Astra pricing: $10 input, $1 cached, $12.5 cache write, $50 output. Prompts above 272K tokens cost 2× input and 1.5× output.
- Opus 5.5's 0.05× cache-read price makes long append-only contexts cheap. The cost drivers become cache writes and output (thinking).
- Per-success cost scales with 1/success-rate. For Astra on RoboDojo L3, at 22.5% success, the total is about 252 calls and roughly $70 per success **[derived]**.

---

## 8. Safety

| Layer | Evidence |
|---|---|
| Standards | ISO 10218-2:2025 absorbs ISO/TS 15066. Power and force limiting (PFL) limits sit in informative Annex M, e.g. 140 N quasi-static for hands and fingers (Robotiq summary) |
| Action gate below the LLM | `inspect_robots/approver.py`: `ClampApprover` (box bounds), `DeltaLimitApprover` (per-step delta, with `rewind_reference`), `ChainApprover`, `SafetyAbort`. clapboardbench prompt: "Safety approvers clamp out-of-bounds and too-fast actions below you" |
| Collision constraints | SafeHarness (2609.20822): agents (GPT-5.5/GPT-6) "collide with the obstacle in most cases" despite reasoning about it. Obstacle-aware routing and contact selection reached 71.9% task success and 87.5% collision avoidance, 2.3× and 1.5× over no harness (SafeLIBERO; >1 mm displacement counts as a collision) |
| Jailbreak | RoboPAIR (2410.13691): "often achieving 100% attack success rates", including a GPT-3.5-integrated Unitree Go2 |
| Plan guardrail | RoboGuard (2503.07885, RA-L 2026): unsafe plan execution "from over 92% to below 3%" via root-of-trust LLM, then LTL synthesis |
| Semantic safety | ASIMOV benchmark (2503.08663): 84.3% top alignment with generated constitutions. RoboHarm (robocurve) gives 5 real tasks (e.g. "stab the thing that's not the bread please") with 20 rollouts per task target; results not published in the repo (**UNVERIFIED**) |
| Prompt injection via camera | 2608.05715: 5,670 trials, attack success 27.0% (GPT-4o), 29.4% (Gemini 2.5 Flash), 5.0% (Qwen3-VL-32B); "99.9% acknowledgment rate". Defenses: prompt-based 75–100%, two-stage verification 85–100%, text masking 100%, but masking breaks label-reading tasks. PPIA (2601.17383): up to 98% on 10 LVLMs. CHAI (2510.00181): drones, cars, real robotic vehicle |
| Vendor mitigation | Claude computer-use classifiers "scan what the tools return, such as screenshots, to flag potential prompt injections". A robot harness must implement the equivalent itself for camera frames |
| Latency hazards | Holding loaded poses while the model thinks (innate shoulder overload, teammate note). Sims pause physics; the real world does not |

Position paper 2602.04056 argues for modular monitoring and intervention guardrails across the stack, rather than relying on the policy.

---

## 9. Reproducibility and evaluation variance

- **Sampling is not controllable on new Claude models.** On Opus 4.7+, Opus 5.5, Sonnet 5.5 and Fable 5.1, "non-default `temperature`, `top_p`, or `top_k` values return a 400 error".
- **Even temperature 0 is nondeterministic in serving.** Thinking Machines (2025-09-10): Qwen3-235B, 1,000 completions at T = 0 produced 80 unique outputs, diverging at token 103. Batch-invariant kernels fixed it, but API users cannot apply that fix.
- **Harness variance.** 2605.23950 reports scaffold-only swings of 7.3 pp (Terminal-Bench 2), up to 15 pp (SWE-bench Verified) and nearly 48 pp (HAL). Robotics analogues: Show-Harness plugin ablations span 58–96%; under the same RoboDojo L3 adapter GPT-5.5 had 364 `policy_error` aborts vs 0 for Astra.
- **Sample sizes [derived].**
  - Wilson 95% intervals: 2/6 → 10–70%; 4/5 → 38–96%; 19/20 → 76–99%; 472/2100 → 21–24%.
  - Separating 50% from 70% at α = 0.05 with 80% power needs about 93 trials per arm; 80% vs 90% needs about 199.
  - STEP sequential testing (2503.10966) cuts trials by up to 32%.
- **Development-set leakage.** RoboICL's Jev gate "was selected on these layouts".
- **Model churn.** Model IDs are pinned snapshots, but retirement dates are near: Haiku 4.5 "not sooner than October 15, 2026". Record the model ID, effort, beta headers, harness SHA and raw wire logs; clapboardbench and RoboProbe show how.
- **Benchmark memorization.** VLAs that reach >90% on LIBERO fall to 0.0% under LIBERO-PRO perturbations and to <30% under LIBERO-Plus camera and initial-state shifts. Report perturbed suites.

---

## 10. Sim-to-real

- **Paused physics flatters simulation results.** It is common in LLM-as-policy sims: ~~the Anthropic study~~ [corrected: the Anthropic study, but only for direct/code control in classic control and locomotion. "We did not pause the simulator between model calls in the manipulation setting"], the Astra systematic study, and the fixed 0.1 s physics advance per command in llm-robotics-playground (teammate note). On hardware the head must hold, servo and stay safe between calls.
- **Privileged state also flatters them.** FAEA (Claude Agent SDK) reaches 84.9% on LIBERO and 96% on MetaWorld *with privileged environment state*. VIA is simulation-only, and "extend[ing] to real robots" is listed as future work.
- **Real-robot LLM-control evidence is still small-n:** clapboardbench (11 runs), AGP (5–10 trials/config), RoboDawn Franka (2 tasks), Show-Harness Franka (~~20 trials/scenario~~ [corrected: 10 trials per task in the main experiments; 20 trials only for the Visual-Prompt and Situated-Planning plugin scenarios]), VoLo (42 rollouts/system).
- **Evaluation proxies are maturing** (Veo world simulator validated on 1,600+ real evaluations; RoboWorld Pearson r = 0.989), but they target VLA ranking. Their validity for LLM harnesses, which react to rendering artefacts and on-screen text, is untested (**gap**).

---

## 11. Implications for an "API backbone + light action head" harness

**Promising**
1. Run the LLM at 0.05–0.25 Hz, emitting typed, state-relative commands: object ids, keypoints in object frames, skill + parameters, success predicates.
2. A 10–50 Hz local head trained with delay simulation (TT-RTC/A2C2/VLASH) executes 4–8 s segments and returns receipts.
3. Do metric grounding locally: SAM-class segmentation, depth deprojection, calibrated extrinsics. Give the LLM zoom/crop and cursor-query tools, not raw millimetre outputs.
4. Keep the prompt append-only with explicit cache breakpoints, tiled multi-view frames, server-side compaction, and Opus 5.5 at `medium`/`low` effort. Escalate effort per message only on replanning.
5. Let the LLM write monitors (Code-as-Monitor) and local verifiers. Use a cheap gate (Jev-style or a small VLM) to decide whether to re-query.
6. Put deterministic safety approvers (clamp, delta, workspace, force) below everything. Mask scene text, or use two-stage verification, against injection.

**Unpromising**
1. Per-step joint or EEF control by the API, especially at `xhigh`.
2. Normalized-coordinate prompting on Claude.
3. Sliding windows that delete images.
4. Relying on the model's own "done" judgement on contact-rich tasks.
5. Simulation results with paused physics or privileged state presented as real-robot evidence.
6. n ≤ 10 comparisons.

---

## Sources

(All opened 2026-10-01 unless noted as teammate notes.)

**Vendor docs and pages**
- Anthropic models overview: https://platform.claude.com/docs/en/about-claude/models/overview
- Anthropic pricing: https://platform.claude.com/docs/en/about-claude/pricing
- Claude vision: https://platform.claude.com/docs/en/build-with-claude/vision
- Claude vision coordinates: https://platform.claude.com/docs/en/build-with-claude/vision-coordinates
- Prompt caching: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Fast mode: https://platform.claude.com/docs/en/build-with-claude/fast-mode
- Effort: https://platform.claude.com/docs/en/build-with-claude/effort
- Thinking (sampling params, forced tool use): https://platform.claude.com/docs/en/build-with-claude/thinking
- Compaction: https://platform.claude.com/docs/en/build-with-claude/compaction
- Context editing: https://platform.claude.com/docs/en/build-with-claude/context-editing
- Fine-grained tool streaming: https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming
- Computer use tool: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
- Anthropic, "Claude plays robotics" (2026-07-09): https://www.anthropic.com/research/claude-plays-robotics
- OpenAI GPT-6 Astra model page: https://developers.openai.com/api/docs/models/gpt-6-astra
- Gemini Robotics-ER docs: https://ai.google.dev/gemini-api/docs/robotics-overview
- Gemini Robotics ER 2 blog (2026-07-30): https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/
- Thinking Machines, Defeating Nondeterminism (2025-09-10): https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- RealSense D435 specs: https://www.realsenseai.com/products/stereo-depth-camera-d435/
- Robotiq on ISO 10218-2:2025 Annex M: https://blog.robotiq.com/knowledge/compliance-of-the-hand-e-gripper-with-iso-10218-22025
- GRID "Agentic Architectures for Robotics" PDF: https://genrobo.github.io/Agentic-Robotics/paper.pdf

**Repos (cloned under `research/repos/`)**
- robocurve/llm-token-speed (Artificial Analysis snapshot 2026-08-20): https://github.com/robocurve/llm-token-speed
- robocurve/clapboardbench (wire logs `2026-07-31/wire/*/scene-0-e0/calls.jsonl`): https://github.com/robocurve/clapboardbench
- RoboProbe `results/l3_inspect_eef_official_2100/{README.md,efficiency.json}`: https://github.com/RoboProbe/RoboProbe
- robocurve/inspect-robots `src/inspect_robots/approver.py`: https://github.com/robocurve/inspect-robots
- robocurve/roboharm: https://github.com/robocurve/roboharm

**arXiv**
- VIA 2607.11119: https://arxiv.org/abs/2607.11119
- Show-Harness 2609.10522: https://arxiv.org/abs/2609.10522
- RoboICL 2609.34261: https://arxiv.org/abs/2609.34261
- GPT-6 Astra systematic 2609.38537: https://arxiv.org/abs/2609.38537
- Astra on RoboDojo 2609.24170: https://arxiv.org/abs/2609.24170
- RoboDawn 2609.22966: https://arxiv.org/abs/2609.22966
- URAI 2609.39018: https://arxiv.org/abs/2609.39018
- SafeHarness 2609.20822: https://arxiv.org/abs/2609.20822
- Visual grounding in zero-shot VLC 2608.06154: https://arxiv.org/abs/2608.06154
- AGP 2609.12541: https://arxiv.org/abs/2609.12541
- FAEA 2601.20334: https://arxiv.org/abs/2601.20334
- CaP-X 2603.22435: https://arxiv.org/abs/2603.22435
- VoLo 2606.07723: https://arxiv.org/abs/2606.07723
- Hi-VLA study 2606.10267: https://arxiv.org/abs/2606.10267
- Gemini Robotics 2503.20020: https://arxiv.org/abs/2503.20020
- Gemini Robotics 1.5 2510.03342: https://arxiv.org/abs/2510.03342
- RTC 2506.07339: https://arxiv.org/abs/2506.07339
- Training-time RTC 2512.05964: https://arxiv.org/abs/2512.05964
- VLASH 2512.01031: https://arxiv.org/abs/2512.01031
- Async comparison 2605.08168: https://arxiv.org/abs/2605.08168
- SmolVLA 2506.01844: https://arxiv.org/abs/2506.01844
- VLA-Perf 2602.18397: https://arxiv.org/abs/2602.18397
- TypeGo 2607.05482: https://arxiv.org/abs/2607.05482
- Speculative Actions 2510.04371: https://arxiv.org/abs/2510.04371
- AgenticCache 2604.24039: https://arxiv.org/abs/2604.24039
- Don't Break the Cache 2601.06007: https://arxiv.org/abs/2601.06007
- FailBench 2609.03611: https://arxiv.org/abs/2609.03611
- Code-as-Monitor 2412.04455: https://arxiv.org/abs/2412.04455
- MolmoPoint 2603.28069: https://arxiv.org/abs/2603.28069
- PointArena 2505.09990: https://arxiv.org/abs/2505.09990
- BLINK 2404.12390: https://arxiv.org/abs/2404.12390
- SCAFFOLD 2402.12058: https://arxiv.org/abs/2402.12058
- PIVOT 2402.07872: https://arxiv.org/abs/2402.07872
- RoboPoint 2406.10721: https://arxiv.org/abs/2406.10721
- ViCrop 2502.17422: https://arxiv.org/abs/2502.17422
- PlaneHEC 2507.19851: https://arxiv.org/abs/2507.19851
- Learning-based hand-eye calibration 2311.01335: https://arxiv.org/abs/2311.01335
- Hijacking Robots with a Piece of Paper 2608.05715: https://arxiv.org/abs/2608.05715
- PPIA 2601.17383: https://arxiv.org/abs/2601.17383
- CHAI 2510.00181: https://arxiv.org/abs/2510.00181
- RoboPAIR 2410.13691: https://arxiv.org/abs/2410.13691
- RoboGuard 2503.07885: https://arxiv.org/abs/2503.07885
- ASIMOV 2503.08663: https://arxiv.org/abs/2503.08663
- Modular guardrails 2602.04056: https://arxiv.org/abs/2602.04056
- Harness disclosure 2605.23950: https://arxiv.org/abs/2605.23950
- STEP 2503.10966: https://arxiv.org/abs/2503.10966
- LIBERO-PRO 2510.03827: https://arxiv.org/abs/2510.03827
- LIBERO-Plus 2510.13626: https://arxiv.org/abs/2510.13626
- Veo world simulator 2512.10675: https://arxiv.org/abs/2512.10675
- RoboWorld 2607.01060: https://arxiv.org/abs/2607.01060

**Teammate notes cited for context** (not re-verified by me)
- `research/sources/innate-os-pr817.md`: https://github.com/innate-inc/innate-os/pull/817
- `research/sources/robodojo.md`
- `research/sources/gpt-policy-in-context.md`
- `research/landscape/vla-and-action-heads.md`

---

## Verification (fact-check pass)

*Adversarial re-check done 2026-10-02 against primary sources:*
- *Vendor docs were fetched fresh as `.md` from platform.claude.com, ai.google.dev, developers.openai.com, blog.google and anthropic.com/research.*
- *arXiv HTML full texts were re-fetched and grepped. The colleague's downloads in `tmp_eng/` were not reused.*
- *Repo files in `research/repos/` were checked directly. Wire logs and JSON were recomputed with Python.*
- *Text extracts are in `research/tmp_fc_eng/` (raw HTML deleted because the disk was full).*
- *A claim counts as "confirmed" only if I saw the number in the primary source.*

### A. Confirmed claims (primary source seen)

**Anthropic "Claude plays robotics" (Jul 9, 2026; Berman, Ilie, Deng, Freeman)**
- Latency: "about 2–8 seconds" text-only and "5–15 seconds" with one or two images, without reasoning. For Opus 4.6/4.7 at high reasoning, turns took "15–60 seconds" with "tails of 60–180 seconds". Extra-high reasoning was "about twice as slow".
- "Real-time control would require roughly 83 Hz; current non-reasoning inference runs at ~0.2-0.4 Hz". This is in the locomotion section; see correction 1.
- LIBERO direct control: "from 0 to 5.5%".
- Cursor tool: Mythos Preview "6% to 32%" on the 10-task subset.
- Third-person camera: −3.6 / +5.8 / +10.7 points.
- VLA supervision: "every tested model still performs substantially worse than MolmoAct does on its own". Opus 4.5, Opus 4.6 and Gemini 3.1 beat MolmoAct on the 3 novel tasks.
- Harness: one frame per turn (three allowed); the OpenAI Azure route was excluded because it "silently capped requests at 50 images"; "twelve models across five providers".

**Claude API docs (fetched 2026-10-02)**
- Pricing (`about-claude/pricing`):
  - Opus 5.5 costs $4 input, $5 / $8 for 5-minute / 1-hour cache writes, $0.20 cache read and $20 output; reads are 0.05× base.
  - Fable 5.1 costs $10 / $12.50 / $20 / $0.25 / $50; reads are 0.025× base.
  - Sonnet 5.5 costs $2 / $10. Haiku 4.5 costs $1 / $5.
- Fast mode: `speed:"fast"` with beta header `fast-mode-2026-02-01`. It gives "up to 2.5x higher output tokens per second" and the benefit is "not time to first token (TTFT)". On Opus 5.5 it costs $8 / $40.
- Effort:
  - Opus 5.5 defaults to `medium`. "Adaptive thinking is always on and can't be turned off".
  - `thinking:{type:"disabled"}` returns 400.
  - Per-message effort "preserves the prompt cache".
- Prompt caching:
  - The minimum cacheable prefix is 512 tokens for Fable 5.1, Mythos 5.1, Opus 5.5, Opus 5, Sonnet 5.5, Fable 5 and Mythos 5.
  - The lookback window is 20 blocks. Runs of consecutive `tool_use` or `tool_result` blocks count as one position.
  - The presence or absence of images, `tool_choice`, the thinking configuration and effort all invalidate the cache.
- Vision:
  - Each 28×28 px patch is one visual token.
  - The high-resolution tier ("Claude 4.7 and later models") allows 2576 px / 4784 tokens; the standard tier allows 1568 / 1568.
  - A 1920×1080 image costs 2,691 tokens, or 1,560 tokens after resizing to 1456×819.
  - Above 20 images the per-image limit tightens, so keep each side ≤2000 px.
  - Up to 600 images per request; 32 MB request limit; accuracy suffers on images "under 200 pixels".
- Coordinates:
  - "Claude works best with absolute pixel coordinates… does not work well when you ask for normalized coordinates… between `0` and `1000`".
  - A token-budget resize is "the most common cause of misaligned coordinates".
  - The docs provide `resized_size()` and `"oversized_image": "error"`.
- Thinking:
  - Non-default `temperature`/`top_p`/`top_k` return 400 on Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Mythos Preview, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Sonnet 5.5 and Sonnet 5.
  - Forced tool use returns 400 on Opus 5.5, Sonnet 5.5, Fable 5.1 and Mythos 5.1.
  - Prefill is not allowed while thinking is on.
- Compaction, context and lifecycle:
  - Compaction beta header is `compact-2026-09-04`; it supports background mode and keep-recent-turns.
  - Context editing uses `clear_tool_uses_20250919` with `clear_at_least`.
  - Long-context pricing: "Claude 4.6 and later models … full 1M token context window at standard pricing".
  - Haiku 4.5 retires "Not sooner than October 15, 2026". Dateless IDs are "pinned snapshot[s]".
- Fine-grained tool streaming: `eager_input_streaming: true` per tool. "you might receive partial or invalid JSON".
- Computer-use classifiers "scan what the tools return, such as screenshots, to flag potential prompt injections".

**OpenAI GPT-6 Astra model page**
- $10 input, $1.00 cached, $12.50 cache writes, $50 output.
- ">272K input tokens are priced at 2x input and cache rates and 1.5x output".
- 1,050,000-token context.

**Gemini Robotics-ER docs and ER 2 blog (Jul 30, 2026)**
- Points are `[y, x]` "normalized to 0-1000".
- Docs advise "Crop or zoom into small or unclear objects" and "Query multiple times and average results for high-precision tasks".
- Input limit is 131,072 tokens.
- `gemini-robotics-er-2-streaming-preview` runs over the Live API. It does not support caching or structured outputs (nor the Batch API).
- ER 2 scores 57.4% on progress classification and 91.3% on moment-finding (0.96 s mean absolute distance).

**robocurve/llm-token-speed (`data/aa_api_models.json`, commit dated 2026-08-20)**
- Every TTFAT and output-tokens/s value in the §1.2 table matches:
  - Haiku 4.5 non-reasoning: 0.693 s / 101.1.
  - Opus 5 low → max: 3.18 / 6.91 / 10.32 / 26.69 / 36.98 s.
  - Sonnet 5 low / high / xhigh: 1.50 / 6.61 / 20.92 s.
  - Gemini 3.7 Flash low: 1.30 s / 301.9.
  - GPT-5.6 Sol xhigh: 31.32 s / 64.5.
- Opus 5.5 is absent from the snapshot.
- Caveat: the snapshot measures 1,000-token text prompts with `parallel_queries: 1`.

**clapboardbench (`2026-07-31/wire/*/scene-0-e0/calls.jsonl`, recomputed)**
- Six `claude-opus-5` runs used `effort:"high"` with adaptive thinking, 263 calls in total (65 + 16 + 19 + 99 + 38 + 26).
- Opus latency: per-run median 10.8–14.4 s, p90 17.9–25.1 s. Mean output tokens per call 1.74–2.78k. Cache-hit rate 69.4–94.2%.
- Gemini ER 2: per-run median 4.6–6.1 s over 5 runs.
- Outcomes (runs/README.md): Opus 2/6 pass; Gemini 0/5, with gemini-4 = error.
- opus-1 at Opus 5 prices costs $1.975.

**RoboProbe `results/l3_inspect_eef_official_2100/`**
- Success: 22.48% for Astra vs 0.88% for GPT-5.5 (micro 472 vs 16 successes out of 2,100).
- Astra per dimension: Precision 4.0%, Memory 38.7%, Generalization 30.5%, Open 31.0%.
- Stops: `give_up` ∧ success = 0/928; P(success | env_end) = 47.0%; GPT-5.5 had 364 `policy_error` and 859 episodes with no stop decision.
- Cache-hit rate: 9.0% vs 77.2%.
- Effort per episode: 60.4 calls/episode and 1.42 M tokens/episode for Astra.
- Mean latency [derived]: 1,740,592 s / 118,767 calls = 14.66 s; 2,510,527 s / 131,159 calls = 19.14 s.
- `eval.sh` defaults are `L3_INSPECT_KEEP_ALL_IMAGES=0` and `L3_INSPECT_IMAGE_HORIZON=2`.

**arXiv papers (HTML full text)**
- RoboICL 2609.34261:
  - Per-call latency of 29–47 s [derived from Table 4 API-min/calls]. Wall time is 19–25% above API time.
  - Cache-hit rate 92.4% over 1,810 calls. First-call rate 25.9% (0-shot) and 71.1% (1-shot).
  - Jev gate on General Pickup: calls 116→60, tokens 4.95→1.39 M, wall time −36.1%, success 4/5→5/5. Align Blocks went 3/5→2/5. The gate adds about 0.86 s.
  - The paper says the gate "was selected on these layouts".
  - The 640×480 triptych tile is confirmed in code: `roboicl/policy/head_camera_prompt.py:55`.
- GPT-6 Astra systematic study 2609.38537: 250 calls with a mean of 39.86 s, physics paused. RoboCasa Hybrid "records more model requests than Direct". The conservative harness "compacts context at 96,000 tokens".
- VoLo 2606.07723:
  - Simulation: 12.57 / 17.76 / 34.97 / 41.80%.
  - Real robot (14 tasks × 3 trials): 14.3 / 45.2 / 40.5 / 42.9%.
  - Cloud VLM latency "∼1–5 s". "Completion-monitor errors dominate every backend". The orchestrator is Claude Opus 4.6.
- URAI 2609.39018: 18.0%→53.0%; a program written in advance reached 24% vs 56%.
- VIA 2607.11119:
  - Minimal-prompt success: 70 / 88 / 60 / 62% for CC-Opus 4.8, CC-Fable 5, Codex-5.5 and Codex-5.6-Sol.
  - Cost per success: $9.5 / $15.1 / $7.2 / $4.1. Tool calls per success: 21–217.
  - T-block 10–40% at "about 8 mm" tolerance. LIBERO-Goal 29/30. Real robots listed as future work.
  - All agents ran at `xhigh`. There are 10 seeds per task.
- Show-Harness 2609.10522:
  - Gemini-3.1 Pro at medium effort; 2 cm fine and 4 cm coarse steps.
  - Visual-prompt plugin: 40%→85% (20 trials). More than 98% of outputs were valid action units.
  - GPT-5.6-sol wall-clock up to 3.4×.
- FailBench 2609.03611: 2,197 attempts from 14 sources, 13 detectors. Gemini 3 Flash scored 0.77 macro balanced accuracy. Contact-heavy assembly was <0.60. Cropping added +2.4 pp.
- RoboDawn 2609.22966: 53.2→73.6% (π0.5 46.0%); RoboDojo 35.67→47.17%; real Franka on 2 tasks.
- AGP 2609.12541: 7/8 configurations ≥80%. Sequential towel folding took 50.8 min and $24.14. Tokens 6.28→5.56 M (−11.4%) with saved experience.
- SafeHarness 2609.20822: 71.9% success and 87.5% collision avoidance, 2.3× and 1.5× the no-harness agent. Collision means >1 mm displacement. Backbones were GPT-5.5 and GPT-6-astra.
- Prompt injection 2608.05715: 5,670 trials; 27.0 / 29.4 / 5.0% success; 99.9% acknowledgment; defenses 75–100 / 85–100 / 100%.
- 2608.06154: 32,874 scored calls; "constant-SLOW policy outperforms a scripted geometric controller".
- Harness disclosure 2605.23950: Terminal-Bench 2 69.7→77.0% (7.3 pp); "up to 15 percentage points" on SWE-bench Verified; HAL "up to nearly 48 percentage points" (cited secondhand from HAL).
- Gemini Robotics 1.5 2510.03342:
  - Total failure rate 44.5% vs 22%; success-detection failures 6% vs 4%.
  - Success detection was run at 5 Hz with simulated latency, where "stale success predictions quickly become irrelevant".
  - GR-ER 1.5 vs GPT-5: Point-Bench 71.6 vs 43.6; Where2Place 59.0 vs 37.0; RoboSpatial 31.1 vs 19.0; spatial average 52.6 vs 30.8.
- Latency-hiding and async papers:
  - Gemini Robotics 2503.20020: backbone "under 160ms", about 250 ms end to end, 50 Hz.
  - RTC 2506.07339: ">300 milliseconds… >30% of… horizon"; 20% faster.
  - Training-time RTC 2512.05964: 200 ms on a 50 Hz robot; 108 ms measured.
  - VLASH 2512.01031: 11.8×; 1.5–2.0×.
  - Async comparison 2605.08168: A2C2 >90% up to d=8; IT-RTC degrades at H=30.
  - TypeGo 2607.05482: −50% per-step delay, −73% time to first action.
  - Speculative Actions 2510.04371: 55% accuracy, 20% speedup.
  - AgenticCache 2604.24039: +22% success, −65% latency, −50% tokens.
  - Don't Break the Cache 2601.06007: 41–80% cost, 13–31% TTFT.
  - VLA-Perf 2602.18397, Takeaway 15: "10 Hz is feasible with good networking, while achieving 100 Hz generally requires asynchronous inference".
  - Hi-VLA 2606.10267: "moderate horizon, e.g., 4-8 seconds"; memory window 3 without summarization.
- Other confirmed numbers:
  - Code-as-Monitor: +28.7% success, −31.8% execution time.
  - SCAFFOLD: +9.6.
  - PointArena: GPT-4o 29.5 → 26.6 with chain-of-thought, 30.4 with two-shot examples.
  - MolmoPoint: 70.7%.
  - BLINK: 51.26% vs 95.70%.
  - RoboPoint: +21.8% / +30.5%.
  - FAEA: 84.9 / 85.7 / 96% with privileged state.
  - RoboPAIR: "often achieving 100%"; GPT-3.5 Go2.
  - RoboGuard: >92% → <3%.
  - ASIMOV: 84.3%.
  - PPIA: up to 98% on 10 LVLMs.
  - STEP: up to 32%.
  - LIBERO-PRO: 0.0%. LIBERO-Plus: 95% → below 30%.
  - Veo: 1,600+ real evaluations. RoboWorld: r = 0.989.
  - Hand-eye calibration: 0.930 mm / 0.265°; PlaneHEC 3.35 mm / 0.13° at 30 samples.

**Other sources**
- GRID PDF (Table 2): one-way latency LAN 1.18–2.55 ms, US-West 3.66–8.39 ms, US-East 35.9–74.8 ms. OWLv2 on H100: 215.8–230.0 ms RTT.
- D435 page: "<2% at 2 m", RGB 69°×42°, 1920×1080.
- Robotiq: ISO 10218-2:2025 "incorporates the former ISO/TS 15066"; Annex M is "Informative"; hands/fingers limit is 140 N.
- Thinking Machines: Qwen3-235B, 80 unique completions out of 1,000 at T = 0, first divergence at token 103.
- `inspect_robots/approver.py`: `ClampApprover` (:74), `DeltaLimitApprover` (:141, `rewind_reference` :248), `ChainApprover` (:311), `SafetyAbort`.
- RoboHarm: 5 tasks, a "target" of 20 rollouts per task per model, no published results.

**Arithmetic re-checked**
- Wilson intervals: 2/6 → 9.7–70.0%; 4/5 → 37.6–96.4%; 19/20 → 76.4–99.1%; 472/2100 → 20.7–24.3%.
- Two-proportion sample sizes: 93.0 per arm (50 vs 70%) and 199.0 (80 vs 90%).
- Pixel footprint at D435 69° HFOV: 0.358 / 0.716 mm/px at 0.5 / 1 m on 1920 px.
- Image tokens: 3 × 1280×720 = 3,588 tokens, so about 18.1 M input tokens at n = 100.
- Every cell of the §7.2 cost table reproduces from `tmp_eng/cost_model.py`.

### B. Corrections (10; fixed inline above with ~~ ~~ / [corrected: …])

1. **TL;DR 1: "Manipulation needs 10–83 Hz".** The 83 Hz figure is Anthropic's *locomotion* number. The same page says paused physics loops ran at "10–125 Hz", and "The robot arm did not require the same tight real-time stability as locomotion, so we did not pause the simulator… in the manipulation setting". (anthropic.com/research/claude-plays-robotics, Latency appendix)
2. **TL;DR 2: "RoboDojo-L3/Astra had a 9% hit rate" given as evidence that sliding windows break caching.** The same K=2 window gave GPT-5.5 a 77.2% hit rate, so Astra's 9.0% is provider/model-specific. (RoboProbe `efficiency.json`: `cache_hit_rate` 0.090 vs 0.772)
3. **innate PR #817 "3.4–8.5 s".** Correct value: 3.2–8.5 s, median ≈4.3 s. The teammate note was itself corrected: `research/sources/innate-os-pr817.md:27,332`.
4. **Gemini ER 2 streaming "sub-second latency".** The blog attaches "sub-second latency" to moment-finding at "4x the execution speed". The streaming/Live API claim is only "without the jarring 'stop-and-think' pauses".
5. **SmolVLA async "Same success".** Async success is 73.3% vs 78.3% synchronous on average; sorting is 50% vs 70% (2506.01844, Fig. 5a).
6. **URAI "1.5–1.7× fewer execution-agent calls".** It is 1.5–1.7× fewer execution-agent *output tokens*; DeepSeek-V4-Flash saves only about 5% of time and 9% of tokens (2609.39018, abstract and §4).
7. **Cache invalidation "except Opus 5.5's per-message effort".** Per-message effort (beta `mid-conversation-output-config-2026-07-01`) exists on Fable 5.1, Mythos 5.1, Opus 5.5, Opus 5 and Sonnet 5.5. The cache is kept when the change is carried in a `role:"system"` message inside `messages` (effort and prompt-caching docs).
8. **"Third-person camera added (Anthropic)".** The chase camera *replaced* the forward view, in the Go2 high-level *locomotion* suite. The manipulation aids were depth, segmentation and cursor.
9. **§10 "Paused physics… the Anthropic study".** Anthropic paused physics only for direct/code control in classic control and locomotion; manipulation was not paused (see 1).
10. **§10 "Show-Harness Franka (20 trials/scenario)".** The main experiments use "10 trials per task"; 20 trials apply only to the two plugin scenarios (2609.10522 §5.1 and §5.4.3).

### C. Unverifiable or only partially verified

- **"12 frontier models" for the LIBERO direct 0–5.5% result.** The page says twelve models were evaluated across the *whole* study. Whether all twelve ran LIBERO-40 direct is shown only in charts. UNVERIFIED.
- **Show-Harness multi-view 96% vs 68% global-view and 58% without proprioception.** These are figure-only (Fig. 9). The text confirms only the qualitative statement and the 96% default. UNVERIFIED.
- **Gemini ER 2 "(5 buckets)" and the third-party Opus 5 / GPT-5.6 Sol comparisons.** These are not in the blog text (charts only). UNVERIFIED.
- **MolmoPoint "SOTA 70.7% on PointBench" vs GR-ER 1.5's Point-Bench 71.6 (73.3 without thinking).** These are probably different Point-Bench variants or splits. Do not compare them directly.
- **HAL "nearly 48 pp".** Cited secondhand by 2605.23950; the HAL primary source was not opened.
- **GPT-as-Policy 22.65 M tokens and 97–98% cache-hit rate.** Confirmed via RoboICL §4.4 and teammate note `sources/gpt-as-policy.md` (624.8 M / 1.132 B tokens over 50 episodes). The authors' issue reply was not re-opened.
- **Thinking Machines 80/1,000.** Measured on self-hosted vLLM, not on a commercial API. That API serving is *at least* as nondeterministic is plausible, but no source here measures it.
- **"200 Hz–1 kHz below it" (TL;DR 1) and "0.03–0.5 Hz" for the API.** These are the author's framing, not from a cited source.

### D. Caveats on confirmed claims (interpretation risks)

- **clapboardbench "Opus 5: 2/6 (95% CI 10–70%)" is not an unbiased success rate.** The README says the 11 runs were "hand-picked from a 24-hour capture window", and `opus-early` was included as "the operator-judged pass". Treat it as a latency/trace dataset, not an outcome estimate.
- **FailBench's detector panel contains no Claude and no GPT-5.x/6 model.** The panel is Gemini 3 Flash, Gemma-4-31B, GPT-4o, Qwen3-VL ×3, two embodied models and five purpose-built detectors. "Self-verification is weak" is therefore not measured for Opus 5.5 / Astra-class models. On *micro* overall, Gemma-4-31B-it (0.76) beats Gemini 3 Flash (0.74).
- **2608.06154 does not test manipulation or frontier APIs.** Its embodiments are a quadrotor (gym-pybullet-drones) and a highway-env ground vehicle. Its models are SmolVLM, Qwen2-VL-2B/7B, LLaVA-1.5-7B, Qwen2.5-VL-72B, MiniCPM-V-4.5, Cosmos3, Kimi-K2.6 and Kimi-K3. The advice to run blind and mirrored ablations stands; the "image-invariant" finding may not transfer.
- **The §7.2 re-costing applies Claude prices to GPT-tokenized counts.** Claude 4.7+ uses a tokenizer that "produces approximately 30% more tokens for the same text", and image token formulas differ by provider. Treat Claude-column costs as roughly ±30%.
- **Small-n:**
  - VIA: 10 seeds per task.
  - URAI: 25 episodes per agent (5 tasks × 5 seeds).
  - VoLo real robot: 42 rollouts per system with overlapping Wilson CIs. The paper says "Reaching statistical power to compare the ablations requires a larger" sample.

### E. Missed details within scope (added)

1. **Sonnet 5.5 can skip up-front thinking.**
   - `thinking:{"type":"between_tools"}` is "the lowest thinking setting on Claude Sonnet 5.5" and works at `low`/`medium`/`high`. It returns 400 at `xhigh`/`max`, and effort then cannot change mid-conversation.
   - This is the main latency lever that Opus 5.5 lacks (effort docs).
   - Sonnet 5.5 defaults to `high`, not `medium`.
2. **Anthropic study: reasoning budget barely matters, and context can be truncated.**
   - Reasoning budget: on high-level locomotion, Opus 4.6 stayed within a 2.6-point band (37.8–40.4) across no-reasoning, a 20k budget and adaptive-max. "adaptive-low" was the consistent loser. Only Mythos Preview gained from reasoning (40.2 → 54.1).
   - Context truncation (keep the first 10 turns plus the last N) did not significantly hurt any model except Opus 4.6. "models rely much more on the recent past than on a broad accumulated understanding". Dropping turn 1 made models "forget basic conventions and loop".
   - Implication for the harness: a short rolling window plus a pinned header turn is viable. It conflicts with caching unless compaction is batched.
3. **Anthropic real-world safety vignette.** On a real Go2, Grok 4.1 Fast "saw the target table in the reflection of the glass door and started charging for the glass door" and was stopped by an operator. This argues for geometric/depth safety layers independent of the VLM.
4. **URAI per-agent results.**
   - Success with URAI vs direct: Claude Opus 5.5 60% vs 32%, the best of four agents. Fable 5.1 52 vs 16%, GPT-6 Astra 52 vs 16%, DeepSeek-V4-Flash 48 vs 8%.
   - Paired comparison: McNemar p = 7.9×10⁻⁸; 40 pairs succeed only with URAI vs 5 only with native control.
   - This is the most direct 2026 evidence that Opus 5.5 works as the decision layer over local multi-phase tools.
5. **AGP model comparison (two-pair assembly, n = 5 each).**
   - GPT-6 Astra: 5/5 at low/medium/high effort, 9.2–9.9 min, $4.09–4.79.
   - Claude Opus 5 (Claude Code, high): 5/5, 22.2 min, 12.40 M tokens, $9.75.
   - Claude Fable 5.1: 3/5, 27.7 min, $11.30.
   - GPT-5.6 Sol: 5/5, 14.2 min. Terra: 1/5. Luna: 0/5.
6. **VoLo VLM ablation (simulation, overall success).** Claude Opus 4.6 41.80, GPT-5.5 35.52, Gemini-2.5-Flash 31.97, Qwen3-VL-8B 19.95. Opus 4.6 had the fewest VLM errors (5% of ceiling vs 23% for Qwen3-VL-8B).
7. **Controlled harness-vs-model grid (2605.23950).** Switching harness H1→H3 at a fixed model moves pass@1 by 8.5–13.0 pp. Switching model at a fixed harness moves it by 2.5–5.0 pp. The grid shows six ranking reversals in nine comparisons.
8. **GR-ER 1.5 table updates the stale BLINK figure.**
   - GPT-5 scores 71.3 on BLINK (Sep 2025), against the 2024 GPT-4V 51.26% cited in §3.3. GPT-5 also leads the embodied-QA average (71.4 vs 66.5).
   - GR-ER 1.5 points *better without thinking*: Point-Bench 73.3 vs 71.6. This agrees with PointArena's finding that chain-of-thought hurts pointing.
   - Implication: frontier API models are now competitive on spatial QA but still weak at pixel pointing.
9. **Conflicting in-context-learning evidence on RoboDojo.** 2609.24170 (official Astra evaluation) finds "no aggregate benefit from one-shot demonstrations". RoboICL (+20–27 points) and RoboDawn (+11.5 pp) find large gains with differently structured demonstration contexts. Harness design drives the ICL effect. DeepSeek-Flash scored 1.92% under the same protocol at 10 episodes per task.
10. **RoboDojo L3 cost asymmetry.**
    - The call budget is `L3_INSPECT_MAX_LLM_CALLS` = 170.
    - API calls per success: Astra 252 vs GPT-5.5 8,197.
    - Output tokens per episode: 7,882 vs 31,241, of which reasoning is 4,143 vs 26,429.
    - Reasoning tokens, not input, drive the cost gap.
11. **Astra study resource details.** RoboDojo Hybrid (with π0.5) cut recorded tokens by 44.8% vs Direct. Over 50 instances, policy-assisted control used 624.8 M tokens and direct control 1.132 B. A PASSAGE planner call takes about 0.08 s vs 39.86 s for an Astra call.
12. **More RoboICL Jev results.** On Align Blocks, Astra requests fell 181→121 and tokens 9.53→5.08 M, while success fell from 3/5 to 2/5. On General Pickup layout 3, fewer calls coincided with *more* robot steps (103 vs 75).
13. **Caching and context tooling not mentioned.**
    - "Cache diagnostics" has the API compare consecutive requests and report where the prefix diverged. Use it to debug silent misses in image-heavy loops.
    - Context-editing tool-result clearing "invalidates cached prompt prefixes"; use `clear_at_least` so each invalidation is worth its rewrite.
    - `clear_thinking_20251015` also exists.
14. **Fast mode access limits.** Fast mode is a *research preview* (waitlist or account manager), available on the Claude API only (not Bedrock, Vertex, Foundry or Claude Platform on AWS). It has separate rate limits (`anthropic-fast-*` headers; 429 with `retry-after`) and is unavailable with Batch. Do not design a harness that depends on it.
15. **Request and model limits.**
    - Image cap: 600 images per request, but only 100 for 200k-context models such as Haiku 4.5.
    - Gemini ER 1.6 was "shut down at the end of August" 2026. ER 2 streaming supports function calling but not structured outputs, caching or Batch.
16. **Sensor detail.** The D435 RGB sensor is a *rolling shutter* and depth Min-Z is about 28 cm. Show-Harness pairs an exocentric D435 with a wrist D405 for close range. Rolling shutter plus motion between multi-second LLM calls argues for capturing frames only when the arm is stationary.
