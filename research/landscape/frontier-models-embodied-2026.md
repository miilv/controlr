# Frontier LLM/VLM APIs as robot brains: model-level landscape (as of 2026-10-01)

Scope: which closed and open models people use in 2025-2026 to drive robots, what they can do spatially, the numbers that exist, and what each API exposes (price, image limits, latency, streaming, tool calling). This note is for an "LLM API backbone (Claude Opus class) + optional light learned action head" harness. Harness-level deep dives (Robocurve, RoboDojo, GPT-as-Policy, piper-astra-jev, quackd and others) are in `research/sources/*.md`. Here they appear only as model-level evidence.

Conventions. Every 2025-2026 number below comes from a page I opened on 2026-10-01; the URLs are in Sources. "Vendor" means the model's own maker reported the number. "Indep." means a third party measured it. **UNVERIFIED** marks anything I could not confirm at a primary source. Two caveats apply throughout. Effort or reasoning settings and the harness change results a lot, often more than the choice of model. And most closed-loop robot numbers come from 20-75 trials, so differences of 10-20 points between models fall within noise.

---

## 0. TL;DR for the harness design

1. **In closed-loop direct control, GPT-6 Astra leads clearly in every independent head-to-head I found.** Claude Opus 5 and Fable 5.1 usually come second. Sol, Gemini Flash, Grok, Kimi, Qwen and DeepSeek trail well behind. The figures:
   - LIBERO-Agent Performance Score: Astra 45.0, Opus 5 19.3, Fable 5.1 15.8.
   - CodeActionBench: Astra 73.3%, Opus 5 49.3%, Gemini 3.6 Flash 20.0%, Grok 4.6 12.0%, Sonnet 5 2.7%.
   - Robocurve block-in-bowl: 19/20 vs 8/20.
   - DrivingBench: Astra was the only model to finish the course. Fable 5.1 reached 45%.

   No independent closed-loop robot result yet exists for Opus 5.5 (released 2026-09-22) or Sonnet 5.5 (2026-09-28).
2. **Claude is competitive or better on the "System 2" sub-skills.**
   - Writing code for robots. On generalized TAMP, Claude Code with Opus 5 reached 82% vs the 47% planner baseline (Astra 95%, Sol 56%).
   - Success detection. On DeepMind's own chart, Opus 5's video success detection is 81.0%, second only to ER 2's 82.4% and ahead of GPT-5.6 Sol's 74.7%.
   - Following safety instructions: Opus 5 95.9% vs Sol 91.4%.
   - Claude is weaker at precise pointing. In an independent low-effort evaluation, RefSpatial was Opus 5 56.3 vs Astra 78.0 and Gemini 3.6 Flash 76.4. Opus 5 also scored anomalously low on video spatial reasoning (VSI-Bench 21.3).

   Anthropic's own docs state that Claude's "coordinate and localization outputs are approximate" and that it "does not work well" with normalized coordinates.
3. **Google is the only frontier lab shipping a robotics-specific API model.**
   - Gemini Robotics ER 2: `gemini-robotics-er-2-preview`, built on Gemini 3.5 Flash. It costs $1/$5 per 1M tokens in 2026.
   - It has a Live-API streaming twin, `gemini-robotics-er-2-streaming-preview`.
   - It outputs points natively in `[y, x]` 0-1000 format, plus boxes, trajectories, video progress and success detection.

   That makes it the obvious candidate for a perception or verification co-processor next to an Opus planner. Neither OpenAI's nor Anthropic's frontier models expose a realtime video endpoint: `gpt-6-astra` does not support Realtime or Live, and I found no Claude realtime endpoint. [fact-check: correct for Astra, but OpenAI's non-frontier Realtime models do take images over the Realtime API. `gpt-realtime-2.1` lists input modalities "text, audio, image" and costs $5/1M image input; `gpt-realtime-2.1-mini` costs $0.80/1M image input (model card and pricing page). Google's `gemini-3.8-live` takes image/video at $1.00/1M or $0.002/min.]
4. **Latency is the binding constraint.** Frontier calls take about 3-40 s per decision in practice. Measured figures:
   - Opus 5 time-to-first-token is 3.2 s at low effort and 10.3 s at high.
   - Astra's measured mean was 14.7 s per call on RoboDojo, and DrivingBench recorded 4.9 s median per command.
   - Fable 5.1 recorded 22.2 s median per command in DrivingBench.
   - Anthropic's Embody study reports about 0.2-0.4 Hz, against roughly 83 Hz needed for real-time control.

   This forces the architecture into a fast layer (servo/IK, VLA or decision model) plus a slow LLM layer.
5. **Tools and perception move results more than switching models does.**
   - Robo-Harness K1's depth, anchor and grasp tools took Astra from 61.1% to 88.9% on LIBERO-PRO, and let Gemini 3.7 Flash reach 77.8%.
   - A cursor tool raised Mythos Preview from 6% to 32% on a LIBERO subset.
6. **The new "decision models" are a cheap, text-only fast path for discrete gating, not perception.** Jev (TypeSafe), Laya (Convai) and Kev (Jared Palmer) all expose `/v1/systemone` with noul, choice and score questions. Jev takes 70-500 ms and costs $0.042 per 1M input tokens. Laya takes about 33-40 ms on a T4, and Kev-4B about 18 ms on an H100. [fact-check: these are not like-for-like. Kev's 18.1 ms is server-side model time for 6 questions about a new short text, excluding a ~65 ms network round trip (kev README "Serving Performance"). Jev's 70-500 ms is vendor end-to-end time, and independent benchmarks cited in the Laya README measured Jev at a p50 of 236-276 ms.]

---

## 1. Closed frontier general models

### 1.1 OpenAI GPT-6 Astra (`gpt-6-astra`), the current robotics outlier

**Release.**
- Unveiled 2026-09-03 as a limited preview; ~~the API and paid tiers followed on 2026-09-04~~ [corrected: the OpenAI API changelog (developers.openai.com/api/docs/changelog, "### Sep 3") lists `gpt-6-astra` as released in the API (v1/responses, v1/chat/completions) on 2026-09-03; per Wikipedia's GPT-6 article, the public release to paid ChatGPT users followed on 2026-09-04] (community announcement thread; Wikipedia).
- The official launch post at openai.com returned a Cloudflare challenge and 403 from here, so it is **unreachable/UNVERIFIED**.
- Press reports of a "recurrent depth"/looped architecture are **UNVERIFIED**; the system card does not discuss architecture.

**API surface** (developers.openai.com model card):
- Context and output: 1.05M context, 922K max input, 128K output, knowledge cutoff 2026-04-30.
- Reasoning effort: `low|medium|high|xhigh|max`. `none` is rejected ("use `low` instead").
- Endpoints: Responses, Chat Completions and Batch. **Realtime and Live are not supported.**
- Tool calling "requires Responses". Async tools (`async: true`) and mid-turn steering over WebSocket exist (latest-model guide).
- Price: $10 input, $1 cached, $12.50 cache write and $50 output per 1M tokens. Above 272K input the rate is 2x input and 1.5x output. Batch and Flex are half price. Fast mode is 2x, ~~and an "Ultrafast" mode also exists with "no latency SLA"~~ [corrected: the "does not include a latency SLA" sentence in the latest-model guide refers to *Fast mode* for GPT-6 Astra. Ultrafast is a separate service tier (`service_tier: "ultrafast"`, Responses API, added 2026-09-29 per the API changelog). It costs $60 input, $6 cached, $75 cache write and $300 output per 1M tokens at ≤272K (6x standard), runs at low rate limits, supports US/global processing only, and OpenAI recommends using it over WebSockets (ultrafast-mode guide; pricing page)].
- Built-in tools include `computer_use`, `mcp` and `code_interpreter`.

**Images** (images-vision guide):
- Patches are 32 px with a 1.2 multiplier. A 1024² image costs 1,229 tokens at `high`.
- `high` has a 2,500-patch budget. `original` keeps native resolution up to 65,535 px per side, and `auto` behaves as `original` on Astra.
- Up to 1,500 images per request, 512 MB payload, 30,000 patches per image.
- The guide still warns: "The model struggles with tasks requiring precise spatial localization."

**Embodied-specific features: none official.**
- The system card (deploymentsafety.openai.com) contains no robotics, embodied or spatial evaluations.
- I found no OpenAI robotics guide or cookbook.
- All robotics evidence is third-party.

**Third-party results.** All are indep.; where a number comes from the authors of the work being evaluated, it is author-reported.

| Setting | Result | Source |
|---|---|---|
| Robocurve, bimanual YAM, EEF `move_to`, medium effort, 20 trials/task | Bowl: 19/20 (Fable 5.1 8/20, Fable 5 1/20). Puzzle: 2/20 (Fable 5.1 2/20). Astra $0.94/run and 2.5 min/trial vs Fable 5.1 $2.12 and 6.8 min. Not interleaved; graded by an operator who knew the model. | openai.robocurve.org |
| StationeryBench (Robocurve), 5 bimanual tasks × 20 trials | Astra completed 7/100; mean progress 46/100 vs MolmoAct2 (zero-shot) 0/100 and 12/100 | openai.robocurve.org/stationerybench |
| RoboDojo, all 42 sim tasks, 2,100 trials, single seed | 22.48% SR / 28.97 Score. GPT-5.5 0.88%, DeepSeek-Flash 1.92%. Strong on semantic tasks, about 4% on Precision tasks | arXiv 2609.24170 |
| GPT-as-Policy (π0.5 proposes, Astra gates/edits) | Hybrid 48% on a RoboDojo subset. A 30 s locomotion run needed 250 calls averaging 39.86 s, with physics paused | arXiv 2609.38537 |
| LIBERO-Agent, 30 tasks × 3 rollouts, high effort, vendor harnesses | Score 45.0 vs Opus 5 19.3, Fable 5.1 15.8, Sol 10.8, Qwen3.8-Max 10.5, Kimi K3 5.5, DeepSeek-V4.1-Flash 4.5. Astra used 1.38M tokens and 11.5 min per episode | arXiv 2609.39507 |
| CodeActionBench, 25 tasks × 3 attempts | Astra (Codex CLI) 55/75 = 73.3%, 22/25 tasks covered. Correct final self-assessment in 68/75 | arXiv 2609.33807 |
| DrivingBench, real Toyota Corolla, MCP tools | Only Astra finished, on its 2nd attempt, in 322 s. Median 4.9 s per command. It refused to drive under several framings | arXiv 2609.38948 |
| Generalized TAMP, Codex | Astra 95% vs Opus 5 (Claude Code) 82%, Sol 56%, planners 47% | arXiv 2609.30233 |
| VLN-CE R2R-CE-100 | 81.3% SR at "ultra" reasoning | arXiv 2609.29861 |
| Robo-Harness K1 (LIBERO-PRO) | RGB-only 61.1% → 88.9% with K1 perception tools | arXiv 2609.29389 |
| Grounding (34 benchmarks) | Astra 71.54 vs GroundingPI-4B 73.68 | arXiv 2609.39601 |

**Measured speed** (Artificial Analysis): 43-51 output tokens/s; time to first answer token 2.96 s at low effort.

**Relevance.** Astra is the strongest proof that a frontier API model can close the loop at the end-effector level. Its failure modes are consistent across studies: precision and contact, dynamic control, latency, and refusals when "real physical stakes" are emphasized (DrivingBench §6).

### 1.2 Other OpenAI models: GPT-5.6 Sol, GPT-6 Sol/Luna, GPT-6.1 Sol, GPT-Live-1

- **GPT-5.6 Sol** (2026-07-09) is the usual "previous-generation" baseline.
  - It is weak at direct control: LIBERO-Agent 10.8, CodeActionBench 16%, DrivingBench at most 6% of the course.
  - It scores 43.2% on ERQA in DeepMind's chart, against 74.7% for video success detection.
- **GPT-6 Sol and Luna** were released 2026-09-22 (Wikipedia). **GPT-6.1 Sol** is documented in the "Using GPT-6" guide with effort `low…max` and default `medium`. I found no robotics data for any of them.
- **`gpt-live-1`** is OpenAI's full-duplex voice model.
  - It uses the `v1/live/sessions` endpoint and costs $0.05/min.
  - It takes **no image or video input**, so it does not provide a video stream for robots.

### 1.3 Anthropic Claude: current lineup and API facts

These come from the docs at platform.claude.com, read 2026-10-01.

| Model (ID) | Released | $/MTok in/out | Context / out | Thinking / default effort | Comparative latency |
|---|---|---|---|---|---|
| Fable 5.1 (`claude-fable-5-1`) | 2026-09-01 | 10 / 50 (cache read 0.25) | 1M / 128K | adaptive, always on / `high` | Slower |
| Opus 5.5 (`claude-opus-5-5`) | 2026-09-22 | 4 / 20 (cache read 0.20); fast mode 8 / 40, "up to 2.5x speed" | 1M / 128K | always on, cannot be disabled / **`medium`** | Moderate |
| Sonnet 5.5 (`claude-sonnet-5-5`) | 2026-09-28 | 2 / 10 | 1M / 128K | adaptive; `between_tools` turns off up-front thinking / `high` | Fast |
| Haiku 4.5 (`claude-haiku-4-5-20251001`) | 2025-10 | 1 / 5 | 200K / 64K | extended (`budget_tokens`) | Fastest |
| Opus 5 (legacy, `claude-opus-5`) | 2026-07 | 5 / 25 | 1M / 128K | adaptive | — |

**Vision.**
- Images are tokenized in 28×28 patches.
- The high-resolution tier (Claude 4.7 and later) allows 2576 px on the long edge and 4,784 visual tokens. Haiku is on the standard tier: 1568 px and 1568 tokens.
- A request can carry 600 images, but above 20 images a stricter limit of about 2000 px per image applies.
- `"transformations": {"oversized_image": "error"}` makes silent server-side resizing an error instead.
- Coordinates: "Claude works best with absolute pixel coordinates… does not work well when you ask for normalized coordinates" (vision-coordinates guide).

**Breaking changes relevant to a control loop** (Opus 5.5 and Fable 5.1):
- Forced `tool_choice` (`any` or `tool`) returns a 400; use `auto` plus `strict: true`.
- Thinking cannot be disabled on Opus 5.5.
- Thinking blocks are bound to the model and the conversation, so harnesses must be append-only.
- ~~Computer use only works through `computer_toolset_20260801`.~~ [corrected: this change applies only to Opus 5.5, and only on the Claude API and Google Cloud. There a `computer_20251124` tool returns a 400, while Amazon Bedrock still accepts `computer_20251124` on Opus 5.5. The Opus 5.5 overview says only the first three breaking changes (thinking can't be disabled, forced tool use returns an error, thinking blocks tied to model and conversation) "also apply on Claude Fable 5.1" (platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5.md).]

**Published vision benchmarks.** The Opus 5.5 launch post lists only Chartography (89.0) and OSWorld 2.1 (81.8, "partial"). It contains no robotics or spatial benchmark.

**Measured speed** (Artificial Analysis snapshot of 2026-08-20, bundled in `robocurve/llm-token-speed`):

| Opus 5 effort | Output speed | Time to first token |
|---|---|---|
| low | ~50 tok/s | 3.18 s |
| medium | ~50 tok/s | 6.91 s |
| high | ~51 tok/s | 10.32 s |
| xhigh | ~53 tok/s | 26.7 s |
| max | ~54 tok/s | 37.0 s |

- Sonnet 5 at low effort: 1.50 s to first token, 62 tok/s.
- Haiku 4.5 non-reasoning: 0.69 s, 101 tok/s.
- **No Opus 5.5 or Sonnet 5.5 speed measurement was available to me (UNVERIFIED).**

**Robot results with Claude.**
- **Robocurve, Fable 5 vs 5.1** (2026-09-03, 80 interleaved trials): bowl 1/20 → 8/20, puzzle 0/20 → 2/20 at medium effort. Fable 5 used up its 20-call budget in 18 of 20 bowl trials.
- **Robocurve, Opus 5 test-time scaling** (2026-08-19). This ran on raw joint targets with no IK, 5 runs per condition and 224×224 images. The mean stacking score rose with effort: low 46, medium 64, high 76, against 60 for Sol at high. Fast mode was used.
- **LIBERO-Agent and CodeActionBench** (see §1.1):
  - Opus 5 is clearly second.
  - Its harness matters: 49.3% with the reference harness vs 45.3% with Claude Code.
  - Sonnet 5 nearly collapsed at 2.7%, mostly by hitting the budget.
  - Opus over-claims success. In 34.2% of its failed attempts it reported success, vs 15.0% for Astra.

**Relevance.**
- Opus 5.5 is the cheapest Opus to date ($4/$20). It ~~also brings~~ [corrected: supports; per-message effort is not new with Opus 5.5. The effort doc lists per-message effort (beta header `mid-conversation-output-config-2026-07-01`) for Fable 5.1, Mythos 5.1, Opus 5.5, Opus 5 and Sonnet 5.5] per-message effort (beta), mid-conversation system messages and server-side refusal fallbacks, all useful for a long-lived control session.
- Expect Claude to be strongest as planner, code-writer, verifier and safety layer.
- Expect it to need external grounding for precise pointing, depth and contact.

### 1.4 Anthropic's own robotics work

- **Project Fetch, phase 1** (2025-11-12). This was an uplift study: 8 staff with no robotics background on a quadruped, one team with Claude and one without.
  - Team Claude finished 7/8 tasks and Team Claude-less 6/8.
  - On the tasks both teams completed, Team Claude took about half the time.
- **Project Fetch, phase 2** (2026-06-18). Opus 4.7 ran alone in Claude Code at max effort, 3 trials.
  - It took 9 min 35 s on the 4 shared tasks, 18.9x faster than Team Claude.
  - It still failed the closed-loop ball fetch: "Claude struggled to capture this subtlety."
- **"Claude plays robotics" / Embody** (2026-07-09, Frontier Red Team).
  - Twelve models from five providers were tested on MuJoCo classic control, G1/Go2 in simulation, a LIBERO-style Franka arm and a real Go2.
  - Four interfaces were compared: direct control, programmatic `controller(obs)->action`, policy supervision (Go2 gait and MolmoAct VLA), and RL supervision via PPO.
  - Composite score: Mythos Preview 0.389 (best), Opus 4 0.115.
  - LIBERO direct control: 0-5.5% success; a cursor tool raised it from 6% to 32%.
  - Every Claude model *supervising* MolmoAct did worse than MolmoAct alone. Mythos overrode the VLA too often, while Opus 4.5 and 4.6 deferred more.
  - Physics was paused between calls in the direct and code modes.
  - Latency was about 2-8 s per turn without reasoning (5-15 s with images) and 15-60 s at high reasoning.
  - The authors' conclusion: "only get traction at the higher-abstraction interfaces".
  - The code is promised at `github.com/safety-research/embody` (**not yet released**).

**Relevance.** Anthropic's own data says the same as third parties: Claude is weak as a low-level controller and good at abstractions and code. Naive "LLM overrides VLA" supervision can *hurt*.

### 1.5 Google Gemini 3.x (general) and Gemini 4 Argon

- **Lineup and prices** (pricing page):
  - Gemini 3.1 Pro Preview: $2/$12.
  - Gemini 3.5 Flash: $1.50/$9.
  - Gemini 3.8 Flash: $0.75/$3.75 until 2026-12-31. It is the newest Flash; the 2026-09-02 launch date comes from search snippets of 9to5Google only, so it is UNVERIFIED. [fact-check: now verified at a primary source. The Gemini API release notes (ai.google.dev/gemini-api/docs/changelog) say "September 2, 2026 Gemini 3.8 Flash generally available (GA)". `gemini-3.8-live` went GA on 2026-09-15.]
  - Live models: `gemini-3.8-live`, which accepts image and video at $1.00 per 1M tokens or $0.002/min.
- **Speed** (Artificial Analysis snapshot): Gemini 3.7 Flash runs at 302-339 tok/s, with 1.3 s to first token at low effort. That is about 6x Opus-class output speed.
- **Embodied numbers.** In an independent evaluation, Gemini 3.6 Flash at *minimal* thinking roughly matches Astra at low effort on 28 embodied benchmarks (73.0 vs 73.3; see §4). On closed-loop agentic manipulation it is weaker: 20.0% on CodeActionBench.
- **Gemini 4 Argon** was announced 2026-09-30 (TechCrunch). Access is restricted to Fairwind cyber partners, there is no model ID, and there is no robotics data.

### 1.6 xAI Grok (Grok 4.6 / 4.7)

- **API** (docs.x.ai): `grok-4.7` and `grok-4.6` both have 500K context and cost $2/$6 per 1M tokens. Images may be up to 20 MiB, jpg/png only, with no stated count limit. Separate realtime voice models exist. The page does not mention robotics.
- **Measured** (Artificial Analysis snapshot): Grok 4.6 at high effort runs at 66 tok/s with **39.8 s to first token**.
- **Robot evidence** (indep.):
  - DrivingBench: 8-11% of the course, 12.5 s median per command.
  - CodeActionBench: 9/75 (12%).
- **Optimus integration.** Claims that Grok serves as Optimus's voice and "System 2" come only from blogs, so they are **UNVERIFIED**. Grok 5 has not shipped as of late September 2026 (secondary sources, **UNVERIFIED**).
- **Relevance.** No reason to prefer Grok as a robot backbone today.

---

## 2. Robotics-specialized API models (Google)

### 2.1 Gemini Robotics-ER lineage: 1.5 (2025-09) → 1.6 (2026-04-14) → ER 2 (2026-07-30)

- **ER 1.5 (2025 baseline).** The Gemini Robotics 1.5 report, Table 19, was accessed in September 2025:

  | Benchmark | GR-ER 1.5 (thinking) | GPT-5 | Gemini 2.5 Pro |
  |---|---|---|---|
  | Point-Bench | 71.6 | 43.6 | 62.7 |
  | RefSpatial | 48.5 | 23.5 | 33.6 |
  | RoboSpatial-Pointing | 31.1 | 19.0 | 8.3 |
  | Where2Place | 59.0 | 37.0 | 37.0 |
  | ERQA | 54.8 | 59.0 | 56.0 |
  | VSI-Bench | 45.8 | 52.9 | 51.1 |
  | Spatial average | 52.6 | 30.8 | 35.4 |

  In 2025, specialized pointing beat general frontier models by about 20 points, while the general models were already better at ERQA and VSI-style question answering.
- **ER 1.6** added instrument reading, built with Boston Dynamics (blog.google). The `gemini-robotics-er-1.6-preview` endpoint was scheduled to shut down at the end of August 2026.
- **ER 2** (docs, model card, deepmind.google). It is "based on Gemini 3.5 Flash". Its two endpoints:

  | | `gemini-robotics-er-2-preview` | `gemini-robotics-er-2-streaming-preview` |
  |---|---|---|
  | Context | 131,072 input / 65,536 output | same |
  | Inputs | text, image, video, audio | same |
  | Supported features | caching, code execution, computer use, function calling, structured outputs, Batch | Live API, bidirectional streaming with function calling |
  | Not supported | Live API | caching, code execution |
  | Price per 1M (paid tier) | $1.00 / $5.00 through 2026-12-31, then $2 / $10 | same |

  - Output conventions: points as `[y, x]` normalized to 0-1000, plus boxes and trajectories, via the `interactions` API with `thinking_level`.
  - The docs recommend `medium` thinking for the latency/quality balance.
  - The docs also recommend "Query multiple times and average results for high-precision tasks."
- **ER 2 vendor-reported comparison** (deepmind.google chart alt-text):

  | Metric | Opus 5 | GPT-5.6 Sol | ER 1.6 | Gemini 3.6 Flash | **ER 2** |
  |---|---|---|---|---|---|
  | Success detection, image | 83.6 | 83.1 | 82.9 | 83.3 | **87.7** |
  | Success detection, video | 81.0 | 74.7 | 76.0 | 75.4 | **82.4** |
  | ERQA | 67.2 | 43.2 | 72.5 | 73.0 | **78.5** |
  | Instrument reading | 53.0 | 61.5 | 52.8 | 52.0 | **65.7** |
  | Progress classification | 37.1 | 46.2 | 42.7 | 43.9 | **57.4** |
  | Safety instruction following | 95.9 | 91.4 | 47.2 | — | **97.9** |
  | Human proximity (1 m) | 77.1 | 83.4 | 51.1 | — | **93.0** |

  - Moment finding: 91.3%, with 0.96 s mean absolute error.
  - Orchestration success, ER 1.6 → ER 2: real VLA 48.6% → 60.0%, sim VLA 37.4% → 42.9%, human tele-op 63.6% → 74.0%.
  - GPT-6 Astra was not in the comparison.
- **Independent checks.**
  - "Hard Vision, Easy Vision" used ER 2 as the *specialist reference* for embodied understanding; Astra beat it by 3.8 points.
  - On RoboAbstention, ER 1.6 abstained on only 16.5% of infeasible or ambiguous instructions, rising to 93.6% with defensive prompting.
- **Relevance.** This is the most "harness-ready" robotics model API: native pointing formats, video success and progress signals, Live streaming, and a design explicitly meant to orchestrate VLAs. It is cheap enough to call at every step as a perception or verification sidecar to Claude.

### 2.2 Gemini Robotics On-Device (2025-06-24) and On-Device 2 (2026-07-30)

- These are on-device VLAs, not ER models. They take text, images and proprioception and output actions.
- On-Device 1 was "based on" Gemini Robotics and adapted with "as few as 50 to 100 demonstrations". An SDK is available to trusted testers.
- On-Device 2 is "based on Gemini Robotics 1.5 technology and our on-device Gemma models". It is distributed only to trusted testers.
- Neither is an API option for a third-party harness today. They show the reference architecture: ER as the brain, an on-device VLA as the action head.

---

## 3. "System One" decision models: Jev, Laya, Kev

All three take text state plus typed questions (`noul` yes/no, `choice`, `score`) and return a probability distribution without generating text. All three speak the same `/v1/systemone` protocol.

| Model | Org / date | What it is | Latency | Price / license | Limits |
|---|---|---|---|---|---|
| **Jev 1.13** (`jev-1.13.0`, `jev-latest`) | TypeSafe AI (founder Diogo Almeida), 2026-09-15 | Hosted "System One" model trained with "RLCD" (calibrated-decision RL); choice cardinality up to 255 | "70ms-500ms" end-to-end (vendor) | $0.042 per 1M input tokens, output free | **Text only**. 64K/request (32K state plus longest question). Docs now say 100K tok/s and 40 req/s; the limits changed after launch |
| **Laya** | Convai Innovations, about 2026-09-18 (date from secondary sources) | ModernBERT-large encoder (395M) plus decision head = 421M. Multilingual mmBERT variant is 322M | T4: 39.5 ms (EN) / 32.8 ms (multilingual) for 1 question; ~~7.2 ms/question at 10 questions~~ [corrected: 7.2 ms/question at 10 questions is the multilingual mmBERT checkpoint (72.3 ms per 10 questions). The English `laya` checkpoint takes 15.9 ms/question (158.6 ms per 10) (HF README)] | Apache-2.0, self-host; `laya-serve` speaks `/v1/systemone` | 512-1024 token context. Ships overconfident (ECE 0.466 → 0.081 after refit). Zero-shot on unseen tasks only 0.362 |
| **Kev** (0.8B/4B/9B/27B) | Jared Palmer, ~~about 2026-09-24~~ [corrected: first GitHub release v0.1.0 ("kev-0.5b") was 2026-09-17, the "Kev family" release 2026-09-20 and the Kev 1.0 release (0.8B/4B/9B/27B, whose numbers are quoted here) 2026-10-01T17:25Z (`gh api repos/jaredpalmer/kev/releases`). The repo was created 2026-09-17.] | Rank-16 LoRA plus "pointer head" on Qwen3.5 bases; Kev-27B is fully fine-tuned on Qwen3.8-27B | Kev-4B H100: 18.1 ms; Kev-27B B200: 46.5 ms | Apache-2.0 | Out-of-domain accuracy ~~Kev-4B 0.838 vs Jev 0.857 (dev)~~ [corrected: like-for-like on the development split, Kev-4B scores 0.817 vs Jev 0.857. 0.838 is Kev-4B's *test* score, and Jev was run only on dev. Kev-27B scores 0.851 dev / 0.889 test (kev README "Accuracy: New Sources", cells = dev / test)]; held-out index 38.0 vs Jev 54.0 |

**Robotics use** (community, all indep.; details in teammates' piper-astra-jev and quackd notes and the awesome-jev index):
- Jev chooses discrete primitives from JSON state, at about 150 ms to 1 s per decision.
- One MuJoCo apple-to-plate seed reported $0.0188 for Jev vs $5.93 for Astra.
- No peer-reviewed robotics evaluation exists.

**Relevance.** These suit a fast gate in the "light head" position: the LLM writes the menu and the state schema, and the decision model picks among them at 5-50 Hz. They cannot see, so perception must be symbolic first.

---

## 4. Open embodied-reasoning VLMs (local fast perception / pointing candidates)

**Independent per-benchmark comparison.** PhysBrain 1.5, arXiv 2609.14973, Table 4, re-evaluated every model with one metric. Closed models ran at their *lowest* thinking setting: Gemini minimal, GPT low, Claude adaptive-low.

| Benchmark | Gemini 3.6 Flash | GPT-6 Astra | Claude Opus 5 | PhysBrain 1.5 (8B) | RoboBrain 2.5 (8B) | MiMo-Embodied (7B) | Embodied-R1.5 (8B) | Hy-Emb-VLM-1.0 (30B-A3B) | Qwen3-VL-8B |
|---|---|---|---|---|---|---|---|---|---|
| ERQA | 72.3 | **75.8** | 61.5 | 52.8 | 44.3 | 44.8 | 42.3 | 56.3 | 42.5 |
| PointBench | 69.6 | **71.9** | 68.6 | 64.3 | 61.7 | 51.6 | 64.5 | 61.1 | 61.5 |
| RefSpatial-Bench | 76.4 | **78.0** | 56.3 | 50.9 | 59.9 | 40.0 | 55.2 | 47.4 | 43.6 |
| Where2Place | 73.4 | **76.9** | 69.0 | 72.1 | 72.0 | 63.9 | 75.0 | 65.4 | 64.7 |
| RoboSpatial-Home | 71.0 | 73.7 | 72.0 | **73.9** | 67.3 | 60.1 | 72.0 | 71.2 | 67.7 |
| VSI-Bench | 51.8 | 59.8 | **21.3** | 61.9 | 43.9 | 45.0 | 56.3 | 55.7 | 55.1 |
| PixMo-Points | 74.8 | **75.2** | 56.7 | 62.2 | 57.3 | 51.5 | 65.3 | 55.8 | 53.7 |
| Part-Affordance | 64.7 | 55.0 | 78.1 | **84.0** | 24.7 | 61.9 | 83.4 | 62.8 | 56.4 |
| VABench-Point | 60.7 | 65.3 | 64.7 | 65.2 | 10.6 | 47.7 | **75.7** | 60.5 | 46.3 |
| ShareRobot-Traj | 82.8 | 83.1 | 81.9 | 84.9 | **85.2** | 69.0 | 84.0 | 84.5 | 78.2 |
| EgoPlan-Bench2 | 53.1 | **69.3** | 48.4 | 62.1 | 33.8 | 37.9 | 53.1 | 47.5 | 32.1 |
| **Overall (28 benchmarks)** | 73.0 | **73.3** | 67.9 | 72.5 | 58.2 | 57.4 | 64.9 | 66.0 | 59.5 |

How to read this table:
- Low-effort evaluation understates the frontier models. DeepMind's chart has Opus 5 at 67.2 on ERQA, vs 61.5 here.
- Opus 5's VSI-Bench score of 21.3 is a video spatial benchmark. Treat it as a red flag to test, not as settled: frame sampling and harness could be the cause.
- Even so, Claude's pointing and RefSpatial gap (about 20 points) is consistent with Anthropic's own docs.

**Model entries (open).**

| Model | Org, date | Approach | Code / license | Notes for the harness |
|---|---|---|---|---|
| **Qwen3-VL** (2B-32B dense, 30B-A3B, 235B-A22B) | Alibaba, 2025-11 (arXiv 2511.21631) | 256K interleaved context, interleaved-MRoPE, DeepStack | open weights | Base for Cosmos-Reason2, Cosmos 3 and PhysBrain 1.5. Qwen3.5-9B was the student in K1 |
| **Cosmos-Reason2** (2B/8B/32B) → **Cosmos 3 Nano** (8B reasoner + 8B generator) | NVIDIA, 2025-12-19 / 2026-06-01 (arXiv 2606.02800) | CR2: Qwen3-VL post-trained with SFT and RL; 2D/3D points and boxes; 256K context. Cosmos 3: mixture-of-transformers "omnimodal" model with Action-CoT 2D end-effector plans | NVIDIA Open Model License / OpenMDW-1.1 | CR2-8B: Where2Place 50.0, ERQA 44.0, below Qwen on some benchmarks. Cosmos3 Nano overall 62.3 in PhysBrain's table. The CR2 collection is archived in favor of Cosmos 3 |
| **RoboBrain 2.0 → 2.5** | BAAI / FlagOpen, 2025-07 / 2026-01-20 | 2.5 adds depth-aware 3D coordinates, metric constraints, 3D keypoint traces and dense temporal value (progress) | GitHub FlagOpen/RoboBrain2.5 | Strong on trajectories (ShareRobot 85.2), weak on part-affordance (24.7) |
| **MiMo-Embodied** (7B) | Xiaomi, 2025-11-20 | Cross-embodiment model for driving plus embodied tasks, CoT and RL | GitHub XiaomiMiMo/MiMo-Embodied | 57.4 overall in PhysBrain's table |
| **Embodied-R1 (3B) → R1.5 (8B)** | pickxiguapi, 2025-08 / 2026-06 | Pointing as an embodiment-agnostic intermediate representation; RFT; R1.5 adds a Planner-Grounder-Corrector loop and can be fine-tuned into a VLA | open | R1: 56.2% SimplerEnv, 87.5% on 8 real XArm tasks zero-shot. R1.5: best VABench-Point (75.7) |
| **RynnBrain 1.1** (2B/9B/122B-A10B) | Alibaba DAMO Academy, 2026-07-20 | Contact-point prediction, native 3D grounding, RynnBrain-VLA | GitHub alibaba-damo-academy/RynnBrain | Authors claim the 122B model beats all evaluated models on VSI-Bench, MMSI and RefSpatial (author-reported). On RoboChrono: Frame Matching 95.4%, Frame Ordering 32.9% |
| **Hy-Embodied-VLM-1.0** (30B, 3B active MoE) | Tencent Hunyuan / Robotics X, 2026-07-14 | Hy3-A3B LLM plus Hy-ViT2; action-centric taxonomy; 38 benchmarks | GitHub Tencent-Hunyuan/HY-Embodied | Best open model before PhysBrain 1.5 (66.0); latency-oriented MoE |
| **PhysBrain 1.5** (8B) | DeepCybo, 2026-09-14 | Qwen3-VL-8B; human-video pre-training; discrete end-effector motion and dense visual target tokens | GitHub DeepCybo-PhysAI/PhysBrain-1.5 plus PhysBrainEvalKit | Best open overall (72.5); a candidate local pointing and affordance service |
| **ACE-Brain-0.5** (8B) | DAXIAO Robotics, 2026-07-05 | Five functions (perception, decision, interaction, self-monitoring, self-improvement); SSR+ merging | open | 59.0 overall in PhysBrain's table |
| **MolmoER / MolmoAct2** | Ai2, 2026-05-04 | MolmoER is the embodied-reasoning backbone; MolmoAct2 adds a flow-matching action expert plus the OpenFAST tokenizer; 720 h BimanualYAM data | fully open | Authors claim MolmoER beats GPT-5 and ER-1.5 on 13 embodied-reasoning benchmarks (author-reported). MolmoAct2 is the VLA baseline Astra beat in StationeryBench (out-of-distribution scenes) |
| **GroundingPI** (4B) | groundingpi, 2026-09-30 | Points and boxes as quantized coordinate tokens; GRPO | GitHub groundingpi/GroundingPI | 73.68 average on 34 grounding benchmarks vs Astra 71.54 (author-reported). The best evidence that a small local grounding model can match the frontier at pointing |
| **Pelican-VL 1.0** (7B-72B) | 2025-10 | "DPPO" metaloop RL | open | Earlier lineage; no 2026 head-to-head found |

---

## 5. Cross-model head-to-heads (closed-loop or agentic)

| Study (date) | Setup | Models → result | Notes |
|---|---|---|---|
| LIBERO-Agent (2026-09-30) | 30 sim tasks × 3 rollouts; native OSC_POSE actions; vendor harnesses (Codex, Claude Code, DeepSeek); high effort | Astra 45.0 > Opus 5 19.3 > Fable 5.1 15.8 > Sol 10.8 > Qwen3.8-Max 10.5 > Kimi K3 5.5 > DS-V4.1-Flash 4.5 | Adding depth gave +20 points SR to Astra, Opus and Fable. Video demos helped Astra (36.0 → 66.2 stage completion) |
| CodeActionBench (2026-09-27) | 25 tasks; code-as-policy; RGB plus calibrated geometry API | Astra (Codex) 73.3% > Opus 5 (ref) 49.3% > Opus 5 (Claude Code) 45.3% > Gemini 3.6 Flash 20.0% > Sol 16% > Qwen3.8-Max 14.7% > Grok 4.6 = Kimi K3 12% > Sonnet 5 2.7% | Gemini 3.6 Flash had the cheapest total ($52) |
| DrivingBench (2026-09-30) | Real car, 3 MCP tools, latency counts | Astra finished; Fable 5.1 45%; Grok 4.6 ≤11%; Sol 6% | Median s/command: Astra 4.9, Sol 5.4, Grok 12.5, Fable 22.2 |
| Robocurve (2026-09-03/04) | Real YAM bimanual arms; 224×224 frames [fact-check: UNVERIFIED for these runs. Neither the Astra nor the Fable 5.1 page states a resolution. 224×224 is stated only on the Opus 5 stack-blocks page, and it is the default `cam_width: int = 224` in repos/inspect-robots-yam/src/inspect_robots_yam/config.py:148]; medium effort | Bowl: Astra 19/20, Fable 5.1 8/20, Fable 5 1/20 | Astra vs Fable runs were not interleaved; operator grading |
| Robocurve Opus 5 scaling (2026-08-19) | Raw joint targets, n=5 per condition | Opus high 76 > medium 64 > Sol high 60 > Opus low 46 | Effort matters for Claude in this setup |
| Gen-TAMP coding agents (2026-09-24) | 28 environments, 98,000 episodes | Astra 95%, Opus 5 82%, Sol 56% vs planners 47% (16 environments) | Code-as-policy is where Claude is closest to Astra |
| RoboDojo (2026-09-21) | 42 tasks × 50 episodes | Astra 22.48% SR; GPT-5.5 0.88%; DeepSeek-Flash 1.92% | No Claude or Gemini entry on the board |
| Embody (Anthropic, 2026-07-09) | 12 models, 4 interfaces | Mythos 0.389 composite; LIBERO direct control ≤5.5% | Programmatic control was the strongest interface |
| ER 2 chart (vendor, 2026-07-30) | Success detection, ERQA, safety | ER 2 > Opus 5 ≈ Sol on most metrics; Opus is best non-ER on video success detection and safety instruction following | Vendor-selected metrics |
| RoboChrono (2026-09-29) | 18 VLMs, streaming task understanding | Astra Frame Matching 98.3% vs Ordering 68.3% | Temporal ordering is weaker than matching |

---

## 6. Analysis: what this means for an "Opus backbone + light action head" harness

**Promising**
1. **A hierarchical split, with Claude at the slow layer.** Every source agrees, including Anthropic's Embody, DeepMind's ER-plus-VLA design and the Understanding Robots synthesis. Abstraction and code are where Claude is near the frontier (TAMP 82%; best non-specialist on video success detection). Direct end-effector or joint control at 0.2-0.4 Hz is where it lags.
2. **Outsourcing pixels.** Use Gemini ER 2 (cheap, Live-streamable, native points) or a local 4-8B grounding or embodied model (GroundingPI-4B, PhysBrain 1.5, RoboBrain 2.5, Embodied-R1.5) for points, boxes, affordances and trajectories. Feed metric 3D from depth into Claude as text or tool results. K1 shows +27.8 points for Astra from perception tools alone. LIBERO-Agent shows +20 points from calibrated depth for Opus and Fable.
3. **A light learned head where data exists.** The K1 student result is the most direct evidence for the user's "light action head" idea: a Qwen3.5-9B distilled from 107 teacher episodes of tool-calls beat OpenVLA (44.2% vs 30.2%). Gemini On-Device (50-100 demos) and Embodied-R1.5 (VLA fine-tune) point the same way.
4. **Decision models as reflex gates.** Kev and Laya are self-hostable at about 20-40 ms; Jev is hosted at 70-500 ms. They can gate done/abort/primitive choices between LLM calls. They are text-only, so a symbolic state must be built first.
5. **Exploiting Claude API features for loops.**
   - Per-message effort lets the harness use low effort for routine steps and high effort for replanning.
   - Mid-conversation system messages and prompt caching keep the stable prefix.
   - Use pre-resized images with `oversized_image: "error"`.
   - Request absolute pixel coordinates.
   - Use `strict` tools in place of forced `tool_choice`, which returns a 400 on Opus 5.5 and Fable 5.1.

**Unpromising or risky**
1. **Claude as a direct low-level controller.** Embody LIBERO direct control was ≤5.5%, and Opus trails Astra by 2-3x in closed-loop studies. Opus 5.5 has not been measured, so do not assume parity.
2. **Naive LLM-overrides-VLA supervision.** In Embody, every Claude model supervising MolmoAct did worse than MolmoAct alone. Bounded edits (as in GPT-as-Policy) or a defer-by-default policy are needed.
3. **Normalized-coordinate pointing from Claude.** The vendor explicitly warns against it, and the independent RefSpatial and PixMo gaps are about 20 points.
4. **Video-based spatial reasoning from Claude without testing.** VSI-Bench 21.3 at low effort is a warning sign.
5. **Relying on any frontier API for anything time-critical.** Measured gaps run from 3 s to 40 s, and "physics paused" is common in the published results.
6. **Prompt framings that emphasize real physical stakes.** These can trigger refusals: Astra in DrivingBench, and Fable and Opus classifier `refusal` stop reasons. Operational tool framing worked better.

---

## 7. Open gaps / UNVERIFIED

- The OpenAI Astra launch page (openai.com/index/gpt-6-astra) is blocked by Cloudflare, so any robotics or "Put That There" content in it is UNVERIFIED. The "recurrent depth" architecture claim is UNVERIFIED.
- There are no independent embodied numbers for Opus 5.5, Sonnet 5.5, Fable 5.1 on pointing benchmarks, GPT-6 Sol/Luna/6.1 Sol, Gemini 3.8 Flash or Gemini 4 Argon.
- ER 2 has no published Point-Bench or RefSpatial numbers, and no Astra comparison.
- Grok/Optimus integration and Grok 5 status are from secondary blogs: UNVERIFIED.
- ~~Kev on OpenRouter ($0.042/M, 8K context, 2026-09-25) is from search snippets of an OpenRouter X post I did not open: UNVERIFIED.~~ [corrected: this looks mis-attributed. OpenRouter's public model list (`GET https://openrouter.ai/api/v1/models`, queried 2026-10-02) has no Kev model. What it does list is `typesafe/jev-router` ("TypeSafe: Jev Router"), created 2026-09-25 19:12 UTC, with 1,000,000 context and router pricing (−1). Its description says it "picks the best model and reasoning effort for each request … runs on Jev". $0.042/M is Jev's own price (docs.typesafe.ai/models).]
- Anthropic's Embody code is not yet released.
- The DeepMind ER 2 blog body could not be rendered from here; numbers were taken from the deepmind.google model page's chart alt-text and from blog.google.

---

## Sources

OpenAI
- https://community.openai.com/t/introducing-gpt-6-astra-the-most-intelligent-and-aligned-model-in-the-world/1394703
- https://developers.openai.com/api/docs/models/gpt-6-astra
- https://developers.openai.com/api/docs/guides/latest-model
- https://developers.openai.com/api/docs/guides/images-vision
- https://developers.openai.com/api/docs/models/gpt-live-1
- https://deploymentsafety.openai.com/gpt-6-astra
- https://en.wikipedia.org/wiki/GPT-6_Astra
- https://openai.com/index/gpt-6-astra/ (unreachable: 403/Cloudflare)
- https://artificialanalysis.ai/models/releases/gpt-6-astra

Anthropic
- https://platform.claude.com/docs/en/about-claude/models/overview.md
- https://platform.claude.com/docs/en/models/opus-5-5/overview.md
- https://platform.claude.com/docs/en/models/fable-5-1/overview.md
- https://platform.claude.com/docs/en/models/sonnet-5-5/overview.md
- https://platform.claude.com/docs/en/build-with-claude/vision.md
- https://platform.claude.com/docs/en/build-with-claude/vision-coordinates.md
- https://www.anthropic.com/news/claude-opus-5-5
- https://www.anthropic.com/research/project-fetch-robot-dog
- https://www.anthropic.com/research/project-fetch-phase-two
- https://www.anthropic.com/research/claude-plays-robotics

Google
- https://ai.google.dev/gemini-api/docs/robotics-overview
- https://ai.google.dev/gemini-api/docs/pricing
- https://deepmind.google/models/model-cards/gemini-robotics-er-2/
- https://deepmind.google/models/gemini-robotics/embodied-reasoning/
- https://deepmind.google/models/model-cards/gemini-robotics-on-device-2/
- https://deepmind.google/discover/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/
- https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/
- https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-1-6/
- https://arxiv.org/html/2510.03342 (Gemini Robotics 1.5, App. C, Tables 19-21)
- https://techcrunch.com/2026/09/30/google-releases-gemini-4-argon-called-its-most-powerful-model-yet/

xAI
- https://docs.x.ai/docs/models

Decision models
- https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://docs.typesafe.ai/models
- https://huggingface.co/convaiinnovations/laya
- https://github.com/jaredpalmer/kev
- https://github.com/Frank-ZY-Dou/awesome-jev (local clone: research/repos/awesome-jev)

Independent evaluations
- https://openai.robocurve.org/gpt-6-astra/
- https://openai.robocurve.org/stationerybench/
- https://anthropic.robocurve.org/fable-5.1/
- https://anthropic.robocurve.org/stack-blocks/
- https://github.com/robocurve/llm-token-speed (Artificial Analysis API snapshot of 2026-08-20; local clone)
- https://arxiv.org/abs/2609.24170 (RoboDojo Astra eval)
- https://arxiv.org/abs/2609.38537 (GPT-6 Astra as embodied policies)
- https://arxiv.org/html/2609.39507v1 (LIBERO-Agent)
- https://arxiv.org/html/2609.33807v1 (CodeActionBench)
- https://arxiv.org/html/2609.38948v1 (DrivingBench)
- https://arxiv.org/html/2609.30233v1 (coding agents for generalized TAMP)
- https://arxiv.org/html/2609.29389v1 (Robo-Harness K1)
- https://arxiv.org/html/2609.35718v1 (Hard Vision, Easy Vision)
- https://arxiv.org/abs/2609.29861 (Astra VLN-CE)
- https://arxiv.org/abs/2609.36605 (RoboChrono)
- https://arxiv.org/abs/2609.39601 (GroundingPI)
- https://arxiv.org/abs/2605.20544 (RoboAbstention)
- https://arxiv.org/abs/2609.31770 (XLeRobot with Astra)
- https://arxiv.org/abs/2609.20822 (SafeHarness)
- https://arxiv.org/abs/2609.34276 (NavHarness)
- https://www.understandingrobots.org/p/openais-astra-model-is-shockingly

Open embodied models
- https://arxiv.org/html/2609.14973v1 (PhysBrain 1.5, Table 4)
- https://arxiv.org/abs/2511.21631 (Qwen3-VL)
- https://huggingface.co/nvidia/Cosmos-Reason2-8B
- https://arxiv.org/abs/2606.02800 (Cosmos 3)
- https://arxiv.org/abs/2601.14352 (RoboBrain 2.5)
- https://arxiv.org/abs/2507.02029 (RoboBrain 2.0)
- https://arxiv.org/abs/2511.16518 (MiMo-Embodied)
- https://arxiv.org/abs/2508.13998 (Embodied-R1)
- https://arxiv.org/abs/2606.11324 (Embodied-R1.5)
- https://arxiv.org/abs/2607.17977 (RynnBrain 1.1)
- https://arxiv.org/abs/2607.12894 (Hy-Embodied-VLM-1.0)
- https://arxiv.org/abs/2607.04426 (ACE-Brain-0.5)
- https://arxiv.org/abs/2605.02881 (MolmoAct2)
- https://arxiv.org/abs/2508.07917 (MolmoAct)
- https://arxiv.org/abs/2506.04308 (RoboRefer / RefSpatial)
- https://arxiv.org/abs/2511.00108 (Pelican-VL)
- GitHub orgs checked via `gh search repos`: FlagOpen/RoboBrain2.5, XiaomiMiMo/MiMo-Embodied, alibaba-damo-academy/RynnBrain, Tencent-Hunyuan/HY-Embodied, DeepCybo-PhysAI/PhysBrain-1.5, pickxiguapi/Embodied-R1.5, DAXIAORobotics/ACE-Brain-0.5, groundingpi/GroundingPI

---

## Verification (fact-check pass)

Adversarial re-check done 2026-10-01/02. Every item below was re-opened at a primary source: arXiv HTML/abs, vendor docs as `.md`, vendor blogs and chart alt-text, GitHub/HF APIs, or local repo files. Raw fetches are in `research/tmp_fc_models/`. Overall the note is accurate. All benchmark numbers re-checked from papers and vendor charts matched exactly. The errors are in dates, product-tier attribution and a few like-for-like comparisons.

### Confirmed claims (seen in a primary source)

**Closed-loop and agentic studies**
1. **LIBERO-Agent** (arXiv 2609.39507v1, 30 Sep 2026).
   - Table 1 Performance Score: Astra 45.0, Opus 5 19.3, Fable 5.1 15.8, Sol 10.8, Qwen3.8-Max 10.5, Kimi K3 5.5, DeepSeek-V4.1-Flash 4.5.
   - Astra used 11.5 min and 1.38M tokens per episode. Protocol: 30 tasks × 3 rollouts, high effort, a 1,800 s limit per rollout, and native OSC_POSE in batches of 1-50.
   - Table 2: calibrated depth adds +20 pp short-horizon SR for each agent (Astra 50→70, Opus 10→30, Fable 0→20).
   - Table 3: video demos take Astra's stable stage completion from 36.0 to 66.2.
2. **CodeActionBench** (2609.33807v1, 27 Sep 2026).
   - Table 10, successes out of 75: Astra (Codex) 55, Opus 5 reference harness 37, Opus 5 (Claude Code) 34, Gemini 3.6 Flash 15, Sol 12, Qwen3.8-Max 11, Grok 4.6 9, Kimi K3 9, Sonnet 5 2.
   - Overclaims on failed attempts: Opus (reference) 13/38 = 34.2% vs Astra 3/20 = 15.0%. Astra made a correct final claim in 68/75.
   - Gemini 3.6 Flash had the lowest total cost, $52.00. Sonnet 5 left 70/75 attempts without a claim, and 69 of those hit a budget limit.
3. **DrivingBench** (2609.38948v1, 30 Sep 2026).
   - Astra finished on attempt 2: 322 s, 134.7 m, 24 commands.
   - Fable 5.1 reached 9%, 10%, 45% over its attempts; Grok 4.6 8%, 11%, 10%; Sol 6% on all three.
   - Median s/command: Astra 4.9, Sol 5.4, Grok 12.5, Fable 22.2.
   - Harnesses were Codex, Claude Code and Cursor, with 3 MCP tools.
4. **Generalized TAMP** (2609.30233v1, 24 Sep 2026). On the 16 environments that have a planner: Claude Code (Opus 5) 82%, Astra 95%, Sol 56%, planners 47%. The study covers 28 environments and 98,000 episodes.
5. **RoboDojo Astra** (2609.24170v1, 21 Sep 2026).
   - Astra: 22.48% SR / 28.97 Score over 2,100 trials, with Precision at 4.00% SR. GPT-5.5 scored 0.88% and DeepSeek-Flash 1.92%.
   - The "14.7 s per call" figure is not in the paper text. It comes from `repos/RoboProbe/results/l3_inspect_eef_official_2100/efficiency.json` (`planners.astra.overall.latency_s` = 1,740,592 s over 118,767 calls = 14.66 s).
6. **GPT-as-Policy** (2609.38537v1, Galbot Team).
   - Hybrid succeeded on 24/50 (48%) vs Direct 13/50 (26%).
   - The locomotion run used 250 synchronous calls with a mean of 39.86 s each, with physics paused.
7. **Robo-Harness K1** (2609.29389v1).
   - Matched subset of n=18: RGB+Astra 11/18 (61.1%), K1+Gemini 3.7 Flash 14/18 (77.8%; 139/180 = 77.2% on the full Gemini eval), K1+Astra 16/18 (88.9%).
   - Qwen3.5-9B K1 RUA trained on 107 episodes: 19/43 (44.2%) vs OpenVLA 13/43 (30.2%).
   - n is small: 18 and 43 cases.

**Robocurve (real YAM arms)**
8. Astra page (4 Sep) and Fable 5.1 page (3 Sep).
   - Bowl: Astra 19/20, Fable 5.1 8/20, Fable 5 1/20. Puzzle: Astra 2/20, Fable 5.1 2/20.
   - Cost and time per run: $0.94 and 2.5 min vs $2.12 and 6.8 min.
   - The Astra runs were not interleaved with the Fable runs, and grading was operator-judged with the model known. Fable 5 exhausted its 20-call budget in 18/20 bowl trials.
9. Opus 5 stack-blocks page (19 Aug). The per-run scores reproduce the means: low 46, medium 64, high 76, Sol (high) 60.
   - Setup: 5 runs per condition, 224×224 RGB, fast mode, raw joint targets with no IK, inspect-robots 0.53.0.
10. StationeryBench (10 Sep). Astra completed 7/100 vs MolmoAct2 0/100; mean progress 46 vs 12. MolmoAct2 ran zero-shot with `allenai/MolmoAct2-BimanualYAM`.

**Anthropic API**
11. Models overview and per-model pages (`.md`).
   - IDs: `claude-fable-5-1`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-4-5-20251001`.
   - Prices: $10/$50, $4/$20, $2/$10, $1/$5. Cache reads: $0.25 for Fable 5.1 and $0.20 for Opus 5.5.
   - Default effort: high (Fable 5.1), medium (Opus 5.5), high (Sonnet 5.5).
   - Release dates: 2026-09-01 (Fable 5.1), 2026-09-22 (Opus 5.5), 2026-09-28 (Sonnet 5.5). Opus 5 (legacy) was released July 24, 2026 at $5/$25.
   - Behavior: forced `tool_choice` returns 400 on Opus 5.5, and thinking cannot be disabled. Sonnet 5.5's `between_tools` is valid at `high` effort or below.
   - Fast mode on Opus 5.5 costs $8/$40, gives "up to 2.5x" OTPS, and requires `speed:"fast"` plus beta header `fast-mode-2026-02-01`.
12. Vision docs.
   - Images use 28×28 patches. High-resolution tier: 2576 px / 4784 tokens. Standard tier: 1568 px / 1568 tokens.
   - 600 images per request. More than 20 images triggers a 2000 px per-image limit.
   - `"transformations": {"oversized_image": "error"}` exists.
   - Quotes confirmed: "coordinate and localization outputs are approximate" and "does not work well when you ask for normalized coordinates".
13. Opus 5.5 launch post: Chartography 89.0% (with tools) and OSWorld 2.1 81.8% (partial). It has no robotics benchmarks.

**OpenAI API**
14. GPT-6 Astra model card (`.md`).
   - 1,050,000 context, 922,000 max input, 128,000 output, knowledge cutoff Apr 30 2026.
   - $10 / $1 / $12.5 / $50 per 1M. Above 272K input: 2x input, 1.5x output. Batch/Flex at 50%; Fast mode 2x.
   - Live and Realtime are not supported.
   - Latest-model guide: `none` effort is unsupported ("use `low`"), tool calling requires Responses, and `async: true` tools and mid-turn steering over WebSocket exist.
15. Images-vision guide (`.md`).
   - 32 px patches with a 1.2 multiplier; 1024×1024 at `high` costs 1,229 tokens.
   - `high` has a 2,500-patch budget, and `auto` behaves as `original` on Astra.
   - Limits: 1,500 images, 512 MB, 30,000 patches.
   - Confirmed quote: "struggles with tasks requiring precise spatial localization".
16. `gpt-live-1`: $0.05/min; image and video listed as unsupported modalities.
17. Astra system card (deploymentsafety.openai.com/gpt-6-astra, 286K chars of text): zero hits for "robot", "embodied" or "spatial".

**Google**
18. robotics-overview page.
   - Two ER 2 endpoints, each 131,072 in / 65,536 out, with the feature matrix as stated. ER 2 is "Builds on Gemini 3.5 Flash" (model card: "based on Gemini 3.5 Flash").
   - Output is `[y, x]` normalized to 0-1000 via `client.interactions.create(... generation_config={"thinking_level": ...})`.
   - Recommendations confirmed: use `medium` thinking for the latency/quality balance, and "Query multiple times and average results for high-precision tasks." ER 1.6 shuts down "at the end of August".
   - Pricing: $1.00/$5.00 through 2026-12-31, then $2/$10. A free tier exists.
19. deepmind.google ER page chart alt-text. All 7×5 values in the note's ER 2 table match, and so do the orchestration figures (48.6→60.0, 37.4→42.9, 63.6→74.0).
   - blog.google ER 2 post (2026-07-30): moment finding 91.3%, 0.96 s mean absolute distance.
   - ER 1.6 post (2026-04-14): instrument reading via Boston Dynamics.
20. Gemini Robotics 1.5 report, Table 19 (arXiv 2510.03342): all values in the note's ER 1.5 table match.
21. On-Device post (2025-06-24): "as few as 50 to 100 demonstrations", trusted testers.
   - On-Device 2 card (30 Jul 2026): "based on Gemini Robotics 1.5 technology and our on-device Gemma models", Trusted Testers only.
22. Gemini pricing.
   - 3.1 Pro Preview $2/$12; 3.5 Flash $1.50/$9; 3.6/3.7/3.8 Flash $0.75/$3.75 through 2026-12-31.
   - `gemini-3.8-live` image/video input: $1.00 per 1M or $0.002/min.
   - TechCrunch: Argon is limited to Fairwind cyber partners.
23. xAI docs: `grok-4.7` has 500k context at $2/$6 per 1M. Images: 20 MiB max, no count limit, jpg/png only.

**Speed measurements (Artificial Analysis)**
24. Snapshot `repos/llm-token-speed/data/aa_api_models.json`, commit 2026-08-20.
   - Opus 5 time to first token: low 3.182 s, medium 6.909 s, high 10.316 s, xhigh 26.692 s, max 36.976 s. Output speed 50.1-54.4 tok/s.
   - Sonnet 5 at low: 1.496 s, 61.6 tok/s. Haiku 4.5 non-reasoning: 0.693 s, 101.1 tok/s.
   - Gemini 3.7 Flash: 301.9-339.4 tok/s, 1.303 s at low. Grok 4.6 (high): 66.0 tok/s, 39.819 s.
   - The AA Astra page: 43-51 t/s; low effort 2.96 s to first answer token.

**Anthropic research posts**
25. Embody / "Claude plays robotics" (Jul 9, 2026).
   - Chart alt-text: "Mythos Preview scores the highest at 0.389, while Opus 4 scores the lowest at 0.115".
   - Twelve models across five providers. The cursor tool took Mythos from 6% to 32% on the 10-task subset, and direct-control task completion was 0-5.5%.
   - Every model supervising MolmoAct did worse than MolmoAct alone.
   - "roughly 83 Hz" needed vs "~0.2-0.4 Hz" achieved; simulator paused.
   - Latency: 2-8 s / 5-15 s / 15-60 s. The code repo is "once released".
26. Project Fetch phase 1 (Nov 12, 2025): 8 researchers; Team Claude 7/8 vs Team Claude-less 6/8.
   - Phase 2 (Jun 18, 2026): Opus 4.7 at max effort, 3 trials. Chart alt-text: 9 min 35 s vs Team Claude 181 min (18.9x) and Team Claude-less 361 min (37.7x).

**Open models and smaller evaluations**
27. PhysBrain 1.5, Table 4 (2609.14973v1): every reproduced cell (11 benchmarks × 9 models plus the overall row) matches. Closed models ran at Gemini minimal, Astra low and Claude adaptive-low.
28. Other spot checks, all matching:
   - GroundingPI: 73.68 vs Astra 71.54 over 34 benchmarks and 44 baselines; authors are XPeng / PKU / HKU / Berkeley / Princeton et al.
   - VLN-CE: 81.3% at ultra (75.7% at medium) on R2R-CE-100.
   - RoboChrono: Astra 98.3 / 68.3, RynnBrain-122B 95.4 / 32.9, 18 VLMs.
   - Hard Vision: Astra +3.8 over the ER 2 reference.
   - RoboAbstention: ER 1.6 at 16.5% → 93.6% over 6,069 instructions.
   - Embodied-R1: 56.2% SimplerEnv, 87.5% on 8 XArm tasks.
   - MolmoAct2: 720 h BimanualYAM; MolmoER beats GPT-5 and ER-1.5 on 13 benchmarks.
   - RynnBrain 1.1: 122B tops VSI-Bench, MMSI and RefSpatial-Bench.
   - Hy-Embodied: Hy3-A3B + Hy-ViT2, 38 benchmarks.
   - arXiv submission dates as listed in §4.
29. Decision models.
   - Jev blog (Sep 15, 2026; Diogo Almeida): RLCD, 70-500 ms, $0.042/MTok input, output free, cardinality 255.
   - Jev docs: `jev-1.13.0`/`jev-latest`; 64k per request, 32k for state plus the longest question; 100K tok/s and 40 req/s; text only.
   - Laya HF README: ModernBERT-large 395M → 421M; mmBERT-base 322M; ECE 0.466 → 0.081; zero-shot 0.362; Apache-2.0; `laya-serve` exposes `/v1/systemone`. HF repo created 2026-09-18.
   - Kev README: rank-16 LoRA plus pointer head; Kev-27B fully fine-tuned on `Qwen/Qwen3.8-27B`; held-out index Kev-4B 38.0 vs Jev 54.0.
   - OpenRoboto `docs/RESULTS.md`: Jev $0.018825 vs Astra $5.933624, one seed-0 trial each.

### Corrections (claim → correct value, evidence)
1. **§1.1 Astra API date.** "the API and paid tiers followed on 2026-09-04" → the API release was **2026-09-03**: the OpenAI API changelog places "Released GPT-6 Astra" (v1/responses, v1/chat/completions) under "### Sep 3". Only the paid-ChatGPT public release was 2026-09-04 (Wikipedia, GPT-6 article). Fixed inline.
2. **§1.1 Ultrafast.** "an 'Ultrafast' mode also exists with 'no latency SLA'" → the no-SLA sentence is about **Fast mode** for Astra (latest-model guide: "Fast mode for GPT-6 Astra does not include a latency SLA").
   - Ultrafast = `service_tier: "ultrafast"`, added 2026-09-29.
   - Price: **$60 / $6 / $75 / $300 per 1M** (≤272K), i.e. 6x standard; $120/$450 for long context.
   - Low rate limits, WebSockets recommended, no EU residency.
   - Fixed inline.
3. **§1.3 computer-use breaking change.** It is not an Opus 5.5 + Fable 5.1 change. It applies to **Opus 5.5 only, on the Claude API and Google Cloud only**; Bedrock still accepts `computer_20251124` (whats-new-opus-5-5.md). Fixed inline.
4. **§1.3 "Opus 5.5 brings per-message effort".** Per-message effort (beta `mid-conversation-output-config-2026-07-01`) is also on Opus 5, Fable 5.1, Mythos 5.1 and Sonnet 5.5 (effort.md). It is not an Opus 5.5 novelty. Fixed inline.
5. **§3 Kev date.** "about 2026-09-24" → v0.1.0 on **2026-09-17**, "Kev family" on 2026-09-20, **Kev 1.0 on 2026-10-01** (GitHub releases API). Fixed inline.
6. **§3 Kev vs Jev accuracy.** "Kev-4B 0.838 vs Jev 0.857 (dev)" mixes splits. On dev it is **Kev-4B 0.817 vs Jev 0.857**; 0.838 is Kev-4B's test score, and Jev has no test score (kev README table "development / test"). Fixed inline.
7. **§3 Laya.** "7.2 ms/question at 10 questions" is the **multilingual** checkpoint. English `laya` = **15.9 ms/question** (158.6 ms per 10). Fixed inline.
8. **§7 "Kev on OpenRouter".** It is not Kev. OpenRouter's `/api/v1/models` lists **`typesafe/jev-router`** (created 2026-09-25 19:12 UTC, 1M context, router pricing) and no Kev entry. The "8K context" figure matches neither. Fixed inline.

### Unverifiable / not confirmed
- **224×224 frames in the Astra/Fable Robocurve runs** (§5): not stated on either page. It is plausible, as the inspect-robots-yam default (`repos/inspect-robots-yam/src/inspect_robots_yam/config.py:148`). Marked inline.
- **openai.com/index/gpt-6-astra**: not re-tried, since the note already marks it unreachable. The "recurrent depth" claim exists only as press reports cited on Wikipedia (Fortune, The Information, TechCrunch 2026-09-02/03). It is not in any OpenAI primary doc I opened, so it stays UNVERIFIED.
- **Cosmos-Reason2-8B card numbers** (Where2Place 50.0, ERQA 44.0, "archived" collection): the HF raw README returned 401 (gated), so they were not re-checked.
- **Jev rate limits "changed after launch"**: only the current docs could be seen (100K tok/s, 40 req/s); no earlier snapshot was available.
- **Grok-as-Optimus "System 2" and Grok 5 status**: secondary sources only, as the note says.
- **"No independent closed-loop robot result for Opus 5.5 / Sonnet 5.5"**: consistent with an arXiv API search (`all:"Opus 5.5"`, 0 hits) and a web search on 2026-10-02. A negative cannot be proven.
- **Understanding Robots synthesis, XLeRobot, SafeHarness, NavHarness**: listed in Sources but not used for numbers in the body. Not re-checked.

### Added missed details (within scope)
1. **OpenAI's cheaper GPT-6 tier shipped, with an image bug.**
   - `gpt-6.1-sol` was released **2026-09-29** at $2 input / $0.10 cached / $2.50 cache write / $10 output per 1M (≤272K). It needs Responses for tool calling, has no `none` effort, and has multi-agent in beta.
   - `gpt-6-sol` costs $2/$10 and `gpt-6-luna` $0.10/$0.50 (pricing page).
   - On **2026-09-25** OpenAI fixed "a bug in image encoding that degraded image understanding in GPT-6 Sol and GPT-6 Luna" and recommends rerunning image evals. Any robot or vision result on Sol/Luna from 2026-09-22 to 09-25 is suspect (API changelog).
2. **OpenAI does have a realtime path with images**, just not on Astra.
   - `gpt-realtime-2.1`: input "text, audio, image", 128K context, $4/$24 text, $5 per 1M image input.
   - `gpt-realtime-2.1-mini`: $0.80 per 1M image input.
   - Together with `gemini-3.8-live` (GA 2026-09-15) and `gemini-robotics-er-2-streaming-preview`, these are the only streaming multimodal endpoints for a low-latency perception sidecar.
3. **Anthropic Opus 5.5 behaviors that matter for a control loop** (whats-new-opus-5-5.md, fast-mode.md):
   - It thinks *more* per turn than Opus 5 at the same effort, "most of all at xhigh and max", so re-sweep effort and do not carry it over.
   - Text between tool calls now arrives as `thinking` blocks that are empty at the default `display: "omitted"`, so streamed progress notes go silent unless `thinking.display` is set.
   - Fast mode raises output tokens/s, "not time to first token". It barely helps short robot-action turns, which are dominated by TTFT.
   - The minimum cacheable prompt is 512 tokens.
   - From 2026-08-31, new accounts get a 400 by default if anything before a replayed thinking block changed. Use mid-conversation system messages, or the `thinking-binding-controls-2026-08-01` beta with `prefix_mismatch_behavior: "drop_block"`.
   - Beta `inline-tools-2026-09-15` lets you add or change tools mid-conversation without losing the cache.
   - Beta `compact-2026-09-04` provides on-demand compaction.
4. **Claude Mythos 5.1** exists, with the same capabilities as Fable 5.1, but only for Project Glasswing participants (fable-5-1 overview). Its cache reads are 2.5% of input, like Fable 5.1. The Embody leader was "Mythos Preview"; Mythos-class models are not generally available as API backbones.
5. **Claude image limit for Haiku:** 600 images per request applies to 1M-context models; models with a 200k context (Haiku 4.5) allow **100** (vision.md).
6. **Caveats missing from the Robocurve rows:**
   - The bowl comparison was **not on the same rig**: Fable bowl trials ran on rig-3, which was unavailable for Astra. The puzzle comparison was same-rig.
   - Costs are list price, $10/$50 for all three models, with no Anthropic prompt caching, while about a fifth of Astra's input was cached and not discounted.
7. **TAMP beyond the 16 planner environments.** The Q5 paragraph of 2609.30233 gives no-source mean success of Opus 74% vs Astra 86%. The paragraph compares across "27 of 28 environments", so the denominator appears to be all 28. With environment source code it rose to 84% and 95%. LLMGenPlan with Opus 5 and source code reached only 28% (2609.30233).
8. **GPT-as-Policy hybrid details.**
   - 10 RoboDojo tasks × 5 paired trials.
   - Of 42,750 executed control steps, 36,576 (85.6%) came from π0.5 and 6,174 (14.4%) were generated or corrected by Astra.
   - π0.5's published results, reweighted to the same mixture, give 15.67% SR. Direct Astra scored 26%.
   - This is the best published evidence for "defer by default" supervision.
9. **RoboDojo protocol nuance:** DeepSeek-Flash ran 10 episodes per task, not 50. All three LLMs ran on a single evaluation seed (2609.24170 §3.3).
10. **Safety and refusal asymmetry:** DrivingBench §2 cites RoboHarm (Sun et al., 2026): Astra "refused only 2 of 100 harmful robot-arm instructions" when simply asked, yet refused to drive in an empty lot until reframed. The RoboHarm harness is in `repos/roboharm` (5 tasks, 20 rollouts per task per model). RoboHarm's own result table was not re-opened here.
11. **Decision-model latency comparability:** Kev latencies are server-side model time for 6 questions, plus ~65 ms network via Modal. Kev-27B's B200 row was measured on the previous checkpoint. Independent Jev p50 is 236-276 ms (AbdelStark/jev-benchmarks and nibzard/decision-model-benchmark, cited in the Laya README). Kev-27B dev/test accuracy on new sources is 0.851/0.889, the closest to Jev.
