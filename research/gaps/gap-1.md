# Gap 1: Measured per-call latency and cost of the candidate backbones under the harness's real request shape

*2026-10-02. "Derived" = computed by me from public raw data; UNVERIFIED = not confirmed in a primary source.*

## Status

| Part | Status |
|---|---|
| 1. Mine the public Tier 1 trials | **Done.** All 360 `data/runs/*.json` files, `data/trials.csv`, and all 360 per-trial "Log" transcripts. The transcripts were streamed and parsed: 11,768 wire attempts, each with its `call N attempt M · endpoint · status · latency` line. |
| 2–4. Live 50-call benchmark of 14 configurations | **Not run.** There are no OpenAI or Google keys in the environment. The only Anthropic credential is this agent session's own `ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_BASE_URL`, which is not Ilia's API account and may add a gateway hop. The workspace rule is read-only toward the outside world, and nobody approved spending money. I wrote a ready-to-run benchmark instead (`gaps/gap-1-bench.py`; dry-run budget about $285). I also extracted every other public per-call measurement I could find (Section 5). |

## 1. Bottom line

1. **On the real Tier 1 rig, Opus 5.5 at medium effort takes p50 5.28 s, p90 11.72 s, p99 21.1 s and at most 60.3 s per call** (n = 4,105 calls, 120 trials).
   - GPT-6 Astra: 6.53 / 11.28 / 19.6 / 38.9 s (n = 3,437).
   - Opus 5: 8.77 / 20.79 / 40.5 / 67.8 s (n = 4,226).
   - The critique's 7.6 s (wall ÷ calls) overstates the LLM call by ~44%: it folds in 28 s of motion and ~14 s of per-trial overhead.
2. **About 84% of trial wall time is LLM waiting** (Opus 5.5: 84.5%, Astra 83.2%, Opus 5 89.9%). The arm moves for about 28 s out of 269 s.
   - Realized decision rate including motion: **7.6 calls/min (0.127 Hz)** for Opus 5.5, 6.9/min for Astra, 4.9/min for Opus 5.
3. **Per-call latency is mostly decode time for output tokens (thinking).** Across trials, Opus 5.5 latency ≈ **1.8 s + output_tokens / 87 tok/s** (R² = 0.91). It averages 424 output tokens per call.
   - Artificial Analysis independently lists Opus 5.5 at 92.2 tok/s.
   - The lever is output length (effort, fast mode, terse tool calls), not prompt size: ~87% of input is a cache read.
4. **The first call is about 3× slower:** Opus 5.5 p50 16.1 s (p90 24.1 s) vs a steady state of 5.13 s. Every Opus 5.5 trial's first call exceeded 8 s.
   - The final `done`/`give_up` call (which includes a hindsight note) has p50 12.1 s.
5. **An "8 s park" rule would fire on 23% of Opus 5.5 calls** (7.9 per trial, in 120/120 trials). It would fire on 31% for Astra and 56% for Opus 5.
   - Tails are thin: P(>15 s) = 4.7%, P(>20 s) = 1.4%, P(>30 s) = 0.15%.
6. **Reliability was perfect in this sample.** All 11,768 calls returned status 200 on the first attempt, with zero retries and zero timeouts.
   - None of the 360 transcripts contains the harness's "Respond with exactly one tool call." nudge. So zero no-tool-call turns occurred under `tool_choice:auto` with 3 tools.
7. **Cost under the Tier 1 shape: Opus 5.5 costs $0.0262 per call and $0.898 per trial** (matching the page's $0.90).
   - **54% of that is cache writes** (2,826 tokens written per call). Output is 32% and cache reads 14%.
   - Opus 5 costs $0.050 per call. Astra costs $0.107 per call using its recorded cache usage; the page's cache-adjusted estimate is about $0.040.
8. **There are still no measurements for:**
   - Opus 5.5 at low or high effort, with per-message effort, or in fast mode;
   - Sonnet 5.5 `between_tools`, or Haiku 4.5 in a robot loop;
   - Astra `ultrafast`;
   - ER 2 *pointing*;
   - TTFT for any model under the robot request shape;
   - any model with a 1920×480 tile.

   The only adjacent data: Opus 5 **fast mode** at high effort has p50 12.5 s with 2,404 output tokens per call (an implied 233 tok/s marginal rate). ER 2 at `reasoning_effort=high`, used as a joint controller, has p50 5.8 s and p90 9.6 s (n = 91).

## 2. Data and method

**Source.** Robocurve, "Opus 5.5 on RoboDojo-RC Tier 1", 2026-09-23. The setup:
- 6 tasks × 20 trials × 3 models, on six bimanual I2RT YAM rigs, with each task fixed to one rig.
- Inspect Robots 0.58.0, agent policy at **medium** effort.
- 40-call budget, 900-step cap, 25% speed cap.
- Frames are 224×224.

Model blocks ran in overlapping windows starting on Opus 5.5's launch day (UTC): Astra 09-22 16:13–21:56, Opus 5.5 09-22 18:18–09-23 00:14, Opus 5 09-22 20:40–09-23 03:19. Hourly medians are stable (Opus 5.5 4.6–5.8 s, Opus 5 7.6–9.2 s).

**Effective configuration.** It was identical across each model's 120 transcript headers:
- `effort=medium`, `image_horizon=2`, `images=always`, `max_llm_calls=40`, `max_speed_frac=0.25`, `speed=null`.
- Claude models: `wire=messages` at `https://api.anthropic.com/v1`, `max_output_tokens=16000`.
- Astra: `wire=responses` at `https://api.openai.com/v1`.
- Each wire call shows `tool count 3` (`move_to`, `done`, `give_up`). Each turn adds three 224×224 PNG frames, i.e. 3 × 64 image tokens.

**What "latency" measures.** In `inspect_robots_agent/_anthropic.py` (HEAD `095172f`), each attempt is one blocking `httpx` `POST /messages` (no streaming), wrapped in `t_start = time.time()` … `duration_s = time.time() - t_start`. `_responses.py` does the same for `POST /responses`.
- So it is **client wall time per HTTP attempt** (upload, queueing, prefill, adaptive thinking, tool call). **TTFT is not recoverable.**
- The body sets `thinking:{type:"adaptive"}`, `output_config:{effort}`, and `cache_control:ephemeral` on the system prompt; read timeout max(120, 16000/125) = 128 s.

**Token fields.**
- Per-trial `input_tokens` is the total prompt, i.e. uncached + `cache_creation_input_tokens` + `cache_read_input_tokens`. For example, run `adhoc_00188e02` has 973,438 = 78 + 124,942 + 848,418.
- Output totals include reasoning tokens.
- Per-call tokens are unpublished (`wire/.../calls.jsonl`), so token-to-latency relations below are trial-level regressions.

**Prices used** ($/MTok). The Opus 5.5 cache-read rate is 0.05× input, not the usual 0.1×.

| Model | Uncached | Cache write (5 min) | Cache read | Output | Source |
|---|---|---|---|---|---|
| Opus 5.5 | 4 | 5 | 0.20 | 20 | Anthropic pricing via the bundled API reference; AA lists a "95%" cache discount |
| Opus 5 | 5 | 6.25 | 0.50 | 25 | same |
| Astra | 10 | 12.50 | 1 | 50 | as stated on the Tier 1 page |

## 3. Per-call latency, Tier 1 (derived from 11,768 logged attempts)

| Model (medium) | n calls | p50 | p90 | p99 | max | Steady-state p50 / p90 (excl. first and last call) | First call p50 / p90 | Last call p50 | Output tokens/call | Input tokens/call | Cache-read share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `claude-opus-5-5` | 4,105 | **5.28** | **11.72** | **21.10** | 60.33 | 5.13 / 10.07 | 16.10 / 24.09 | 12.09 | 424 | 20,970 | 0.865 |
| `gpt-6-astra` | 3,437 | 6.53 | 11.28 | 19.59 | 38.89 | 6.31 / 11.03 | 10.02 / 13.26 | 8.50 | 134 | 11,534 | 0.328 |
| `claude-opus-5` | 4,226 | 8.77 | 20.79 | 40.49 | 67.76 | 8.45 / 19.32 | 23.99 / 34.77 | 14.62 | 662 | 26,147 | 0.865 |

All times are in seconds.

**Exceedance, P(latency > t):**

| t | Opus 5.5 | Astra | Opus 5 |
|---|---|---|---|
| 4 s | 0.78 | 0.90 | 0.92 |
| 6 s | 0.40 | 0.58 | 0.74 |
| **8 s** | **0.23** | 0.31 | 0.56 |
| 10 s | 0.15 | 0.16 | 0.43 |
| 15 s | 0.047 | 0.033 | 0.21 |
| 20 s | 0.014 | 0.010 | 0.11 |
| 30 s | 0.0015 | 0.0017 | 0.034 |

**Clustering.** Lag-1 correlation of consecutive mid-trial latencies is weak: 0.18 for Opus 5.5, 0.15 for Astra, 0.20 for Opus 5. For Opus 5.5, P(next > 8 s | this > 8 s) is 0.31, against a base rate of 0.18.

**Per-task Opus 5.5 p50.** It ranges from 4.62 s (Pack And Pour, 345 output tokens/call) to 6.87 s (Cap Pen, 578 output tokens/call). It tracks output length.

**Depth.** Opus 5.5 p50 by call index:

| Call index | 5–9 | 10–14 | 15–19 | 20–24 | 25–29 | 30–34 | 35–39 |
|---|---|---|---|---|---|---|---|
| p50 (s) | 4.95 | 5.12 | 5.22 | 5.29 | 5.31 | 5.63 | 6.45 |

That is +14% from index 5–9 to 30–34. Only 55/120 trials reach index 35, so the last bucket is confounded by survivorship. Depth cost was small here because history was mostly cache reads and images were evicted after 2 turns.

**Latency model** (least squares over 120 trials per model: Σlatency = a·calls + b·output_tokens):

| Model | Fixed per call | Marginal decode rate | R² |
|---|---|---|---|
| Opus 5.5 | 1.76 s | 87 tok/s | 0.91 |
| Opus 5 | 1.37 s | 68 tok/s | 0.94 |
| Astra | 2.46 s | 28 tok/s | 0.90 |

- Opus 5.5's 87 tok/s matches Artificial Analysis's independent 92.2 tok/s.
- Astra's 28 tok/s is half of AA's 51.3 tok/s (max); undercounted output or different prefill (UNVERIFIED).

**Wall-time decomposition (mean per trial).**

| Model | Wall time | LLM | Motion (steps/10) | Other (reset, IO, grading) |
|---|---|---|---|---|
| Opus 5.5 | 268.8 s | 84.5% | 28.0 s | 13.7 s |
| Astra | 250.7 s | 83.2% | 27.7 s | 14.4 s |
| Opus 5 | 435.2 s | 89.9% | 30.8 s | 13.4 s |

## 4. Cost per call (derived, recorded cache usage, list prices)

| Model | $ per call | $ per trial | Cache write | Output | Cache read | Tokens written to cache / read / output per call |
|---|---|---|---|---|---|---|
| Opus 5.5 | **0.0262** | 0.898 | $0.483 (54%) | $0.290 (32%) | $0.124 (14%) | 2,826 / 18,140 / 424 |
| Opus 5 | 0.0499 | 1.758 | $0.777 | $0.583 | $0.398 | 3,531 / 22,615 / 662 |
| Astra (recorded) | 0.1073 | 3.074 | $2.774 | $0.191 | $0.108 | 7,747 / 3,784 / 134 |

- The Astra page's cache-adjusted figure is $1.14 per trial.
- Opus 5.5 / Opus 5 = 0.51, which reproduces the page's "51% of the cost".

**Why writes dominate.** ~2.8k written tokens per call exceeds one new turn (3 × 64 image tokens + state + previous assistant turn). This fits critique Error 5: `policy.py:_evicted_view` rewrites user turns as frames age out of `image_horizon=2`, forcing re-writes from there on.

**Projection to the design's request shape** (derived, UNVERIFIED). Assumptions:
- an append-only history;
- a 10k-token cached prefix;
- a 1920×480 tile per turn, ≈ 69 × 18 = 1,242 image tokens under the ~28×28-px-per-token rule (verify with `count_tokens`);
- ~300 text tokens per turn;
- ~424 replayed output tokens per turn.

Then each call writes about 2.0k tokens and reads 10k + depth × ~2k:

| Depth | Approximate cost per call |
|---|---|
| 5 | $0.022 |
| 20 | $0.028 |
| 35 | $0.034 |

Per-call cost stays within 1.3× of Tier 1, so the design's ~$900 API budget is plausible; robot-hours remain the binding constraint.

## 5. Other configurations: what public data exists

| Configuration | Best available evidence | Status |
|---|---|---|
| Opus 5.5 low / high / per-message effort | None. AA lists only "Adaptive Reasoning, Max Effort" (92.2 tok/s; TTFT 704.69 s at max) | UNMEASURED |
| Opus 5.5 `speed:"fast"` | Docs: "up to 2.5x higher output tokens per second", "not time to first token"; $8/$40; research preview by account manager or waitlist; Claude API only; separate `anthropic-fast-*` limits; switching speed breaks the cache. **Adjacent measurement:** clapboardbench Opus 5 fast/high, 263 calls in 6 runs: p50 12.48 s, p90 24.46 s, p99 39.07 s, max 55.87 s; 2,404 output tokens/call (93% thinking); 92.6k input; fit **4.25 s + output/233 tok/s** (R² 0.95); about $0.31/call. If Opus 5.5 keeps its ~1.8 s fixed cost, fast mode at the Tier 1 shape projects to ≈1.8 + 424/218 ≈ **3.7 s mean**, vs 6.6 s observed (derived) | Projection only |
| `claude-sonnet-5-5` + `between_tools` (low) | Docs: "the lowest thinking setting on Claude Sonnet 5.5"; valid at low/medium/high; with it, per-message effort changes 400. AA lists only Sonnet 5.5 max (138.4 tok/s). Released 2026-09-28 | UNMEASURED |
| `claude-haiku-4-5` | AA snapshot 2026-08-20: 0.69 s TTFT, 101 tok/s, text only | No robot-shape data |
| `gpt-6-astra` low / medium | Tier 1 medium (above). Robocurve Astra/Fable report, medium: 5.3 / 11.4 s (bowl) and 6.1 / 10.0 s (puzzle) | Medium only |
| Astra `service_tier:"ultrafast"` | OpenAI guide: "fastest API service tier"; "strongly recommend WebSockets"; 500K TPM at tiers 1–3; US/global processing only; 6× price ($60/$6/$75/$300). No latency number. OrcaRouter (2026-09-30): no independent party has measured it. An OpenRouter row (1.65 s "P50 latency", 55 tps) is UNVERIFIED | UNMEASURED |
| `gemini-robotics-er-2-preview` pointing | Docs: `[y, x]` normalised 0–1000; "For thinking level use medium for a good balance between latency and performance"; "Query multiple times and average results"; caching and structured output supported (not on the streaming endpoint); $1 / $5 (cache $0.10) through 2026-12-31. **Adjacent measurement:** clapboardbench ER 2 as a joint controller at `reasoning_effort=high` via OpenAI-compat: 91 calls, 5 runs; **p50 5.81 s, p90 9.55 s, p99 15.6 s**; ~17k prompt tokens, 6–12 frames, 95 visible completion tokens (thinking not reported; latency uncorrelated with visible tokens, R² 0.03) | Pointing UNMEASURED |
| `gemini-3.8-flash` | AA (High): 244.7 tok/s, TTFT 17.48 s; $0.75 / $3.75 (cache $0.075) | No low-effort or robot-shape data |
| RoboDojo official Astra "14.66 s" | `RoboProbe .../efficiency.json`, ByteDance proxy. Twice Tier 1 Astra's 7.28 s mean on the native API | Proxy figure, as the critique says |

## 6. Consequences for HARNESS_DESIGN timing decisions

1. **Duty cycle (0.03–0.2 Hz).**
   - At medium effort the upper bound is not reachable. The LLM-only p50 rate is 0.19 Hz with zero motion, and the realized rate was 0.127 Hz.
   - Plan on ≈0.1 Hz at medium. Reaching 0.2 Hz requires cutting about 60% of output tokens (low effort, per-message effort for routine steps, a terse tool schema) or fast mode, and both need measuring.
2. **"Park if a loaded arm waits > 8 s."**
   - At the Tier 1 shape this fires about 8 times per trial. It would fire on every first call (cold cache plus initial planning), which always exceeded 8 s.
   - Issue the planning call while the arm is at rest. Pre-warm the prefix with a `max_tokens:0` request.
   - Base park decisions on load and torque, not on a fixed 8 s timer. A 15 s timer fires on 4.7% of calls.
3. **"2–4 calls per pick-place."** At steady state this costs ≈10–21 s of LLM time at p50 and ≈20–40 s at p90, plus about 0.8 s of motion per call. It is the dominant term in every task.
4. **Speculative next calls.** Latency autocorrelation is weak (0.18), so slow calls cannot be predicted from the previous call. Output-token count can be: it explains 91% of between-trial variance. Bound output length rather than speculate.
5. **Routing pointing to ER 2.** No ER 2 pointing latency exists. Its only real-robot calls (high thinking) had p50 5.8 s, about Opus 5.5's 5.3 s. A serial ER 2 query plus an Opus call roughly doubles a decision's latency, and k = 3 averaging triples the ER 2 share.
   - Run ER 2 concurrently, or only on explicit `point` requests.
   - Measure `thinking_level` low and minimal (Gap 2 overlaps).
6. **Caching.** Append-only history with a 10k prefix should roughly halve the 54% write share (UNVERIFIED projection). Mutating eviction should also be removed for the Opus 5.5 preserved-thinking check (critique Error 5).

## 7. The benchmark that still needs running (parts 2–4)

`gaps/gap-1-bench.py` is a PEP 723 script, run with `uv run`. It is byte-compiled and its `--dry-run` works offline. It is **untested against any API**.

**Per rollout** it drives a *live* append-only 35-turn conversation: one 1920×480 JPEG tile (from `--frames-dir`, e.g. Tier 1 MP4 frames) plus a state block per turn, a ~10k-token system prefix plus 8 `strict` tools, `tool_choice:auto`, genuine thinking/reasoning items replayed. Depths 5, 20 and 35 come from the same rollouts, so no synthetic thinking blocks are needed.

**Per call it logs** first-event, first-block and first-`tool_use` times, total time, status, tool-call count, uncached/cache-read/cache-write/output/thinking tokens, and dollars.

**Default matrix:** 10 rollouts × 35 calls = 350 calls per configuration (50 calls per ER 2 pointing variant).
- `opus55-{low,medium,high}`
- `opus55-permsg-low`: medium, then an effort-only system message to `low`, with beta `mid-conversation-output-config-2026-07-01`
- `opus55-medium-fast`
- `sonnet55-between-low`
- `haiku45`
- `astra-{low,medium}` and `astra-low-ultrafast`
- `er2-point-{low,medium}-k1` and `er2-point-low-k3`
- `gemini38flash-low`

The dry-run estimate is ≈ $285, of which ultrafast is $161.

**Run first (≈ $50):** `opus55-low`, `opus55-medium`, `opus55-permsg-low`, `opus55-medium-fast`, `sonnet55-between-low` and `er2-point-low-k1`.

**Analysis:** p50/p90/p99 per configuration and depth bucket, P(>8 s), decisions/min (60 / p50); report call 0 separately.

## Files

- `research/gaps/gap-1-latency-table.csv`: per-configuration summary for the 5 measured configurations.
- `research/gaps/gap-1-tier1-per-call.csv.gz`: 11,768 rows with `run_id, model, task, rig, call, attempt, endpoint, status, latency_s`, plus the trial-level output and input tokens per call.
- `research/gaps/gap-1-bench.py`: the unexecuted benchmark.

## Sources

- Robocurve, "Opus 5.5 on RoboDojo-RC Tier 1" (2026-09-23): https://robocurve.org/opus-5-5-robodojo-rc-tier-1/. Also used: `data/trials.csv`, `data/cells.csv`, 360 × `data/runs/*.json`, and 360 per-trial logs at `https://robodojo-tier1-artifacts.pages.dev/log/<host>/<rig>/<run>/`. All fetched 2026-10-02.
- Inspect Robots source, `plugins/inspect-robots-agent/src/inspect_robots_agent/{_anthropic.py,_responses.py,policy.py}` at `095172f`: https://github.com/robocurve/inspect-robots (local clone `repos/inspect-robots`).
- clapboardbench wire logs `2026-07-31/wire/*/scene-0-e0/calls.jsonl`: https://github.com/robocurve/clapboardbench (local clone).
- Robocurve GPT-6 Astra vs Fable 5.1 report, via `sources/robocurve-gpt6-astra.md`: https://openai.robocurve.org/gpt-6-astra/
- Anthropic fast mode docs: https://platform.claude.com/docs/en/build-with-claude/fast-mode
- Anthropic effort docs (per-message effort, Sonnet 5.5 `between_tools`): https://platform.claude.com/docs/en/build-with-claude/effort
- Anthropic pricing and prompt-caching rules, via the bundled Claude API reference (Opus 5.5 $4/$20, cache read $0.20; writes 1.25×).
- Artificial Analysis model pages, fetched 2026-10-02:
  - https://artificialanalysis.ai/models/claude-opus-5-5
  - https://artificialanalysis.ai/models/claude-sonnet-5-5
  - https://artificialanalysis.ai/models/gpt-6-astra
  - https://artificialanalysis.ai/models/gemini-3-8-flash
- Artificial Analysis snapshot 2026-08-20: https://github.com/robocurve/llm-token-speed
- OpenAI ultrafast guide: https://developers.openai.com/api/docs/guides/ultrafast-mode
- OrcaRouter ultrafast notes: https://www.orcarouter.ai/blog/gpt-6-astra-ultrafast-vs-gpt-6-astra
- OpenRouter Astra page: https://openrouter.ai/openai/gpt-6-astra (UNVERIFIED row)
- Gemini Robotics-ER docs: https://ai.google.dev/gemini-api/docs/robotics-overview
- Gemini pricing: https://ai.google.dev/gemini-api/docs/pricing
- RoboProbe efficiency file, via `landscape/engineering-challenges.md`: `repos/RoboProbe/results/l3_inspect_eef_official_2100/efficiency.json`
