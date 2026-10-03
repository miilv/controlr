# 2026-10-03 — thinking controls for Sonnet 5.5 through omniroute, token budget per turn

**Done:** 21 probe calls on real turns of eval run `20261003T012711Z_sim_waffle_yaw` (turn 7, 13 and
the turn that thought most, user turn 12); latency/token accounting over the 4 successful eval runs
(83 turns).

**Findings — thinking (user turn 12, the hard one):**

| Request | reasoning tokens | wall s | Reply |
|---|---|---|---|
| `claude/claude-sonnet-5-5`, no effort field (= current default) | 396, 293 | 5.7, 4.6 | clean |
| `reasoning_effort: high` | 244 | 4.3 | clean |
| `reasoning_effort: medium` | 0 | 2.2 | clean |
| `reasoning_effort: low` (×3) | 0, 0, 0 | 2.1, 2.0, 2.1 | clean |
| `thinking: {type: between_tools}` | 0 | 2.4 | prose reasoning written into the visible reply |
| `output_config: {effort: low}` | 425 | 6.1 | ignored by the router |
| `no-think/claude/claude-sonnet-5-5` | 221 (turn 13: 97) | 4.0 | still thinks |
| `claude/claude-sonnet-5-5-low/-medium/-high` | — | — | HTTP 400 "not available in the active live catalog" |
| `thinking: {type: disabled}` | — | — | HTTP 400: use `between_tools` |

So the eval ran at the model's default effort, which behaves like `high` (adaptive: median 118,
p90 364, max 675 reasoning tokens per turn). `reasoning_effort: low|medium` removes thinking on this
turn; whether medium still thinks on harder turns is untested (n=1 per level except low).

**Findings — latency:** over 83 eval turns, LLM time = 3.3 s mean (p50 2.85, p90 5.0);
fit: ≈ 2.0 s + reasoning_tokens / 116 tok/s (r = 0.99). Time to response headers is a stable
1.3–2.0 s (router + model TTFT + prefill of a ~10k-token, 94 %-cached prompt).

**Findings — tokens per turn:** turn 0 = 6,952 prompt tokens (manual + task + plan + one image).
Each later turn adds ≈ 290 input tokens: image ≈ 200 (448×336 ≈ w·h/750), STATE + labels ≈ 40,
the previous visible reply ≈ 40 (thinking is not replayed). New content per turn ≈ 70 % image /
30 % text; at turn 18 (~12k tokens) the context is ≈ 58 % fixed prefix, 30 % images, 12 % turn text.
Visible output ≈ 40 tokens per turn.

**Decisions:** none yet — `reasoning_effort` is a config field already (`llm.extra_body`); the next
experiment compares default vs `low` (and `medium`) on the same seeds.

**Spend:** 21 calls (Sonnet 5.5), ~180k prompt tokens.
