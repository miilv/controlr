# Review: LLM layer and caching (lens "caching-llm"), 2026-10-02

Scope: `controlr/llm/*`, the caching-relevant parts of `controlr/loop.py`, `controlr/bench/*`,
`controlr/runlog.py`, and the live run logs in `runs/` (the same 7 live runs exist on
compute3:`~/controlr/runs`). I made no live LLM calls and changed no code.

How I checked the logs: for each turn of each live run I took the normalised usage and computed:
- `total` = prompt_tokens (reads and writes included);
- `unc` = total − read − write;
- `Δ` = total_N − total_{N−1};
- the previous visible reply size = completion − reasoning tokens.

Key identity found (holds for every steady-state turn of all 7 live runs):

```
Δ − unc  ≈ previous assistant reply (+≈5 framing tokens)
write_N  ≈ unc_{N−1} + reply_{N−1}
```

Example, 20261002T125503Z turn 5: write 414 = unc_4 399 + 15. Example, 20261002T125321Z turn 3:
Δ 363, unc 307, reply_2 61.

So the uncached part is exactly the newest user message, and the cached prefix ends right after the
newest assistant reply. The steady-state read share per turn is 0.87–0.94 (median ≈ 0.89 for Haiku and 0.91 for Sonnet). That matches the
integration notes. Their explanation of why does not match.

No blockers found. Majors: 1–5.

---

## 1. MAJOR — The cache breakpoints come from omniroute's `cc` provider, not from `apply_cache_markers`. The docs credit the wrong marker.

**Where:**
- `controlr/llm/caching.py:13-21`
- `ARCHITECTURE.md:97-108`
- `docs/journal/2026-10-02-v0-build-integration.md:72-80`
- `configs/base.yaml` `llm.cache`, `llm.cache_ttl`

**Problem.** controlr marks three places: the system message, the last user message and the
second-to-last user message. The logs show the cache boundary somewhere else: at the end of the
**last assistant message**, which controlr never marks.

**Evidence.**
- The identity above. If the second-to-last-user marker set the boundary, the uncached part would be
  `reply_{N−1} + user_N`. The logs show `user_N` only:
  - 125321 turn 3: unc 307 while Δ is 363.
  - Sonnet runs: Δ − unc = 23–28, which is the visible reply length.
- Turns that followed an empty `(no reply)` placeholder show Δ − unc = 5. That is the placeholder's
  token count.
- The planner requests carry **no markers** (`run_planner` sends `tr.to_messages()` raw), yet every
  planner call wrote its system prompt to the cache:
  - 110436: write 5225
  - 125503: write 5358
  - 131109: write 5552
  - 130156 then read 5358.
- The response headers show `x-omniroute-provider: cc` on every turn, including the `claude/` and
  `no-think/claude/` ids. `cc` is a Claude-Code-style upstream (`anthropic-ratelimit-unified-*`
  headers), which places its own `cache_control`.

**Consequences.**
- `llm.cache: none` versus `anthropic` and `cache_ttl: 1h` are probably no-ops on Claude routes. A
  sweep over these axes would measure nothing.
- The 1 h TTL has never been verified.
- The "keep a read point inside the 20-block lookback" rationale in `caching.py` is not what keeps
  reads working.
- If omniroute changes its strategy, caching changes silently.
- If omniroute forwards our 3 markers plus its own, the request may exceed the 4-marker limit, and
  which markers survive is undefined.

**Fix.**
1. Run 2 small probes (about 6 calls) with the existing `bench-cache`:
   - `cache=none` on `claude/claude-sonnet-5`: if read shares are unchanged, our markers are ignored.
   - `cache_ttl=1h`, then a 6-minute pause, then 1 call: does it still read?
2. Rewrite the Caching contract and the docstring to describe what is observed. Keep
   `apply_cache_markers` for non-`cc` Anthropic upstreams.
3. Add a per-turn boundary check to the run log and summary: `unc ≈ tokens of newest user turn`.
   This makes any future change visible.

## 2. MAJOR — An unreported cache split inside one episode. The code does not detect it.

**Where:**
- `runs/20261002T105745Z_sim_reach/turns.jsonl`. This run is missing from the notes' live-run table.
- `controlr/runlog.py:219-242`: the summary computes only the episode share.

**Evidence** (Haiku, normalised usage):

| turn | read | write | note |
|---|---|---|---|
| 3 | 4460 | 413 | prefix 4873 cached |
| 4 | **0** | **5250** | full miss, although the 4873 entry exists |
| 5 | 4873 | 721 | reads turn 3's entry, so it was still alive |
| 6 | 5594 | 350 | |
| 7 | 5250 | 1074 | reads **turn 4's** entry, not 5944 from turn 6 |

There are two independent caches in one episode. Requests alternated between at least 2 upstream
accounts or orgs.

This contradicts INTEGRATION_NOTES "from the first cacheable turn on, every call reads everything up
to the previous user turn". Headers were not logged then; they are logged now.

The current headers show that `x-omniroute-session-id` changes between turn 1 and turn 2 in every run.
It looks content-derived. The CORS allow-list advertises `X-OmniRoute-Lease-Owner`,
`X-OmniRoute-Lease-Generation` and `x-omniroute-connection` request headers, which are candidates for
pinning a run to one upstream.

**Fix.**
- Check in `summarize_turns`: flag turns with `read_N < read_{N−1} + write_{N−1} − 64`, ignoring the
  first cacheable turn. Report `cache_regressions` in `summary.json` and print a CLI warning.
- Find out whether omniroute honours a lease or session header, and send one stable value per run.
- Until then, report the per-turn read share (not only the episode share) in the notes.

## 3. MAJOR — Latency logs drop the fields needed to interpret TTFT. On thinking routes, "TTFT" includes thinking.

**Where:**
- `controlr/loop.py:109-111` (`_timings_dict`). This also feeds `planner.json`.
- `controlr/runlog.py:236-238`

**Problem.** `Timings.ttft` is the first *content* delta. On `claude/claude-sonnet-5` and Opus that
point comes after thinking:
- 131109 turn 6: ttft 9.40 s with 658 reasoning tokens.
- t_complete − ttft ≈ 0.1 s.

`_timings_dict` keeps only `ttft`, `t_complete` and `t_end`. It drops:
- `ttft_any` (first reasoning or content byte);
- `t_wall` (includes retries and backoff);
- `reasoning_text` and its length.

So the logs cannot separate queue and network latency from thinking time. They also cannot show
whether the router streams thinking at all, which matters for finding 7.

The notes' table labels the Sonnet numbers 4.02 / 6.06 s as TTFT. Those are mostly thinking time.

`percentiles()` also silently skips `None`: the 6 `finish_reason=length` turns of 130156 have no
ttft. As a result, `llm_ttft_s` and `llm_complete_s` are computed over a different and biased subset
from `llm_end_s`.

**Fix.**
- Log every `Timings` field, plus `reasoning_chars` and the `reasoning_tokens` already in usage.
- Add `llm_ttft_any_s` to the summary, plus a count of `None` values next to each percentile.
- Rename the column in the notes.

## 4. MAJOR — The latency benchmark is invalid for thinking routes and for Haiku as configured

**Where:**
- `configs/bench/latency.yaml:17` (`max_tokens: 40`)
- `controlr/bench/cache_probe.py:29-40` (`PROBE_SYSTEM`)
- `controlr/bench/latency.py:1-11` (claims steady-state cached)

**Problem.**
- **max_tokens.** `max_tokens` includes thinking. `claude/claude-sonnet-5` used 14–695 reasoning
  tokens per turn, and thinking cannot be disabled on `claude/claude-opus-5-5`. With 40 tokens these
  cells end at `length` with empty text: ttft is None, "status p50" is None, and "end" is the time
  to 40 thinking tokens. This is the integration "fix 4" bug, still present in the bench.
- **Prompt too short for Haiku's cache.**
  - `PROBE_SYSTEM` is about 2.2–2.4k tokens: 8708 characters, `estimate_tokens` gives 2419. Its
    comment claims "> 4096 tokens with a few images".
  - Haiku 4.5 needs 4096, and the runs show the first write only at about 4.45k.
  - Images are about 50 tokens at 224 px, about 200 at 448 px, and about 450 at 672 px.
  - So these Haiku cells are measured **uncached**, while the bench reports them as steady-state:
    - h=1 at all sizes;
    - h=10 at 224 px, and borderline at 448 px.
  - Haiku is the model the ~1 s target depends on.

**Fix.**
- Use `max_tokens` ≥ 2000. Early stop at STATUS keeps the cost low.
- Pad `PROBE_SYSTEM` to at least 5k tokens, using deterministic reference text.
- In `markdown_table`, flag measured cells with share < 0.5 as "UNCACHED".

## 5. MAJOR — The benchmarks send byte-identical requests across runs, so omniroute replays responses

**Where:**
- `controlr/bench/cache_probe.py:126-147`
- `controlr/bench/latency.py:130-145`

**Problem.** The probe and the latency matrix use a fixed `PROBE_SYSTEM`, deterministic synthetic
frames, a fixed `PROBE_REPLY` and no nonce. A second `bench-cache` or `bench-latency` run, or a
repeated sweep, sends exactly the requests of the first run.

omniroute answers repeated requests from its response cache: about 0.4 s and no cache counters. This
is documented in `LLMConfig.request_nonce`, INTEGRATION_NOTES fix 1 and the `test_llm_live.py`
docstring. The loop got the nonce; the benches did not. The bench rows would then report about 0.4 s
latency and usage without cache counters, with no error. The `x-omniroute-cache: HIT|MISS` header,
which would expose this, is not recorded by either bench.

**Fix.**
- Put `RUN <out_dir name>` at the start of the first user turn, not in the system prompt, so the
  provider's prompt cache is still shared.
- Record `x-omniroute-cache` per call. Exclude HIT rows from the medians, or fail on them.

## 6. MINOR — Planner isolation: the docs claim cache sharing that cannot happen

**Where:**
- `controlr/llm/caching.py:14-15`: "also shared with the planner call's system prompt".
- INTEGRATION_NOTES: "With the planner, the system prompt was cached from the planner call; the
  second Sonnet run read 5358 of 5706".

**Problem.**
- The planner runs on a different model (opus-5-5-xhigh). Prompt caches are per model.
- The planner system text is manual + `---` + planner instructions. Its breakpoint (omniroute's)
  sits at a different position.
- The 5358 read was the 130156 **planner** reading the 125503 **planner's** entry. Control turn 0
  read 0 in both Sonnet runs.
- Isolation of the control transcript itself is correct: only the plan text enters user turn 0.
- Each planner call pays a 1.25× write of about 5.5k tokens that is reused only if another planner
  call runs within 5 minutes.

**Fix.** Correct both texts.

## 7. MINOR (plausible, unverified) — Planner read timeout, then a full-cost retry

**Where:**
- `controlr/llm/client.py:299-300, 420-422`
- `controlr/loop.py:180`

**Problem.**
- The planner shares the 180 s per-read timeout.
- Opus xhigh produced its first content at 54–83 s, at about 85 reasoning tokens/s.
- With `max_tokens: 16000`, a long think lasts about 190 s.
- If the router does not forward thinking deltas or pings (unknown, finding 3), httpx raises
  `ReadTimeout`. The client treats that as `_Retryable` and re-runs the whole planner up to 3 more
  times: about 12 minutes and 4× the cost, then no plan.

**Fix.**
- Give the planner its own timeout (`planner.timeout_s`, ≥ 600).
- Do not retry a `ReadTimeout` that happens after a 200 response arrived for non-idempotent-cost
  calls, or cap planner retries at 1.
- Log `ttft_any`.

## 8. MINOR — The usage grace period is on the robot's critical path and is not a real deadline

**Where:** `controlr/llm/client.py:410-413`

**Problem.**
- The grace condition is evaluated only when a `data:` chunk arrives. SSE comments, pings and a
  stalled upstream do not trigger it, so after STATUS the call can block for up to `timeout_s`
  (180 s) while the robot waits.
- In the current logs t_end − t_complete is 0–0.2 s. That is real robot idle time, spent on
  statistics.

**Fix.** Either:
- enforce a wall-clock deadline: read the stream in a thread, or use a short per-read timeout during
  grace; or
- return the frozen text immediately and finish draining usage in the background, then attach it to
  the turn record before the next request.

## 9. MINOR — Early-stop truncation is chunk-granular

**Where:** `controlr/llm/client.py:414-417`, `controlr/protocol/grammar.py:447-456`

**Problem.**
- The stored reply is cut at whatever chunk completed the STATUS word, for example
  `STATUS OK moving above est` (131109 turn 0).
- This text is cached and re-sent on every later turn. The model sees its own truncated notes, and
  the cut point depends on the router's chunking.
- Usage counts the tokens after the cut, which were discarded.
- Caching itself is not affected.

**Fix.** Freeze at a deterministic point: the end of the STATUS line, or strip the trailing partial
note to `STATUS <WORD>`. Document that `completion_tokens` includes discarded tokens.

## 10. MINOR — Usage normalisation depends on route-specific heuristics

**Where:** `controlr/llm/client.py:128-133`

**Problem.**
- "Writes excluded from `prompt_tokens`" was measured only on provider `cc`.
- A route that includes writes but reports `cache_creation_input_tokens` would double-count them.
- "`raw < read` ⇒ reads excluded" is ambiguous when a route excludes reads but `uncached > read`:
  reads are then not added, and the total is undercounted.

**Fix.**
- Decide by key semantics: `prompt_tokens_details.cached_tokens` present means OpenAI semantics
  (reads included).
- Log `x-omniroute-provider` with the usage.
- Add an in-run consistency check: total_N ≈ total_{N−1} + Δ, the same arithmetic as this review.

## 11. MINOR — Turns without usage silently drop out of cache stats. FakeLLM diverges from the real client.

**Where:** `controlr/runlog.py:219-242`, `controlr/llm/fake.py:82`

**Problem.**
- `token_totals` and `cache_read_share` skip turns with `usage=None`, with no count reported.
- `FakeLLM` returns `usage=None` on every early stop, while the real client with grace returns usage.
  Fake-run summaries therefore have almost no token or cache data, although the fake.py docstring
  says "every field populated".

**Fix.**
- Add `n_turns_without_usage` to the summary.
- Have FakeLLM return usage on early stop when the client's `usage_grace_s` > 0, or always.

## 12. MINOR — No guard for the 20-block lookback

**Where:** `controlr/loop.py:153-160`, `controlr/observation/renderers.py:257-280`

**Problem.**
- Blocks per turn = 1 assistant + feedback text + images + legend.
- 6 cameras × (raw + diff + heatmap) without `tile` gives 18 images + 2 text blocks + 1 assistant
  reply = 21 blocks between consecutive breakpoints. Anthropic would then miss the previous entry
  every turn, whichever side places the markers.
- Today's configs use 3–4 blocks per turn.

**Fix.** At renderer construction, warn or raise when the maximum number of blocks per turn exceeds
18. Suggest `tile: true`.

## 13. MINOR — Cross-episode system-prompt stability depends on the measured reset state

**Where:** `controlr/prompts/builder.py:393-419`

**Problem.**
- The tool-axis lines are computed from `obs.state.tcp_rotvec` after reset, a physics-settled
  measurement, and printed with 2 decimals.
- Settle jitter near a rounding boundary changes the system prompt. That loses the cross-episode
  system cache, which the contract claims ("identical across episodes of one config").

**Fix.** Compute the line from the commanded start pose (`task.params.start_q` → FK), or round
coarser, for example to whole degrees.

## 14. MINOR — What `ttft` includes, and a stale comment

**ttft includes the request upload.**
- `t0` is taken before `httpx` serialises and uploads the body.
- Each 448×336 frame is about 31.5 KB of JPEG, about 42 KB as base64. A 40-turn transcript is about
  1.7 MB per request, all re-uploaded every turn.
- On a slow uplink that adds tenths of a second, which appear as model TTFT.
- Fix: log request body bytes per turn, and possibly the time to first response byte (headers)
  separately from the first content delta.

**A comment in `tests/test_llm_live.py` contradicts the logs.** It says the
`no-think/claude/*` route "returns only prompt/completion/total — cache counters are dropped". Every
live Haiku run here (`no-think/claude/claude-haiku-4-5-20251001`) has
`cache_read_input_tokens`, `cache_creation_input_tokens` and `prompt_tokens_details.cached_tokens`.

---

### Verified OK

- **Transcript byte-stability.** Images are encoded once, the data URL is computed once, and past
  messages are pure functions of the stored records. The tests pass:
  `tests/test_llm_{caching,client,transcript,fake}.py` and `tests/test_loop.py`, 73 passed.
- **Marker count.** Local script, 40 turns: 3 markers on [system, user N−1, user N]. The input is
  never mutated.
- **Legend text.** The renderer legend text is always non-empty, so our last-user marker lands on a
  text part, not an image.
- **Float formatting.** `fmt_num` gives fixed decimals with no `-0`.
- **Nonce placement.** The nonce is in user turn 0, not in the system prompt.
- **Planner transcript isolation.** The control transcript receives only the plan text.
- **Request serialisation cost.** 0.04 s at 40 turns, even with large noise images.
