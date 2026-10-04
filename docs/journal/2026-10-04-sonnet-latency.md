# 2026-10-04 — Sonnet 5.5 through the router: where a turn's time goes, what helps

Same breakdown as for Luna ([luna-latency](2026-10-04-luna-latency.md),
[turn-latency](2026-10-04-turn-latency.md)), for `claude/claude-sonnet-5-5` (the router's `claude`
provider = a Claude subscription). Dev box unless noted; replays of the Sonnet eval episode
`20261003T012436Z` (seed 0), same session per replay, 2 s between turns, turns 1–11.

## Router and network

- verf-new → api.anthropic.com: RTT 6.5 ms, new TLS 55 ms (Anthropic's edge is in Frankfurt too).
- Router call log, our key: Sonnet TTFT today (tiny prompts) avg 0.91 s, **min 0.55 s**; the
  2026-10-03 eval (default effort) avg 2.7 s, min 1.06 s; no router-added waits. The router does not
  log the effort it sends to Claude.
- Tiny prompt from the dev box: first token ≈ 0.92–1.39 s, end ≈ 1.4–1.5 s (Luna: 1.39 / 1.82 s).

## Thinking: `low`, never `none`

| `extra_body.reasoning_effort` | headers | first token | **STATUS word p50 / p90 / min** | turns that thought |
|---|---|---|---|---|
| `low` (2 replays) | 1.33 | 1.68 | **1.78 / 2.55 / 1.63** | 2/22 (333 tokens) |
| `low` on the `cc/` route | 1.30 | 1.67 | 1.75 / 1.91 / 1.53 | 1/11 |
| unset (= adaptive thinking, the 10-03 eval) | 1.29 | 3.39 | 3.39 / 5.10 / 1.70 | 8/11 |
| **`none`** | 1.28 | 4.12 | **4.12 / 5.48 / 1.67** | **17/22 (4845 tokens)** |

`none` (and presumably `minimal`) is not a Claude effort: the router falls back to the default and
Sonnet thinks MORE. `config.validate` now rejects `none`/`minimal` on Claude routes. Cache share
0.97 in every arm. Through the router, Sonnet `low` reaches the STATUS word about as fast as Luna
`none` (≈ 1.85 s) and its call ends earlier (1.85 vs 2.38 s).

## What the ~0.7 s above a tiny prompt is

Turn 11 with a warm prefix (the last user message varied by whitespace so the router cannot
replay a response), STATUS word, n = 4–6 per variant, two sessions an hour apart:

| variant | bytes | prompt tokens (cached) | STATUS word |
|---|---|---|---|
| all images, 448 px (as run) | 520 KB | 10.1k (9.9k) | 2.14–2.86 s; 2.21–2.56 s |
| only the newest image | 63 KB | 8.0k (7.7k) | 1.90–2.09 s |
| text only | 21 KB | 7.8k (7.7k) | 1.42–1.53 s; 1.76–1.91 s |
| all images, 336 px | 379 KB | 9.1k (8.9k) | 1.68–2.76 s (p50 2.17) |
| all images, 224 px | 232 KB | 8.4k (8.3k) | **3.8–5.5 s: thought on 5 of 6 calls** |

So the newest frame costs Sonnet ≈ 0.4–0.5 s to process (it is the uncached part), the cached older
frames + the upload another ≈ 0.2–0.4 s. 336 px saves ≈ 0.2 s (within the hour-to-hour drift of
0.3–0.4 s); 224 px makes Sonnet reason about what it cannot see — slower. Also: a stray "(check N)"
appended to a turn was enough to make `low` think on 3 of 12 calls.

## Optimisations for Sonnet, by size

1. **`reasoning_effort: low`** (vs the default the eval used): STATUS word 1.78 vs 3.39 s p50 —
   the biggest single lever. Its effect on success is the open BACKLOG effort experiment.
2. **Execute on the STATUS word** (`llm.overlap_tail`, merged): −0.1–0.3 s (Sonnet's usage chunk
   trails by ≤ 0.2 s).
3. **compute3's uplink**: the router path re-uploads the whole transcript every turn (Anthropic has
   no server-side conversation state); at ≥ 1 MB/s that is ≤ 0.5 s at turn 16, at 50 KB/s 15 s.
   `scripts/net_check.sh` measures it without LLM calls. compute3's VPN at 16:00: 740 KB in
   14.7–15.5 s (≈ 50 KB/s), TLS handshake 1.0–1.9 s, `GET /models` 7.7–8.8 s.
4. **Frame size 336 px** as an A/B for success (≈ −0.2 s); not 224.
5. Not worth it: a direct Anthropic path (the router adds ~10 ms + ~60 ms; it would need a
   separate subscription token — the agent session's credentials are off-limits), `speed`/
   `service_tier` fields (accepted, no visible effect, not logged upstream), the `cc/` route.

Floor for Sonnet through this router: ≈ 1.4–1.5 s to the STATUS word on a text-only turn, ≈ 1.6 s
with one image (best calls); < 1 s is not reachable here either.

**Spend:** ≈ 100 Sonnet calls (6 replays × 12 turns, 2 × 4 × 5–7 warm-prefix calls, 5 tiny calls);
router read-only (SQL timing columns for our key, no bodies).
