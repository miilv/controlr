# 2026-10-04 — GPT-6 Luna through the router; why compute3 turns took 10–20 s

**Task:** run the rotation pick-to-box with `gpt-6-luna` ("fast mode, low thinking") as the control
model (Ilia). The first live episodes took 10–20 s per LLM call; this entry is the debugging.

## Luna through omniroute (probes on recorded turns of `20261003T012711Z` / `20261004T094646Z`)

- Catalog: `cx/gpt-6-luna` (alias `codex/gpt-6-luna`; `-low…-max` ids), `cxa/gpt-6-luna`. `cx/` is
  listed as `api_format: responses`, but `/chat/completions` works — the router translates.
  `cxa/` → HTTP 401 "No active credentials for provider: codex-app-server".
- No `-fast` / `-priority` id for gpt-6 (those exist only for `dva/gpt-5-6-*`, which have no vision).
  `extra_body.service_tier: priority` (Codex "fast mode") is accepted, but the router reports no tier
  (`service_tier: null` on `/responses`), and 5 vs 5 calls showed no latency difference.
- Thinking (16 calls per arm, turns 1/4/11/13 × 4):

| `reasoning_effort` | calls that thought | p50 s | max s |
|---|---|---|---|
| `none` | 0/16 | 2.38 | 5.26 |
| `low` | 5/16 (113–282 tokens) | 2.46 | 10.07 |

  `low` thinking is hidden and arrives as one burst (+4–8 s per thinking turn). `minimal` is
  accepted (n=1, 0 reasoning). The `-low` id behaves like the `low` field (n=1). No field = thinks more.
- The usage chunk trails the STATUS line by up to ~0.5 s on `cx/` (Sonnet ≤ 0.2 s): with
  `usage_grace_s: 0.3` six of 30 turns logged no usage (cache share in `summary.json` wrong).
  The Luna config uses 0.6 s.

## Why 10–20 s: compute3's route to the router

Per turn the loop re-sends the whole transcript with every image as base64: 60 KB at turn 0,
~42 KB more per turn, 730 KB at turn 16. In the episodes the call time grew with the turn number.

| same request (turn 16, 726 KB) | header / end s |
|---|---|
| compute3, its own route | Luna 13.4–17.7 / 13.7–20.4; **Sonnet 9.4–9.9 / 9.8–10.3** |
| compute3 via an ssh reverse tunnel to the dev box's proxy | Luna 2.0–2.1 / 2.3–2.7 |
| dev box | Luna 1.6–2.1 / 2.3–2.6; Sonnet 1.6 / 2.0–2.4 |
| (yesterday, compute3, Sonnet eval turn 16, ~730 KB) | 1.6–1.8 / 2.1–2.2 |

compute3 has no wired link (`eno1` no carrier); it is on Wi-Fi (−43 dBm, fine) and reaches the
router through a VPN client (`tun0` → sing-box → xray, Happ). Upload of a 740 KB body to the router:
**42–138 KB/s** through that VPN (5–17 s, varying within the hour), 1.0 MB/s through the tunnel;
compute3 → dev box over ssh: 3 MB in 0.53 s. So the VPN exit's upload is the bottleneck, it varies
by day (yesterday's Sonnet runs did not see it), and it hits every model; Luna was only slower on
top of it (thinking turns + a ~0.3 s higher fixed cost).

**Workaround (used for the Luna round):** `ssh -R 127.0.0.1:18809:127.0.0.1:10809 compute3` from the
dev box (its HTTP proxy) and `CONTROLR_LLM_PROXY=http://127.0.0.1:18809 scripts/remote_run.sh …`,
which sets `HTTPS_PROXY` for the controlr process only (RUNBOOK §3). Fixing compute3's link
(cable, or a different route for the router host) is the owner's call.

## What the remaining ~2.4 s is (dev box, no thinking)

| part | measured |
|---|---|
| network + router, trivial `GET /models` (warm TLS) | 0.3 s first byte (TLS 0.24 s once) |
| fixed cost of any generation, 31-token prompt | Luna 1.15–1.4 s to first token; Sonnet 0.87–0.94 s |
| prefill of a real turn (~8.7k tokens, 17 images; only 2.8k cached on a fresh session) | ≈ +0.6–0.8 s |
| streaming ~25 tokens + router chunking | ≈ 0.3–0.4 s |

About 1.4–1.5 s per call is router + upstream overhead we don't control (`cx/` goes through the
Codex backend). Below 1 s needs a direct API, fewer image tokens, or server-side conversation state
(only the new frame uploaded per turn) — BACKLOG.

## Is ~1.5 s the floor? (test stand `scripts/router_latency_stand.py`, dev box, p50)

| measurement | result |
|---|---|
| `GET /models`, warm connection / new connection | 0.09 s / 0.37 s (TCP+TLS through the proxy chain ≈ 0.25 s) |
| chat to a nonexistent model (router parses, resolves, rejects; no upstream) | 0.09 s |
| Luna tiny prompt, Responses API, n=15: headers / first token / end | 1.07 / 1.39 / 1.82 s (p10 first token 1.18) |
| Sonnet tiny prompt, chat, interleaved n=15: first token / end | 0.99 / 1.45 s |
| Luna real turn 12 (7.7k tokens, 13 images, fresh session): first token / end | Responses 2.86 / 3.85 s, chat 2.06 / 2.94 s |
| episode replay turns 2–12, as today vs `prompt_cache_key` + `x-session-id` + echoed `x-codex-turn-state` | first token 1.71 vs 1.78 s, end 2.38 vs 2.40 s, cache share 0.82 vs 0.82 — no effect |
| gzip / deflate request body | router answers "Invalid JSON body" — not supported |
| 726 KB body, warm, dev box → router (no LLM) | 0.12–0.34 s (one stall of 19.5 s) |

Topology: compute3 (VPN exit Helsinki) or the dev box (exit NL) → omniroute in Frankfurt (OmniRoute
3.8.51 behind Caddy) → chatgpt.com Codex backend on a ChatGPT **Team** subscription
(`x-codex-plan-type: team`). OmniRoute source (tag v3.8.51, read by a research agent, not measured):
the pre-upstream work is a few ms (in-process SQLite, translation); it opens a new TCP+TLS connection
to chatgpt.com on most calls (32 single-connection agents, 4 s idle timeout); it holds a Codex stream
back until the first text delta (so `response.created` timing is not the backend's accept time);
a `keepalive` chat chunk means the handler took > 1 s. Per-phase timing exists server-side
(`OMNIROUTE_TRACE=true`, `/api/telemetry/summary` — 401 with our key). A WebSocket transport to Codex
exists, opt-in per connection (`codexTransport=websocket`). `service_tier` is forwarded
(`fast` → `priority`); `reasoning_effort: none` is forwarded as is.

Budget of a Luna loop turn (turn 12–16, via the tunnel): upload of the transcript 0.4–0.7 s (+0.04 s
per turn); router + Codex backend + model to the first token ≈ 1.0–1.4 s (tiny prompt ≈ 1.1–1.4 s, so
almost all of it is fixed); prefill of the uncached tail 0.2–0.5 s; ~25 tokens 0.2–0.4 s; waiting
for the usage chunk after the reply is complete 0.1–0.7 s (mean 0.3 s). httpx's 5 s keep-alive
expiry re-handshakes when calls are > 5 s apart (+0.23 s) — only 1 % of loop turns.

**Verdict.** Through `cx/` the first token never came under ~0.9 s even for a 31-token prompt, so
**< 1 s per call is not reachable on this route**; the best case for a loop turn is ≈ 1.3–1.6 s
(first token + the action line). What would move it: (1) harness — execute when the action line is
complete instead of waiting for STATUS + usage (−0.3–0.6 s), JPEG quality 75 at the same 448 px
(−41 % bytes, same image tokens), keep-alive 120 s; (2) router admin — `OMNIROUTE_TRACE` to split
OmniRoute vs Codex exactly, longer `FETCH_KEEPALIVE_TIMEOUT_MS` / fewer dispatcher connections,
Codex WebSocket transport; (3) a different upstream — OpenAI's API platform with an API key
(no Codex front door; needs a key and per-token spend), or another model family. Sonnet through
`claude/` is ~0.4 s faster to the first token than Luna on the same router.

**Side notes.** The aborted first round (effort `low`, through compute3's VPN; latency unusable,
outcomes valid since the robot waits): seed 0 and seed 1 both ended `unstable` at a re-grasp
`GRIP close` (runs `20261004T094646Z`, `20261004T095130Z`). A Kit `omni.telemetry.transmitter`
outlived its Isaac server by a few minutes (ppid 1), then exited by itself.

**Spend:** latency test stand ≈ 101 calls (tiny prompts mostly; 34 real-turn replays); probes ≈ 98 calls (Luna 83–85 incl. 2 tiny `/responses` and 5 tiny chat calls, 1 rejected `cxa/`
call; Sonnet 13), mostly 5–9k-token prompts; the aborted round 31 Luna calls (17 + 13 + 1).
