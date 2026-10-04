# 2026-10-04 — turn latency: harness overlap, the router's share, a direct Codex client

Follow-up of [luna-latency](2026-10-04-luna-latency.md): where the remaining ~1.5 s per call goes
and what can be cut (Ilia: harness changes; router settings if harmless to other clients; can the
Codex backend be reached directly through the subscription).

## 1. Harness: execute on the STATUS word (`llm.overlap_tail`, **new default: on**)

Before: the loop waited for the call to return — after the STATUS word the client kept reading up
to `usage_grace_s` for the rest of the STATUS note and the usage chunk (0.1–0.7 s, mean 0.3 s on
`cx/`, ~10 % of a Luna call). Now the call runs on a worker thread; a wrapped `stop_when` hands the
loop the reply as it stands at the STATUS word; the loop parses and executes it while the call
finishes. The transcript gets the call's final reply, so stored messages and executed actions are
the same as without overlap (unit tests: the robot starts moving while the call is still open;
overlap on/off give byte-identical `messages.jsonl` and the same executed actions; replies that
never complete fall back to the classic path). The record gets `timings.exec_overlapped` and
`timings.llm_tail`; `timings.llm_wall` is now the time the loop waited for the reply.

**Default change** (rule 1): `llm.overlap_tail: true` in `config.py` and `configs/base.yaml`.
Comparability: what the model sees and what the robot does are unchanged; turn cycle times get
shorter by the usage wait, and `timings.llm_wall` means "until the reply was complete" (before:
until the call returned). Runs before 2026-10-04 used the old meaning.

Also: `LLMClient` keeps idle connections 120 s (`KEEPALIVE_S`; httpx's default 5 s re-handshakes,
+0.23 s through the proxy chain, when a slow robot turn is longer than 5 s).

## 2. The router's share (verf-new, read-only)

omniroute = the central OmniRoute gate on verf-new (Frankfurt), shared by every client agent.
Its own telemetry (`/api/telemetry/summary`, last 19 requests): parse 0, validate 7, policy 0,
resolve 0, finalize 0 ms; `connect` (upstream until it answers) p50 1476 ms. `call_logs` for our
key (`robot`), 242 Luna calls today: router-measured TTFT avg 1.9 s, min 0.72 s, `added_wait_ms`
null (the router adds no waits). From verf-new to chatgpt.com: RTT 6.6 ms, a new TLS connection
≈ 60 ms (OmniRoute reconnects on most calls), an unauthenticated POST to the Codex endpoint is
rejected in 60–75 ms. So of a ~1.5 s call: router ≈ 10 ms + reconnect ≈ 60 ms; the Codex backend
(auth, queue, prefill, first token) is the rest.

**Nothing changed on the router.** The only levers (`FETCH_KEEPALIVE_TIMEOUT_MS`, dispatcher
connections) save ≤ 60 ms per call and need a restart of the shared gate (in-flight requests of
every client drop); a longer keep-alive also adds a small stale-socket risk for everyone. The Codex
WebSocket transport is a per-connection setting, and both Codex connections are shared.

## 3. Direct to the Codex backend

The protocol is in the open-source Codex CLI (`codex-rs`, v0.160.0; read by a research agent):
`POST https://chatgpt.com/backend-api/codex/responses` (Responses API, SSE) or a WebSocket to the
same URL (`OpenAI-Beta: responses_websockets=2026-02-06`, messages `{"type":"response.create",…}`);
headers `Authorization`, `ChatGPT-Account-ID`, `originator: codex_cli_rs`, `version`, `session-id`;
`store:false`. On a WebSocket, a turn whose input extends the previous one can be sent as only the
new items + `previous_response_id` (the server keeps the state per connection; 60-min connection
limit, `previous_response_not_found` → resend in full). Auth: ChatGPT OAuth; refresh tokens are
single-use and rotate (`refresh_token_reused`), so controlr has its own device-code login
(`controlr/llm/codex_auth.py`, `scripts/codex_login.py`); the router's login is never reused.
From compute3 chatgpt.com needs the VPN or the tunnel (Skoltech Wi-Fi blocks it).

Stand: `scripts/codex_direct_stand.py` replays an episode (manual, recorded user turns, the model's
own replies) over HTTP (whole conversation per turn) or WebSocket (new turn only).

Login: device code approved by Ilia; token file `~/.controlr/codex_auth.json` on the dev box (access
token valid 240 h). For the compute3 runs it was copied to `~/controlr-secrets/` and deleted
afterwards (no refresh could happen in between).

Results (Luna, effort `none`, replay of `20261004T103138Z` turns 0–11, the model's own replies;
"complete" = the STATUS word, i.e. when the loop starts the robot; turns 1–11, n = 11 per run):

| path | sent per turn | created p50 | first token p50 | **complete p50 / p90 / min** |
|---|---|---|---|---|
| router (chat), dev box (earlier replay of the same run) | 0.1–0.7 MB | — | 1.71 | ≈ 1.85 (end 2.38) |
| direct HTTP, dev box | 309 KB (whole conversation) | 1.56 | 1.79 | 2.03 / 2.96 / 1.31 |
| direct WebSocket, dev box, priority (2 runs) | 57 KB (new turn only) | 0.51 / 0.64 | 1.09 / 1.54 | 1.21 / 1.70; p90 2.6–3.3; min 1.04 |
| direct WebSocket, dev box, no tier | 57 KB | 0.54 | 1.25 | 1.50 / 1.76 / 1.27 |
| direct WebSocket, 2 s gaps, no tier (2 runs) | 57 KB | 0.55 / 0.49 | 1.10 / 1.29 | 1.39 / 1.70 |
| + `generate:false` prewarm after each reply (2 runs) | 57 KB | 0.52 / 0.51 | 1.11 / 1.33 | 1.35 / 1.49 (cache share unchanged 0.86) |
| + `detail: low` images (2 runs) | 57 KB | 0.50 / 0.49 | 1.03 / 1.05 | 1.26 / 1.25 (input tokens unchanged) |
| direct WebSocket from **compute3, own VPN** | 57 KB | 0.84 | 1.63 | 1.85 / 2.63 / 1.51 |
| direct WebSocket from **compute3, tunnel to the dev box** | 57 KB | 0.54 | 1.65 | 2.03 / 2.16 / 1.55 |
| (router from compute3 via tunnel, Luna round: first token ≈ 2.1–2.4) | | | | |

Network: dev box → chatgpt.com (through its proxy) 0.3 s first byte on a new connection, ~0.1 s
warm. So on a WebSocket ≈ 0.4 s goes to the backend before `response.created`, ≈ 0.5–0.8 s from
there to the first token (prefill of ~7k tokens, 86 % cached, + the new image), ≈ 0.15–0.25 s to
the STATUS word.

**Verdict.** Direct WebSocket saves ~0.3–0.6 s per turn against the router (dev box 1.2–1.5 s vs
≈ 1.85 s; compute3 1.85–2.0 s vs ≈ 2.3 s) and makes compute3's slow VPN irrelevant (57 KB per
turn). It does **not** reach < 1 s: the best single turn was 1.04 s. `service_tier: priority`,
a prewarm and low-detail images showed no effect beyond noise. Below 1 s would need fewer prompt
tokens per call, a faster model, or OpenAI's API platform.

**Landmines.**
- **Moderation false positives**: `invalid_prompt: your prompt was flagged as potentially violating
  our usage policy` on ordinary robot turns — 3 of ~70 direct calls in the first runs, 1 of 36 in
  a dedicated run (≈ 3–4 %), sometimes after the reply had started streaming. Through the router:
  0 in ~400 calls. A direct transport needs a retry (new connection, full input) for it.
- **The manual must go in `instructions`.** As a developer message with empty `instructions` (the
  CLI's layout) Luna wrote prose instead of commands on all 36 turns.
- Even with `instructions`, one turn 0 came back as prose ("We need command move toward …").
- After an error on a WebSocket, the failed response can keep streaming on the same socket: drop
  the connection and resend the whole conversation on a new one (`previous_response_not_found`
  otherwise).
- `reasoning.context: "all_turns"` is mandatory with the `x-openai-internal-codex-responses-lite`
  header (400 otherwise).

**Spend (this entry):** direct stand ≈ 270 calls on controlr's own login (20 replays × 12 turns + 22
prewarm `generate:false` requests + smoke tests), 5–8k-token prompts; router read-only (one telemetry
GET, SQL reads of timing columns for our key only).
