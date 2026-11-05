# 2026-11-05 — Can cx/gpt-6-sol replace the agentic perceiver route?

**Recommendation:** keep the pinned default until a rendered dev A/B validates a route change.
Use cx/ for that A/B; do not spend Claude capacity during the sealed measurement.
A cx/ fallback after 429 is a plausible hybrid experiment, but is not validated or enabled here.
A weekly quota cannot be repaired by short backoff.

## Call map and configuration

Audited base: `97107b8e70f2cf1771a1c3d36f82e0d395c327c7`.
The supplied incident count (466 calls / 29 cells, two cells losing policy calls) was not
independently re-read: sealed and scoring-only data remained outside this investigation.

| Component | Invocation / configuration | Request |
|---|---|---|
| Agentic scene perceiver | `RuntimeConfig.agentic="budget"`, `agentic_max` (default 2, submit presets 3), `agentic_count_held`; policy `_agentic` → perception service `agentic_estimate` → `record.agentic_estimate` → `j1.perceive_impl` → `runner.run_agent` | Fresh conversation, 8 logical turns by default (`AGENTIC_MAX_TURNS`); 6000 output tokens per request, including reasoning; at most 6 commands and 6 attached tool images per turn; history accumulates |
| Held check | `RuntimeConfig.held_look` / service → `j1.held_look_impl`; `AGENTIC_HELD` default `b1_then_agent`; confident B1 avoids the model | Asked-arm wrist + head (up to 2 full PNGs); 4000 output tokens; up to 2 logical calls to obtain yes/no |
| Supervisor | `dojo.runtime.supervisor.RuleSupervisor` | Deterministic instruction/world/event planner; **zero LLM calls** |
| Optional AgentPolicy | `dojo.agent.loop.AgentConfig.model`, default `cx/gpt-6.1-sol`; separate from the rule-policy candidate | 16000 output tokens, manual_v6, max 30 calls; 3 images/turn, first long edge 640 px then 448, JPEG q80; `llmio.make_client` defaults 600 s / 3 extra attempts |
| Generic harness | `controlr.config.LLMConfig.model` default Sonnet; `PlannerConfig.model` default Opus xhigh | Separate control/planner path, not the candidate's agentic perception configuration |

Scene and held calls share the **existing model configuration field** `j1._cfg()["model"]`.
Precedence: explicit `model=` keyword → `AGENTIC_MODEL` → `claude/claude-sonnet-5-5`.
The scene record function already accepts `model=`; both entry points use `_cfg`.
No model-route code change is needed. A candidate file may pin the service process with
`--setenv AGENTIC_MODEL=cx/gpt-6-sol`; candidate parity resolves its `setenv`, and
`freeze_manifest.routes` records the effective model and source. No CANDIDATE.txt lives in this
checkout, and no active candidate file on compute4 was opened. An omitted setting uses the code
default, not `controlr.config.LLMConfig.model`. `AGENTIC_EXTRA_BODY` supplies optional effort etc.
The credential dotenv reader reads only OMNIROUTE names, so placing AGENTIC_MODEL in that file
alone does not configure the service.

`prompt.system_prompt(kind)` concatenates the versioned preamble and the entire vendored
PERCEIVER-PROTOCOL, including for held checks:

| Kind | Preamble | System chars | System MD5 |
|---|---|---:|---|
| robodojo | runner_robodojo_v1.md | 28134 | 908d4d286b80e5aa56075e265142c2a5 |
| robodojo_multi (scan) | runner_robodojo_v2.md | 28694 | b5c5ce4168b0529c0cdcd054da57bea2 |
| held | runner_held_v1.md | 24964 | 841740ffbffa7954bd55259c453e3261 |

Scene inputs are unresized RGB PNGs: normally head + both wrists (640×480 in this sample),
plus any scan observations, instruction, gripper commands, measured flange poses, B1/prior hints.
Tool images may be zoomed. There is no fixed total image or input-token limit across history.
The 13 replayed requests had 3–6 images (median 4), 1.46–2.02 MB serialized requests and
10,059–13,635 cx/ input tokens (median 11,431). Token counts depend on route; system characters
are the reproducible prompt-size measurement, not a token estimate.

## Failure handling and the fix

`make_chat()` chooses httpx `OmniChat`, falling back to stdlib `UrllibChat` when unavailable or
`AGENTIC_HTTP=urllib`. Both have a 240 s read timeout and **2 extra attempts** by default.

The primary httpx path already retries 429, every 5xx, transport failures, 408/409/425 and
retryable empty stream errors. It uses exponential jittered backoff (0.5 s base, 20 s cap) and
Retry-After seconds / HTTP-date (60 s cap). Exhaustion returns an error; other HTTP 4xx return
without retry. The existing unit tests cover this path. No extra retry wrapper was added.

Before this change, urllib retried 408/429 and selected 5xx/network failures, but ignored
Retry-After and slept even after its final failed attempt. It now accepts seconds/date
Retry-After (60 s cap), retries all 5xx, closes HTTP error responses, and sleeps only before
another attempt. Without a usable header it retains its 2/4/... s backoff (20 s cap).
`max_retries=0` still disables transport retries. This is a pure robustness fix: no successful
request's body, prompt, model, parsing or result is changed; only failed-request recovery and
waiting change. No experimental switch or default-model change is introduced.

After transport exhaustion, `run_agent` retries an empty errored logical turn with unchanged
messages; two accumulated errors terminate it (`llm_error`), spending that logical budget.
Normally two failed logical calls can therefore mean **6 HTTP attempts**. On no final answer,
`perceive_impl` uses B1 if `AGENTIC_FALLBACK=b1` and a hint exists; otherwise no agent objects.
The runtime `_agentic` also retains B1 after a perception-service exception. Agentic estimate
budget is still consumed. A held check tries twice at the logical layer and then returns
`held=None`, confidence 0, error "no answer", not false; runtime consumers retain uncertainty.
This explains why surviving an episode can still mean losing model-assisted perception.
`ChatReply` discards httpx attempt counts, so report `calls` counts logical calls, not HTTP
attempts. Held log summaries do not preserve the full raw request/reply conversation.

## Replay protocol and spend

No renderer, GPU, simulator, tools or policy ran. Read only old dev workspaces at
`compute4:~/dojo-data/integ-c4/runs2/c4b_{swap,ttt}_gcon_10000{0,1,2,3,4}/agentic`.
Copied 20 MB locally and to `compute3:~/controlr-route-probe/data`; all compute3 writes stayed
under `~/controlr-route-probe`. compute3 had 123 GB free (above the 10 GB guard).
No sealed directories, measurement ids, layout ids >=100100, other processes or GPU were used.

Selected one request per workspace: the last stored Sonnet reply containing an answer.
Reconstructed its preceding conversation with original Sonnet assistant text and stored tool
results. Verified system prompt MD5, untruncated outputs, saved PNGs and serialized request
byte counts for **every preceding request**. 13 of 17 workspaces passed; one lacked a saved
projected image and three had a request-size mismatch. The probe also refuses repeated tool
image filenames because overwritten images cannot reconstruct their earlier contents.
Historical logs contain no wire hash: the saved request SHA256 is of the reconstruction,
and matching byte counts plus prompt pins are consistency checks, not proof of byte identity.
Missing extra-body settings or overwritten files cause rejection where detectable.

Changed only `model` to cx/gpt-6-sol; output budget remained 6000; no added effort, temperature
or nonce. `OmniChat(max_retries=0)` prevented hidden automatic retries. Source data are
October 28; cx/ replies were collected November 5 from compute3 using OMNIROUTE keys from
`~/controlr/.env`. **13 live cx/ HTTP calls; 0 Claude/cc calls.** No aborted paid runs or probes.
151,325 input tokens (93,696 cached), 5,388 output tokens including 3,913 reported reasoning
 tokens. Dollar billing is not exposed by these replies; no dollar estimate is inferred.

Command (with the repository on PYTHONPATH and AGENTIC_ENV_FILE set):

```bash
python scripts/perception/route_request_replay.py ~/controlr-route-probe/data \
  --live --limit 13 --model cx/gpt-6-sol --out ~/controlr-route-probe/results.jsonl
```

Local raw evidence: `runs/route-replay/results.jsonl`; remote: `~/controlr-route-probe/results.jsonl`.
Compact per-request provenance and metrics: [JSON](2026-11-05-dojo-route-replay.json).
No frames or raw live transcripts were committed. No videos apply to a request-only probe.

## Results and limits

| Metric | Result |
|---|---:|
| HTTP errors / empty replies | 0 / 13 |
| Replies parsed by existing JSON/markup parser | 13 / 13 |
| Answer vs Sonnet's answer at this request | 7 / 13 (53.8%) |
| More tool commands vs Sonnet's answer | 6 / 13 (46.2%) |
| Object-count agreement among paired answers | 7 / 7 |
| Exact geometry/identity entry-set agreement among paired answers | 4 / 7 (57.1%); 4 / 13 overall |
| Latency median [min, max], stored Sonnet | 15.60 [5.87, 23.92] s |
| Latency median [min, max], live cx/ | 13.22 [9.95, 22.09] s |

Geometry/identity agreement expands B1 `keep` references using the stored hint, then compares
sorted `(category, asset, pos, quat, size, top_z, value)` entries exactly. It ignores names,
confidence and prose; this is agreement with Sonnet, **not correctness against truth**.
The four exact matches were 10-object tic-tac-toe hints retained by both models.
Two swap answers preserved all six B1 objects but omitted Sonnet's size/top-height overrides
and wrist-look suggestions. The remaining tic-tac-toe answer found the same two objects but
used B1 positions rather than Sonnet's measured board/piece positions, and suggested a
different occlusion look. None is established as a better answer by this probe.

The six continuations wanted more geometric measurements (`locate`, `btn.py`, etc.). They
are valid runner actions, not schema failures, but their latency and eventual answers are
unknown because tools were deliberately not executed. Two replies carried trailing non-JSON
characters; the existing outer-object parser recovered their command JSON. There were no
HTTP failures to measure fallback effectiveness.

This is 13 decisions from 9 eligible episode rows, 2 tasks, one draw per request, with some
within-episode dependence. It conditions on Sonnet's history rather than cx/'s own trajectory.
Timing compares different dates, cache states and routes; it cannot establish a latency win.
There is no held_look accuracy test, ground-truth scoring or episode-success measurement.

## Smallest useful next dev A/B

After the measurement, use compute3 rendered dev camera rows on swap_blocks and
play_tic_tac_toe, ids 100000–04, 2 repeats each: 20 episodes per arm, identical candidate,
calibration, seeds, runtime preset and prompt; vary only AGENTIC_MODEL. Pin run configs and
freeze manifests. Read held checks, full agent turns, logical and HTTP attempts, failure rate,
latency, token usage and wall-budget exhaustion; assess harmful belief writes and success.
This is a gate for gross regressions, not a powered success claim. A Sonnet A arm must wait
for explicit release of the Claude budget. Until then, a cx/-only dev smoke can establish tool
execution compatibility. A hybrid fallback additionally needs injected-429 tests and explicit
source-route reporting; silently routing a measured candidate after 429 would change its
inference procedure. No active measurement or default was changed by this work.
