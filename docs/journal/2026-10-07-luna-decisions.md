# 2026-10-07 — GPT-6 Luna Decisions as a decision head

**What the model is.** `openai/gpt-6-luna-decisions` (released 2026-10-06) is GPT-6 Luna served
through OpenRouter's alpha **Decisions API** (`POST /api/alpha/decisions`), not chat completions.
A request carries a `state` (text / JSON / images) and up to 200 named typed questions — `noul`
(P(yes)), `choice` (a probability per option + the likeliest), `score` (a probability per level of
an ordered rubric + the expected level). No text comes back. OpenRouter lists $0.10 / M input,
output free, P50 latency 0.26 s, 1.05 M context, ~95 % availability over the last 3 days.

**Access.** omniroute has no such model (its `cx/` / `codex/` / `cxa/gpt-6-luna*` are chat
variants) and answers 404 `unknown_route` on `/v1/decisions` and `/v1/alpha/decisions`. The decision
head therefore calls OpenRouter directly with `OPENROUTER_API_KEY` (agreed with the owner today;
CLAUDE.md rule 3, ENVIRONMENT.md, `.env.example`, CI secret scan updated). The key is not set up
yet: no live call has been made.

**Design (the model is still the controller).** A decision model cannot write `MOVE ee_delta 12 0
-8`, so the action becomes questions: `dx dy dz` (+ `dyaw`) as `score` over fixed levels (default
-30 -10 -3 0 3 10 30 mm, ±20 / ±5 deg), `grip` keep/open/close, `status` CONTINUE/DONE/FAIL — one
request per turn. `reduce=expected` takes the probability-weighted level, so uncertainty shrinks
the step on its own. The answers are rendered as ordinary grammar text, so parsing, the safety
envelope, execution, feedback and the run log are untouched; `DecisionsClient` has the
`LLMClient.complete` signature (`controlr/llm/decisions.py`, ARCHITECTURE "Decision head").
Alternatives considered: one `choice` over ~13 discrete moves (coarser, no expected value) —
rejected for now; it can come back as a question file + reduction mode.

**What changes structurally.** The API keeps no history, so the request is rebuilt each turn:
manual + turn-0 text (task, plan) + the last `decisions.history` (action, feedback) pairs + the
newest frame. Prompt caching plays no part; the transcript stays append-only for the log. The
planner stays a chat call (Opus through omniroute): `run_episode(planner_llm=...)`,
`make_control_client(cfg)`. Shorter cycles: `configs/sim_waffle_yaw_luna_dec.yaml` — 150 turns
instead of 30, no streaming / overlap.

**Open (BACKLOG P1 "Decision head, first light").** The Decisions schema does not show how images
go into `state`; `decisions.image_mode` (content parts vs a data-URL field) is a guess until
`tests/test_llm_decisions_live.py` (4 calls, left/right red-square check) passes for one mode.
Then a mock episode, then the 4 rotation seeds with pinned plans against the chat models.

**Update — key in place, image probe (4 calls, $0.0001).** `OPENROUTER_API_KEY` is set locally and in
`~/controlr/.env` on compute3. `tests/test_llm_decisions_live.py`, a red square left vs right in a
320x240 frame, asked "where is the red square?":
- `image_mode: parts` (JSON text part + `image_url` parts as `state`): left 1.00 / right 1.00 —
  the model sees the frame. 200 OK in 0.49 s and 0.28 s, 235 input tokens, $0.0000235 per call.
- `image_mode: field` (data URLs as a JSON field): left 0.59 / left 0.62, confidence 0.18-0.24 — it
  does not see the image (the URL is read as text, if at all).
The default (`parts`) is the working one; `field` is useless as it stands.
