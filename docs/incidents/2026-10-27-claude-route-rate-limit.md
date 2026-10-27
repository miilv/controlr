# 2026-10-27 — omniroute's Claude account rate-limited for 61 h (every claude/ and cc/ route)

**Symptom:** from 18:24Z every `claude/*` and `cc/*` call through omniroute answered 429 "This request would exceed your
account's rate limit ... reset after 61h". cc/ maps to the same account. The agentic perceiver of `submit_v2` (Sonnet
5.5) fell back to B1 after about 130 s of retries per call, on every box.

**Root cause:** the Claude subscription behind omniroute has its own weekly limit. The perception bake-off's
Perceiver-models stream ran 6-13 parallel agentic sessions (Sonnet, Opus, Fable) under "no LLM cap". Between 17:30Z and
18:24Z it made about 1,900 calls: 29.6M prompt tokens (25.0M cache reads) and 1.86M completion. Integration's rendered
runs used the same account at the same time. Nobody knew the limit; the "no cap" decision was based on the OpenAI
subscription's weekly limits.

**How it was fixed:** all of the stream's LLM runs were stopped at about 18:58Z. The 429 fallbacks were excluded from the
results. The lead stopped all LLM calls until a decision: wait for the reset (about 30 Oct 08:00Z), move the agentic
perceiver to a cx/ route, or set agentic off.

**What changed so it won't repeat:** DEVELOPMENT.md pitfall. Before a multi-arm LLM sweep on `claude/`, estimate its
tokens against the account's weekly budget and check `retry_after` / 429s after the first minutes.

**Lessons:** "no cap" applied to the OpenAI subscription, not to the Claude account. Parallel Opus/Fable sessions burn
the weekly budget in an hour. A 429 with a multi-day reset should stop the runners at once instead of retrying.
