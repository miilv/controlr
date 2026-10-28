# 2026-10-28: the cx/ (Codex) route was mostly unavailable for the agentic perceiver (stream S, side task)

**Context.** The Claude route's weekly limit is binding: about 1,230 agentic calls today, and 2,500+ projected for the
queued runs. So the lead asked for an offline check of cx/ models as the agentic perceiver, on the 94-scene / 84-gross
bench of 2026-10-27-perceiver-models: same frames and B1 hint, H2, `arbitrate(risky_b1=0.2)`. A cx/ model would become
a candidate for a new preset only if its fixed >= Sonnet's - 2 and broken <= Sonnet's + 1. Sonnet's recorded arms
score 36 / 4 and 37 / 3; the paired scorer reproduces those exactly. 0 Claude calls.

**What happened**
- The router lists vision cx/ models gpt-6.1-sol / -terra, gpt-6-sol / astra / luna, and gpt-5.5 / 5.6-*. I chose
  cx/gpt-6.1-sol.
- 14:09-14:56Z: 8 perceive calls (`dev_d0 run`, compute4, CPU only). Only 2 answered. The other 6 fell back to plain B1
  on HTTP errors:
  - 503 "[codex/gpt-6.1-sol] Unavailable (reset after 2-7 min)": 9 calls;
  - 429 "All credentials for model gpt-6.1-sol are cooling down" and "Codex is at capacity": 2;
  - 402 "Payment Required (reset after 2 min)": 2.
- 14:31Z: one ping each to gpt-6.1-sol, gpt-6-sol, gpt-6-astra-high and gpt-5.5-high. All four got 503 with the same
  reset time, so the outage is backend-wide, not per model. At 14:13Z, 6.1-sol and 6-sol had answered and
  astra-high / 6.1-terra / 5.5 had not.
- The client's 3 quick retries cannot bridge resets of 2-10 min. A one-scene-at-a-time runner with 5-min waits managed
  about 1 scene per 10 min, about 15 h for the bench, so it was stopped.
- When it answered: about 13 s per call median, 3-7 calls per perceive.
- Spend: 30 cx calls, 187k prompt tokens (158k of them cache reads), 3k completion.

**Finding.** Today a cx/ perceiver preset would mostly run as plain B1: 6 of 8 perceives fell back. Availability decides
a runtime preset as much as quality does. The recurring 402 suggests the Codex subscription is out of quota or has a
billing problem; Ilia has been asked.

**Next.** A health probe on compute4 (`~/dojo-data/pmodels/cx_probe_then_bench.sh`): one 1-token call to cx/gpt-6.1-sol
every 30 min, at most 20, status codes logged to `cx_probe.log`. After 3 successes in a row it runs the bench at full
pace (about 1-1.5 h), then H2 + arbitration risky_b1 0.2 and the paired score against Sonnet / Opus
(`paired_cxsol61.md`).
