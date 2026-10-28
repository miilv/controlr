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

## Verdict, after the route came back (bench 16:34-17:31Z)

The health probe got 3 × 200 in a row (15:34, 16:04, 16:34Z), and the bench then ran at full pace. All 94 scenes
answered: the 6 outage fallbacks were re-run, and 2 first-turn "no JSON object" replies recovered. Paired against the
recorded arms on the same frames (B1: 290 within tolerance, 84 gross), with risky_b1 0.2:

| arm | within tol | gross | fixed / broken / still gross | phantoms | LLM calls (med / max per scene) | prompt / completion k-tok | s per call med | s per scene med / p90 |
|---|---|---|---|---|---|---|---|---|
| Sonnet 5.5, run 1 | 310 | 52 | 36 / 4 / 48 | 13 | 307 (3 / 6) | 4365 / 265 | 9.3 | 31 / 57 |
| Sonnet 5.5, run 2 | 310 | 50 | 37 / 3 / 47 | 10 | 305 (3 / 6) | 4432 / 265 | 9.1 | 31 / 54 |
| Opus 5.5 | 314 | 43 | 42 / 1 / 42 | 14 | 400 (4 / 8) | 6456 / 333 | 10.8 | 47 / 80 |
| **cx/gpt-6.1-sol** | **306** | **62** | **24 / 2 / 60** | 8 | 293 (3 / 8) | 3167 / 76 | 14.3 | 44 / 83 |

Fixed / broken per task, cx/gpt-6.1-sol: pack_random 10 / 1, organize_table 5 / 1, deposit_coin 2 / 0,
general_pickup 2 / 0, store_tools 2 / 0, build_tower 1 / 0, put_bottles 1 / 0, stack_bowls 1 / 0, all others 0 / 0.

- **Rule.** The pre-registered rule needed fixed >= 34 and broken <= 4-5. cx/gpt-6.1-sol fixes 10 too few, so it is
  **not a candidate**.
- **No second cx model.** 6.1-sol was the strongest listed; a weaker sibling is unlikely to gain 10 fixes (lead).
- **Total cx spend:** 328 calls (incl. the outage attempts and 13 pings / probes), 3.3M prompt tokens (2.8M cache
  reads), 78k completion. Files: compute4 `~/dojo-data/pmodels/paired_cxsol61.md`, `cxstat_cxsol61.txt`.
