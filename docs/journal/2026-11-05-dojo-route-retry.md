# 2026-11-05 — ROUTE: agentic route audit, bounded retry and cx/ replay

The agentic perceiver and held checks already share a configurable route (`_cfg.model`,
explicit model keyword or AGENTIC_MODEL, default Sonnet unchanged). Candidate `--setenv`
pins it, and freeze manifests record it. The supervisor is deterministic; AgentPolicy uses
a separate cx/ default. Full map, sizes, failure semantics, evidence and recommendation:
[report](../experiments/2026-11-05-dojo-route-replay.md).

httpx already had bounded transient retries and Retry-After. Corrected only the stdlib
fallback: seconds/date Retry-After capped at 60 s, every 5xx, closing error responses,
no final exhausted sleep. This is a robustness correction with no successful-request
serialization or result changes, so no new experimental flag. Existing constructor
max_retries=0 disables transport retries. New tests use httpx.MockTransport exclusively
(with an in-memory urllib adapter, no socket).

Added a dry-run-first teacher-forced request replay probe. It validates prompt pins,
request sizes, complete tool outputs and saved images; refuses overwritten tool images,
accepts only dev ids, and allows only cx/ live calls with at most 200 requests and zero
automatic retries. A reconstruction hash is saved; historical logs do not contain wire
hashes. It cannot execute tools or evaluate episode success.

Copied only explicitly named old dev rows from compute4's integ-c4/runs2 (read-only),
20 MB. No active measurement data, scoring-only ids, processes, GPU or writes on compute4.
CPU-only paid probe on compute3 under ~/controlr-route-probe, above the 10 GB disk guard:
13 cx/gpt-6-sol calls, 0 Claude/cc calls, 151325 input / 5388 output tokens.
All parsed; 7 answered immediately, 6 continued with tools. Exact geometry/identity
agreement 4/7 paired answers (4/13 overall); object-count agreement 7/7.
Median 13.22 s vs stored Sonnet 15.60 s is descriptive only (different dates/cache states).
Recommendation: keep defaults; validate cx/ on a paired dev camera A/B before adopting
or adding a fallback. No candidate, active deployment or measurement configuration changed.

Validation: focused client/perceiver tests passed; full local suite and CI results are
recorded in the PR. No robot/, loop.py, safety or Isaac changes; no deployment / GPU tests.
PR is left unmerged for lead review.
