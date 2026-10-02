# 2026-10-02 — project structure modelled on Blick

**Done:** `CLAUDE.md` (+ `AGENTS.md` as a symlink), `HUMAN.md`, `ARCHITECTURE.md` at the root;
`docs/{DEVELOPMENT,EXPERIMENTS,RUNBOOK,ENVIRONMENT,BACKLOG}.md`; directories `docs/journal/`,
`docs/experiments/` (reports + index), `docs/reviews/`, `docs/incidents/` (two postmortems);
CI (`.github/workflows/ci.yml`: unit tests, compileall, bash -n, secret scan) and a PR template.
Old reports moved with dates: SMOKE_REPORT/ROTATION_REPORT → `docs/experiments/`,
INTEGRATION_NOTES → journal, review_*/FIXLOG → `docs/reviews/`; references in code/tests/docs
updated. `REPOS_MANIFEST.md` moved out of the ignored `research/repos/` into `research/`.

**Decisions:** a separate `docs/experiments/` genre (Blick doesn't have one) — this is a
research project, and run results shouldn't drown in the work log. All docs in English (first
drafted in Russian like Blick, switched at Ilia's request).

**Not done:** the `run-experiment` skill, Isaac tests in CI — in BACKLOG.
