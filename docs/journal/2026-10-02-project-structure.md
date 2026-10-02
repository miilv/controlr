# 2026-10-02 — структура проекта по образцу Blick

**Сделано:** `CLAUDE.md` (+ `AGENTS.md` симлинком), `HUMAN.md`, `ARCHITECTURE.md` в корне;
`docs/{DEVELOPMENT,EXPERIMENTS,RUNBOOK,ENVIRONMENT,BACKLOG}.md`; каталоги `docs/journal/`,
`docs/experiments/` (отчёты + индекс), `docs/reviews/`, `docs/incidents/` (два постмортема);
CI (`.github/workflows/ci.yml`: юнит-тесты, py_compile, bash -n, secret scan) и шаблон PR.
Старые отчёты перенесены с датами: SMOKE_REPORT/ROTATION_REPORT → `docs/experiments/`,
INTEGRATION_NOTES → journal, review_*/FIXLOG → `docs/reviews/`; ссылки в коде/тестах/доках
обновлены. `REPOS_MANIFEST.md` вынесен из игнорируемого `research/repos/` в `research/`.

**Решения:** отдельный жанр `docs/experiments/` (в Blick его нет) — проект исследовательский,
результаты прогонов не должны тонуть в журнале работ. Стабильные доки — по-русски (как в Blick),
`ARCHITECTURE.md` и `research/` — по-английски (контракты и источники, уже написаны так).

**Не доделано:** скилл `run-experiment`, Isaac-тесты в CI — в BACKLOG.
