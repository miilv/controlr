# How to develop controlr (playbook for humans and agents)

Rules — [CLAUDE.md](../CLAUDE.md), design — [ARCHITECTURE.md](../ARCHITECTURE.md),
experiments — [EXPERIMENTS.md](EXPERIMENTS.md). This file: how to make changes.

## 0. Bootstrap

```bash
git clone git@github.com:miilv/controlr.git && cd controlr
uv sync --extra dev                 # python >= 3.10; deps: httpx, numpy, pillow, pyyaml
cp .env.example .env                # OMNIROUTE_BASE_URL / OMNIROUTE_API_KEY (key from the owner)
uv run pytest -q                    # must be green with no network and no GPU
uv run controlr run -c configs/mock.yaml --fake-llm   # the full loop on the mock robot, 0 calls
```

Isaac and the GPU are only on `compute3` (`ssh compute3`, see [ENVIRONMENT.md](ENVIRONMENT.md)).
You don't need Isaac locally: the mock and replay robots cover everything except physics.

## 1. Task workflow

```bash
git status                                            # don't touch anyone else's changes
git worktree add ../controlr-<task> -b feat/<task>    # or just a branch if it's the only task
# ... changes + tests for them ...
uv run pytest -q                                      # unit
uv run controlr prompt -c configs/sim_waffle.yaml     # touched prompt/grammar/feedback? re-read it as the model
scripts/deploy.sh                                     # touched robot/, loop.py, safety, isaac/
ssh compute3 'cd ~/controlr && CONTROLR_ISAAC=1 .venv/bin/python -m pytest -q tests/'
gh pr create --fill                                   # open the PR yourself; CI = unit tests + compileall + secret scan
gh pr merge --merge --delete-branch                   # after green CI
git worktree remove ../controlr-<task>
```

Live runs (money) — only in a task with a budget, following [EXPERIMENTS.md](EXPERIMENTS.md).

## 2. Tests: three tiers

| Tier | What | How to run | Where |
|---|---|---|---|
| unit | everything without network or Isaac: grammar, feedback, cache markers, transcript, kinematics, safety, loop on `FakeLLM`+`MockRobot`, client on `httpx.MockTransport` | `uv run pytest -q` | locally + CI |
| isaac | Isaac server, motion tracking, kinematics-vs-sim consistency, scripted expert | `CONTROLR_ISAAC=1 pytest -m isaac` (full run ~20 min) | compute3 only |
| live | real calls through the router | `CONTROLR_LIVE=1 pytest -m live` | by budget, manually |

Rule: new logic → a unit test. Changed motion execution / contacts / scene → an isaac test.
Changed message serialisation → a prefix stability test (`tests/test_llm_transcript.py`) and
one live run checking `cache_read` (see RUNBOOK §cache).

## 3. Module map: what to change and how to verify

| I want to | Touch | Verify |
|---|---|---|
| a new action format / mode | `protocol/grammar.py` (+ `config.ActionConfig`), `robot/safety.py` | `test_grammar` (spec examples parse under every config), `test_safety`; `controlr prompt` |
| feedback text / STATE line | `protocol/feedback.py` | `test_feedback` (determinism!), `controlr prompt` |
| robot manual / planner | `prompts/system_v0.md`, `prompts/planner_v0.md`, `prompts/builder.py` | `test_prompts`; a new prompt variant = a new `system_vN.md` + `prompt.system: system_vN`; never edit the old one (comparability) |
| observation (overlays, diff, tiling) | `observation/renderers.py` | `test_renderers` (determinism, projecting a known point) |
| a new Isaac task | `robot/isaac/tasks.py` (+ server if new objects are needed) | `test_isaac_tasks` (no GPU) + scripted expert in the isaac tests; look at the frames (`docs/img/`) |
| a new backend (real UR3 etc.) | `robot/<backend>.py`, factory in `robot/__init__.py` | the `robot/base.py` contract; mock tests as the template |
| cache / router / timings | `llm/` | `test_llm_*`; live `bench-cache` (budget) |
| an experiment axis | field in `config.py` + `configs/base.yaml` | `test_cli` / `test_loop`; the default must not change behaviour |

## 4. Pitfalls (already stepped on)

- **omniroute replays responses** to byte-identical requests (~0.4 s, no usage). That's why
  `llm.request_nonce: true` puts the run id into the first turn. Benchmarks too.
- **Effort through the router is a model-id suffix** (`claude/claude-opus-5-5-xhigh`);
  `extra_body.reasoning_effort` is effectively ignored.
- **Opus 5.5 can't turn thinking off**; Sonnet 5.5 thinks and can exhaust `max_tokens` → an empty
  reply. `llm.max_tokens` counts thinking tokens too. For fast turns use `no-think/...` routes.
- **On `claude/` / `cc/` the router sets the cache boundary itself**; our markers there seem to
  have no effect (not confirmed by a probe — see BACKLOG). Haiku 4.5 doesn't cache prefixes
  under 4096 tokens.
- **Isaac runs at ~0.24× real time**: a 100 mm move ≈ 4 s — often more than the LLM.
- **PhysX blows up** on a squeezed object; such episodes end as `unstable`, and the server now
  survives NaNs — but check the server log when things look odd.
- **The prompt and grammar must match the config**: examples in the manual are generated from
  `grammar_spec`; don't hand-write examples.
- **A single angled D435** gives poor depth: 80–200 mm depth errors are the main failure cause
  so far (hypothesis in BACKLOG).
- **`uv run` in the repo root creates `.venv`**; on compute3 `deploy.sh` builds `~/controlr/.venv`.

## 5. What lives where

| What | Where | Rule |
|---|---|---|
| agent rules | `CLAUDE.md` (`AGENTS.md` is a symlink) | short, no dates |
| architecture and contracts | `ARCHITECTURE.md` | updated in the PR that changes behaviour |
| playbook / operations / environment | `docs/DEVELOPMENT.md`, `RUNBOOK.md`, `ENVIRONMENT.md` | stable, no dates |
| experiment protocol | `docs/EXPERIMENTS.md` | stable |
| experiment reports | `docs/experiments/YYYY-MM-DD-<slug>.md` + index in README | dated, never edited after the fact |
| work log | `docs/journal/YYYY-MM-DD-<slug>.md` | dated |
| code reviews and fixlog | `docs/reviews/YYYY-MM-DD-<slug>.md` | dated |
| incidents | `docs/incidents/YYYY-MM-DD-<slug>.md` | postmortem within a day |
| task queue | `docs/BACKLOG.md` | strike items in the task's PR |
| literature | `research/` | `research/repos/` are clones, not in git |
| images / videos for docs | `docs/img/`, `docs/video/` | only what a report links to |
| runs | `runs/` | not in git; reports reference the run dir name |
