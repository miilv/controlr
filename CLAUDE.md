# controlr — a robotics harness

A harness in the spirit of coding agents, but for a robot: camera frames + a system prompt
(the "robot operating manual") + a task → a vision LLM over an API → **low-level numeric
actions** (`MOVE ee_delta dx dy dz`, `GRIP`, `STATUS`) → safety envelope → robot → new frame +
feedback. Turn-based (the robot holds still while the model thinks); the transcript is
append-only and fully prompt-cached. The goal is an experimentation platform: can large API
models, with the right prompt and architecture, act as the controller without a VLA?

Today: UR3 CB3 + Robotiq in Isaac Sim 6.0 (PHANTOM's calibrated scene) on `compute3`;
models through one OpenAI-compatible router (omniroute). Next: the real UR3.

## Components

| Code | What |
|---|---|
| `controlr/llm/` | streaming client (timings, usage, early stop), decision head (`decisions.py`: Decisions API → grammar text), cache markers, append-only transcript, `FakeLLM` |
| `controlr/protocol/` | reply grammar (MOVE/GRIP/HOLD/STATUS) and feedback text |
| `controlr/prompts/` | system prompt = robot operating manual (`system_v0.md`), planner prompt |
| `controlr/observation/` | frame renderers: resize, grid, ee_marker, axes, diff, heatmap, tile |
| `controlr/robot/` | `Robot` ABC, `spec`, UR3 kinematics, `SafetyEnvelope`, mock, replay, `isaac/` (server in Isaac's python + client) |
| `controlr/loop.py`, `runlog.py`, `cli.py`, `bench/` | episode (planner + turn loop), run logs, CLI, cache/latency benchmarks, sweeps |
| `configs/` | experiments as YAML (`extends:`); every experiment axis is a config field |
| `scripts/` | deploy and run on compute3, Isaac server, run videos |
| `research/` | literature review and teardown of other harnesses |
| `runs/` | run outputs — **not in git** |

## Commands

```bash
uv sync --extra dev                         # environment
uv run pytest -q                            # unit tests (live/isaac skip themselves) — same as CI
uv run controlr prompt -c configs/sim_waffle.yaml     # the manual as the model sees it (no robot, no LLM)
uv run controlr run -c configs/mock.yaml --fake-llm   # dry run of the whole pipeline, no network

# compute3 (Isaac): deploy, then run — the Isaac server starts and stops by itself
scripts/deploy.sh
scripts/remote_run.sh run -c configs/sim_waffle_yaw.yaml --seeds 0,1,2,3
ssh compute3 'cd ~/controlr && CONTROLR_ISAAC=1 .venv/bin/python -m pytest -q tests/'   # ~20 min

# analysis
uv run controlr report runs/<run_dir> [--csv out.csv]
uv run --no-project --with pillow python scripts/turn_video.py runs/<run_dir> docs/video/<name>.mp4 "<title>"
uv run controlr bench-cache --model claude/claude-sonnet-5-5 --turns 20   # ⚠️ live calls
```

## Rules (mandatory)

**Step 0 of any code task: read [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) and [ARCHITECTURE.md](ARCHITECTURE.md) BEFORE the first change.** For an experiment task, also [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

1. **The model is the controller.** No high-level skills ("pick up the box") and no VLA in the
   action path. Alternatives are config switches, not code forks. Every experiment axis is a
   field in `controlr/config.py`; defaults never change silently (that breaks comparability
   across runs) — changing a default = a journal entry.
2. **Workflow:** branch (`git worktree add ../controlr-<task> -b feat/<task>` for parallel tasks)
   → tests locally → if you touched `robot/`, `loop.py`, safety or Isaac: `scripts/deploy.sh` and
   the Isaac tests on compute3 → **open the PR yourself** → green CI → merge. Direct push to
   `main` — docs only (journal, experiments, incidents).
3. **Live LLM calls cost money, but limits are generous.** Soft cap: don't burn more than
   ~5k calls a day, and don't waste calls on runs that can't teach anything. Unit
   tests use only `FakeLLM` / `httpx.MockTransport`; live tests are marked `@pytest.mark.live`
   and run only with `CONTROLR_LIVE=1`. Keys: only `OMNIROUTE_*` (and `OPENROUTER_API_KEY` for the decision head) from `.env`,
   **never** the agent session's credentials (`ANTHROPIC_AUTH_TOKEN` etc.). Actual spend (calls, tokens) goes into
   the report.
4. **compute3 is a shared machine** (someone else's work runs there too). Write only under
   `~/controlr*`; the GPU is ours, CPU/RAM in moderation; never touch other processes; stop the
   Isaac server when done. `apt` / drivers / reboot — only on explicit request
   ([RUNBOOK](docs/RUNBOOK.md)). PHANTOM (`~/phantom-icra-2027` on compute3,
   `~/skoltech/research` locally) is **never modified** — import it or copy with attribution.
5. **Secrets:** `.env` lives only locally and in `~/controlr/.env` on compute3 — not in dev
   copies, git, `runs/` or logs. New variable → `.env.example` +
   [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
6. **Cache invariants:** the transcript is append-only, history is never edited; an image is
   encoded once and stored as bytes; feedback text is deterministic (fixed decimals);
   reasoning effort never changes within an episode. Any change to serialisation → a prefix
   stability test + a `cache_read` check on a live run.
7. **Safety lives in code, not in the prompt.** Every action goes through `SafetyEnvelope`;
   clamps and stops are reported back to the model as feedback. The real UR3 — only through
   PHANTOM's drivers + `SafetyMonitor` and only with explicit per-session permission (a human
   at the e-stop).
8. **An experiment is reproducible or it doesn't count:** config in git, seeds × repeats, the
   same plan (`planner.plan_file`) when comparing control models, n and spread in the report;
   "2/4" is noise, not a finding. Report: `docs/experiments/YYYY-MM-DD-<slug>.md` + a row in the
   index; videos of successes and a typical failure via `scripts/turn_video.py`.
9. **Research agents have no side effects:** don't download models/video/audio, don't install
   torch/CUDA, clones are shallow in `/tmp` and get deleted; keep disk use in check (see
   [incidents](docs/incidents/)).
10. **Loose ends go in the same PR:** closed a [BACKLOG](docs/BACKLOG.md) item — strike it,
    found a landmine — add it; changed behaviour — update [ARCHITECTURE](ARCHITECTURE.md) /
    [DEVELOPMENT](docs/DEVELOPMENT.md); significant work — `docs/journal/YYYY-MM-DD-<slug>.md`;
    incident — `docs/incidents/`. Stable docs carry no dates or statuses; everything
    time-bound goes to journal/experiments.
11. **Git hygiene:** `git status` first; `git add` file by file (no `-A` / `.`); never commit
    `runs/`, `.env`, `research/repos/`; videos only when short and needed by a report.

## Where to look

| When | File |
|---|---|
| Any code task | [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — playbook: setup, workflow, tests, typical changes, pitfalls |
| How the system works, module contracts | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Design / read an experiment | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), reports in [docs/experiments/](docs/experiments/) |
| Something broke (compute3, Isaac, router, cache) | [docs/RUNBOOK.md](docs/RUNBOOK.md) + [docs/incidents/](docs/incidents/) |
| Pick a task / known landmines | [docs/BACKLOG.md](docs/BACKLOG.md) |
| Env variables, machines, paths | [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) |
| Context of past work | [docs/journal/](docs/journal/), code reviews in [docs/reviews/](docs/reviews/) |
| Why it's built this way, what others tried | [research/README.md](research/README.md) |
| How to give the agent tasks (for humans) | [HUMAN.md](HUMAN.md) |
