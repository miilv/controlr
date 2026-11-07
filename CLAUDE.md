# controlr: a robot harness where a model is the controller

Two lines of work share this repo:

- **RoboDojo (the main work).** The `dojo/` stack plays the RoboDojo benchmark: Isaac Sim 5.1, two ARX X5 arms,
  three RGB cameras and 42 tasks / 54 cells. It has a runtime and planner, a skill library, camera-only perception
  (silhouette fitting B1, a twin-trained segmenter, glyph readers, an agentic perceiver on Claude Sonnet), a
  generator for dev layouts, and a camera A/B pipeline on the GPU boxes. This is joint work with Mikhail's team.
- **The original controlr.** A vision LLM emits low-level numeric actions (`MOVE ee_delta dx dy dz`, `GRIP`,
  `STATUS`) for a UR3 in Isaac (PHANTOM's scene) and a real SO-101, through a safety envelope; turn-based, with a
  fully prompt-cached transcript. See "Original controlr" below.

## THE HARD RULE (Ilia, 7 Nov 2026; it cannot be changed)

Our RoboDojo system is **one general system**:

- a model planner composes **general skills** (pick, place, push, pour, press, open, insert, handover, ...) and
  perception queries;
- it writes code against the same primitives only when no skill fits or a skill fails;
- code that works becomes a candidate general skill, and is promoted only if it holds on other layouts (the ratchet);
- one perception path and one object catalogue serve every episode.

**Forbidden:**

- task names in prompts or code: `if task == ...`, task-named vocabularies, task-keyed switches or routes, per-task
  prompt banks;
- per-task builds or entries;
- values taken from task configs or checkers (positions, counts, slots, thresholds); perceive or measure them;
- evaluator-termination feedback ("the episode is still running, so the attempt failed");
- the three PyRUA-RSI tricks: picking blocks into the dustpan instead of sweeping, kong's slot→group table, and
  termination as a retry signal.

The board says the same. RoboDojo's submission rules, verbatim:

> "Each submission shall represent a single fixed system ... follow the exact same inference procedure across all
> tasks. An agent or planner inside that system is allowed ... We do not allow a router that switches between
> different policies or checkpoints by task."

A Codex audit found that our D-C19 system (48.44) is *not* defensibly a single system: tasks.json lookups,
`ported_plan` and checker rules. See [audit](docs/journal/2026-11-07-codex-legality-audit-controlr.md) and
[decision](docs/journal/2026-11-07-general-system-rule.md). The task-keyed parts listed in BACKLOG are to be
retired. Per-cell work is how we discover a general mechanism, never a task program.

## Joint work with Mikhail (Misha's team)

- **Joint system:** their model planner (`planner="model"`, a skill registry described in scene words, no task names)
  over a **shared general skill library** that both sides co-own.
- **Repo:** we are moving into `github.com/msannikov03/robot-harness`; follow its `AGENTS.md` and PR protocol once
  the move lands. Until then, work here.
- **Bus:** `ssh anywherevla@100.64.0.2`, `~/data2/coord/inbox/{ilia,misha}/`. One file per message, with
  `from` / `type` / `re` front matter; move processed files to `done/`.
  - Re-read the inbox at least every 2 h and after any context compaction.
  - Only the lead writes to the bus.
- **Joint decisions** (D-C*) live in robot-harness `coord/DECISIONS.md`.
- **Standing guidance:** say yes to Misha's requests, share everything except secrets, and never use his key or give
  him ours.

## Machines (check the latest user word before putting load on any box)

| Box | Use | Rules |
|---|---|---|
| compute4 (`haptics`) | 2 camera lanes, `c4b_` | Shared with Mikhail's side: lane gate `~/coord/lane_gate.stamp` only, no other `~/coord` access. Their final tests get priority when asked. |
| compute6 (`robotics`) | 2 camera lanes, `c6b_` | Ours. Log power (lab power budget). |
| compute3 (`physicalai`, a colleague's box) | physics-only G3 and CUDA inference; Isaac RTX crashes on driver 595.99 | Check `who` / GPU first; never touch PID 3613 or anything that isn't ours. Driver rollback only if the colleague provably runs nothing and would not notice. |
| compute7 (`robotics`, a desktop user) | partial use: 1 GPU job, ≤ 8 cores / 16 GB, nice | Yield to the user; never touch the desktop session. |
| compute1 / compute5 / compute8 | not ours right now | |

- **Kill by PID only,** from pid files: never `pkill -f` or `killall`.
- **No apt / drivers / reboot** without explicit request.
- **Nothing autostarts on reboot.**
- **Write only under** `~/controlr*`, `~/dojo-*`, `~/dojo-data/<stream>`.

## Experiments (RoboDojo)

- **Layout ids.** Rendered or camera runs use ONLY `OUR_MEASUREMENT_SLICES` (100000-09, 100020-29), made fresh by a
  new salt. Mikhail's slices are 100010-19 and 100050-99; 100030-49 is the training block. Ids ≥ 100100 are sealed
  joint runs. Never read sealed data (`~/dojo-data/dc19`, `~/coord/scored`, `~/rh-misha`, `~/dojo-official-rh-*`).
- **Camera A/B protocol** ([EXPERIMENTS.md](docs/EXPERIMENTS.md)):
  - pre-register before any data;
  - paired fresh layouts;
  - the score-based gain rule (pooled +5, wins ≥ 2× losses, net ≥ 2);
  - guards, plus parity guards for scoped levers;
  - a per-writer write audit with the identity instrument (#417);
  - one re-test per rejected lever.

  Each A/B keeps the rule it registered with.
- **Route:** the candidate perceiver is `claude/claude-sonnet-5-5`. Verify the route of every camera row from its
  call logs.
- **G3** (perfect-perception physics) and offline replays on recorded frames come first; spend camera time only on
  the survivors.
- **Yardstick:** PyRUA-RSI's per-task programs (`research/robodojo-teardown/pyrua-rsi.md`,
  `docs/experiments/data/rsi-yardstick/`) are a measure of what is physically reachable. They are never a design to
  copy: their task routing is not board-legal. Write our own code only, with no attribution lines.
- **Landmine:** our generated imitate demos run 669-728 actions, while the official ones are likely ≤ 640. Absolute
  imitate numbers on our layouts may not transfer.

## Agents (the lead delegates; see memory for details)

- **Who does what.**
  - Claude Opus agents (T3 `claudeAgent`, through omniroute) do the creative per-mechanism work.
  - Codex agents (T3 `codex`, gpt-6.1-sol) do procedure: preregs, dispatch, verdicts, moves, infra.
  - Several Claude agents at once is fine (omniroute balances about 5 accounts). Stop them at the first HTTP 429.
- **Every brief states:**
  - the stopping condition: a result at the bar, or a written physical limit with numbers; not a time box;
  - several mechanisms to probe;
  - allowed concurrency;
  - the hard rule;
  - layout ids;
  - box rules;
  - "no per-event watchers that wake you".
- **The lead owns PR merges and CI watching**; sub-agents cannot watch PRs.

## Components

| Code | What |
|---|---|
| `dojo/runtime/` | episode runtime, planner / supervisor, belief state, `RuntimeConfig` |
| `dojo/skills/` | skill library, routes, G3 (perfect-perception physics) |
| `dojo/perception/` | B1 silhouette fitting, segmenter / SAM proposers, glyph readers, write audit |
| `dojo/simlab/` | dev layout generator, presettle, id slices (`paths.py`) |
| `dojo/agent/`, `dojo/bridge/` | agentic perceiver and manual, RoboDojo bridge |
| `scripts/dojo/queues/` | camera lane scripts (`lane2.sh`), queue tools |
| `controlr/` | the original harness: LLM client, protocol, prompts, robots (UR3 / SO-101), safety, loop |
| `configs/`, `research/`, `runs/` (not in git) | experiment YAML, literature and teardowns, outputs |

## Commands

```bash
uv sync --extra dev
uv run pytest -q                                      # unit tests, same as CI (live / isaac tests skip themselves)
uv run controlr run -c configs/mock.yaml --fake-llm   # original harness, dry run
scripts/bootstrap_gpu_box.sh HOST [--dry-run]         # provision a GPU box (docs/RUNBOOK-new-gpu-box.md)
uv run controlr report runs/<run_dir> [--csv out.csv]
```

## Rules (mandatory)

**Step 0 of any code task:** read [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) and [ARCHITECTURE.md](ARCHITECTURE.md).
For RoboDojo work, also read [docs/ROBODOJO.md](docs/ROBODOJO.md) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

1. **The hard rule above.** Every experiment axis is a config field. Defaults never change silently; changing a
   default needs a journal entry.
2. **Workflow:** branch / worktree → local tests → open the PR yourself → green CI → merge. If you touched the runtime,
   skills, safety or Isaac code, deploy and test on a box. Direct pushes to `main` are for docs only: journal,
   experiments, incidents.
3. **Live LLM calls cost money; the limits are generous.** Don't waste calls on runs that teach nothing.
   - Unit tests use only `FakeLLM` / mocks.
   - Keys: only `OMNIROUTE_*` (and `OPENROUTER_API_KEY`) from `.env`, **never** the agent session's credentials.
   - Report actual spend.
4. **Shared boxes:** see the Machines table above. PHANTOM (`~/phantom-icra-2027` on compute3,
   `~/skoltech/research` locally) is never modified.
5. **Secrets:** `.env` lives only locally, on the boxes' `~/controlr/.env` / lane env files, and in
   `~/data2/ilia/.env` on isr-lab (mode 600). Never in git, `runs/`, logs or chat. Never print key values. A new
   variable goes into `.env.example` and [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
6. **Cache invariants (original harness):**
   - the transcript is append-only;
   - images are encoded once;
   - feedback text is deterministic;
   - effort never changes within an episode.
7. **Safety lives in code.** Every real-robot action goes through `SafetyEnvelope` / PHANTOM's `SafetyMonitor`, with
   explicit per-session permission and a human at the e-stop or power switch.
8. **Reproducible or it doesn't count:**
   - the config is in git;
   - seeds × repeats;
   - n and spread are in the report;
   - the report goes in `docs/experiments/YYYY-MM-DD-<slug>.md` plus an index row.
9. **Research agents have no side effects.**
   - No model / media downloads unless the lead allows it.
   - Clones are shallow, in `/tmp`, and get deleted.
   - Keep disk use bounded.
10. **Loose ends go in the same PR:** BACKLOG, ARCHITECTURE / DEVELOPMENT updates, a journal entry for significant
    work, incidents. Stable docs carry no dates or statuses.
11. **Git hygiene:**
    - `git status` first;
    - `git add` file by file;
    - never commit `runs/`, `.env` or `research/repos/`.

## Original controlr (UR3 / SO-101)

The original rule still holds for that harness: the model emits low-level actions, with no high-level skills and no
VLA in the action path. UR3 in Isaac on compute3 (`scripts/deploy.sh`, `scripts/remote_run.sh`); real SO-101
([docs/SO101.md](docs/SO101.md)); the real UR3 only through PHANTOM's drivers with explicit permission.

## Where to look

| When | File |
|---|---|
| Any code task | [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md), [ARCHITECTURE.md](ARCHITECTURE.md) |
| RoboDojo stack, bridge, lanes | [docs/ROBODOJO.md](docs/ROBODOJO.md), [docs/BRIDGE-RUNBOOK.md](docs/BRIDGE-RUNBOOK.md), [dojo/README.md](dojo/README.md) |
| Experiment rules and reports | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), [docs/experiments/](docs/experiments/) |
| Something broke | [docs/RUNBOOK.md](docs/RUNBOOK.md), [docs/incidents/](docs/incidents/) |
| Open tasks and landmines | [docs/BACKLOG.md](docs/BACKLOG.md) |
| Past work | [docs/journal/](docs/journal/) |
| Other harnesses, RoboDojo teardowns | [research/README.md](research/README.md), [research/robodojo-teardown/](research/robodojo-teardown/) |
| Env variables and machines | [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) |
