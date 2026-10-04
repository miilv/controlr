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
- **Thinking control through the router differs per model.** Opus planner: effort only via the id
  suffix (`claude/claude-opus-5-5-xhigh`); `reasoning_effort` barely changes it. Sonnet 5.5:
  `extra_body: {reasoning_effort: low|medium|high}` works (no field = behaves like high); the
  `-low/-medium/-high` id suffixes are listed but rejected; `no-think/` still thinks; `thinking:
  {type: disabled}` is a 400 — the API wants `{type: between_tools}`, which moves the reasoning
  into the visible reply as prose; `output_config.effort` is dropped by the router
  ([journal](journal/2026-10-03-thinking-controls.md)).
- **GPT-6 Luna (`cx/gpt-6-luna`)**: chat-completions works although the catalog lists the route as
  Responses-only; `reasoning_effort: none` = no thinking, `low` still thinks on ~1/3 of turns
  (hidden, +4–8 s); `service_tier: priority` is accepted but unverifiable; the usage chunk can trail
  the reply by ~0.5 s (raise `llm.usage_grace_s`) ([journal](journal/2026-10-04-luna-latency.md)).
- **Direct Codex calls (chatgpt.com) from compute3 need the VPN or the tunnel**: Skoltech Wi-Fi
  blocks chatgpt.com. controlr's own Codex login (`scripts/codex_login.py`) has a rotating refresh
  token — keep the token file on ONE machine; a second copy that refreshes kills the first, and the
  router's login must never be reused (it would break the router for every client).
- **LLM latency measured on compute3 depends on its network.** compute3 is on Wi-Fi behind a VPN
  proxy whose upload varied 40 KB/s – >1 MB/s; each turn re-uploads the transcript with every image
  (~42 KB per image, ~730 KB by turn 16). Before comparing latency across runs, check the upload
  (RUNBOOK §3); use `CONTROLR_LLM_PROXY` + a tunnel when it is slow.
- **Opus 5.5 can't turn thinking off**; Sonnet 5.5 thinks and can exhaust `max_tokens` → an empty
  reply. `llm.max_tokens` counts thinking tokens too. For fast Sonnet 5.5 turns use
  `llm.extra_body.reasoning_effort: low` (≈2 s, 0 thinking), not the `no-think/` route.
- **On `claude/` / `cc/` the router sets the cache boundary itself**; our markers there seem to
  have no effect (not confirmed by a probe — see BACKLOG). Haiku 4.5 doesn't cache prefixes
  under 4096 tokens.
- **Isaac speed**: with PHANTOM's settings (1 ms steps, every step through `World.step`, USD
  write-back, a contact read per body) the sim ran at ~0.18× real time; see the
  contacts-and-speed report for what each speed setting buys and which defaults were chosen.
- **PhysX blows up** on a squeezed object; such episodes end as `unstable`, and the server now
  survives NaNs — but check the server log when things look odd.
- **The prompt and grammar must match the config**: examples in the manual are generated from
  `grammar_spec`; don't hand-write examples.
- **Feedback can leak simulator ground truth.** STOP texts name the object hit, box WARNs use the
  box's true pose, `task not complete yet` uses the sim's goal check — none of it exists on the
  real rig (journal/2026-10-03-feedback-realism.md, BACKLOG P1). New feedback must be derivable
  from real sensors (RTDE pose/joints, wrist force, Robotiq position/fault, the harness's own rules).
- **A single angled D435** gives poor depth: 80–200 mm depth errors are the main failure cause
  so far (hypothesis in BACKLOG).
- **`uv run` in the repo root creates `.venv`**; on compute3 `deploy.sh` builds `~/controlr/.venv`.
- **YAML reads a bare `off` as `false`** (`box_collision: off`, `--set safety.box_collision=off`);
  `config.validate` maps it back to `"off"`. Quote other on/off-like strings in YAML.
- **The manual must describe exactly the feedback the model gets.** `system_v0.md` has
  placeholders for every feedback-dependent sentence; with the legacy settings it must render
  byte-identical (`test_prompts::test_legacy_settings_reproduce_the_old_manual_byte_for_byte`,
  hashes of the old manuals). Changing a legacy sentence = a new `system_vN.md`, not an edit.
- **The blue box moves.** It is a dynamic body in Isaac: anything that uses box geometry must
  take the CURRENT pose (server `state()["bin"]`, `IsaacRobot.obstacles()`, `tasks.bin_info_at`),
  never `scene_info["bin"]` (the authored pose). Resting partly on the 3 mm mat it tips ~0.7°, so
  containment is checked in the box's own frame (`tasks.to_box_interior`) — a yaw-only test put a
  packet standing on the floor 0.6 mm "below" it and failed the scripted expert.
- **`pkill -f <pattern>` over ssh kills the ssh session's own shell** (its command line contains
  the pattern). Find PIDs with `pgrep -af`, then `kill <pid>`; a client killed without `close()`
  leaves its Isaac server running — kill that PID too.
- **compute3's CPU governor is `powersave`** and other users' jobs run there: Isaac timings vary
  with machine load — compare speed settings within one session, and note the load.

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
