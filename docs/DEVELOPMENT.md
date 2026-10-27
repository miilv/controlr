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

Live runs (money) — soft cap ~5k calls a day, following [EXPERIMENTS.md](EXPERIMENTS.md).

## 2. Tests: three tiers

| Tier | What | How to run | Where |
|---|---|---|---|
| unit | everything without network or Isaac: grammar, feedback, cache markers, transcript, kinematics, safety, loop on `FakeLLM`+`MockRobot`, client on `httpx.MockTransport` | `uv run pytest -q` | locally + CI |
| isaac | Isaac server, motion tracking, kinematics-vs-sim consistency, scripted expert | `CONTROLR_ISAAC=1 pytest -m isaac` (full run ~20 min) | compute3 only |
| live | real calls through the router | `CONTROLR_LIVE=1 pytest -m live` | manually |

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
| RoboDojo (shim, frames, two-arm grammar) | `robot/robodojo/`, `protocol/grammar.py` (`_parse_arm_line`), `prompts/system_v1.md` | `test_robodojo` (shim vs a fake `TASK_ENV`, a whole episode over a real connection); then `scripts/robodojo/run.sh --fake-llm` on compute2 ([ROBODOJO.md](ROBODOJO.md)) |
| dojo behind the official RoboDojo client (robot-harness's strict server) | `dojo/bridge/` (client, `contract.py` firewall, `RobotServerBackend`, driver), `scripts/dojo/bridge_episode.sh` | `test_bridge` (a fake strict server over a real socket: firewall, `joint_steps`, runtime / agent / twin episodes); `scripts/dojo/bridge_episode.sh --fake`; then a practice episode on a 5.1-rendering GPU ([BRIDGE-RUNBOOK.md](BRIDGE-RUNBOOK.md)) |
| decision head (questions, reduction, state) — tune with `scripts/decisions_probe.py` (one call per scene) and `scripts/decisions_perception.py` (image relations per camera) before episodes | `llm/decisions.py`, `prompts/decisions_vN.yaml` (+ `config.DecisionsConfig`) | `test_llm_decisions`; image format / that the model sees the frame: `CONTROLR_LIVE=1 pytest tests/test_llm_decisions_live.py -s` (4 calls) |
| cache / router / timings | `llm/` | `test_llm_*`; live `bench-cache` |
| an experiment axis | field in `config.py` + `configs/base.yaml` | `test_cli` / `test_loop`; the default must not change behaviour |

## 4. Pitfalls (already stepped on)

- **A test must not import from `tests.`** (`from tests.dojo.test_skills2 import ...`): CI collects with no repo root on
  `sys.path`, so collection dies with `No module named 'tests'` (fold, ttt: 2026-10-27). Give the test its own helpers or
  import from `dojo/`.
- **CI never runs a G3 / Isaac episode**, so a name that exists only on your measurement base passes CI and breaks main
  (#85 → #89: `belief_drop` in g3.py before #79 landed; every mode-A run died with NameError). Before a PR: run one G3
  episode from the PR head itself, and `uvx pyflakes` the files you touched (it flags undefined names).
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
- **chatgpt.com and api.anthropic.com answer 403 from Skoltech without a VPN**: compute3's machine-wide
  `controlr-vpn` (RUNBOOK §3a) handles it; if direct Codex calls get 403 there, the VPN is down. controlr's own Codex login (`scripts/codex_login.py`) has a rotating refresh
  token — keep the token file on ONE machine; a second copy that refreshes kills the first, and the
  router's login must never be reused (it would break the router for every client).
- **LLM latency measured on compute3 depends on its network.** Each turn re-uploads the transcript with
  every image (~42 KB per image, ~730 KB by turn 16). Runs before 2026-10-04 17:20 MSK went through the
  Happ VPN (40–140 KB/s upload); since then the router is reached directly (~1.5 MB/s, `controlr-vpn` routes it direct). Before
  comparing latency across runs, check `scripts/net_check.sh` (RUNBOOK §3a).
- **GPT-6 Luna Decisions is not a chat model.** It answers typed questions with probabilities on
  OpenRouter's `/api/alpha/decisions` (alpha API); chat SDKs and omniroute do not work with it
  (omniroute 404s every decisions path). It cannot write numbers — steps are levels we define
  (`decisions.levels`), so the step size resolution is a config axis. The schema does not document
  images in `state`: only `image_mode: parts` (content parts after a JSON text part) reaches the
  model; with `field` (data URLs inside the JSON) it does not see the frame and guesses
  (`tests/test_llm_decisions_live.py`).
  Question wording is a prompt: a new wording = a new `decisions_vN.yaml`.
- **The Decisions API has no prompt cache**: an identical request is billed in full at the same
  latency (probe 2026-10-07). History in `state` (`state_layout: transcript`) costs input tokens
  every turn (~200 per turn with one 448 px image); thin past images with `history_image_every`.
- **Decision-head probes must be balanced.** With the target on the same side in most samples a
  constant answer scores "24/24" (it happened). `scripts/decisions_probe.py` / `decisions_perception.py`
  place the start around the aim point with a random sign per axis; always read the accuracy on the
  minority class. The decision head locates only VISIBLE, concretely named things ("the red ball"):
  "the point 50 mm above the box rim" stays at chance (journal 2026-10-07-decision-head-tuning).
- **Sonnet 5.5 on `claude/` turns a "commands" prompt into call markup.** Asked for `{"commands": [...]}` JSON with tools
  described in the system prompt, it can write `<invoke name=...><parameter name="commands">[...]</parameter>` in the
  text and then repeat empty `<invoke>` / `<parameter>` blocks until `max_tokens` (40 s lost per turn). The agentic
  perceiver parses the first non-empty `<parameter>` as the reply and stops such a stream early
  (`dojo.perception.agentic.llm.degenerate`). `dva/gemini-*` answer 401 (no provider credentials); `auto/gemini` works.
- **Codex-served GPT-6 (`cx/gpt-6-astra-*`, `cx/gpt-6.1-sol`) answers "write a code cell" as a call of a `python`
  tool.** Without a declared tool omniroute drops the call: the stream has only `reasoning_content`, `finish_reason:
  stop`, ~100 completion tokens and empty text (2026-10-25: 24 of 30 agent replies lost). Declare a `python(code)`
  function tool (`dojo.agent.loop.PYTHON_TOOL`); `LLMClient` assembles streamed `tool_calls` into `LLMResult.tool_calls`.
  JSON-only prompts (Phase 1 grounding) are unaffected. Sonnet on `claude/` sometimes writes the same call as
  `<invoke name="python"><parameter name="code">` markup in the text; `dojo.agent.loop.extract_code` accepts both.
- **Sonnet 5.5: `reasoning_effort: low`, never `none`.** Through the router `none`/`minimal` fall back to
  the default and Sonnet thinks on most turns (17/22); `config.validate` rejects them on Claude routes.
  Very small frames (224 px) also make `low` think ([journal](journal/2026-10-04-sonnet-latency.md)).
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
- **Shell under `set -eo pipefail`** (all our step scripts): `cmd | grep -q x` fails when grep exits early (the writer
  gets SIGPIPE), and `find | xargs grep -l` returns 123 when some file does NOT match — both kill a `set -e` step on
  the good path. Count with `grep -c`, or turn pipefail off in that one substitution. `find -size -1k` matches only
  EMPTY files (sizes round up to whole KiB): use bytes (`-size -400c`). `{ cat a; cat b 2>/dev/null; } | sort` fails
  too when `b` does not exist yet (a first run): `cat b 2>/dev/null || true`. Test steps for real (the bootstrap's ssh
  shim runs them in sandbox HOMEs, `tests/test_bootstrap_fleet.py`; journal 2026-10-25-dojo-infra), not only as dry runs.
- **Never rewrite `~/.ssh/authorized_keys` through a temp file** (`grep -v x f > t; cat t > f`): if grep fails, `t` is
  empty and the box locks you out. Append, and remove with `sed -i '/marker$/d'` (it replaces the file atomically).
- **Two light lanes at once never come free by polling.** The g3 chains re-take a released light lock within seconds
  (blocked `flock -w` waiters and 15-s pollers). A job that needs two lanes (an Isaac 6.0 render of ours) takes them
  as they free up with a 0.5-s poller on fds the shell holds (`scripts/perception/held_render.sh`: a flock on an
  inherited fd stays held after the helper exits).
- **The physics-only twin writes nothing back to USD** (`updateToUsd = false`, omni.physx.fabric loaded), and RoboDojo
  reads some state from USD: articulation link poses (`usd_sync_links` answers them from the physics view) and a
  CPU-device garment's particles (`GarmentObject.sample_mesh_vertices`). For configs that spawn a Garment the runner
  launches the physics-only kit with `--/physics/updateToUsd=true --/physics/updateParticlesToUsd=true`
  (`runner.GARMENT_KIT_SETTINGS`); without them the cloth reads as its spawn mesh forever. `SIMLAB_PHYSICS_KIT_ARGS`
  adds any other Kit setting to a physics-only run (dev).
- **Mesh vertices are not surface samples.** A box-like catalogue mesh (the charger) has vertices only at its edges,
  so a vertex test of "what the pads touch" found nothing on its flat sides and fell back to a wrong width (the held
  object then hid under a finger mask). Sample the surface (`held.surface_points`).
- **Two Isaac processes that start the same SDF task at once can fail to cook its SDF colliders** (nuts, bolts, the tube rack: `omni.physx.cooking.plugin`, "UjitsoMeshCookingContext: cooking failure for .../collision/collisions"). The uncooked object then has no proper collision, and every episode measures the infrastructure: fasten_screws G3 mode B on compute6 was 1/15 in two jobs started the same second, against 1.0 for the same layout alone. g3 reads its own Kit log (`dojo.simlab.kitlog.KitLogWatch`, carb `/log/file`): such episodes are `category` `infra_failure` and stay out of the summary's n (`infra_failures`, `infra_seeds`), and the twin service flags its load. Stagger same-task SDF jobs by a minute, and re-run the infra seeds.
- **A catalogue colour of exactly 0.7 grey means "colour not read".** Some assets take their colour from an MDL preset
  (`info:mdl:sourceAsset:subIdentifier`, e.g. `ABS_Soft_Leather_Pop_Orange`), with no shader inputs on the USD
  side. `catalog.mdl_preset_color` reads the preset's flat colour argument (linear -> sRGB). Before it existed, the
  four play_stacking_toy pieces (purple, yellow, blue, orange-red in the render) were all grey, and B1 told them
  apart by silhouette alone. A preset whose colour comes from a texture (copper, marble, wood) still reads grey.
  Rebuild an asset after a builder change (`catalog build --force --keys-json`), then sync the boxes
  (`catalog_check`).
- **compute3's CPU governor is `powersave`** and other users' jobs run there: Isaac timings vary
  with machine load — compare speed settings within one session, and note the load.
- **G3 episodes are bit-deterministic, with two exceptions.** The same layout replays the same actions on a loaded
  box (stacking 100001: 927 identical steps over 6 runs and 6 commits). Outcomes still differ when:
  - a search hits a wall-clock cap. `library.RELAY_PLAN_S` (relay chains), `HANDOVER_PLAN_S`, `HANG_SEARCH_S` and
    hang's per-round budget: a loaded box explores less before the cap;
  - meshes are missing. `AssetDB` falls back to bboxes when no catalogue has an asset's `geom.npz`. On a box whose
    catalogue sync was still running, the pieces' servo tail hovered 4 mm instead of 22 mm and three layouts changed.
    Now every miss is printed (`[AssetDB] WARNING no mesh for ...`) and listed in the episode records
    (`missing_meshes`, G3 and `dojo.runtime.eval`). G3 refuses to run when the full catalogue has the mesh
    (`dojo/skills/data/mesh_less.json` lists those it never has; `scripts/skills/mesh_less.py` regenerates it);
    `--allow-bbox` plans on the bboxes anyway.
  Before calling a difference a regression, diff the saved actions (`<task>_<seed>_actions.json.gz`) of the two runs:
  the first differing step points at the skill that diverged.
- **`scripts/deploy.sh` installs only `.[dev]`.** A rendered `integ_eval.sh` starts the perception service from the
  deploy dir's `.venv`, and that needs `.[perception]` (opencv, trimesh, ...). Without it the episode ends at step 0,
  `cat=error ... No module named 'cv2'`. Before a rendered run: `uv pip install --python .venv/bin/python -e
  ".[dev,perception]"` in the deploy dir. G3 runs in RoboDojo's python and does not need it.

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
