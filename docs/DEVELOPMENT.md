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

### Task-scoped key switches

`RuntimeConfig.key_scope_task` accepts `""` (legacy global behavior) or
`"insert_key"`. With the latter, `key_handover` / `key_lean` are copied into
Skills only for the exact instruction-routed key vocabulary; all other routes
resolve `v1/off`, including Skills rebuilds. Scope alone enables neither switch.
Use runtime config for camera scoping; G3's direct `--skill-attr` overrides
remain global. Defaults and presets retain legacy behavior.

### Testing belief-object routing

G3 uses --skill-attr plan_world=objects and optionally
--skill-attr plan_solver=curobo. The corresponding dojo RuntimeConfig fields
have defaults table and routes. Table retains the legacy skill behavior.
plan_profile=legacy (default) retains the two-second GPU/caching settings;
recover opts into coverage recovery with a ten-second request budget, diverse
IK retry banks inside current native limits, geometry proxies and bounded,
monotone start-contact proposals. Retracts/repairs execute only after the complete
path passes the original validator. plan_capture=requests (default off)
adds numeric belief payloads, native goal checks, actual legacy fallback results
and proposed joints for diagnosis.
Keep captured request replays separate from fully validated path coverage, and
report actual queries, inherited cached failures and native validation rejects.
The installed v2 has raw stage success/feasibility flags rather than MotionGenStatus
codes; do not relabel discarded historical no_plan records as known causes.

tests/dojo/test_world_plan.py covers CPU dispatch, geometry, attachments and
the port's belief-only/native-TCP conversion. tests/dojo/test_curobo_world.py
mocks GPU proposals and checks box/sphere transforms, limits, collisions,
deadline fallback and owned-worker cleanup; it needs no GPU dependencies.

The optional GPU worker requires the deployed cuRobo v2 and its public X5
configuration in the current Python environment. Its process isolation keeps a
fatal CUDA failure from poisoning Isaac. The installed MotionPlanner attachment
accessor refers to a missing solver property, so the adapter binds managers to
the available trajectory/IK/graph models. Mesh rendering links are excluded
from IK tool targets; one trajectory seed avoids invalid graph-seed batches.

Validate on paired settled dev layouts. Report success/score, spread, steps,
overall plan_s, added-search/startup time and fallback reasons. Request counts
include speculative skill candidates. Do not treat shared-machine timing as
a controlled latency benchmark. The GPU request deadline and CPU cooperative
deadline have different startup/overrun semantics; report them explicitly.

## 4. Pitfalls (already stepped on)

- **Camera lane profiles:** `scripts/dojo/queues/lane2.sh` retains compute4 defaults.
  For compute6 set `LANE_STREAM=integ-c6`, `LANE_STATE`, `LANE_RUNS`, `LANE_DATA`
  and the pinned `LANE_REPO`. The compute6 profile skips the compute4 boot stamp
  and accepts only `~/dojo-gpu.lock` / `~/dojo-gpu-heavy2.lock`; light locks remain
  independent. Both profiles keep the 16 GiB-used admission gate and 50 GB disk
  floor. `shared_side.sh` and `ping_side.py` manage shared candidate services from
  the same deployment. `power_watch.py STATE` samples every five minutes and
  drains heavy2 after two consecutive readings above 700 W, without killing
  another stream's work. A drained lane finishes its current job before exiting.

- **A full disk takes the whole box down.** compute3's `/` hit 100 % (2026-10-29: other tools' data plus ours). `integ_eval.sh`
  and `rdpy.sh` GPU jobs now refuse to start below `DOJO_MIN_FREE_GB` (default 10) free on the `$HOME` / `$DOJO_DATA`
  filesystem (`scripts/dojo/disk_guard.sh`, exit 6). Boxes with a small data disk set their own floor (compute1: 2).
  Clean your old deploy copies / run dirs by explicit path, never by pattern.
- **A test must not import from `tests.`** (`from tests.dojo.test_skills2 import ...`): CI collects with no repo root on
  `sys.path`, so collection dies with `No module named 'tests'` (fold, ttt: 2026-10-27). Give the test its own helpers or
  import from `dojo/`.
- **CI never runs a G3 / Isaac episode**, so a name that exists only on your measurement base passes CI and breaks main
  (#85 → #89: `belief_drop` in g3.py before #79 landed; every mode-A run died with NameError). Before a PR: run one G3
  episode from the PR head itself, and `uvx pyflakes` the files you touched (it flags undefined names).
- **The sim stops joint 5 at +-pi/2, not at the asset's +-10 rad.** The stop holds on both sides and at any wrist roll;
  it is likely self-contact, since the URDF / ARX.usd declare +-10 rad and nothing clamps the commands.
  `kinematics.JOINT_LIMITS` ("wide") lets the IK go to 1.9 rad, so a plan past pi/2 leaves the arm stuck there, with
  the TCP 20-100 mm from where the planner believes it is: fill_pen drops let go beside the holder. Check commanded
  against measured joints (`q` / `qm` in the G3 trace frames) before blaming the skill. `RuntimeConfig.joint_limits`
  "measured" keeps joint 5 within +-1.56.
- **`timeout N scripts/skills/g3.sh ...` does not bound a G3 job.** The TERM reaches g3.sh, whose trap forwards it to
  the Isaac python and waits, but the python keeps running its episodes: a `timeout 1900` job ran 40 min, on into the
  next layout, until its python was killed by PID. Use `timeout -k 60 N` (a KILL to the process group 60 s later), or
  give each job its own process group / systemd scope and stop that.
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
- **omniroute's Claude account (`claude/*`, `cc/*` = the same account) has its own weekly limit.** On 2026-10-27 a
  multi-arm perceiver sweep (6-13 parallel Sonnet / Opus / Fable sessions, ~1,900 calls, 29.6M prompt tokens of which
  25M cache reads, in under an hour) plus the rendered runs hit it: 429 "reset after 61h" on every route for everyone
  ([incident](incidents/2026-10-27-claude-route-rate-limit.md)). "No LLM cap" meant the OpenAI subscription. Before a sweep
  on `claude/`, estimate its tokens against the weekly budget; a 429 with a multi-day `retry_after` should stop the runner.
- **Codex-served GPT-6 (`cx/gpt-6-astra-*`, `cx/gpt-6.1-sol`) answers "write a code cell" as a call of a `python`
  tool.** Without a declared tool omniroute drops the call: the stream has only `reasoning_content`, `finish_reason:
  stop`, ~100 completion tokens and empty text (2026-10-25: 24 of 30 agent replies lost). Declare a `python(code)`
  function tool (`dojo.agent.loop.PYTHON_TOOL`); `LLMClient` assembles streamed `tool_calls` into `LLMResult.tool_calls`.
  JSON-only prompts (Phase 1 grounding) are unaffected. Sonnet on `claude/` sometimes writes the same call as
  `<invoke name="python"><parameter name="code">` markup in the text; `dojo.agent.loop.extract_code` accepts both.
- **No checker knowledge in any prompt (dojo agent).** The manual, the appendix and the skill docs carry no checker
  thresholds, tolerances, score ladders, reward predicates or per-instruction-pattern recipes (robot-harness
  RULES-robodojo §14.5). A number that describes a robot action (a skill's motion, an API default) is allowed in skill
  code, not in prompt text. `tests/dojo/test_agent_manual_v6.py` guards it; manual_v2-v4 broke it
  ([journal](journal/2026-10-28-dojo-agent-manual-v6.md)).
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
- **RoboDojo's table generator turns an object about its ORIGIN, not about the point xlim / ylim constrain.**
  `ClutteredGenerator.add_model` puts the place frame's centre at the sampled point, then turns only the orientation
  by the random yaw (the position stays). The constrained point is origin + R0 @ centre whatever the yaw; an asset whose
  origin sits at one end (the phone: 58 mm from its centre) lies anywhere on a circle around it. B1's region checks
  use this for table / ground placements (`inverse_layout._gen_offset`); placements on another object
  (`gen_layouts.place_on_object`) turn about the placement point. `scripts/perception/region_check.py TASK` lists the
  settled dev objects outside region + margin under both rules.
- **RoboDojo poses an Articulation on its articulation-root prim, not on the asset's origin.** Its `ArticulationObject`
  is an Isaac `SingleArticulation`, whose view resolves the prim path to the prim with `PhysicsArticulationRootAPI`;
  the record's `default_pos/ori` lands there and `objects_gt` reads it back. On laptop/00000-03 that is the base link,
  77 mm off the default prim, where our catalogue meshes, bboxes and `links.json` live: the mesh sits at the record
  pose + (0, +78, -11) mm (object frame), so perception's (correct) laptop read scored "80 mm off" and G3 planned on a
  laptop 8 cm away. `dojo/skills/artroot.py` converts (`to_asset` / `to_root`, table `data/articulation_roots.json`
  from the USDs by `scripts/skills/articulation_roots.py`): our side is the asset frame, RoboDojo's side (records the
  sim loads, `objects_gt` rows) the root frame. New code that pairs a GT row or a record with a mesh goes through
  `artroot.row_to_asset`; a pose handed to RoboDojo through `to_root`. Re-run the script after an asset update.
- **A catalogue colour of exactly 0.7 grey means "colour not read".** Some assets take their colour from an MDL preset
  (`info:mdl:sourceAsset:subIdentifier`, e.g. `ABS_Soft_Leather_Pop_Orange`), with no shader inputs on the USD
  side. `catalog.mdl_preset_color` reads the preset's flat colour argument (linear -> sRGB). Before it existed, the
  four play_stacking_toy pieces (purple, yellow, blue, orange-red in the render) were all grey, and B1 told them
  apart by silhouette alone. A preset whose colour comes from a texture (copper, marble, wood) still reads grey.
  Rebuild an asset after a builder change (`catalog build --force --keys-json`), then sync the boxes
  (`catalog_check`).
- **compute3's CPU governor is `powersave`** and other users' jobs run there: Isaac timings vary
  with machine load — compare speed settings within one session, and note the load.
- **Physics is deterministic per box CLASS, not across classes.** The same commit and settled layout replays to 0.0 mm
  on compute3 vs compute4 (5090, i9-14900KF), but compute2 (4090, i9-13900KF) drifts: 2 mm at step 161, 77 mm by the
  end of an episode (stream C, 2026-10-28). So: A/B arms run on the same box class (best the same box); a replay or a
  `seq_capture` runs on the recording box's class; a number from another class is a different measurement. What it does
  to the twin verifier's decisions (independent place / release outcomes, not end-of-episode drift):
  docs/experiments/2026-10-28-twin-crossbox.md (`dojo.simlab.twin_crossbox`): flat places agree (0 / 229), drops
  into containers flip (30 / 76).
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
- **A box's `bg_folds*` must carry the `scene/` backgrounds, or every belt read is blind.** The bootstrap copies
  compute3's folds once, so backgrounds built after a box's copy are not there. compute4's `bg_folds_51` had no
  `scene/` (2026-10-28). Belt and frame-0 reads then segment the belt against the table background, find nothing,
  and match_and_pick ends with "nothing moves". The perception service now prints `WARNING: no scene background for
  [...] ... belt reads are blind`, and the episode report has a `belt_blind` event and `track_read` `blind: true`. Before
  a conveyor run: `ls $DOJO_BG_FOLDS/fold*/scene/` (Dynamic_conveyor_00000 for match_and_pick; pick_from_conveyor_by_image
  needs Dynamic_conveyor_00001).
- **A robot-harness skill never holds the raw server client.** `RobotServerBackend.robot` is a `ForeignRobot`. It
  refuses `success` / `end` / `reset` / `close` before sending, so a `--dev` server's grader stays out of reach, and
  it checks every reply against `contract.BANNED`. Only the driver ends an episode. A new verb that skills need goes
  through the same proxy (`tests/dojo/test_g5_review.py`).
- **Perception ops have a reply deadline.** `PerceptionClient(op_timeout_s=600)`: a reply later than that
  (the slowest op in 306 rendered dev episodes took 243 s) closes the connection and raises `TimeoutError`. Later ops in that episode raise
  at once, and `RuntimePolicy.reset` reconnects. A stalled service no longer holds an episode past its wall cap
  indefinitely.
- **No geometry is invented for an asset the catalogue lacks.** With an `AssetDB`, missing or bbox-less metadata
  gives `AssetInfo.unknown_geometry`: its 4 cm box only keeps the object an obstacle. Its estimate has confidence 0,
  and pick / transfer / topple / push refuse it (`status "unknown_geometry"`). Of the 749 RoboDojo assets, only
  `Clutter/vehicle/00000` has no aligned bbox.
- **The perception calibration is one frozen set, never chosen per episode.** The runtime's perception service reads
  `$DOJO_BG_FOLDS/$DOJO_CALIB_SET` (backgrounds, `color_M.json`, `manifest.json`) for every request; the policy gets the
  instruction and the step limit only. The old per-episode `fold = crc32(layout id) % 2` made the layout id a policy
  input (legality audit, journal 2026-10-28-legality-calib-fold). Sets are built only from the calibration-only ids
  100040-49 (D-C10b; never evaluated: the drivers refuse them) with `python -m dojo.perception.calib_set build` (a set
  is never rebuilt in place), pinned in `scripts/infra/pins/calib_sets.txt` and installed by `box_steps.sh calib_sets`;
  a box without the set refuses perception runs.
- **Offline perception on a box (B1, belt_read, embed renders) takes the GPU.** It renders through pyrender EGL: a
  ~300 MiB graphics context per process, even with CUDA hidden. compute3 has no OSMesa, so take a light slot (flock)
  for it like any GPU job.
- **Learned identity (`EstimatorConfig.embed_model`) needs torch in the perception venv.** `.[perception]` has no
  torch: with the flag set and no torch the first belt read raises `ModuleNotFoundError: torch`. On a box (RTX 5090,
  driver >= 570): `uv pip install --python .venv/bin/python "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
  --extra-index-url https://pypi.org/simple && uv pip install --python .venv/bin/python "transformers>=4.56"` (6.8 GB of
  venv; `uv pip uninstall triton` saves 0.6 GB, eager inference does not use it), plus the weights in
  `$DOJO_EMBED_MODELS/dinov2-small` (85 MB: `hf download facebook/dinov2-small config.json preprocessor_config.json
  model.safetensors --local-dir ...`). Put the uv cache (`UV_CACHE_DIR`) on the venv's filesystem: it hardlinks, so the
  cache costs nothing extra. Crop style matters more than the model: on the dark belt the segmentation mask drops a
  black camera's body, so a masked crop scores 0.41-0.53 where the raw crop against renders over the background scores
  0.76-0.80 (`embed_crop`, journal 2026-10-27-dojo-pbake-identity).

- **A config label can stand for a numbered series.** `select_mode.label_prefix` with `select_instance_nums` makes the
  generator expand one label into `<prefix>_<k>` objects (sweep_blocks' `rigid2` -> `cube_0..cube_4`, classify_objects'
  `rigid0` -> `cat0_*`, `cat1_*`, `cat2_*`). B1 searched for one object per label (2 of 3 cubes never looked for), and
  `eval_g1` dropped the series' GT objects (their labels are not config labels), so the misses never showed in G1.
  `EstimatorConfig.series_count` and `eval_g1.series_labels` / `gt_label` handle them; `scripts/perception/sam/series_tasks.py`
  lists the tasks (journal 2026-10-27-dojo-sam).
- **SAM flags at runtime need their server.** `sam_fg` / `sam_propose` / `series_count=sam` are estimator-config fields
  (`DOJO_PERCEPTION_CFG='{"sam_propose": "sam2.1_small"}'` for the perception service), but they only act when a SAM
  server answers at `DOJO_SAM_ADDR`: start `python -m dojo.perception.sam_masks serve --sock ... --models sam2.1_small`
  (1-6 GB VRAM by models; `scripts/perception/sam/setup_venv.sh`) next to the perception service and export
  `DOJO_SAM_ADDR` into the service's environment. Without one the flags are a no-op (plain B1, `series_count=sam`
  falls back to `min`).
- **A grasp is the LAST closing gripper command of a run, not the first one below half open.** Some skills pre-shape the
  jaws on the approach (hang_mugs' wall grasp: 1.0 -> 0.43 at the start of the approach, 0.0 at the real close ~30
  steps later). A "first command below 0.5" rule takes the pre-shape as the close: `dojo.runtime.eval.GTProbe`'s
  `dev_gt_probe` closes still do (P3r3 hang_mugs 100003: "(42, right, mug2, 128.5 mm)" is the pre-shape), and the
  first version of the grasp-time metric scored no hang_mugs event at all. `grasp_metric.grip_events` takes the last
  closing command before the jaws open again.
- **The grasp-time metric rebuilds the runtime's belief from the report, and only from what the report logs**
  (`grasp_metric.BeliefTracker`): the planning record, every accepted refine (pick / place / held), the jaws (an
  attached object rides the commanded TCP), `belief_drop`, `post_place`, held_look drops. A divergence's re-estimate and
  the twin's nominal outcome are not in the report: events after a `resync` divergence carry `after_divergence`.
  The conveyors' riders are tracked between reads and their beliefs are not logged per step (`moving`): do not read
  belt tasks' numbers as perception numbers until the runtime logs its belief per event.
- **Physics-only replays are CPU-bound** (RoboDojo's PhysX runs on the CPU): on a loaded compute3 (load 30) one episode
  replays at ~2-5 steps/s. `scripts/perception/m95_replay.py` loads many episodes of a task in one runner and stops
  after the episode's last gripper change (`--full` for the whole list).
- **RoboDojo's gripper observation is the APPLIED control, and that makes it a jaw sensor.** `{arm}_ee_joint_state` is
  the measured finger value at most 0.2 of the range past the target (`process_gripper_val`). Closed on nothing it reads
  the command itself (0.0 in 87 of 98 empty closes in stream M's replays); closed on an object w wide about
  `(w/2 + 0.01) / 0.054 - 0.2` (`belief.expected_official`; a 40 mm block 0.33, a 3-5 mm mug wall 0.003-0.05). A
  1.9 mm coin reads 0.0 held or not, and a mug taken by its hook reads 0.0 a step before the jaws open. The strict
  server v1 / `DOJO_GRIP_SOURCE=command` hide it (the command in its place); strict v2 / `DOJO_GRIP_SOURCE=v2` add it
  as `<arm>_ee_joint_state_official` (`RuntimeConfig.grip_check` reads it). Never put it in place of the command
  (`DOJO_GRIP_SOURCE=robodojo`): held_look and the robot mask read that key as the command (the 14/24 vs 17/25 A/B,
  experiments 2026-10-27-grip-source-ab). A deployment path that hands the policy RoboDojo's observation as it is has
  no `_official` key, and grip_check is then a silent no-op. `RuntimeConfig.grip_official_key="ee_joint_state"` makes it
  read `<arm>_ee_joint_state` against the runtime's own last command (`KinematicBelief.g`). Set the key per path; it
  cannot be detected, because under the command semantics the two values are equal, which is exactly an empty grip.
- **A garbage-collected pyrender renderer kills every GL context of the process.** `OffscreenRenderer.delete()` (run by
  `__del__`) calls `eglTerminate` on the EGL display, which all pyrender contexts of a process share; the next render
  of ANY other renderer (GL silhouettes, `ColorMeshRenderer`, a `Gallery`) fails with `GLError 12289` in
  `eglMakeCurrent`. Keep renderers process-wide (module globals like `render.gl()`, `wrist_check._GALLERY`), never per
  object that may be dropped (E1 of the wrist check: a per-episode `Gallery` broke every later episode).
- **CPU silhouettes (`DOJO_GL=0`) are only cheap for box-like meshes.** A wrist refine renders ~550 silhouettes; with a
  25k-face silhouette mesh (mugs, bowls, most organize / pack items) that is 13 s on CPU against 0.4-0.8 s with GL
  (EGL). build_tower's blocks (12 faces) are fine on CPU. Anything else offline needs a GL context, i.e. a light GPU job
  in a slot.
- **Per-step frames of a recorded episode: replay its actions, don't re-run the policy.** `dojo.perception.seq_capture`
  (mode gt, `--policy dojo.perception.seq_capture:SeqReplayPolicy`, `scripts/perception/seq_render.sh`) reproduces the
  recorded run exactly (same `dev_gt_probe`) and keeps every camera + state per step (`steps.jsonl`, no GT) and the dev
  GT apart (`gt.jsonl`, for scoring only). Frames are PNG (lossless) by default: JPEG q95 4:4:4 changes neither B1
  nor refine measurably (journal 2026-10-28-dojo-bgperc §1), but it moves the v2b segmenter's borderline detections
  (§7), so `DOJO_SEQ_FMT=jpg` only for frames no detector reads. On top of the stride it always observes the decision frames (the recorded run's logged
  steps, k-1 of every grip event, release+1..5: `meta.json` "forced_steps"). Gate every replay with
  `scripts/perception/bgperc/replay_check.py` and replay only on the box that recorded it (BACKLOG: compute2's physics).
- **The segmenter flag needs its server too.** `seg_propose` acts only when `python -m dojo.perception.seg_twin serve
  --sock S --weights W --classes classes.json` answers at `DOJO_SEG_ADDR` (torch + ultralytics in that venv); without
  one it is a no-op. Its classes are catalogue keys from the training export (`classes.json`): an asset that never
  appeared in the training renders cannot be named, only its category (`_seg_groups` falls back to the category).
- **Additive segmentation uses independent checkpoints.** `EstimatorConfig.seg_additive` defaults to false.
  Supply `serve --extra-weights W --extra-classes C` (or those leading options to `with_seg_server.sh`) to load a
  disjoint new-only vocabulary. Enabled requests append extra masks only below `seg_additive_iou` overlap with
  every base mask and above `seg_additive_min_score`. Base mask/key/score/box entries remain an unchanged prefix;
  combined detections have no shared cap or NMS. Consumer deduplication gives base entries priority. Extras require
  exact public task candidates and never category fallback; wrist pools and pad extension remain base-only.
  A requested additive model that was not loaded fails loudly. Raw detector preservation does not establish
  downstream pose/count invariance: gate every candidate on paired dev frames and report phantoms separately.
  `new_only_export.py` deliberately retains empty-label old-only scenes as negatives for the separate new model.
- **The glyph flag needs its server too.** `glyph_read` acts only when `python -m dojo.perception.glyph_read serve --sock S
  --model kong_cls/model.pt` answers at `DOJO_GLYPH_ADDR` (torch + torchvision); without one, or with a back / grazing
  / low-margin crop, the tile stays unread ("unknown"). The margins (per view kind) live in the checkpoint and were
  chosen on 100030-49 validation draws; never pick them on the held-out ids. One server serves one model (kong tiles
  or digits / operators); a second reader takes its own socket (`GlyphClient(addr)` as `client_`).
- **A straight-down wrist look sees the table upside down.** `regrasp._look_move` tries wrist yaw 0 first, and that
  pose rolls the wrist camera 180 deg against the head camera and the home wrist (world +y points down in the image).
  The digit reader was trained on head / home-roll crops with a small rotation augmentation. Seen that way, a 6 reads as
  a 9 at a confident margin, and 2 / 3 / 4 / 5 / 7 fall below the margin. `EstimatorConfig.glyph_upright` "on" turns
  the runtime wrist crop of a flat glyph so world +y is up first (`glyph_read.upright_deg` / `turn_upright`). Any new
  wrist read of an oriented glyph needs the same thought: which roll was the reader trained on?
- **A knocked-over object is not where it stood.** The watcher (`RuntimeConfig.watch`) lays a knocked object face up
  at its slot, but make_kong's discard lies 48 mm toward the robot in every dev render. A read or plan at the laid pose
  looks at the empty slot (head "back", wrist "edge"). `RuntimeConfig.watch_relocate` locates it in the head view
  (`task_state.laid_pose`: new tile-face pixels since frame 0 near the slot) until two reads agree.
- **A set side-server address fails loudly.** With `DOJO_SEG_ADDR` / `DOJO_GLYPH_ADDR*` unset the flags are no-ops;
  with one SET but unreachable, gone mid-run or answering an error, the clients raise `sam_masks.SideServerDown`
  (never a silent run without the server). Callers re-raise it past their "a failure never takes perception down"
  handlers; the perception service writes `side_server_down` next to `DOJO_PERCEPTION_READY` and exits; eval marks the
  row `infra_failure` and `integ_eval.sh` exits 4 (the runner re-queues once). One `sam_masks.serve` server may serve
  several lanes: the model runs one request at a time (`model_lock`), the image cache sits under its own lock.
- **RoboDojo scores a goal the moment it holds.** An episode ends in success at the step its checker passes, e.g.
  make_kong's stand at the release of the drawn tile. A read or confirm planned AFTER the final placement never runs,
  and the tile was used unread. Read before the final place, on a spot the checker does not score
  (`library.kong_draw_read`'s read spot).
- **A native success latch does not prove a stable released placement.** For a conveyor
  container, archive the first latch, the release segment and an independently measured
  post-chain hold. A development-only continuation may suppress success termination while
  retaining failure/step-limit stops; label it unscored and keep the official G3 panel on
  the unmodified native environment. A later read of latched stage count cannot establish
  current containment or lift. Check the unlatched geometry and report both outcomes.
- **`classify_mask` needs the arm mask.** The digit / operator margins were calibrated without the crops it refuses:
  masks cut by the image border (`edge`) and masks whose 3-px ring is >= 30 % arm (`robot`, only when `robot=` is
  passed: `~known` of a B1 view or `render.robot_mask`). Without `robot=`, a tile half hidden by the gripper is read
  at a high margin and often wrong (a cut 3 is a 9).
- **CPU OpenGL without OSMesa.** The boxes have no OSMesa, but Mesa's EGL vendor exposes llvmpipe as EGL device 1:
  `LIBGL_ALWAYS_SOFTWARE=1 EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json` gives pyrender with no GPU process
  (GT-mask renders: ~0.2 s for a frame's 3 cameras). For B1 itself `DOJO_GL=0` (its CPU silhouettes) is ~5x faster
  than llvmpipe; the two B1 backends differ on ~5 % of objects (D0 frame 0: 891 vs 903 within tolerance), so compare
  arms on one.
- **RTX leaves ghost trails of a moved arm.** A frame captured with one render pass after a 40-step move (D0's frames
  1-3, `make_d0`) shows the arm smeared along its path (temporal accumulation). `seg_render` captures 40 % of its
  arm-pose frames that way and the rest after 8 extra `render_for_capture` passes.
- **A segmenter class increment costs its look-alikes.** Fine-tuning v2b with new asset classes
  (`export_yolo --base-classes`, `expand_head.py`) moves detections from an old class in another category to the
  new one: car 10-15 took classify's toy_cars. Leaving an asset unlabelled teaches it as background. `--new-keep`
  keeps only the listed new assets and leaves out every image that shows another one. Gate any increment on the
  full frame-0 bake, every task, 0 losers.
- **Provider responses and child processes are credential sinks.** A gateway can echo our `Authorization` header in an
  error body, and a child inherits the whole environment (the router key, the path of the file that holds it). Any new
  text from a response that reaches a log goes through `controlr.keyguard` (`clip`: redact, THEN cut, else a key that
  straddles the cut leaves its head; `safe_label` for server-chosen ids), and any new `subprocess` / `Popen` in dojo
  that does not call the model takes `env=child_env({...})`. A key is read from one named file or the environment,
  never a mix and never a home-dir `.env` nobody named ([ENVIRONMENT.md](ENVIRONMENT.md), credentials in processes).
  Tests: fake keys only, and assert on names / fragments, never print an environment (`tests/test_sec_redaction.py`,
  `tests/dojo/test_sec_env.py`).
- **`inverse_layout.coarse_fit` assumes the table plane.** A mask whose ray meets the table plane far away (a wrist
  camera looking along the table) gives it a huge ROI (minutes per fit, A1's first offline run). Callers that fit
  arbitrary masks guard the view first (`redetect.fit_in_mask`: table-plane point within 1.5 m, projected extent
  < 2 x the image).
- **Identical instances fool a nearest-GT error.** A belief write that puts a ttt piece (or a block) onto ANOTHER identical
  piece "improves" its error to the nearest GT of its group while making two events wrong. Seeds from unexplained masks
  need the object's exact asset class, and associated objects refit only locally (`redetect`).
- **`pyrender` pins PyOpenGL 3.1.0**, whose array handlers fail on numpy 2 in textured renders (every press_by_number
  frame: `ArgumentError ... No array-type handler` / `glGenTextures`). `pyproject.toml` overrides it to 3.1.9
  (`[tool.uv] override-dependencies`), so `uv sync` / `uv lock` venvs get the working version; a venv built with plain
  `uv pip install -e .` before that override still has 3.1.0 (check `python -c "import OpenGL; print(OpenGL.__version__)"`).
- **A held_look `held` has three values.** `None` (with `error`) is "no answer": no evidence (a uniform wrist frame, a
  silhouette fit that singles out no opening is not used, the fingertips out of view), the agent gave no yes / no, B1
  failed, or the asked arm had no verdict. Never read it as "not held" (`if not r["held"]` does), and with `arm` read
  only that arm's entry: the other arm's answer is not this arm's (robot-harness G2-3 / G2-10, P1S I1-3). Likewise an
  orientation or a confidence the perceiver did not measure is never filled in with a default (`merge_prior`'s
  `prior_unusable`, the bridge's `_orientation_unknown`).
- **A number card's asset index is not its digit unless someone read the glyph.** B1 picks a card among identical
  boards by textured render-and-compare. With no texture score (no GL, a hidden or flat glyph) every candidate ties, and
  the first one wins as a placeholder. Take a digit only from the digit fields (`dojo.perception.digits`: record
  `_perception.digit` / `digit_source`, `ObjectState.digit`, J1 `value` / `attributes.digit`), never from `category_idx`.
  `digit_source == "unread"` means unknown, and the planner then reports the press count unknown. Where the agentic
  perceiver's `value` and B1's read disagree, `digits.resolve` decides; don't add a second rule. A card's index follows
  its digit (`idx_for_digit`, `asset_for_digit`) so the twin and the planner agree (legality audit P2-6, G2-2, G2-4;
  journal 2026-10-28-legality-card-digit).
- **solve_equation / arrange_largest_number's tiles are slippery and the pads sit at the edge of the reach.** The
  `number` tiles' metadata friction is 0.0, and the operator tiles (plus / minus / multiplication / division / equal)
  are 60 mm discs 8 mm thick. Off-centre grasps, the category's "top" grasps and in-air handovers let them slip.
  solve_equation's pads lie at y 0: neither arm reaches (0, 0) pointing straight down, and the default 6 cm pre-place
  over mat1 / mat2 has no IK. `dojo.skills.tiles` takes centred box grasps, a 3 cm pre-place, a relay patch between the
  arms, and an in-air handover only as the last attempt. main's generator had also lost solve_equation's task
  semantics (all four tiles and no missing slot, on which RoboDojo's `run_reward` crashes); generator v15 has them
  again. Its other tasks' draws are byte-identical to v14 (journal 2026-10-28-dojo-newskill-tiles-align).

### Imitation twin look

`RuntimeConfig.imitate_tie_look` (default false; G3 `--skill-attr imitate_tie_look=on`) extends the
one-to-one imitation pairing look to equal-cost two-pair swaps, including same-asset matches. It enables the existing
mismatched-asset look pass too, after the demonstration wait, with at most three looks under its copy-budget guard.
It uses observed identities only. Run it with `imitate_pairing=one_to_one` and `imitate_wait=home`; it has no effect
on greedy pairing. Reassignment follows the looks. Their explicit `imitate` refine kind enables the existing identity classifier
(`refine_identity` must be on) over the label's normal candidate group; ordinary `look` reads only refit its current
asset. This kind is gated by the switch, and uses the existing identity/pose acceptance and logging path. Cycles requiring
three or more pairs and unassigned objects are outside this tie detector. Camera validation must audit the existing
look writer and budget costs before adoption; G3's perfect identities only test the no-tie path.

### Imitation IMIT2 switches

All default to today's behaviour; G3 mirrors them with `--skill-attr` (`imitate_pairing=cues`, `imitate_order=...`,
`imitate_carry=reach`, `imitate_retry=refind`).

- `imitate_pairing: cues` — twins by appearance: the runtime wires `provider.twin_costs` (perception op `twin_costs`,
  `dojo.perception.twin_match`) to the episode start's frame; `copy_demonstration` reads it once before the
  demonstrator moves anything. Colour histograms and plane-corrected size at each believed pose, one Hungarian
  assignment, swap margins / ties logged (`twin_costs` event). No belief write. Without a reader (G3) or on a read that
  does not cover the labels it falls back to `one_to_one`, so G3 cannot measure it; the believed-identity look passes
  (`imitate_look`, `imitate_tie_look`) do not run when the cue read paired the twins.
- `imitate_order: persist | elim | persist_elim` — `persist`: the departure read goes back to perception without its
  gone list, and a spot counts as left after `PERSIST_READS` reads in a row, dated by the first; `elim`: with all but
  one demonstrator object read leaving, the last one goes last. G3's watch is an arrival read, so only `elim` acts there.
- `imitate_carry: reach` — for an object `REACH_MARGIN` nearer the other arm's base, `put_in_open` tries the
  container's arm with centred grasps only, then a table handover by the side arm (phones: the handover first).
- `imitate_retry: refind` — up to `RETRY_TRIES` puts per twin while `RETRY_MIN_STEPS` remain; after a failed put the
  low hands rise before going home, and a belief moved > 40 mm from the twin's start spot is proposed back through
  `provider.confirm_pose` (a belief writer: include `confirm_pose` in the write audit).

Dev scorers for recorded camera rows (spatial GT from our layouts, demo order from the pickle's `aim_label`, never a
policy input) are in the IMIT2 report; `scripts/skills/imit2_g3.sh` runs our-slice G3 panels on compute3 light locks.

### Stacking-toy held-ring read

`RuntimeConfig.peg_held_read: on` (default `off`; G3 `--skill-attr peg_held_read=on`) asks the existing held-pose protocol
for a ring-to-TCP measurement at the peg hover in mode B. It requires lateral uncertainty <= 0.5 mm and axis uncertainty
<= 7.5 degrees, may use the existing bounded look/show protocol, and retains guarded threading on an accepted transform.
A rejected/missing read uses the original predicted transform. It does not enable seating or change stuck-piece retries.
Use with `held_pose: request` and the adopted base refinement; validate camera scores and belief writes before adoption.
The dev-only `scripts/skills/toy_release_audit.py` compares hover attachment and pre-release hole/peg errors against a
physics replay captured with `m95_replay.py --every 1`; its GT is never used by a policy.
The provider can reanchor belief on an ok read before the tail's uncertainty gate. For this lever, explicit
`belief_write` events with writer `held_pose` capture those changes, including tail-rejected reads; include them in the
harmful-write audit. The safe tail fallback alone does not establish harmless belief writes.

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

### Segmenter dev gate

`scripts/perception/seg/bake_seg.py run` selects frame 0 on dev / training ids
100000–04 and 100030–49 only, including the w51 / p60 gate sets when selected.
Use `scripts/perception/seg/dev_gate.py cut SOURCE OUT` to archive that slice of
an existing bake with provenance hashes; this reuses predictions, not inference.
`dev_gate.py compare BASE CANDIDATE` requires matching frame/object keys, unique
objects and no errored frames, and reports identity and 10 mm position separately.
`--diagnostic` can inspect errored archives but marks them invalid for adoption.
Use `--by-asset` to split categories by retained GT asset index; report both
resolutions so a win on one asset cannot hide a loss on another. The genuine-dev
new-class hitcheck remains a separate required gate.

Before incremental segmenter training, run `audit_rehearsal.py --base-classes
BASE/classes.json --export EXPORT --out support.json`. It checks the exact old
class prefix, variant training ids, unique-image support and the validation split;
exit 1 means at least one old class has zero training support. RFS repeats and
validation images cannot satisfy rehearsal. This audits manifests, not whether
image files still exist. Keep the support report and source hashes with the recipe.

GPU / software-GL bake workers explicitly register PyOpenGL's installed ctypes
parameter handler. Older PyOpenGL plugin lists can omit Python 3.12's output
pointer type, causing `glGenTextures` to fail on textured assets even with working
CUDA / EGL. Changing this setup requires a fresh paired baseline, not dropping
failed frames from the gate.
For a matched rehearsal ablation, `narrow_rehearsal.py` derives the control from
an already labelled full export. It retains every appended-class image and its
exact full-export RFS repetition count; only old-only rehearsal differs.
`audit_rehearsal.py --allow-missing-old` is for this diagnostic control only;
vocabulary and split validation still apply. Candidate training uses strict audit.

`train.py --batch -1 --batch-headroom-gb 8` opts into auto batch sizing and an
allocator cap from available VRAM minus that reserve. It records the resolved
integer in `resolved_batch.json` for reuse in matched arms, and stops if free
VRAM drops below 6 GiB. Without the option, training defaults are unchanged.
Training extras explicitly marked by `extra_groups` can resolve canonical Rigid
keys stored under Clutter; ordinary task generation keeps its original lookup.

## RoboDojo research board

The durable 54-cell research board lives in [scripts/dojo/board](../scripts/dojo/board/README.md).
It replays a hashed recovery checkpoint and can refresh only the permitted compute4 integration
rows over read-only SSH. State defaults to `~/dojo-board`; `--state-dir` changes it.
Sealed measurement ids are excluded before report reads and at ingestion; sealed scoring is unsupported.
Historical pre-registered decisions are preserved; new experiments follow the Camera A/B rule.

## Replaying agentic route requests

`scripts/perception/route_request_replay.py ROOT --out rows.jsonl` reconstructs
teacher-forced answer requests from copied dev workspaces (`ROOT/<name>-<layout>/call_*/*`).
It runs no tools, renderer or robot. It requires the original prompt pin, saved
PNGs, untruncated tool output and historical request byte counts. It rejects
repeated tool-image filenames (the stored image may have been overwritten).
Only ids 100000–04 / 100030–49 are accepted. `--live --model cx/gpt-6-sol`
permits at most 200 cx/ requests, with automatic retries disabled; credentials
come only from the named `AGENTIC_ENV_FILE` or OMNIROUTE environment source.
Saved rows include historical replies, new replies, latency and token usage.
The probe compares single decisions conditioned on the stored Sonnet history;
it cannot measure a new model's full tool trajectory or episode success.

### Aggregate-only imitation infrastructure census

`scripts/dojo/imit_infra_census.py ROOT` summarizes old `c4b_imitatesg` candidate attempts using only category,
success, rc and steps_used. It restricts reads to measurement ids 100005-09 / 100020-29 and known suffixes,
separates archived rows, and emits no episode ids or diagnostics. Pipe it over SSH to inspect a read-only host
without copying a script there. It does not select the board's counted family or diagnose scoring episodes.

### Wide conveyor interceptions

`RuntimeConfig.belt_waist_first` (`off` by default; G3 `--skill-attr belt_waist_first=on`)
tries mesh waist grasps before shallow slice/rim grasps when the rider has no top grasp.
An unreachable waist falls back to the existing surface search. Narrow top grasps retain
their search. This changes grasp selection, not tracking or identity; keep `belt_wrist_veto`
independently configured. Both conveyor skills share this interception planner.

### Integration camera queue

The compute4 Integration runner is versioned at `scripts/dojo/queues/lane2.sh`.
Set `LANE_REPO` to a merged deployment and use private heavy1/heavy2 locks.
Admission waits at used VRAM >16384 MiB (failed queries also wait), checks a
50 GB disk floor, and serializes startups until perception readiness plus a
150-second startup window. These guards coordinate our lanes; other GPU users
must retain their own admission gates. `wave_status.py` reads only our lane logs
and reports launcher completion/rc and prospective remaining occupancy.

The shared boot protocol reads only `~/coord/lane_gate.stamp`, waits until its
epoch is at least 120 seconds old, then writes `<epoch> <iso> ilia-ab`. While the
other team runs its single Isaac, use one Integration lane (`lane2.caps: 1 4`).

### Pen-holder G3 switches

Use `--skill-attr fill_grasp=vertical` (or `rolled` for rolled shafts only, `rolled_tilt` to add approach tilts), `fill_holder=stay`, `fill_drop=center` / `inset`, or
`fill_insert=safe` to screen fill-container changes independently. Defaults are
`off`; runtime equivalents live in `RuntimeConfig`. The vertical preference keeps
the original candidate set as fallback. Check both pick feasibility and final
pens-in counts: reducing slip alone can still lose reachable transfers. Keep
raw variant records and their presettled records together under a private data
root; pass that layout directory explicitly to both paired G3 arms.

`--skill-attr fill_guard=geometry` screens the conditional profile; leave the four
independent fill switches off (mixed settings are rejected). Runtime uses the same
field. It selects the profile from holder position, arm reach and flat slender
shaft geometry, then restores the scoped switches. Inspect the per-episode
`fill_guard` measurements alongside scores and holder orientation: a score tie
can hide a new tip after a reward tier has already been banked.

### Matching-slot Cartesian approaches

`RuntimeConfig.slot_approach` is `off` by default. `linear` makes matching-slot transfers use
`plan_place(approach_line=True)`: a collision-checked Cartesian path to the placement hover.
It applies to level carries and the reorienting/relay fallback. An infeasible Cartesian path
rejects that candidate; it never falls back to a joint-space transit for the same candidate.
G3 exposes the same switch with `--skill-attr slot_approach=linear`. Pick, descent, release,
held-pose measurements and all non-slot places retain their existing settings.

`RuntimeConfig.slot_hover=low` (default `off`) independently sets matching-slot pre-place
height to 15 mm and support clearance to 3 mm. It retains collision-checked approaches;
the container top and tool's lowest corner still set the minimum hover. G3 uses
`--skill-attr slot_hover=low`. Do not combine it with Cartesian approaches without a separate comparison.

`RuntimeConfig.slot_place=servo` (default `off`) uses the existing measured placement tail
for matching slots, disables insertion probes, and requests the existing held-pose protocol
at the hover. G3 re-measures the tool between tail moves; perception mode can only use the
provider's accepted held transform. Configure a held-pose provider to test the camera effect;
this switch does not turn on a perception service or supply ground truth.

`RuntimeConfig.slot_carry_read=on` (default `off`) splits the slot approach into at most
four checked Cartesian hops. It interpolates object orientation from lift to hover and
updates attachment between hops. Perception uses gated current-view held reads only;
it adds no show/observer moves. An unavailable first read keeps the legacy approach;
a later missing read retains the last accepted attachment. Infeasible hops stop motion
while retaining the grasp for the existing recovery path. G3 re-measures from its provider.

### Egg-holder G3 switches

For egg-holder feasibility screens, `RuntimeConfig.egg_grasp_height` (`off` / `raised`)
adds upper-shell pinch candidates, and `egg_grasp_angles` (`off` / `wide`) adds steeper
round grasps. `egg_wall_model` (`off` / `shell`) tests the lower/upper radial mesh envelope;
`egg_drop_approach` (`off` / `low`) lowers the preapproach; `egg_drop_spin` (`off` / `axis`)
adds spins about the round object's long axis. All default off and map to G3 `--skill-attr` switches. Dense wall checks,
collision planning, pick verification and the episode budget still apply. Pair settled
layouts from the same generator; v6 egg records must not be mixed with v15 records.

The `integ-c3` camera profile accepts only heavy1 on `~/dojo-gpu.lock`, skips
coord, admits at most 14 GiB used VRAM, and logs `who`/GPU status before each
job. `LANE_ENV_FILE` selects a private box environment without writing a shared
home-level file; other profiles retain their existing default.

### Alternate handle orientation on hooks

`RuntimeConfig.hang_half_turn` (default false; G3 `--skill-attr hang_half_turn=on`)
adds a second handle orientation to threaded-hook transfer planning. It turns the
handle frame 180 degrees about its upward axis: the hole center and threading
line stay fixed, while the wrist gets another approach. Original orientations
are tried first within each lift strategy. Cradle placements are unaffected.
This expands geometric candidates without changing hook eligibility, collision
checks, release motion or scoring. Camera adoption requires a paired experiment.

`RuntimeConfig.hang_handle_grasp` (default false; G3 `hang_handle_grasp=on`)
adds pinches across the handle's outer bar before the body-wall candidates.
The catalogue's handle frame proposes the location; opposed mesh faces must
confirm a 4–84 mm width. Missing meshes or unsuitable frames add no candidates.
The ordinary pick and transfer planners retain their collision checks.

The dev-only `scripts/skills/hang_phase_report.py` audits G3 handle geometry
from saved true poses. It reconstructs current action intervals from trajectory
segment lengths because frame `seg` names the previously completed segment.
Its post-hoc phase bins and continuous coordinates are diagnostics only; no
policy consumes them.

`RuntimeConfig.hang_carry_slow` (default false; G3 `hang_carry_slow=on`)
halves the threaded-hook reorienting carry rate to 0.03 rad/step. Cradle,
pick, slide, lower and release behavior retain their existing settings. The
switch changes carry pacing only; the episode step budget stays fixed.

For the opt-in egg endpoint-search experiment, use G3
`--skill-attr egg_reach=search` (runtime `egg_reach: search`). The default is `off`.
`tests/dojo/test_egg_reach.py` covers scoped practical bounds and endpoint rejection;
`test_egg_grasp_height.py` covers unchanged baseline geometry. Its endpoint screen
must never bypass the subsequent collision and motion planner. Clear the IK cache
when entering and leaving a scoped joint envelope: cache keys omit joint limits.

`RuntimeConfig.headset_seat` (default `off`; G3 `headset_seat=on`) changes
cradle hangs only. It retains grasps at least 35 mm wide, adds eight squeeze
steps before lift, halves the cradle carry rate, and lowers the measured bridge onto the cradle before
opening with longer settle pauses. Threaded hooks retain their original pick
and release behavior. Use with `hang_cradle=straddle` to screen the seating
change independently of cradle collision feasibility. Missing wide grasps
refuse the transfer rather than falling back to the slipping narrow pinches.

`RuntimeConfig.headset_carry` defaults to `legacy`; G3 exposes the same enum.
`staged` scopes a two-leg carry to non-threaded cradle hangs: rotate above the
held object's current xy, then translate to the hover along a checked Cartesian
line at fixed orientation. A failed leg rejects the whole placement. This
retains the existing cradle contact allowance and all hand/arm collision checks;
it does not enable straddling or change grasp/loose verification by itself.

`RuntimeConfig.headset_grasp` defaults to `legacy`; G3 mirrors it. `dense`
changes only cradle proposals to 400 opposed-face samples and thirteen
approach angles, retaining box/slice candidates at widths >=35 mm. It screens
up to 24 full hang plans instead of eight. Threaded-hook proposals retain
the legacy generator. This does not bypass physical lift/loose verification
or enable staged carries, straddling or seating independently.

`RuntimeConfig.headset_table` defaults to `bbox`; G3 mirrors it. `mesh`
scopes a full-mesh held-object/table check to cradle hangs and restores the
previous collision mode on return or exception. It retains the arm/finger
checks and table tolerance. Every catalogue vertex is transformed; any mesh
XY bbox overlap with the finite table checks global minimum z conservatively,
covering triangle crossings with no inside vertex. Missing/invalid meshes
retain the legacy bbox check. This avoids empty OBB corners blocking a lift;
it is not a smaller margin or an unchecked contact move. Geometry hanging
outside an edge can be conservatively refused. Native table/obstacle checks
still validate every planned waypoint. Threaded hooks retain their old mode.
### Multi-bottle pour profiles

`RuntimeConfig.ported_pour_profile` / G3 `--skill-attr ported_pour_profile=...`
accept `off` (legacy dispatch), `sparse` (Mikhail's multi-bottle v2 roll profile,
without via points), and `via` (v2 including target-row carry waypoints).
They supersede `ported_pour_pace` when enabled; `ported_pour_aim` remains independent.
The existing default-off `new_routes: [pour_order]` / G3 `--routes pour_order`
routes the three color-named bottle/bowl clauses in instruction order. It refuses
incomplete or repeated-color assignments. No task/checker labels ground targets.

The `checked` pour profile uses the original strict RGB wrapper's underlying
v5 skill: staged rolls, mandatory waypoint retries, checked put-downs. These
profiles are dev feasibility switches; they do not select a camera candidate.
`scripts/skills/pour_g3_job.sh` serializes the POUR stream on compute3's light7
lock, checks <=24 GiB used VRAM and >=10 GB free disk before each scoped job,
and uses an 8G/600% physics-only process with a bounded KILL tail.

For ordered pours, `checked` runs the original v5 composite (adaptive drain,
pre-flow-only retry, stop on an unsuccessful pair, checked held-bottle return).
A still-held bottle suppresses generic finish, which otherwise opens both hands.
Single-cup checked pours use the v5 bottle skill directly.

`ported_pour_pace=budget` is an opt-in overlap derivative. Flow angles, flow
speed, hold count, drain and staged untilt angles remain the overlap profile;
lift uses normal speed and staged untilt uses the existing full held-speed
limit. The last pour alone receives `profile_reserve=0`: the 60-step put-back
reserve no longer cuts its flow profile short. The put-back reserve and all
move-budget/lease checks remain in force. Single-cup routing retains v1 pacing.

`scripts/skills/pour_g3_panel.sh` executes an existing private
`~/dojo-data/pour/panel-registration.json`. It verifies source, stable-layout
hashes and a fixed A/B order, alternates AB/BA by layout, and refuses duplicate
or mixed-source/settings rows. Completed episodes are skipped on resume;
admission or process failures stop the driver for diagnosis. It never generates
layouts or selects a candidate from outcomes.

`python3 scripts/skills/pour_g3_report.py --root ~/dojo-data/pour` reads the
registration and retained rows without simulator imports. It reports paired
coverage, official/full-only means and sample SD, >=10-point wins/losses,
clean final-predicate successes and first-blocking-stage diagnostics. Missing
or infrastructure pairs cannot yield a passing completed decision.

The basket-image experiment uses `--routes basket_image` and
`--skill-attr belt_container=on --skill-attr belt_deposit=on` in G3;
`--skill-attr belt_lift=braced_hook` selects its candidate lift. Runtime mirrors
these fields; all are off by default. CPU guards live in `test_basket_hook.py`.
Belt riders cannot pass the stationary presettle check: use a documented
belt-relative rest check and identical frozen inputs in both comparison arms.

The lift-first profile is independently enabled with `belt_first=on` and
`belt_target=match` (both default off), mirrored by RuntimeConfig and G3.
`test_basket_first.py` guards dispatch, legacy target/planner arguments and
restoration/intersection of the temporary measured joint envelope.
`settle_native_phase.py --ids ...` removes total belt travel from authored
positions, avoiding the extra horizontal phase introduced by the round-1
snapshot method; its pure correction has a CPU test.

`headset_stage=center` (default `off`, mirrored by Skills and G3) adds one
checked table rest when pickup and cradle favor different arms and their xy
separation exceeds 0.40 m. It tries up to eight native free spots but executes
at most one transfer in legacy retry mode, retaining the lying orientation and physical verification.
Only a verified stage enables additional narrow ear-cup proposals: 18–35 mm,
at least 60 mm below the bridge, broad pad patch and >=6 mm contact depth.
The original loose-hold and method-slip rejection remain. Staging failures
return to the ordinary hang planner from the refreshed world; threaded hooks
never enter this path. `test_headset_stage.py` covers gates, checked planning,
failed physical holds, scope/defaults and proposal geometry.


`headset_grasp=cups` adds geometry-verified narrow cup proposals even before
a successful stage; `legacy` and `dense` retain their original banks.
`headset_retry=sites` (default `legacy`) keeps failed pinch neighborhoods in
the object frame per arm, including the actual refined pick attachment.
It excludes nearby equivalent approaches/closing axes and permits four physical
staging and four hang attempts. Plan refusals do not trigger repeated identical
searches; place/budget/end failures are not retried. Every attempt retains
native planning and the original lift/loose verifier and episode step budget.

`headset_grasp=cups` opens to the full 88 mm jaw gap before descent; all
finger/table/scene checks still apply. `headset_stage=mesh_center` preserves
the centre-stage geometry and sets its rest height from all mesh vertices
plus 3 mm clearance; invalid geometry uses the previous bbox target.
With `headset_retry=sites`, a failed measured-attachment place re-plan puts
the object down and rejects that pickup site instead of carrying a nominal
attachment. Legacy retry retains the original fallback.
`headset_lift=slow` (default `legacy`, mirrored in Skills/G3) uses a checked
0.03 rad/step lift in cradle pickups and staging; threaded hooks keep their
original lift. Closure, lift height and physical hold thresholds are unchanged.
`headset_stage=mesh_clear` keeps the mesh-ground target and plans a 25 mm
release gap. Endpoint screening uses the release pose; the rebuilt full
place and pickup repair retain the ground target for physical verification.
Every leg uses native hand, held-object and scene checks.

`headset_hold=loaded` (default `legacy`, mirrored in Skills/G3) can prove a
nominally loose headset pickup through actual following motion. It applies
only where the original loose rule would put the object down; tolerant relays
retain their original execution. Only an originally successful loose lift qualifies: hold30 stationary steps, then
plan/check a20 mm vertical raise/return with8 steps at each endpoint. Refreshed
attachment samples must stay within3 mm/3 degrees, the object must rise17 mm
and its full mesh clear the table20 mm. Original hard rise/30 mm slip gates
run before and after the challenge. Accepted results retain nominal drift
in the log and require a measured-attachment place plan. Missing meshes,
static cached beliefs, unstable holds or failed native paths cannot prove a
hold. The existing step observer is preserved/restored; default and other
object transfers execute their old path. This is a G3 mechanical proof;
camera use needs fresh physical held-pose observations.

### Write-audit identity and held transport

Held-pose `belief_write.before7` is the prior attachment synchronized to the current measured TCP; `before_sync7`
retains the cached raw world pose. Computing the synchronized prior is logging-only and does not add a live belief
synchronization or observation. `m95_score.py audit` associates labels chronologically: unambiguous <=10 mm proximity
anchors are provisional until the first consumed pick or saved grasp-close evidence finalizes the identity. Subsequent
retries cannot rebind it. Every row records its association source/step and fallback flag; `association.json` counts
fallbacks and missing body geometry. A missing already-anchored GT object fails rather than selecting a neighbour.

`scripts/perception/audit_supplementary.py` re-scores copied frozen ledgers against saved exact write frames without
starting a simulator. It accepts frozen body-bounds/group caches, and keeps the 2 mm body-centre definitions unchanged.
Its optional legacy-show repair records exact-flange versus commanded-anchor-FK provenance; approximate legacy
reconstructions are supplementary diagnostics, never retroactive qualification of a registered verdict.

### match_and_pick MATCH2 switches

Six `RuntimeConfig` fields, all `off` by default, each a skill parameter (Skills attributes, mirrored by G3
`--skill-attr`; none keyed by task name, none reading a task config). G3's GT provider has no belt tracker, so
`belt_ref`, `belt_veto`, `belt_speed` and `belt_wrist_id` are inert there:

- `belt_ref=tracked`: the remembered first object is the leading start rider the tracker still holds after its first
  reads (a frame-0 phantom ahead of it is dropped, never confirmed), and its key is the leading asset of its own
  belt-read history (`RuntimeProvider.identity_dist`). Matches keep today's key / vote-share rule. `belt_ref=similar`
  also scores matches against that history (`interactive.ref_match`: Bhattacharyya similarity plus a
  competing-category veto); offline on recorded camera reads it took the true match less often than the key rule.
- `belt_veto=evidence` (with `belt_wrist_veto=on`): an empty pick look vetoes only when the prior was in the wrist
  view (`visible_frac` >= 0.5: not under our fingers, not out of view) or the head tracker has not matched the track
  for 30 steps (`RuntimeProvider.track_age`).
- `belt_speed=lock`: the belt speed is measured once from the first long, clean track (8 reads over 36 steps, median
  line residual <= 4 mm), projected on the belt axis and kept (event `belt_lock`).
- `belt_grasp=solid`: a rider with no top grasp that is not a flat dish (`interactive.flat_rider`) is intercepted with
  mesh-antipodal grasps ranked by `grasps.lift_hold`, then a waist, rim pinches last. A plate keeps today's order.
- `belt_retry=window`: up to six match rounds while 60 steps remain.
- `belt_wrist_id=on`: a moving rider's pick look re-identifies it over every rider candidate
  (`policy.observed_candidates`: the identities the belt tracker read this episode, never a task config's candidates); a confident other category than the first object's is not picked (status `other`).

`save_belt_look_frames: N` (dev) keeps the first N belt pick-look frames. `scripts/skills/match2_decision_replay.py`
replays match_and_pick's decision (first object, speed, the match it would take) on recorded camera reports (frame-0
record + logged reads, robot at home, stub pick); its GT is the layout moved along the belt, for scoring only.
`scripts/skills/match2_g3_job.sh` runs one scoped compute3 physics job on light3 / light4.

### Thin-disc contact switches (COIN2)

Six default-off `RuntimeConfig` fields, copied onto Skills and mirrored in G3 `--skill-attr`
(`dojo/skills/coin_contact.py`): `disc_align=walls`, `disc_snap=on`, `disc_lift=shallow`, `disc_seat=table`,
`disc_release=staged`, `disc_reserve=on`. They act only inside `precision.disc_into_slot` /
`plan_disc_handover`, which `insert_into` reaches by geometry (a thin disc, `disc_axis`, standing in a fixed holder,
`holder_of`), never by task name. A planner selects them per call with `insert_into(..., disc_contact="on")`
(`coin_contact.PROFILE`, scoped to the call and restored on exit); `None` keeps the configured switches. The agent
manual and the J2 registry do not document the argument yet: adding it there changes the chooser prompt for every
task and needs its own manual version and A/B.

- `disc_align=walls` reads `<arm>_ee_joint_state_official` (strict v2, `DOJO_GRIP_SOURCE=v2` in G3). Without the key
  the probe ends with "no official jaw reading" and writes nothing. Its writes go through `provider.set_pose`
  (runtime: a `belief_write` with writer `wall_probe`, before7 / after7; G3 `OffsetBeliefProvider`: the offset is
  re-derived; the GT provider has none). Include `wall_probe` in the write audit.
- The V's slope varies by layout and arm (0.0016-0.0021 per degree in the twin): the fit takes it free within
  `WALL_B_RANGE`. The intercept (pads square and centred) comes from the holder's catalogue mesh at the squeeze
  height (`wall_intercept`); `WALL_A` only when the mesh is unknown.
- `coin_contact.insert_thin(obj, receptacle, slot_pose7=None, contact="on")` is the general entry: a perceived slot
  pose gives the targets (`slot_targets_from_pose`), else the receptacle's perceived pose and catalogue frame. Only
  discs run the contact sequence; another thin shape is refused with a named stage.
- `plan_pick` gained `snap_in`, `snap_rad`, `lift_step` and `post_lift`; all default to today's plan (the op list is
  unchanged, `tests/dojo/test_coin_contact.py`). `plan_place` / `_place_tail` gained `release_stages`.
- G3 `--belief-offset-file F` (dev) gives per-layout belief offsets `{seed: {label: [dx, dy, dyaw]}}`, on top of
  `--belief-offset`. Camera-like coin errors: `docs/experiments/data/coin2/camera_pick_errors.json`.
- `scripts/skills/coin2_stages.py` scores the contact stages from `m95_replay.py --every 1 --full` replays (GT, dev
  only): fingertip clearance at each close, hold loss stage, slip, end state, release pose in the bank's frame.
  `scripts/skills/coin2_g3_job.sh` runs one scoped compute3 job on light5 / light6.

### Offline SAM task-prompt comparison

`scripts/perception/sam/collect_task_frames.py ROOT` exports existing `c4b_` /
`c6b_` frame-zero look captures as a tar stream without writing to the source.
It reads only dev IDs 100000–09 and 100020–24, excluding the historical v2b
training range. GT comes from the exact `layout` path in each episode manifest,
including salted inputs; a same-numbered standard layout is insufficient.
Loaded raw and settled inputs are marked separately. `--phase-look` exports
a small pre-grasp survey panel; report its unchanged-object-pose assumption
separately from frame-zero evidence.

`task_prompt_eval.py prepare ROOT` uses `seg_twin.gt_code_map` on CPU Mesa
(`PYOPENGL_PLATFORM=egl`, Mesa vendor, `LIBGL_ALWAYS_SOFTWARE=1`,
`EGL_DEVICE_ID=1` on compute3). `pred ROOT --out ROWS --model sam3|sam3.1
--checkpoint CKPT --weights V2B --classes CLASSES` runs the paired component
comparison. SAM3.1 takes the original multiplex safetensors, constructs its
text detector without the unused tracker, and checks checkpoint compatibility.
Use an 8 GiB / 600% systemd scope on shared compute3, one model arm at a time.
Inference checks the 24 GiB-used admission ceiling and 10 GiB disk floor and
caps its CUDA allocator at 30% of device capacity. Store caches under the
experiment directory, not a shared package environment.

`task_prompt_summary.py ROWS... --out DIR` emits summary, frame and instance
CSVs. Prompt categories are distinct from asset identities: ambiguous category
sets are not correct class reads. Union de-duplication retains v2b identities
first and adds disjoint SAM masks at IoU <=0.5. This is an offline component
comparison, not a runtime proposer or an end-to-end task-score evaluation.
### Internal RSI yardstick launcher
`baselines/rsi/` measures the external fixed-program baseline on our hashed
inputs through the official eval client; its programs never enter policy code.
All host/state/lane/lock paths are explicit arguments. The desktop profile
shares one prep/policy slot in a bounded user scope and yields to desktop
load; the camera profile uses the existing lane2 caps and startup/boot gates.
See [the install recipe](../baselines/rsi/INSTALL.md). CPU tests mock telemetry
and verify admission, credential isolation, portable commands and input hashes.
Running a recipe still requires current permission for the destination box.

### Task-scoped SAM camera proposals and shared service

**Diagnostic-only; prohibited for a board candidate.** `sam_task_routes` and
`sam_task_banks.json` are historical component experiments, default-off. Never enable
these fields or their task-keyed prompt lists for board episodes. The board requires
one general perception path, with no task names, task-keyed code, task-config values
or checker values. A general mask proposer must consume only episode instruction text
and the one shared catalogue; thresholds and post-filters must be uniform. The legacy
inverse-layout fitter also reads task-config candidate pools/priors, so replacing only
its mask prompts does not by itself establish board compliance.


`EstimatorConfig.sam_task_routes` defaults to `{}`. Keys are the instruction-routed
vocabulary passed to perception (as for other task switches), never launcher task IDs.
Each entry accepts `mode: v2b|sam|union`, `bank` (same task), `model: sam3|sam3.1_text`,
`threshold`, `dedup_iou` and conveyor `roi` / optional `labels`. Example perception
JSON overlay: `{"sam_task_routes":{"pour_by_language":{"mode":"sam","model":"sam3","threshold":0.2}}}`.
Merge it into the candidate's `DOJO_PERCEPTION_CFG`; do not discard the candidate fields.
No entry means the existing v2b path. `sam` replaces detector proposals; `union` retains
v2b first and adds SAM masks at IoU <=0.5. Background proposals remain unchanged.
SAM categories restrict the public candidate fits and cannot supply an exact-asset hint.
Existing silhouette acceptance and appearance identity stages still apply. The committed
`dojo/perception/data/sam_task_banks.json` is SAMBAKE's phrase inventory; runtime reads
no evaluation manifests or GT. Conveyor ROI filters are separately registered hypotheses.
The existing `sam_fg` and `sam_propose` flags remain independent and unchanged.
Requested SAM routes require a live service and fail as infrastructure errors if it dies.

On a **lead-allocated, drained compute4 or compute6**, deploy the merged code to a pinned
`~/controlr*` directory. Never update a running lane's deployment. Install the server's
separate venv with Python 3.12, torch **2.9.1+cu128**, torchvision, safetensors, timm,
ftfy, regex, iopath, einops, psutil, hydra-core, omegaconf, Pillow, numpy and Meta SAM3
at commit `f1647e8d0c1d280991f4d1b7d974c2704537fe66` (no SAM2 needed for these routes).
Use a dedicated `~/controlr-sam/venv`; do not alter simulator Python. Copy the already
verified checkpoints from compute3's `~/dojo-data/sambake/models/sam3.pt` and
`sam31.safetensors` to `~/dojo-data/models/sam3/sam3.pt` and
`~/dojo-data/models/sam3.1/sam3.1_multiplex_fp16.safetensors`, respectively. The lifecycle
script verifies both SHA256s before starting; no model download is performed.

```bash
export LANE_REPO=$HOME/controlr-<merged-pin> SAM_PY=$HOME/controlr-sam/venv/bin/python
export LANE_STATE=$HOME/dojo-integ-c6-state LANE_STREAM=integ-c6  # c4: c4-state / integ-c4
export LANE_SHARED_SAM=sam3  # egg/conveyor wave: sam3.1_text; drain before switching
bash "$LANE_REPO/scripts/dojo/queues/sam_shared.sh" start
bash "$LANE_REPO/scripts/dojo/queues/sam_shared.sh" status
# Both lane queue rows use --sam sam3 (or both sam3.1_text), never a per-lane SAM process.
# After STOP/drain and completion of both jobs:
bash "$LANE_REPO/scripts/dojo/queues/sam_shared.sh" stop
```

Start also maintains the existing shared v2b/digit/kong services, using the pinned lane
venv, and enables `shared.on` only after SAM health. Stop stops the shared bundle after
the lanes drain. `shared_side.sh` serializes lifecycle operations; lane `shared_ok()`
checks PID, socket info and exact loaded models before/after each job. The service
serializes inference across clients. A dead/replaced service requeues the row and stops
the lanes. `side.shared-sam` counts one big service in `sides()`; nvidia-smi admission
includes its actual residency. Default admission remains unchanged. A measured
single-checkpoint wave may explicitly export `LANE_SAM_LANE_MIB=12288` to reserve 12 GiB
free per new rendered lane (20 GiB used ceiling), only with healthy shared SAM. Keep
this opt-in unset until the box's actual lane footprint is checked. Do not load both
checkpoints for a two-lane wave: their measured 9.46 GiB service residency leaves too
little room for two lanes and the other services. Load one checkpoint per box/wave.
Use `scripts/perception/sam/shared_bench.py` under a light lock to reproduce startup,
request and two-client timings on an existing permitted RGB. It terminates only its
own server child. Detailed measurements and parked specs: `docs/experiments/data/samint/`.


The SAM route key enters at the perception RPC's explicit `vocabulary` field.
`_Perception.estimate` forwards it as `inverse_layout.estimate(..., task=vocabulary)`;
`sam_routing.resolve(config.sam_task_routes, vocabulary)` selects the configuration.
It is derived from permitted instruction routing in the current runtime, not launcher
metadata or an environment task ID. A model planner using a skill registry can supply
the same canonical perception vocabulary explicitly: the estimator does not require
an instruction-pattern branch to choose SAM. Registry entries must map to an existing
public candidate vocabulary/bank key; an arbitrary skill display name is not a bank.
The task key also selects the existing public candidate pool. SAM's route map chooses
mode/checkpoint/thresholds/filters, while the existing fit stage chooses asset identity.

The key is per **estimate request**, including candidate vocabularies evaluated by
`RuntimePolicy._explain_vocabulary`, before the final route is chosen. For a scoped
SAM parity guard, inspect every `vocab_explain.candidates` entry as well as the final
vocabulary and record routing counters. A final inactive vocabulary alone does not
prove that an earlier candidate estimate made zero SAM requests. An active bank
in a guard's candidate exploration blocks the zero-request parity proof.
