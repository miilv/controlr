# Bridge runbook: our full system behind the official RoboDojo eval client

Our full system: LLM agent + dojo skills + perception + twin. It runs behind robot-harness's strict robot server,
which in turn runs behind RoboDojo's OFFICIAL eval client (XPolicyLab policy `harness_strict`, Isaac 5.1 with
rendering). Code: `dojo/bridge/` (package doc in `dojo/bridge/__init__.py`), tests `tests/dojo/test_bridge.py`.
Rules for Mikhail's side: robot-harness `AGENTS.md` (tiers, blind runs, md5 freeze) and `coord/J1-RUNBOOK.md`.

Marks on commands:
- **[CPU]**: runs on mocks (fake server, kinematic env), no GPU.
- **[needs #8]**: needs the `joint_steps` verb (robot-harness issue #8, §9).
- **[GPU]**: needs a GPU box with Isaac 5.1 rendering (see the journal for what has run).

## 0. The picture

One GPU box, one session lock (`~/dojo-gpu.lock`), five processes:

| process | owner | python | what |
|---|---|---|---|
| strict robot server (`robodojo_server_j2.py --strict`, requested) | Mikhail | `$RD_HOME/policy_venv` | the J1 server (strict tier B + `perceive` + B1 `held_look`) plus `joint_steps`; socket `$BENCH/run/<id>.sock` |
| no-op ws policy server + **RoboDojo eval client** | Mikhail (launcher) | RoboDojo conda env | the official episode: RoboDojo's reset, step accounting, `step_lim`, success latch, `_result.json` |
| perception service (`dojo.runtime.perception_service`) | us | `$CR/.venv` (perception extra, EGL) | estimate / refine / check / look poses / search regions on contract frames |
| twin service (`dojo.simlab.twin_service`) | us | RoboDojo conda env (`scripts/simlab/rdpy.sh`) | physics-only `Twin` for the scene derived from the instruction; spawned by the driver |
| bridge driver (`dojo.bridge.driver`) | us | `$CR/.venv` | `RuntimePolicy` / `AgentPolicy` unchanged, over the socket; omniroute for the agent |

Data path: the official env renders a step. The shim serves it to the server, and the server strips everything
outside strict tier B. The `obs` verb then gives us PNG paths plus `eef.<arm>.{q, flange, grip_cmd}`.
`dojo.bridge.contract` builds the runtime's contract observation from exactly those fields, and the policy runs.
Its per-step joint action lists go out through `joint_steps`: one RoboDojo `take_action` per entry, verbatim.

## 1. Prerequisites

| what | note |
|---|---|
| `joint_steps` verb on the strict server (+ shim op), and a start script for that server (`bin/robot-start-j2`) | Mikhail's lane, robot-harness issue #8; we never fork his server |
| robot-harness launchers runnable on the box | `bin/robot-start-j2` / `harness/robodojo_{start,launch}_j2.sh` source `${HARNESS_ENV:-/home/anywherevla/data2/bench/env.sh}`: on our box export `HARNESS_ENV=<an env.sh with this box's BENCH and RD_HOME>` |
| a GPU box with driver 570/580 | Isaac 5.1 RTX rendering (robot-harness D-P10) |
| our side | `dojo/bridge/`, `scripts/dojo/bridge_episode.sh`; CPU-tested (§3) |

Current state of each: the latest `docs/journal/*-dojo-bridge.md` entry.

## 2. One-time setup on the GPU box [GPU]

```bash
# RoboDojo + Isaac 5.1 + cuRobo + policy_venv as on isr-lab (robot-harness docs/SETUP-compute.md; RoboDojo f57e163,
# XPolicyLab 7074afb). RD_HOME = that install.
export RD_HOME=<robodojo install>
# robot-harness at a pinned commit (Mikhail's code; read-only for us) + an env file for this box (BENCH, RD_HOME, caches)
git clone https://github.com/msannikov03/robot-harness $BENCH && git -C $BENCH checkout <commit with joint_steps>
sed -e "s|^export BENCH=.*|export BENCH=$BENCH|" -e "s|^export RD_HOME=.*|export RD_HOME=$RD_HOME|" $BENCH/env.sh > ~/bench_env.sh
export HARNESS_ENV=~/bench_env.sh          # read by bin/robot-start-j2 and its launcher (and inherited by bridge_episode.sh)
# controlr at the commit under test (for a scored run: a pinned checkout nobody edits, md5-frozen)
git clone git@github.com:miilv/controlr.git $CR && cd $CR && git checkout <commit> && uv sync --extra dev --extra perception
# perception data (2.9 GB catalogue + backgrounds + colour maps) from compute3
rsync -a compute3:dojo-perception-data/{catalog,bg_folds,bg_folds_51} ~/dojo-perception-data/
# omniroute keys for the agent: OMNIROUTE_BASE_URL / OMNIROUTE_API_KEY only, in ~/controlr/.env (dojo/agent/llmio.py
# reads nothing else from it). A new machine holding .env needs the owner's OK (CLAUDE.md rule 5): ask first.
```

The venv needs the perception extra, which carries RoboDojo's generator deps (`tqdm`, `transforms3d`). The twin-layout
template (`dojo.runtime.estimate._template`) imports RoboDojo's `utils.cluttered_generator` in OUR venv. Without them,
every twin verify of a bridge episode fails. The proxy never showed this, because there the policy runs in RoboDojo's
python.

### 2b. isr-lab (Mikhail's 4090; the Unix user is shared): our tree only, his code unmodified

Everything of ours lives under `~/data2/ilia`. `~/data2/bench` and `~/data2/controlr-robodojo` are read-only for us.

| what | where |
|---|---|
| controlr checkout + `.venv` (py3.11, `uv sync --extra dev --extra perception`, `UV_CACHE_DIR` in our tree) | `~/data2/ilia/bridge/controlr` (`COMMIT` file; deployed as a `git archive` of the branch) |
| perception data: task-scoped catalogue subset (no `visual.glb`), `bg_folds{,_51}`, `color_calib{,_51}.json` | `~/data2/ilia/bridge/pdata` (~285 MB) |
| BENCH / RD_HOME: symlink farms over his installs (`scripts/dojo/isr_farm.sh`) | `~/data2/ilia/bench` (`run/` ours), `~/data2/ilia/rdhome` (`RoboDojo/eval_result/`, `RoboDojo/XPolicyLab/policy/` ours; `setup_policy_server.py` a byte copy) |
| the environment of every process of ours | `scripts/dojo/isr_env.sh`, with `HARNESS_ENV=scripts/dojo/isr_harness_env.sh` and `HARNESS_SHIM_HOME=~/data2/ilia/home` |
| GPU: the coord lock `~/data2/coord/gpu.lock` + `gpu.owner` + COORDINATION.md lines, inbox / `gpu.reserve` gate | `scripts/dojo/isr_job.sh NAME ETA cmd` (one hold) |
| several practice episodes in one hold (<= 2 h), gate + disk floor (7 GB) before each, prune + summary row after | `scripts/dojo/isr_block.sh BLOCK ETA SPEC` (SPEC: `ID TASK <bridge_episode.sh options>` per line; seed 0 layout 9 fixed) |
| one episode | `scripts/dojo/isr_episode.sh ID TASK <options>` |
| rendered 5.1 dev eval on OUR settled layouts (>= 100000): the proxy driver with `--renderer isaac51` (`RenderedEpisode`: one EvalEnv with RTX, the eval client's app flags), perception service + twin, no robot server; the gate runs before every episode (`DOJO_EVAL_GATE`) | `scripts/dojo/isr_dev_eval.sh NAME ETA --task T --layouts L... --policy P --config JSON [--twin-envs 8]` |
| rendered vs physics-only kit: replay an episode's executed actions, per-step object poses, divergence + phase | `scripts/dojo/replay_kits.py run ... [--physics-only] [--update-to-usd] [--no-fabric-ext] [--obs-every K]`, then `compare A B` |
| regenerate the exact contract observations of a dev-eval episode (physics is deterministic, so they are bit-exact) | `replay_kits.py run ... --save-obs-at s1,s2,...`; which checks: `scripts/dojo/frames_plan.py plan --budget-mb N` (negatives first), then `index` (matched by asset key + position, never by label) |
| stop a running block before its next episode | `touch $DOJO_DATA/bridge/STOP` (read by `bridge_episode.sh`; remove it to resume) |
| robot-harness's lock-aware starters under our hold (mock, `ROBOT_START_DRYRUN`) | `scripts/dojo/isr_lockmock.py` |
| rows (task, score, steps, wall, perception error at planning vs `dev.jsonl` after the close, verify, LLM calls) | `scripts/dojo/isr_summary.py` -> `~/data2/ilia/dojo-data/bridge/rows.jsonl` |

Why the farms: robot-harness's launchers write the server's run dir under `$BENCH/run/<id>/`. They also copy the shim
into `$RD/XPolicyLab/policy/harness_strict_j2/`, and the eval client writes its result under `$RD/eval_result/`,
relative to its cwd. With `BENCH` / `RD_HOME` pointing at the farms, all three land in our tree:
- RoboDojo's own path logic keeps the farm: `pwd`-based roots in its shell scripts, `abspath(__file__)` in
  `global_configs.py`;
- `PYTHONDONTWRITEBYTECODE=1` keeps `__pycache__` out of his trees.

The session lock `DOJO_GPU_LOCK` (`~/data2/ilia/home/dojo-gpu.lock`, fd 9) is ours. The coord lock is held around it,
so nothing of ours writes into the coord lock file.

Rules there (Mikhail's conditions for our use of his box):
- <= 8 runnable processes of ours, thread pools <= 4; the box hard-resets under all-core load;
- CPUs 0-11 only (`taskset`, set by `isr_job.sh`; Mikhail's jobs use 12-23, 24-31 stay free); never widen it;
- a load watchdog in `isr_job.sh` stops our tree when the 1-min load is above 12, or our CPU% above 1200, for more than
  30 s;
- never prune Isaac's shader / texture caches (their rebuild is an all-core spike); prune only run dirs and PNGs;
- deploy only between holds; record the commit in the checkout's `COMMIT` before a run;
- Mikhail's launcher files are md5-pinned (`isr_job.sh --pin`). A change, or his gpu.lock-taking starter appearing,
  refuses the hold (rc 78);
- disk: Mikhail's floor is 6 GB free on data2; ours is `ISR_DISK_FLOOR_MB` (default 7168, checked in MB before every episode);
- holds <= 2 h;
- `DOJO_ILIA=1` on every PID;
- kill only our own PIDs (`scripts/simlab/ilia_stop.sh`).

## 3. CPU preflight, no GPU, about 1 min [CPU]

```bash
cd $CR
uv run pytest -q tests/dojo/test_bridge.py                       # 14 tests: protocol, firewall, joint_steps, episodes
# the whole launcher path on the fake strict server (our runtime, the server's perceive route, no twin):
BENCH=/tmp/rbfake scripts/dojo/bridge_episode.sh --id fk-1 --fake --policy dojo.runtime.policy:RuntimePolicy \
    --perception server --verify off --out /tmp/rbfake/out
# -> "[bridge] fk-1: end=ended steps=172/550 ended=True ..." (the fake's latch: stacked + home), driver rc=0
```
With a robot-harness checkout that has `joint_steps`, the same file also runs his offline J2 stack
(`harness/j2_mock_shim.py`: the real J2 server + J2 shim on a fake task env) twice: once for the raw contract
(refusal codes, `actions_md5`, `tracking_rad`, `eef`), once for a whole RuntimePolicy episode through his validator.

Our backend against robot-harness's REAL strict J1 server code. His files are copied into a temp `$BENCH`, never
modified, and run with his offline mock shim; only `strict_calib.py` is stubbed:
```bash
CONTROLR_ROBOT_HARNESS=<robot-harness checkout> uv run pytest -q tests/dojo/test_bridge_rh.py
```
The agent on the fake server (live calls: about 1 per episode for the stack task; cx/gpt-6.1-sol by default). The agent
reads the router key only from the file `DOJO_AGENT_DOTENV` names, or from exported `OMNIROUTE_*`, never both and never
a home-dir `.env` by itself ([ENVIRONMENT.md](ENVIRONMENT.md)):
```bash
.venv/bin/python -m dojo.bridge.fake_server --id fk-2 --bench /tmp/rbfake &
DOJO_AGENT_DOTENV=$HOME/controlr/.env DOJO_AGENT_LOG=/tmp/rbfake/agent .venv/bin/python -m dojo.bridge.driver --server-id fk-2 --bench /tmp/rbfake \
    --policy dojo.agent.policy:AgentPolicy --perception server --twin off --config '{"extra":{"agent":{"max_calls":4}}}' \
    --out /tmp/rbfake/out --close
```

## 4. A strict full-system episode through the official client [GPU] [needs #8]

Practice only: **seed 0 layout 9** (robot-harness AGENTS.md rule 1). Scored seeds and layouts are run once, blind,
md5-frozen, on the lead's go (§6).

### 4.0 First on a new box: the `joint_steps` smoke (about 70 steps, no perception / twin / LLM)
```bash
scripts/dojo/bridge_episode.sh --id br-prac-smoke-s0L9-1 --task stack_blocks --layout 9 --seed 0 \
    --policy dojo.bridge.smoke:JointStepsSmoke --perception server --verify off
```
`dojo.bridge.smoke.JointStepsSmoke` moves each arm's base and wrist-roll joints out and back at 0.18 rad/step (the
skills' free-space rate) and toggles each gripper with a step command, one `joint_steps` chunk per phase. Read
`bridge.json` → `joint_steps_chunks`: per phase `tracking_rad` / `tracking_max_rad` (measured vs commanded after the
chunk), `max_step_dq`, `grip_change_while_moving`. Every chunk should run in full (`ran == n`). If tracking after the
3 hold steps stays far above the proxy's, re-time the skills (the twin verifies the same lists) before running 4a.
Then RuntimePolicy (`--policy dojo.runtime.policy:RuntimePolicy`, 0 calls), then AgentPolicy.

### 4a. One command
```bash
cd $CR
export BENCH=<robot-harness install> RD_HOME=<robodojo install> DOJO_PERCEPTION_DATA=~/dojo-perception-data
scripts/dojo/bridge_episode.sh --id br-prac-st-s0L9-1 --task stack_blocks --layout 9 --seed 0 \
    --policy dojo.agent.policy:AgentPolicy --perception local --verify bundle \
    --config '{"refine":true,"search":true,"post_place":true,"max_rehovers":2,"refine_max_shift_m":0.07}'
```
The config is the closed-loop arm of the proxy runs (`B_*_closed`, journal 2026-10-24-dojo-integ §6), the same for
every task. `--task / --layout / --seed` only start the official run, which picks the eval layout. Nothing of ours
reads them, and neither the driver's `info` nor any reply carries them.

What the script does, in order:
1. Takes `~/dojo-gpu.lock` for the whole episode.
2. `$BENCH/bin/robot-start-j2 --task T --layout L --seed S --dev --reserve 20 --id ID --wait-gpu 900`: the server,
   the ws policy server and the eval client (Isaac boots in 1-4 min). `--dev` writes truth to `dev.jsonl` for later
   evaluation only; we never call `success` or `end`.
3. Starts our perception service and waits for its ready file.
4. Starts the driver: it waits for `episode_pending`, then `reset` returns the instruction and step budget. It spawns
   the twin for the scene the instruction names (`--hold-gpu-lock`: the twin does not queue on the lock this session
   holds), then runs the policy until it is done or RoboDojo ends the episode. Then it calls `close`: RoboDojo writes
   `_result.json`, and an unfinished episode counts as failed.
5. Waits for the launcher's `launch_done.json` and stops what it started (only its own PIDs).

### 4b. The same by hand (one shell, in this order)
```bash
exec 9>>~/dojo-gpu.lock && flock -w 3600 9                       # the session lock (children inherit fd 9)
cd $CR; ID=br-prac-st-s0L9-1; OUT=~/dojo-data/bridge/runs; mkdir -p $OUT/$ID
export J1_PERCEIVER=dojo.perception.j1_adapter:perceive J1_PERCEIVER_PATH=$CR   # the server's perceive verb (required by robot-start-j1/j2)
$BENCH/bin/robot-start-j2 --task stack_blocks --layout 9 --seed 0 --dev --reserve 20 --id $ID --wait-gpu 900
# perception service (local route)
PD=~/dojo-perception-data; export DOJO_CATALOG=$PD/catalog DOJO_X5_DIR=$PD/catalog/robot/x5 \
  DOJO_BG_FOLDS=$PD/bg_folds DOJO_COLOR_CALIB=$PD/catalog/color_calib.json \
  DOJO_PERCEPTION_PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])') \
  DOJO_PERCEPTION_AUTHKEY=$(python3 -c 'import secrets;print(secrets.token_hex(16))') DOJO_PERCEPTION_READY=$OUT/$ID/perception.ready
DOJO_ILIA=1 PYOPENGL_PLATFORM=egl nice .venv/bin/python -u -m dojo.runtime.perception_service > $OUT/$ID/perception_service.log 2>&1 &
PERC=$!; until [ -f $DOJO_PERCEPTION_READY ]; do sleep 2; done
# the driver (spawns the twin for the instruction's scene; closes the server at the end)
DOJO_ILIA=1 DOJO_AGENT_DOTENV=$HOME/controlr/.env DOJO_AGENT_LOG=$OUT/$ID/agent .venv/bin/python -u -m dojo.bridge.driver --server-id $ID --bench $BENCH \
    --policy dojo.agent.policy:AgentPolicy --perception local --verify bundle --twin spawn --hold-gpu-lock \
    --config '{"refine":true,"search":true,"post_place":true,"max_rehovers":2,"refine_max_shift_m":0.07}' --out $OUT --close
until [ -f $BENCH/run/$ID/launch_done.json ]; do sleep 2; done; kill $PERC
```
Variants (one flag each):
- `--policy dojo.runtime.policy:RuntimePolicy`: the rule supervisor instead of the agent, 0 LLM calls.
- `--perception server`: estimates from the server's `perceive` verb, i.e. our B1 adapter inside Mikhail's server,
  logged in `run/<id>/perceive.jsonl`. Every other op (refine, check, search, scan, agentic, held_look, watch, ...)
  then comes from the local service if one runs. Without it, refine / search are off and the resync check is logged
  `unverified`; the driver refuses a config that enables a local-only feature (`perception.LOCAL_ONLY`). The adapter (`j1_adapter.task_for`) routes by the six J1 patterns, then by the runtime's
  `perception_vocabulary`, and uses the scene backgrounds under `DOJO_BG/scene/` (the conveyor belt).
- `--verify off`: no twin.

### 4c. The official path on OUR dev layouts, on any 5.1-rendering box of ours [GPU]

`robot-start-j2` runs RoboDojo's client on `Assets/Eval_Layout` (and needs it on disk, which our boxes do not have).
`scripts/dojo/official_eval.sh` runs the same chain on our settled dev layouts (>= 100000), several episodes per eval
client, so a dev gain can be checked on the path the leaderboard scores:

```bash
source ~/dojo-env.sh && cd ~/dojo-<stream>
# once: robot-harness at a pinned commit, read-only (a git archive + a COMMIT file; never a working clone of his repo)
mkdir -p ~/dojo-official-rh && tar -xzf rh-<sha>.tar.gz -C ~/dojo-official-rh && echo <sha> > ~/dojo-official-rh/COMMIT
scripts/dojo/official_eval.sh --task press_by_number --layouts "100000 100001 100002" \
    [--policy dojo.agent.policy:AgentPolicy] [--preset submit_v1] [--config JSON]
python -m dojo.bridge.official compare --official ~/dojo-data/official/runs/<id> --integ <integ_eval out dir> --md pair.md
```

**A sealed joint block (D-C19).** This is the only way ids >= 100100 run; every other call refuses them. The format and
rules were agreed with Mikhail's side (bus 30 Oct 0008Z / 0035Z), and each side runs its own code to the same rules.

```bash
COORD_DIR=~/coord scripts/dojo/official_eval.sh --candidate CANDIDATE.txt --task T --layouts "<tranche ids>" \
    --sealed-block DIR --manifest MANIFEST --lock ~/coord/dc19.lock
python -m dojo.simlab.sealed verify --dir DIR --manifest MANIFEST     # before the first episode and after the last
```

- **Manifest, schema `dc19-sealed/1` (`dojo.simlab.sealed`):**
  - generator provenance, `seed_salt_sha256`, `created_utc`;
  - per task: `settled`, `records {id: md5}`, `tranche_a` / `tranche_b`, `unstable_count`, and `shortfall` for a short
    cell.
  - Ids lie in 100100-100119. Tranches are ascending and disjoint, A is below B, and together they equal the records:
    5 each unless the shortfall says otherwise.
  - Tranche B runs only once `tranche_b.preregistered` exists beside the manifest.
  - Position k of a tranche is its k-th id.
- **Staging:** only runnable ids are staged. Each record is md5-checked before staging into the run's layout dir
  (read-only, nowhere else), and checked again in RoboDojo's process (`SIMLAB_SEALED_MANIFEST`). Raw records are used
  where `settled` is false (make_toast, the conveyors).
- **Run once:** before anything starts, an O_EXCL marker `$COORD_DIR/scored/<block>/<task>-<id>.json` is created. A
  second claim refuses. The one disclosed infrastructure re-run (before any verdict was read) needs
  `DC19_INFRA_RERUN=<reason>`.
- **No dev path:** robot-harness's server runs without `--dev`, and HARNESS_DEV / HARNESS_PRIVILEGED are refused.
  `--lock` has no default.
- **`verify`** prints `task id md5 pass/fail`, then its own md5. It fails on an unlisted file or task folder and on an
  empty record. It never prints content.
- **Generation:** `scripts/dojo/dc19_generate.py`. It runs once, only on the lead's `--go`, after both freezes:
  - draws the salt from /dev/urandom into `DIR/.salt` (0400, never printed);
  - generates salted records (fill_egg_holder on its v6 tree), then presettles them (pass 1, then keep-raw-unmoved);
  - selects the first 10 stable ids and writes DIR plus the manifest;
  - prints only sha256(manifest) and per-cell n_stable / n_unstable.
- **Pre-flight before tranche A:** `scripts/dojo/route_probe.py` makes one tiny call on the agentic route.
  `freeze_manifest.py --route-probe` records it.

**The candidate on this path (D-C19).** `--candidate ~/dojo-integ-c4-state/CANDIDATE.txt` takes Integration's file
verbatim:
- preset, `--config`, and `--setenv` (for our processes only);
- the `--seg` / `--glyph` / `--glyph-kong` side servers, started as lane2.sh starts them. One that does not come up
  ends the call before robot-harness's server starts;
- RuntimePolicy.

Before a scored run:

```bash
scripts/dojo/official_eval.sh --candidate CANDIDATE.txt --dry-run             # the resolved call, nothing started
python scripts/dojo/candidate_parity.py --candidate CANDIDATE.txt             # integ_eval vs official: zero diff = go
python scripts/dojo/freeze_manifest.py --candidate CANDIDATE.txt --rd-python <RoboDojo python> --out manifest.json
```

The manifest records:
- the sha, the candidate text, and the resolved launcher line with its full RuntimeConfig;
- the md5 / sha256 of every side-server weight;
- the parity diff;
- model routes by name;
- python, Isaac, RoboDojo and robot-harness versions;
- that the strict v2 server supplies `ee_joint_state_official`.

Run it from a clean checkout at the frozen sha. The parity check also compares the perception data. Integration's
lane2.sh runs `DOJO_BG_FOLDS=~/dojo-data/bg_folds_51_ilia` (the 5.1 set with the scene backgrounds), so on compute4 export
the same before official_eval.sh, or parity fails.

| piece | whose | how |
|---|---|---|
| eval client | RoboDojo's `main.py`, unchanged | `scripts/simlab/dev_eval.sh` (`dojo/simlab/dev_eval.py`: SeedManager reads `OUT/layouts/<task>/<id>.settled.json`, ids ascending) |
| strict J2 server + shim `harness_strict_j2` | Mikhail's files, unmodified (`$BENCH`, default `~/dojo-official-rh`) | the J2 launcher's md5 checks, its env (`HARNESS_STRICT=1`, `HARNESS_DEV=1` for `dev.jsonl` only); the shim is copied into OUR RoboDojo farm |
| RoboDojo install | the box's `~/dojo-robodojo`, read-only | a symlink farm `~/dojo-official-rdhome` (`scripts/dojo/rd_farm.sh`): `XPolicyLab/policy/` and `eval_result/` are ours |
| bridge driver, perception service, twin | ours | one driver per episode; the next driver's `reset` ends the previous episode on the server (the shim books an unfinished one as failed), the last one `--close`s; the perception calibration is the service's one frozen set (`DOJO_CALIB_SET`, `scripts/dojo/calib_set.sh`), the same for every episode |

- Everything runs under the heavy lock and in one systemd scope (24G / 1400 %). Keep a call <= 30 min.
- Outputs, under `~/dojo-data/official/runs/<id>/`:
  - `_result.json`: RoboDojo's own. `details[i].layout_id` is the index of the dev layout in id order.
  - `rows.jsonl`: per episode, the verdict joined with our side (`python -m dojo.bridge.official rows`).
  - `eps/<layout>/<id>/`: the driver's `bridge.json`, `report.json.gz`, `actions.json.gz`, frames, agent.
  - `server_run/`: the server's run dir (`episode.jsonl`, `dev.jsonl`), `eval/`: the eval client's logs and mp4s.
- `compare` pairs it with an integ_eval run on the same layouts: verdicts, steps, and the first executed action where the
  two paths differ (tol 1e-4: the server rounds measured joints / flange to 1e-5). Physics is bitwise deterministic
  (journal 2026-10-26-bridge-isr A2), so an outcome difference with identical actions is a scoring difference, and one
  after a divergence is a decision difference with its cause at that step.

## 5. What is strict, and where it is checked [CPU]

| check | where | test |
|---|---|---|
| server tier must be `B-strict`, `strict.finger_measurement: false` | `contract.check_tier` at `connect()` | `test_non_strict_or_verbless_servers_are_refused` |
| every reply our code reads is scanned for banned keys: the strict server's BANNED list (objects, calib, finger_q, gap_m, grasp_detected, fingers_stopped, blocked, grasps) plus dev / grader keys (success, reward, score, layout_id, seed, objects_dev...); a hit raises, nothing is stripped and carried on | `contract.check_reply` in `backend._call` | `test_a_leaking_server_is_refused` (objects, finger_q, calib), `test_scan_finds_banned_keys_at_any_depth` |
| the observation holds exactly: 3 RGB frames, measured joints, measured flange, COMMANDED gripper, instruction, `steps_used`, `step_lim`; TCP, `grip_obs`, home distances, `held_look`, paths stay in the backend | `contract.contract_obs` | `test_observation_carries_only_the_contract` |
| the runtime's own firewall checks every observation again (mode perception: any extra key raises) | `dojo.runtime.firewall.check_obs` in `RuntimePolicy.next` | same test + the episode tests |
| camera calibration never comes from a reply: our perception computes K / T from the public config + measured joints, the same inputs as the server's `strict_calib.py` | `perception_service` (`cameras_from_state`) | the runtime's tests |
| `info` = instruction + step limit only, no task name; the twin scene follows the instruction | `driver.run_episode`, `twin.twin_scene` | `test_runtime_episode_through_the_server`, `test_twin_scene_follows_the_instruction` |
| the runtime moves only through `joint_steps`; GT is never read (patched to raise) | `backend._run` | `test_runtime_episode_through_the_server` |
| no GT accessor, eval file or grader verb in the bridge sources | static check | `test_bridge_sources_never_read_gt_or_grader_state` |

`ended` (RoboDojo's natural end: success latch or step limit) is read by the driver only, to stop calling the
policy, exactly as an eval harness stops calling `next`. It is visible by design (robot-harness AGENTS.md rule 4);
disclose it.

## 6. Outputs, and the rules for a scored run

`$OUT/<id>/` (ours, verdict-free):
- `bridge.json`: end reason, steps, wall / policy / exec / observe seconds, twin boot, server call count, backend stats,
  and `env`: the run's data-set / perceiver settings (`driver.ENV_NOTES`: names and paths). `AGENTIC_EXTRA_BODY`
  is recorded only as `sha256:<16 hex>`, since a request body can carry a credential.
- `report.json.gz`: the policy's episode report (perception records, events, verify stats, agent usage).
- `actions.json.gz`: every executed action.
- `calls.jsonl`: one line per server call.
- `frames/`: every 20th observation's PNGs; `--keep-frames k` changes the rate.
- `agent/`: transcript, calls, cells.
- logs: `perception_service.log`, `twin_service_<scene>.log`, `bridge_episode.log`.

`$BENCH/run/<id>/` (Mikhail's): `episode.jsonl`, `perceive.jsonl`, `robodojo_result/_result.json` (THE score),
logs.

For a scored block, follow robot-harness AGENTS.md:
- Freeze our checkout and data dirs with md5 lists, as J1's `freeze_j1.sh` does for the adapter.
- Run once and blind: never read `_result.json`, `robodojo_result/`, `dev.jsonl` or the server logs until the
  block's last close.
- Ledger lines go to `results/ledger.jsonl`, with tier `official` or `B_camera`.
- Live LLM calls count toward the daily cap (CLAUDE.md rule 3). Each episode's usage is in
  `report.json.gz` → `agent.usage`.

## 7. Timing and resources (measured on isr-lab: RTX 4090, Isaac 5.1 RTX; journal 2026-10-26-bridge-isr)

| item | value |
|---|---|
| eval client boot (`robot-start-j2` to `episode_pending`) | < 1 min with a warm shader cache. The first RTX boot fills ~1.6 GB of Kit caches under the shim HOME |
| observation | `obs` writes 3 PNGs into our `frames/` (`save_dir`) and we decode them: ~0.07 s per observation. A stack episode has ~15 observations |
| twin boot | 4-5 s after `reset` (physics-only, 4 envs) |
| perception / verify per episode | stack_blocks: perceive ~3 s, verify ~31 s for 6 segments / 48 rollouts (4 envs); agentic frame-0 estimate ~39 s (3 Sonnet calls) |
| VRAM | 10.1-12.4 GB peak: the eval client (it runs RoboDojo's 10-env scene for one layout), the twin and perception EGL |
| CPU | peak <= 3 runnable processes of ours, load average <= 3.5 |
| disk | ~4 MB per run dir with `--keep-frames 20` (+ ~2 MB of agentic frames). Prune with `scripts/dojo/isr_prune.py` |
| robot-start busy check | refuses when < 6 GB VRAM is free or another eval client runs, so start the official run BEFORE our services |

## 8. Known gaps

- **Without `joint_steps` on the server (#8)**, `connect()` refuses with a pointer to #8. There is no
  fallback through `move_to`: it re-plans with cuRobo at <= 0.08 rad/step, so the executed paths would differ from
  the twin-verified ones and cost about 2x the steps.
- **Gripper semantics.** `joint_steps` sends our gripper command per step, verbatim, without the shim's 0.25-per-step
  ramp, as in the proxy runs (RoboDojo's own `take_action`).
  - The skills' 0-step grip ops put a new gripper command on the FIRST step of a free-space move: `open_for_grasp`
    before a pick approach, `grip_for_home`, `close_for_push` / `close_for_press`. This happens from rest or at a via
    point between two moves.
  - The server counts these per arm (`grip_change_while_moving`, the shim's invariant 3) and does not refuse them.
    `bridge.json` → `backend.grip_change_while_moving` holds the episode's total. A stack episode has >= 3, which is
    expected; a refusal rule would need >= 1 grip step per such op in the skills.
- **The server's `perceive` route has no refine / check / search** unless the local service also runs. Its routing
  (`j1_adapter.task_for`) falls back to the runtime's `perception_vocabulary`, and its backgrounds follow the scene
  (`backgrounds_for`). The scene background dirs must be shipped with its `DOJO_BG`.
- **`held_look`**: the server's verb (B1 inside his server) is available as `backend.held_look`, but the runtime does
  not call it. The runtime's own held check (`RuntimeConfig.held_look` = `b1` | `b1_then_agent`) is a perception
  SERVICE op. Like `agentic`, it exists only with `--perception local`.
- **Instructions without a runtime perception vocabulary** (`dojo.runtime.supervisor.VOCABULARIES`) cannot run in
  mode perception, so they cannot run on this path at all: make_toast, make_toast_random, store_laptop_and_headphones,
  hang_mugs.
- **Cross-interpreter pickles.** The driver and perception run in our venv (numpy 2.x); the twin service runs in
  RoboDojo's python (numpy 1.26). `TwinClient` sends plain lists and numbers (`twin_service.plain`), because a numpy-2
  array does not unpickle under numpy 1.26 (`No module named 'numpy._core'`). Replies the other way are fine.

## 9. The `joint_steps` contract (robot-harness #8 / HARNESS-CONTRACT 1.1; mirrored by `dojo/bridge/fake_server.py`)

```
joint_steps actions=[{"left_arm_joint_state":[6],"left_ee_joint_state":[1],"right_arm_joint_state":[6],"right_ee_joint_state":[1]}, ...]
            [reserve=<int, default --reserve>] [max_dq=0.2]
```
1. The whole list is validated before any step. A failure replies `{steps: 0, refused: "<code>", hint, index}`, with
   codes `keys`, `shape`, `non_finite`, `gripper_range`, `jump`, `budget`, `too_long`, `empty`. Checks:
   - all four keys per entry, with 6 / 1 finite numbers;
   - gripper in [0, 1];
   - per-arm joint jump between consecutive commands (the first entry vs the current hold) <= `max_dq`, capped at
     0.2 rad (the X5 actuator, 5 rad/s x 40 ms);
   - length <= steps_left - reserve (`budget`), and <= 2000.
2. Then one `take_action` per entry, verbatim: no IK, no re-timing, no gripper ramp. `get_obs` runs once per step,
   `is_episode_end` after each step, and the run stops when it is true. The hold stays at the last entry.
3. Reply: `{steps, executed, requested, refused: null, tracking_rad, tracking_max_rad, tracking_max_at,
   start_tracking_rad, grip_change_while_moving, max_step_dq, max_dq, reserve, actions_md5, eef: {arm: {q, flange,
   grip_cmd}}}` plus `step`, `steps_left`, `ended`.

Every chunk's `actions_md5` (md5 of `json.dumps(actions, sort_keys=True, separators=(",", ":"))`) is checked
against the reply and kept in `bridge.json` → `joint_steps_chunks`, to match against the server's `episode.jsonl`.
After a robot-harness skill moves an arm through `backend.robot`, our Skills must be rebuilt from proprio. Otherwise the
executor's first entry jumps from a stale commanded pose and is refused (`jump`).

We always pass `reserve=0`. The runtime keeps its own fixed reserve for `finish()` (`RuntimeConfig.reserve` = 30, the
same for every episode). The backend sends at most 500 entries per call, and actions past the step limit are never
sent: the episode ends there, as in any eval harness.
