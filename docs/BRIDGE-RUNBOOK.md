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
The agent on the fake server (live calls: about 1 per episode for the stack task; cx/gpt-6.1-sol by default):
```bash
.venv/bin/python -m dojo.bridge.fake_server --id fk-2 --bench /tmp/rbfake &
DOJO_AGENT_LOG=/tmp/rbfake/agent .venv/bin/python -m dojo.bridge.driver --server-id fk-2 --bench /tmp/rbfake \
    --policy dojo.agent.policy:AgentPolicy --perception server --twin off --config '{"extra":{"agent":{"max_calls":4}}}' \
    --out /tmp/rbfake/out --close
```

## 4. A strict full-system episode through the official client [GPU] [needs #8]

Practice only: **seed 0 layout 9** (robot-harness AGENTS.md rule 1). Scored seeds and layouts are run once, blind,
md5-frozen, on the lead's go (§6).

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
DOJO_ILIA=1 DOJO_AGENT_LOG=$OUT/$ID/agent .venv/bin/python -u -m dojo.bridge.driver --server-id $ID --bench $BENCH \
    --policy dojo.agent.policy:AgentPolicy --perception local --verify bundle --twin spawn --hold-gpu-lock \
    --config '{"refine":true,"search":true,"post_place":true,"max_rehovers":2,"refine_max_shift_m":0.07}' --out $OUT --close
until [ -f $BENCH/run/$ID/launch_done.json ]; do sleep 2; done; kill $PERC
```
Variants (one flag each):
- `--policy dojo.runtime.policy:RuntimePolicy`: the rule supervisor instead of the agent, 0 LLM calls.
- `--perception server`: estimates from the server's `perceive` verb, i.e. our B1 adapter inside Mikhail's server,
  logged in `run/<id>/perceive.jsonl`. Refine / check / search then come from the local service if one runs, else they
  are off. The adapter's own instruction routing covers only stack_blocks, push_T, press_by_number, swap_T,
  pour_balls_into_vase and general_pickup.
- `--verify off`: no twin.

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
- `bridge.json`: end reason, steps, wall / policy / exec / observe seconds, twin boot, server call count, backend stats.
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

## 7. Timing and resources (expected; measure on the first practice episode)

| item | value |
|---|---|
| observation | `obs` writes 3 PNGs and we decode them: ~0.1-0.2 s per observation (fake server: 4 ms at 48x64) |
| twin boot | 1-2 min after `reset`; the sim is paused meanwhile (shim idle timeout 3600 s) |
| perception / verify per episode | as in the proxy (journal 2026-10-24-dojo-integ: 34-54 s per stack episode) |
| VRAM | eval client with RTX + twin (8 envs) + perception EGL on one GPU. The bundle peaked at 11.7 GB without RTX; check `nvidia-smi` before the first run |
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
- **The server's `perceive` route has no refine / check / search** unless the local service also runs. Its task routing
  (`j1_adapter.infer_task`) covers 6 tasks, while the runtime's vocabularies cover more. Unifying them is BACKLOG.
- **`held_look`** (B1) is available as `backend.held_look`, but the runtime does not call it: its slip checks read
  predictions; see the integ journal's open items.

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
