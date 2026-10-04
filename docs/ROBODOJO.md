# RoboDojo: controlr as RoboDojo's controller

[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) (HKU MMLab) is a sim benchmark of 42
manipulation tasks on two ARX X5 arms in Isaac Sim 5.1 (background:
[research/sources/robodojo.md](../research/sources/robodojo.md)). controlr plugs into it as an
XPolicyLab policy, so **RoboDojo's own eval client owns every episode**: it picks the layout,
counts env steps against the task's `step_lim`, latches success with the task's reward, writes
`_result.json` and the mp4. controlr only decides the motions. The model's task input is
RoboDojo's instruction string; nothing task-specific is added (no recipes, no score ladders).

## How it fits together

```
RoboDojo eval client (conda env, Isaac Sim 5.1)          controlr robodojo-serve (our venv)
  per episode: reset(layout) -> eval_one_episode() ------>  accept: hello {task, layout, instruction, step_lim}
  XPolicyLab/policy/controlr/deploy.py (the shim)            run_episode(cfg, RoboDojoRobot(conn))  -> runs/<...>_<task>_L<layout>/
     serves observe / execute / check_goal  <------------    requests over multiprocessing.connection (localhost, HMAC)
     execute: cuRobo plan per arm -> joint waypoints         done{outcome} -> shim returns -> RoboDojo scores, saves mp4
     -> take_action() per env step (both arms, both grippers)
XPolicyLab ws policy server (no-op model: RoboDojo's handshake only)
```

- **Shim** (`controlr/robot/robodojo/shim/deploy.py`, copied into `XPolicyLab/policy/controlr/` by
  `scripts/robodojo/run.sh`): stdlib + numpy; follows the motion rules of RoboDojo's own LLM
  adapter (`GPT_6_Astra_Direct_EEF`): a target is a flange pose planned by RoboDojo's cuRobo, the
  joint path is resampled so no joint moves more than 0.05 rad per env step, grippers move after
  the arm arrives (≤ 0.25 of their range per step), every step commands both arms and both
  grippers (an arm not commanded holds its last target), `get_obs` after every step (that is
  what writes RoboDojo's mp4). An episode controlr ends early (FAIL, max_turns, error) is marked
  failed in RoboDojo, as RoboDojo's own adapters do.
- **Client** (`RoboDojoRobot`): flange (link6) ↔ TCP = grasp point 150.1 mm along the flange +x;
  tool z = that axis; grippers 0..1 ↔ width (`robot.params.gripper_max_mm`, nominal 80); arms
  `L`/`R` ↔ `left`/`right`. Feedback adds `STEPS: <n> of <lim> left`. An unreachable target
  (planner failure) is a WARN for that arm; an arm that ends > 10 mm / 5 deg from its target too.
- **Envelope**: `RobotSpec.kinematics = "backend"` → `CartesianEnvelope` (step limits, workspace
  box, table clearance; no IK — cuRobo plans). It knows neither the objects nor the other arm.
- **Episode end**: RoboDojo's success latch or step limit → `Robot.episode_over()` → outcome
  `env_end`; `summary.success` is RoboDojo's verdict. The number to report is RoboDojo's
  `_result.json` (`success_rate`, `score`), and `summary.success` must agree with it per episode.

## Install (once, compute2)

```bash
scp scripts/robodojo/install.sh compute2:/root/controlr-robodojo/install.sh
ssh compute2 'cd /root/controlr-robodojo && setsid nohup bash install.sh > install.log 2>&1 < /dev/null &'
```

Everything lands in `/root/controlr-robodojo` (own Miniconda + env `miniconda3/envs/robodojo`,
RoboDojo clone with its IsaacLab / cuRobo / XPolicyLab submodules, pip cache, the eval assets
— every `Assets/` folder (Robots, Object, Material, Room, Background, Sensor, Traj) plus the
seed-0 layouts, ~41 GB, as a sparse git-lfs clone in
`hf_git/`: anonymous `hf download` is rate limited by Hugging Face after ~650 files, the LFS batch
API is not). Nothing touches `~/.bashrc` or a shared conda.
Steps are resumable: `bash install.sh isaaclab curobo assets check`.

## Run

```bash
CONTROLR_HOST=compute2 scripts/deploy.sh /root/controlr          # code + venv
ssh compute2 'cd /root/controlr && scripts/robodojo/run.sh --task general_pickup --seed 0 --eval-num 2 --fake-llm'
ssh compute2 'cd /root/controlr && scripts/robodojo/run.sh --task general_pickup --seed 0 --eval-num 5'
rsync -az compute2:/root/controlr/runs/ runs/                     # run dirs + runs/robodojo_eval/<stamp>/
```

`run.sh` starts `controlr robodojo-serve`, the no-op ws policy server and RoboDojo's eval client,
and stops all three on exit. Per evaluation: `runs/robodojo_eval/<stamp>_<task>_s<seed>/`
(serve / policy server / eval client logs, `robodojo_result/` with `_result.json` + mp4s);
per episode: a normal controlr run dir.

## Pitfalls

- `general_pickup` has 200 env steps (8 s of arm motion): long moves eat the budget.
- RoboDojo's EE-pose action mode silently drops an arm command when IK fails — that is why the
  shim plans itself and sends joint targets.
- Camera extrinsics come from RoboDojo's `get_camera_extrinsics` (camera-to-world, converted from
  the USD to the OpenCV camera axes in the shim); overlays that project points rely on it.
- RoboDojo's `init_assets.sh` checks only Robots / Object / Material / Eval_Layout, but scenes
  also load `Assets/Room`; and `Assets/Robots/**/curobo.yml` must be generated from the
  `*_tmp.yml` templates (`utils/update_embodiment_config_path.py`). `install.sh` does both.
- XPolicyLab's ws policy server needs `websockets>=14`, RoboDojo's env pins 12.0: the no-op
  policy server runs in `policy_venv`.
- Eval layouts are consumed in order: `--seed S --eval-num N` = layouts 0..N-1 of seed S.
