# dojo: the RoboDojo v2 stack

Design: [../docs/design/robodojo-v2.md](../docs/design/robodojo-v2.md). Evidence: [../research/robodojo-teardown/](../research/robodojo-teardown/README.md).

| Module | Owner stream | What |
|---|---|---|
| `dojo/simlab/` | simlab | Dev-layout generator, headless RoboDojo runner, GT dumps (dev only), `Twin` API, determinism tests |
| `dojo/perception/` | perception | Asset catalogue (meshes, stable poses), inverse layout: RGB + proprio → layout record, accuracy eval |
| `dojo/skills/` | skills, skills2 | WorldState, own-IK motion layer, grasps (top / side / rim / wall / box / slice), skills incl. re-orienting transfers (`plan_transfer`), relay chains and in-air handovers (`library.place_oriented`), topple, mesh peg-in-hole fit (`insertion.py`), a generic skill library behind one planner entry point (`planner.solve`: instruction + WorldState -> skill), plan + verify (`TwinBackend`), G3 harness (`g3.py`, `scripts/skills/g3*.sh`), kinematic dry runs (`scripts/skills/dryrun.py`) |
| `dojo/runtime/` | integration | The one proxy_eval policy (`RuntimePolicy`): perceive (+ look macro) → WorldState → supervisor (instruction-only) → skills with twin verify (bundle) → resync → finish; firewall; eval driver `scripts/dojo/integ_eval.sh` / `launch.sh` (modes gt / perception) |
| `baselines/pyrua/` | pyrua | PyRUA-Lean reproduction on our infra (the floor) |

Hard rule (the only one): at eval time the policy uses only the official observation contract. Never read
`Assets/Eval_Layout/**` or eval-env state for development. Dev work uses our own generated layouts, where ground
truth is ours.

Shared machine conventions:
- compute2 (4090 24 GB): RoboDojo install at `/root/controlr-robodojo` (read-only, shared); data at
  `/root/dojo-data/<stream>/`; code deploys at `/root/controlr-<stream>`.
- compute3 (5090 32 GB, shared with others): `~/dojo-*`.
- Before starting Isaac or a large model, check free VRAM with `nvidia-smi`; never touch other people's processes;
  stop what you started.

## simlab quick reference

Run everything through `scripts/simlab/rdpy.sh` (RoboDojo's conda python + the team GPU lock). Use
`physics_only=True` / `--physics-only` on compute3, where Isaac Sim 5.1's RTX does not start.

| What | Entry point |
|---|---|
| layout schema, perturbation | `dojo/simlab/layout.py` |
| new dev layouts (seeds >= 100000) | `-m dojo.simlab.gen_layouts --task T --n 25` |
| pre-settled records `<id>.settled.json` (prefer these) | `scripts/simlab/presettle_all.sh [task ...]` (`-m dojo.simlab.presettle --task T`) |
| load layouts, official obs vs dev-only GT | `dojo.simlab.runner.RoboDojoRunner` |
| twin: load / snapshot / restore / step / verify / load_from_estimate | `dojo.simlab.twin.Twin` |
| scripted motion, replayable action lists | `dojo/simlab/motion.py` |
| D0 dataset | `scripts/simlab/d0_all.sh` (compute2), `scripts/simlab/d0_proxy.sh` (compute3: Isaac 6.0 renders) |
| G2 determinism / sensitivity | `scripts/simlab/g2_all.sh OUT [--physics-only]` |
| twin throughput | `scripts/simlab/twin_bench.sh` |
| RoboDojo's eval client on dev layouts (official, needs 5.1 rendering) | `scripts/simlab/dev_eval.sh --task T --policy P` |
| PROXY dev-eval on compute3 (5.1 physics + 6.0 render on demand) | `scripts/simlab/proxy_eval.sh --task T --layouts ... --policy module:Class [--bundle]` |
| twin as a service for a policy (inside `--bundle`) | `dojo.simlab.twin_service.TwinClient` |
| minimal RoboDojo on another GPU | `scripts/simlab/install_min.sh` |

### Proxy dev-eval with your own policy (compute3)

```python
# mypkg/mypolicy.py, importable from the repo root (scripts add it to PYTHONPATH)
class MyPolicy:
    def reset(self, info):        # {"task", "layout_id", "instruction", "step_lim", ...}
        ...
    def next(self, obs):          # obs: images {cam_head, cam_left_wrist, cam_right_wrist: 480x640x3 uint8},
        ...                       #      state (official joints / flange poses / grippers), instruction,
                                  #      steps_used, step_lim
        return {"actions": [{"left_arm_joint_state": [6], "left_ee_joint_state": [g],
                             "right_arm_joint_state": [6], "right_ee_joint_state": [g]}, ...]}
        # or {"done": True}; one observation is rendered per next() call (~0.12 s)
```

```bash
ssh compute3 'cd ~/dojo-simlab && scripts/simlab/proxy_eval.sh --task stack_blocks \
    --layouts $(ls ~/dojo-data/simlab/layouts/stack_blocks/1000*.json | head -10) --policy mypkg.mypolicy:MyPolicy'
# -> ~/dojo-data/simlab/proxy_eval/<stamp>/_result.json (RoboDojo-style, marked "proxy")
```

Every episode goes through RoboDojo's own reset, `take_action` step accounting, `step_lim` and reward
latches; only the frames come from Isaac 6.0 (RoboDojo's stage, cameras and render settings). Episodes
are serial.

`--bundle [--twin-envs N]` adds a twin service (physics-only `Twin`, N <= 16, default 8) for the same task;
the whole bundle (episode env + render service + twin) is ONE GPU job: one lock, one scope (24 GB / 14 cores).
Inside the policy:

```python
from dojo.simlab.twin_service import TwinClient
tc = TwinClient()                                  # env vars set by the bundle
tc.load_from_estimate(layout_estimate, robot={"left_q": q_l, "right_q": q_r, "left_grip": 1.0, "right_grip": 1.0})
outs = tc.verify({"layout": layout_estimate, "robot": {...}}, [{"actions": seg, "meta": {...}}, ...],
                 perturb={"pos_mm": 2.0, "yaw_deg": 1.0}, k=4)    # outcome fields: see Twin.verify
```

`dojo.simlab.proxy_eval:VerifyReplayPolicy` (spec `verifyreplay:TRAJ.json[:k]`) is a dev demo of the
bundle. `info["layout_file"]` exists only in dev runs; a real policy passes its perception estimate.
Layout lists: prefer `<id>.settled.json` (objects at the poses RoboDojo's settle gives them).

Findings and numbers: [docs/journal/2026-10-24-dojo-simlab.md](../docs/journal/2026-10-24-dojo-simlab.md).
