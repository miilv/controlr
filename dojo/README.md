# dojo: the RoboDojo v2 stack

Design: [../docs/design/robodojo-v2.md](../docs/design/robodojo-v2.md). Evidence: [../research/robodojo-teardown/](../research/robodojo-teardown/README.md).

| Module | Owner stream | What |
|---|---|---|
| `dojo/simlab/` | simlab | Dev-layout generator, headless RoboDojo runner, GT dumps (dev only), `Twin` API, determinism tests |
| `dojo/perception/` | perception | Asset catalogue (meshes, stable poses), inverse layout: RGB + proprio → layout record, accuracy eval |
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
| load layouts, official obs vs dev-only GT | `dojo.simlab.runner.RoboDojoRunner` |
| twin: load / snapshot / restore / step / verify / load_from_estimate | `dojo.simlab.twin.Twin` |
| scripted motion, replayable action lists | `dojo/simlab/motion.py` |
| D0 dataset | `scripts/simlab/d0_all.sh` (compute2), `scripts/simlab/d0_proxy.sh` (compute3: Isaac 6.0 renders) |
| G2 determinism / sensitivity | `scripts/simlab/g2_all.sh OUT [--physics-only]` |
| twin throughput | `scripts/simlab/twin_bench.sh` |
| RoboDojo's eval client on dev layouts | `scripts/simlab/dev_eval.sh --task T --policy P` |
| minimal RoboDojo on another GPU | `scripts/simlab/install_min.sh` |

Findings and numbers: [docs/journal/2026-10-24-dojo-simlab.md](../docs/journal/2026-10-24-dojo-simlab.md).
