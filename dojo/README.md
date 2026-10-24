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
