# configs

One YAML = one experiment. Resolution order: defaults in `controlr/config.py` ← `extends:` ←
the file ← `--set key=value`. An unknown key is an error (a typo must not silently fall back to
a default). All axes — [../docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md) §2.

| File | What |
|---|---|
| `base.yaml` | shared defaults: model, planner, observation, actions, safety, episode |
| `sim_waffle.yaml` | Isaac: waffle packet → box, fixed tool orientation |
| `sim_waffle_yaw.yaml` | the same + `rotation=yaw`, packet yaw spread ±40°, start yaw ±20° |
| `sim_waffle_yaw_luna_dec.yaml` | the same task with GPT-6 Luna Decisions as a decision head (`llm.backend=decisions`): ~0.3 s turns, small steps, 150 turns |
| `sim_reach.yaml` | Isaac: move the TCP to a marker |
| `mock.yaml` | kinematic mock robot without physics (for `--fake-llm` and cheap checks) |
| `bench/latency.yaml` | latency matrix: models × frame size × history length |
| `sweeps/*.yaml` | sweeps: `config` + `set` + `grid` × `seeds` → `controlr sweep` |

A new experiment = a new file via `extends:`, not an edit of an existing one: old reports
reference these configs.
