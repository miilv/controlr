# RSI installation and lanes

These are recipes, not authorization to start a box. Confirm the current
permission and that its camera lanes have drained before installing/running.
Do not bootstrap, change drivers, install system packages, or read sealed data.
Use a merged controlr deployment for `GLUE`. Keep the existing measured
layout-generator deployment for `CODE` (the initial preparation used controlr
`7e16003ebc46f59567f5e50fd3536c9f65dfb324`); do not mix generators on resume.
All paths below expand on the destination box, not on this agent machine.

## Install on compute4 or compute6, after authorization

```bash
ROOT="$HOME/controlr-rsi-repro"
MODELS="$HOME/dojo-data/models"
STATE="$HOME/dojo-data/rsi-repro"
# GLUE = merged checkout's baselines/rsi; CODE = pinned generator/runtime checkout
bash "$GLUE/setup.sh" --root "$ROOT" --models "$MODELS" --uv "$HOME/.local/bin/uv"
bash "$CODE/scripts/dojo/rd_farm.sh" "$HOME/dojo-robodojo" "$ROOT/rdhome"
ln -s "$ROOT/PyRUA-Lean/integrations/xpolicylab/PyRUA_RSI" \
  "$ROOT/rdhome/RoboDojo/XPolicyLab/policy/PyRUA_RSI"
```

Venv: `ROOT/venv` (managed CPython 3.11; torch 2.8.0+cu128, torchvision 0.23.0,
numpy<2). Runtime: `ROOT/runtime` at c686120; program bundle:
`ROOT/PyRUA-Lean` at e754cf43. SAM3 package: RLinf/sam3 c1f6b1e.
Checkpoint: `~/dojo-data/models/sam3.pt`, 3,450,062,241 bytes, SHA-256
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
If a verified shared checkpoint already exists, setup reuses it. Download via
upstream `download_assets.sh modelscope` only if absent. No API keys or dotenv.
This recipe is not run on compute7: its existing venv/checkpoint stay in place.

Before later rehosting, transfer only our own frozen layouts, manifests,
demos and preparation checkpoints from the permitted RSI state directory.
Verify every SHA-256 and the source hashes. Never read dc19, coord/scored,
rh-misha or dojo-official-rh paths. No new draws replace already frozen cells.

## Two camera workers

Use the same `STATE`, `LANE_STATE`, and `wave-start.lock` as the existing camera
profile. Do not change caps, queues, boot stamps, STOP files or lane ownership
to force admission. Locks must already exist as regular non-symlink files.
Use exactly the two heavy locks; prep and policy cannot borrow light slots.

| Box | stream | lane state | heavy1 lock | heavy2 lock | boot stamp |
|---|---|---|---|---|---|
| compute4 | integ-c4 | `~/dojo-integ-c4-state` | `~/dojo-integ-c4-heavy1.lock` | `~/dojo-integ-c4-heavy2.lock` | `~/coord/lane_gate.stamp` |
| compute6 | integ-c6 | `~/dojo-integ-c6-state` (confirm deployment override) | `~/dojo-gpu.lock` | `~/dojo-gpu-heavy2.lock` | none |

After permission and drain, run one foreground scope per lane, under an owned
SSH/session launcher with its PID recorded. Example for compute4 heavy1:

```bash
COMMON=(--root "$ROOT" --state "$STATE" --code "$CODE" --rdhome "$ROOT/rdhome"
        --checkpoint "$MODELS/sam3.pt" --profile camera
        --lane-state "$HOME/dojo-integ-c4-state" --stream integ-c4
        --boot-stamp "$HOME/coord/lane_gate.stamp")
systemd-run --user --scope --unit=integ-c4-rsi-heavy1 \
  -p CPUQuota=1400% -p MemoryMax=24G \
  nice -n 10 ionice -c3 env -i HOME="$ROOT/home" PATH=/usr/bin:/bin PYTHONPATH="$CODE" \
  python3 "$GLUE/runner.py" batch "${COMMON[@]}" \
  --lane heavy1 --lock "$HOME/dojo-integ-c4-heavy1.lock" --shards 2 --shard 0 --episodes 5
```

Heavy2 uses a distinct scope, heavy2 lock and `--shard 1`. For compute6 use its
stream/state/lock paths and omit `--boot-stamp`. Counts, memory, free VRAM,
STOPs and disk can reduce actual concurrency below two. Do not widen the
lane gate or VRAM thresholds when SAM3 does not fit.

This command splits cells, not the two halves of one cell. The first pass is
positions 0–4 of each frozen ten. Running the remaining positions 5–9 needs a
separately authorized position-preserving client invocation; the first-pass
command must not be reused on a reindexed five-layout directory.

## Desktop partial-use profile

Reuse the existing install and checkpoint. One `heavy1` lock for all prep and
policy jobs, no second lane. Run the whole batch in an RSI user scope with
`-p CPUQuota=800% -p MemoryMax=16G`, under `nice -n 10 ionice -c3`. Pass
`--profile desktop --episodes 5` and all explicit paths. The executable checks
the actual scope's cgroup memory/CPU limits before GPU admission. It waits
five minutes on >2 GiB pre-start total used VRAM, load1>8 or power>300 W;
`admission.jsonl` records every decision. Only our recorded children are stopped.
Nothing is enabled at reboot. The existing desktop processes are never touched.
