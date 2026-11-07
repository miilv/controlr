#!/usr/bin/env bash
# CLASSIFY3 offline frame-0 estimate (dev) on compute3: v2b segmenter server + (for the SAM arms) one SAM3 text server,
# then frame0_e2e.py per arm, then both servers stopped by their own PIDs. One free light lock (light3 / light4), a
# systemd user scope 12G / 600 %; B1's silhouettes on CPU (DOJO_GL=0: no rendering), SAM / v2b on CUDA (SAM <= 10 GB).
#   REPO=~/controlr-classify3-samint scripts/perception/classify3/e2e_job.sh ROWS OUT_DIR ARM[=ROUTES_JSON] ...
#   e.g. ... v2b 'union={"classify_objects_by_language": {"mode": "union", "model": "sam3"}}'
set -uo pipefail
REPO="${REPO:?the deployed tree with SAM routing}"
ROWS="$1"; OUTD="$2"; shift 2
PY="${E2E_PY:-$HOME/controlr-identity/.venv/bin/python}"
VEND="$HOME/controlr-samint/vendor:$HOME/controlr-sambake/vendor"
SEGW="${SEG_WEIGHTS:-$HOME/dojo-data/integ3-1106/segtrain/models/v2b/best.pt}"
mkdir -p "$OUTD"
free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9'); [[ "$free" -ge 10 ]] || { echo "disk ${free} GB" >&2; exit 6; }
LOCK=""
for L in "$HOME/dojo-gpu-light3.lock" "$HOME/dojo-gpu-light4.lock"; do
  exec 9>>"$L"; if flock -n 9; then LOCK="$L"; break; fi; exec 9>&-
done
[[ -n "$LOCK" ]] || { echo "[e2e] no free light lock" >&2; exit 75; }
printf 'pid=%s stream=CLASSIFY3 start=%s cmd=e2e\n' "$$" "$(date -u +%FT%TZ)" > "$LOCK"
SEG_SOCK="$OUTD/seg.sock"; SAM_SOCK="$OUTD/sam.sock"; rm -f "$SEG_SOCK" "$SAM_SOCK"
start_srv() {  # name sock cmd...: start, wait for its socket, echo "pid starttime"
  local name="$1" sock="$2"; shift 2
  ( cd "$REPO" && exec "$@" ) > "$OUTD/$name.log" 2>&1 &
  local p=$!
  for _ in $(seq 240); do [[ -S "$sock" ]] && break; kill -0 $p 2>/dev/null || { echo "[e2e] $name died" >&2; return 1; }; sleep 0.5; done
  [[ -S "$sock" ]] || return 1
  echo "$p $(awk '{print $22}' /proc/$p/stat)"
}
stop_srv() {  # "pid starttime": never signal a recycled PID
  local p="${1%% *}" st="${1##* }"
  [[ -n "$p" && "$(awk '{print $22}' /proc/$p/stat 2>/dev/null)" == "$st" ]] && { kill "$p"; wait "$p" 2>/dev/null; }
}
SEG=$(PYTHONPATH="$REPO:$VEND" YOLO_CONFIG_DIR="$OUTD/ultra_cfg" start_srv seg "$SEG_SOCK" "$PY" -m dojo.perception.seg_twin serve \
      --sock "$SEG_SOCK" --weights "$SEGW" --classes "$(dirname "$SEGW")/classes.json" --max-hours 3) || exit 1
SAM=""
trap 'stop_srv "$SEG"; [[ -n "$SAM" ]] && stop_srv "$SAM"; rm -f "$SEG_SOCK" "$SAM_SOCK"' EXIT
export DOJO_BG_FOLDS="$HOME/dojo-perception-data/bg_folds_51" DOJO_CALIB_SET=frozen_d0_100040_49 DOJO_GL=0
export DOJO_CATALOG="${DOJO_CATALOG:-$HOME/dojo-perception-data/catalog}" DOJO_SEG_ADDR="$SEG_SOCK"
for spec in "$@"; do
  arm="${spec%%=*}"; routes=""; [[ "$spec" == *=* ]] && routes="${spec#*=}"
  if [[ -n "$routes" && -z "$SAM" ]]; then
    SAM=$(PYTHONPATH="$REPO:$VEND" DOJO_SAM_MODELS="$HOME/controlr-samint/models" start_srv sam "$SAM_SOCK" "$PY" -m dojo.perception.sam_masks \
          serve --sock "$SAM_SOCK" --models sam3 --max-hours 3) || exit 1
  fi
  cfg=""; [[ -n "$routes" ]] && cfg="{\"sam_task_routes\": $routes}"
  ( cd "$REPO" && DOJO_SAM_ADDR="$([[ -n "$routes" ]] && echo "$SAM_SOCK")" DOJO_PERCEPTION_CFG="$cfg" PYTHONPATH="$REPO:$VEND" \
      systemd-run --user --scope -q -p MemoryMax=12G -p CPUQuota=600% \
      "$PY" scripts/perception/classify3/frame0_e2e.py --rows "$ROWS" --layouts "$HOME/dojo-data/simlab/layouts" --arm "$arm" \
      --out "$OUTD/$arm.jsonl" ) >> "$OUTD/e2e.log" 2>&1
done
echo "[e2e] done"
