#!/usr/bin/env bash
# Explicit opt-in shared SAM lifecycle; run only on a lead-allocated box with drained lanes.
set -euo pipefail
: "${LANE_REPO:?set pinned deployment}"
: "${LANE_STATE:?set box state directory}"
: "${LANE_STREAM:?set box stream}"
: "${SAM_PY:?set dedicated torch 2.9.1+cu128 venv python}"
export LANE_SHARED_SAM=${LANE_SHARED_SAM:-sam3}
export DOJO_SAM_MODELS=${DOJO_SAM_MODELS:-$HOME/dojo-data/models}
export PYTHONPATH="$LANE_REPO${PYTHONPATH:+:$PYTHONPATH}"
case "${1:-status}" in
  start)
    [[ ! -e "$LANE_STATE/running.heavy1" && ! -e "$LANE_STATE/running.heavy2" ]] || { echo 'drain lanes before start' >&2; exit 2; }
    # Pin the measured artifacts; no downloads and no LLM calls.
    echo "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e  $DOJO_SAM_MODELS/sam3/sam3.pt" | sha256sum -c -
    echo "9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03  $DOJO_SAM_MODELS/sam3.1/sam3.1_multiplex_fp16.safetensors" | sha256sum -c -
    mkdir -p "$LANE_STATE"
    cp "$LANE_REPO/scripts/dojo/queues/ping_side.py" "$LANE_STATE/ping_side.py"
    bash "$LANE_REPO/scripts/dojo/queues/shared_side.sh" start
    timeout 15 "$SAM_PY" "$LANE_STATE/ping_side.py" "$LANE_STATE/shared/sam.sock"
    touch "$LANE_STATE/shared.on"
    ;;
  stop)
    [[ ! -e "$LANE_STATE/running.heavy1" && ! -e "$LANE_STATE/running.heavy2" ]] || { echo 'drain lanes before stop' >&2; exit 2; }
    bash "$LANE_REPO/scripts/dojo/queues/shared_side.sh" stop ;;
  status) bash "$LANE_REPO/scripts/dojo/queues/shared_side.sh" status ;;
  *) exit 2 ;;
esac
