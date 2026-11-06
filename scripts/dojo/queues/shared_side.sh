#!/usr/bin/env bash
# box-level shared side servers for camera lanes (lead 29 Oct, after #301's locks): seg v2b, glyph digit_v2,
# glyphk kong_v1b, from LANE_REPO, each in its own stream-owned scope.
#   shared_side.sh start | stop | status
S=${LANE_STATE:-$HOME/dojo-integ-c4-state}; R=${LANE_REPO:?set LANE_REPO}; M=${LANE_MODELS:-$HOME/dojo-data/segtrain/models}; D=$S/shared
STREAM=${LANE_STREAM:-integ-c4}
mkdir -p "$D"
start_one() {   # KIND MODEL CMD...
  local k=$1 m=$2; shift 2
  if kill -0 "$(cat "$D/$k.pid" 2>/dev/null)" 2>/dev/null; then
    timeout 15 "$R/.venv/bin/python" "$S/ping_side.py" "$D/$k.sock" >/dev/null 2>&1 || return 1
    echo "$k already up (pid $(cat "$D/$k.pid"))"; return 0
  fi
  rm -f "$D/$k.sock"
  ( cd "$R" && exec systemd-run --user --scope -q --unit="${STREAM}_shared_${k}_$$" -p MemoryMax=16G -p CPUQuota=400% "$@" ) \
    >> "$D/$k.log" 2>&1 < /dev/null &
  local pid=$! i
  echo "${STREAM}_shared_${k}_$$" > "$D/$k.unit"; echo "$pid" > "$D/$k.pid"
  for i in $(seq 1 60); do [[ -S "$D/$k.sock" ]] && break; sleep 2; done
  [[ -S "$D/$k.sock" ]] || { echo "$k NOT ready (see $D/$k.log)"; return 1; }
  # the python inside the scope: the server's own PID (the scope wrapper execs it)
  echo "$m" > "$D/$k.model"; echo "$k up (pid $pid, model $m)"
}
case "${1:-status}" in
  start)
    start_one seg v2b env YOLO_CONFIG_DIR="$S/ultra_cfg" "$R/.venv/bin/python" -u -m dojo.perception.seg_twin serve \
      --sock "$D/seg.sock" --weights "$M/v2b/best.pt" --classes "$M/v2b/classes.json" --conf 0.25 --max-hours 48 || exit 1
    start_one glyph digit_v2 env PYTHONPATH="$R" "$R/.venv/bin/python" -u -m dojo.perception.glyph_read serve \
      --sock "$D/glyph.sock" --model "$M/digit_v2/model.pt" --max-hours 48 || exit 1
    start_one glyphk kong_v1b env PYTHONPATH="$R" "$R/.venv/bin/python" -u -m dojo.perception.glyph_read serve \
      --sock "$D/glyphk.sock" --model "$M/kong_v1b/model.pt" --max-hours 48 || exit 1 ;;
  stop)
    rm -f "$S/shared.on"
    for k in seg glyph glyphk; do
      u=$(cat "$D/$k.unit" 2>/dev/null); p=$(cat "$D/$k.pid" 2>/dev/null)
      [[ "$u" == "${STREAM}_shared_${k}_"* && "$p" =~ ^[0-9]+$ ]] || { echo "invalid $k ownership" >&2; exit 2; }
      # The recorded wrapper must still belong to this exact scope before stopping it.
      if kill -0 "$p" 2>/dev/null; then
        grep -qF "$u.scope" "/proc/$p/cgroup" || { echo "PID $p ownership changed" >&2; exit 2; }
        systemctl --user stop "$u.scope"
      fi
    done
    rm -f "$D"/*.pid "$D"/*.sock; echo stopped ;;
  status)
    for k in seg glyph glyphk; do p=$(cat "$D/$k.pid" 2>/dev/null); a=$(kill -0 "$p" 2>/dev/null && echo alive || echo dead)
      timeout 15 "$R/.venv/bin/python" "$S/ping_side.py" "$D/$k.sock" >/dev/null 2>&1 && pg=ok || pg=FAIL
      echo "$k pid $p $a ping $pg model $(cat "$D/$k.model" 2>/dev/null)"; done ;;
esac
