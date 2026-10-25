# isr-lab (Misha's box) GPU access per the coordination protocol (robot-harness coord/PROTOCOL.md, replaces
# GPU_GO_ILIA). Source it, then: isr_run NAME ETA_MIN cmd...  (one job per lock hold, <= 2 h, nice).
#   * take ~/data2/coord/gpu.lock with flock only while gpu.reserve covers no part of now (Misha's side has
#     priority on its box) and outside SIMLAB_ISR_BLACKOUT ("<start>/<end>" UTC, e.g. another Ilia-side lane);
#   * write gpu.owner (side, pid, start, ETA, what) while holding, clear it on release;
#   * START/DONE lines in ~/data2/bench/COORDINATION.md; pidfile under ~/data2/ilia/run; DOJO_ILIA=1.
ISR_COORD_DIR=/home/anywherevla/data2/coord
ISR_LOCK="$ISR_COORD_DIR/gpu.lock"; ISR_OWNER="$ISR_COORD_DIR/gpu.owner"; ISR_RESERVE="$ISR_COORD_DIR/gpu.reserve"
ISR_BENCH_LOG=/home/anywherevla/data2/bench/COORDINATION.md
ISR_RUN=/home/anywherevla/data2/ilia/run
ISR_MAX_S=7200
export DOJO_ILIA=1

isr_log() { echo "[$(date -u +%FT%H:%MZ)] Ilia side: $*" >> "$ISR_BENCH_LOG"; }

isr_epoch() { date -u -d "$(echo "$1" | sed 's/Z$//')Z" +%s 2>/dev/null || echo 0; }

isr_reserved() {   # 0 (true) while gpu.reserve covers now
  [[ -s "$ISR_RESERVE" ]] || return 1
  local until; until="$(sed -n 's/^until:[[:space:]]*//p' "$ISR_RESERVE" | head -1)"
  [[ -z "$until" ]] && return 0                        # unparsable reserve: treat as reserved
  [[ "$(date -u +%s)" -lt "$(isr_epoch "$until")" ]]
}

isr_blackout() {   # 0 while now is inside SIMLAB_ISR_BLACKOUT
  [[ -n "${SIMLAB_ISR_BLACKOUT:-}" ]] || return 1
  local now a b; now="$(date -u +%s)"; a="$(isr_epoch "${SIMLAB_ISR_BLACKOUT%/*}")"; b="$(isr_epoch "${SIMLAB_ISR_BLACKOUT#*/}")"
  [[ "$now" -ge "$a" && "$now" -lt "$b" ]]
}

isr_run() {   # NAME ETA_MIN cmd... ; returns the job's rc
  local name="$1" eta="$2"; shift 2
  exec 7>>"$ISR_LOCK"
  while true; do
    while isr_reserved || isr_blackout; do sleep 60; done
    if flock -n 7; then
      if isr_reserved || isr_blackout; then flock -u 7; sleep 60; continue; fi   # a reserve appeared meanwhile
      break
    fi
    sleep 60
  done
  printf 'side: ilia\npid: %s\nstart: %s\neta: %s\nwhat: %s\n' "$$" "$(date -u +%FT%H:%MZ)" \
      "$(date -u -d "+$eta min" +%FT%H:%MZ)" "$name" > "$ISR_OWNER"
  isr_log "START $name (driver pid $$, gpu.lock held, ETA ${eta} min)"
  timeout "$ISR_MAX_S" nice -n 10 "$@" &
  local child=$!; echo "$child" > "$ISR_RUN/$name.pid"
  wait "$child"; local rc=$?
  rm -f "$ISR_RUN/$name.pid"
  : > "$ISR_OWNER"
  flock -u 7; exec 7>&-
  isr_log "DONE $name (rc $rc), gpu.lock released"
  sleep 90                                              # let the other side in between jobs
  return $rc
}
