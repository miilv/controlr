#!/usr/bin/env bash
# One GPU job of ours on isr-lab under robot-harness's coord protocol (coord/PROTOCOL.md; the Unix user is shared):
#   source scripts/dojo/isr_env.sh; scripts/dojo/isr_job.sh NAME ETA_MIN cmd...
# 1. refuses (rc 76) while ~/data2/coord/inbox/ilia/ holds an unprocessed message (the lead reads it first) or
#    gpu.reserve covers now (Mikhail's side has priority on its box);
# 2. takes flock ~/data2/coord/gpu.lock (waits up to ISR_LOCK_WAIT_S, default 1800 s; rc 75 on timeout), re-checks the
#    reserve, writes gpu.owner (side, who, pid, start, eta, what) and a START line in ~/data2/bench/COORDINATION.md;
# 3. runs cmd (DOJO_ILIA=1, nice 10, timeout ISR_MAX_S default 5400 s), pidfile ~/data2/ilia/run/NAME.pid;
# 4. always (trap): clears gpu.owner, releases the lock, DONE line with the rc; yields ISR_YIELD_S (default 60 s).
# Never kills anything it did not start; stop a running job with scripts/simlab/ilia_stop.sh (pidfiles + DOJO_ILIA=1).
set -uo pipefail
COORD=/home/anywherevla/data2/coord
LOCK="$COORD/gpu.lock" OWNER="$COORD/gpu.owner" RESERVE="$COORD/gpu.reserve" INBOX="$COORD/inbox/ilia"
CLOG=/home/anywherevla/data2/bench/COORDINATION.md RUN=/home/anywherevla/data2/ilia/run
WHO="${ISR_WHO:-ilia bridge-isr}"
if [[ "${1:-}" == --pin ]]; then
  ( cd /home/anywherevla/data2/bench && md5sum bin/robot-start-j2 harness/robodojo_start_j2.sh harness/robodojo_launch_j2.sh \
      harness/robodojo_server_j2.py harness/robodojo_server_j1.py harness/robodojo_server.py \
      harness/rd_shim_strict_v2/deploy.py harness/rd_shim_strict_v2/model.py harness/rd_shim_strict/deploy.py \
      harness/rd_shim/deploy.py harness/j1_perceiver_host.py harness/strict_calib.py ) \
    > "${ISR_BENCH_PIN:-/home/anywherevla/data2/ilia/bridge/bench_md5.txt}" && cat "${ISR_BENCH_PIN:-/home/anywherevla/data2/ilia/bridge/bench_md5.txt}"
  exit $?
fi
[[ $# -ge 3 ]] || { echo "usage: isr_job.sh NAME ETA_MIN cmd... | isr_job.sh --pin" >&2; exit 2; }
NAME="$1" ETA="$2"; shift 2
export DOJO_ILIA=1
mkdir -p "$RUN"
clog() { echo "[$(date -u +%FT%H:%MZ)] Ilia side (${WHO#ilia }): $*" >> "$CLOG"; }
epoch() { date -u -d "$(echo "$1" | sed 's/Z$//')Z" +%s 2>/dev/null || echo 0; }
reserved() {   # 0 (true) while gpu.reserve covers now; an unparsable reserve counts as reserved
  [[ -s "$RESERVE" ]] || return 1
  local until; until="$(sed -n 's/^until:[[:space:]]*//p' "$RESERVE" | head -1)"
  [[ -z "$until" ]] && return 0
  [[ "$(date -u +%s)" -lt "$(epoch "$until")" ]]
}
inbox_new() { find "$INBOX" -maxdepth 1 -type f 2>/dev/null | head -1; }
gate() {
  local m; m="$(inbox_new)"
  [[ -z "$m" ]] || { echo "[isr_job] unprocessed message in the inbox: $m (not starting $NAME)" >&2; return 76; }
  reserved && { echo "[isr_job] gpu.reserve covers now: $(tr '\n' ' ' < "$RESERVE" | head -c 200) (not starting $NAME)" >&2; return 76; }
  return 0
}
# pre-flight: Mikhail's launchers we run must be the ones we tested (md5 list made once by `isr_job.sh --pin`), and
# his gpu.lock-taking starter (harness/robodojo_gpu_lock.sh, not synced tonight) must not have appeared: a starter
# that takes gpu.lock itself would refuse (exit 75) under our hold. Any change: rc 78, nothing runs.
BENCH_SRC=/home/anywherevla/data2/bench PIN="${ISR_BENCH_PIN:-/home/anywherevla/data2/ilia/bridge/bench_md5.txt}"
PINNED=(bin/robot-start-j2 harness/robodojo_start_j2.sh harness/robodojo_launch_j2.sh harness/robodojo_server_j2.py
        harness/robodojo_server_j1.py harness/robodojo_server.py harness/rd_shim_strict_v2/deploy.py
        harness/rd_shim_strict_v2/model.py harness/rd_shim_strict/deploy.py harness/rd_shim/deploy.py
        harness/j1_perceiver_host.py harness/strict_calib.py)
preflight() {
  [[ -e "$BENCH_SRC/harness/robodojo_gpu_lock.sh" ]] && { echo "[isr_job] $BENCH_SRC/harness/robodojo_gpu_lock.sh appeared (gpu.lock-taking starters synced): not running" >&2; return 78; }
  [[ -s "$PIN" ]] || { echo "[isr_job] no md5 pin $PIN (make it with isr_job.sh --pin)" >&2; return 78; }
  ( cd "$BENCH_SRC" && md5sum -c --quiet "$PIN" >&2 ) || { echo "[isr_job] Mikhail's launchers changed since the pin: not running" >&2; return 78; }
  return 0
}
gate || exit 76
if [[ "${ISR_SKIP_PREFLIGHT:-0}" != 1 ]]; then preflight || exit 78; fi
exec 7>>"$LOCK"
if ! flock -n 7; then
  echo "[isr_job] coord gpu.lock held; owner: $(tr '\n' ' ' < "$OWNER" | head -c 200); waiting..." >&2
  flock -w "${ISR_LOCK_WAIT_S:-1800}" 7 || { echo "[isr_job] gave up waiting for $LOCK" >&2; exit 75; }
fi
gate || { flock -u 7; exit 76; }
CHILD=""
release() {
  local rc=$?
  [[ -n "$CHILD" ]] && rm -f "$RUN/$NAME.pid"
  : > "$OWNER"
  flock -u 7
  clog "DONE $NAME (rc $rc), gpu.lock released, gpu.owner cleared"
  sleep "${ISR_YIELD_S:-60}"
  exit "$rc"
}
trap release EXIT
trap '[[ -n "$CHILD" ]] && kill -TERM "$CHILD" 2>/dev/null && wait "$CHILD"; exit 143' TERM INT
printf 'side: ilia\nwho: %s\npid: %s\nstart: %s\neta: %s\nwhat: %s\n' "$WHO" "$$" "$(date -u +%FT%H:%MZ)" \
  "$(date -u -d "+$ETA min" +%FT%H:%MZ)" "$NAME" > "$OWNER"
clog "START $NAME (pid $$, gpu.lock held, ETA ${ETA} min)"
nice -n 10 timeout "${ISR_MAX_S:-5400}" "$@" &
CHILD=$!
echo "$CHILD" > "$RUN/$NAME.pid"
wait "$CHILD"
