#!/usr/bin/env bash
# Stop OUR processes on Misha's box (shared Unix user): only PIDs from ~/data2/ilia/run/*.pid whose
# /proc/<pid>/environ contains DOJO_ILIA=1 (and their children). Never by name pattern.
set -uo pipefail
RUN=/home/anywherevla/data2/ilia/run
ours() { grep -qz '^DOJO_ILIA=1$' "/proc/$1/environ" 2>/dev/null; }   # match only, never print the env
for f in "$RUN"/*.pid; do
  [[ -f "$f" ]] || continue
  p=$(cat "$f")
  if [[ -d /proc/$p ]] && ours "$p"; then
    for c in $(ps -o pid= --ppid "$p"); do ours "$c" && kill "$c" 2>/dev/null; done
    kill "$p" && echo "stopped $p ($(basename "$f"))"
  else
    echo "skip $p ($(basename "$f")): gone or not ours"
  fi
  rm -f "$f"
done
