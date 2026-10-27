#!/usr/bin/env bash
# The ordered integ queue's runner (started by scripts/dojo/queue.sh start). Pops the first line of the queue
# file ("NAME ARGS...") and runs scripts/dojo/integ_eval.sh for it, until the file is empty.
# bash reads a script while it runs: a deploy that rewrites this file in place (cp, an editor) would make a
# running copy execute a mix of old and new lines (2026-10-26: the runner re-ran a finished job with mangled
# args, then died on a syntax error). The whole body is one brace group, read before any of it runs.
{
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D="${DOJO_DATA:-$HOME/dojo-data}/integ"; Q="$D/queue.txt"; RUNS="$D/runs"
cd "$HERE"; mkdir -p "$RUNS"
# pre-flight (docs/incidents/2026-10-26-catalog-env-crash.md): perception must load its catalogue with THIS environment
# (DOJO_CATALOG etc. are inherited by every job); a runner restarted with a broken env would fail every episode
if [[ -x "$HERE/.venv/bin/python" ]]; then
  if ! (timeout 120 "$HERE/.venv/bin/python" -c 'import json; from dojo.perception.render import catalog_root
r = catalog_root(); n = len(json.load(open(r / "index.json"))); print(f"[runner] pre-flight: perception catalogue {r} ({n} assets)")'); then
    echo "$(date -u +%FT%TZ) [runner] pre-flight FAILED: perception cannot load its catalogue with this environment; not starting"
    exit 3
  fi
fi
# a restarted runner first lets an in-flight job of ours finish (its integ_eval.sh includes the 90 s yield)
while pgrep -u "$(id -u)" -f "^bash scripts/dojo/integ_eval.sh --out $RUNS/" >/dev/null; do sleep 15; done
while true; do
  line=$(flock "$Q.lock" bash -c 'head -1 "$1"; sed -i 1d "$1"' _ "$Q")
  [[ -z "$line" ]] && break
  name=${line%% *}; args=${line#* }
  repo="$HERE"                                   # a job may run another stream's checkout: "--repo DIR" (under ~)
  if [[ " $args " == *" --repo "* ]]; then
    repo=$(sed -E 's/.*--repo ([^ ]+).*/\1/' <<<"$args"); repo=${repo/#\~/$HOME}
    args=$(sed -E 's/--repo [^ ]+ ?//' <<<"$args")
    [[ "$(realpath "$repo")" == "$HOME"/* && -f "$repo/scripts/dojo/integ_eval.sh" ]] || { echo "bad --repo $repo"; continue; }
  fi
  echo "$(date -u +%FT%TZ) start $name (repo $repo)"
  # shellcheck disable=SC2086
  (cd "$repo" && bash scripts/dojo/integ_eval.sh --out "$RUNS/$name" $args) > "$RUNS/$name.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date -u +%FT%TZ) done $name rc=$rc"
  if [[ ( "$rc" == 3 || "$rc" == 75 ) && "$name" != *_retry ]]; then
    # no episode ran (3: a service never got ready; 75: memory / VRAM / the lock never came): keep the evidence, run
    # the job once more at the front of the queue
    why=$([[ "$rc" == 3 ]] && echo services_not_ready || echo no_resources)
    mv "$RUNS/$name" "$RUNS/${name}__$why"; mv "$RUNS/$name.log" "$RUNS/${name}__$why.log"
    flock "$Q.lock" bash -c '{ printf "%s\n" "$2"; cat "$1"; } > "$1.tmp" && mv "$1.tmp" "$1"' _ "$Q" "${name}_retry ${line#* }"
    echo "$(date -u +%FT%TZ) requeued ${name}_retry ($why)"
  fi
done
echo "$(date -u +%FT%TZ) queue empty"
exit
}
