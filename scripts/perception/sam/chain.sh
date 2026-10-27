#!/usr/bin/env bash
# chain.sh NAME:CFG_JSON ... : SAM bake-off variant runs one after another (8-core scope each), all sets incl. laptop.
set -u
export DOJO_SAM_ADDR=$HOME/dojo-data/sam/sam.sock
for spec in "$@"; do
  name=${spec%%:*}; cfg=${spec#*:}
  [ -f ~/dojo-data/sam/$name/frames.jsonl ] && [ "$(wc -l < ~/dojo-data/sam/$name/frames.jsonl)" -ge 222 ] && continue
  echo "$(date -u +%FT%TZ) start $name $cfg"
  ~/dojo-data/sam/cpu8.sh .venv/bin/python -W ignore scripts/perception/sam/bake.py run --out ~/dojo-data/sam/$name \
      --sets d0_51,d0_ext,coin,laptop --workers ${WORKERS:-4} --cfg "$cfg" > ~/dojo-data/sam/$name.log 2>&1
  echo "$(date -u +%FT%TZ) done $name rc=$?"
done
