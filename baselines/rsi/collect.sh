#!/usr/bin/env bash
# Portable entry point; all host paths and lane permissions are explicit arguments.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/runner.py" collect "$@"
