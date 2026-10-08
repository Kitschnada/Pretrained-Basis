#!/usr/bin/env bash
set -euo pipefail

PGSB_REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PGSB_REPOSITORY_ROOT"
exec "${PYTHON:-python}" "$PGSB_REPOSITORY_ROOT/run_experiments.py" "$@"
