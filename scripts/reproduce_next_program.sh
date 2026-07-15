#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export TWISTEDMERGE_DATA_ROOT="${TWISTEDMERGE_DATA_ROOT:-${ROOT}/data}"
exec "${PYTHON_BIN}" "${ROOT}/experiments/run_next_twistedmerge_program.py" --tier all --resume "$@"
