#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "STOP: missing virtualenv Python at $PYTHON" >&2
  exit 2
fi

cd "$REPO_ROOT" || exit 2
exec "$PYTHON" scripts/gpt2/run_phase_c_bounded.py "$@"
