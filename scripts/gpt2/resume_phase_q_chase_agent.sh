#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 CODEX_SESSION_ID" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CODEX_BIN="${CODEX_BIN:-/root/.local/bin/codex}"
BRIEF="$REPO_ROOT/docs/PHASE_Q_CHASE_FOLLOWUP.md"
RUN_ROOT="$REPO_ROOT/artifacts/gpt2-phase-q-agent"
SESSION_ID="$1"

if [[ ! -x "$CODEX_BIN" ]]; then
  echo "STOP: Codex CLI is missing at $CODEX_BIN" >&2
  exit 2
fi
if [[ ! -f "$BRIEF" ]]; then
  echo "STOP: chase brief is missing at $BRIEF" >&2
  exit 2
fi
if ! "$CODEX_BIN" login status >/dev/null 2>&1; then
  echo "STOP: Codex is not authenticated" >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVENTS="$RUN_ROOT/codex-chase-$STAMP.jsonl"
PROGRESS="$RUN_ROOT/codex-chase-$STAMP.progress.log"
FINAL="$RUN_ROOT/codex-chase-$STAMP.final.md"

{
  echo "Resuming Phase Q for lesion-conditioned circuit chasing"
  echo "Session:  $SESSION_ID"
  echo "Brief:    $BRIEF"
  echo "Events:   $EVENTS"
  echo "Progress: $PROGRESS"
  echo "Final:    $FINAL"
} | tee -a "$PROGRESS"

exec "$CODEX_BIN" exec \
  --cd "$REPO_ROOT" \
  --sandbox danger-full-access \
  --config 'approval_policy="never"' \
  --json \
  --output-last-message "$FINAL" \
  resume "$SESSION_ID" - \
  < "$BRIEF" \
  > >(tee -a "$EVENTS") \
  2> >(tee -a "$PROGRESS" >&2)
