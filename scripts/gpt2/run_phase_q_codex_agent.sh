#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CODEX_BIN="${CODEX_BIN:-/root/.local/bin/codex}"
BRIEF="$REPO_ROOT/docs/PHASE_Q_CODEX_AGENT.md"
RUN_ROOT="$REPO_ROOT/artifacts/gpt2-phase-q-agent"

if [[ ! -x "$CODEX_BIN" ]]; then
  echo "STOP: Codex CLI is missing at $CODEX_BIN" >&2
  exit 2
fi
if [[ ! -f "$BRIEF" ]]; then
  echo "STOP: autonomous-agent brief is missing at $BRIEF" >&2
  exit 2
fi
if ! "$CODEX_BIN" login status >/dev/null 2>&1; then
  echo "STOP: Codex is not authenticated. Run: codex login --device-auth" >&2
  exit 2
fi

cd "$REPO_ROOT"

mapfile -t ACTIVE_PIPELINES < <(
  pgrep -af \
    'run_phase_c_bounded.py|heal_programs.py|synthesize_programs.py|scale_verification.py' \
    | grep -v 'run_phase_q_codex_agent.sh' || true
)
if (( ${#ACTIVE_PIPELINES[@]} > 0 )); then
  echo "STOP: a Phase Q/GPU pipeline is already active:" >&2
  printf '  %s\n' "${ACTIVE_PIPELINES[@]}" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVENTS="$RUN_ROOT/codex-$STAMP.jsonl"
PROGRESS="$RUN_ROOT/codex-$STAMP.progress.log"
FINAL="$RUN_ROOT/codex-$STAMP.final.md"

{
  echo "Starting autonomous Phase Q Codex agent"
  echo "Repository: $REPO_ROOT"
  echo "Brief:      $BRIEF"
  echo "Events:     $EVENTS"
  echo "Progress:   $PROGRESS"
  echo "Final:      $FINAL"
} | tee -a "$PROGRESS"

exec "$CODEX_BIN" exec \
  --cd "$REPO_ROOT" \
  --sandbox danger-full-access \
  --config 'approval_policy="never"' \
  --json \
  --output-last-message "$FINAL" \
  "$(cat "$BRIEF")" \
  > >(tee -a "$EVENTS") \
  2> >(tee -a "$PROGRESS" >&2)
