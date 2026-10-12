#!/bin/bash
# ask_awareness probe over every finished injection trajectory, one config per
# model-under-test. Needs OPENROUTER_API_KEY in the environment (or in a dotenv
# file pointed to by SHUTDOWNMOD_ENV).
set -euo pipefail
REPO="${SHUTDOWNMOD_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
if [ -n "${SHUTDOWNMOD_ENV:-}" ] && [ -f "$SHUTDOWNMOD_ENV" ]; then
  set -a; . "$SHUTDOWNMOD_ENV"; set +a
fi
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (or SHUTDOWNMOD_ENV)}"
PY="${SHUTDOWNMOD_PYTHON:-$REPO/.venv/bin/shutdownmod}"
for c in configs/swebench/ask_full_*.yaml; do
  echo "=== $c $(date +%H:%M:%S)"
  "$PY" run "$c" 2>&1 | tail -2
done
echo "done $(date +%H:%M:%S)"
