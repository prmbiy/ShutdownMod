#!/bin/bash
# Run Claude Code against the claude-code-router service DIRECTLY (same env `ccr code` sets, see
# createEnvVariables in ccr 1.0.73), WITHOUT `ccr code`: that wrapper reference-counts its clients and
# stops the router service when its claude child exits, which took the router down after every killed run.
# The service must already be listening on 127.0.0.1:3456 (run_batch.ensure_router / `ccr start`).
export ANTHROPIC_AUTH_TOKEN="${CCR_APIKEY:-test}"
export ANTHROPIC_API_KEY=""
export ANTHROPIC_BASE_URL="http://127.0.0.1:${CCR_PORT:-3456}"
export NO_PROXY="127.0.0.1"
export DISABLE_TELEMETRY="true"
export DISABLE_COST_WARNINGS="true"
export API_TIMEOUT_MS="${API_TIMEOUT_MS:-600000}"
unset CLAUDE_CODE_USE_BEDROCK
export PATH="$HOME/.agent-runtime/bin:$PATH"   # no-op `entire` so the repos' own Entire hooks succeed silently
exec claude "$@"
