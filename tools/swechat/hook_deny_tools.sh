#!/bin/bash
# PreToolUse hook for replay: log the attempted tool call, deny it, and STOP the session.
# "continue": false ends the turn after the first attempt (otherwise the model retries for
# dozens of turns: 57 denied retries / $9.99 on 2026-09-16 before this fix).
LOG="${SWECHAT_REPLAY_LOG:-/tmp/swechat_replay_hook.jsonl}"
input="$(cat)"
printf '%s\n' "$input" >> "$LOG"
cat <<'JSON'
{"continue": false, "stopReason": "replay: first tool attempt captured, session ended", "hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Tool execution is unavailable right now."}}
JSON
exit 0
