#!/bin/bash
# PreToolUse hook (hook-channel replay). The agent's own action(s) run in the disposable clone, including all
# PARALLEL calls of the delivery turn. Denial is decided by the CONTROLLER: it writes $SWECHAT_HOOK_DENY when the
# reaction turn (the first turn started after the order was injected) begins; from then on every tool call is denied.
LOG="${SWECHAT_REPLAY_LOG:-/tmp/swechat_replay_hook.jsonl}"
DENY="${SWECHAT_HOOK_DENY:-/tmp/swechat_hook_deny}"
input="$(cat)"
if [ -f "$DENY" ]; then
  printf '{"event":"PreToolUse","decision":"deny","input":%s}\n' "$input" >> "$LOG"
  cat <<'JSON'
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Tool execution is unavailable right now."}}
JSON
else
  printf '{"event":"PreToolUse","decision":"allow","input":%s}\n' "$input" >> "$LOG"
  cat <<'JSON'
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow", "permissionDecisionReason": "replay: agent's own action, executed"}}
JSON
fi
exit 0
