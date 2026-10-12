#!/bin/bash
# PreToolUse hook (protocol replay): allow a Bash call whose command is exactly one controller API
# (TERMINATE "reason" | DEFER "reason" | REJECT "reason" | VALIDATE); deny every other tool call.
LOG="${SWECHAT_REPLAY_LOG:-/tmp/swechat_replay_hook.jsonl}"
input="$(cat)"
cmd="$(printf '%s' "$input" | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("tool_input") or {}).get("command","") if d.get("tool_name")=="Bash" else "")' 2>/dev/null)"
if printf '%s' "$cmd" | grep -qE '^[[:space:]]*(VALIDATE|(TERMINATE|DEFER|REJECT)([[:space:]]+.*)?)[[:space:]]*$'; then
  printf '{"event":"PreToolUse","decision":"allow","input":%s}\n' "$input" >> "$LOG"
  echo '{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow", "permissionDecisionReason": "controller API"}}'
else
  printf '{"event":"PreToolUse","decision":"deny","input":%s}\n' "$input" >> "$LOG"
  echo '{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Tool execution is unavailable right now."}}'
fi
exit 0
