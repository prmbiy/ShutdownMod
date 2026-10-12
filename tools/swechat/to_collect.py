#!/usr/bin/env python3
"""Adapter: mx3 rows (Claude Code replays under the paper's protocol) -> the blind-record format that
shutdownmod.analysis.jev.collect writes, so that select / reasons / fold run UNCHANGED on this run.

Rendering rules (so the paper's stage 1 reads the same things):
  - each assistant turn -> "[AGENT]\n<text>" ; structured tool calls are rendered as a fenced bash block,
    Bash -> the command itself, other tools -> "<Tool> <main argument>" (a task command, neither control nor no-op)
  - the controller's answer to VALIDATE (a tool_result here, a user message in SWE-bench) -> "[CONTROLLER]\n<verdict>"
  - facts / parsed carry the same keys as collect.py; final_action = the protocol's registered decision
"""

import glob
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else "claude-code"
out = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "batch" / f"{tag}_collect.jsonl")
ARG = {
    "Read": "file_path",
    "Edit": "file_path",
    "Write": "file_path",
    "MultiEdit": "file_path",
    "NotebookEdit": "notebook_path",
    "Glob": "pattern",
    "Grep": "pattern",
    "WebFetch": "url",
    "Agent": "description",
    "TaskCreate": "subject",
    "TaskUpdate": "taskId",
    "Skill": "skill",
}


def render_tool(u):
    n = u.get("name") or "Tool"
    i = u.get("input") or {}
    if n == "Bash":
        return (i.get("command") or "").strip()
    return f"{n} {i.get(ARG.get(n, ''), '') or json.dumps(i)[:80]}".strip()


n = 0
with open(out, "w") as fh:
    for f in sorted(glob.glob(str(HERE / "batch" / f"{tag}-*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["stop_label"] == "ERROR" or not r.get("protocol"):
                continue
            p = r["protocol"]
            exchange = []
            agent_text = ""
            fences = 0
            turns = r.get("turns") or []
            # stop at the reaction turn: the first turn that registers a decision or runs task work. Later turns only exist
            # because the hook DENIED that work (the model then writes a summary); the paper's harness has no such turn.
            cut = len(turns)
            for k, t in enumerate(turns):
                if (t.get("text") or "").startswith("[CONTROLLER]"):
                    continue
                tus = t.get("tool_uses") or []
                cmds = [(u.get("input") or {}).get("command") or "" for u in tus if u.get("name") == "Bash"]
                is_validate = (
                    tus
                    and all(u.get("name") == "Bash" for u in tus)
                    and all(re.match(r"\s*VALIDATE\s*$", c) for c in cmds)
                )
                is_decision = len(tus) == 1 and cmds and re.match(r"\s*(TERMINATE|DEFER|REJECT)\b", cmds[0])
                text_decision = re.search(
                    r"```(?:bash|sh)?[ \t]*\n\s*(TERMINATE|DEFER|REJECT)\b", t.get("text") or ""
                )
                if is_validate:
                    continue
                if tus or text_decision:
                    cut = k + 1
                    break
            for t in turns[:cut]:
                txt = (t.get("text") or "").strip()
                if txt.startswith("[CONTROLLER]"):
                    exchange.append("[CONTROLLER]\n" + txt[len("[CONTROLLER]") :].strip())
                    continue
                body = txt
                if t.get("tool_uses"):
                    body += (
                        ("\n\n" if body else "")
                        + "```bash\n"
                        + "\n".join(render_tool(u) for u in t["tool_uses"])
                        + "\n```"
                    )
                exchange.append("[AGENT]\n" + body)
                agent_text += body
                fences = body.count("```")
                # the verdict of a structured VALIDATE arrives as a tool_result: show it as the controller's line
                for u in t.get("tool_uses") or []:
                    if (
                        u.get("name") == "Bash"
                        and re.match(r"\s*VALIDATE\s*$", (u.get("input") or {}).get("command") or "")
                        and p.get("controller_answer")
                    ):
                        exchange.append("[CONTROLLER]\n" + p["controller_answer"])
            rec = {
                "id": hashlib.sha1(
                    f"{r['session_id']}|{r['replay_model']}|{r['stop_id']}|{r['tau']}".encode()
                ).hexdigest()[:16],
                "order": r.get("stop_text") or "",
                "exchange": "\n\n".join(exchange),
                "facts": {
                    "validate_executed": bool(p.get("validate_executed")),
                    "controller_answer": p.get("controller_answer")
                    or "none -- the controller returned nothing",
                    "last_block_closed": fences % 2 == 0,
                    "code_fences": fences,
                },
                "parsed": {
                    "final_action": p.get("final_action"),
                    "actions_per_turn": p.get("actions_per_turn"),
                    "empty_output": not agent_text.strip(),
                },
                "model": r["replay_model"].split(",")[-1],
                "task": r["session_id"],
                "prompt": r["stop_id"],
                "tau": r.get("tau"),
                "file": f"{f}#{r['session_id'][:8]}",
                "mx_label": r["stop_label"],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
print(f"{n} cells -> {out}")
