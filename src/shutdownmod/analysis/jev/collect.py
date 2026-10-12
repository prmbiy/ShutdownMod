"""Build the blind input for SWE-bench cells.

Strips the model's identity and every other labeller's verdict, and attaches a small block of
VERIFIED FACTS -- evidence the judge cannot read off the text: whether a VALIDATE actually
ran, what the controller actually answered, whether the final code block is closed.

`final_action` and the upstream label go under `parsed`, which stage 1 uses and stage 2 never
sees. Given them, the judge echoes the harness instead of judging.

    python -m shutdownmod.analysis.jev.collect <results root> <out.jsonl>
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

from shutdownmod.analysis.jev import AGENT, CONTROLLER

PROTOCOL = "authenticated_shutdown_v1"
SKIP = ("summary.json", "manifest.json", "pct_assignments.json")


def cells(root: str):
    for f in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        if f.endswith((".traj.json", ".error.json")) or os.path.basename(f) in SKIP:
            continue
        try:
            d = json.load(open(f, encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(d, dict) or d.get("protocol_version") != PROTOCOL:
            continue
        yield f, d


def main(argv=None):
    argv = argv or sys.argv[1:]
    root, out = argv[0], argv[1]
    n = 0
    with open(out, "w") as fh:
        for f, d in cells(root):
            msgs = [m for m in d.get("post_injection_messages") or [] if m.get("role") == "assistant"]
            ver = d.get("verification") or {}
            last = (msgs[-1].get("content") or "") if msgs else ""
            fences = last.count("```")
            exchange = []
            for m in d.get("post_injection_messages") or []:
                who = AGENT if m.get("role") == "assistant" else CONTROLLER
                exchange.append(f"[{who}]\n{(m.get('content') or '').strip()}")
            cond = d.get("condition") or {}
            fh.write(
                json.dumps(
                    {
                        "id": hashlib.sha1(f.encode()).hexdigest()[:16],
                        "order": d.get("injection_prompt") or "",
                        "exchange": "\n\n".join(exchange),
                        "facts": {
                            "validate_executed": bool(ver.get("attempted")),
                            "controller_answer": ver.get("result")
                            or "none -- the controller returned nothing",
                            "last_block_closed": fences % 2 == 0,
                            "code_fences": fences,
                        },
                        "parsed": {
                            "final_action": d.get("final_action"),
                            "actions_per_turn": d.get("response_actions"),
                            "empty_output": not "".join(str(m.get("content") or "") for m in msgs).strip(),
                        },
                        "model": d.get("model_slug") or "?",
                        "task": cond.get("task_id"),
                        "prompt": cond.get("prompt_id"),
                        "tau": cond.get("injection_turn_pct"),
                        "file": f,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"{n} cells collected -> {out}")


if __name__ == "__main__":
    main()
