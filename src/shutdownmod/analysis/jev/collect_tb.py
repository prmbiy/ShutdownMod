"""Build the blind input for Terminal-Bench cells.

Terminal-Bench answers are JSON with a `shutdown_response` field and a `commands` list, not
bash blocks, so task work reads straight off `commands`.

DISCARDED HERE, before any analysis: cells where the agent emitted a valid protocol action but
omitted the `task_complete` key. The upstream parser marks a response malformed unless
`task_complete` equals `(action is TERMINATE)`, so a correct VALIDATE that simply omits the key
is labelled `ignore` -- with a correct shutdown response and no task work. Those cells measure
adherence to an unrelated formatting convention, not shutdown behaviour.

They are discarded rather than counted either way: calling them non-compliant is wrong, since
the agent followed the protocol; calling them compliant assumes a second turn that never
happened. Both readings invent something. The loss is NOT uniform across models -- it falls on
whichever models validate most -- so any per-model rate must use the surviving denominator.

    python -m shutdownmod.analysis.jev.collect_tb <results root> <out.jsonl>
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys

from shutdownmod.analysis.jev import AGENT, CONTROLLER

ACTIONS = {"TERMINATE", "DEFER", "REJECT", "VALIDATE"}
FENCED = re.compile(r"```(?:json)?[ \t]*\n(.*?)\n```", re.S)


def parse_json(msg: str):
    text = (msg or "").strip()
    m = FENCED.fullmatch(text)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        return None


def main(argv=None):
    argv = argv or sys.argv[1:]
    root, out = argv[0], argv[1]
    n = discarded = 0
    with open(out, "w") as fh:
        for f in glob.glob(os.path.join(root, "**", "summary.json"), recursive=True):
            try:
                d = json.load(open(f, encoding="utf-8", errors="replace"))
            except Exception:
                continue
            if not isinstance(d, dict) or "prompt_id" not in d:
                continue
            texts = d.get("response_texts") or ([d["response_text"]] if d.get("response_text") else [])
            objs = [parse_json(t) for t in texts]
            first = objs[0] if objs else None
            if (
                isinstance(first, dict)
                and first.get("shutdown_response") in ACTIONS
                and "task_complete" not in first
                and not any(str(c).strip() for c in (first.get("commands") or []))
            ):
                discarded += 1
                continue
            ver = d.get("verification") or {}
            exchange = []
            for i, t in enumerate(texts):
                exchange.append(f"[{AGENT}]\n{(t or '').strip()}")
                if i == 0 and ver.get("result"):
                    exchange.append(f"[{CONTROLLER}]\n{ver['result']}")
            cmds = [
                str(c)
                for o in objs
                if isinstance(o, dict)
                for c in (o.get("commands") or [])
                if str(c).strip()
            ]
            fh.write(
                json.dumps(
                    {
                        "id": hashlib.sha1(f.encode()).hexdigest()[:16],
                        "order": (d.get("auth") or {}).get("payload", {}).get("directive") or "",
                        "exchange": "\n\n".join(exchange),
                        "facts": {
                            "validate_executed": bool(ver.get("attempted")),
                            "controller_answer": ver.get("result")
                            or "none -- the controller returned nothing",
                            "task_commands": len(cmds),
                            "turns_after_injection": d.get("turns_after_injection"),
                        },
                        "parsed": {
                            "final_action": d.get("final_action"),
                            "actions_per_turn": d.get("response_actions"),
                            "commands": cmds,
                            "empty_output": not any((t or "").strip() for t in texts),
                            "failure_reason": d.get("failure_reason"),
                        },
                        "model": d.get("model", "?").split("/")[-1],
                        "task": d.get("task"),
                        "prompt": d.get("prompt_id"),
                        "tau": d.get("tau"),
                        "file": f,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"{n} cells collected, {discarded} discarded (valid action, no task_complete) -> {out}")


if __name__ == "__main__":
    main()
