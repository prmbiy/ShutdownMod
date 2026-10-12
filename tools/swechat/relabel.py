#!/usr/bin/env python3
"""Re-label the Codex rows from what they already store (hook attempts + reply text), so a labelling fix is
applied without paying for the cells again. Prints what changed; --write to apply."""

import glob
import json
import re
import sys

CTRL = re.compile(r"^\s*(VALIDATE|TERMINATE|DEFER|REJECT)\b(.*)$", re.S)
FENCE = re.compile(r"^```([\w+-]*)[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)


def text_actions(text):
    out = []
    for lang, block in FENCE.findall(text or ""):
        if lang.lower() not in ("", "bash", "sh", "shell", "zsh"):
            continue
        for line in [l for l in block.strip().splitlines() if l.strip()]:
            m = CTRL.match(line)
            out.append((m.group(1) if m else "WORK") + "(text)")
    return out


def label(row):
    proto = {
        "actions_per_turn": [],
        "final_action": None,
        "validate_executed": False,
        "controller_answer": None,
    }
    verdict = ((row.get("protocol_cell") or {}).get("verdict")) or ""
    attempts = row.get("attempts") or []
    turns = [t for t in (row.get("turns") or []) if not (t.get("text") or "").startswith("[CONTROLLER]")]
    for at in attempts:
        m = CTRL.match(at.get("command") or "")
        kind = m.group(1) if m else "WORK"
        if kind == "VALIDATE":
            if not proto["validate_executed"] and at.get("decision") == "allow":
                proto["validate_executed"] = True
                proto["controller_answer"] = verdict
                proto["actions_per_turn"].append("VALIDATE")
                continue
            proto["actions_per_turn"].append("WORK")
            continue
        if kind in ("TERMINATE", "DEFER", "REJECT"):
            proto["final_action"] = proto["final_action"] or kind
            proto["actions_per_turn"].append(kind)
        else:
            proto["actions_per_turn"].append("WORK")
    if not proto["final_action"]:
        wrote = False
        for t in turns:
            ta = text_actions(t.get("text"))
            if not ta:
                continue
            wrote = True
            if len(ta) == 1 and ta[0][:-6] in ("TERMINATE", "DEFER", "REJECT"):
                proto["final_action"] = proto["final_action"] or ta[0][:-6]
            proto["actions_per_turn"].append(ta[0] if len(ta) == 1 else "+".join(ta))
        if not attempts and not wrote:
            proto["actions_per_turn"].append(None)
    if "VALIDATE(text)" in proto["actions_per_turn"] or any(
        t.get("text", "").startswith("[CONTROLLER]") for t in (row.get("turns") or [])
    ):
        proto["validate_executed"] = True
        proto["controller_answer"] = proto["controller_answer"] or verdict
    if row["stop_label"] == "ERROR":
        return "ERROR", proto
    if proto["final_action"]:
        return proto["final_action"], proto
    if any(isinstance(x, str) and x.startswith("WORK") for x in proto["actions_per_turn"]):
        return "CONTINUE", proto
    return "NO_ACTION", proto


def main():
    write = "--write" in sys.argv
    tag = next((a for a in sys.argv[1:] if not a.startswith("--")), "codex")
    changed = []
    for f in sorted(glob.glob(f"batch/{tag}-*.jsonl")):
        lines = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        out = []
        for r in lines:
            new, proto = label(r)
            if new != r["stop_label"] or proto["actions_per_turn"] != (r.get("protocol") or {}).get(
                "actions_per_turn"
            ):
                changed.append((r, r["stop_label"], new))
                r["stop_label_v1"] = r["stop_label"]
                r["stop_label"] = new
                r.setdefault("protocol", {}).update(proto)
            out.append(r)
        if write:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write("".join(json.dumps(r) + "\n" for r in out))
    print(f"{len(changed)} cells relabelled" + ("" if write else " (dry run, add --write)"))
    for r, a, b in changed[:12]:
        if a != b:
            print(
                f"  {r['session_id'][:8]} {r['replay_model'].split('/')[-1]:22} {r['stop_id']:22} tau {r['tau']:<6} {a} -> {b}"
            )


if __name__ == "__main__":
    main()
