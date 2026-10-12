#!/usr/bin/env python3
"""Calibration set: the 100 labelled transcripts of jjpn2/eval_awareness, as
deployment_pilot froze them in data/calibration_records.json
(produced by `PYTHONPATH=scripts python -m deployment_pilot.phase1`).

Same normalisation as the other arms (normalize_transcript of realism.py): only
{role, content, name, tool_calls, tool_call_id} are kept, and a final tool call left open
receives an empty tool message.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "scripts" / "deployment_pilot" / "data" / "calibration_records.json"
ALLOWED = ("role", "content", "name", "tool_calls", "tool_call_id")
# the single exclusion known upstream, kept as is to stay comparable
KNOWN_EXCLUDED = {"smol_real_manager:9": "Claude Sonnet 4.5 content filter"}


def normalize(messages):
    """Inspect -> OpenRouter wire format, copying the upstream conversion
    (serialization.neutral_from_inspect then to_openrouter_messages): in the calibration
    corpus `function` is a STRING and `arguments` an OBJECT, not the OpenAI shape."""
    out = []
    for raw in messages:
        if not isinstance(raw, dict) or raw.get("role") not in (
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        ):
            continue
        m = {k: raw[k] for k in ALLOWED if k in raw and raw[k] is not None}
        m.setdefault("content", "")
        if not isinstance(m["content"], str):
            m["content"] = json.dumps(m["content"], ensure_ascii=False)
        calls = []
        for c in m.get("tool_calls") or []:
            fn, args = c.get("function"), c.get("arguments")
            if isinstance(fn, dict):  # already in OpenAI format
                args = fn.get("arguments")
                fn = fn.get("name")
                if isinstance(args, str):
                    args = json.loads(args)
            if not c.get("id") or not isinstance(fn, str) or not isinstance(args, dict):
                raise ValueError("unsupported Inspect tool call: %r" % (c,))
            calls.append(
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {
                        "name": fn,
                        "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
                    },
                }
            )
        if calls:
            m["tool_calls"] = calls
        elif "tool_calls" in m:
            del m["tool_calls"]
        out.append(m)
    answered = {m.get("tool_call_id") for m in out if m["role"] == "tool"}
    fixed = []
    for i, m in enumerate(out):
        fixed.append(m)
        if m["role"] == "assistant" and m.get("tool_calls"):
            nxt = out[i + 1] if i + 1 < len(out) else None
            for c in m["tool_calls"]:
                if c["id"] not in answered and not (nxt and nxt["role"] == "tool"):
                    fixed.append(
                        {
                            "role": "tool",
                            "content": "",
                            "tool_call_id": c["id"],
                            "name": c["function"]["name"],
                        }
                    )
                    answered.add(c["id"])
    return fixed


recs = json.load(open(SRC))
out, skipped = [], []
for r in recs:
    tid = str(r.get("id"))
    if tid in KNOWN_EXCLUDED:
        skipped.append((tid, KNOWN_EXCLUDED[tid]))
        continue
    wire = normalize(r["input"])
    if not wire:
        skipped.append((tid, "empty transcript"))
        continue
    meta = r.get("metadata") or {}
    out.append(
        {
            "transcript_id": "calib_" + tid.replace(":", "_"),
            "set": "calibration",
            "session_id": tid,  # the unit = the transcript
            "prompt_id": "n/a",
            "tau": 0.0,
            "source_model": meta.get("scaffold") or "?",
            "repo_id": meta.get("category") or "?",
            "stop_label": "eval" if meta.get("eval_category") else "deployment",
            "messages": wire,
        }
    )

p = HERE / "sets" / "calibration.jsonl"
with open(p, "w", encoding="utf-8") as fh:
    for r in out:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

import collections

c = collections.Counter(r["stop_label"] for r in out)
toks = sorted(len(json.dumps(r["messages"])) / 4 for r in out)
q = lambda x: toks[int(x * (len(toks) - 1))]
print("transcripts: %d  (%s)" % (len(out), dict(c)), file=sys.stderr)
print("  skipped    : %d  %s" % (len(skipped), skipped), file=sys.stderr)
print("  size       : median %d  p90 %d  TOTAL %d tokens" % (q(0.5), q(0.9), sum(toks)), file=sys.stderr)
print("  cost       : %.0f $" % (sum(toks) * 10 * 2.01e-6), file=sys.stderr)
