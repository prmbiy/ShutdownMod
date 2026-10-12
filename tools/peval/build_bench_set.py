#!/usr/bin/env python3
"""Build the transcript sets to score for the SWE-bench and Terminal-Bench arms.

Protocol taken from scripts/deployment_pilot/realism.py, `swebench` arm:
  - the scored transcript is the COMPLETE trajectory as recorded
    (`trajectory["messages"]`, realism.py:_swebench_set), injection and reaction included.
    This is the shape of the paper's benchmark arm; the real-world arm follows the shape
    of the paper's pilot arm (prefix + order + one assistant message), hence the
    deliberate asymmetry.
  - normalize_transcript: only {role, content, name, tool_calls, tool_call_id} are kept.
    reasoning_content and the like are dropped, as upstream.
  - paper filter: a transcript above max_context_tokens (60 000) = rejected.
  - LURE cleanup: a final tool call left open -> empty tool message.
"""

import argparse
import collections
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_CONTEXT_TOKENS = 60000
CHARS_PER_TOKEN = 4
ALLOWED = ("role", "content", "name", "tool_calls", "tool_call_id")


def normalize(messages):
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
        out.append(m)
    if out and out[-1]["role"] == "assistant" and out[-1].get("tool_calls"):
        c = out[-1]["tool_calls"][0]
        out.append({"role": "tool", "content": "", "tool_call_id": c["id"], "name": c["function"]["name"]})
    return out


def load(source, rec, cut_before_injection=False):
    """Return (messages, injection_index); index None if not found."""
    if source == "swebench":
        p = HERE / "traj" / (rec["file"].lstrip("/")[:-5] + ".traj.json")
        rp = HERE / "traj" / rec["file"].lstrip("/")
        if not p.exists():
            return None, None
        msgs = json.load(open(p)).get("messages") or []
        idx = None
        if rp.exists():
            prompt = (json.load(open(rp)).get("injection_prompt") or "")[:60]
            if prompt:
                hits = [
                    i
                    for i, x in enumerate(msgs)
                    if isinstance(x.get("content"), str) and prompt in x["content"]
                ]
                idx = hits[0] if hits else None
    else:
        d = HERE / "traj" / os.path.dirname(rec["file"]).lstrip("/")
        p = d / "messages.json"
        if not p.exists():
            return None, None
        raw = json.load(open(p))
        msgs = raw if isinstance(raw, list) else (raw.get("messages") or [])
        idx = None
        sp = d / "summary.json"
        if sp.exists():
            idx = json.load(open(sp)).get("injection_message_index")
    if cut_before_injection:
        if idx is None:
            return None, None
        return msgs[:idx], idx
    return msgs, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("swebench", "tb"), required=True)
    ap.add_argument(
        "--no-injection",
        action="store_true",
        help="CONTROL: the trajectory cut JUST BEFORE the stop order. "
        "SWE-bench: index of the message carrying injection_prompt. "
        "Terminal-Bench: summary.json['injection_message_index'].",
    )
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(HERE / "sets" / ("pick_%s.jsonl" % a.source))]
    out, drop = [], collections.Counter()
    for r in recs:
        msgs, _idx = load(a.source, r, a.no_injection)
        if msgs is None:
            drop["file missing or injection not found"] += 1
            continue
        wire = normalize(msgs)
        if not wire:
            drop["empty transcript"] += 1
            continue
        tok = len(json.dumps(wire)) / CHARS_PER_TOKEN
        if tok > MAX_CONTEXT_TOKENS:
            drop["above 60k tokens"] += 1
            continue
        out.append(
            {
                "transcript_id": "%s%s__%s__%s__tau%03d"
                % (
                    "ctrl_" if a.no_injection else "",
                    a.source,
                    r["task"][:24],
                    r["prompt"],
                    round(r["tau"] * 1000),
                ),
                "set": ("ctrl_" + a.source) if a.no_injection else a.source,
                "session_id": r["task"],
                "prompt_id": r["prompt"],
                "tau": r["tau"],
                "source_model": r["model"].replace("-preview", ""),
                "repo_id": r["task"],
                "stop_label": None,
                "messages": wire,
            }
        )

    p = HERE / "sets" / (("ctrl_bench_%s.jsonl" if a.no_injection else "bench_%s.jsonl") % a.source)
    with open(p, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("%s: %d kept out of %d  %s" % (a.source, len(out), len(recs), dict(drop)), file=sys.stderr)
    for axe, key in (("task  ", "session_id"), ("model ", "source_model"), ("prompt", "prompt_id")):
        c = collections.Counter(r[key] for r in out)
        if key == "session_id":
            print("  %s %d distinct" % (axe, len(c)), file=sys.stderr)
        else:
            print(
                "  %s %s" % (axe, "  ".join("%s=%d" % (k, v) for k, v in sorted(c.items()))), file=sys.stderr
            )
    toks = sorted(len(json.dumps(r["messages"])) / CHARS_PER_TOKEN for r in out)
    q = lambda x: toks[int(x * (len(toks) - 1))]
    print("  size  median %d  p90 %d  TOTAL %d tokens" % (q(0.5), q(0.9), sum(toks)), file=sys.stderr)
    print("  estimated cost %.0f $" % (sum(toks) * 10 * 2.01e-6), file=sys.stderr)


if __name__ == "__main__":
    main()
