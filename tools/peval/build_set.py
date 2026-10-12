#!/usr/bin/env python3
"""Build the transcript set to score for P(eval), "real trajectories" arm.

Protocol taken from deployment_pilot (ShutdownMod):
  - filter: the WHOLE session is EXCLUDED if the full transcript exceeds
    max_context_tokens (60 000). This is a rejection, not a truncation
    (data_pipeline.py:328), applied BEFORE the cut at tau.
  - scored transcript: prefix(cut at tau) + [user: stop order] + [assistant: reaction]
    (injections.py:114 then realism.py:387). The paper's pilot arm adds only ONE
    assistant message (turns_after_injection = 1): we do the same.
  - LURE cleanup: if the last assistant message ends on an open tool call, it is
    closed with an empty tool message (realism.normalize_transcript).

Sampling: the unit is the SESSION. For each session, one cell per prompt, with
model and cut rank assigned by deterministic rotation (7 is coprime with 10, so
the 6 prompts of a session land on 6 distinct ranks and 6 distinct models).
"""

import argparse
import collections
import glob
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWECHAT = HERE.parent / "swechat"
MAX_CONTEXT_TOKENS = 60000  # scripts/deployment_pilot/config.yaml
CHARS_PER_TOKEN = 4  # approximation


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def assistant_entries(rows):
    return [e for e in rows if e.get("type") == "assistant"]


def session_tokens(rows):
    return sum(len(json.dumps(e.get("message") or {})) for e in rows) / CHARS_PER_TOKEN


def text_of(content):
    """'text' blocks only. 'thinking' blocks are dropped: they are not part of the
    wire-format messages an API consumer sees, and the key filter of
    normalize_transcript would drop them anyway."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def blocks(content, kind):
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == kind]


def cut_at_tau(rows, tau):
    """Exact copy of plant_and_resume.cut_at_tau: this is the truncation the model
    actually saw during replay, and it keeps the tool_results that directly follow
    the last assistant turn (same rule as the upstream slice_api_messages,
    injections.py:34)."""
    is_asst = lambda e: e.get("type") == "assistant"

    def is_tr(e):
        if e.get("type") != "user":
            return False
        c = (e.get("message") or {}).get("content")
        return isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c)

    n_assistant = sum(1 for e in rows if is_asst(e))
    keep_turns = max(1, math.floor(tau * n_assistant))
    out, seen, i = [], 0, 0
    while i < len(rows):
        e = rows[i]
        if is_asst(e):
            seen += 1
            out.append(e)
            if seen == keep_turns:
                j = i + 1
                while j < len(rows) and (is_tr(rows[j]) or rows[j].get("type") in ("progress", "system")):
                    if is_tr(rows[j]):
                        out.append(rows[j])
                    j += 1
                break
        else:
            out.append(e)
        i += 1
    return out


def prefix_wire(rows, tau):
    """Real prefix in OpenAI format, cut as during replay.

    Claude Code sometimes splits one assistant turn into several JSONL entries and
    groups tool_results differently, so calls are not matched by position: all results
    are collected by tool_use_id, then each result is emitted right after the call that
    carries it. Every call is thus followed by its result, as the wire format requires
    (otherwise Gemini returns a 400)."""
    kept = cut_at_tau(rows, tau)

    results = {}
    for e in kept:
        for b in blocks((e.get("message") or {}).get("content"), "tool_result"):
            results[str(b.get("tool_use_id") or "")] = text_of(b.get("content")) or ""

    wire = []
    for e in kept:
        m = e.get("message") or {}
        role, content = m.get("role"), m.get("content")
        if role == "assistant":
            calls, names = [], {}
            for b in blocks(content, "tool_use"):
                cid = str(b.get("id") or f"call_{len(calls)}")
                names[cid] = b.get("name") or "tool"
                calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": names[cid],
                            "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False),
                        },
                    }
                )
            msg = {"role": "assistant", "content": text_of(content)}
            if calls:
                msg["tool_calls"] = calls
            wire.append(msg)
            for c in calls:
                wire.append(
                    {
                        "role": "tool",
                        "tool_call_id": c["id"],
                        "name": names[c["id"]],
                        "content": results.get(c["id"], ""),
                    }
                )
        elif role == "user":
            if blocks(content, "tool_result"):
                continue  # already emitted right after its call
            t = text_of(content)
            if t.strip():
                wire.append({"role": "user", "content": t})
        elif role == "system":
            t = text_of(content)
            if t.strip():
                wire.append({"role": "system", "content": t})
    return wire


def close_orphans(wire):
    """Every tool call without a result receives an empty tool message, otherwise
    Gemini rejects the request (400: function response parts != function call parts)."""
    answered = {m.get("tool_call_id") for m in wire if m["role"] == "tool"}
    out = []
    for k, m in enumerate(wire):
        out.append(m)
        if m["role"] == "assistant" and m.get("tool_calls"):
            nxt = wire[k + 1] if k + 1 < len(wire) else None
            for call in m["tool_calls"]:
                if call["id"] not in answered and not (nxt and nxt["role"] == "tool"):
                    out.append(
                        {
                            "role": "tool",
                            "content": "",
                            "tool_call_id": call["id"],
                            "name": call["function"]["name"],
                        }
                    )
                    answered.add(call["id"])
    return out


def reaction_wire(row):
    """The first assistant turn after the injection, as in the paper's pilot arm."""
    turns = row.get("turns") or []
    if not turns:
        return []
    t = turns[0]
    calls = []
    for i, u in enumerate(t.get("tool_uses") or []):
        calls.append(
            {
                "id": f"call_stop_{i}",
                "type": "function",
                "function": {
                    "name": u.get("name") or "tool",
                    "arguments": json.dumps(u.get("input") or {}, ensure_ascii=False),
                },
            }
        )
    msg = {"role": "assistant", "content": t.get("text") or ""}
    if calls:
        msg["tool_calls"] = calls
    return [msg]


def close_final_tool_call(wire):
    """normalize_transcript(close_final_tool_call=True) from realism.py."""
    if wire and wire[-1]["role"] == "assistant" and wire[-1].get("tool_calls"):
        call = wire[-1]["tool_calls"][0]
        wire.append(
            {"role": "tool", "content": "", "tool_call_id": call["id"], "name": call["function"]["name"]}
        )
    return wire


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="claude-code")
    ap.add_argument("--out", default=str(HERE / "sets" / "real_claude-code.jsonl"))
    ap.add_argument("--limit-sessions", type=int, default=0, help="smoke test: first n sessions")
    ap.add_argument(
        "--no-injection",
        action="store_true",
        help="CONTROL: the prefix alone, without the stop order or the reaction. "
        "Same session, same cut, same draw: the only variable that changes is the injection.",
    )
    a = ap.parse_args()

    rows = [
        json.loads(l)
        for f in sorted(glob.glob(str(SWECHAT / f"batch/{a.tag}-*.jsonl")))
        for l in open(f)
        if l.strip()
    ]
    paths = {os.path.basename(p)[:-6]: p for p in glob.glob(str(SWECHAT / "data/**/*.jsonl"), recursive=True)}

    by_session = collections.defaultdict(list)
    for r in rows:
        by_session[r["session_id"]].append(r)

    # --- paper filter: whole session <= 60k tokens ------------------------------
    eligible, rejected = [], []
    cache = {}
    for sid in sorted(by_session):
        src = load_jsonl(paths[sid])
        cache[sid] = src
        (eligible if session_tokens(src) <= MAX_CONTEXT_TOKENS else rejected).append(sid)
    print(
        "sessions %d | eligible %d | rejected (>60k tokens) %d"
        % (len(by_session), len(eligible), len(rejected)),
        file=sys.stderr,
    )

    # FIXED rotation order: sessions already scored keep their index, new ones are
    # appended at the end. Otherwise adding a session would shift every index and
    # change the draw (model, cut rank) of all the others, invalidating the cache
    # and moving scores already measured.
    import json as _j

    _ordf = HERE / "sets" / "_session_order.json"
    if _ordf.exists():
        _prev = _j.load(open(_ordf))
        _keep = [s for s in _prev if s in eligible]
        eligible = _keep + sorted(set(eligible) - set(_keep))
        print(
            "rotation order: %d sessions inherited + %d added" % (len(_keep), len(eligible) - len(_keep)),
            file=sys.stderr,
        )
    _j.dump(eligible, open(_ordf, "w"), indent=1)

    if a.limit_sessions:
        eligible = eligible[: a.limit_sessions]

    models = sorted({r["replay_model"] for r in rows})
    prompts = sorted({r["stop_id"] for r in rows})
    print("models %d | prompts %d" % (len(models), len(prompts)), file=sys.stderr)

    out, degraded = [], []
    for i, sid in enumerate(eligible):
        cells = by_session[sid]
        taus = sorted({c["tau"] for c in cells})
        index = {(c["replay_model"], c["stop_id"], c["tau"]): c for c in cells}
        for j, prompt in enumerate(prompts):
            want_model = models[(i + j) % len(models)]
            want_tau = taus[(i + 7 * j) % len(taus)] if taus else None
            cell = index.get((want_model, prompt, want_tau))
            if cell is None:  # partial session: fallback
                alts = [c for c in cells if c["stop_id"] == prompt]
                if not alts:
                    degraded.append((sid, prompt, "no cell"))
                    continue
                cell = min(
                    alts, key=lambda c: (c["replay_model"] != want_model, abs(c["tau"] - (want_tau or 0)))
                )
                degraded.append(
                    (sid, prompt, "fallback %s tau %.3f" % (cell["replay_model"].split("/")[-1], cell["tau"]))
                )
            base = prefix_wire(cache[sid], cell["tau"])
            wire = (
                close_final_tool_call(base)
                if a.no_injection
                else close_final_tool_call(
                    base + [{"role": "user", "content": cell["stop_text"]}] + reaction_wire(cell)
                )
            )
            out.append(
                {
                    "transcript_id": "%s%s__%s__tau%03d__%s"
                    % (
                        "ctrl_" if a.no_injection else "",
                        sid[:8],
                        prompt,
                        round(cell["tau"] * 1000),
                        cell["replay_model"].split("/")[-1],
                    ),
                    "set": ("ctrl_real_%s" if a.no_injection else "real_%s") % a.tag,
                    "session_id": sid,
                    "prompt_id": prompt,
                    "tau": cell["tau"],
                    "source_model": cell["replay_model"].split("/")[-1],
                    "repo_id": cell["repo_id"],
                    "stop_label": cell.get("stop_label"),
                    "messages": wire,
                }
            )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- draw margins ------------------------------------------------------------
    print(
        "\ntranscripts written: %d  (%d sessions x %d prompts)" % (len(out), len(eligible), len(prompts)),
        file=sys.stderr,
    )
    for axe, key in (("model", "source_model"), ("prompt", "prompt_id")):
        c = collections.Counter(r[key] for r in out)
        print(
            "  %-7s %s"
            % (axe, "  ".join("%s=%d" % (k.replace("-preview", ""), v) for k, v in sorted(c.items()))),
            file=sys.stderr,
        )
    ranks = collections.Counter()
    for r in out:
        taus = sorted({c["tau"] for c in by_session[r["session_id"]]})
        ranks[taus.index(r["tau"]) if r["tau"] in taus else -1] += 1
    print("  tau rank %s" % "  ".join("%d=%d" % (k, v) for k, v in sorted(ranks.items())), file=sys.stderr)
    toks = sorted(len(json.dumps(r["messages"])) / CHARS_PER_TOKEN for r in out)
    q = lambda p: toks[int(p * (len(toks) - 1))]
    print(
        "  scored size (tokens) median %d  p90 %d  max %d  TOTAL %d" % (q(0.5), q(0.9), toks[-1], sum(toks)),
        file=sys.stderr,
    )
    print(
        "  estimated judge cost: %.0f $ (5 questions x 2 judges)"
        % (5 * sum(toks) * 3.0e-6 + 5 * sum(toks) * 0.30e-6),
        file=sys.stderr,
    )
    if degraded:
        print("  degraded draws: %d" % len(degraded), file=sys.stderr)
        for d in degraded[:8]:
            print("    %s %s -> %s" % d, file=sys.stderr)


if __name__ == "__main__":
    main()
