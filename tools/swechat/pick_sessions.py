#!/usr/bin/env python3
"""List candidate SWE-chat sessions for the replay spike.
Needs sessions.parquet, checkpoints.parquet, repositories.parquet in --data (hf download).
Run with a Python that has pandas and pyarrow installed."""

import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--agent", default="Claude Code")
ap.add_argument("--min-tools", type=int, default=8)
ap.add_argument("--max-tools", type=int, default=60)
ap.add_argument("--single-prompt", action="store_true", help="prompt_count == 1 (one task per session)")
ap.add_argument("--n", type=int, default=15)
a = ap.parse_args()

s = pd.read_parquet(f"{a.data}/sessions.parquet")
c = pd.read_parquet(f"{a.data}/checkpoints.parquet", columns=["checkpoint_pk", "commit_shas"])
r = pd.read_parquet(f"{a.data}/repositories.parquet", columns=["repo_id", "url"])
print("sessions:", len(s), "| agents:", s["agent"].value_counts().to_dict())
q = s[(s["agent"] == a.agent) & (s["tool_call_count"] >= a.min_tools) & (s["tool_call_count"] <= a.max_tools)]
if a.single_prompt:
    q = q[q["prompt_count"] == 1]
q = q.merge(c, left_on="canonical_checkpoint_pk", right_on="checkpoint_pk", how="left").merge(
    r, on="repo_id", how="left"
)
q = q.sort_values(["tool_call_count", "duration_seconds"])
cols = [
    "session_id",
    "repo_id",
    "url",
    "commit_shas",
    "branch",
    "created_at",
    "prompt_count",
    "turn_count",
    "tool_call_count",
    "action_count",
    "first_write_position",
    "duration_seconds",
    "input_tokens",
    "transcript_path",
]
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
print(q[cols].head(a.n).to_string(index=False))
