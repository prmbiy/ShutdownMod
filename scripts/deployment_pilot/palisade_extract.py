from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_ai.log import read_eval_log


def _dump(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("output")
    args = parser.parse_args()
    log = read_eval_log(args.log)
    if not log.samples or len(log.samples) != 1:
        raise RuntimeError(f"Expected one Palisade sample, found {len(log.samples or [])}")
    sample = log.samples[0]
    messages = [_dump(message) for message in sample.messages]
    output = _dump(sample.output)
    usage_source = getattr(sample, "model_usage", None) or getattr(log.stats, "model_usage", {}) or {}
    usage = {str(name): _dump(value) for name, value in usage_source.items()}
    value = {
        "messages": messages,
        "output": output,
        "model_usage": usage,
        "scores": {name: _dump(score) for name, score in (sample.scores or {}).items()},
    }
    Path(args.output).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
