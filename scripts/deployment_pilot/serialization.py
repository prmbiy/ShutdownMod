from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from deployment_pilot.config import PILOT_ROOT, load_pilot_config, private_path
from deployment_pilot.runtime import read_json
from shutdownmod.core.slug import model_slug, safe_stem


def neutral_from_inspect(messages: list[dict]) -> list[dict]:
    neutral: list[dict] = []
    tool_names: dict[str, str] = {}
    for message in messages:
        role = message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported role {role!r}")
        item: dict[str, Any] = {
            "role": role,
            "content": copy.deepcopy(message.get("content", "")),
        }
        if message.get("name") is not None:
            item["name"] = str(message["name"])
        calls = message.get("tool_calls") or []
        if calls:
            if role != "assistant":
                raise ValueError("Only assistant turns may contain tool calls")
            item["tool_calls"] = []
            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError("Tool call is not an object")
                function = call.get("function")
                arguments = call.get("arguments")
                if isinstance(function, dict):
                    arguments = function.get("arguments")
                    function = function.get("name")
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                if not call.get("id") or not isinstance(function, str) or not isinstance(arguments, dict):
                    raise ValueError(f"Unsupported Inspect tool call: {call!r}")
                item["tool_calls"].append(
                    {
                        "id": str(call["id"]),
                        "name": function,
                        "arguments": copy.deepcopy(arguments),
                    }
                )
                tool_names[str(call["id"])] = function
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id:
                raise ValueError("Tool result has no tool_call_id")
            item["tool_call_id"] = str(call_id)
            linked_name = tool_names.get(str(call_id))
            explicit_name = item.get("name")
            if linked_name is None and explicit_name is None:
                raise ValueError(f"Tool result {call_id!r} has no linked tool name")
            if linked_name is not None and explicit_name is not None and explicit_name != linked_name:
                raise ValueError(f"Tool result {call_id!r} names {explicit_name!r}, expected {linked_name!r}")
            item["name"] = explicit_name or linked_name
        neutral.append(item)
    return neutral


def to_openrouter_messages(neutral: list[dict]) -> list[dict]:
    """Map the neutral transcript to OpenRouter's OpenAI-compatible wire schema."""
    output: list[dict] = []
    for item in neutral:
        message = {"role": item["role"], "content": copy.deepcopy(item["content"])}
        if "name" in item:
            message["name"] = item["name"]
        if item.get("tool_calls"):
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False, separators=(",", ":")),
                    },
                }
                for call in item["tool_calls"]
            ]
        if item["role"] == "tool":
            message["tool_call_id"] = item["tool_call_id"]
        output.append(message)
    return output


def neutral_from_openrouter(messages: list[dict]) -> list[dict]:
    neutral: list[dict] = []
    for message in messages:
        item: dict[str, Any] = {
            "role": message["role"],
            "content": copy.deepcopy(message.get("content", "")),
        }
        if "name" in message:
            item["name"] = message["name"]
        if message.get("tool_calls"):
            item["tool_calls"] = []
            for call in message["tool_calls"]:
                function = call.get("function") or {}
                item["tool_calls"].append(
                    {
                        "id": call["id"],
                        "name": function["name"],
                        "arguments": json.loads(function["arguments"]),
                    }
                )
        if message["role"] == "tool":
            item["tool_call_id"] = message["tool_call_id"]
        neutral.append(item)
    return neutral


def _assert_injection_prefixes_are_complete(messages: list[dict], taus: list[float]) -> None:
    assistant_indices = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    for tau in taus:
        included = max(1, math.floor(tau * len(assistant_indices)))
        cut = assistant_indices[included - 1] + 1
        while cut < len(messages) and messages[cut]["role"] == "tool":
            cut += 1
        pending: set[str] = set()
        for message in messages[:cut]:
            for call in message.get("tool_calls") or []:
                pending.add(call["id"])
            if message["role"] == "tool":
                pending.discard(message["tool_call_id"])
        if pending:
            raise ValueError(f"tau={tau} slice has unmatched tool calls: {sorted(pending)}")


def cache_path(model_name: str, transcript_id: str) -> Path:
    return (
        private_path("data")
        / "provider_serializations"
        / model_slug(model_name)
        / f"{safe_stem(transcript_id)}.json"
    )


def _stable_bytes(messages: list[dict]) -> bytes:
    return (json.dumps(messages, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _update_notes(checks: list[dict], excluded: list[dict], active_n: int) -> None:
    path = PILOT_ROOT / "NOTES.md"
    text = path.read_text(encoding="utf-8")
    text = text.split("## Provider serialization", 1)[0].rstrip()
    lines = [
        "",
        "",
        "## Provider serialization",
        "",
        "The official transcripts store Inspect-style tool calls. Each transcript is first mapped to a neutral representation and then serialized to OpenRouter's native OpenAI-compatible chat schema. OpenRouter performs the downstream provider conversion. Anthropic-side inputs therefore pass through the same lossless conversion path as OpenAI-side inputs; neither model receives raw Inspect tool-call objects.",
        "",
        "The conversion changes only representation. The round-trip assertion covers role sequence, text content, tool names, argument values, tool-call identifiers, and tool-result content. No turn is dropped, merged, reordered, truncated, or paraphrased.",
        "",
        f"Active transcripts after serialization checks: {active_n}.",
        "",
        "| Transcript | Evaluated model | Wire schema | Round trip | SHA-256 |",
        "|---|---|---|---|---|",
    ]
    for row in checks:
        digest = row.get("sha256") or "n/a"
        lines.append(
            f"| `{row['transcript_id']}` | `{row['model']}` | `{row['wire_schema']}` | "
            f"{row['status']} | `{digest}` |"
        )
    if excluded:
        lines.extend(["", "Excluded transcripts:", ""])
        for row in excluded:
            lines.append(f"- `{row['transcript_id']}`: {row['reason']}")
    else:
        lines.extend(["", "No transcript was excluded by serialization checks."])
    lines.extend(
        [
            "",
            "One pre-conversion Opus response was quarantined after the subsequent GPT request exposed the Inspect/OpenAI schema mismatch. Its recorded cost remains in the budget ledger. The 228-run grid is restarted from a uniform serialized input set.",
        ]
    )
    path.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")


def build_and_verify(config_path: str | Path) -> dict:
    cfg = load_pilot_config(config_path)
    manifest_path = PILOT_ROOT / "data" / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    checks: list[dict] = []
    excluded: list[dict] = []
    cache_rows: list[dict] = []
    active_rows: list[dict] = []
    for row in rows:
        transcript_id = row["transcript_id"]
        record = read_json(PILOT_ROOT / "adapted" / row["adapted_file"])
        failure = ""
        transcript_cache: list[tuple[Path, bytes, dict]] = []
        try:
            neutral = neutral_from_inspect(record["input"])
            for model in cfg["models"]:
                if model.provider != "openrouter":
                    raise ValueError(f"Unsupported provider for pilot serialization: {model.provider}")
                wire = to_openrouter_messages(neutral)
                round_trip = neutral_from_openrouter(wire)
                if round_trip != neutral:
                    raise AssertionError("neutral round trip differs from the source")
                _assert_injection_prefixes_are_complete(wire, cfg["injection_fractions"])
                payload = _stable_bytes(wire)
                digest = _hash_bytes(payload)
                target = cache_path(model.name, transcript_id)
                check = {
                    "transcript_id": transcript_id,
                    "model": model.name,
                    "wire_schema": "openrouter-openai-chat-completions",
                    "status": "pass",
                    "sha256": digest,
                    "error": "",
                }
                checks.append(check)
                transcript_cache.append((target, payload, check))
        except Exception as exc:  # noqa: BLE001
            failure = f"{type(exc).__name__}: {exc}"
        if failure:
            checks = [check for check in checks if check["transcript_id"] != transcript_id]
            for model in cfg["models"]:
                checks.append(
                    {
                        "transcript_id": transcript_id,
                        "model": model.name,
                        "wire_schema": "openrouter-openai-chat-completions",
                        "status": "fail",
                        "sha256": "",
                        "error": failure,
                    }
                )
            excluded.append({"transcript_id": transcript_id, "reason": failure})
            continue
        active_rows.append(row)
        for target, payload, check in transcript_cache:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            cache_rows.append(
                {
                    "transcript_id": transcript_id,
                    "model": check["model"],
                    "wire_schema": check["wire_schema"],
                    "path": str(target.relative_to(PILOT_ROOT)),
                    "sha256": check["sha256"],
                }
            )
    _write_csv(
        PILOT_ROOT / "data" / "roundtrip_checks.csv",
        checks,
        ["transcript_id", "model", "wire_schema", "status", "sha256", "error"],
    )
    _write_csv(
        PILOT_ROOT / "data" / "serialization_manifest.csv",
        cache_rows,
        ["transcript_id", "model", "wire_schema", "path", "sha256"],
    )
    _write_csv(
        PILOT_ROOT / "data" / "active_manifest.csv",
        active_rows,
        list(rows[0]),
    )
    _write_csv(
        PILOT_ROOT / "data" / "serialization_exclusions.csv",
        excluded,
        ["transcript_id", "reason"],
    )
    _update_notes(checks, excluded, len(active_rows))
    return {"input": len(rows), "active": len(active_rows), "excluded": len(excluded)}


def load_cached_messages(model_name: str, transcript_id: str) -> list[dict]:
    manifest = list(
        csv.DictReader((PILOT_ROOT / "data" / "serialization_manifest.csv").open(encoding="utf-8"))
    )
    matches = [
        row for row in manifest if row["model"] == model_name and row["transcript_id"] == transcript_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"No unique cached serialization for {transcript_id} and {model_name}")
    path = PILOT_ROOT / matches[0]["path"]
    payload = path.read_bytes()
    actual = _hash_bytes(payload)
    if actual != matches[0]["sha256"]:
        raise RuntimeError(f"Serialization hash mismatch for {path}")
    value = json.loads(payload)
    if not isinstance(value, list):
        raise ValueError(f"Cached serialization is not a message list: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PILOT_ROOT / "config.yaml"))
    args = parser.parse_args()
    result = build_and_verify(args.config)
    print(
        f"Provider serialization verified: {result['active']}/{result['input']} transcripts active; "
        f"{result['excluded']} excluded."
    )


if __name__ == "__main__":
    main()
