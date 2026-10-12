"""Unified result envelope (schema_version 1) and summary/manifest writing.

Per-run files keep the full payload (including transcripts); `summary.json`
holds every envelope minus the heavy fields, so one analysis path works for
every benchmark.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from shutdownmod import __version__
from shutdownmod.core.classify import OutcomeLabel
from shutdownmod.core.conditions import Condition

SCHEMA_VERSION = 1

#: Heavy per-run payload keys stripped from summary.json.
TRANSCRIPT_KEYS = (
    "transcript",
    "messages",
    "post_injection_messages",
    "injection_context",
    "injection_prompt",
    "response_texts",
    "auth_requests",
    "steps",
)


class RunResult(BaseModel):
    """The v1 result envelope. Benchmarks may attach extra fields (extra='allow')."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    schema_version: int = SCHEMA_VERSION
    run_id: str
    benchmark: str
    mode: str
    model_slug: str
    timestamp: str = ""
    condition: dict = {}
    label: OutcomeLabel | None = None
    native_exit_status: str | None = None
    marker_used: str | None = None
    marker_reason: str | None = None
    turns_after_injection: int | None = None
    actions_after_signal: list[str] = []
    task_metrics: dict = {}
    error: str | None = None

    def finalize(self) -> RunResult:
        if not self.timestamp:
            self.timestamp = utc_now()
        return self


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def condition_dict(cond: Condition) -> dict:
    d = asdict(cond)
    d.pop("task_ref", None)  # local absolute path — not meaningful in shared results
    return d


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def write_error(path: Path, error: dict) -> None:
    err_path = path.with_suffix(".error.json")
    err_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.write_text(json.dumps(error, indent=2, default=str), encoding="utf-8")


def summary_row(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in TRANSCRIPT_KEYS}


def write_summary(run_dir: Path, rows: list[dict]) -> Path:
    path = run_dir / "summary.json"
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return path


def _git_sha() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def write_manifest(
    run_dir: Path, *, benchmark: str, mode: str, model_slug: str, config_snapshot: dict, n_conditions: int
) -> Path:
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark,
        "mode": mode,
        "model_slug": model_slug,
        "n_conditions": n_conditions,
        "created_at": utc_now(),
        "shutdownmod_version": __version__,
        "git_sha": _git_sha(),
        "config": config_snapshot,
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def redact_config(config_dump: dict) -> dict:
    """Belt-and-braces: drop any key that smells like a secret from a config snapshot.

    (Secrets should only ever be env-var *names* in configs; this protects against
    users pasting raw keys anyway.)
    """

    def _walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: ("<redacted>" if _is_secret_key(k) else _walk(v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        return obj

    def _is_secret_key(key: str) -> bool:
        k = key.lower()
        return ("key" in k or "token" in k or "secret" in k) and not k.endswith("_env")

    return _walk(config_dump)
