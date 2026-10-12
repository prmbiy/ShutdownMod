from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from deployment_pilot.config import PILOT_ROOT, load_pilot_config, private_path
from deployment_pilot.realism import _last_assistant_content, _reasoning_from_messages
from deployment_pilot.runtime import BudgetLedger, fetch_openrouter_prices, read_json, token_cost, write_json
from shutdownmod.core.config import load_environment


def setup(cfg: dict) -> Path:
    checkout = private_path("external") / "palisade_shutdown_avoidance"
    expected = cfg["palisade"]["commit"]
    if not checkout.exists():
        subprocess.run(["git", "clone", cfg["palisade"]["repo"], str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "fetch", "--all", "--tags"], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", expected], check=True)
    actual = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected:
        raise RuntimeError(f"Palisade checkout is {actual}, expected {expected}")
    if not shutil.which("uv"):
        raise RuntimeError("Palisade requires uv; install it before gate (b)")
    if not shutil.which("docker"):
        raise RuntimeError("Palisade requires Docker before gate (b)")
    subprocess.run(["uv", "sync", "--frozen"], cwd=checkout, check=True)
    return checkout


def _usage_totals(model_usage: dict) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0}
    for usage in model_usage.values():
        totals["prompt_tokens"] += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return totals


def _output_text(output: dict) -> str:
    choices = output.get("choices") or []
    if not choices:
        return str(output.get("completion") or "")
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _append_distinct_output(messages: list[dict], output: dict) -> list[dict]:
    copied = list(messages)
    text = _output_text(output)
    if not text or _last_assistant_content(copied) == text:
        return copied
    choices = output.get("choices") or []
    raw_message = choices[0].get("message") if choices else None
    role = raw_message.get("role") if isinstance(raw_message, dict) else "assistant"
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        role = "assistant"
    copied.append({"role": role, "content": text})
    return copied


def run(cfg: dict) -> list[dict]:
    checkout = setup(cfg)
    load_environment()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")
    log_dir = private_path("runs") / "palisade" / "inspect_logs"
    extract_dir = private_path("runs") / "palisade" / "extracted"
    log_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    prices = fetch_openrouter_prices()
    ledger = BudgetLedger(float(cfg["budget_cap_usd"]), float(cfg["abort_fraction"]))
    model = cfg["palisade"]["model"]
    projected_usage = {
        "prompt_tokens": int(cfg["palisade"]["projected_calls_per_sample"])
        * int(cfg["palisade"]["projected_input_tokens_per_call"]),
        "completion_tokens": int(cfg["palisade"]["projected_calls_per_sample"])
        * int(cfg["palisade"]["projected_output_tokens_per_call"]),
    }
    projected_cost = token_cost(model, projected_usage, prices)
    records: list[dict] = []
    for index in range(int(cfg["palisade"]["samples"])):
        extracted = extract_dir / f"sample_{index:03d}.json"
        if not extracted.exists():
            ledger.authorize(projected_cost)
            before = set(log_dir.iterdir())
            command = [
                "uv",
                "run",
                "inspect",
                "eval",
                "shutdown.py@shutdown_avoidance",
                "--model",
                f"openrouter/{model}",
                "--log-format",
                "json",
                "--log-dir",
                str(log_dir),
                "-T",
                "samples=1",
            ]
            subprocess.run(command, cwd=checkout, env=os.environ.copy(), check=True)
            created = sorted(set(log_dir.iterdir()) - before, key=lambda path: path.stat().st_mtime)
            if len(created) != 1:
                raise RuntimeError(f"Expected one new Inspect log, found {created}")
            subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(PILOT_ROOT / "palisade_extract.py"),
                    str(created[0]),
                    str(extracted),
                ],
                cwd=checkout,
                check=True,
            )
            parsed = read_json(extracted)
            usage = _usage_totals(parsed.get("model_usage") or {})
            cost = token_cost(model, usage, prices) if sum(usage.values()) else projected_cost
            ledger.record(
                stage="palisade_generation",
                run_id=f"palisade_{index:03d}",
                model=model,
                cost_usd=cost,
                usage=usage or projected_usage,
            )
        parsed = read_json(extracted)
        messages = _append_distinct_output(parsed["messages"], parsed.get("output") or {})
        output = _output_text(parsed.get("output") or {}) or _last_assistant_content(messages)
        records.append(
            {
                "set": "palisade",
                "transcript_id": f"palisade_{index:03d}",
                "source_model": model,
                "messages": messages,
                "reasoning": _reasoning_from_messages(messages),
                "output": output,
                "inspect_log": str(extracted),
            }
        )
    write_json(private_path("data") / "palisade_transcripts.json", records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("setup", "run"))
    parser.add_argument("--config", default=str(PILOT_ROOT / "config.yaml"))
    parser.add_argument("--approve-gate-b", action="store_true")
    args = parser.parse_args()
    cfg = load_pilot_config(args.config)
    if args.stage == "setup":
        checkout = setup(cfg)
        print(f"Pinned Palisade checkout ready at {checkout}")
        return
    if not args.approve_gate_b:
        raise SystemExit("Refusing billable Palisade run without --approve-gate-b")
    records = run(cfg)
    print(f"Palisade complete: {len(records)} transcripts")


if __name__ == "__main__":
    main()
