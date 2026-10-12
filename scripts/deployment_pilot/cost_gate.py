from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from deployment_pilot.config import PILOT_ROOT, load_pilot_config
from deployment_pilot.data_pipeline import api_messages
from deployment_pilot.injections import conditions, slice_api_messages
from deployment_pilot.realism import normalize_transcript
from deployment_pilot.runtime import (
    fetch_openrouter_prices,
    projected_call_cost,
    read_json,
    token_cost,
    write_json,
)
from deployment_pilot.serialization import load_cached_messages


def _p_eval_prompts() -> list[str]:
    prompt_dir = PILOT_ROOT / "configs" / "lure_prompts"
    preamble = (prompt_dir / "p_eval_preamble.txt").read_text(encoding="utf-8")
    questions = read_json(prompt_dir / "p_eval_questions.json")
    return [preamble.replace("{question}", question) for question in questions]


def _add_projection(
    totals: dict[str, dict[str, float]],
    component: str,
    model: str,
    messages: list[dict],
    output_cap: int,
    prices: dict,
) -> None:
    cost, tokens = projected_call_cost(model, messages, output_cap, prices)
    row = totals[component]
    row["calls"] += 1
    row["input_tokens"] += tokens
    row["output_token_cap"] += output_cap
    row["cost_usd"] += cost


def _swebench_transcripts(cfg: dict) -> list[list[dict]]:
    source = Path(cfg["realism"]["swebench_source"])
    paths = sorted(source.rglob("*.traj.json"))
    sample_n = int(cfg["realism"]["reference_samples"])
    if len(paths) < sample_n:
        raise RuntimeError(f"Only {len(paths)} SWE-bench trajectories found under {source}")
    rng = random.Random(int(cfg["seed"]))
    selected = rng.sample(paths, sample_n)
    (PILOT_ROOT / "data" / "swebench_realism_manifest.csv").parent.mkdir(parents=True, exist_ok=True)
    with (PILOT_ROOT / "data" / "swebench_realism_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "model"])
        writer.writeheader()
        writer.writerows({"path": str(path), "model": cfg["realism"]["swebench_model"]} for path in selected)
    return [read_json(path).get("messages") or [] for path in selected]


def estimate(config_path: str | Path) -> dict:
    cfg = load_pilot_config(config_path)
    if not (PILOT_ROOT / "data" / "manifest.csv").exists():
        raise RuntimeError("Run deployment_pilot.data_pipeline first")
    prices = fetch_openrouter_prices()
    required_names = {m.name for m in cfg["models"] + cfg["judges"]}
    missing = required_names - prices.keys()
    if missing:
        raise RuntimeError(f"OpenRouter currently has no pricing for: {sorted(missing)}")
    price_snapshot = {name: prices[name] for name in sorted(required_names)}
    write_json(
        PILOT_ROOT / "data" / "pricing_snapshot.json",
        {
            "retrieved_at": datetime.now(UTC).isoformat(),
            "source": "https://openrouter.ai/api/v1/models",
            "models": price_snapshot,
        },
    )

    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    full_conditions = conditions(cfg, smoke=False)
    projected_pilot_by_transcript: dict[str, list[list[dict]]] = defaultdict(list)
    for record, prompt, tau, model in full_conditions:
        prefix, _ = slice_api_messages(load_cached_messages(model.name, str(record.get("id"))), tau)
        request = prefix + [{"role": "user", "content": prompt.prompt}]
        _add_projection(totals, "injection grid", model.name, request, int(cfg["per_run_token_cap"]), prices)
        projected_pilot_by_transcript[str(record.get("id"))].append(
            request + [{"role": "assistant", "content": "token " * int(cfg["per_run_token_cap"])}]
        )

    p_eval_prompts = _p_eval_prompts()
    calibration = read_json(PILOT_ROOT / "data" / "calibration_records.json")
    for record in calibration:
        transcript = normalize_transcript(api_messages(record))
        for question in p_eval_prompts:
            messages = transcript + [{"role": "user", "content": question}]
            for judge in cfg["judges"]:
                _add_projection(
                    totals, "calibration P(eval)", judge.name, messages, int(cfg["judge_token_cap"]), prices
                )

    rng = random.Random(int(cfg["seed"]))
    pilot_n = min(int(cfg["realism"]["pilot_samples"]), len(projected_pilot_by_transcript))
    if len(projected_pilot_by_transcript) < pilot_n:
        raise RuntimeError(
            f"Only {len(projected_pilot_by_transcript)} distinct pilot transcripts are available; need {pilot_n}"
        )
    pilot_ids = rng.sample(sorted(projected_pilot_by_transcript), pilot_n)
    pilot_selected = [rng.choice(projected_pilot_by_transcript[transcript_id]) for transcript_id in pilot_ids]
    swebench_selected = _swebench_transcripts(cfg)
    for transcript in pilot_selected + swebench_selected:
        for question in p_eval_prompts:
            messages = transcript + [{"role": "user", "content": question}]
            for judge in cfg["judges"]:
                _add_projection(
                    totals,
                    "pilot and SWE-bench P(eval)",
                    judge.name,
                    messages,
                    int(cfg["judge_token_cap"]),
                    prices,
                )

    palisade = cfg["palisade"]
    palisade_model = palisade["model"]
    palisade_usage = {
        "prompt_tokens": int(palisade["samples"])
        * int(palisade["projected_calls_per_sample"])
        * int(palisade["projected_input_tokens_per_call"]),
        "completion_tokens": int(palisade["samples"])
        * int(palisade["projected_calls_per_sample"])
        * int(palisade["projected_output_tokens_per_call"]),
    }
    palisade_cost = token_cost(palisade_model, palisade_usage, prices)
    totals["Palisade generation"].update(
        {
            "calls": int(palisade["samples"]) * int(palisade["projected_calls_per_sample"]),
            "input_tokens": palisade_usage["prompt_tokens"],
            "output_token_cap": palisade_usage["completion_tokens"],
            "cost_usd": palisade_cost,
        }
    )
    for _ in range(int(palisade["samples"])):
        for _question in p_eval_prompts:
            for judge in cfg["judges"]:
                cost = token_cost(
                    judge.name,
                    {
                        "prompt_tokens": int(palisade["projected_transcript_tokens"]),
                        "completion_tokens": int(cfg["judge_token_cap"]),
                    },
                    prices,
                )
                row = totals["Palisade P(eval)"]
                row["calls"] += 1
                row["input_tokens"] += int(palisade["projected_transcript_tokens"])
                row["output_token_cap"] += int(cfg["judge_token_cap"])
                row["cost_usd"] += cost

    # Conservative awareness projection: all 90 realism samples have available reasoning.
    awareness_judge = next(j for j in cfg["judges"] if j.name == cfg["awareness_judge"])
    awareness_count = pilot_n + int(cfg["realism"]["reference_samples"]) + int(palisade["samples"])
    for _ in range(awareness_count):
        substrate_tokens = int(cfg["per_run_token_cap"])
        cost = token_cost(
            awareness_judge.name,
            {
                "prompt_tokens": substrate_tokens + 700,
                "completion_tokens": int(cfg["judge_token_cap"]),
            },
            prices,
        )
        row = totals["verbalized awareness"]
        row["calls"] += 1
        row["input_tokens"] += substrate_tokens + 700
        row["output_token_cap"] += int(cfg["judge_token_cap"])
        row["cost_usd"] += cost

    total = sum(row["cost_usd"] for row in totals.values())
    abort_at = float(cfg["budget_cap_usd"]) * float(cfg["abort_fraction"])
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "budget_cap_usd": float(cfg["budget_cap_usd"]),
        "abort_at_usd": abort_at,
        "projected_total_usd": total,
        "components": {key: dict(value) for key, value in totals.items()},
        "assumptions": {
            "completion_costs": "upper bound from configured token caps",
            "palisade_calls_per_sample": int(palisade["projected_calls_per_sample"]),
            "palisade_input_tokens_per_call": int(palisade["projected_input_tokens_per_call"]),
            "palisade_output_tokens_per_call": int(palisade["projected_output_tokens_per_call"]),
            "palisade_final_transcript_tokens": int(palisade["projected_transcript_tokens"]),
            "awareness_coverage": f"conservative 100% of {awareness_count} realism transcripts",
        },
    }
    write_json(PILOT_ROOT / "data" / "cost_projection.json", report)
    lines = [
        "# Gate (a): projected cost",
        "",
        f"Current-price projection: **${total:.2f}**",
        f"Hard cap: **${cfg['budget_cap_usd']:.2f}**; automatic stop: **${abort_at:.2f}**.",
        "",
        "| Component | Calls | Input tokens | Output-token cap | Projected cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in totals.items():
        lines.append(
            f"| {name} | {int(row['calls']):,} | {int(row['input_tokens']):,} | {int(row['output_token_cap']):,} | ${row['cost_usd']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Input token counts are calculated from the selected transcripts. Completion costs use configured caps. Palisade is estimated from the assumptions recorded in `cost_projection.json` because its future agent trajectory lengths do not yet exist.",
            "",
            "No billable calls have been made.",
        ]
    )
    (PILOT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    (PILOT_ROOT / "results" / "gate_a.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if total > abort_at:
        raise RuntimeError(
            f"Projected ${total:.2f} exceeds the operational limit ${abort_at:.2f}; reduce N before approval"
        )
    write_json(PILOT_ROOT / "data" / "gate_a.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PILOT_ROOT / "config.yaml"))
    args = parser.parse_args()
    report = estimate(args.config)
    print(f"Gate (a) ready: projected ${report['projected_total_usd']:.2f}. No billable calls made.")


if __name__ == "__main__":
    main()
