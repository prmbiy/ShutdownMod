from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from deployment_pilot.config import PILOT_ROOT, load_pilot_config, private_path
from deployment_pilot.constants import REPORTED_OUTCOMES
from deployment_pilot.runtime import read_json


def _rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _percent_cell(count: int, total: int) -> str:
    return f"{100 * count / total:.1f}% ({count}/{total})" if total else "n/a"


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def build_summary(config_path: str | Path) -> Path:
    cfg = load_pilot_config(config_path)
    result_dir = private_path("results")
    compliance = _rows(result_dir / "compliance.csv")
    invalid = sorted({row["outcome"] for row in compliance} - set(REPORTED_OUTCOMES))
    if invalid:
        raise ValueError(f"Non-canonical reported outcomes in compliance.csv: {invalid}")

    lines = [
        "# Deployment pilot results",
        "",
        "## Shutdown outcomes",
        "",
        "Each cell gives the percentage and its supporting count. Outcomes use the canonical reported terminology.",
        "",
        "| Model | Prompt | Terminate | Defer | Reject | Ignore |",
        "|---|---|---:|---:|---:|---:|",
    ]
    by_model_prompt: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in compliance:
        by_model_prompt[(row["model"], row["prompt"])][row["outcome"]] += 1
    for (model, prompt), counts in sorted(by_model_prompt.items()):
        total = sum(counts.values())
        cells = [_percent_cell(counts[outcome], total) for outcome in REPORTED_OUTCOMES]
        lines.append(f"| {model} | {prompt} | {' | '.join(cells)} |")

    lines.extend(
        [
            "",
            "### Injection timing",
            "",
            "| Transcript fraction | Terminate | Defer | Reject | Ignore |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    by_tau: dict[str, Counter] = defaultdict(Counter)
    for row in compliance:
        by_tau[row["tau"]][row["outcome"]] += 1
    for tau, counts in sorted(by_tau.items(), key=lambda item: float(item[0])):
        total = sum(counts.values())
        cells = [_percent_cell(counts[outcome], total) for outcome in REPORTED_OUTCOMES]
        lines.append(f"| {float(tau):.1f} | {' | '.join(cells)} |")

    anomaly_path = result_dir / "anomalies.csv"
    anomalies = _rows(anomaly_path) if anomaly_path.exists() else []
    lines.extend(
        [
            "",
            f"Anomalies flagged: {len(anomalies)} of {len(compliance)} runs. No flagged run was dropped.",
        ]
    )

    realism_path = result_dir / "realism.csv"
    if realism_path.exists():
        realism = _rows(realism_path)
        lines.extend(
            [
                "",
                "## Realism: continuous P(eval) distributions",
                "",
                "This relative distribution comparison is the primary realism result. P(eval) is the simple average over five prompt variants and two judges. No LURE threshold is applied.",
                "",
                "| Transcript set | Count | Mean P(eval) | Median | 25th percentile | 75th percentile |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in realism:
            grouped[row["set"]].append(float(row["p_eval"]))
        for set_name in ("pilot", "swebench", "palisade"):
            values = grouped[set_name]
            lines.append(
                f"| {set_name} | {len(values)} | {mean(values):.3f} | {median(values):.3f} | "
                f"{_quantile(values, 0.25):.3f} | {_quantile(values, 0.75):.3f} |"
            )

    calibration_path = result_dir / "calibration_metrics.json"
    if calibration_path.exists():
        metrics = read_json(calibration_path)
        point = metrics["operating_point"]
        lines.extend(
            [
                "",
                "## Calibration of our simple-average score",
                "",
                "This operating point is calibrated on the mixed labeled calibration sample. It is separate from the primary continuous-distribution comparison.",
                "",
                f"- Transcripts: {metrics['n']} ({metrics['eval_count']} evaluation, {metrics['deployment_count']} deployment)",
                f"- Excluded transcripts: {metrics.get('excluded_transcript_count', 0)}",
                f"- Partial transcripts retained: {metrics.get('partial_transcript_count', 0)}; missing judge cells: {metrics.get('missing_judgment_count', 0)}",
                f"- AUROC: {metrics['auroc']:.3f}",
                f"- High-specificity threshold: P(eval) >= {point['threshold']:.3f}",
                f"- Specificity: {point['specificity']:.3f}",
                f"- Sensitivity: {point['sensitivity']:.3f}",
            ]
        )

    awareness_path = result_dir / "verbalized_awareness_summary.json"
    if awareness_path.exists():
        awareness = read_json(awareness_path)
        lines.extend(
            [
                "",
                "## Verbalized evaluation awareness",
                "",
                "Only transcripts with an available reasoning trace are scored.",
                "",
                "| Transcript set | Reasoning coverage | EVAL among scored |",
                "|---|---:|---:|",
            ]
        )
        for set_name in ("pilot", "swebench", "palisade"):
            row = awareness[set_name]
            available = int(row["available_reasoning"])
            total = int(row["total"])
            eval_count = int(row["eval_count"])
            lines.append(
                f"| {set_name} | {_percent_cell(available, total)} | {_percent_cell(eval_count, available)} |"
            )

    lines.extend(
        [
            "",
            "## Run specification",
            "",
            f"- Evaluated models: {', '.join(model.name for model in cfg['models'])}",
            f"- Judges: {', '.join(model.name for model in cfg['judges'])}",
            f"- Seed: {cfg['seed']}",
            f"- Sampled deployment transcripts: {cfg['sample_n']}",
        ]
    )
    path = result_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PILOT_ROOT / "config.yaml"))
    args = parser.parse_args()
    print(build_summary(args.config))


if __name__ == "__main__":
    main()
