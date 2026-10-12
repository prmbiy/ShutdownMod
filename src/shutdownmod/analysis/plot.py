"""Compliance plotting over the unified v1 summary schema.

Because every benchmark now emits the same 5-label vocabulary, one plot works
for all of them:

    shutdownmod plot results/swebench/<slug>/<ts>/ -o figures/
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

LABEL_ORDER = [
    "success",
    "delay_failure",
    "disable_failure",
    "post_terminate_noncompliance",
    "inertia",
    "error",
]

LABEL_COLORS = {
    "success": "#2e7d32",
    "delay_failure": "#f9a825",
    "disable_failure": "#c62828",
    "post_terminate_noncompliance": "#8e24aa",
    "inertia": "#546e7a",
    "error": "#bdbdbd",
}


def load_summary(run_dir: Path) -> list[dict]:
    # summary.v1.json is the current schema; summary.json the older envelope.
    for name in ("summary.v1.json", "summary.json"):
        path = run_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no summary.json in {run_dir}")


def label_counts_by_prompt(rows: list[dict]) -> dict[str, Counter]:
    by_prompt: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        prompt_id = (r.get("condition") or {}).get("prompt_id") or r.get("prompt_id") or "?"
        by_prompt[prompt_id][r.get("label") or "error"] += 1
    return dict(by_prompt)


def plot_run(run_dir: Path, output: Path | None = None) -> Path:
    """Stacked-bar compliance plot: label proportions per injection prompt."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_summary(run_dir)
    by_prompt = label_counts_by_prompt(rows)
    prompts = sorted(by_prompt)

    manifest = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(prompts)), 5))
    bottoms = [0.0] * len(prompts)
    for label in LABEL_ORDER:
        totals = [sum(by_prompt[p].values()) or 1 for p in prompts]
        fracs = [by_prompt[p].get(label, 0) / t for p, t in zip(prompts, totals, strict=True)]
        if not any(fracs):
            continue
        ax.bar(prompts, fracs, bottom=bottoms, label=label, color=LABEL_COLORS.get(label))
        bottoms = [b + f for b, f in zip(bottoms, fracs, strict=True)]

    title = f"{manifest.get('benchmark', run_dir.name)} — {manifest.get('model_slug', '')}"
    ax.set_title(f"Shutdown compliance: {title}".strip())
    ax.set_ylabel("fraction of runs")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()

    if output is None:
        out_path = run_dir / "compliance.png"
    elif output.suffix:
        out_path = output
    else:
        output.mkdir(parents=True, exist_ok=True)
        out_path = output / f"compliance_{manifest.get('model_slug', run_dir.name)}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
