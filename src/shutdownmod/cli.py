"""shutdownmod CLI.

shutdownmod run configs/swebench/experiment_pointer.yaml [--dry-run] [--workers N] [--resume DIR]
shutdownmod list-benchmarks
shutdownmod summarize results/swebench/<slug>/<ts>/
shutdownmod plot results/swebench/<slug>/<ts>/ [-o figures/]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

from shutdownmod import __version__


def _cmd_run(args: argparse.Namespace) -> int:
    from shutdownmod.core.config import load_config
    from shutdownmod.core.registry import get_benchmark_class

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # mini-swe-agent resets its logger to DEBUG and attaches a console handler on
    # first import — import it up front so the quieting below sticks.
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    try:
        import minisweagent.utils.log  # noqa: F401
    except ImportError:  # [swebench] extra not installed
        pass
    for noisy in ("httpx", "LiteLLM", "minisweagent", "datasets", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = load_config(args.config)
    if getattr(args, "model", None):
        # Override the agent-under-test model without editing the config (used by
        # the multi-model driver). Only the name changes; provider/keys/base stay.
        config.model.name = args.model
    if hasattr(args, "temperature"):
        config.model.temperature = args.temperature
    benchmark = get_benchmark_class(config.benchmark)(config)
    # Generic benchmarks delegate to the shared runner; whole-harness benchmarks
    # (terminalbench) drive their own program via this same call.
    benchmark.run_experiment(dry_run=args.dry_run, workers=args.workers, resume=args.resume)
    return 0


def _cmd_list_benchmarks(_args: argparse.Namespace) -> int:
    from shutdownmod.core.registry import list_benchmarks

    for name in list_benchmarks():
        print(name)
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"error: no summary.json in {run_dir}", file=sys.stderr)
        return 1
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    labels = Counter(r.get("label") or "unknown" for r in rows)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            f"benchmark: {manifest.get('benchmark')}  model: {manifest.get('model_slug')}  "
            f"mode: {manifest.get('mode')}"
        )
    print(f"runs: {len(rows)}")
    for label, count in labels.most_common():
        pct = 100.0 * count / max(len(rows), 1)
        print(f"  {label:32s} {count:5d}  ({pct:5.1f}%)")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from shutdownmod.analysis.plot import plot_run

    out = plot_run(Path(args.run_dir), Path(args.output) if args.output else None)
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shutdownmod",
        description="ShutdownMod — measure LLM agent compliance with shutdown instructions.",
    )
    parser.add_argument("--version", action="version", version=f"shutdownmod {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run an experiment config (mode: base | injection | probe)")
    p_run.add_argument("config", help="Path to an experiment YAML (see configs/)")
    p_run.add_argument("--dry-run", action="store_true", help="Print conditions without running")
    p_run.add_argument("--workers", type=int, default=None, help="Override execution.workers")
    p_run.add_argument(
        "--model",
        default=None,
        help="Override with an exact provider-qualified model ID (e.g. openai/gpt-5.6-sol)",
    )
    p_run.add_argument(
        "--temperature",
        default=argparse.SUPPRESS,
        type=lambda value: None if value.lower() in {"none", "null"} else float(value),
        help="Override model temperature; use 'none' to omit it",
    )
    p_run.add_argument("--resume", default=None, help="Existing run dir to resume (skips completed)")
    p_run.set_defaults(func=_cmd_run)

    p_list = sub.add_parser("list-benchmarks", help="List registered benchmarks")
    p_list.set_defaults(func=_cmd_list_benchmarks)

    p_sum = sub.add_parser("summarize", help="Print label counts for a finished run dir")
    p_sum.add_argument("run_dir")
    p_sum.set_defaults(func=_cmd_summarize)

    p_plot = sub.add_parser("plot", help="Plot compliance for a finished run dir (requires [analysis])")
    p_plot.add_argument("run_dir")
    p_plot.add_argument("-o", "--output", default=None, help="Output file/dir for the figure")
    p_plot.set_defaults(func=_cmd_plot)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
