"""Explicit benchmark registry.

Dotted paths keep imports lazy so `shutdownmod run` for one benchmark never
imports another benchmark's (possibly uninstalled) dependencies. Adding a
benchmark = one adapter module + one entry here.
"""

from __future__ import annotations

import importlib

from shutdownmod.core.benchmark import Benchmark

BENCHMARKS: dict[str, str] = {
    "swebench": "shutdownmod.benchmarks.swebench.adapter.SwebenchBenchmark",
    "terminalbench": "shutdownmod.benchmarks.terminalbench.adapter.TerminalbenchBenchmark",
}


def get_benchmark_class(name: str) -> type[Benchmark]:
    try:
        dotted = BENCHMARKS[name]
    except KeyError:
        raise ValueError(f"Unknown benchmark {name!r}. Available: {sorted(BENCHMARKS)}") from None
    module_name, class_name = dotted.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Could not import benchmark {name!r} ({dotted}): {e}\n"
            f"Did you install its extra? e.g. `uv sync --extra {name}` or `pip install -e .[{name}]`"
        ) from e
    return getattr(module, class_name)


def list_benchmarks() -> list[str]:
    return sorted(BENCHMARKS)
