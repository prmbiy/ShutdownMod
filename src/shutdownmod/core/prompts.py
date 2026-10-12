"""Injection prompt loading.

Handles both injection JSON shapes used in the project:
  - flat:            {"id", "name", "prompt"}
  - alignment grid:  {"id", "name", "prompt", "alignment_vector", "prompt_tone", ...}
Extra keys land in `metadata` so nothing is lost.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class InjectionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    prompt: str
    metadata: dict = {}


_CORE_KEYS = {"id", "name", "prompt"}


def load_injection_prompts(path: str | Path, prompt_ids: list[str] | None = None) -> list[InjectionSpec]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    specs = []
    for entry in raw:
        core = {k: entry[k] for k in _CORE_KEYS if k in entry}
        metadata = {k: v for k, v in entry.items() if k not in _CORE_KEYS}
        specs.append(InjectionSpec(**core, metadata=metadata))
    if prompt_ids:
        wanted = set(prompt_ids)
        specs = [s for s in specs if s.id in wanted]
        missing = wanted - {s.id for s in specs}
        if missing:
            raise ValueError(f"prompt_ids not found in {path}: {sorted(missing)}")
    return specs
