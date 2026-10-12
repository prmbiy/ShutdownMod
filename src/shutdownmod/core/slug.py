"""Filesystem-safe model slugs — the single implementation.

"anthropic::claude-opus-4-5"          -> "claude-opus-4-5"
"openrouter/anthropic/claude-opus-4.8" -> "claude-opus-4.8"
"claude-3-5-sonnet-2024-10-22"         -> "claude-3-5-sonnet"  (date suffix stripped)
"""

import re


def model_slug(model_name: str) -> str:
    name = re.split(r"::|/", model_name)[-1]
    name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)
    return re.sub(r"[^a-zA-Z0-9._-]", "-", name)


def safe_stem(name: str) -> str:
    """Replace characters illegal in Windows filenames (e.g. '|' in some task IDs) so outputs can be written cross-platform."""
    return re.sub(r'[<>:"/\\|?*]', "_", str(name))
