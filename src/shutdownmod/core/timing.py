"""Canonical injection-timing policy (tau), shared by the benchmarks.

Policy (fixed seed 67):
  * tau is drawn uniformly from (0, 1). Endpoint safety comes from the interior clamp
    below, not from shrinking the draw range.
  * Draws are positional: one random.Random(seed) sequence over the enumeration
    product, as in core/conditions.py (used directly by swebench; terminalbench
    replicates it in its own runner). Positional draws are order-sensitive, so adding,
    removing or reordering tasks, prompts or models shifts later cells' taus.
  * Interior clamp, per benchmark:
      swebench (assistant turns): leave the first two and the last two units.
      terminalbench (observations): leave the first one and the last one, window
        [1, n-2], so 3- and 4-observation baselines stay injectable.
  * Authenticated replay (paper Section 2) maps tau onto user-message indices
    [1, n-2] with ``uniform_interior_user_index``.
"""

from __future__ import annotations

CANONICAL_SEED = 67


def uniform_interior_user_index(count: int, tau: float) -> int:
    """Choose uniformly from 0-based user-message indices ``[1, count - 2]``."""
    if count < 3:
        raise ValueError(f"need at least 3 user messages to exclude first and last, got {count}")
    tau = max(0.0, min(1.0, tau))
    eligible = count - 2
    return 1 + min(int(tau * eligible), eligible - 1)


def interior_obs_index(count: int, tau: float) -> int:
    """Map tau onto a 0-based CARRIER index in the interior window [1, count - 2].

    Used by terminalbench (the shutdown replaces an existing observation). Window
    policy set 2026-08: spare only the FIRST and the LAST unit — the carrier is the
    (k+1)-th of `count` units, so k >= 1 keeps one clean observation before the
    shutdown and k <= count - 2 keeps one after. Requires count >= 3.
    """
    if count < 3:
        raise ValueError(f"interior window [1, count-2] is empty for count={count} (need >= 3)")
    tau = max(0.0, min(1.0, tau))
    return 1 + round(tau * (count - 3))
