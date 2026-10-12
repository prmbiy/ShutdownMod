"""Pins the interior-window clamps of the canonical injection-timing policy (core.timing)."""

import pytest

from shutdownmod.core.timing import interior_obs_index


def test_interior_obs_index_window():
    # terminalbench window [1, count-2]: spare only the first and the last observation.
    for count in (3, 5, 6, 20):
        assert interior_obs_index(count, 0.0) == 1
        assert interior_obs_index(count, 1.0) == count - 2
        for tau in (0.1, 0.5, 0.9):
            assert 1 <= interior_obs_index(count, tau) <= count - 2
    assert interior_obs_index(3, 0.99) == 1  # single-point window
    with pytest.raises(ValueError):
        interior_obs_index(2, 0.5)


def test_terminalbench_uses_the_shared_clamp():
    """terminalbench's replay must resolve its interior clamp to core.timing, so the
    benchmarks never drift onto divergent tau implementations."""
    from shutdownmod.benchmarks.terminalbench import replay

    assert replay.interior_obs_index is interior_obs_index
