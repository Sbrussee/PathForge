"""Unit tests for lightweight annotation helpers."""

from __future__ import annotations

from pathbench.core.annotations.binning import bin_times


def test_bin_times_quantile_returns_expected_bin_count() -> None:
    """Quantile binning should return one bin id per input time."""
    times = [5.0, 8.0, 10.0, 20.0, 25.0, 30.0]

    bins = bin_times(times, n_bins=3, method="quantile")

    assert len(bins) == len(times)
    assert min(bins) >= 1
    assert max(bins) <= 3


def test_bin_times_linear_spreads_values_across_bins() -> None:
    """Linear binning should produce ordered bin assignments for ordered times."""
    times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    bins = bin_times(times, n_bins=3, method="linear")

    assert len(bins) == len(times)
    assert bins[0] <= bins[-1]
    assert set(bins).issubset({1, 2, 3})
