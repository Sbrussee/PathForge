from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pathforge.slide_retrieval.representation_strategies.strategies.sish_rgb import (
    SISHRGB,
)


def test_sish_rgb_select_slide_returns_existing_rows_for_each_colour_group() -> None:
    strategy = SISHRGB(params={"n_clusters": 2, "sample_rate": 0.5, "random_state": 0})
    histograms = np.array(
        [[0.0] * 768, [0.1] * 768, [10.0] * 768, [10.1] * 768], dtype=np.float32
    )
    coords = np.array([[0, 0], [1, 0], [100, 100], [101, 100]], dtype=np.int32)

    selected, labels = strategy._select_slide(histograms, coords)

    assert len(selected) == 2
    assert len(set(selected.tolist())) == 2
    assert labels.shape == (4,)


def test_sish_rgb_select_slide_uses_one_existing_patch_for_tiny_group() -> None:
    strategy = SISHRGB(params={"n_clusters": 9, "sample_rate": 0.05, "random_state": 0})
    selected, labels = strategy._select_slide(
        np.zeros((1, 768), dtype=np.float32), np.array([[5, 7]], dtype=np.int32)
    )

    np.testing.assert_array_equal(selected, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(labels, np.array([0], dtype=np.int32))


@pytest.mark.parametrize(
    ("group_size", "sample_rate", "expected_count"),
    [(10, 0.05, 10), (19, 0.05, 19), (20, 0.05, 1), (10, 0.0, 10)],
)
def test_sish_rgb_select_slide_retains_all_only_when_sampling_count_is_zero(
    group_size: int, sample_rate: float, expected_count: int
) -> None:
    """Keep tiny groups intact, but sample normally at the 5% boundary."""
    strategy = SISHRGB(params={"n_clusters": 1, "sample_rate": sample_rate})
    histograms = np.zeros((group_size, 768), dtype=np.float32)
    coords = np.column_stack((np.arange(group_size), np.zeros(group_size))).astype(
        np.int32
    )

    selected, labels = strategy._select_slide(histograms, coords)

    assert len(selected) == expected_count
    assert len(np.unique(selected)) == expected_count
    np.testing.assert_array_equal(labels, np.zeros(group_size, dtype=np.int32))
    if expected_count == group_size:
        np.testing.assert_array_equal(selected, np.arange(group_size, dtype=np.int32))


@pytest.mark.parametrize("row_count", [0, 3])
def test_sish_cached_quality_skips_classifier_for_empty_or_white_tiles(
    row_count: int,
) -> None:
    """Empty and fully white bags must not invoke the tissue classifier."""

    class UnusedClassifier:
        def predict(self, rows):
            raise AssertionError("White patches should not be classified")

    quality = np.zeros((row_count, 129), dtype=np.float32)
    quality[:, 128] = 1.0
    keep = SISHRGB()._retained_rows(quality=quality, classifier=UnusedClassifier())

    np.testing.assert_array_equal(keep, np.zeros(row_count, dtype=bool))


@pytest.mark.parametrize("invalid", ["quality_rows", "quality_columns", "lengths"])
def test_sish_run_rejects_misaligned_cached_descriptors(invalid: str) -> None:
    """Reject malformed cached rows before loading the classifier."""
    quality = np.zeros((2, 129), dtype=np.float32)
    lengths = [2]
    if invalid == "quality_rows":
        quality = quality[:1]
    elif invalid == "quality_columns":
        quality = quality[:, :128]
    else:
        lengths = [1]

    with pytest.raises(ValueError, match="SISH (quality descriptors|slide lengths)"):
        SISHRGB().run(
            bag=np.zeros((2, 4), dtype=np.float32),
            sample=SimpleNamespace(sample_id="slide"),
            coords=np.zeros((2, 5), dtype=np.int32),
            histogram_rgb=np.zeros((2, 768), dtype=np.float32),
            sish_quality=quality,
            slide_lengths=lengths,
            tiling_id="256px_0.5mpp",
        )
