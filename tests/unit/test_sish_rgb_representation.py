from __future__ import annotations

import numpy as np

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
