from __future__ import annotations

import numpy as np
import pytest

from pathforge.slide_retrieval.search_strategies.strategies.sish.sish_bits import (
    pack_adjacent_feature_bits,
)


def test_sish_bits_use_adjacent_value_encoding() -> None:
    packed = pack_adjacent_feature_bits(np.asarray([[2.0, 1.0, 3.0, 3.0]]))

    np.testing.assert_array_equal(packed, np.asarray([[48]], dtype=np.uint8))


def test_sish_bits_reject_non_matrix_input() -> None:
    with pytest.raises(ValueError, match="2D feature matrix"):
        pack_adjacent_feature_bits(np.asarray([1.0, 2.0]))
