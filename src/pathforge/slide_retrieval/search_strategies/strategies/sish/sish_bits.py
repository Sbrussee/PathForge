"""Binary companion-feature encoding used by SISH Hamming matching."""

from __future__ import annotations

import numpy as np

from pathforge.slide_retrieval.representations.minmax import encode_minmax


def pack_adjacent_feature_bits(features: np.ndarray) -> np.ndarray:
    """Pack SISH/Yottixel adjacent-value bits for each patch feature row.

    Inputs:
        features: Float feature matrix shaped ``(N, D)``.

    Output:
        Packed unsigned-byte matrix shaped ``(N, ceil(D / 8))``. The first bit
        is zero; each following bit indicates whether the current feature is at
        least its predecessor.

    Example:
        ``pack_adjacent_feature_bits(np.asarray([[2.0, 1.0, 3.0]]))``.
    """
    return encode_minmax(np.asarray(features, dtype=np.float32)).packed


def pack_adjacent_feature_row_bits(feature_row: np.ndarray) -> bytes:
    """Return the SISH/Yottixel packed bit code for one feature row.

    Input:
        feature_row: One-dimensional feature array shaped ``(D,)``.

    Output:
        Packed byte string with ``ceil(D / 8)`` bytes.

    Example:
        ``pack_adjacent_feature_row_bits(np.asarray([2.0, 1.0]))``.
    """
    return pack_adjacent_feature_bits(np.asarray(feature_row)[None, :])[0].tobytes()
