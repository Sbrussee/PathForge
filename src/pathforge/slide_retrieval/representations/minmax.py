"""Yottixel/SISH-compatible MinMax barcode encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EncodedBarcodes:
    """Packed MinMax barcodes and their pre- and post-packing dimensions."""

    packed: np.ndarray
    feature_dimension: int
    faiss_bit_dimension: int


def encode_minmax(features: np.ndarray) -> EncodedBarcodes:
    """Encode finite feature rows into big-endian MinMax barcodes.

    The first bit of every barcode is zero. Each subsequent bit is one when
    the corresponding feature is greater than or equal to its predecessor.
    This is the convention used by SISH and SalvDataset-CBIR.
    """
    _validate_features(features)
    feature_dimension = int(features.shape[1])
    faiss_bit_dimension = ((feature_dimension + 7) // 8) * 8
    bits = np.zeros((features.shape[0], faiss_bit_dimension), dtype=np.uint8)
    if feature_dimension > 1:
        bits[:, 1:feature_dimension] = features[:, 1:] >= features[:, :-1]
    return EncodedBarcodes(
        packed=np.ascontiguousarray(np.packbits(bits, axis=1, bitorder="big")),
        feature_dimension=feature_dimension,
        faiss_bit_dimension=faiss_bit_dimension,
    )


def _validate_features(features: object) -> None:
    if (
        not isinstance(features, np.ndarray)
        or features.ndim != 2
        or not features.shape[0]
        or not features.shape[1]
    ):
        raise ValueError("features must be a non-empty two-dimensional array.")
    if (
        not np.issubdtype(features.dtype, np.number)
        or np.issubdtype(features.dtype, np.bool_)
        or np.issubdtype(features.dtype, np.complexfloating)
    ):
        raise ValueError("features must have a real numeric dtype.")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite.")
