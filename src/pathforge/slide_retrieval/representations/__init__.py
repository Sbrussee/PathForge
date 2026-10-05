"""Shared retrieval representations independent of one search method."""

from pathforge.slide_retrieval.representations.minmax import (
    EncodedBarcodes,
    encode_minmax,
)

__all__ = ["EncodedBarcodes", "encode_minmax"]
