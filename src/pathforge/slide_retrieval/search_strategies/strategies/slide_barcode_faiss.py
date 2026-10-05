"""Exact FAISS Hamming retrieval for one global feature vector per WSI."""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np

from pathforge.slide_retrieval.hyperparams import HyperParam
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.representations.minmax import encode_minmax
from pathforge.slide_retrieval.search_strategies.base import BaseSearchStrategy
from pathforge.slide_retrieval.search_strategies.registry import (
    register_search_strategy,
)
from pathforge.slide_retrieval.search_strategies.types import (
    SearchDatabaseItem,
    SearchHit,
    SearchResult,
)

_SEARCH_DEPTH_INCREMENT = 200


@register_search_strategy("slide-barcode-faiss")
class SlideBarcodeFaissSearch(BaseSearchStrategy):
    """Find exact Hamming-nearest WSIs from SISH/MinMax barcodes.

    Raw `(1, D)` slide vectors remain in retrieval-representation artifacts.
    This adapter creates packed barcodes only while building and querying its
    in-memory FAISS binary-flat index.
    """

    name = "slide-barcode-faiss"
    supported_representation_kinds = frozenset({"slide_vector"})
    k = HyperParam(int, default=10, min=1, help="top-k nearest WSIs")

    def _validate_representations(
        self,
        representations: list[RetrievalRepresentation],
    ) -> None:
        super()._validate_representations(representations)
        for representation in representations:
            if representation.representation_type != "slide_vector":
                raise ValueError(
                    "slide-barcode-faiss requires representation type "
                    f"'slide_vector', got {representation.representation_type!r}."
                )
            self._feature_row(representation.data)

    def build_index(self) -> None:
        """Encode references and create one exact binary-flat FAISS index."""
        if not self.search_database:
            raise ValueError("slide-barcode-faiss requires at least one reference WSI.")
        features = np.concatenate(
            [self._feature_row(item.data) for item in self.search_database], axis=0
        )
        encoded = encode_minmax(features)
        self._feature_dimension = encoded.feature_dimension
        self._index = faiss.IndexBinaryFlat(encoded.faiss_bit_dimension)
        self._index.add(encoded.packed)
        self._items_by_faiss_row = tuple(self.search_database)

    def search(
        self,
        query_representation: RetrievalRepresentation,
        **kwargs: Any,
    ) -> SearchResult:
        """Return the K nearest eligible reference WSIs by Hamming distance."""
        _ = kwargs
        query_item = self.prepare_query(query_representation)
        query_features = self._feature_row(query_item.data)
        encoded = encode_minmax(query_features)
        if encoded.feature_dimension != self._feature_dimension:
            raise ValueError(
                "Query feature dimension is "
                f"{encoded.feature_dimension}, expected {self._feature_dimension}."
            )
        return SearchResult(
            query_sample_id=query_item.sample_id,
            hits=self._search_eligible(query_item, encoded.packed),
        )

    def rank(
        self,
        query_item: SearchDatabaseItem,
        database_items: list[SearchDatabaseItem],
        **kwargs: Any,
    ) -> list[SearchHit]:
        """Rank exactly the eligible subset supplied by the base strategy."""
        _ = kwargs
        encoded = encode_minmax(self._feature_row(query_item.data))
        if not database_items:
            return []
        features = np.concatenate(
            [self._feature_row(item.data) for item in database_items], axis=0
        )
        candidates = encode_minmax(features)
        if candidates.feature_dimension != encoded.feature_dimension:
            raise ValueError(
                "Query feature dimension is "
                f"{encoded.feature_dimension}, expected {candidates.feature_dimension}."
            )
        subset_index = faiss.IndexBinaryFlat(candidates.faiss_bit_dimension)
        subset_index.add(candidates.packed)
        distances, row_ids = subset_index.search(
            encoded.packed, min(int(self.k), len(database_items))
        )
        return [
            SearchHit(
                sample_id=database_items[int(row_id)].sample_id,
                score=float(distance),
                rank=rank,
                metadata=database_items[int(row_id)].metadata,
            )
            for rank, (distance, row_id) in enumerate(
                zip(distances[0], row_ids[0]), start=1
            )
            if row_id >= 0
        ]

    def _search_eligible(
        self,
        query_item: SearchDatabaseItem,
        packed_query: np.ndarray,
    ) -> list[SearchHit]:
        total_rows = len(self._items_by_faiss_row)
        search_depth = min(total_rows, max(int(self.k), _SEARCH_DEPTH_INCREMENT))
        while True:
            distances, row_ids = self._index.search(packed_query, search_depth)
            hits: list[SearchHit] = []
            for distance, row_id in zip(distances[0], row_ids[0]):
                if row_id < 0:
                    continue
                item = self._items_by_faiss_row[int(row_id)]
                if (
                    query_item.exclusion_key is not None
                    and item.exclusion_key == query_item.exclusion_key
                ):
                    continue
                hits.append(
                    SearchHit(
                        sample_id=item.sample_id,
                        score=float(distance),
                        rank=len(hits) + 1,
                        metadata=item.metadata,
                    )
                )
                if len(hits) == self.k:
                    return hits
            if search_depth == total_rows:
                return hits
            search_depth = min(total_rows, search_depth + _SEARCH_DEPTH_INCREMENT)

    @staticmethod
    def _feature_row(data: Any) -> np.ndarray:
        features = np.asarray(data)
        if features.ndim != 2 or features.shape[0] != 1 or features.shape[1] <= 0:
            raise ValueError(
                "slide-barcode-faiss requires a raw feature matrix with shape (1, D). "
                f"Got {features.shape}."
            )
        return features
