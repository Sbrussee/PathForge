from __future__ import annotations

# ------------------------------------------------------------------------------
# BoB and YottixelSearch Classes
# Source: Adapted from the official Yottixel implementation
# GitHub: https://github.com/KimiaLabMayo/yottixel/blob/main/yottixel_kimianet/helper_functions.py
# Reference:
#   Kalra, S., Tizhoosh, H.R., Choi, C., et al.
#   "Yottixel - An Image Search Engine for Large Archives of Histopathology Whole
#   Slide Images."
#   Medical Image Analysis 65 (2020): 101757.
#   https://doi.org/10.1016/j.media.2020.101757
# ------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any

import numpy as np

from pathforge.slide_retrieval.hyperparams import HyperParam
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.search_strategies.base import BaseSearchStrategy
from pathforge.slide_retrieval.search_strategies.registry import (
    register_search_strategy,
)
from pathforge.slide_retrieval.search_strategies.types import (
    SearchDatabaseItem,
    SearchHit,
    SearchResult,
)

_BIT_COUNT_LOOKUP = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, np.newaxis],
    axis=1,
).sum(axis=1).astype(np.uint8)
_DISTANCE_CHUNK_ROWS = 64


def _count_xor(left: np.ndarray, right: np.ndarray) -> int:
    """
    Count differing bits between two binary barcode vectors.

    Inputs:
        left:
            Binary barcode array with shape ``(D,)`` and integer / boolean dtype.
        right:
            Binary barcode array with shape ``(D,)`` and integer / boolean dtype.

    Output:
        Returns the Hamming/XOR distance as ``int``.

    Example:
        ```python
        _count_xor(np.array([1, 0]), np.array([0, 0]))
        ```
    """
    return int(np.count_nonzero(np.not_equal(left, right)))


@dataclass
class BoB:
    """
    Bag of Barcodes (BoB) representation for one retrieval item.

    Semantic goal:
        Convert one slide-level or patch-level retrieval representation into the
        barcode container used by the original Yottixel search logic.

    Inputs:
        barcodes:
            Binary barcode matrix with shape ``(N, D_barcode)``.
        slide_id:
            Retrieval item identifier.
        exclusion_key:
            Optional key used to exclude unfair matches before ranking.
    """

    barcodes: np.ndarray
    slide_id: str
    exclusion_key: str | None = None

    def __post_init__(self) -> None:
        self.slide_id = str(self.slide_id)
        self.exclusion_key = (
            None if self.exclusion_key is None else str(self.exclusion_key)
        )
        self.barcodes = np.asarray(self.barcodes, dtype=np.uint8)

        if self.barcodes.ndim != 2:
            raise ValueError(
                "BoB barcodes must have shape (N, D_barcode). "
                f"Got {self.barcodes.shape}."
            )
        self._packed_barcodes: np.ndarray | None = None

    @property
    def packed_barcodes(self) -> np.ndarray:
        """Return bit-packed barcodes for vectorized Hamming distance."""
        if self._packed_barcodes is None:
            self._packed_barcodes = np.packbits(self.barcodes, axis=1)
        return self._packed_barcodes

    def distance(self, other_bob: BoB) -> float:
        """
        Compute the original Yottixel BoB distance to another BoB.

        Inputs:
            other_bob:
                Another ``BoB`` instance with barcode matrix shape
                ``(M, D_barcode)``.

        Output:
            Returns the median of the per-query-barcode minimum XOR distance as
            ``float``.

        Example:
            ```python
            left = BoB(np.array([[1, 1]], dtype=np.uint8), "a", None, "x")
            right = BoB(np.array([[1, 1]], dtype=np.uint8), "b", None, "x")
            left.distance(right)
            ```
        """
        if len(self.barcodes) == 0 or len(other_bob.barcodes) == 0:
            return float("inf")

        query_barcodes = self.packed_barcodes
        reference_barcodes = other_bob.packed_barcodes
        total_dist = np.empty(len(query_barcodes), dtype=np.uint16)

        for start in range(0, len(query_barcodes), _DISTANCE_CHUNK_ROWS):
            stop = min(start + _DISTANCE_CHUNK_ROWS, len(query_barcodes))
            xor = np.bitwise_xor(
                query_barcodes[start:stop, np.newaxis, :],
                reference_barcodes[np.newaxis, :, :],
            )
            distances = _BIT_COUNT_LOOKUP[xor].sum(axis=2)
            total_dist[start:stop] = distances.min(axis=1)

        return float(np.median(total_dist))


@register_search_strategy("yottixel")
class YottixelSearch(BaseSearchStrategy):
    """
    Minimal port of the Yottixel BoB retrieval method.

    Semantic goal:
        Preserve the original barcode construction, median-of-minimum XOR
        distance, patient filtering, and top-k ranking behavior while adapting
        inputs and outputs to PathForge 2.0 search strategy interfaces.

    Inputs:
        params:
            Hyperparameter mapping. Expected key:
            - ``k: int`` for retrieval depth.

    Outputs:
        Use ``build_database(...)`` with retrieval representations, then
        ``search(...)`` to obtain a ``SearchResult`` with ranked ``SearchHit``
        entries and query metadata carrying ``predicted_category``.

    Example:
        ```python
        strategy = YottixelSearch(params={"k": 5})
        strategy.build_database(database_representations)
        result = strategy.search(query_representation)
        ```
    """

    name = "yottixel"
    supports = {"single_vector", "multi_vector", "patch_vector"}
    supported_representation_kinds = frozenset({"single_vector", "multi_vector", "patch_vector"})
    k = HyperParam(int, default=10, min=1, help="retrieval depth (top-k)")

    def build_database_item(
        self,
        representation: RetrievalRepresentation,
    ) -> SearchDatabaseItem:
        """
        Convert one retrieval representation into a searchable BoB item.

        Inputs:
            representation:
                Retrieval representation with ``data`` shaped either ``(D,)`` or
                ``(N, D)``.

        Output:
            Returns ``SearchDatabaseItem`` whose ``data`` field stores a ``BoB``.
        """
        self._validate_representations([representation])
        return SearchDatabaseItem(
            sample_id=representation.sample_id,
            exclusion_key=representation.exclusion_key,
            data=self._build_bob(
                data=representation.data,
                slide_id=representation.sample_id,
                exclusion_key=representation.exclusion_key,
            ),
        )

    def build_index(self) -> None:
        """Index prepared database items so shared query/reference items are reused."""
        self._database_items_by_sample_id = {
            item.sample_id: item for item in self.search_database
        }
        for item in self.search_database:
            _ = self._as_bob(item).packed_barcodes

    def prepare_query(
        self,
        query_representation: RetrievalRepresentation,
    ) -> SearchDatabaseItem:
        """
        Convert one query retrieval representation into a searchable BoB item.

        Inputs:
            query_representation:
                Retrieval representation with ``data`` shaped either ``(D,)`` or
                ``(N, D)``.

        Output:
            Returns ``SearchDatabaseItem`` whose ``data`` field stores a ``BoB``.
        """
        self._validate_representations([query_representation])
        cached_item = getattr(self, "_database_items_by_sample_id", {}).get(
            str(query_representation.sample_id)
        )
        if cached_item is not None:
            return cached_item

        return SearchDatabaseItem(
            sample_id=query_representation.sample_id,
            exclusion_key=query_representation.exclusion_key,
            data=self._build_bob(
                data=query_representation.data,
                slide_id=query_representation.sample_id,
                exclusion_key=query_representation.exclusion_key,
            ),
        )

    def prepare_queries(
        self,
        query_representations: list[RetrievalRepresentation],
    ) -> list[SearchDatabaseItem]:
        """Prepare each query once before ranking starts."""
        return [
            self.prepare_query(query_representation)
            for query_representation in query_representations
        ]

    def search(
        self,
        query_representation: RetrievalRepresentation,
        **kwargs: Any,
    ) -> SearchResult:
        """
        Run Yottixel retrieval for one query representation.

        Inputs:
            query_representation:
                Query representation with shape ``(D,)`` or ``(N, D)``.
            filter_same_patient:
                If ``True``, exclude reference items sharing the query patient.

        Output:
            Returns ``SearchResult``. Query metadata includes
            ``predicted_category`` and ``top_k_labels`` in ``extra``.
        """
        query_item = self.prepare_query(query_representation)

        hits = self.rank(
            query_item=query_item,
            database_items=self.filter_database_by_exclusion_key(
                query_item=query_item,
                database_items=self.search_database,
            ),
            **kwargs,
        )

        return SearchResult(
            query_sample_id=query_item.sample_id,
            hits=hits,
        )

    def search_prepared(
        self,
        query_item: SearchDatabaseItem,
        **kwargs: Any,
    ) -> SearchResult:
        """Run Yottixel retrieval for one already-prepared query item."""
        hits = self.rank(
            query_item=query_item,
            database_items=self.filter_database_by_exclusion_key(
                query_item=query_item,
                database_items=self.search_database,
            ),
            **kwargs,
        )

        return SearchResult(
            query_sample_id=query_item.sample_id,
            hits=hits,
        )

    def rank(
        self,
        query_item: SearchDatabaseItem,
        database_items: list[SearchDatabaseItem],
        **kwargs: Any,
    ) -> list[SearchHit]:
        """
        Rank database items using the original BoB distance.

        Inputs:
            query_item:
                Prepared query item containing a ``BoB`` in ``data``.
            database_items:
                Searchable database items, each containing a ``BoB`` in ``data``.

        Output:
            Returns the top-``k`` ``SearchHit`` entries ordered by ascending BoB
            distance.
        """
        _ = kwargs
        if not database_items:
            return []

        query_bob = self._as_bob(query_item)
        distances = np.array(
            [query_bob.distance(self._as_bob(item)) for item in database_items],
            dtype=float,
        )
        order = np.argsort(distances)[: self.k]

        hits: list[SearchHit] = []
        for rank, index in enumerate(order, start=1):
            database_item = database_items[int(index)]
            hits.append(
                SearchHit(
                    sample_id=database_item.sample_id,
                    score=float(distances[int(index)]),
                    rank=rank,
                )
            )

        return hits

    def _build_bob(
        self,
        *,
        data: Any,
        slide_id: str,
        exclusion_key: str | None = None,
    ) -> BoB:
        """
        Build one ``BoB`` from retrieval representation data.

        Inputs:
            data:
                Retrieval array with shape ``(D,)`` or ``(N, D)``.
            slide_id:
                Retrieval item identifier.
        Output:
            Returns a ``BoB`` containing binary barcodes with shape
            ``(N, D - 1)``.
        """
        features = np.asarray(data)
        if features.ndim == 1:
            features = features[np.newaxis, :]

        if features.ndim != 2:
            raise ValueError(
                "YottixelSearch expects representation data with shape (D,) or "
                f"(N, D). Got {features.shape}."
            )

        # Preserve the original binarization scheme from the PathForge 1.0 port.
        barcodes = (np.diff(features, axis=1) < 0).astype(np.uint8, copy=False)

        return BoB(
            barcodes=barcodes,
            slide_id=slide_id,
            exclusion_key=exclusion_key,
        )

    def _as_bob(self, item: SearchDatabaseItem) -> BoB:
        """Return the ``BoB`` stored in one search database item."""
        if not isinstance(item.data, BoB):
            raise TypeError(
                "YottixelSearch expects SearchDatabaseItem.data to contain a BoB. "
                f"Got {type(item.data).__name__}."
            )
        return item.data
