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

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from pathbench.slide_retrieval.hyperparams import HyperParam
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.search_strategies.base import BaseSearchStrategy
from pathbench.slide_retrieval.search_strategies.registry import (
    register_search_strategy,
)
from pathbench.slide_retrieval.search_strategies.types import (
    SearchDatabaseItem,
    SearchHit,
    SearchResult,
)
from pathbench.slide_retrieval.types import RetrievalItemMetadata


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
        patient_id:
            Patient identifier used for exclusion during evaluation.
        label:
            Ground-truth category / diagnosis label.

    Output:
        Instantiated ``BoB`` object exposing ``distance(other_bob) -> float``.

    Example:
        ```python
        bob = BoB(
            barcodes=np.array([[1, 0, 1]], dtype=np.uint8),
            slide_id="slide-1",
            patient_id="patient-1",
            label="tumor",
        )
        ```
    """

    barcodes: np.ndarray
    slide_id: str
    patient_id: str | None
    label: str | None

    def __post_init__(self) -> None:
        self.slide_id = str(self.slide_id)
        self.patient_id = (
            None if self.patient_id is None else str(self.patient_id)
        )
        self.label = None if self.label is None else str(self.label)
        self.barcodes = np.asarray(self.barcodes, dtype=np.uint8)

        if self.barcodes.ndim != 2:
            raise ValueError(
                "BoB barcodes must have shape (N, D_barcode). "
                f"Got {self.barcodes.shape}."
            )

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

        total_dist: list[int] = []
        for feat in self.barcodes:
            # Compute XOR distance between this barcode and all in the other BoB.
            distances = [_count_xor(feat, other) for other in other_bob.barcodes]
            total_dist.append(int(np.min(distances)))

        return float(np.median(total_dist))


@register_search_strategy("yottixel")
class YottixelSearch(BaseSearchStrategy):
    """
    Minimal port of the Yottixel BoB retrieval method.

    Semantic goal:
        Preserve the original barcode construction, median-of-minimum XOR
        distance, patient filtering, and top-k ranking behavior while adapting
        inputs and outputs to PathBench 2.0 search strategy interfaces.

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
    supports = {"single_vector", "multi_vector"}
    supported_representation_kinds = frozenset({"single_vector", "multi_vector"})
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
        metadata = RetrievalItemMetadata.from_any(representation.metadata)

        return SearchDatabaseItem(
            item_id=representation.sample_id,
            search_type=representation.representation_type,
            data=self._build_bob(
                data=representation.data,
                slide_id=representation.sample_id,
                metadata=metadata,
            ),
            metadata=metadata,
        )

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
        metadata = RetrievalItemMetadata.from_any(query_representation.metadata)

        return SearchDatabaseItem(
            item_id=query_representation.sample_id,
            search_type=query_representation.representation_type,
            data=self._build_bob(
                data=query_representation.data,
                slide_id=query_representation.sample_id,
                metadata=metadata,
            ),
            metadata=metadata,
        )

    def search(
        self,
        query_representation: RetrievalRepresentation,
        *,
        filter_same_patient: bool = True,
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

        database_items = self.search_database
        if filter_same_patient:
            database_items = self.filter_database_by_patient(
                query_item=query_item,
                database_items=database_items,
            )

        hits = self.rank(
            query_item=query_item,
            database_items=database_items,
            **kwargs,
        )
        predicted_category = self._predict_category(
            query_item=query_item,
            hits=hits,
        )

        return SearchResult(
            query_id=query_item.item_id,
            hits=hits,
            metadata=query_item.metadata.copy(
                predicted_category=predicted_category,
                top_k_labels=[hit.metadata.category for hit in hits],
            ),
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
                    item_id=database_item.item_id,
                    score=float(distances[int(index)]),
                    rank=rank,
                    metadata=database_item.metadata,
                )
            )

        return hits

    def _build_bob(
        self,
        *,
        data: Any,
        slide_id: str,
        metadata: RetrievalItemMetadata,
    ) -> BoB:
        """
        Build one ``BoB`` from retrieval representation data.

        Inputs:
            data:
                Retrieval array with shape ``(D,)`` or ``(N, D)``.
            slide_id:
                Retrieval item identifier.
            metadata:
                Normalized retrieval metadata.

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

        # Preserve the original binarization scheme from the PathBench 1.0 port.
        barcodes = (np.diff(features, axis=1) < 0).astype(np.uint8, copy=False)

        return BoB(
            barcodes=barcodes,
            slide_id=slide_id,
            patient_id=metadata.patient_id,
            label=metadata.category,
        )

    def _as_bob(self, item: SearchDatabaseItem) -> BoB:
        """Return the ``BoB`` stored in one search database item."""
        if not isinstance(item.data, BoB):
            raise TypeError(
                "YottixelSearch expects SearchDatabaseItem.data to contain a BoB. "
                f"Got {type(item.data).__name__}."
            )
        return item.data

    def _predict_category(
        self,
        *,
        query_item: SearchDatabaseItem,
        hits: list[SearchHit],
    ) -> str | None:
        """
        Predict the query category using majority vote over the ranked hits.

        Inputs:
            query_item:
                Prepared query item carrying query metadata.
            hits:
                Ranked search hits.

        Output:
            Returns the majority-vote category. Falls back to the query category
            when no eligible atlas items remain after filtering.
        """
        if not hits:
            return query_item.metadata.category

        top_k_labels = [hit.metadata.category for hit in hits]
        return Counter(top_k_labels).most_common(1)[0][0]
