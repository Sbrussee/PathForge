from __future__ import annotations

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
from pathforge.slide_retrieval.types import RetrievalItemMetadata


@dataclass(slots=True)
class PBSSSearchItem:
    """Prototype statistics needed for PBSS ranking."""

    slide_id: str
    exclusion_key: str | None
    metadata: RetrievalItemMetadata
    proto_mean: np.ndarray
    proto_cov: np.ndarray
    include_mask: np.ndarray | None = None


def _per_proto_diag_mahalanobis(
    mu_a: np.ndarray,
    cov_a: np.ndarray,
    mu_b: np.ndarray,
    cov_b: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    """Compute same-prototype diagonal Mahalanobis distances."""
    proto_count = min(mu_a.shape[0], mu_b.shape[0])
    feature_dim = min(mu_a.shape[1], mu_b.shape[1])
    mu_a = mu_a[:proto_count, :feature_dim]
    mu_b = mu_b[:proto_count, :feature_dim]
    cov_a = cov_a[:proto_count, :feature_dim]
    cov_b = cov_b[:proto_count, :feature_dim]

    denom = np.clip(cov_a, 0.0, None) + np.clip(cov_b, 0.0, None) + float(eps)
    diff = mu_a - mu_b
    distances_squared = np.sum((diff * diff) / denom, axis=1)
    return np.sqrt(np.maximum(distances_squared, 0.0))


def _softmin(distances: np.ndarray, *, tau: float) -> float:
    """Stable soft-min aggregation over per-prototype distances."""
    values = np.asarray(distances, dtype=np.float32)
    if values.size == 0:
        return float("inf")
    minimum = float(np.min(values))
    scaled = np.exp(-(values - minimum) / max(float(tau), 1e-8))
    return float(-float(tau) * np.log(np.mean(scaled)) + minimum)


@register_search_strategy("pbss")
class PrototypeSimilaritySearch(BaseSearchStrategy):
    """
    Prototype-Based Slide Similarity search for PBMS/PANTHER representations.

    PBSS compares each slide's PANTHER-adapted prototype means and diagonal
    covariances, then ranks candidate slides by ascending soft-min aggregated
    same-prototype Mahalanobis distance.
    """

    name = "pbss"
    supports = {"patch_vector"}
    supported_representation_kinds = frozenset({"patch_vector"})

    k = HyperParam(int, default=10, min=1, help="retrieval depth")
    eps = HyperParam(
        float,
        default=1e-6,
        min=1e-12,
        help="variance floor in diagonal Mahalanobis distance",
    )
    tau = HyperParam(
        float,
        default=0.5,
        min=1e-4,
        help="soft-min temperature for prototype-distance aggregation",
    )
    use_proto_labels = HyperParam(
        bool,
        default=True,
        help="exclude prototypes with prototype_labels == 'exclude'",
    )

    def build_database_item(
        self,
        representation: RetrievalRepresentation,
    ) -> SearchDatabaseItem:
        self._validate_representations([representation])
        return SearchDatabaseItem(
            sample_id=representation.sample_id,
            metadata=representation.metadata,
            exclusion_key=representation.exclusion_key,
            data=self._build_pbss_item(representation),
            additional_data=representation.additional_data,
        )

    def prepare_query(
        self,
        query_representation: RetrievalRepresentation,
    ) -> SearchDatabaseItem:
        self._validate_representations([query_representation])
        cached_item = getattr(self, "_database_items_by_sample_id", {}).get(
            str(query_representation.sample_id)
        )
        if cached_item is not None:
            return cached_item

        return SearchDatabaseItem(
            sample_id=query_representation.sample_id,
            metadata=query_representation.metadata,
            exclusion_key=query_representation.exclusion_key,
            data=self._build_pbss_item(query_representation),
            additional_data=query_representation.additional_data,
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

    def build_index(self) -> None:
        """Index database items for reuse when query and reference sets overlap."""
        self._database_items_by_sample_id = {
            item.sample_id: item for item in self.search_database
        }

    def search_prepared(
        self,
        query_item: SearchDatabaseItem,
        **kwargs: Any,
    ) -> SearchResult:
        """Run PBSS retrieval for one already-prepared query item."""
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
        _ = kwargs
        if not database_items:
            return []

        query_pbss = self._as_pbss_item(query_item)
        distances = np.array(
            [
                self._distance(query_pbss, self._as_pbss_item(database_item))
                for database_item in database_items
            ],
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
                    metadata=database_item.metadata,
                )
            )

        return hits

    def _distance(self, query: PBSSSearchItem, reference: PBSSSearchItem) -> float:
        joint_mask = self._joint_include_mask(query, reference)
        query_mean, query_cov = self._apply_mask(query, joint_mask)
        ref_mean, ref_cov = self._apply_mask(reference, joint_mask)

        if query_mean.size == 0 or ref_mean.size == 0:
            query_mean, query_cov = query.proto_mean, query.proto_cov
            ref_mean, ref_cov = reference.proto_mean, reference.proto_cov

        proto_distances = _per_proto_diag_mahalanobis(
            query_mean,
            query_cov,
            ref_mean,
            ref_cov,
            eps=float(self.eps),
        )
        return _softmin(proto_distances, tau=float(self.tau))

    def _joint_include_mask(
        self,
        query: PBSSSearchItem,
        reference: PBSSSearchItem,
    ) -> np.ndarray | None:
        if not bool(self.use_proto_labels):
            return None

        query_mask = query.include_mask
        reference_mask = reference.include_mask
        if query_mask is None and reference_mask is None:
            return None

        proto_count = min(query.proto_mean.shape[0], reference.proto_mean.shape[0])
        if query_mask is None:
            mask = reference_mask[:proto_count]
        elif reference_mask is None:
            mask = query_mask[:proto_count]
        else:
            mask = query_mask[:proto_count] & reference_mask[:proto_count]

        return mask if np.any(mask) else None

    def _apply_mask(
        self,
        item: PBSSSearchItem,
        mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        proto_count = int(item.proto_mean.shape[0])
        if mask is not None:
            proto_count = min(proto_count, int(mask.shape[0]))
        mean = item.proto_mean[:proto_count, :]
        cov = item.proto_cov[:proto_count, :]
        if mask is not None:
            clipped_mask = mask[:proto_count]
            mean = mean[clipped_mask]
            cov = cov[clipped_mask]
        return mean, cov

    def _build_pbss_item(
        self,
        representation: RetrievalRepresentation,
    ) -> PBSSSearchItem:
        additional_data = dict(representation.additional_data or {})
        proto_mean = self._required_matrix(
            additional_data,
            key="panther_proto_mean",
            sample_id=representation.sample_id,
        )
        proto_cov = self._required_matrix(
            additional_data,
            key="panther_proto_cov",
            sample_id=representation.sample_id,
        )
        if proto_mean.shape != proto_cov.shape:
            raise ValueError(
                "PBSS requires panther_proto_mean and panther_proto_cov to have "
                f"matching shapes for {representation.sample_id}. Got "
                f"{proto_mean.shape} and {proto_cov.shape}."
            )

        include_mask = None
        if bool(self.use_proto_labels):
            labels = additional_data.get("prototype_labels")
            if labels is not None:
                labels_array = np.asarray(labels, dtype=str).reshape(-1)
                if labels_array.shape[0] >= proto_mean.shape[0]:
                    include_mask = labels_array[: proto_mean.shape[0]] != "exclude"
                    if not np.any(include_mask):
                        include_mask = None

        return PBSSSearchItem(
            slide_id=str(representation.sample_id),
            exclusion_key=representation.exclusion_key,
            metadata=representation.metadata,
            proto_mean=proto_mean,
            proto_cov=proto_cov,
            include_mask=include_mask,
        )

    def _required_matrix(
        self,
        additional_data: dict[str, Any],
        *,
        key: str,
        sample_id: str,
    ) -> np.ndarray:
        if key not in additional_data:
            raise ValueError(
                f"PBSS requires pbms-features additional_data['{key}'] for "
                f"sample {sample_id}. Use retrieval_representation: pbms-features."
            )

        matrix = np.asarray(additional_data[key], dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(
                f"PBSS requires additional_data['{key}'] with shape (P, D) for "
                f"sample {sample_id}. Got {matrix.shape}."
            )
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(
                f"PBSS requires non-empty additional_data['{key}'] for sample "
                f"{sample_id}. Got {matrix.shape}."
            )
        return matrix

    def _as_pbss_item(self, item: SearchDatabaseItem) -> PBSSSearchItem:
        if not isinstance(item.data, PBSSSearchItem):
            raise TypeError(
                f"PBSS expected SearchDatabaseItem.data to be PBSSSearchItem for "
                f"{item.sample_id}."
            )
        return item.data
