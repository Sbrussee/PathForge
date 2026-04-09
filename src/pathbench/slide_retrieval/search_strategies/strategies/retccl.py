from __future__ import annotations

# ------------------------------------------------------------------------------
# RETCCL Retrieval:
#   Minimal refactor of the PathBench 1.0 RETCCL-style search strategy.
#   The ranking procedure is intentionally kept close to the original code path.
#
#   Source Paper:
#     Wang, X., Du, Y., Yang, S., et al.
#     "RetCCL: Clustering-Guided Contrastive Learning for Whole-Slide Image
#     Retrieval."
#     Medical Image Analysis 83 (2023): 102645.
#     https://doi.org/10.1016/j.media.2022.102645
# ------------------------------------------------------------------------------

from collections import Counter
from dataclasses import dataclass
from statistics import mean, mode
from typing import Any

import numpy as np
from numpy.linalg import norm

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
)


def _safe_mean(values: list[float]) -> float:
    """
    Compute a float mean while preserving the PathBench 1.0 empty-list fallback.

    Inputs:
        values (list[float]): Numeric values with shape ``(N,)``.

    Outputs:
        float: ``np.mean(values)`` when ``N > 0``, else ``0.0``.

    Example:
        >>> _safe_mean([0.5, 1.0])
        0.75
    """
    return float(np.mean(values)) if len(values) else 0.0


@dataclass(slots=True)
class RetCCLSearchItem:
    """
    Keep RETCCL search state for one slide together in one record.

    Inputs:
        slide_id (str): Aggregated retrieval item identifier.
        patient_id (str | None): Patient identifier used for LOPO-style filtering.
        label (str | None): Slide category / diagnosis label.
        features (np.ndarray): Feature matrix with shape ``(N, D)``.
        metadata (dict[str, Any]): Original retrieval metadata.

    Outputs:
        None. This dataclass is consumed internally by ``RetCCLSearch``.

    Example:
        >>> item = RetCCLSearchItem(
        ...     slide_id="slide-1",
        ...     patient_id="patient-1",
        ...     label="a",
        ...     features=np.ones((2, 3), dtype=float),
        ...     metadata={},
        ... )
        >>> item.features.shape
        (2, 3)
    """

    slide_id: str
    exclusion_key: str | None
    features: np.ndarray


@register_search_strategy("retccl")
class RetCCLSearch(BaseSearchStrategy):
    """
    RETCCL-based multi-vector retrieval adapted to the PathBench 2.0 interface.

    Semantic goal:
        Preserve the original PathBench 1.0 RETCCL ranking logic as closely as
        possible. The database is flattened to patch level, query-patch bags are
        filtered by cosine threshold, ordered by weighted entropy, pruned by the
        original eta threshold, and converted back to slide-level ranked hits.

    Inputs:
        database representations:
            ``RetrievalRepresentation`` items with ``representation_type ==
            "multi_vector"`` and ``data`` shape ``(N, D)``.
        query representation:
            One ``RetrievalRepresentation`` with the same shape convention.

    Outputs:
        ``SearchHit`` list ranked by descending average similarity. Each hit
        score is the original RETCCL ``avg_sim`` value.

    Example:
        >>> strategy = RetCCLSearch(params={"k": 2})
        >>> strategy.hyperparam_values()["k"]
        2
    """

    name = "retccl"
    supported_representation_kinds = frozenset({"multi_vector", "patch_vector"})
    k = HyperParam(int, default=5, min=1, help="retrieval depth")
    cosine_threshold = HyperParam(
        float,
        default=0.7,
        min=0.0,
        max=1.0,
        help="min cosine similarity to accept a patch match",
    )
    class_weight_factor = HyperParam(
        float,
        default=10.0,
        min=0.0,
        help="inverse-frequency reweighting strength",
    )
    topk_per_patch = HyperParam(
        int,
        default=5,
        min=1,
        help="how many top sims per patch to use",
    )

    def build_index(self) -> None:
        """
        Build the in-memory slide index and class weights from the search database.

        Inputs:
            None. Uses ``self.search_database`` populated by ``build_database``.

        Outputs:
            None. Sets:
            - ``self.slide_index: dict[str, RetCCLSearchItem]``
            - ``self.class_weight: dict[str | None, float]``
        """
        self.slide_index: dict[str, RetCCLSearchItem] = {}

        for item in self.search_database:
            features = self._as_feature_matrix(
                representation_data=item.data,
                item_id=item.item_id,
            )
            self.slide_index[item.item_id] = RetCCLSearchItem(
                slide_id=item.item_id,
                exclusion_key=item.exclusion_key,
                features=features,
            )

        self.class_weight = self._compute_class_weights(
            factor=self.class_weight_factor
        )

    def build_database_item(
        self,
        representation: RetrievalRepresentation,
    ) -> SearchDatabaseItem:
        """
        Convert one retrieval representation into a searchable RETCCL item.

        Inputs:
            representation (RetrievalRepresentation):
                Retrieval item with ``data`` shape ``(N, D)``.

        Outputs:
            SearchDatabaseItem: Standard search-database container.
        """
        features = self._as_feature_matrix(
            representation_data=representation.data,
            item_id=representation.sample_id,
        )
        return SearchDatabaseItem(
            sample_id=representation.sample_id,
            exclusion_key=representation.exclusion_key,
            data=features,
        )

    def rank(
        self,
        query_item: SearchDatabaseItem,
        database_items: list[SearchDatabaseItem],
        **kwargs: Any,
    ) -> list[SearchHit]:
        """
        Rank database slides for one query using the original RETCCL procedure.

        Inputs:
            query_item (SearchDatabaseItem):
                Query item with ``data`` shape ``(N_q, D)``.
            database_items (list[SearchDatabaseItem]):
                Candidate database items, typically already filtered by patient.

        Outputs:
            list[SearchHit]:
                Ranked hits ordered by descending average similarity.

        Example:
            >>> strategy = RetCCLSearch(params={"k": 1})
            >>> strategy.rank  # doctest: +ELLIPSIS
            <bound method RetCCLSearch.rank of ...>
        """
        _ = kwargs

        query_slide = RetCCLSearchItem(
            slide_id=query_item.item_id,
            exclusion_key=query_item.exclusion_key,
            features=self._as_feature_matrix(
                representation_data=query_item.data,
                item_id=query_item.item_id,
            ),
        )

        candidate_ids = {item.item_id for item in database_items}
        flat_feats: list[np.ndarray] = []
        flat_meta: list[tuple[str, int]] = []

        for slide_id, slide in self.slide_index.items():
            if slide_id not in candidate_ids:
                continue
            if slide.slide_id == query_slide.slide_id:
                continue

            features = slide.features
            for patch_idx in range(features.shape[0]):
                flat_feats.append(features[patch_idx])
                flat_meta.append((slide.slide_id, patch_idx))

        if not flat_feats:
            return []

        flat_feats_array = np.asarray(flat_feats, dtype=float)
        flat_norms = norm(flat_feats_array, axis=1) + 1e-12

        bag_matches: dict[int, list[tuple[int, float]]] = {}
        bag_entropy: dict[int, float] = {}

        for patch_idx, query_feature in enumerate(query_slide.features):
            query_norm = norm(query_feature) + 1e-12
            similarities = (flat_feats_array @ query_feature) / (flat_norms * query_norm)

            mask = similarities >= self.cosine_threshold
            matched_indices = np.where(mask)[0]
            bag = [(int(index), float(similarities[index])) for index in matched_indices]
            bag.sort(key=lambda value: value[1], reverse=True)
            bag_matches[patch_idx] = bag

            if not bag:
                bag_entropy[patch_idx] = 0.0
                continue

            weight_sums: dict[str | None, float] = {}
            total_weight = 0.0
            for flat_index, similarity in bag:
                slide_id, _patch_index = flat_meta[flat_index]
                label = None
                weight_similarity = (
                    ((similarity + 1.0) / 2.0) * self.class_weight.get(label, 0.0)
                )
                weight_sums[label] = weight_sums.get(label, 0.0) + weight_similarity
                total_weight += weight_similarity

            entropy = 0.0
            for weight in weight_sums.values():
                probability = weight / total_weight
                entropy -= probability * np.log(probability)
            bag_entropy[patch_idx] = entropy

        bag_matches = dict(
            sorted(
                bag_matches.items(),
                key=lambda item: bag_entropy[item[0]],
                reverse=True,
            )
        )

        num_query_patches = query_slide.features.shape[0]
        eta_threshold = 0.0
        for patch_idx in range(num_query_patches):
            topk_scores = [
                score
                for _flat_index, score in bag_matches.get(patch_idx, [])[
                    : self.topk_per_patch
                ]
            ]
            eta_threshold += _safe_mean(topk_scores)
        eta_threshold /= max(num_query_patches, 1)

        drop_ids: list[int] = []
        for patch_idx, bag in bag_matches.items():
            topk_scores = [score for _flat_index, score in bag[: self.topk_per_patch]]
            if _safe_mean(topk_scores) < eta_threshold:
                drop_ids.append(patch_idx)
        for patch_idx in drop_ids:
            del bag_matches[patch_idx]

        wsi_retrieval: dict[str, tuple[str, float, float]] = {}
        for _patch_idx, bag in bag_matches.items():
            topk_matches = bag[: self.topk_per_patch]
            if not topk_matches:
                continue

            match_labels: list[str | None] = []
            match_slides: list[str] = []
            similarities: list[float] = []

            for flat_index, similarity in topk_matches:
                slide_id, _patch_index = flat_meta[flat_index]
                match_labels.append(None)
                match_slides.append(slide_id)
                similarities.append(similarity)

            matched_label = mode(match_labels)
            chosen_index = match_labels.index(matched_label)
            chosen_slide = match_slides[chosen_index]
            if chosen_slide not in wsi_retrieval:
                wsi_retrieval[chosen_slide] = (
                    chosen_slide,
                    similarities[chosen_index],
                    mean(similarities),
                )

        sorted_hits = sorted(
            wsi_retrieval.values(),
            key=lambda value: value[2],
            reverse=True,
        )

        hits: list[SearchHit] = []
        for rank, (slide_id, _score, avg_similarity) in enumerate(
            sorted_hits[: self.k],
            start=1,
        ):
            hits.append(
                SearchHit(
                    sample_id=slide_id,
                    score=float(avg_similarity),
                    rank=rank,
                )
            )

        return hits

    def _compute_class_weights(
        self,
        factor: float = 10.0,
    ) -> dict[str | None, float]:
        """
        Mirror the original RETCCL inverse-frequency class reweighting.

        Inputs:
            factor (float): Normalization factor for the weight sum.

        Outputs:
            dict[str | None, float]:
                Mapping ``label -> normalized inverse-frequency weight``.
        """
        _ = factor
        return {None: 1.0}

    def _as_feature_matrix(
        self,
        *,
        representation_data: Any,
        item_id: str,
    ) -> np.ndarray:
        """
        Normalize RETCCL feature storage to a 2D float matrix.

        Inputs:
            representation_data (Any):
                Retrieval representation data with shape ``(N, D)`` or ``(D,)``.
            item_id (str): Identifier used in validation errors.

        Outputs:
            np.ndarray: Float feature matrix with shape ``(N, D)``.

        Example:
            >>> strategy = RetCCLSearch()
            >>> strategy._as_feature_matrix(
            ...     representation_data=np.array([1.0, 2.0]),
            ...     item_id="q",
            ... ).shape
            (1, 2)
        """
        features = np.asarray(representation_data, dtype=float)

        if features.ndim == 1:
            features = features[None, :]

        if features.ndim != 2:
            raise ValueError(
                "RetCCLSearch expects retrieval items with shape (N, D). "
                f"Got {features.shape} for item '{item_id}'."
            )

        return features
