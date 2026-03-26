from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans

from pathbench.core.experiments.combo_ids import build_tiling_id
from pathbench.core.datasets.bag_dataset import BagDataset, BagSample
from pathbench.slide_retrieval.hyperparams import HyperParam
from pathbench.slide_retrieval.io import load_sample_patch_coords
from pathbench.slide_retrieval.representation_strategies.base import (
    BaseRetrievalRepresentationStrategy,
)
from pathbench.slide_retrieval.representation_strategies.registry import (
    register_representation_strategy,
)
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)


logger = logging.getLogger(__name__)


@register_representation_strategy("hshr_features")
class HSHRFeatures(BaseRetrievalRepresentationStrategy):
    """
    HSHR-Features retrieval representation.

    Semantic goal:
        Cluster patch-level features with k-means and select the single patch
        nearest to each cluster centroid.

    Inputs:
        bag:
            Feature matrix with shape ``(N, D)`` as ``torch.Tensor`` or
            ``numpy.ndarray``.
        sample:
            Optional ``BagSample`` used for ``sample_id`` and artifact-path
            lookup.
        bag_dataset:
            Optional ``BagDataset`` used to resolve ``tiling_id`` when loading
            coordinates.

    Outputs:
        Returns a ``RetrievalRepresentation`` with:
        - ``representation_type="multi_vector"``
        - ``data`` of shape ``(K, D)``
        - ``additional_data["selected_indices"]`` of shape ``(K,)``
        - ``additional_data["group_ids"]`` of shape ``(N,)``
        - ``additional_data["coords"]`` of shape ``(N, 2)``

    Example:
        ```python
        strategy = HSHRFeatures(params={"n_patches": 4})
        representation = strategy.run(
            bag=torch.randn(32, 512),
            sample=sample,
            bag_dataset=bag_dataset,
        )
        ```

    Notes:
        This is a minimal PathBench 1.0 port. The clustering and representative
        patch selection logic are kept intentionally close to the original code.
    """

    name = "hshr_features"
    n_patches = HyperParam(
        int,
        default=25,
        min=1,
        help="Desired number of patches to select (k in k-means).",
    )
    supported_feature_levels = frozenset({"patch"})
    output_representation_kind = "multi_vector"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(params=params, **kwargs)
        self.config = kwargs.get("config")
        self.random_state = self._resolve_random_state(self.config)

    def run(
        self,
        bag: torch.Tensor | np.ndarray,
        sample: BagSample | None = None,
        **kwargs: Any,
    ) -> RetrievalRepresentation:
        """
        Compute the HSHR feature-selected multi-vector representation.

        Inputs:
            bag:
                Feature matrix with shape ``(N, D)``.
            sample:
                Optional sample metadata for ``sample_id`` and coordinate lookup.

        Output:
            Retrieval representation with ``data`` shape ``(K, D)`` where
            ``K = min(max(n_patches, 1), N)``.
        """
        features, coords = self._prepare_selection_inputs(
            bag=bag,
            sample=sample,
            bag_dataset=kwargs.get("bag_dataset"),
            combo_cfg=kwargs.get("combo_cfg"),
        )
        bag_id = (
            str(kwargs.get("bag_dataset").tiling_id)
            if kwargs.get("bag_dataset") is not None
            else build_tiling_id(kwargs.get("combo_cfg"))
        )
        if len(features) == 0:
            return self._build_representation_from_indices(
                sample=sample,
                features=np.empty((0, 0), dtype=np.float32),
                selected=[],
                group_ids=np.array([], dtype=int),
                coords=np.empty((0, 2), dtype=int),
                bag_id=bag_id,
            )

        # Cluster feature rows and choose one member nearest each centroid.
        k = int(self.n_patches) if self.n_patches is not None else 1
        k = max(1, min(k, len(features)))
        km = KMeans(
            n_clusters=k,
            n_init="auto",
            random_state=self.random_state,
        )
        labels = km.fit_predict(features)
        centers = km.cluster_centers_
        unique_bins, group_ids = np.unique(labels, return_inverse=True)
        groups = {g: np.where(group_ids == g)[0] for g in range(len(unique_bins))}

        selected: list[int] = []
        for g, member_idx in groups.items():
            if member_idx.size == 0:
                continue
            feats_g = features[member_idx]
            center_g = centers[g]
            d2 = np.sum((feats_g - center_g) ** 2, axis=1)
            pick_local = int(np.argmin(d2))
            selected.append(int(member_idx[pick_local]))

        return self._build_representation_from_indices(
            sample=sample,
            features=features,
            selected=selected,
            group_ids=group_ids,
            coords=coords,
            bag_id=bag_id,
        )

    def _prepare_selection_inputs(
        self,
        *,
        bag: torch.Tensor | np.ndarray,
        sample: BagSample | None,
        bag_dataset: BagDataset | None,
        combo_cfg: Any | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare the aligned feature and coordinate arrays used by HSHR.

        Outputs:
            tuple[np.ndarray, np.ndarray]:
                Feature matrix with shape ``(N, D)`` and aligned coordinate
                matrix with shape ``(N, 2)``.
        """
        features = self._as_feature_matrix(bag)
        if len(features) == 0:
            logger.warning("HSHRFeatures: empty patch list.")
            return features, np.empty((0, 2), dtype=int)

        coords = self._load_coords(
            sample=sample,
            bag_dataset=bag_dataset,
            combo_cfg=combo_cfg,
            num_patches=int(features.shape[0]),
        )
        return features, coords

    def _build_representation_from_indices(
        self,
        *,
        sample: BagSample | None,
        features: np.ndarray,
        selected: list[int],
        group_ids: np.ndarray,
        coords: np.ndarray,
        bag_id: str,
    ) -> RetrievalRepresentation:
        """
        Convert HSHR centroid-selection outputs into a retrieval representation.

        Outputs:
            RetrievalRepresentation with selected features, compact group ids,
            and selected patch coordinates.
        """
        selected_indices = np.asarray(selected, dtype=int)
        if selected:
            selected_features = features[selected_indices].astype(np.float32, copy=False)
            selected_coords = np.asarray(coords[selected_indices], dtype=int)
        else:
            feature_dim = 0 if features.ndim != 2 else int(features.shape[1])
            selected_features = np.empty((0, feature_dim), dtype=np.float32)
            selected_coords = np.empty((0, 2), dtype=int)

        return RetrievalRepresentation(
            sample_id=self._resolve_sample_id(sample=sample),
            representation_type="multi_vector",
            data=selected_features,
            additional_data={
                "selected_indices": selected_indices,
                "group_ids": group_ids.astype(int),
                "selected_coords": selected_coords,
                "bag_id": str(bag_id),
            },
        )

    def _load_coords(
        self,
        *,
        sample: BagSample | None,
        bag_dataset: BagDataset | None,
        combo_cfg: Any | None,
        num_patches: int,
    ) -> np.ndarray:
        """
        Load patch coordinates aligned with the bag rows.

        Outputs:
            Integer array with shape ``(N, 2)``. Falls back to zeros with the
            same shape when coordinates are unavailable in the current context.
        """
        if sample is None:
            return np.zeros((num_patches, 2), dtype=int)

        if bag_dataset is None and combo_cfg is None:
            raise ValueError(
                "HSHRFeatures requires combo_cfg when bag_dataset is not provided."
            )

        tiling_id = (
            str(bag_dataset.tiling_id)
            if bag_dataset is not None
            else build_tiling_id(combo_cfg)
        )
        all_coords = load_sample_patch_coords(
            sample=sample,
            bag_id=tiling_id,
        )
        if int(all_coords.shape[0]) != num_patches:
            raise ValueError(
                "HSHRFeatures: number of coords does not match number of bag rows. "
                f"Got {all_coords.shape[0]} coords and {num_patches} features."
            )
        return all_coords

    def _as_feature_matrix(self, bag: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert one bag into a 2D float feature matrix with shape ``(N, D)``."""
        if isinstance(bag, torch.Tensor):
            features = bag.detach().cpu().numpy()
        else:
            features = np.asarray(bag)

        if features.ndim != 2:
            raise ValueError(
                f"HSHRFeatures expects a 2D feature matrix with shape (N, D). Got {features.shape}."
            )

        return np.asarray(features, dtype=float)

    def _resolve_sample_id(self, *, sample: BagSample | None) -> str:
        """Resolve the sample identifier used in the retrieval representation."""
        if sample is None:
            return ""
        return str(sample.sample_id)

    def _resolve_random_state(self, config: Any) -> int | None:
        """Resolve the experiment random state from repo config when available."""
        if config is None:
            return None

        if isinstance(config, dict):
            experiment_cfg = config.get("experiment", {}) or {}
            value = experiment_cfg.get("random_state", None)
            return None if value is None else int(value)

        experiment_cfg = getattr(config, "experiment", None)
        if experiment_cfg is None:
            return None

        value = getattr(experiment_cfg, "random_state", None)
        return None if value is None else int(value)
