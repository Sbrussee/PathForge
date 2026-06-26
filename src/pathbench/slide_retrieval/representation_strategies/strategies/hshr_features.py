from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans

from pathbench.core.experiments.combo_ids import build_tiling_id
from pathbench.core.datasets.bag_dataset import BagDataset, BagSample
from pathbench.core.io.slide_artifacts import features as features_io
from pathbench.core.io.slide_artifacts import tiles as tiles_io
from pathbench.core.io.slide_artifacts.base import FileHandleH5
from pathbench.slide_retrieval.hyperparams import HyperParam
from pathbench.slide_retrieval.representation_strategies.base import (
    BaseRetrievalRepresentationStrategy,
)
from pathbench.slide_retrieval.representation_strategies.registry import (
    register_representation_strategy,
)
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)

@register_representation_strategy("hshr-features")
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

    name = "hshr-features"
    n_patches = HyperParam(
        int,
        default=25,
        min=1,
        help="Desired number of patches to select (k in k-means).",
    )
    supported_feature_levels = frozenset({"patch"})
    output_representation_kind = "patch_vector"

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
        coords = np.asarray(kwargs.get("coords"), dtype=int)
        features = self._as_feature_matrix(bag)
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

    def load_sample(
        self,
        *,
        index: int,
        sample: BagSample,
        base_dataset: BagDataset,
    ) -> dict[str, Any]:
        """Load the bag tensor and aligned coordinates for one retrieval item."""
        del index

        bag_parts: list[np.ndarray] = []
        coord_parts: list[np.ndarray] = []
        for artifact_path in sample.artifact_paths:
            with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                feature_matrix = features_io.read_features(
                    slide_artifact,
                    bag_id=base_dataset.tiling_id,
                    extractor_name=base_dataset.extractor_name,
                )
                coords = tiles_io.read_coords(
                    slide_artifact,
                    bag_id=base_dataset.tiling_id,
                )
            bag_parts.append(np.asarray(feature_matrix, dtype=np.float32))
            coord_parts.append(np.asarray(coords[:, :2], dtype=int))

        if not bag_parts:
            raise RuntimeError(f"No bags found for sample '{sample.sample_id}'.")

        return {
            "bag": (
                np.concatenate(bag_parts, axis=0)
                if bag_parts
                else np.empty((0, 0), dtype=np.float32)
            ),
            "coords": (
                np.concatenate(coord_parts, axis=0)
                if coord_parts
                else np.empty((0, 2), dtype=int)
            ),
        }

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
            data=selected_features,
            additional_data={
                "selected_indices": selected_indices,
                "group_ids": group_ids.astype(int),
                "selected_coords": selected_coords,
                "bag_id": str(bag_id),
            },
        )

    def _as_feature_matrix(self, bag: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert one bag into a 2D float feature matrix with shape ``(N, D)``."""
        return self.as_numpy_feature_matrix(bag)

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
