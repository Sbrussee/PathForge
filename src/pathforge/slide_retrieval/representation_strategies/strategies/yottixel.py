from __future__ import annotations

# ------------------------------------------------------------------------------
# Yottixel Mosaic Selection:
#   Based on code from the official Yottixel repository:
#     https://github.com/KimiaLabMayo/yottixel
#   Source Paper:
#     Kalra, S., Tizhoosh, H.R., Choi, C., et al.
#     "Yottixel - An Image Search Engine for Large Archives of Histopathology
#     Whole Slide Images."
#     Medical Image Analysis 65 (2020): 101757.
#     https://doi.org/10.1016/j.media.2020.101757
#
# RetCCL-inspired Yottixel variant:
#   Source Paper:
#     Wang, X., Du, Y., Yang, S., et al.
#     "RetCCL: Clustering-Guided Contrastive Learning for Whole-Slide Image
#     Retrieval."
#     Medical Image Analysis 83 (2023): 102645.
#     https://doi.org/10.1016/j.media.2022.102645
# ------------------------------------------------------------------------------
import logging
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans

from pathforge.core.datasets.bag_dataset import BagDataset, BagSample
from pathforge.core.io.slide_artifacts import features as features_io
from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.slide_retrieval.hyperparams import HyperParam
from pathforge.slide_retrieval.representation_strategies.base import (
    BaseRetrievalRepresentationStrategy,
)
from pathforge.slide_retrieval.representation_strategies.histogram_rgb import (
    resolve_sample_patch_histogram_rgb,
)
from pathforge.slide_retrieval.representation_strategies.mean_rgb import (
    resolve_sample_patch_mean_rgb,
)
from pathforge.slide_retrieval.representation_strategies.registry import (
    register_representation_strategy,
)
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)

logger = logging.getLogger(__name__)


class _BaseYottixelRepresentationStrategy(BaseRetrievalRepresentationStrategy):
    """
    Base adapter for minimally ported Yottixel retrieval representations.

    Args:
        params: Hyperparameter mapping. Expected keys:
            - `n_clusters: int`
            - `perc_selected: float`

    Returns:
        None. Subclasses expose `run(...) -> RetrievalRepresentation`.

    """

    supported_feature_levels = frozenset({"patch"})
    output_representation_kind = "patch_vector"
    n_clusters = HyperParam(
        int,
        default=9,
        min=1,
        help="KMeans clusters (stage 1)",
    )
    perc_selected = HyperParam(
        float,
        default=1.0,
        min=0.0,
        max=100.0,
        help="Percent reps per group",
    )

    def __init__(self, params: dict[str, Any] | None = None, **kwargs) -> None:
        super().__init__(params=params, **kwargs)
        self.random_state = None

    def _resolve_random_state(self, combo_cfg: Any | None) -> int | None:
        """
        Resolve the experiment random state from the combo context when available.

        Args:
            combo_cfg: Combination config passed by the retrieval task.

        Returns:
            int | None: Random seed or `None` when unavailable.

        """
        _ = combo_cfg

        config = self.extra.get("config")
        experiment_config = getattr(config, "experiment", None)
        if experiment_config is None and isinstance(config, dict):
            experiment_config = config.get("experiment")

        if experiment_config is None:
            return None

        if isinstance(experiment_config, dict):
            return experiment_config.get("random_state")

        return getattr(experiment_config, "random_state", None)

    def _prepare_selection_inputs(
        self,
        *,
        bag: torch.Tensor | np.ndarray,
        sample: BagSample,
        coords: np.ndarray,
        selection_label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare the row-aligned bag and coordinate arrays for Yottixel selection.

        Args:
            bag: Patch-level bag tensor with shape `(N, D)`.
            sample: Bag sample identifying the aggregated retrieval item.
            combo_cfg: Combination config used to recover the tiling id.
            selection_label: Human-readable label for warnings.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                Feature matrix with shape `(N, D)` and coordinate array with
                shape `(N, 2)`.

        """
        bag_array = self.as_numpy_feature_matrix(bag)
        coords = np.asarray(coords, dtype=np.int32)

        if bag_array.ndim != 2:
            raise ValueError(f"bag must have shape (N, D). Got {bag_array.shape}.")

        if int(bag_array.shape[0]) != int(coords.shape[0]):
            raise ValueError(
                "Bag rows and coordinate rows must match for Yottixel selection. "
                f"Got {bag_array.shape[0]} and {coords.shape[0]}."
            )

        if len(bag_array) == 0:
            logger.warning(
                "Empty patch list provided to %s selection.", selection_label
            )

        return bag_array, np.asarray(coords, dtype=np.int32)

    def _build_representation_from_indices(
        self,
        *,
        bag_array: np.ndarray,
        sample: BagSample,
        selected: list[int],
        group_ids: np.ndarray,
        coords: np.ndarray,
        bag_id: str,
    ) -> RetrievalRepresentation:
        """
        Build one Yottixel retrieval representation from selected patch indices.

        Args:
            bag_array: Full patch feature matrix with shape `(N, D)`.
            sample: Bag sample identifying the retrieval item.
            selected: Global selected patch indices with length `(K,)`.
            group_ids: Compact first-stage cluster ids with shape `(N,)`.
            coords: Patch coordinate array with shape `(N, 2)`.

        Returns:
            RetrievalRepresentation: Multi-vector output with selected bag rows
            and Yottixel auxiliary arrays. Stored coordinates are restricted to
            the selected patches with shape `(K, 2)`.

        """
        if selected:
            representation_data = np.asarray(bag_array[selected], dtype=np.float32)
            selected_indices = np.asarray(selected, dtype=np.int32)
            selected_coords = np.asarray(coords[selected], dtype=np.int32)
        else:
            representation_data = np.empty((0, bag_array.shape[1]), dtype=np.float32)
            selected_indices = np.array([], dtype=np.int32)
            selected_coords = np.empty((0, 2), dtype=np.int32)

        return RetrievalRepresentation(
            sample_id=sample.sample_id,
            data=representation_data,
            additional_data={
                "selected_indices": selected_indices,
                "group_ids": group_ids.astype(np.int32, copy=False),
                "selected_coords": selected_coords,
                "bag_id": str(bag_id),
            },
        )


@register_representation_strategy("yottixel-rgb")
class YottixelRGB(_BaseYottixelRepresentationStrategy):
    """
    Yottixel RGB mosaic selection.

    This is a minimal port of the original two-stage Yottixel selector. In this
    repo, the first-stage colour descriptors are resolved from persisted,
    row-aligned RGB descriptors. The selected patch indices are then applied
    to the configured foundation-model feature bag used by Yottixel search.

    Args:
        bag: `torch.Tensor` or `np.ndarray` of patch-level foundation-model
            features shaped `(N, D)`.
        sample: `BagSample` describing the retrieval item and source artifacts.
        combo_cfg: Combo config exposing `tile_px` and `tile_mpp`.

        Returns:
            RetrievalRepresentation: Multi-vector representation with
        `data.shape == (K, D)` for selected foundation-model patch rows.

    """

    name = "yottixel-rgb"
    colour_descriptor = HyperParam(
        str,
        default="mean_rgb",
        choices=("mean_rgb", "histogram_rgb"),
        help=(
            "RGB descriptor used only for first-stage colour clustering. "
            "Selected patches always retain their foundation-model features."
        ),
    )

    def run(
        self,
        bag: torch.Tensor | np.ndarray,
        sample: BagSample | None = None,
        **kwargs,
    ) -> RetrievalRepresentation:
        """
        Run the minimally adapted Yottixel RGB selection.

        Args:
            bag: Patch-level foundation-model tensor with shape `(N, D)`.
            sample: Bag sample carrying artifact paths for coordinate loading.
            **kwargs: Must include `combo_cfg`, `coords` shaped `(N, 2)`,
                `tiling_id`, and `colour_descriptors` shaped `(N, 3)` for
                `mean_rgb` or `(N, 768)` for `histogram_rgb`.

        Returns:
            RetrievalRepresentation: Selected foundation-model rows shaped
            `(K, D)` plus Yottixel auxiliary arrays in `additional_data`.

        Example:
            >>> strategy = YottixelRGB({"colour_descriptor": "mean_rgb"})
            >>> representation = strategy.run(
            ...     bag=np.ones((2, 4)), sample=sample,
            ...     coords=np.array([[0, 0], [1, 1]]),
            ...     colour_descriptors=np.ones((2, 3)), tiling_id="256px_0.5mpp",
            ... )
        """
        if sample is None:
            raise ValueError("sample is required for Yottixel_rgb.")

        combo_cfg = kwargs.get("combo_cfg")
        self.random_state = self._resolve_random_state(combo_cfg)
        tiling_id = str(kwargs.get("tiling_id"))
        feature_bag = self.as_numpy_feature_matrix(bag)
        colour_descriptors = np.asarray(
            kwargs.get("colour_descriptors"), dtype=np.float32
        )
        selection_bag, coords = self._prepare_selection_inputs(
            bag=colour_descriptors,
            sample=sample,
            coords=np.asarray(kwargs.get("coords"), dtype=np.int32),
            selection_label=f"yottixel {self.colour_descriptor}",
        )
        if feature_bag.shape[0] != selection_bag.shape[0]:
            raise ValueError(
                "Foundation-model bag rows and RGB descriptor rows must match for "
                f"Yottixel RGB selection. Got {feature_bag.shape[0]} and "
                f"{selection_bag.shape[0]}."
            )
        if len(selection_bag) == 0:
            return self._build_representation_from_indices(
                bag_array=feature_bag,
                sample=sample,
                selected=[],
                group_ids=np.array([], dtype=np.int32),
                coords=coords,
                bag_id=tiling_id,
            )

        # Stage 1: cluster patches using the selected RGB descriptor.
        n_clusters = min(int(self.n_clusters), len(selection_bag))
        kmeans_first_stage = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        first_stage_labels_raw = kmeans_first_stage.fit_predict(selection_bag)
        unique_bins, group_ids = np.unique(first_stage_labels_raw, return_inverse=True)

        # Stage 2: within each RGB cluster, spatially pick representative patches.
        selected: list[int] = []
        for g in range(len(unique_bins)):
            member_idx = np.where(group_ids == g)[0]
            if member_idx.size == 0:
                continue

            cluster_coords = np.asarray(coords[member_idx], dtype=float)
            n_select = max(1, int(len(member_idx) * self.perc_selected / 100))
            kmeans_loc = KMeans(
                n_clusters=n_select,
                random_state=self.random_state,
                n_init=10,
            )
            dists = kmeans_loc.fit_transform(cluster_coords)

            used_local: set[int] = set()
            for c in range(n_select):
                sorted_local = np.argsort(dists[:, c])
                for sidx in sorted_local:
                    if int(sidx) not in used_local:
                        used_local.add(int(sidx))
                        selected.append(int(member_idx[int(sidx)]))
                        break

        return self._build_representation_from_indices(
            bag_array=feature_bag,
            sample=sample,
            selected=selected,
            group_ids=group_ids,
            coords=coords,
            bag_id=tiling_id,
        )

    def load_sample(
        self,
        *,
        index: int,
        sample: BagSample,
        base_dataset: BagDataset,
    ) -> dict[str, Any]:
        """Load row-aligned FM bags, RGB descriptors, and coordinates.

        Args:
            index: Position of `sample` in `base_dataset`; only used to satisfy
                the retrieval dataset loader contract.
            sample: Retrieval sample whose artifacts provide the patch rows.
            base_dataset: Dataset providing the canonical `tiling_id` and
                `extractor_name` used to read foundation-model features.

        Returns a payload with `bag` shaped `(N, D)`, `colour_descriptors`
        shaped `(N, 3)` or `(N, 768)`, and `coords` shaped `(N, 2)`. These
        arrays share exactly the stored patch-row order.

        Example:
            >>> payload = strategy.load_sample(
            ...     index=0, sample=sample, base_dataset=base_dataset
            ... )
            >>> payload["bag"].shape[0] == payload["colour_descriptors"].shape[0]
            True
        """
        _ = index
        tiling_id = str(base_dataset.tiling_id)
        feature_parts: list[np.ndarray] = []
        coord_parts: list[np.ndarray] = []
        for artifact_path in sample.artifact_paths:
            with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                feature_parts.append(
                    np.asarray(
                        features_io.read_features(
                            slide_artifact,
                            bag_id=tiling_id,
                            extractor_name=base_dataset.extractor_name,
                        ),
                        dtype=np.float32,
                    )
                )
                coords = tiles_io.read_coords(slide_artifact, bag_id=tiling_id)
            coord_parts.append(np.asarray(coords[:, :2], dtype=np.int32))

        resolver = (
            resolve_sample_patch_mean_rgb
            if self.colour_descriptor == "mean_rgb"
            else resolve_sample_patch_histogram_rgb
        )
        colour_descriptors = resolver(
            sample=sample,
            bag_id=tiling_id,
            config=self.extra.get("config"),
        )

        return {
            "bag": (
                np.concatenate(feature_parts, axis=0)
                if feature_parts
                else np.empty((0, 0), dtype=np.float32)
            ),
            "colour_descriptors": np.asarray(colour_descriptors, dtype=np.float32),
            "coords": (
                np.concatenate(coord_parts, axis=0)
                if coord_parts
                else np.empty((0, 2), dtype=np.int32)
            ),
            "tiling_id": tiling_id,
        }


@register_representation_strategy("yottixel-features")
class YottixelFeatures(_BaseYottixelRepresentationStrategy):
    """
    Yottixel-Features mosaic selection (RetCCL-inspired).

    This keeps the original first-stage feature clustering and second-stage
    spatial clustering unchanged, while adapting inputs to this repo's retrieval
    strategy interface.

    Args:
        bag: `torch.Tensor` with shape `(N, D)` containing patch embeddings.
        sample: `BagSample` describing the retrieval item and source artifacts.
        combo_cfg: Combo config exposing `tile_px` and `tile_mpp`.

    Returns:
        RetrievalRepresentation: Multi-vector representation with
        `data.shape == (K, D)` for the selected patches.

    Example:
        >>> strategy = YottixelFeatures(params={"n_clusters": 9})
        >>> strategy.output_representation_kind
        'patch_vector'
    """

    name = "yottixel-features"

    def run(
        self,
        bag: torch.Tensor,
        sample: BagSample | None = None,
        **kwargs,
    ) -> RetrievalRepresentation:
        """
        Run the minimally adapted RetCCL-inspired Yottixel selection.

        Args:
            bag: Patch-level tensor with shape `(N, D)`.
            sample: Bag sample carrying artifact paths for coordinate loading.
            **kwargs: Must include `combo_cfg`.

        Returns:
            RetrievalRepresentation: Selected patch rows plus Yottixel auxiliary
            arrays in `additional_data`.
        """
        if sample is None:
            raise ValueError("sample is required for Yottixel_features.")

        self.random_state = self._resolve_random_state(kwargs.get("combo_cfg"))
        bag_array, coords = self._prepare_selection_inputs(
            bag=bag,
            sample=sample,
            coords=np.asarray(kwargs.get("coords"), dtype=np.int32),
            selection_label="Yottixel feature",
        )
        if len(bag_array) == 0:
            return self._build_representation_from_indices(
                bag_array=bag_array,
                sample=sample,
                selected=[],
                group_ids=np.array([], dtype=np.int32),
                coords=coords,
                bag_id=str(kwargs.get("tiling_id")),
            )

        # Stage 1: cluster patches using learned feature embeddings.
        patch_features = bag_array
        n_clusters = min(int(self.n_clusters), len(patch_features))
        kmeans_first_stage = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        first_stage_labels_raw = kmeans_first_stage.fit_predict(patch_features)
        unique_bins, group_ids = np.unique(first_stage_labels_raw, return_inverse=True)

        # Stage 2: within each feature cluster, spatially pick representatives.
        selected: list[int] = []
        for g in range(len(unique_bins)):
            member_idx = np.where(group_ids == g)[0]
            if member_idx.size == 0:
                continue

            cluster_coords = np.asarray(coords[member_idx], dtype=float)
            n_select = max(1, int(len(member_idx) * self.perc_selected / 100))

            kmeans_loc = KMeans(
                n_clusters=n_select,
                random_state=self.random_state,
                n_init=10,
            )
            dists = kmeans_loc.fit_transform(cluster_coords)

            used_local: set[int] = set()
            for c in range(n_select):
                sorted_local = np.argsort(dists[:, c])
                for sidx in sorted_local:
                    if int(sidx) not in used_local:
                        used_local.add(int(sidx))
                        selected.append(int(member_idx[int(sidx)]))
                        break

        return self._build_representation_from_indices(
            bag_array=bag_array,
            sample=sample,
            selected=selected,
            group_ids=group_ids,
            coords=coords,
            bag_id=str(kwargs.get("tiling_id")),
        )

    def load_sample(
        self,
        *,
        index: int,
        sample: BagSample,
        base_dataset: BagDataset,
    ) -> dict[str, Any]:
        """Load feature bags and coordinates for one Yottixel feature item."""
        del index
        tiling_id = str(base_dataset.tiling_id)
        bag_parts: list[np.ndarray] = []
        coord_parts: list[np.ndarray] = []
        for artifact_path in sample.artifact_paths:
            with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                feature_matrix = features_io.read_features(
                    slide_artifact,
                    bag_id=tiling_id,
                    extractor_name=base_dataset.extractor_name,
                )
                coords = tiles_io.read_coords(
                    slide_artifact,
                    bag_id=tiling_id,
                )
            bag_parts.append(np.asarray(feature_matrix, dtype=np.float32))
            coord_parts.append(np.asarray(coords[:, :2], dtype=np.int32))

        return {
            "bag": (
                np.concatenate(bag_parts, axis=0)
                if bag_parts
                else np.empty((0, 0), dtype=np.float32)
            ),
            "coords": (
                np.concatenate(coord_parts, axis=0)
                if coord_parts
                else np.empty((0, 2), dtype=np.int32)
            ),
            "tiling_id": tiling_id,
        }
