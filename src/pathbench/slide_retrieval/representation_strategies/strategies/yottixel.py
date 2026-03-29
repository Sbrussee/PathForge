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
from sklearn.cluster import KMeans
import torch

from pathbench.core.experiments.combo_ids import build_tiling_id
from pathbench.core.datasets.bag_dataset import BagDataset, BagSample
from pathbench.slide_retrieval.hyperparams import HyperParam
from pathbench.slide_retrieval.io import load_sample_patch_coords
from pathbench.slide_retrieval.mean_rgb import resolve_sample_patch_mean_rgb
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
        bag: torch.Tensor,
        sample: BagSample,
        bag_dataset: BagDataset | None,
        combo_cfg: Any,
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
        bag_array = np.asarray(bag.detach().cpu().numpy(), dtype=float)
        tiling_id = (
            str(bag_dataset.tiling_id)
            if bag_dataset is not None
            else build_tiling_id(combo_cfg)
        )
        coords = load_sample_patch_coords(
            sample=sample,
            tile_id=tiling_id,
            dtype=np.int32,
        )
        self.random_state = self._resolve_random_state(combo_cfg)

        if bag_array.ndim != 2:
            raise ValueError(f"bag must have shape (N, D). Got {bag_array.shape}.")

        if int(bag_array.shape[0]) != int(coords.shape[0]):
            raise ValueError(
                "Bag rows and coordinate rows must match for Yottixel selection. "
                f"Got {bag_array.shape[0]} and {coords.shape[0]}."
            )

        if len(bag_array) == 0:
            logger.warning("Empty patch list provided to %s selection.", selection_label)

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
            representation_type=self.output_representation_kind,
            data=representation_data,
            additional_data={
                "selected_indices": selected_indices,
                "group_ids": group_ids.astype(np.int32, copy=False),
                "selected_coords": selected_coords,
                "bag_id": str(bag_id),
            },
        )


@register_representation_strategy("yottixel_rgb")
class YottixelRGB(_BaseYottixelRepresentationStrategy):
    """
    Yottixel RGB mosaic selection.

    This is a minimal port of the original two-stage Yottixel selector. In this
    repo, the first-stage color descriptors are resolved from persisted
    row-aligned per-patch mean RGB values stored in the slide H5 artifacts.

    Args:
        bag: `torch.Tensor` accepted for interface compatibility and ignored by
            this strategy after mean RGB descriptors are resolved.
        sample: `BagSample` describing the retrieval item and source artifacts.
        combo_cfg: Combo config exposing `tile_px` and `tile_mpp`.

        Returns:
            RetrievalRepresentation: Multi-vector representation with
        `data.shape == (K, 3)` for the selected patches.

    """

    name = "yottixel_rgb"

    def run(
        self,
        bag: torch.Tensor,
        sample: BagSample | None = None,
        **kwargs,
    ) -> RetrievalRepresentation:
        """
        Run the minimally adapted Yottixel RGB selection.

        Args:
            bag: Patch-level tensor with shape `(N, D_hist)`.
            sample: Bag sample carrying artifact paths for coordinate loading.
            **kwargs: Must include `combo_cfg`.

        Returns:
            RetrievalRepresentation: Selected patch rows plus Yottixel auxiliary
            arrays in `additional_data`.
        """
        if sample is None:
            raise ValueError("sample is required for Yottixel_rgb.")

        bag_dataset = kwargs.get("bag_dataset")
        combo_cfg = kwargs.get("combo_cfg")
        tiling_id = (
            str(bag_dataset.tiling_id)
            if bag_dataset is not None
            else build_tiling_id(combo_cfg)
        )
        resolved_mean_rgb = resolve_sample_patch_mean_rgb(
            sample=sample,
            bag_id=tiling_id,
            config=self.extra.get("config"),
        )
        bag_array, coords = self._prepare_selection_inputs(
            bag=torch.as_tensor(resolved_mean_rgb, dtype=torch.float32),
            sample=sample,
            bag_dataset=bag_dataset,
            combo_cfg=combo_cfg,
            selection_label="yottixel mean RGB",
        )
        if len(bag_array) == 0:
            return self._build_representation_from_indices(
                bag_array=bag_array,
                sample=sample,
                selected=[],
                group_ids=np.array([], dtype=np.int32),
                coords=coords,
                bag_id=tiling_id,
            )

        # Stage 1: cluster patches using mean RGB descriptors.
        mean_rgb = bag_array
        n_clusters = min(int(self.n_clusters), len(mean_rgb))
        kmeans_first_stage = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
        )
        first_stage_labels_raw = kmeans_first_stage.fit_predict(mean_rgb)
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
            bag_id=tiling_id,
        )


@register_representation_strategy("Yottixel_features")
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

    name = "Yottixel_features"

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

        bag_array, coords = self._prepare_selection_inputs(
            bag=bag,
            sample=sample,
            bag_dataset=kwargs.get("bag_dataset"),
            combo_cfg=kwargs.get("combo_cfg"),
            selection_label="Yottixel feature",
        )
        if len(bag_array) == 0:
            return self._build_representation_from_indices(
                bag_array=bag_array,
                sample=sample,
                selected=[],
                group_ids=np.array([], dtype=np.int32),
                coords=coords,
                bag_id=(
                    str(kwargs.get("bag_dataset").tiling_id)
                    if kwargs.get("bag_dataset") is not None
                    else build_tiling_id(kwargs.get("combo_cfg"))
                ),
            )

        # Stage 1: cluster patches using learned feature embeddings.
        patch_features = bag_array
        n_clusters = min(int(self.n_clusters), len(patch_features))
        kmeans_first_stage = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
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

            if n_select == 1:
                selected.append(int(member_idx[0]))
                continue

            kmeans_loc = KMeans(
                n_clusters=n_select,
                random_state=self.random_state,
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
            bag_id=(
                str(kwargs.get("bag_dataset").tiling_id)
                if kwargs.get("bag_dataset") is not None
                else build_tiling_id(kwargs.get("combo_cfg"))
            ),
        )
