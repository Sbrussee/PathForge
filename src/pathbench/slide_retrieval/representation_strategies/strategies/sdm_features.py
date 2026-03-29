from __future__ import annotations

# ------------------------------------------------------------------------------
# SDM (Selection of Distinct Morphologies):
#   Source Paper:
#     Shafique, A., Frohlich, K., Alsaafin, A., et al.
#     "Selection of Distinct Morphologies to Divide & Conquer Gigapixel Pathology
#     Images."
#     Medical Image Analysis (2023). DOI:10.1016/j.media.2023.102123
#   No official code release; reference implementation provided in this script.
# ------------------------------------------------------------------------------

import logging
from typing import Any

import numpy as np
import torch

from pathbench.core.experiments.combo_ids import build_tiling_id
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
from pathbench.slide_retrieval.types import RetrievalItemMetadata

logger = logging.getLogger(__name__)


@register_representation_strategy("sdm_features")
class SDMFeatures(BaseRetrievalRepresentationStrategy):
    """
    Selection of Distinct Morphologies (SDM) retrieval representation.

    Semantic goal:
        Preserve the PathBench 1.0 SDM morphology-selection logic while adapting
        it to the PathBench 2.0 slide-retrieval strategy interface. The strategy
        returns the selected patch features as a multi-vector representation and
        stores the original grouping outputs for downstream inspection.

    Inputs:
        bag (torch.Tensor):
            Patch feature tensor with shape ``(N, D)``.
        sample:
            Bag sample carrying ``sample_id`` and ``artifact_paths`` so aligned
            patch coordinates can be read from H5 tile storage.

    Outputs:
        RetrievalRepresentation:
            - ``representation_type``: ``"patch_vector"``
            - ``data``: selected feature matrix with shape ``(G, D)``
            - ``additional_data["selected_indices"]``: shape ``(G,)``
            - ``additional_data["group_ids"]``: shape ``(N,)``
            - ``additional_data["coords"]``: shape ``(N, 2)``

    Example:
        >>> strategy = SDMFeatures()
        >>> bag = torch.randn(8, 4)
        >>> # sample must expose sample_id and artifact_paths in real use.
    """

    name = "sdm_features"
    supported_feature_levels = frozenset({"patch"})
    output_representation_kind = "patch_vector"

    def run(
        self,
        bag: torch.Tensor,
        sample=None,
        **kwargs,
    ) -> RetrievalRepresentation:
        """
        Compute one SDM retrieval representation from a patch-level bag.

        Args:
            bag (torch.Tensor): Patch features with shape ``(N, D)``.
            sample: Sample object with ``sample_id`` and ``artifact_paths``.

        Returns:
            RetrievalRepresentation: SDM-selected patch feature representation.
        """
        random_state = self._resolve_random_state(kwargs.get("combo_cfg"))
        features, coords = self._prepare_selection_inputs(
            bag=bag,
            sample=sample,
            bag_dataset=kwargs.get("bag_dataset"),
            combo_cfg=kwargs.get("combo_cfg"),
        )

        if len(features) == 0:
            return self._build_representation_from_indices(
                sample=sample,
                features=np.empty((0, 0), dtype=np.float32),
                selected=[],
                group_ids=np.array([], dtype=np.int64),
                coords=np.empty((0, 2), dtype=np.int64),
                groups={},
                bag_id=(
                    str(kwargs.get("bag_dataset").tiling_id)
                    if kwargs.get("bag_dataset") is not None
                    else build_tiling_id(kwargs.get("combo_cfg"))
                ),
            )

        # Compute the global centroid and bin patches by rounded distance.
        if not np.all(np.isfinite(features)):
            raise ValueError("Non-finite values in features.")

        centroid = features.mean(axis=0)
        dists = np.linalg.norm(features - centroid[None, :], axis=1)
        raw_bin_ids = np.rint(dists).astype(int)
        unique_bins, group_ids = np.unique(raw_bin_ids, return_inverse=True)
        groups = {
            group_id: np.where(group_ids == group_id)[0]
            for group_id in range(len(unique_bins))
        }

        # Select one reproducible representative per morphology group.
        rng = np.random.default_rng(random_state)
        selected = [int(rng.choice(index_array)) for index_array in groups.values()]

        return self._build_representation_from_indices(
            sample=sample,
            features=features,
            selected=selected,
            group_ids=group_ids,
            coords=coords,
            groups=groups,
            bag_id=(
                str(kwargs.get("bag_dataset").tiling_id)
                if kwargs.get("bag_dataset") is not None
                else build_tiling_id(kwargs.get("combo_cfg"))
            ),
        )

    def _prepare_selection_inputs(
        self,
        *,
        bag: torch.Tensor,
        sample: Any,
        bag_dataset: Any | None,
        combo_cfg: Any | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare the aligned feature and coordinate arrays used by SDM.

        Args:
            bag (torch.Tensor): Patch feature tensor with shape ``(N, D)``.
            sample (Any): Sample object exposing ``sample_id`` and artifact paths.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                Feature matrix with shape ``(N, D)`` and coordinate matrix with
                shape ``(N, 2)``.
        """
        features = self._to_numpy_features(bag)
        coords = self._read_sample_coords(
            sample=sample,
            bag_dataset=bag_dataset,
            combo_cfg=combo_cfg,
        )

        if features.shape[0] != coords.shape[0]:
            raise ValueError(
                "SDM requires one coordinate row per patch feature row. "
                f"Got features={features.shape[0]} and coords={coords.shape[0]}."
            )

        if len(features) == 0:
            logger.warning("Empty patch list provided to SDM.")

        return features, coords

    def _build_representation_from_indices(
        self,
        *,
        sample: Any,
        features: np.ndarray,
        selected: list[int],
        group_ids: np.ndarray,
        coords: np.ndarray,
        groups: dict[int, np.ndarray],
        bag_id: str,
    ) -> RetrievalRepresentation:
        """
        Convert SDM grouping outputs into the repo retrieval container.

        Args:
            sample: Sample-like object carrying the retrieval item identifier.
            features (np.ndarray): Full feature matrix with shape ``(N, D)``.
            selected (list[int]): Selected representative row indices.
            group_ids (np.ndarray): Group assignments with shape ``(N,)``.
            coords (np.ndarray): Aligned coordinates with shape ``(N, 2)``.
            groups (dict[int, np.ndarray]): Mapping of group id to member indices.

        Returns:
            RetrievalRepresentation: Multi-vector output with selected feature rows
            and SDM auxiliary arrays.
        """
        selected_array = np.asarray(selected, dtype=np.int64)
        if selected:
            selected_features = np.asarray(
                features[selected_array],
                dtype=np.float32,
            )
            selected_coords = np.asarray(coords[selected_array], dtype=np.int64)
        else:
            feature_dim = 0 if features.ndim != 2 else int(features.shape[1])
            selected_features = np.empty((0, feature_dim), dtype=np.float32)
            selected_coords = np.empty((0, 2), dtype=np.int64)

        return RetrievalRepresentation(
            sample_id="" if sample is None else str(getattr(sample, "sample_id", "")),
            representation_type=self.output_representation_kind,
            data=selected_features,
            metadata=RetrievalItemMetadata(
                extra={
                    "groups": {
                        str(group_id): member_indices.tolist()
                        for group_id, member_indices in groups.items()
                    }
                }
            ),
            additional_data={
                "selected_indices": selected_array,
                "group_ids": group_ids.astype(np.int64, copy=False),
                "selected_coords": selected_coords,
                "bag_id": str(bag_id),
            },
        )

    def _resolve_random_state(self, combo_cfg: Any) -> int | None:
        """
        Resolve the experiment random state if it is available.

        Args:
            combo_cfg (Any): Combo-like config object.

        Returns:
            int | None: Configured random state or ``None``.
        """
        config = self.extra.get("config")
        if config is not None:
            experiment_cfg = getattr(config, "experiment", None)
            if experiment_cfg is not None:
                random_state = getattr(experiment_cfg, "random_state", None)
                if random_state is not None:
                    return random_state

        if combo_cfg is None:
            return None

        experiment_cfg = getattr(combo_cfg, "experiment", None)
        if experiment_cfg is None:
            return getattr(combo_cfg, "random_state", None)

        return getattr(experiment_cfg, "random_state", None)

    def _to_numpy_features(self, bag: torch.Tensor) -> np.ndarray:
        """
        Convert the input bag tensor into a 2D float numpy array.

        Args:
            bag (torch.Tensor): Patch feature tensor with shape ``(N, D)``.

        Returns:
            np.ndarray: Float feature matrix with shape ``(N, D)``.
        """
        if isinstance(bag, torch.Tensor):
            features = bag.detach().cpu().numpy()
        else:
            features = np.asarray(bag)
        features = np.asarray(features)

        if features.ndim != 2:
            raise ValueError(
                f"SDM expects a 2D patch feature matrix. Got shape {features.shape}."
            )

        return features

    def _read_sample_coords(
        self,
        sample: Any,
        bag_dataset: Any | None,
        combo_cfg: Any | None,
    ) -> np.ndarray:
        """
        Read and concatenate aligned patch coordinates for one sample.

        Args:
            sample (Any): Sample object exposing ``artifact_paths``.

        Returns:
            np.ndarray: Coordinate matrix with shape ``(N, 2)``.
        """
        if sample is None:
            raise ValueError(
                "SDM requires a sample with artifact_paths to load coordinates."
            )

        if bag_dataset is None and combo_cfg is None:
            raise ValueError(
                "SDM requires combo_cfg when bag_dataset is not provided."
            )

        tiling_id = (
            str(bag_dataset.tiling_id)
            if bag_dataset is not None
            else build_tiling_id(combo_cfg)
        )

        return load_sample_patch_coords(
            sample=sample,
            tile_id=tiling_id,
            dtype=np.int64,
        )
