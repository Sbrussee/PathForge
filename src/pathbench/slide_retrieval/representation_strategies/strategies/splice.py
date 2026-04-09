from __future__ import annotations

# ------------------------------------------------------------------------------
# SPLICE (Streaming Mosaic Selection):
#   Source Paper:
#     Alsaafin, A., Nejat, P., Shafique, A., et al.
#     "SPLICE -- Streamlining Digital Pathology Image Processing." arXiv, 2024.
#     https://doi.org/10.48550/arXiv.2404.17704
#   No official GitHub available.
# ------------------------------------------------------------------------------

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from pathbench.core.datasets.bag_dataset import BagDataset, BagSample
from pathbench.core.io.slide_artifacts import features as features_io
from pathbench.core.io.slide_artifacts import tiles as tiles_io
from pathbench.core.io.slide_artifacts.base import FileHandleH5
from pathbench.slide_retrieval.hyperparams import HyperParam
from pathbench.slide_retrieval.mean_rgb import (
    _build_slide_processor,
    _resolve_sample_slide_paths,
    _slide_retrieval_artifact_path,
    load_or_create_slide_patch_mean_rgb,
)
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


class _BaseSPLICEStrategy(BaseRetrievalRepresentationStrategy):
    """Shared compatibility contract for SPLICE retrieval representations."""

    supported_feature_levels = frozenset({"patch"})
    output_representation_kind = "patch_vector"
    percentile_threshold = HyperParam(
        float,
        default=25.0,
        min=0.0,
        max=100.0,
        help="Percentile (0-100) of distances used as suppression threshold.",
    )

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(params=params, **kwargs)

    def _prepare_selection_inputs(
        self,
        *,
        features: np.ndarray,
        coords: np.ndarray,
        selection_label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare the row-aligned arrays used by one SPLICE selector.

        Inputs:
        - features: `np.ndarray` with shape `(N, D)` containing the original bag.
        - coords: `np.ndarray` with shape `(N, 2)` containing `(x, y)` locations.
        - selection_label: human-readable selector label used in warnings.

        Returns:
        - tuple[np.ndarray, np.ndarray]:
          validated feature matrix and validated coordinate matrix.
        """
        if features.ndim != 2:
            raise ValueError(f"SPLICE expects a 2D bag matrix. Got shape {features.shape}.")

        if coords.shape[0] != features.shape[0]:
            raise ValueError(
                "SPLICE requires coords rows to match bag rows. "
                f"Got coords={coords.shape[0]} and bag={features.shape[0]}."
            )

        if len(features) == 0:
            logger.warning("Empty patch list provided to %s.", selection_label)

        return features, coords

    def _build_representation_from_indices(
        self,
        *,
        sample: BagSample | None,
        features: np.ndarray,
        selected: list[int],
        group_ids: np.ndarray,
        coords: np.ndarray,
        groups: dict[int, np.ndarray],
        bag_id: str,
    ) -> RetrievalRepresentation:
        """
        Convert the SPLICE selection outputs into the repo retrieval container.

        Inputs:
        - sample: optional sample carrying the retrieval item identifier.
        - features: `np.ndarray` with shape `(N, D)` containing the original bag.
        - selected: `list[int]` of representative row indices.
        - group_ids: `np.ndarray` with shape `(N,)` mapping rows to group ids.
        - coords: `np.ndarray` with shape `(N, 2)` containing `(x, y)` locations.
        - groups: `dict[int, np.ndarray]` mapping each group id to member indices.

        Returns:
        - `RetrievalRepresentation` with:
          - `data`: selected feature rows, shape `(K, D)`
          - `additional_data["selected_indices"]`: shape `(K,)`
          - `additional_data["group_ids"]`: shape `(N,)`
          - `additional_data["selected_coords"]`: shape `(K, 2)`
        """
        selected_array = np.asarray(selected, dtype=int)
        selected_coords = np.asarray(coords[selected_array], dtype=np.int64)
        groups_payload = {
            str(group_id): [int(member) for member in members.tolist()]
            for group_id, members in groups.items()
        }

        return RetrievalRepresentation(
            sample_id="" if sample is None else str(sample.sample_id),
            data=features[selected_array].astype(np.float32, copy=False),
            additional_data={
                "selected_indices": selected_array.astype(np.int64, copy=False),
                "group_ids": group_ids.astype(np.int64, copy=False),
                "selected_coords": selected_coords,
                "bag_id": str(bag_id),
                "groups": groups_payload,
                "selector_name": self.name,
            },
        )

    @staticmethod
    def _resolve_combo_cfg(combo_cfg: Any) -> Any:
        """
        Normalize the combo config used to reconstruct the H5 tiling identifier.

        Inputs:
        - combo_cfg: object with `tile_px` and `tile_mpp`, or `None`.

        Returns:
        - object exposing `tile_px: int` and `tile_mpp: float`.

        Example:
        ```python
        combo_cfg = self._resolve_combo_cfg(combo_cfg)
        ```
        """
        if combo_cfg is None:
            raise ValueError("combo_cfg must be provided for SPLICE retrieval strategies.")

        return SimpleNamespace(
            tile_px=int(combo_cfg.tile_px),
            tile_mpp=float(combo_cfg.tile_mpp),
        )


@register_representation_strategy("splice_rgb")
class SPLICERGB(_BaseSPLICEStrategy):
    """
    SPLICE RGB Mosaic Selection
    --------------------------
    Applies the SPLICE algorithm using patch-level mean RGB descriptors to reduce
    redundancy
    among selected patches. This method iteratively selects patches and excludes
    neighboring ones with similar color profiles, ensuring a diverse subset.

    The SPLICE method works in a streaming fashion, computing the Euclidean
    distance between patches and suppressing nearby redundant patches within a
    distance threshold determined by a user-defined percentile.

    Inputs:
    - bag: `torch.Tensor | np.ndarray` with shape `(N, D)`, where each row is the
      normalized mean RGB vector for one patch.
    - sample: `BagSample` with `sample_id: str`, `artifact_paths: list[Path]`,
      and slide membership metadata.
    - combo_cfg: object with `tile_px: int` and `tile_mpp: float`.

    Returns:
    - `RetrievalRepresentation` with:
      - `data`: selected mean RGB rows with shape `(K, 3)`
      - `additional_data["selected_indices"]`: shape `(K,)`
      - `additional_data["group_ids"]`: shape `(N,)`
      - `additional_data["selected_coords"]`: shape `(K, 2)`
      - `additional_data["groups"]`: group-to-member index mapping

    Example:
    ```python
    strategy = SPLICERGB(params={"percentile_threshold": 25.0})
    representation = strategy.run(bag=rgb_histograms, sample=sample, combo_cfg=combo_cfg)
    ```

    Reference:
        Alsaafin, Areej, Peyman Nejat, Abubakr Shafique, Jibran Khan, Saghir
        Alfasly, Ghazal Alabtah, and H. R. Tizhoosh. "SPLICE -- Streamlining
        Digital Pathology Image Processing." arXiv, April 26, 2024.
        https://doi.org/10.48550/arXiv.2404.17704.
    """

    name = "splice_rgb"

    def run(
        self,
        bag: torch.Tensor,
        sample: BagSample | None = None,
        **kwargs: Any,
    ) -> RetrievalRepresentation:
        tiling_id = str(kwargs.get("tiling_id"))
        color_features = np.asarray(kwargs.get("mean_rgb"), dtype=float)
        coords = np.asarray(kwargs.get("coords"))
        color_features, coords = self._prepare_selection_inputs(
            features=color_features,
            coords=coords,
            selection_label="SPLICE mean RGB",
        )

        if color_features.size == 0 or color_features.shape[0] == 0:
            return self._build_representation_from_indices(
                sample=sample,
                features=np.empty((0, 0), dtype=np.float32),
                selected=[],
                group_ids=np.array([], dtype=np.int64),
                coords=np.empty((0, 2), dtype=np.int64),
                groups={},
                bag_id=tiling_id,
            )

        if self.percentile_threshold is None:
            raise ValueError("percentile_threshold must be specified for SPLICE.")

        # Stream through mean RGB rows and suppress nearby redundant patches.
        num_patches = color_features.shape[0]
        selected: list[int] = []
        excluded = np.zeros(num_patches, dtype=bool)
        group_ids = -1 * np.ones(num_patches, dtype=int)
        groups: dict[int, np.ndarray] = {}

        for i in range(num_patches):
            if excluded[i]:
                continue

            group_id = len(selected)
            selected.append(i)
            group_ids[i] = group_id

            ref_feat = color_features[i]
            remaining_idx = np.where(~excluded)[0]

            distances = np.linalg.norm(color_features[remaining_idx] - ref_feat, axis=1)
            if distances.size == 0:
                groups[group_id] = np.array([i], dtype=int)
                continue

            thresh = np.percentile(distances, self.percentile_threshold)

            members = [i]
            for j, d in zip(remaining_idx, distances):
                if j == i:
                    continue
                if d < thresh:
                    excluded[j] = True
                    group_ids[j] = group_id
                    members.append(j)

            groups[group_id] = np.array(members, dtype=int)

        if (group_ids == -1).any() and len(selected) > 0:
            seeds = np.array(selected, dtype=int)
            seed_feats = color_features[seeds]
            unassigned = np.where(group_ids == -1)[0]
            for idx in unassigned:
                d = np.linalg.norm(seed_feats - color_features[idx], axis=1)
                gid = int(np.argmin(d))
                group_ids[idx] = gid
                groups[gid] = np.concatenate([groups[gid], [idx]])

        return self._build_representation_from_indices(
            sample=sample,
            features=color_features,
            selected=selected,
            group_ids=group_ids.astype(int),
            coords=coords,
            groups=groups,
            bag_id=tiling_id,
        )

    def load_sample(
        self,
        *,
        index: int,
        sample: BagSample,
        base_dataset: BagDataset,
    ) -> dict[str, Any]:
        """Load mean-RGB descriptors and coordinates for one SPLICE RGB sample."""
        _ = index
        tiling_id = str(base_dataset.tiling_id)
        slide_paths_by_id = _resolve_sample_slide_paths(
            sample=sample,
            config=self.extra.get("config"),
        )
        slide_processor = _build_slide_processor(config=self.extra.get("config"))
        coord_parts: list[np.ndarray] = []
        mean_rgb_parts: list[np.ndarray] = []

        try:
            for slide_id, artifact_path in zip(sample.slide_ids, sample.artifact_paths):
                with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                    coords = tiles_io.read_coords(
                        slide_artifact,
                        bag_id=tiling_id,
                    )
                    mean_rgb = load_or_create_slide_patch_mean_rgb(
                        slide_artifact=slide_artifact,
                        retrieval_artifact_path=_slide_retrieval_artifact_path(
                            slide_artifact_path=artifact_path,
                            slide_id=str(slide_id),
                        ),
                        slide_path=slide_paths_by_id.get(str(slide_id)),
                        bag_id=tiling_id,
                        slide_processor=slide_processor,
                        slide_id=str(slide_id),
                    )
                coord_parts.append(np.asarray(coords[:, :2], dtype=np.int64))
                mean_rgb_parts.append(np.asarray(mean_rgb, dtype=np.float32))
        finally:
            close_fn = getattr(slide_processor, "close", None)
            if callable(close_fn):
                close_fn()

        return {
            "mean_rgb": (
                np.concatenate(mean_rgb_parts, axis=0)
                if mean_rgb_parts
                else np.empty((0, 3), dtype=np.float32)
            ),
            "coords": (
                np.concatenate(coord_parts, axis=0)
                if coord_parts
                else np.empty((0, 2), dtype=np.int64)
            ),
            "tiling_id": tiling_id,
        }


@register_representation_strategy("splice_features")
class SPLICEFeatures(_BaseSPLICEStrategy):
    """
    SPLICE Features Mosaic Selection
    -------------------------------
    Applies the SPLICE algorithm using deep learning features instead of RGB
    histograms to identify and retain a diverse set of informative patches.

    This variant operates in the same streaming selection mode as the original
    SPLICE, but uses feature embeddings to compute pairwise distances. A patch is
    only selected if it differs enough from previously selected ones.

    Inputs:
    - bag: `torch.Tensor | np.ndarray` with shape `(N, D)`, where each row is a
      deep feature embedding for one patch.
    - sample: `BagSample` with `sample_id: str`, `artifact_paths: list[Path]`,
      and slide membership metadata.
    - combo_cfg: object with `tile_px: int` and `tile_mpp: float`.

    Returns:
    - `RetrievalRepresentation` with:
      - `data`: selected embedding rows with shape `(K, D)`
      - `additional_data["selected_indices"]`: shape `(K,)`
      - `additional_data["group_ids"]`: shape `(N,)`
      - `additional_data["selected_coords"]`: shape `(K, 2)`
      - `additional_data["groups"]`: group-to-member index mapping

    Example:
    ```python
    strategy = SPLICEFeatures(params={"percentile_threshold": 25.0})
    representation = strategy.run(bag=embeddings, sample=sample, combo_cfg=combo_cfg)
    ```

    Reference:
        Alsaafin, Areej, Peyman Nejat, Abubakr Shafique, Jibran Khan, Saghir
        Alfasly, Ghazal Alabtah, and H. R. Tizhoosh. "SPLICE -- Streamlining
        Digital Pathology Image Processing." arXiv, April 26, 2024.
        https://doi.org/10.48550/arXiv.2404.17704.
    """

    name = "splice_features"

    def run(
        self,
        bag: torch.Tensor,
        sample: BagSample | None = None,
        **kwargs: Any,
    ) -> RetrievalRepresentation:
        tiling_id = str(kwargs.get("tiling_id"))
        features = self.as_numpy_feature_matrix(bag)
        coords = np.asarray(kwargs.get("coords"))
        features, coords = self._prepare_selection_inputs(
            features=features,
            coords=coords,
            selection_label="SPLICE",
        )

        if features.size == 0 or features.shape[0] == 0:
            return self._build_representation_from_indices(
                sample=sample,
                features=np.empty((0, 0), dtype=np.float32),
                selected=[],
                group_ids=np.array([], dtype=np.int64),
                coords=np.empty((0, 2), dtype=np.int64),
                groups={},
                bag_id=tiling_id,
            )

        if self.percentile_threshold is None:
            raise ValueError("percentile_threshold must be specified for SPLICE.")

        # Stream through embedding rows and suppress nearby redundant patches.
        num_patches = features.shape[0]
        selected: list[int] = []
        excluded = np.zeros(num_patches, dtype=bool)
        group_ids = -1 * np.ones(num_patches, dtype=int)
        groups: dict[int, np.ndarray] = {}

        for i in range(num_patches):
            if excluded[i]:
                continue

            group_id = len(selected)
            selected.append(i)
            group_ids[i] = group_id

            ref_feat = features[i]
            remaining_idx = np.where(~excluded)[0]

            distances = np.linalg.norm(features[remaining_idx] - ref_feat, axis=1)
            if distances.size == 0:
                groups[group_id] = np.array([i], dtype=int)
                continue

            thresh = np.percentile(distances, self.percentile_threshold)

            members = [i]
            for j, d in zip(remaining_idx, distances):
                if j == i:
                    continue
                if d < thresh:
                    excluded[j] = True
                    group_ids[j] = group_id
                    members.append(j)

            groups[group_id] = np.array(members, dtype=int)

        if (group_ids == -1).any() and len(selected) > 0:
            seeds = np.array(selected, dtype=int)
            seed_feats = features[seeds]
            unassigned = np.where(group_ids == -1)[0]
            for idx in unassigned:
                d = np.linalg.norm(seed_feats - features[idx], axis=1)
                gid = int(np.argmin(d))
                group_ids[idx] = gid
                groups[gid] = np.concatenate([groups[gid], [idx]])

        return self._build_representation_from_indices(
            sample=sample,
            features=features,
            selected=selected,
            group_ids=group_ids.astype(int),
            coords=coords,
            groups=groups,
            bag_id=tiling_id,
        )

    def load_sample(
        self,
        *,
        index: int,
        sample: BagSample,
        base_dataset: BagDataset,
    ) -> dict[str, Any]:
        """Load feature bags and coordinates for one SPLICE sample."""
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
            coord_parts.append(np.asarray(coords[:, :2], dtype=np.int64))

        return {
            "bag": (
                np.concatenate(bag_parts, axis=0)
                if bag_parts
                else np.empty((0, 0), dtype=np.float32)
            ),
            "coords": (
                np.concatenate(coord_parts, axis=0)
                if coord_parts
                else np.empty((0, 2), dtype=np.int64)
            ),
            "tiling_id": tiling_id,
        }
