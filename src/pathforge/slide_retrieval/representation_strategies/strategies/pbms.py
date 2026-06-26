from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
import torch

from pathforge.core.datasets.bag_dataset import BagDataset, BagSample
from pathforge.core.experiments.combo_ids import build_feature_name, build_tiling_id
from pathforge.core.io.slide_artifacts import features as features_io
from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.slide_retrieval.hyperparams import HyperParam
from pathforge.slide_retrieval.representation_strategies.base import (
    BaseRetrievalRepresentationStrategy,
)
from pathforge.slide_retrieval.representation_strategies.panther import (
    PantherAssignment,
    PantherPrototypeAssigner,
)
from pathforge.slide_retrieval.representation_strategies.prototype_bundles import (
    SalvPrototypeBundle,
    SalvPrototypeBundleResolver,
)
from pathforge.slide_retrieval.representation_strategies.registry import (
    register_representation_strategy,
)
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)

logger = logging.getLogger(__name__)


@register_representation_strategy("pbms-features")
class PBMSFeatures(BaseRetrievalRepresentationStrategy):
    """
    Prototype-Based Mosaic Selection using Salv prototypes and PANTHER assignment.

    Patches assigned to prototypes labeled `exclude` in `proto_labels.json` are
    excluded from representative selection. Included prototype groups use the
    spatial KMeans representative selection from the PBMS refactor.
    """

    name = "pbms-features"
    supported_feature_levels = frozenset({"patch"})
    output_representation_kind = "patch_vector"
    preferred_loader_workers = 1
    preferred_materialization_workers = 1

    perc_selected = HyperParam(
        float,
        default=1.0,
        min=0.0,
        max=100.0,
        help="Percent of patches selected within each included prototype group.",
    )
    em_iter = HyperParam(
        int,
        default=10,
        min=1,
        help="Number of PANTHER EM iterations.",
    )
    tau = HyperParam(
        float,
        default=10.0,
        min=0.0,
        help="PANTHER prior strength.",
    )
    ot_eps = HyperParam(
        float,
        default=0.1,
        min=1e-12,
        help="PANTHER diagonal covariance prior scale.",
    )
    prototype_run_name = HyperParam(
        str,
        default=None,
        help="Optional prototype run directory name/relative path for disambiguation.",
    )
    save_resp = HyperParam(
        bool,
        default=True,
        help="Save full PANTHER responsibility matrix as float16 additional data.",
    )

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(params=params, **kwargs)
        self.config = kwargs.get("config")
        self.random_state = self._resolve_random_state()
        self._bundle: SalvPrototypeBundle | None = None
        self._tiling_id: str | None = None
        self._feature_name: str | None = None

    def prepare_for_combo(
        self,
        *,
        combo_cfg: Any,
        feature_name: str | None = None,
        tiling_id: str | None = None,
    ) -> None:
        """
        Resolve the prototype bundle before cache IDs are built.

        The resolved bundle hash is included in `hyperparam_values()` so cached
        retrieval representations are invalidated when prototypes or labels
        change.
        """
        resolved_feature_name = feature_name or build_feature_name(combo_cfg)
        resolved_tiling_id = tiling_id or build_tiling_id(combo_cfg)
        prototypes_root = self._resolve_prototypes_root()

        resolver = SalvPrototypeBundleResolver(prototypes_root)
        bundle = resolver.resolve(
            tile_px=int(combo_cfg.tile_px),
            tile_mpp=float(combo_cfg.tile_mpp),
            feature_extraction=str(combo_cfg.feature_extraction),
            color_norm=_normalize_optional_name(combo_cfg.get("color_norm")),
            prototype_run_name=self.prototype_run_name,
        )

        self._bundle = bundle
        self._tiling_id = str(resolved_tiling_id)
        self._feature_name = str(resolved_feature_name)
        logger.info(
            "PBMS resolved prototype bundle %s from %s.",
            bundle.bundle_id,
            bundle.run_dir,
        )

    def hyperparam_values(self) -> dict[str, Any]:
        values = super().hyperparam_values()
        if self._bundle is not None:
            values["prototype_bundle_hash"] = self._bundle.content_hash
            values["prototype_bundle_id"] = self._bundle.bundle_id
        return values

    def run(
        self,
        bag: torch.Tensor | np.ndarray,
        sample: BagSample | None = None,
        **kwargs: Any,
    ) -> RetrievalRepresentation:
        if sample is None:
            raise ValueError("sample is required for pbms-features.")

        features = self.as_numpy_feature_matrix(bag)
        coords = np.asarray(kwargs.get("coords"), dtype=np.int32)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"coords must have shape (N, >=2). Got {coords.shape}.")
        coords = np.asarray(coords[:, :2], dtype=np.int32)
        if int(coords.shape[0]) != int(features.shape[0]):
            raise ValueError(
                "PBMS requires one coordinate row per feature row. "
                f"Got coords={coords.shape[0]} and features={features.shape[0]}."
            )

        bag_id = str(kwargs.get("tiling_id") or self._tiling_id or "")
        if not bag_id:
            bag_id = build_tiling_id(kwargs.get("combo_cfg"))

        if features.shape[0] == 0:
            return self._build_representation(
                sample=sample,
                features=np.empty((0, 0), dtype=np.float32),
                coords=np.empty((0, 2), dtype=np.int32),
                selected_indices=np.array([], dtype=np.int32),
                group_ids=np.array([], dtype=np.int32),
                assignment=None,
                bundle=self._require_bundle(kwargs.get("combo_cfg")),
                bag_id=bag_id,
            )

        bundle = self._require_bundle(kwargs.get("combo_cfg"))
        if int(features.shape[1]) != bundle.feature_dim:
            raise ValueError(
                "PBMS feature dimension does not match prototype bundle. "
                f"Got features dim {features.shape[1]} and prototype dim {bundle.feature_dim} "
                f"for bundle {bundle.run_dir}."
            )

        assignment = PantherPrototypeAssigner(
            prototypes=bundle.prototype_matrix,
            em_iter=int(self.em_iter),
            tau=float(self.tau),
            ot_eps=float(self.ot_eps),
        ).assign(features)

        group_ids = _build_group_ids(
            top1=assignment.top1,
            labels=bundle.labels,
        )
        selected_indices = self._select_representatives(
            group_ids=group_ids,
            coords=coords,
        )

        return self._build_representation(
            sample=sample,
            features=features,
            coords=coords,
            selected_indices=selected_indices,
            group_ids=group_ids,
            assignment=assignment,
            bundle=bundle,
            bag_id=bag_id,
        )

    def load_sample(
        self,
        *,
        index: int,
        sample: BagSample,
        base_dataset: BagDataset,
    ) -> dict[str, Any]:
        """Load feature bags and aligned coordinates for one PBMS item."""
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

    def _select_representatives(
        self,
        *,
        group_ids: np.ndarray,
        coords: np.ndarray,
    ) -> np.ndarray:
        selected: list[int] = []
        for group_id in sorted(int(value) for value in np.unique(group_ids) if value >= 1):
            member_idx = np.where(group_ids == group_id)[0]
            if member_idx.size == 0:
                continue

            n_select = max(1, int(len(member_idx) * (float(self.perc_selected) / 100.0)))
            n_select = min(n_select, int(member_idx.size))

            if n_select == 1:
                selected.append(int(member_idx[0]))
                continue

            loc_features = np.asarray(coords[member_idx], dtype=float)
            kmeans_loc = KMeans(
                n_clusters=n_select,
                random_state=self.random_state,
                n_init="auto",
            )
            dists = kmeans_loc.fit_transform(loc_features)

            used_local: set[int] = set()
            for center_index in range(n_select):
                sorted_local = np.argsort(dists[:, center_index])
                for local_index in sorted_local:
                    local_index_int = int(local_index)
                    if local_index_int not in used_local:
                        used_local.add(local_index_int)
                        selected.append(int(member_idx[local_index_int]))
                        break

        return np.asarray(selected, dtype=np.int32)

    def _build_representation(
        self,
        *,
        sample: BagSample,
        features: np.ndarray,
        coords: np.ndarray,
        selected_indices: np.ndarray,
        group_ids: np.ndarray,
        assignment: PantherAssignment | None,
        bundle: SalvPrototypeBundle,
        bag_id: str,
    ) -> RetrievalRepresentation:
        if selected_indices.size:
            selected_features = np.asarray(
                features[selected_indices],
                dtype=np.float32,
            )
            selected_coords = np.asarray(coords[selected_indices], dtype=np.int32)
        else:
            feature_dim = 0 if features.ndim != 2 else int(features.shape[1])
            selected_features = np.empty((0, feature_dim), dtype=np.float32)
            selected_coords = np.empty((0, 2), dtype=np.int32)

        additional_data: dict[str, Any] = {
            "selected_indices": selected_indices.astype(np.int32, copy=False),
            "selected_coords": selected_coords,
            "group_ids": group_ids.astype(np.int32, copy=False),
            "bag_id": np.asarray([str(bag_id)], dtype=str),
            "prototype_labels": np.asarray(bundle.labels, dtype=str),
            "prototype_bundle_id": np.asarray([bundle.bundle_id], dtype=str),
            "prototype_bundle_hash": np.asarray([bundle.content_hash], dtype=str),
            "prototype_run_dir": np.asarray([str(bundle.run_dir)], dtype=str),
        }

        if assignment is not None:
            additional_data.update(
                {
                    "prototype_ids": assignment.top1.astype(np.int32, copy=False),
                    "prototype_confidence": assignment.top1_prob.astype(
                        np.float32,
                        copy=False,
                    ),
                    "panther_slide_embed": assignment.slide_embed.astype(
                        np.float32,
                        copy=False,
                    ),
                    "panther_proto_mean": assignment.proto_mean.astype(
                        np.float32,
                        copy=False,
                    ),
                    "panther_proto_cov": assignment.proto_cov.astype(
                        np.float32,
                        copy=False,
                    ),
                    "panther_proto_prob": assignment.proto_prob.astype(
                        np.float32,
                        copy=False,
                    ),
                }
            )
            if bool(self.save_resp):
                additional_data["panther_resp"] = assignment.resp.astype(
                    np.float16,
                    copy=False,
                )

        return RetrievalRepresentation(
            sample_id=str(sample.sample_id),
            data=selected_features,
            additional_data=additional_data,
        )

    def _require_bundle(self, combo_cfg: Any | None) -> SalvPrototypeBundle:
        if self._bundle is None:
            if combo_cfg is None:
                raise ValueError(
                    "PBMS prototype bundle has not been resolved and combo_cfg is missing."
                )
            self.prepare_for_combo(combo_cfg=combo_cfg)

        assert self._bundle is not None
        return self._bundle

    def _resolve_prototypes_root(self) -> Path:
        config = self.config
        slide_retrieval_cfg = getattr(config, "slide_retrieval", None)
        if slide_retrieval_cfg is None and isinstance(config, dict):
            slide_retrieval_cfg = config.get("slide_retrieval")

        prototypes_root = None
        if isinstance(slide_retrieval_cfg, dict):
            prototypes_root = slide_retrieval_cfg.get("prototypes_root")
        elif slide_retrieval_cfg is not None:
            prototypes_root = getattr(slide_retrieval_cfg, "prototypes_root", None)

        if prototypes_root is None:
            raise ValueError(
                "pbms-features requires slide_retrieval.prototypes_root to point "
                "at Salv prototype discovery runs."
            )

        return Path(str(prototypes_root)).expanduser().resolve()

    def _resolve_random_state(self) -> int | None:
        config = self.config
        experiment_config = getattr(config, "experiment", None)
        if experiment_config is None and isinstance(config, dict):
            experiment_config = config.get("experiment")

        if isinstance(experiment_config, dict):
            value = experiment_config.get("random_state")
        else:
            value = getattr(experiment_config, "random_state", None)

        return None if value is None else int(value)


def _build_group_ids(
    *,
    top1: np.ndarray,
    labels: tuple[str, ...],
) -> np.ndarray:
    prototype_ids = np.asarray(top1, dtype=np.int32)
    labels_array = np.asarray(labels, dtype=object)
    if np.any(prototype_ids < 0) or np.any(prototype_ids >= labels_array.shape[0]):
        raise ValueError("PANTHER prototype assignments contain out-of-range IDs.")

    excluded_mask = labels_array == "exclude"
    return np.where(~excluded_mask[prototype_ids], prototype_ids + 1, 0).astype(
        np.int32,
        copy=False,
    )


def _normalize_optional_name(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
