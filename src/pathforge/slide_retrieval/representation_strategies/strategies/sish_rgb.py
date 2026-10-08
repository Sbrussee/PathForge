"""SISH-style RGB-histogram retrieval representation."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
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
    SISH_QUALITY_DIM,
    resolve_sample_patch_rgb_descriptors,
)
from pathforge.slide_retrieval.representation_strategies.registry import (
    register_representation_strategy,
)
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.search_strategies.strategies.sish.sish_assets import (
    load_sish_trash_classifier,
)


@register_representation_strategy("sish_rgb")
class SISHRGB(BaseRetrievalRepresentationStrategy):
    """Select existing feature rows using SISH RGB histograms and trash filtering."""

    supported_feature_levels = frozenset({"patch"})
    output_representation_kind = "patch_vector"
    n_clusters = HyperParam(int, default=9, min=1, help="SISH colour KMeans clusters.")
    sample_rate = HyperParam(
        float, default=0.05, min=0.0, max=1.0, help="SISH spatial sampling fraction."
    )
    random_state = HyperParam(int, default=0, help="KMeans seed.")

    def hyperparam_values(self) -> dict[str, Any]:
        """Include the trash-classifier content identity in representation caching."""
        values = super().hyperparam_values()
        # Prevent reuse of mosaics generated with the former minimum-one fallback.
        values["small_group_policy"] = "retain_all"
        from pathforge.slide_retrieval.search_strategies.strategies.sish.sish_assets import (
            resolve_sish_asset_path,
        )

        asset = resolve_sish_asset_path(
            config=self.extra.get("config"), asset="trash_classifier"
        )
        values["trash_classifier"] = (
            hashlib.sha256(asset.read_bytes()).hexdigest()
            if asset.is_file()
            else str(asset)
        )
        return values

    def load_sample(
        self, *, index: int, sample: BagSample, base_dataset: BagDataset
    ) -> dict[str, Any]:
        """Load feature, coordinate, RGB, and quality rows; reuse valid caches.

        RGB rows have shape ``(N, 768)`` and quality rows ``(N, 129)``.
        Example: ``payload = strategy.load_sample(index=0, sample=sample,
        base_dataset=dataset)``. A complete cache requires no source WSI.
        """
        del index
        bag_id = str(base_dataset.tiling_id)
        features: list[np.ndarray] = []
        coords: list[np.ndarray] = []
        lengths: list[int] = []
        for path in sample.artifact_paths:
            with FileHandleH5(path, mode="r") as artifact:
                feature = features_io.read_features(
                    artifact, bag_id=bag_id, extractor_name=base_dataset.extractor_name
                )
                coord = tiles_io.read_coords(artifact, bag_id=bag_id)
            if len(feature) != len(coord):
                raise ValueError("SISH RGB features and coordinate rows must match.")
            features.append(np.asarray(feature, dtype=np.float32))
            coords.append(np.asarray(coord, dtype=np.int32))
            lengths.append(len(coord))
        return {
            "bag": np.concatenate(features, axis=0),
            "coords": np.concatenate(coords, axis=0),
            **resolve_sample_patch_rgb_descriptors(
                sample=sample, bag_id=bag_id, config=self.extra.get("config"),
                include_quality=True,
            ),
            "slide_lengths": lengths,
            "tiling_id": bag_id,
        }

    def run(
        self, bag: np.ndarray, sample: BagSample | None = None, **kwargs: Any
    ) -> RetrievalRepresentation:
        """Filter cached quality descriptors and select existing patch embeddings.

        Args:
            bag: Foundation-model feature matrix with shape ``(N, D)``.
            sample: Sample identifying the output retrieval item.
            **kwargs: Row-aligned ``coords: (N, 5)``, ``histogram_rgb: (N,768)``,
                ``sish_quality: (N,129)``, per-slide ``slide_lengths``, and
                canonical ``tiling_id``. No source WSI is used by this method.

        Returns:
            Selected feature rows ``(K, D)``, indices, and coordinates.

        Example:
            >>> payload = strategy.load_sample(
            ...     index=0, sample=sample, base_dataset=dataset
            ... )
            >>> representation = strategy.run(sample=sample, **payload)
        """
        if sample is None:
            raise ValueError("sample is required for sish_rgb.")
        features = self.as_numpy_feature_matrix(bag)
        coords = np.asarray(kwargs["coords"], dtype=np.int32)
        histograms = np.asarray(kwargs["histogram_rgb"], dtype=np.float32)
        quality = np.asarray(kwargs["sish_quality"], dtype=np.float32)
        lengths = [int(item) for item in kwargs["slide_lengths"]]
        if quality.shape != (len(features), SISH_QUALITY_DIM):
            raise ValueError("SISH quality descriptors must have shape (N, 129).")
        if any(length < 0 for length in lengths) or sum(lengths) != len(features):
            raise ValueError("SISH slide lengths must cover every feature row.")
        if len(features) != len(coords) or len(features) != len(histograms):
            raise ValueError(
                "SISH RGB feature, coordinate, and histogram rows must match."
            )
        classifier = load_sish_trash_classifier(config=self.extra.get("config"))
        selected: list[int] = []
        labels = np.full(len(features), -1, dtype=np.int32)
        offset = 0
        for length in lengths:
            indices = np.arange(offset, offset + length, dtype=np.int32)
            keep = self._retained_rows(quality=quality[indices], classifier=classifier)
            kept = indices[keep]
            if len(kept):
                picked, local_labels = self._select_slide(
                    histograms[kept], coords[kept, :2]
                )
                labels[kept] = local_labels
                selected.extend(kept[picked].tolist())
            offset += length
        selected_array = np.asarray(selected, dtype=np.int32)
        return RetrievalRepresentation(
            sample_id=sample.sample_id,
            data=features[selected_array],
            additional_data={
                "selected_indices": selected_array,
                "selected_coords": coords[selected_array, :2],
                "group_ids": labels,
                "bag_id": str(kwargs["tiling_id"]),
            },
        )

    def _retained_rows(
        self, *, quality: np.ndarray, classifier: Any
    ) -> np.ndarray:
        """Filter cached ``(N,129)`` quality rows into a tissue mask ``(N,)``.

        Columns 0–127 contain LBP histogram bins; column 128 is the white
        fraction. Only nonwhite patches are classified, with label 0 retained.
        Example: ``keep = strategy._retained_rows(quality=rows, classifier=clf)``.
        """
        keep = np.zeros(len(quality), dtype=bool)
        candidates = np.flatnonzero(quality[:, 128] <= 0.9)
        if candidates.size:
            predictions = np.asarray(classifier.predict(quality[candidates, :128]))
            keep[candidates[predictions == 0]] = True
        return keep

    def _select_slide(
        self, histograms: np.ndarray, coords: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        model = KMeans(
            n_clusters=min(int(self.n_clusters), len(histograms)),
            random_state=int(self.random_state),
        ).fit(histograms)
        labels = np.asarray(model.labels_, dtype=np.int32)
        selected: list[int] = []
        for group in range(model.n_clusters):
            members = np.flatnonzero(labels == group)
            count = int(float(self.sample_rate) * len(members))
            if count == 0:
                # SISH retains the entire group when its sampling count is zero.
                selected.extend(members.tolist())
                continue
            spatial = KMeans(n_clusters=count, random_state=int(self.random_state)).fit(
                coords[members]
            )
            used: set[int] = set()
            for center in spatial.cluster_centers_:
                for member in members[
                    np.argsort(np.sum((coords[members] - center) ** 2, axis=1))
                ]:
                    if int(member) not in used:
                        used.add(int(member))
                        selected.append(int(member))
                        break
        return np.asarray(selected, dtype=np.int32), labels
