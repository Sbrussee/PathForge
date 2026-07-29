"""SISH-style RGB-histogram retrieval representation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans

from pathforge.core.datasets.bag_dataset import BagDataset, BagSample
from pathforge.core.datasets.wsi_dataset import WSI
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
    _build_slide_processor,
    _resolve_sample_slide_paths,
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
        """Load row-aligned features, coordinates, and cached-or-created histograms."""
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
            "histogram_rgb": resolve_sample_patch_histogram_rgb(
                sample=sample, bag_id=bag_id, config=self.extra.get("config")
            ),
            "slide_lengths": lengths,
            "tiling_id": bag_id,
        }

    def run(
        self, bag: np.ndarray, sample: BagSample | None = None, **kwargs: Any
    ) -> RetrievalRepresentation:
        """Return selected existing patch embeddings for one retrieval sample."""
        if sample is None:
            raise ValueError("sample is required for sish_rgb.")
        features = self.as_numpy_feature_matrix(bag)
        coords = np.asarray(kwargs["coords"], dtype=np.int32)
        histograms = np.asarray(kwargs["histogram_rgb"], dtype=np.float32)
        lengths = [int(item) for item in kwargs["slide_lengths"]]
        if len(features) != len(coords) or len(features) != len(histograms):
            raise ValueError(
                "SISH RGB feature, coordinate, and histogram rows must match."
            )
        classifier = load_sish_trash_classifier(config=self.extra.get("config"))
        slide_paths = _resolve_sample_slide_paths(
            sample=sample, config=self.extra.get("config")
        )
        processor = _build_slide_processor(config=self.extra.get("config"))
        selected: list[int] = []
        labels = np.full(len(features), -1, dtype=np.int32)
        offset = 0
        try:
            for slide_id, artifact_path, length in zip(
                sample.slide_ids, sample.artifact_paths, lengths, strict=True
            ):
                indices = np.arange(offset, offset + length, dtype=np.int32)
                path = slide_paths.get(str(slide_id))
                if path is None or not path.is_file():
                    raise FileNotFoundError(
                        f"SISH RGB trash filtering requires source slide '{slide_id}'."
                    )
                keep = self._retained_rows(
                    slide_id=str(slide_id),
                    slide_path=path,
                    artifact_path=Path(artifact_path),
                    coords=coords[indices],
                    processor=processor,
                    classifier=classifier,
                )
                kept = indices[keep]
                if len(kept):
                    picked, local_labels = self._select_slide(
                        histograms[kept], coords[kept, :2]
                    )
                    labels[kept] = local_labels
                    selected.extend(kept[picked].tolist())
                offset += length
        finally:
            close = getattr(processor, "close", None)
            if callable(close):
                close()
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
        self,
        *,
        slide_id: str,
        slide_path: Path,
        artifact_path: Path,
        coords: np.ndarray,
        processor: Any,
        classifier: Any,
    ) -> np.ndarray:
        wsi = WSI(
            slide=slide_id,
            patient="",
            category="",
            path=slide_path,
            artifact_path=artifact_path,
        )
        processor.load_wsi(wsi)
        lbp_rows: list[np.ndarray] = []
        candidates: list[int] = []
        try:
            for index, row in enumerate(coords):
                x, y, width, height, level = (int(value) for value in row)
                patch = np.asarray(
                    processor.read_patch_region(
                        wsi, x=x, y=y, width=width, height=height, level=level
                    ),
                    dtype=np.uint8,
                )
                grey = cv.resize(cv.cvtColor(patch, cv.COLOR_RGB2GRAY), (256, 256))
                if float(np.mean(grey > 235)) > 0.9:
                    continue
                lbp = local_binary_pattern(grey, 8, 1, "ror")
                lbp_rows.append(
                    np.histogram(lbp, density=True, bins=128, range=(0, 128))[0]
                )
                candidates.append(index)
        finally:
            processor.close_wsi(wsi)
        keep = np.zeros(len(coords), dtype=bool)
        if candidates:
            predictions = np.asarray(
                classifier.predict(np.asarray(lbp_rows, dtype=np.float32))
            )
            keep[np.asarray(candidates, dtype=np.int32)[predictions == 0]] = True
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
            count = max(1, int(float(self.sample_rate) * len(members)))
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
