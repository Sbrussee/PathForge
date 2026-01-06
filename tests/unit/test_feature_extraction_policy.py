"""Unit tests for the feature extraction policy with a mock slide processor."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from pathbench.config.config import Config
from pathbench.core.experiments.base import Experiment
from pathbench.core.slide_processing.base import SlideProcessorBase
from pathbench.policy.feature_extraction import FeatureExtractionPolicy
from pathbench.policy.utils import load_features_pt
from pathbench.utils.registries import SLIDE_PROCESSORS


class MockSlideProcessor(SlideProcessorBase):
    """Minimal slide processor that returns deterministic tiles and features."""

    def load_wsi(self, wsi) -> None:  # type: ignore[override]
        wsi._obj = {"loaded": True}

    def close_wsi(self, wsi) -> None:  # type: ignore[override]
        wsi._obj = None

    def segment_tissue(self, wsi, config=None):  # type: ignore[override]
        return [np.array([[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]], dtype=np.float32)]

    def extract_patches(self, wsi, tissues, config=None):  # type: ignore[override]
        tiles_df = pd.DataFrame(
            {
                "tile_id": ["0", "1", "2"],
                "x": [0, 256, 512],
                "y": [0, 0, 0],
            }
        )
        tile_spec = json.dumps({"tile_px": config["tile_px"], "tile_mpp": config["tile_mpp"]})
        return tiles_df, tile_spec

    def validate_tile_spec(self, tile_spec, config=None):  # type: ignore[override]
        if tile_spec is None:
            return False
        spec = json.loads(tile_spec)
        return spec.get("tile_px") == config["tile_px"] and spec.get("tile_mpp") == config["tile_mpp"]

    def extract_features(self, wsi, tiles, tile_spec, config=None):  # type: ignore[override]
        features = np.random.RandomState(0).randn(len(tiles), 4).astype(np.float32)
        obs = pd.DataFrame({"tile_id": tiles["tile_id"].astype(str)})
        return ad.AnnData(X=features, obs=obs)

    def extract_cells(self, wsi, config=None):  # type: ignore[override]
        return None

    def inspect_slide(self, wsi) -> None:  # type: ignore[override]
        return None


def _register_mock_backend() -> object | None:
    if SLIDE_PROCESSORS.is_available("lazyslide"):
        original = SLIDE_PROCESSORS._f.get("lazyslide")  # type: ignore[attr-defined]
        SLIDE_PROCESSORS._f["lazyslide"] = MockSlideProcessor  # type: ignore[attr-defined]
        return original
    SLIDE_PROCESSORS.register("lazyslide")(MockSlideProcessor)
    return None


def test_feature_extraction_policy_generates_features(tmp_path):
    """
    Run feature extraction with a mock backend and validate saved artifacts.

    Expected feature shape: (num_tiles=3, feature_dim=4).
    """
    original_backend = _register_mock_backend()

    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    for slide_id in ("slide_a", "slide_b"):
        (slides_dir / f"{slide_id}.svs").write_text("mock")

    annotations = pd.DataFrame(
        {
            "slide": ["slide_a", "slide_b"],
            "patient": ["patient_1", "patient_2"],
            "category": ["class_0", "class_1"],
            "dataset": ["train", "train"],
        }
    )
    ann_path = tmp_path / "annotations.csv"
    annotations.to_csv(ann_path, index=False)

    config = Config.from_dict(
        {
            "experiment": {
                "project_name": "features",
                "annotation_file": str(ann_path),
                "mode": "feature_extraction",
                "project_root": str(tmp_path),
            },
            "slide_processing": {
                "backend": "lazyslide",
                "segmentation_method": "otsu",
            },
            "datasets": [
                {
                    "name": "train",
                    "slide_path": str(slides_dir),
                    "used_for": "training",
                }
            ],
            "search_space": {
                "feature_extraction": ["resnet18"],
                "tile_px": [256],
                "tile_mpp": [0.5],
            },
        }
    )

    experiment = Experiment(config)
    policy = FeatureExtractionPolicy(experiment)
    result = policy.execute()
    assert result["status"] == "feature_extraction_done"

    features_root = Path(experiment.project_root) / "features"
    combo_dir = features_root / "resnet18_256px_0.5mpp"
    for slide_id in ("slide_a", "slide_b"):
        assert (combo_dir / f"{slide_id}.pt").exists()
        assert (combo_dir / f"{slide_id}.index.npz").exists()

    loaded = load_features_pt(combo_dir / "slide_a")
    assert loaded.X.shape == (3, 4)

    if original_backend is not None:
        SLIDE_PROCESSORS._f["lazyslide"] = original_backend  # type: ignore[attr-defined]
    else:
        del SLIDE_PROCESSORS._f["lazyslide"]  # type: ignore[attr-defined]