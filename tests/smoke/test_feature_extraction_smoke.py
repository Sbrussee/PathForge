from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathbench.config.config import Config
from pathbench.core.experiments.base import Experiment
from pathbench.core.experiments.combo_ids import build_feature_name, build_tiling_id
from pathbench.core.io.slide_artifacts.base import FileHandleH5
from pathbench.core.io.slide_artifacts.layout import DEFAULT_LAYOUT
from pathbench.policy.feature_extraction import FeatureExtractionPolicy
from pathbench.utils.registries import FEATURE_EXTRACTORS


class _FakeSlideProcessor:
    def load_wsi(self, wsi) -> None:
        _ = wsi

    def close_wsi(self, wsi) -> None:
        _ = wsi

    def get_base_mpp(self, wsi) -> float:
        _ = wsi
        return 0.5

    def segment_tissue(self, wsi, config=None):
        _ = wsi, config
        return []

    def extract_patches(self, wsi, tissues, config=None):
        _ = wsi, tissues, config
        coords = np.array(
            [
                [0, 0, 256, 256, 0],
                [256, 0, 256, 256, 0],
            ],
            dtype=np.int32,
        )
        tiling_spec = {
            "tile_px": 256,
            "tile_mpp": 0.5,
            "stride_px": 256,
            "coord_space": "level0",
            "backend": "smoke",
        }
        return coords, tiling_spec

    def extract_features(self, wsi, coords, tiling_spec, config=None):
        _ = wsi, tiling_spec, config
        return np.ones((coords.shape[0], 8), dtype=np.float32)

    def get_thumbnail(self, wsi, level=-1):
        _ = wsi, level
        image = np.full((32, 64, 3), 120, dtype=np.uint8)
        return image, 8.0, 8.0


def _ensure_feature_extractor_registered(name: str) -> None:
    if FEATURE_EXTRACTORS.is_available(name):
        return

    @FEATURE_EXTRACTORS.register(name)
    def _dummy():
        return name


@pytest.mark.smoke
def test_smoke_feature_extraction_writes_expected_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor = "resnet18"
    _ensure_feature_extractor_registered(extractor)

    slides_dir = tmp_path / "slides"
    artifacts_dir = tmp_path / "artifacts"
    project_root = tmp_path / "project"
    slides_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    (slides_dir / "S1.svs").write_bytes(b"dummy")
    ann_path = tmp_path / "annotations.csv"
    ann_path.write_text(
        "dataset,slide,patient,category\n"
        "smoke,S1,P0,cat0\n",
        encoding="utf-8",
    )

    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "smoke_project",
                "annotation_file": str(ann_path),
                "project_root": str(project_root),
                "mode": "feature_extraction",
                "report": True,
                "thumbnail": True,
            },
            "slide_processing": {"backend": "lazyslide"},
            "datasets": [
                {
                    "name": "smoke",
                    "slides_dir": str(slides_dir),
                    "artifacts_dir": str(artifacts_dir),
                    "used_for": "all",
                }
            ],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": [extractor],
                "mil": [],
            },
        }
    )

    policy = FeatureExtractionPolicy(Experiment(cfg))
    monkeypatch.setattr(policy, "_build_processor", lambda: _FakeSlideProcessor())
    monkeypatch.setattr(policy, "_build_seg_config", lambda: {"method": "otsu", "params": {}})

    dataset = policy.datasets[0]
    combo = policy.combos[0]
    wsi = dataset.samples[0]
    policy.process_slide(dataset=dataset, wsi=wsi, combo_cfg=combo)

    tiling_id = build_tiling_id(combo)
    feature_name = build_feature_name(combo)
    coords_path = DEFAULT_LAYOUT.coords_dataset(tiling_id)
    tiling_path = DEFAULT_LAYOUT.tiling_spec_dataset(tiling_id)
    overview_path = DEFAULT_LAYOUT.tiles_overview_dataset(tiling_id)
    feats_path = DEFAULT_LAYOUT.features_dataset(tiling_id, feature_name)

    with FileHandleH5(wsi.artifact_path, mode="r") as fh:
        assert DEFAULT_LAYOUT.tissue_dataset in fh.h5
        assert coords_path in fh.h5
        assert tiling_path in fh.h5
        assert overview_path in fh.h5
        assert feats_path in fh.h5
