# tests/smoke/test_feature_extraction_smoke.py
from __future__ import annotations

from pathlib import Path

import pytest

from pathbench.config.config import Config
from pathbench.core.experiments.base import Experiment
from pathbench.policy.feature_extraction import FeatureExtractionPolicy
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5.layout import DEFAULT_LAYOUT
from pathbench.utils.registries import FEATURE_EXTRACTORS
from tests.smoke._smoke_dataset import (
    DownloadedSmokeAssets,
    attach_smoke_outputs,
    capture_smoke_metrics,
    link_or_copy,
)


def _ensure_feature_extractor_registered(name: str) -> None:
    # Config validation requires the name to exist in FEATURE_EXTRACTORS
    if FEATURE_EXTRACTORS.is_available(name):
        return

    @FEATURE_EXTRACTORS.register(name)
    def _dummy():
        return name


@pytest.mark.smoke
def test_smoke_feature_extraction_lazyslide(
    smoke_assets: DownloadedSmokeAssets,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ---- choose a tiny-ish end-to-end workload ----
    extractor = "resnet18"
    tile_px = 256
    tile_mpp = 0.5  # coarse -> fewer tiles

    _ensure_feature_extractor_registered(extractor)

    slide_path = smoke_assets.slides["sample.svs"]

    # ---- build minimal dataset layout expected by your code ----
    slides_dir = tmp_path / "slides"
    artifacts_dir = tmp_path / "artifacts"
    project_root = tmp_path / "project"
    slides_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    local_slide = slides_dir / slide_path.name
    link_or_copy(slide_path, local_slide)

    # annotations.csv must match WSIDataset expectations: dataset, slide, patient, category
    ann_path = tmp_path / "annotations.csv"
    ann_path.write_text(
        "dataset,slide,patient,category\nsmoke,sample,P0,cat0\n",
        encoding="utf-8",
    )

    cfg_data = {
        "experiment": {
            "project_name": "smoke_project",
            "annotation_file": str(ann_path),
            "project_root": str(project_root),
            "mode": "feature_extraction",
            "report": True,  # <-- enable tiles_overview generation
        },
        "slide_processing": {
            "backend": "lazyslide",
        },
        "datasets": [
            {
                "name": "smoke",
                "slides_dir": str(slides_dir),
                "artifacts_dir": str(artifacts_dir),
                "used_for": "all",
            }
        ],
        "benchmark_parameters": {
            "tile_px": [tile_px],
            "tile_mpp": [tile_mpp],
            "feature_extraction": [extractor],
            "mil": [],  # not used in feature_extraction mode
        },
    }

    cfg = Config.model_validate(cfg_data)
    exp = Experiment(cfg)
    policy = FeatureExtractionPolicy(exp)

    # ---- make it deterministic + lightweight ----
    monkeypatch.setattr(
        policy,
        "_build_seg_config",
        lambda: {"method": "otsu", "params": {}},
    )

    assert len(policy.datasets) == 1
    ds = policy.datasets[0]
    assert ds.name == "smoke"
    assert len(ds.samples) == 1

    combo = policy.combos[0]
    wsi = ds.samples[0]

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="legacy_lazyslide_feature_extraction",
        metadata={"extractor": extractor, "tile_px": tile_px, "tile_mpp": tile_mpp},
    ) as metadata:
        policy.process_slide(dataset=ds, wsi=wsi, combo_cfg=combo)
        attach_smoke_outputs(
            metadata,
            step_name="legacy_lazyslide_feature_extraction",
            intermediate={"slides_dir": slides_dir},
            final={"artifact_path": wsi.artifact_path},
        )

    # ---- assertions: H5 exists + layout is consistent ----
    assert wsi.artifact_path.exists()

    bag_id = f"{tile_px}px_{tile_mpp:g}mpp"
    coords_path = DEFAULT_LAYOUT.coords_dataset(bag_id)
    tiling_path = DEFAULT_LAYOUT.tiling_spec_dataset(bag_id)
    overview_path = DEFAULT_LAYOUT.tiles_overview_dataset(bag_id)  # <-- new
    feats_path = DEFAULT_LAYOUT.features_dataset(bag_id, extractor)

    with FileHandleH5(wsi.artifact_path, mode="r") as fh:
        assert DEFAULT_LAYOUT.tissue_dataset in fh.h5
        assert coords_path in fh.h5
        assert tiling_path in fh.h5
        assert overview_path in fh.h5  # <-- new
        assert feats_path in fh.h5

        coords = fh.h5[coords_path][()]
        overview = fh.h5[overview_path][()]  # <-- new
        feats = fh.h5[feats_path][()]

        assert coords.ndim == 2 and coords.shape[1] == 5

        # tiles_overview: compressed image bytes stored as 1D uint8 array
        assert overview.ndim == 1
        assert overview.dtype.name == "uint8"
        assert overview.size > 0

        # JPEG SOI marker (optional but useful sanity check)
        overview_bytes = (
            bytes(overview[:4].tolist())
            if overview.size >= 4
            else bytes(overview.tolist())
        )
        assert overview_bytes[:2] == b"\xff\xd8"

        assert feats.ndim == 2
        assert feats.shape[0] == coords.shape[0]
        assert feats.shape[0] > 0
