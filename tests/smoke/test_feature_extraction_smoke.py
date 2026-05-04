# tests/smoke/test_feature_extraction_smoke.py
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pytest

from pathbench.config.config import Config
from pathbench.core.experiments.base import Experiment
from pathbench.policy.feature_extraction import FeatureExtractionPolicy
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5.layout import DEFAULT_LAYOUT
from pathbench.utils.registries import FEATURE_EXTRACTORS


OPENSLIDE_SAMPLE_URL = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"


def _ensure_feature_extractor_registered(name: str) -> None:
    # Config validation requires the name to exist in FEATURE_EXTRACTORS
    if FEATURE_EXTRACTORS.is_available(name):
        return

    @FEATURE_EXTRACTORS.register(name)
    def _dummy():
        return name


def _download_if_needed(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (intentional for test download)
    tmp.replace(dst)
    return dst


@pytest.mark.smoke
def test_smoke_feature_extraction_lazyslide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ---- choose a tiny-ish end-to-end workload ----
    extractor = "resnet18"
    tile_px = 256
    tile_mpp = 0.5  # coarse -> fewer tiles

    _ensure_feature_extractor_registered(extractor)

    # ---- cache slide so repeated smoke runs don't keep downloading ----
    cache_dir = Path(
        os.environ.get("PATHBENCH_SMOKE_CACHE", "~/.cache/pathbench_smoke")
    ).expanduser()
    slide_path = cache_dir / "CMU-1-Small-Region.svs"
    _download_if_needed(OPENSLIDE_SAMPLE_URL, slide_path)

    # ---- build minimal dataset layout expected by your code ----
    slides_dir = tmp_path / "slides"
    artifacts_dir = tmp_path / "artifacts"
    project_root = tmp_path / "project"
    slides_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    local_slide = slides_dir / slide_path.name
    if not local_slide.exists():
        local_slide.write_bytes(slide_path.read_bytes())

    # annotations.csv must match WSIDataset expectations: dataset, slide, patient, category
    ann_path = tmp_path / "annotations.csv"
    ann_path.write_text(
        "dataset,slide,patient,category\nsmoke,CMU-1-Small-Region,P0,cat0\n",
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

    policy.process_slide(dataset=ds, wsi=wsi, combo_cfg=combo)

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
