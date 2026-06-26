"""Realistic smoke coverage for Hugging Face-backed feature workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pathforge.core.io.h5.base import FileHandleH5
from pathforge.core.io.h5 import features as features_io
from pathforge.core.io.h5.layout import DEFAULT_LAYOUT

from ._smoke_dataset import (
    ExtractedWsiWorkspace,
    PreparedBagWorkspace,
    attach_smoke_outputs,
    capture_smoke_metrics,
    read_h5_coords,
    save_slide_feature_matrix,
)


@pytest.mark.smoke
def test_tile_level_feature_extraction_reuses_downloaded_hf_slides(
    extracted_wsi_workspace: ExtractedWsiWorkspace,
) -> None:
    """Validate tile extraction artifacts for a few small Hugging Face slides."""
    assert extracted_wsi_workspace.metrics_path.exists()
    metrics = json.loads(
        extracted_wsi_workspace.metrics_path.read_text(encoding="utf-8")
    )
    assert metrics["num_slides"] == 45
    assert metrics["elapsed_seconds"] > 0

    bag_id = extracted_wsi_workspace.bag_id
    extractor_name = extracted_wsi_workspace.extractor_name
    coords_dataset = DEFAULT_LAYOUT.coords_dataset(bag_id)
    tiling_spec_dataset = DEFAULT_LAYOUT.tiling_spec_dataset(bag_id)
    overview_dataset = DEFAULT_LAYOUT.tiles_overview_dataset(bag_id)
    features_dataset = DEFAULT_LAYOUT.features_dataset(bag_id, extractor_name)

    for slide_id, artifact_path in extracted_wsi_workspace.artifact_paths.items():
        assert artifact_path.exists(), slide_id
        coords = read_h5_coords(artifact_path, bag_id=bag_id)
        with FileHandleH5(artifact_path, mode="r") as slide_artifact:
            assert DEFAULT_LAYOUT.tissue_dataset in slide_artifact.h5
            assert coords_dataset in slide_artifact.h5
            assert tiling_spec_dataset in slide_artifact.h5
            assert overview_dataset in slide_artifact.h5
            assert features_dataset in slide_artifact.h5
            features = features_io.read_features(slide_artifact, bag_id, extractor_name)
            overview = slide_artifact.h5[overview_dataset][()]

        assert coords.ndim == 2 and coords.shape[1] == 5
        assert coords.shape[0] > 0
        assert np.isfinite(coords).all()
        assert features.ndim == 2
        assert features.shape[0] == coords.shape[0]
        assert np.isfinite(features).all()
        assert overview.ndim == 1
        assert overview.dtype.name == "uint8"
        assert overview.size > 4
        assert bytes(overview[:2].tolist()) == b"\xff\xd8"


@pytest.mark.smoke
def test_slide_level_feature_aggregation_reuses_tile_artifacts(
    extracted_wsi_workspace: ExtractedWsiWorkspace,
    slide_level_feature_matrix: tuple[list[str], np.ndarray],
    tmp_path: Path,
) -> None:
    """Aggregate tile-level H5 features into a compact slide-level matrix."""
    slide_ids, slide_features = slide_level_feature_matrix
    output_path = tmp_path / "slide_level_features.npy"

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_slide_level_feature_aggregation",
        metadata={"num_slides": len(slide_ids)},
    ) as metadata:
        saved_path = save_slide_feature_matrix(
            output_path,
            slide_ids=slide_ids,
            slide_features=slide_features,
        )
        attach_smoke_outputs(
            metadata,
            step_name="hf_slide_level_feature_aggregation",
            intermediate={"source_artifacts_dir": extracted_wsi_workspace.artifacts_dir},
            final={
                "slide_feature_matrix": saved_path,
                "slide_feature_metadata": saved_path.with_suffix(".json"),
            },
        )

    assert slide_ids == sorted(extracted_wsi_workspace.artifact_paths)
    assert slide_features.ndim == 2
    assert slide_features.shape[0] == len(slide_ids)
    assert slide_features.shape[1] > 0
    assert np.isfinite(slide_features).all()
    assert saved_path.exists()
    assert saved_path.with_suffix(".json").exists()


@pytest.mark.smoke
def test_extracted_h5_bags_are_reused_for_downstream_mil_tasks(
    extracted_bag_workspace: PreparedBagWorkspace,
    extracted_wsi_workspace: ExtractedWsiWorkspace,
) -> None:
    """Check that the smoke suite prepared reusable MIL bag tensors."""
    assert extracted_bag_workspace.metrics_path.exists()
    for slide_id in extracted_bag_workspace.slide_ids:
        bag_path = extracted_bag_workspace.feature_dir / f"{slide_id}.pt"
        assert bag_path.exists()
        assert slide_id in extracted_wsi_workspace.artifact_paths
        assert extracted_bag_workspace.bag_lengths[slide_id] > 0
