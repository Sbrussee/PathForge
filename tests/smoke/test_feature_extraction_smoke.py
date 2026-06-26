from __future__ import annotations

import numpy as np
import pytest

from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.io.slide_artifacts.layout import DEFAULT_LAYOUT

from ._smoke_dataset import ExtractedWsiWorkspace, attach_smoke_outputs, capture_smoke_metrics


@pytest.mark.smoke
def test_smoke_feature_extraction_writes_expected_artifacts(
    extracted_wsi_workspace: ExtractedWsiWorkspace,
    tmp_path,
) -> None:
    """Validate that real tile feature extraction produced correct H5 artifacts.

    This test exercises the H5 artifact structure produced by the session-scoped
    ``extracted_wsi_workspace`` fixture, which runs full PathForge feature
    extraction on real GTEx WSIs using resnet18.
    """
    bag_id = extracted_wsi_workspace.bag_id          # e.g. "224px_1mpp"
    extractor_name = extracted_wsi_workspace.extractor_name  # "resnet18"

    coords_path = DEFAULT_LAYOUT.coords_dataset(bag_id)
    tiling_path = DEFAULT_LAYOUT.tiling_spec_dataset(bag_id)
    overview_path = DEFAULT_LAYOUT.tiles_overview_dataset(bag_id)
    feats_path = DEFAULT_LAYOUT.features_dataset(bag_id, extractor_name)

    assert extracted_wsi_workspace.artifact_paths, "No artifact paths in workspace"

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_feature_extraction_artifact_validation",
        metadata={
            "num_slides": len(extracted_wsi_workspace.artifact_paths),
            "bag_id": bag_id,
            "extractor_name": extractor_name,
        },
    ) as metadata:
        per_slide_tile_counts: dict[str, int] = {}
        per_slide_feature_dims: dict[str, int] = {}

        for slide_id, artifact_path in sorted(extracted_wsi_workspace.artifact_paths.items()):
            assert artifact_path.exists(), f"H5 artifact missing for {slide_id}: {artifact_path}"

            with FileHandleH5(artifact_path, mode="r") as fh:
                assert DEFAULT_LAYOUT.tissue_dataset in fh.h5, (
                    f"tissue dataset missing in {artifact_path}"
                )
                assert coords_path in fh.h5, f"coords dataset missing in {artifact_path}"
                assert tiling_path in fh.h5, f"tiling_spec dataset missing in {artifact_path}"
                assert overview_path in fh.h5, f"tiles_overview dataset missing in {artifact_path}"
                assert feats_path in fh.h5, f"features dataset missing in {artifact_path}"

                coords = fh.h5[coords_path][()]
                overview = fh.h5[overview_path][()]
                feats = fh.h5[feats_path][()]

                assert coords.ndim == 2 and coords.shape[1] == 5, (
                    f"coords shape {coords.shape} unexpected in {artifact_path}"
                )
                assert overview.ndim == 1 and overview.dtype == np.uint8, (
                    f"overview dtype/shape unexpected in {artifact_path}"
                )
                assert overview.size > 0, f"overview empty in {artifact_path}"
                assert bytes(overview[:2].tolist()) == b"\xff\xd8", (
                    f"overview not JPEG in {artifact_path}"
                )
                assert feats.ndim == 2, f"features not 2D in {artifact_path}"
                assert feats.shape[0] == coords.shape[0], (
                    f"feature/coord count mismatch in {artifact_path}"
                )
                assert feats.shape[0] > 0, f"no tiles extracted in {artifact_path}"
                assert np.isfinite(feats).all(), f"non-finite features in {artifact_path}"

                per_slide_tile_counts[slide_id] = int(feats.shape[0])
                per_slide_feature_dims[slide_id] = int(feats.shape[1])

        attach_smoke_outputs(
            metadata,
            step_name="hf_feature_extraction_artifact_validation",
            intermediate={"artifacts_dir": extracted_wsi_workspace.artifacts_dir},
            final={},
        )

    assert len(per_slide_tile_counts) == len(extracted_wsi_workspace.artifact_paths)
    feature_dims = set(per_slide_feature_dims.values())
    assert len(feature_dims) == 1, f"Inconsistent feature dims across slides: {feature_dims}"
