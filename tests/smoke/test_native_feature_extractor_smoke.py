"""End-to-end smoke coverage for registered PathForge feature extractors."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import Tensor

from pathforge.core.feature_extractors.base import FeatureExtractorBase
from pathforge.core.feature_extractors.factory import register_feature_extractor
from pathforge.core.feature_extractors.selection import FeatureExtractorSelection
from pathforge.core.io.h5.base import FileHandleH5
from pathforge.core.io.h5.layout import DEFAULT_LAYOUT
from pathforge.core.slide_processing.base import FeatureExtractionRequest

from ._smoke_dataset import ExtractedWsiWorkspace, capture_smoke_metrics, read_h5_coords


@register_feature_extractor("smoke_native_mean_rgb")
class SmokeNativeMeanRgbExtractor(FeatureExtractorBase):
    """Small native extractor used to validate the real LazySlide adapter seam.

    The extractor is deliberately deterministic and produces three features per
    tile, making it safe to run alongside the network-backed smoke suite.
    """

    def build_model(self) -> torch.nn.Module:
        """Return a parameter-free model placeholder for the adapter contract."""
        return torch.nn.Identity()

    def get_transform(self):  # type: ignore[no-untyped-def]
        """Convert canonical RGB uint8 patches into CHW float tensors."""
        return lambda patch: torch.from_numpy(patch).permute(2, 0, 1).float() / 255

    def encode_images(self, images: Tensor) -> Tensor:
        """Return the per-channel mean for a ``[B, C, H, W]`` image batch."""
        return images.mean(dim=(2, 3))


@pytest.mark.smoke
def test_registered_native_extractor_returns_row_aligned_features(
    extracted_wsi_workspace: ExtractedWsiWorkspace,
    tmp_path,
) -> None:
    """Run one real WSI through the registered extractor and verify its output.

    The shared workspace supplies real sample slides and tiles. This test runs
    the public LazySlide processor with a registered PathForge extractor, then
    records timing and memory metrics while checking the H5 feature contract.
    """
    pytest.importorskip("lazyslide")
    from pathforge.core.datasets.wsi_dataset import WSI
    from pathforge.core.slide_processing.factory import build_slide_processor

    slide_id, source_artifact = next(
        iter(extracted_wsi_workspace.artifact_paths.items())
    )
    bag_id = extracted_wsi_workspace.bag_id
    coords = read_h5_coords(source_artifact, bag_id=bag_id)
    with FileHandleH5(source_artifact, mode="r") as artifact:
        tiling_spec = json.loads(
            artifact.h5[DEFAULT_LAYOUT.tiling_spec_dataset(bag_id)][()].decode()
        )

    # Reuse the downloaded physical WSI while keeping this extraction independent
    # from the resnet18 artifact made by the shared smoke fixture.
    slide_path = extracted_wsi_workspace.slides_dir / f"{slide_id}.svs"
    wsi = WSI(
        slide=slide_id,
        patient=slide_id,
        category="smoke",
        path=slide_path,
        artifact_path=tmp_path / f"{slide_id}.h5",
    )
    processor = build_slide_processor("lazyslide")
    processor.load_wsi(wsi)
    try:
        with capture_smoke_metrics(
            tmp_path / "metrics",
            step_name="native_feature_extractor_lazyslide",
            metadata={"slide_id": slide_id, "extractor_name": "smoke_native_mean_rgb"},
        ):
            matrix = processor.extract_features(
                wsi,
                coords,
                tiling_spec,
                FeatureExtractionRequest(
                    selection=FeatureExtractorSelection(
                        name="smoke_native_mean_rgb",
                        source="pathforge-native",
                    ),
                    execution_params={"batch_size": 8, "num_workers": 0},
                ),
            )
    finally:
        processor.close_wsi(wsi)

    assert matrix.shape == (coords.shape[0], 3)
    assert np.isfinite(matrix).all()
    metrics = json.loads(
        (
            tmp_path / "metrics" / "native_feature_extractor_lazyslide.metrics.json"
        ).read_text()
    )
    assert metrics["elapsed_seconds"] > 0
    assert "ru_maxrss_mb" in metrics
