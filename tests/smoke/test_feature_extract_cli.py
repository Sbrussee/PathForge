from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pathbench.core.datasets.wsi_dataset import WSI
from pathbench.core.io.slide_artifacts import thumbnail as thumbnail_io
from pathbench.core.io.slide_artifacts.base import FileHandleH5
from pathbench.policy.feature_extraction import (
    FeatureExtractionPolicy,
    _PendingArtifactWrites,
)


class _SmokeSlideProcessor:
    def get_thumbnail(self, wsi: WSI, level: int = -1):
        thumbnail = np.full((24, 48, 3), 180, dtype=np.uint8)
        return thumbnail, 8.0, 8.0


@pytest.mark.smoke
def test_feature_extraction_thumbnail_write_smoke(tmp_path: Path) -> None:
    policy = FeatureExtractionPolicy.__new__(FeatureExtractionPolicy)
    policy.config = SimpleNamespace(
        experiment=SimpleNamespace(report=False, thumbnail=True)
    )

    wsi = WSI(
        slide="S1",
        patient="P1",
        category="cat",
        path=tmp_path / "S1.svs",
        artifact_path=tmp_path / "S1.h5",
    )
    pending_writes = _PendingArtifactWrites()

    policy._resolve_thumbnail(
        artifact_path=wsi.artifact_path,
        wsi=wsi,
        slide_processor=_SmokeSlideProcessor(),
        pending_writes=pending_writes,
    )

    with FileHandleH5(wsi.artifact_path, mode="a") as slide_artifact:
        policy._write_pending_artifact_updates(
            slide_artifact=slide_artifact,
            tiling_id="256px_0.5mpp",
            extractor_name="dummy_extractor",
            pending_writes=pending_writes,
        )

    with FileHandleH5(wsi.artifact_path, mode="r") as slide_artifact:
        assert thumbnail_io.thumbnail_image_exists(slide_artifact) is True
        assert thumbnail_io.thumbnail_spec_exists(slide_artifact) is True
        spec = thumbnail_io.read_thumbnail_spec(slide_artifact)
        assert spec["image_format"] == "jpeg"
        assert spec["coord_space"] == "level0"
        assert spec["downscale_x"] == 8.0
        assert spec["downscale_y"] == 8.0
