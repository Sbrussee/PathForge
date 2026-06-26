from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from pathforge.cli.features_slide import _build_single_slide_wsi
from pathforge.core.datasets.wsi_dataset import WSI


class DummyDataset:
    def __init__(self, *, samples: list[WSI], artifact_path: Path) -> None:
        self.samples = samples
        self._artifact_path = artifact_path

    def slide_artifact_path(self, slide_id: str) -> Path:
        return self._artifact_path


def test_build_single_slide_wsi_preserves_dataset_fallback_mpp(tmp_path: Path) -> None:
    slide_id = "S1"
    input_path = tmp_path / "staging" / "S1.tiff"
    artifact_path = tmp_path / "artifacts" / "S1.h5"
    dataset = DummyDataset(
        samples=[
            WSI(
                slide=slide_id,
                patient="P1",
                category="C1",
                path=tmp_path / "slides" / "S1.tiff",
                artifact_path=artifact_path,
                fallback_mpp=0.25,
            )
        ],
        artifact_path=artifact_path,
    )
    row = pd.Series({"patient": "P1", "category": "C1", "fallback_mpp": 0.25})

    wsi = _build_single_slide_wsi(
        row=row,
        selected_dataset=dataset,
        slide_id=slide_id,
        input_slide_path=input_path,
    )

    assert wsi.path == input_path
    assert wsi.artifact_path == artifact_path
    assert wsi.fallback_mpp == 0.25


def test_build_single_slide_wsi_reads_fallback_mpp_when_dataset_sample_missing(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "S1.h5"
    dataset = SimpleNamespace(
        samples=[],
        slide_artifact_path=lambda slide_id: artifact_path,
    )
    row = pd.Series({"patient": "P1", "category": "C1", "fallback_mpp": "0.5"})

    wsi = _build_single_slide_wsi(
        row=row,
        selected_dataset=dataset,
        slide_id="S1",
        input_slide_path=tmp_path / "S1.tiff",
    )

    assert wsi.fallback_mpp == 0.5
