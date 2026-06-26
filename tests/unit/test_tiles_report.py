"""Unit tests for tile report collection and CLI helpers."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pathbench.cli.report_tiles import _unique_tiling_ids_from_config as _unique_bag_ids_from_config
from pathbench.core.datasets.wsi_dataset import WSI
from pathbench.core.io.h5 import tiles as tiles_io
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.reports.tiles_report_pdf import (
    collect_tiles_overview_entries,
    create_tiles_report_pdf,
)


def _make_dataset(tmp_path: Path) -> SimpleNamespace:
    """Create a small dataset-like object compatible with the report helpers."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sample_1 = WSI(
        slide="S1",
        patient="P1",
        category="cat1",
        path=tmp_path / "S1.svs",
        artifact_path=artifact_dir / "S1.h5",
    )
    sample_2 = WSI(
        slide="S2",
        patient="P2",
        category="cat2",
        path=tmp_path / "S2.svs",
        artifact_path=artifact_dir / "S2.h5",
    )
    return SimpleNamespace(
        name="dataset_a",
        artifacts_dir=artifact_dir,
        samples=[sample_1, sample_2],
    )


def _write_reportable_artifact(
    artifact_path: Path, *, bag_id: str, image_bytes: bytes
) -> None:
    """Write the minimal H5 contents required for tiles report generation."""
    coords = np.asarray(
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
    }
    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        tiles_io.write_coords(slide_artifact, bag_id, coords)
        tiles_io.write_tiling_spec(slide_artifact, bag_id, tiling_spec)
        tiles_io.write_tiles_overview(slide_artifact, bag_id, image_bytes)


def _valid_overview_bytes(color: tuple[int, int, int] = (10, 120, 200)) -> bytes:
    """Create a small valid PNG payload for tiles_overview tests."""
    image = Image.new("RGB", (16, 16), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_collect_tiles_overview_entries_counts_present_and_missing_slides(
    tmp_path: Path,
) -> None:
    """Collection should include reportable slides and count missing overviews."""
    dataset = _make_dataset(tmp_path)
    bag_id = "256px_0.5mpp"
    _write_reportable_artifact(
        dataset.samples[0].artifact_path,
        bag_id=bag_id,
        image_bytes=_valid_overview_bytes(),
    )

    collection = collect_tiles_overview_entries(dataset=dataset, bag_id=bag_id)

    assert collection.stats.total_slides_expected == 2
    assert collection.stats.included_slides == 1
    assert collection.stats.missing_overview == 1
    assert collection.entries[0].slide_id == "S1"
    assert collection.entries[0].num_tiles == 2


def test_create_tiles_report_pdf_writes_pdf(tmp_path: Path) -> None:
    """A dataset with valid overview bytes should yield a report PDF."""
    dataset = _make_dataset(tmp_path)
    bag_id = "256px_0.5mpp"
    for sample in dataset.samples:
        _write_reportable_artifact(
            sample.artifact_path,
            bag_id=bag_id,
            image_bytes=_valid_overview_bytes(),
        )

    output_path = tmp_path / "tiles_report.pdf"
    written_path = create_tiles_report_pdf(
        dataset=dataset,
        bag_id=bag_id,
        output_path=output_path,
    )

    assert written_path == output_path
    assert written_path.exists()
    assert written_path.stat().st_size > 0


def test_unique_bag_ids_from_config_deduplicates_combinations() -> None:
    """The tiles-report CLI helper should preserve order while deduplicating."""
    cfg = SimpleNamespace(
        benchmark_parameters=SimpleNamespace(
            tile_px=[256, 256],
            tile_mpp=[0.5, 1.0],
        )
    )

    bag_ids = _unique_bag_ids_from_config(cfg)

    assert bag_ids == ["256px_0.5mpp", "256px_1mpp"]


def test_collect_tiles_overview_entries_counts_corrupt_overviews(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    bag_id = "256px_0.5mpp"
    _write_reportable_artifact(
        dataset.samples[0].artifact_path,
        bag_id=bag_id,
        image_bytes=b"not-a-real-image",
    )

    collection = collect_tiles_overview_entries(dataset=dataset, bag_id=bag_id)

    assert collection.stats.total_slides_expected == 2
    assert collection.stats.included_slides == 0
    assert collection.stats.corrupt_overview == 1
    assert collection.stats.missing_overview == 1


def test_create_tiles_report_pdf_raises_when_no_valid_overviews(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)

    with pytest.raises(RuntimeError, match="No tiles_overview entries found"):
        create_tiles_report_pdf(
            dataset=dataset,
            bag_id="256px_0.5mpp",
            output_path=tmp_path / "tiles_report.pdf",
        )
