from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pathforge.config.config import DatasetEntry
from pathforge.core.datasets.bag_dataset import BagDataset
from pathforge.core.datasets.bag_schema import assert_bag_schema
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.core.io.h5 import features as features_io
from pathforge.core.io.h5 import tiles as tiles_io
from pathforge.core.io.h5.base import FileHandleH5

BAG_ID = "256px_0.5mpp"
EXTRACTOR = "test-extractor"
COMBO = ComboConfig(
    tile_px=256, tile_mpp=0.5, feature_extraction=EXTRACTOR, color_norm=None
)


def _dataset(
    tmp_path: Path,
    rows,
    bags,
    *,
    name="dataset",
    task="classification",
    target_column="category",
    bag_size=None,
    time_column=None,
    event_column=None,
    slide_column=None,
):
    artifacts_dir, slides_dir = tmp_path / "artifacts", tmp_path / "slides"
    artifacts_dir.mkdir(exist_ok=True)
    slides_dir.mkdir(exist_ok=True)
    annotations = pd.DataFrame(rows)
    annotations["dataset"] = name
    for slide_id, bag in bags.items():
        coords = np.zeros((bag.shape[0], 5), dtype=np.int32)
        with FileHandleH5(artifacts_dir / f"{slide_id}.h5", mode="a") as artifact:
            tiles_io.write_coords(artifact, BAG_ID, coords)
            tiles_io.write_tiling_spec(
                artifact, BAG_ID, {"tile_px": 256, "tile_mpp": 0.5}
            )
            features_io.write_features(artifact, BAG_ID, EXTRACTOR, bag.numpy())
    return BagDataset(
        ds_cfg=DatasetEntry(
            name=name,
            slides_dir=str(slides_dir),
            artifacts_dir=str(artifacts_dir),
            used_for="training",
        ),
        annotations_df=annotations,
        combo_cfg=COMBO,
        aggregation_level="slide",
        task=task,
        target_column=target_column,
        bag_size=bag_size,
        time_column=time_column,
        event_column=event_column,
        slide_column=slide_column,
    )


def test_bag_dataset_infers_feature_and_output_dimensions(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        [
            {"slide": "S1", "category": 0},
            {"slide": "S2", "category": 1},
            {"slide": "S3", "category": 2},
        ],
        {
            "S1": torch.zeros(4, 8),
            "S2": torch.ones(5, 8),
            "S3": torch.full((6, 8), 2.0),
        },
    )
    sample = dataset[0]
    assert_bag_schema(sample, batched=False)
    assert sample["X"].shape == (4, 8)
    assert sample["Y"].dtype == torch.long
    assert dataset.feature_dim == 8
    assert dataset.output_dim() == 3


def test_bag_dataset_uses_variable_bag_size_by_default(tmp_path: Path) -> None:
    original = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    dataset = _dataset(tmp_path, [{"slide": "S1", "category": 0}], {"S1": original})
    assert torch.equal(dataset[0]["X"], original)


def test_bag_dataset_honors_configured_slide_id_column(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        [{"slide_id": "S1", "category": 1}],
        {"S1": torch.ones(2, 3)},
        slide_column="slide_id",
    )

    assert dataset.samples[0].sample_id == "S1"
    assert torch.equal(dataset[0]["X"], torch.ones(2, 3))


def test_bag_dataset_builds_survival_targets(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        [{"slide": "S1", "os_months": 12.0, "status": 1.0, "category": 0}],
        {"S1": torch.zeros(3, 6)},
        task="survival",
        time_column="os_months",
        event_column="status",
    )
    target = dataset[0]["Y"]
    assert set(target) == {"time", "event"}
    assert target["time"].dtype == torch.float32
    assert target["event"].dtype == torch.float32
    assert dataset.output_dim() == 1


def test_bag_dataset_infers_discrete_survival_output_dim(tmp_path: Path) -> None:
    rows = [
        {"slide": "S1", "time_bin": 0, "os_months": 6.0, "status": 1.0, "category": 0},
        {"slide": "S2", "time_bin": 2, "os_months": 18.0, "status": 0.0, "category": 0},
        {"slide": "S3", "time_bin": 3, "os_months": 30.0, "status": 1.0, "category": 0},
    ]
    dataset = _dataset(
        tmp_path,
        rows,
        {slide: torch.zeros(2, 5) for slide in ("S1", "S2", "S3")},
        task="survival_discrete",
        time_column="time_bin",
        event_column="status",
    )
    target = dataset[1]["Y"]
    assert target["time"].dtype == torch.long
    assert target["continuous_time"].item() == 18.0
    assert dataset.feature_dim == 5
    assert dataset.output_dim() == 4


def test_bag_dataset_builds_continuous_regression_targets(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        [{"slide": "S1", "score": 1.25}, {"slide": "S2", "score": 3.75}],
        {"S1": torch.zeros(2, 7), "S2": torch.ones(3, 7)},
        task="regression",
        target_column="score",
    )
    sample = dataset[1]
    assert sample["X"].shape == (3, 7)
    assert sample["Y"].dtype == torch.float32
    assert float(sample["Y"]) == 3.75
    assert dataset.output_dim() == 1


def test_bag_dataset_materializes_fixed_bag_size_deterministically(
    tmp_path: Path,
) -> None:
    dataset = _dataset(
        tmp_path,
        [{"slide": "S1", "category": 0}, {"slide": "S2", "category": 1}],
        {
            "S1": torch.arange(18, dtype=torch.float32).reshape(6, 3),
            "S2": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        },
        bag_size=4,
    )
    assert torch.equal(dataset[0]["X"][:, 0], torch.tensor([0.0, 6.0, 9.0, 15.0]))
    assert torch.equal(dataset[1]["X"][:, 0], torch.tensor([0.0, 3.0, 0.0, 3.0]))


def test_bag_dataset_rejects_removed_prepared_tensor_constructor() -> None:
    with pytest.raises(TypeError):
        BagDataset("dataset", "features", "annotations.csv", "category")
