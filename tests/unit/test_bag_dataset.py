from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from pathbench.core.datasets.bag_dataset import BagDataset


def test_bag_dataset_infers_feature_and_output_dimensions(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    annotations_path = tmp_path / "annotations.csv"
    pd.DataFrame(
        {
            "slide_id": ["S1", "S2", "S3"],
            "category": [0, 1, 2],
        }
    ).to_csv(annotations_path, index=False)
    torch.save(torch.zeros(4, 8, dtype=torch.float32), feature_dir / "S1.pt")
    torch.save(torch.ones(5, 8, dtype=torch.float32), feature_dir / "S2.pt")
    torch.save(torch.full((6, 8), 2.0, dtype=torch.float32), feature_dir / "S3.pt")

    dataset = BagDataset(
        "classification_ds",
        str(feature_dir),
        str(annotations_path),
        "category",
    )

    bag, target = dataset[0]
    assert bag.shape == (4, 8)
    assert target.dtype == torch.long
    assert dataset.feature_dim == 8
    assert dataset.output_dim() == 3


def test_bag_dataset_uses_variable_bag_size_by_default(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    annotations_path = tmp_path / "annotations.csv"
    pd.DataFrame({"slide_id": ["S1"], "category": [0]}).to_csv(
        annotations_path, index=False
    )
    original_bag = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    torch.save(original_bag, feature_dir / "S1.pt")

    dataset = BagDataset(
        "variable_bag_ds",
        str(feature_dir),
        str(annotations_path),
        "category",
    )

    bag, _ = dataset[0]
    assert bag.shape == (5, 3)
    assert torch.equal(bag, original_bag)


def test_bag_dataset_builds_survival_targets(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    annotations_path = tmp_path / "survival.csv"
    pd.DataFrame(
        {
            "slide": ["S1"],
            "os_months": [12.0],
            "status": [1.0],
            "category": [0],
        }
    ).to_csv(annotations_path, index=False)
    torch.save(torch.zeros(3, 6, dtype=torch.float32), feature_dir / "S1.pt")

    dataset = BagDataset(
        "survival_ds",
        str(feature_dir),
        str(annotations_path),
        "category",
        task="survival",
    )

    _, target = dataset[0]
    assert set(target) == {"time", "event"}
    assert target["time"].dtype == torch.float32
    assert target["event"].dtype == torch.float32
    assert dataset.output_dim() == 1


def test_bag_dataset_infers_discrete_survival_output_dim_from_annotation_bins(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    annotations_path = tmp_path / "survival_discrete.csv"
    pd.DataFrame(
        {
            "slide": ["S1", "S2", "S3"],
            "time_bin": [0, 2, 3],
            "status": [1.0, 0.0, 1.0],
            "category": [0, 0, 0],
        }
    ).to_csv(annotations_path, index=False)
    for slide_id in ("S1", "S2", "S3"):
        torch.save(torch.zeros(2, 5, dtype=torch.float32), feature_dir / f"{slide_id}.pt")

    dataset = BagDataset(
        "survival_discrete_ds",
        str(feature_dir),
        str(annotations_path),
        "category",
        task="survival_discrete",
    )

    _, target = dataset[1]
    assert target["time"].dtype == torch.long
    assert target["event"].dtype == torch.float32
    assert dataset.feature_dim == 5
    assert dataset.output_dim() == 4


def test_bag_dataset_builds_continuous_regression_targets(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    annotations_path = tmp_path / "regression.csv"
    pd.DataFrame(
        {
            "slide_id": ["S1", "S2"],
            "score": [1.25, 3.75],
        }
    ).to_csv(annotations_path, index=False)
    torch.save(torch.zeros(2, 7, dtype=torch.float32), feature_dir / "S1.pt")
    torch.save(torch.ones(3, 7, dtype=torch.float32), feature_dir / "S2.pt")

    dataset = BagDataset(
        "regression_ds",
        str(feature_dir),
        str(annotations_path),
        "score",
        task="regression",
    )

    bag, target = dataset[1]
    assert bag.shape == (3, 7)
    assert target.dtype == torch.float32
    assert float(target) == 3.75
    assert dataset.feature_dim == 7
    assert dataset.output_dim() == 1


def test_bag_dataset_materializes_fixed_bag_size_deterministically(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    annotations_path = tmp_path / "fixed_size.csv"
    pd.DataFrame({"slide_id": ["S1", "S2"], "category": [0, 1]}).to_csv(
        annotations_path, index=False
    )
    torch.save(torch.arange(18, dtype=torch.float32).reshape(6, 3), feature_dir / "S1.pt")
    torch.save(torch.arange(6, dtype=torch.float32).reshape(2, 3), feature_dir / "S2.pt")

    dataset = BagDataset(
        "fixed_bag_ds",
        str(feature_dir),
        str(annotations_path),
        "category",
        bag_size=4,
    )

    larger_bag, _ = dataset[0]
    smaller_bag, _ = dataset[1]

    assert larger_bag.shape == (4, 3)
    assert smaller_bag.shape == (4, 3)
    assert torch.equal(larger_bag[:, 0], torch.tensor([0.0, 6.0, 9.0, 15.0]))
    assert torch.equal(smaller_bag[:, 0], torch.tensor([0.0, 3.0, 0.0, 3.0]))
