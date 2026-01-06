"""Unit tests for BagDataset grouping and sampling strategies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from pathbench.config.config import BagDatasetConfig
from pathbench.core.datasets.bag_dataset import BagDataset


def _write_feature_bag(
    features_dir: Path,
    slide_id: str,
    num_tiles: int,
    feature_dim: int,
    seed: int,
) -> torch.Tensor:
    """
    Create and persist a synthetic bag of tile features.

    Expected feature shape: (num_tiles, feature_dim).
    """
    generator = torch.Generator().manual_seed(seed)
    bag = torch.randn(num_tiles, feature_dim, generator=generator)
    torch.save(bag, features_dir / f"{slide_id}.pt")
    return bag


def _build_dataset(
    tmp_path: Path,
    features_dir: Path,
    annotations: pd.DataFrame,
    config: BagDatasetConfig,
) -> BagDataset:
    """
    Materialize annotations on disk and build a BagDataset instance.
    """
    ann_path = tmp_path / "annotations.csv"
    annotations.to_csv(ann_path, index=False)
    return BagDataset(
        _name="train",
        features_dir=features_dir,
        annotation_path=ann_path,
        config=config,
    )


@pytest.mark.parametrize(
    "grouping_strategy,expected_num_bags,expected_lengths",
    [
        ("slide", 3, [4, 6, 5]),
        ("patient", 2, [10, 5]),
        ("tissue", 2, [10, 5]),
    ],
)
def test_grouping_strategies_build_expected_bags(
    tmp_path,
    grouping_strategy,
    expected_num_bags,
    expected_lengths,
):
    """
    Validate grouping strategies (slide/patient/tissue) create correct bag counts.

    Each bag should be a tensor of shape (num_instances, feature_dim).
    """
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    slide_specs = [
        ("slide_a", 4, "patient_1", "tissue_1", 0),
        ("slide_b", 6, "patient_1", "tissue_1", 0),
        ("slide_c", 5, "patient_2", "tissue_2", 1),
    ]

    for idx, (slide_id, num_tiles, _patient, _tissue, _label) in enumerate(slide_specs):
        _write_feature_bag(features_dir, slide_id, num_tiles, feature_dim=8, seed=idx)

    annotations = pd.DataFrame(
        {
            "slide_id": [spec[0] for spec in slide_specs],
            "patient": [spec[2] for spec in slide_specs],
            "tissue_id": [spec[3] for spec in slide_specs],
            "dataset": ["train"] * len(slide_specs),
            "label": [spec[4] for spec in slide_specs],
        }
    )

    config = BagDatasetConfig(
        id_column="slide_id",
        label_column="label",
        dataset_column="dataset",
        grouping_strategy=grouping_strategy,
        patient_column="patient",
        tissue_column="tissue_id",
    )

    dataset = _build_dataset(tmp_path, features_dir, annotations, config)
    assert len(dataset) == expected_num_bags

    bag_lengths = [dataset[idx][0].shape[0] for idx in range(len(dataset))]
    assert sorted(bag_lengths) == sorted(expected_lengths)


@pytest.mark.parametrize("sampling_strategy", ["first", "random"])
def test_sampling_strategies_respect_max_instances(tmp_path, sampling_strategy):
    """
    Validate that sampling strategies cap bag size to max_instances.

    Expected sampled bag shape: (max_instances, feature_dim).
    """
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    slide_id = "slide_sampling"
    bag = torch.arange(20, dtype=torch.float32).view(10, 2)
    torch.save(bag, features_dir / f"{slide_id}.pt")

    annotations = pd.DataFrame(
        {
            "slide_id": [slide_id],
            "dataset": ["train"],
            "label": [1],
        }
    )

    config = BagDatasetConfig(
        id_column="slide_id",
        label_column="label",
        dataset_column="dataset",
        max_instances=3,
        sampling_strategy=sampling_strategy,
        random_seed=42,
    )

    dataset = _build_dataset(tmp_path, features_dir, annotations, config)
    sampled_bag, _label = dataset[0]
    assert sampled_bag.shape == (3, 2)

    if sampling_strategy == "first":
        assert torch.equal(sampled_bag, bag[:3])
    else:
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(bag.shape[0], generator=generator)[:3]
        assert torch.equal(sampled_bag, bag[indices])


def test_allow_missing_features_returns_empty_bag(tmp_path):
    """
    If allow_missing_features is True, missing feature files yield empty bags.

    Expected empty bag shape: (0, 0).
    """
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    annotations = pd.DataFrame(
        {
            "slide_id": ["missing_slide"],
            "dataset": ["train"],
            "label": [0],
        }
    )

    config = BagDatasetConfig(
        id_column="slide_id",
        label_column="label",
        dataset_column="dataset",
        allow_missing_features=True,
    )

    dataset = _build_dataset(tmp_path, features_dir, annotations, config)
    bag, label = dataset[0]
    assert bag.shape == (0, 0)
    assert label == 0


def test_feature_path_column_loads_relative_path(tmp_path):
    """
    Ensure feature_path_column overrides the default slide_id lookup.

    Expected bag shape: (num_tiles, feature_dim).
    """
    features_dir = tmp_path / "features"
    nested_dir = features_dir / "nested"
    nested_dir.mkdir(parents=True)

    slide_id = "slide_custom"
    bag = torch.randn(5, 3)
    torch.save(bag, nested_dir / f"{slide_id}.pt")

    annotations = pd.DataFrame(
        {
            "slide_id": [slide_id],
            "feature_path": [f"nested/{slide_id}.pt"],
            "dataset": ["train"],
            "label": [1],
        }
    )

    config = BagDatasetConfig(
        id_column="slide_id",
        label_column="label",
        dataset_column="dataset",
        feature_path_column="feature_path",
    )

    dataset = _build_dataset(tmp_path, features_dir, annotations, config)
    loaded_bag, label = dataset[0]
    assert loaded_bag.shape == bag.shape
    assert label == 1


def test_single_instance_feature_promotes_to_bag(tmp_path):
    """
    Single-instance feature vectors should be promoted to (1, feature_dim).

    Expected bag shape: (1, feature_dim).
    """
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    slide_id = "slide_vector"
    vector = torch.randn(5)
    torch.save(vector, features_dir / f"{slide_id}.pt")

    annotations = pd.DataFrame(
        {
            "slide_id": [slide_id],
            "dataset": ["train"],
            "label": [0],
        }
    )

    config = BagDatasetConfig(
        id_column="slide_id",
        label_column="label",
        dataset_column="dataset",
    )

    dataset = _build_dataset(tmp_path, features_dir, annotations, config)
    bag, _label = dataset[0]
    assert bag.shape == (1, 5)