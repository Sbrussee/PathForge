"""Smoke benchmarking workflows backed by reusable extracted bags."""

from __future__ import annotations

from pathlib import Path

import pytest

from ._smoke_dataset import PreparedBagWorkspace, capture_smoke_metrics
from ._smoke_training import fit_smoke_model


def _build_bag_dataset(workspace: PreparedBagWorkspace, *, target_column: str):
    """Construct a production ``BagDataset`` for one target column."""
    pytest.importorskip("torch")
    from pathbench.core.datasets.bag_dataset import BagDataset

    return BagDataset(
        f"smoke_{target_column}",
        str(workspace.feature_dir),
        str(workspace.metadata_csv),
        target_column,
    )


@pytest.mark.smoke
def test_binary_classification_mil_benchmark_grid(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Run a tiny binary MIL benchmark over a small hyperparameter grid."""
    dataset = _build_bag_dataset(extracted_bag_workspace, target_column="binary_label")
    runs = []

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_binary_classification_benchmark",
        metadata={"num_bags": len(dataset), "grid_size": 2},
    ) as metadata:
        for lr, dropout in ((1e-3, 0.0), (5e-4, 0.2)):
            _, result = fit_smoke_model(
                tmp_path / f"binary_{lr}_{dropout}",
                dataset_train=dataset,
                dataset_val=dataset,
                input_dim=extracted_bag_workspace.input_dim,
                output_dim=2,
                task="classification",
                loss_name="CrossEntropyLoss",
                epochs=1,
                lr=lr,
                dropout=dropout,
            )
            runs.append(result)
        metadata["best_scores"] = [run.best_score for run in runs]

    assert len(runs) == 2
    assert all(Path(run.best_model_path).exists() for run in runs)


@pytest.mark.smoke
def test_multiclass_classification_mil_benchmark_grid(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Run a tiny multiclass MIL benchmark using the same extracted bags."""
    import pandas as pd

    dataset = _build_bag_dataset(
        extracted_bag_workspace, target_column="multiclass_label"
    )
    runs = []

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_multiclass_classification_benchmark",
        metadata={"num_bags": len(dataset), "grid_size": 2},
    ):
        for lr, dropout in ((1e-3, 0.0), (5e-4, 0.1)):
            _, result = fit_smoke_model(
                tmp_path / f"multiclass_{lr}_{dropout}",
                dataset_train=dataset,
                dataset_val=dataset,
                input_dim=extracted_bag_workspace.input_dim,
                output_dim=3,
                task="classification",
                loss_name="CrossEntropyLoss",
                epochs=1,
                lr=lr,
                dropout=dropout,
            )
            runs.append(result)

    assert len(runs) == 2
    assert all(Path(run.best_model_path).exists() for run in runs)
    metadata_df = pd.read_csv(extracted_bag_workspace.metadata_csv)
    assert metadata_df["multiclass_label"].nunique() == 3
