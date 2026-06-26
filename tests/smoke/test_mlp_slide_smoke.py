"""Smoke coverage for the SlideVectorMLP model trained via LightningTrainer.

``SlideVectorMLP`` mean-pools the bag (B, N, D) → (B, D) before applying a
small two-layer MLP, so it works as a slide-level predictor without any
attention or graph machinery.  Because it inherits from both
``SlideLevelModel`` and ``MILModelBase`` it plugs directly into the standard
``LightningTrainer.fit`` pipeline.

Tests exercise:
- Binary classification on GTEx bag workspace
- Continuous survival on TCGA READ survival workspace
- Discrete survival on TCGA READ survival workspace
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ._smoke_dataset import PreparedBagWorkspace, attach_smoke_outputs, capture_smoke_metrics
from ._smoke_training import (
    DEFAULT_SMOKE_EPOCHS,
    SurvivalBagDataset,
    fit_smoke_model,
    register_smoke_components,
)


def _build_slide_mlp(input_dim: int, output_dim: int):
    register_smoke_components()
    from pathbench.core.models.slide_mlp import SlideVectorMLP

    return SlideVectorMLP(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_slide_mlp_binary_classification_smoke(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """SlideVectorMLP binary classification through LightningTrainer."""
    from pathbench.core.datasets.bag_dataset import BagDataset

    register_smoke_components()
    dataset = BagDataset(
        "mlp_binary_smoke",
        str(extracted_bag_workspace.feature_dir),
        str(extracted_bag_workspace.metadata_csv),
        "binary_label",
    )

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="slide_mlp_binary_classification",
        metadata={"num_bags": len(dataset)},
    ) as meta:
        model = _build_slide_mlp(
            input_dim=extracted_bag_workspace.input_dim, output_dim=2
        )

        from pathbench.training.lightning import LightningTrainer
        from pathbench.utils.registries import LOSSES

        from ._smoke_training import make_training_config

        cfg = make_training_config(
            tmp_path / "mlp_binary",
            task="classification",
            epochs=DEFAULT_SMOKE_EPOCHS,
            lr=1e-3,
            dropout=0.0,
            batch_size=2,
        )
        cfg.mil.best_epoch_based_on = "balanced_accuracy"
        loss_fn = LOSSES.get("CrossEntropyLoss")()
        trainer = LightningTrainer(cfg)
        best_path, best_score = trainer.fit(model, dataset, dataset, loss_fn)

        artifacts_dir = trainer.metrics_artifacts_dir
        attach_smoke_outputs(
            meta,
            step_name="slide_mlp_binary_classification",
            final={"best_model_path": Path(best_path)},
        )

    assert Path(best_path).exists(), "No model checkpoint saved."
    assert (artifacts_dir / "val_confusion_matrix.png").exists()
    assert (artifacts_dir / "val_roc_auc_curve.png").exists()
    assert (artifacts_dir / "val_metrics.json").exists()


# ---------------------------------------------------------------------------
# Continuous survival
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_slide_mlp_continuous_survival_smoke(
    survival_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """SlideVectorMLP continuous survival training through LightningTrainer."""
    register_smoke_components()
    metadata_df = pd.read_csv(survival_bag_workspace.metadata_csv)
    dataset = SurvivalBagDataset(
        metadata_df,
        feature_dir=survival_bag_workspace.feature_dir,
        time_column="os_months",
        event_column="status",
        discrete_time=False,
    )

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="slide_mlp_continuous_survival",
        metadata={"num_bags": len(dataset)},
    ) as meta:
        _, result = fit_smoke_model(
            tmp_path / "mlp_continuous_survival",
            dataset_train=dataset,
            dataset_val=dataset,
            input_dim=survival_bag_workspace.input_dim,
            output_dim=1,
            task="survival",
            loss_name="CoxPHLoss",
            epochs=DEFAULT_SMOKE_EPOCHS,
            lr=1e-3,
            dropout=0.0,
        )
        attach_smoke_outputs(
            meta,
            step_name="slide_mlp_continuous_survival",
            final={"best_model_path": Path(result.best_model_path)},
        )

    assert Path(result.best_model_path).exists()
    assert Path(result.artifacts_dir, "val_td_auc_curve.png").exists()
    assert Path(result.artifacts_dir, "val_concordance_index.png").exists()
    assert Path(result.artifacts_dir, "val_kaplan_meier.png").exists()


# ---------------------------------------------------------------------------
# Discrete survival
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_slide_mlp_discrete_survival_smoke(
    survival_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """SlideVectorMLP discrete survival training through LightningTrainer."""
    register_smoke_components()
    metadata_df = pd.read_csv(survival_bag_workspace.metadata_csv)
    dataset = SurvivalBagDataset(
        metadata_df,
        feature_dir=survival_bag_workspace.feature_dir,
        time_column="time_bin",
        event_column="status",
        discrete_time=True,
    )
    num_bins = int(metadata_df["time_bin"].max()) + 1

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="slide_mlp_discrete_survival",
        metadata={"num_bags": len(dataset), "num_bins": num_bins},
    ) as meta:
        _, result = fit_smoke_model(
            tmp_path / "mlp_discrete_survival",
            dataset_train=dataset,
            dataset_val=dataset,
            input_dim=survival_bag_workspace.input_dim,
            output_dim=num_bins,
            task="survival_discrete",
            loss_name="DiscreteTimeNLLLoss",
            epochs=DEFAULT_SMOKE_EPOCHS,
            lr=1e-3,
            dropout=0.0,
        )
        attach_smoke_outputs(
            meta,
            step_name="slide_mlp_discrete_survival",
            final={"best_model_path": Path(result.best_model_path)},
        )

    assert Path(result.best_model_path).exists()
    assert Path(result.artifacts_dir, "val_td_auc_curve.png").exists()
    assert Path(result.artifacts_dir, "val_kaplan_meier.png").exists()
