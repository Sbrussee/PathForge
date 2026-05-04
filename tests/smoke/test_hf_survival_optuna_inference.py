"""Smoke coverage for survival, Optuna, and inference heatmaps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ._smoke_dataset import (
    ExtractedWsiWorkspace,
    PreparedBagWorkspace,
    capture_smoke_metrics,
    read_h5_feature_matrix,
)
from ._smoke_training import (
    SurvivalBagDataset,
    fit_smoke_model,
    register_smoke_components,
)


@pytest.mark.smoke
def test_continuous_survival_mil_smoke(
    survival_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Run a tiny continuous-survival MIL training loop on TCGA READ features."""
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
        step_name="hf_continuous_survival_benchmark",
        metadata={"num_bags": len(dataset)},
    ):
        _, result = fit_smoke_model(
            tmp_path / "continuous_survival",
            dataset_train=dataset,
            dataset_val=dataset,
            input_dim=survival_bag_workspace.input_dim,
            output_dim=1,
            task="survival",
            loss_name="SmokeSurvivalMSELoss",
            epochs=1,
            lr=1e-3,
            dropout=0.0,
        )

    assert Path(result.best_model_path).exists()


@pytest.mark.smoke
def test_discrete_survival_mil_smoke(
    survival_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Run a tiny discrete-survival MIL training loop on TCGA READ features."""
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
        step_name="hf_discrete_survival_benchmark",
        metadata={"num_bags": len(dataset), "num_bins": num_bins},
    ):
        _, result = fit_smoke_model(
            tmp_path / "discrete_survival",
            dataset_train=dataset,
            dataset_val=dataset,
            input_dim=survival_bag_workspace.input_dim,
            output_dim=num_bins,
            task="survival_discrete",
            loss_name="SmokeDiscreteSurvivalLoss",
            epochs=1,
            lr=1e-3,
            dropout=0.1,
        )

    assert Path(result.best_model_path).exists()


@pytest.mark.smoke
def test_binary_classification_optuna_smoke(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Run a very small Optuna study over binary MIL hyperparameters."""
    optuna = pytest.importorskip("optuna")
    pytest.importorskip("torch")
    from pathbench.core.datasets.bag_dataset import BagDataset

    register_smoke_components()
    dataset = BagDataset(
        "smoke_optuna_binary",
        str(extracted_bag_workspace.feature_dir),
        str(extracted_bag_workspace.metadata_csv),
        "binary_label",
    )
    objective_root = tmp_path / "optuna_runs"

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.25)
        _, result = fit_smoke_model(
            objective_root / f"trial_{trial.number}",
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
        return result.best_score

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_binary_classification_optuna",
        metadata={"n_trials": 2},
    ):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=2)

    assert len(study.trials) == 2
    assert all(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    assert "lr" in study.best_params
    assert "dropout" in study.best_params


@pytest.mark.smoke
def test_trained_mil_inference_heatmap_cli(
    extracted_bag_workspace: PreparedBagWorkspace,
    extracted_wsi_workspace: ExtractedWsiWorkspace,
    tmp_path: Path,
) -> None:
    """Attach a visual heatmap to an extracted slide artifact via the inference CLI."""
    torch = pytest.importorskip("torch")
    from pathbench.cli.inference import main as inference_main
    from pathbench.core.datasets.bag_dataset import BagDataset
    from pathbench.core.io.h5.base import FileHandleH5
    from pathbench.core.io.h5 import heatmaps as heatmap_io

    dataset = BagDataset(
        "smoke_inference_binary",
        str(extracted_bag_workspace.feature_dir),
        str(extracted_bag_workspace.metadata_csv),
        "binary_label",
    )

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_inference_heatmap_cli",
        metadata={"num_bags": len(dataset)},
    ):
        model, result = fit_smoke_model(
            tmp_path / "inference_train",
            dataset_train=dataset,
            dataset_val=dataset,
            input_dim=extracted_bag_workspace.input_dim,
            output_dim=2,
            task="classification",
            loss_name="CrossEntropyLoss",
            epochs=1,
            lr=1e-3,
            dropout=0.0,
        )

        slide_id = extracted_bag_workspace.slide_ids[0]
        artifact_path = extracted_wsi_workspace.artifact_paths[slide_id]
        bag_tensor = torch.load(
            extracted_bag_workspace.feature_dir / f"{slide_id}.pt"
        ).unsqueeze(0)
        instance_scores = (
            model.instance_scores(bag_tensor)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        tile_features = read_h5_feature_matrix(
            artifact_path,
            bag_id=extracted_wsi_workspace.bag_id,
            extractor_name=extracted_wsi_workspace.extractor_name,
        )
        assert instance_scores.shape[0] == tile_features.shape[0]

        scores_path = tmp_path / f"{slide_id}_scores.npy"
        output_json = tmp_path / f"{slide_id}_prediction.json"
        heatmap_json = tmp_path / f"{slide_id}_heatmap.json"
        checkpoint_path = tmp_path / "smoke_model.ckpt"
        np.save(scores_path, instance_scores)
        model.save(str(checkpoint_path))

        exit_code = inference_main(
            [
                "--model_path",
                str(checkpoint_path),
                "--input",
                str(artifact_path),
                "--output",
                str(output_json),
                "--heatmap-backend",
                "smoke_heatmap",
                "--bag-id",
                extracted_wsi_workspace.bag_id,
                "--scores",
                str(scores_path),
                "--heatmap-name",
                "smoke_attention",
                "--heatmap-output",
                str(heatmap_json),
            ]
        )

    assert exit_code == 0
    assert Path(result.best_model_path).exists()
    assert output_json.exists()
    assert heatmap_json.exists()

    prediction_payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert prediction_payload["status"] == "ok"
    assert prediction_payload["heatmap"]["num_points"] == int(instance_scores.shape[0])

    with FileHandleH5(artifact_path, mode="r") as slide_artifact:
        heatmap = heatmap_io.read_prediction_heatmap(
            slide_artifact,
            extracted_wsi_workspace.bag_id,
            "smoke_attention",
        )

    assert heatmap["coords"].shape[0] == instance_scores.shape[0]
    assert heatmap["scores"].shape == (instance_scores.shape[0],)
    assert np.isfinite(heatmap["scores"]).all()
    assert float(heatmap["scores"].min()) >= 0.0
    assert float(heatmap["scores"].max()) <= 1.0
