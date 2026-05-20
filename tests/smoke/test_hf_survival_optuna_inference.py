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
    attach_smoke_outputs,
    capture_smoke_metrics,
    read_h5_feature_matrix,
)
from ._smoke_training import (
    DEFAULT_SMOKE_EPOCHS,
    SurvivalBagDataset,
    fit_smoke_model,
    register_smoke_components,
    training_artifact_outputs,
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
    ) as metadata:
        _, result = fit_smoke_model(
            tmp_path / "continuous_survival",
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
            metadata,
            step_name="hf_continuous_survival_benchmark",
            intermediate={"survival_metadata_csv": survival_bag_workspace.metadata_csv},
            final={
                "best_model_path": Path(result.best_model_path),
                **training_artifact_outputs(result),
            },
        )

    assert Path(result.best_model_path).exists()
    assert Path(result.artifacts_dir, "val_td_auc_curve.png").exists()
    assert Path(result.artifacts_dir, "val_concordance_index.png").exists()
    assert Path(result.artifacts_dir, "val_kaplan_meier.png").exists()


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
    ) as metadata:
        _, result = fit_smoke_model(
            tmp_path / "discrete_survival",
            dataset_train=dataset,
            dataset_val=dataset,
            input_dim=survival_bag_workspace.input_dim,
            output_dim=num_bins,
            task="survival_discrete",
            loss_name="DiscreteTimeNLLLoss",
            epochs=DEFAULT_SMOKE_EPOCHS,
            lr=1e-3,
            dropout=0.1,
        )
        attach_smoke_outputs(
            metadata,
            step_name="hf_discrete_survival_benchmark",
            intermediate={"survival_metadata_csv": survival_bag_workspace.metadata_csv},
            final={
                "best_model_path": Path(result.best_model_path),
                **training_artifact_outputs(result),
            },
        )

    assert Path(result.best_model_path).exists()
    assert Path(result.artifacts_dir, "val_td_auc_curve.png").exists()
    assert Path(result.artifacts_dir, "val_concordance_index.png").exists()
    assert Path(result.artifacts_dir, "val_kaplan_meier.png").exists()


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
            epochs=DEFAULT_SMOKE_EPOCHS,
            lr=lr,
            dropout=dropout,
        )
        return result.best_score

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_binary_classification_optuna",
        metadata={"n_trials": 2},
    ) as metadata:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=2)
        best_trial_root = objective_root / f"trial_{study.best_trial.number}"
        best_artifacts_dir = best_trial_root / "project" / "training_artifacts"
        attach_smoke_outputs(
            metadata,
            step_name="hf_binary_classification_optuna",
            intermediate={"objective_root": objective_root},
            final={
                "best_trial_root": best_trial_root,
                "training_artifacts_dir": best_artifacts_dir,
                "val_metrics_json": best_artifacts_dir / "val_metrics.json",
                "val_curves_json": best_artifacts_dir / "val_curves.json",
                "val_confusion_matrix_png": best_artifacts_dir
                / "val_confusion_matrix.png",
                "val_roc_auc_curve_png": best_artifacts_dir / "val_roc_auc_curve.png",
                "val_pr_auc_curve_png": best_artifacts_dir / "val_pr_auc_curve.png",
                "val_calibration_curve_png": best_artifacts_dir
                / "val_calibration_curve.png",
            },
        )

    assert len(study.trials) == 2
    assert all(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    assert "lr" in study.best_params
    assert "dropout" in study.best_params
    assert Path(best_artifacts_dir, "val_confusion_matrix.png").exists()
    assert Path(best_artifacts_dir, "val_roc_auc_curve.png").exists()
    assert Path(best_artifacts_dir, "val_pr_auc_curve.png").exists()
    assert Path(best_artifacts_dir, "val_calibration_curve.png").exists()


@pytest.mark.smoke
def test_trained_mil_inference_heatmap_cli(
    extracted_bag_workspace: PreparedBagWorkspace,
    extracted_wsi_workspace: ExtractedWsiWorkspace,
    tmp_path: Path,
) -> None:
    """Attach a visual heatmap to an extracted slide artifact via the inference CLI."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchmil")
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
    ) as metadata:
        model, result = fit_smoke_model(
            tmp_path / "inference_train",
            dataset_train=dataset,
            dataset_val=dataset,
            input_dim=extracted_bag_workspace.input_dim,
            output_dim=2,
            task="classification",
            loss_name="CrossEntropyLoss",
            epochs=DEFAULT_SMOKE_EPOCHS,
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
        heatmap_png = tmp_path / f"{slide_id}_heatmap.png"
        heatmap_smoothed_png = tmp_path / f"{slide_id}_heatmap_smoothed.png"
        heatmap_top_tiles_png = tmp_path / f"{slide_id}_heatmap_top_tiles.png"
        checkpoint_path = tmp_path / "smoke_model.ckpt"
        np.save(scores_path, instance_scores)
        from pathbench.inference.model_package import save_packaged_model

        save_packaged_model(
            path=checkpoint_path,
            model=model,
            config=result.config,
            model_name="VarMIL",
            input_dim=extracted_bag_workspace.input_dim,
            output_dim=2,
            loss_name="CrossEntropyLoss",
        )

        exit_code = inference_main(
            [
                "--model_path",
                str(checkpoint_path),
                "--input",
                str(artifact_path),
                "--slide-path",
                str(extracted_wsi_workspace.slides_dir / f"{slide_id}.svs"),
                "--output",
                str(output_json),
                "--heatmap-backend",
                "torchmil",
                "--bag-id",
                extracted_wsi_workspace.bag_id,
                "--scores",
                str(scores_path),
                "--heatmap-name",
                "smoke_attention",
                "--heatmap-output",
                str(heatmap_json),
                "--heatmap-image-output",
                str(heatmap_png),
            ]
        )
        attach_smoke_outputs(
            metadata,
            step_name="hf_inference_heatmap_cli",
            intermediate={
                "training_checkpoint": Path(result.best_model_path),
                "scores_path": scores_path,
            },
            final={
                **training_artifact_outputs(result),
                "prediction_json": output_json,
                "heatmap_json": heatmap_json,
                "heatmap_png": heatmap_png,
                "heatmap_smoothed_png": heatmap_smoothed_png,
                "heatmap_top_tiles_png": heatmap_top_tiles_png,
                "artifact_path": artifact_path,
            },
        )

    assert exit_code == 0
    assert Path(result.best_model_path).exists()
    assert output_json.exists()
    assert heatmap_json.exists()
    assert heatmap_png.exists()
    assert heatmap_smoothed_png.exists()
    assert heatmap_top_tiles_png.exists()

    prediction_payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert prediction_payload["status"] == "ok"
    assert prediction_payload["heatmap"]["num_points"] == int(instance_scores.shape[0])
    assert prediction_payload["heatmap"]["image_output_path"] == str(heatmap_png)
    assert prediction_payload["heatmap"]["smoothed_image_output_path"] == str(
        heatmap_smoothed_png
    )
    assert prediction_payload["heatmap"]["top_tiles_output_path"] == str(
        heatmap_top_tiles_png
    )

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


@pytest.mark.smoke
def test_gtex_survival_mil_smoke(
    gtex_survival_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Run continuous-survival MIL on tile-level GTEx features with synthetic labels.

    Reuses the tile-level bags from ``extracted_bag_workspace`` (no extra
    feature extraction).  Survival labels are derived deterministically from
    GTEx metadata: calcification slides are treated as events, age bracket
    encodes observation time.
    """
    metadata_df = pd.read_csv(gtex_survival_workspace.metadata_csv)
    dataset = SurvivalBagDataset(
        metadata_df,
        feature_dir=gtex_survival_workspace.feature_dir,
        time_column="os_months",
        event_column="status",
        discrete_time=False,
    )

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_gtex_continuous_survival_benchmark",
        metadata={"num_bags": len(dataset)},
    ) as metadata:
        _, result = fit_smoke_model(
            tmp_path / "gtex_continuous_survival",
            dataset_train=dataset,
            dataset_val=dataset,
            input_dim=gtex_survival_workspace.input_dim,
            output_dim=1,
            task="survival",
            loss_name="CoxPHLoss",
            epochs=DEFAULT_SMOKE_EPOCHS,
            lr=1e-3,
            dropout=0.0,
        )
        attach_smoke_outputs(
            metadata,
            step_name="hf_gtex_continuous_survival_benchmark",
            intermediate={
                "survival_metadata_csv": gtex_survival_workspace.metadata_csv
            },
            final={**training_artifact_outputs(result)},
        )

    assert Path(result.best_model_path).exists()
    assert Path(result.artifacts_dir, "val_td_auc_curve.png").exists()
    assert Path(result.artifacts_dir, "val_concordance_index.png").exists()
    assert Path(result.artifacts_dir, "val_kaplan_meier.png").exists()


@pytest.mark.smoke
def test_gtex_survival_heatmap(
    gtex_survival_workspace: PreparedBagWorkspace,
    extracted_wsi_workspace: ExtractedWsiWorkspace,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Generate tile-level survival heatmaps on GTEx slides using reused features.

    Trains a VarMIL survival model on GTEx tile features, then writes per-tile
    attention heatmaps back into the slide H5 artifacts and exports PNG overlays
    with the slide thumbnail as background.
    """
    torch = pytest.importorskip("torch")
    from pathbench.adapters.torchmil.heatmap_explainer import (
        register_torchmil_heatmap_explainer,
    )
    from pathbench.core.io.h5 import heatmaps as heatmap_io
    from pathbench.core.io.h5.base import FileHandleH5
    from pathbench.inference.heatmaps import create_inference_heatmap

    register_torchmil_heatmap_explainer()
    monkeypatch.setattr(
        "pathbench.inference.heatmaps.require_torchmil",
        lambda feature: None,
    )
    monkeypatch.setattr(
        "pathbench.adapters.torchmil.heatmap_explainer.require_torchmil",
        lambda feature: None,
    )

    metadata_df = pd.read_csv(gtex_survival_workspace.metadata_csv)
    dataset = SurvivalBagDataset(
        metadata_df,
        feature_dir=gtex_survival_workspace.feature_dir,
        time_column="os_months",
        event_column="status",
        discrete_time=False,
    )
    model, _result = fit_smoke_model(
        tmp_path / "gtex_survival_heatmap_train",
        dataset_train=dataset,
        dataset_val=dataset,
        input_dim=gtex_survival_workspace.input_dim,
        output_dim=1,
        task="survival",
        loss_name="CoxPHLoss",
        epochs=DEFAULT_SMOKE_EPOCHS,
        lr=1e-3,
        dropout=0.0,
    )

    slide_id = gtex_survival_workspace.slide_ids[0]
    artifact_path = extracted_wsi_workspace.artifact_paths[slide_id]
    bag_path = gtex_survival_workspace.feature_dir / f"{slide_id}.pt"
    bag_tensor = torch.load(bag_path, weights_only=True).unsqueeze(0)
    num_tiles = int(bag_tensor.shape[1])

    attention_scores = (
        model.instance_scores(bag_tensor)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    if attention_scores.ndim > 1:
        attention_scores = attention_scores.squeeze(-1)

    scores_path = tmp_path / f"{slide_id}_survival_scores.npy"
    heatmap_png = tmp_path / f"{slide_id}_survival_heatmap.png"
    np.save(scores_path, attention_scores)

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_gtex_survival_heatmap",
        metadata={"slide_id": slide_id, "num_tiles": num_tiles},
    ) as sm:
        create_inference_heatmap(
            artifact_path=artifact_path,
            bag_id=extracted_wsi_workspace.bag_id,
            scores_path=scores_path,
            heatmap_backend="torchmil",
            heatmap_name="smoke_gtex_survival_attention",
            image_output_path=heatmap_png,
            slide_path=extracted_wsi_workspace.slides_dir / f"{slide_id}.svs",
        )
        attach_smoke_outputs(
            sm,
            step_name="hf_gtex_survival_heatmap",
            intermediate={"artifact_path": artifact_path},
            final={"heatmap_png": heatmap_png},
        )

    assert heatmap_png.exists()
    with FileHandleH5(artifact_path, mode="r") as slide_artifact:
        heatmap = heatmap_io.read_prediction_heatmap(
            slide_artifact,
            extracted_wsi_workspace.bag_id,
            "smoke_gtex_survival_attention",
        )
    assert heatmap["coords"].shape[0] == num_tiles
    assert heatmap["scores"].shape == (num_tiles,)
    assert np.isfinite(heatmap["scores"]).all()
    assert float(heatmap["scores"].min()) >= 0.0
    assert float(heatmap["scores"].max()) <= 1.0
