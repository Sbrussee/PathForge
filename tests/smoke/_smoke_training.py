"""Reusable training primitives for the smoke-test suite.

The production benchmarking and optimization policies are still evolving, so
the smoke suite exercises the stable registry, dataset, trainer, and inference
contracts directly with tiny deterministic models and losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SmokeTrainingResult:
    """Summary of one tiny MIL training run.

    Attributes:
        best_model_path: Best checkpoint returned by ``LightningTrainer.fit``.
        best_score: Scalar validation score selected by the checkpoint callback.
        output_dim: Number of output channels produced by the trained model.
        task_name: PathBench task name used for the run.
    """

    best_model_path: str
    best_score: float
    output_dim: int
    task_name: str


class SurvivalBagDataset:
    """Small bag dataset for survival smoke tests.

    Args:
        metadata_df: DataFrame containing ``slide_id``, one time column, and one
            event column.
        feature_dir: Directory containing one ``{slide_id}.pt`` tensor per row.
        time_column: Survival time column name.
        event_column: Event indicator column name.
        discrete_time: When true, return integer time bins for discrete
            survival; otherwise return floating survival times.

    Returns from ``__getitem__``:
        Tuple of bag tensor shaped ``[num_instances, feature_dim]`` and a target
        dict with ``time`` and ``event`` tensors shaped ``[]``.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        *,
        feature_dir: Path,
        time_column: str,
        event_column: str,
        discrete_time: bool,
    ) -> None:
        import torch

        self._torch = torch
        self.metadata_df = metadata_df.reset_index(drop=True).copy()
        self.feature_dir = feature_dir
        self.time_column = time_column
        self.event_column = event_column
        self.discrete_time = discrete_time

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        row = self.metadata_df.iloc[index]
        bag = self._torch.load(self.feature_dir / f"{row['slide_id']}.pt")
        time_value = row[self.time_column]
        event_value = row[self.event_column]
        target = {
            "time": self._torch.tensor(
                int(time_value) if self.discrete_time else float(time_value),
                dtype=self._torch.long if self.discrete_time else self._torch.float32,
            ),
            "event": self._torch.tensor(float(event_value), dtype=self._torch.float32),
        }
        return bag.float(), target


def register_smoke_components() -> None:
    """Register tiny smoke-test models, losses, and explainers once."""
    import torch
    import torch.nn as nn

    import pathbench.core.losses.classification  # noqa: F401

    from pathbench.core.losses.base import (
        SurvivalContinuousLoss,
        SurvivalDiscreteLoss,
    )
    from pathbench.core.models.mil_base import MILModelBase
    from pathbench.utils.registries import EXPLAINERS, LOSSES, MODELS

    if not MODELS.is_available("SmokePoolingMIL"):

        @MODELS.register("SmokePoolingMIL")
        class SmokePoolingMIL(MILModelBase):
            """A tiny mean-pooling MIL model for smoke tests.

            Args:
                input_dim: Feature dimension ``D`` for one instance.
                output_dim: Number of output channels.
                hidden_dim: Hidden representation width.
                dropout: Dropout probability in ``[0, 1]``.
            """

            def __init__(
                self,
                input_dim: int,
                output_dim: int,
                hidden_dim: int = 32,
                dropout: float = 0.0,
            ) -> None:
                super().__init__()
                self.instance_encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.classifier = nn.Linear(hidden_dim, output_dim)

            @property
            def bag_size(self) -> int | None:
                return None

            def encode_instances(self, bag: torch.Tensor) -> torch.Tensor:
                if bag.ndim != 3:
                    raise ValueError(
                        f"Expected bag shaped [B, N, D], got {tuple(bag.shape)}."
                    )
                return self.instance_encoder(bag.float())

            def instance_scores(self, bag: torch.Tensor) -> torch.Tensor:
                encoded = self.encode_instances(bag)
                logits = self.classifier(encoded)
                if logits.shape[-1] == 1:
                    return logits.squeeze(-1)
                return logits[..., -1]

            def forward_bag(
                self,
                bag: torch.Tensor,
                mask: torch.Tensor | None = None,
                coords: torch.Tensor | None = None,
                label: torch.Tensor | None = None,
                loss_fn: nn.Module | None = None,
            ) -> torch.Tensor:
                encoded = self.encode_instances(bag)
                if mask is None:
                    pooled = encoded.mean(dim=1)
                else:
                    weights = mask.float().unsqueeze(-1)
                    pooled = (encoded * weights).sum(dim=1) / torch.clamp(
                        weights.sum(dim=1), min=1.0
                    )
                return self.classifier(pooled)

    if not LOSSES.is_available("SmokeSurvivalMSELoss"):

        @LOSSES.register("SmokeSurvivalMSELoss")
        class SmokeSurvivalMSELoss(SurvivalContinuousLoss):
            """Continuous survival smoke loss.

            This objective is intentionally simple and deterministic: it regresses
            normalized survival time and lightly penalizes event disagreement.
            """

            def __init__(self) -> None:
                super().__init__()
                self.mse = nn.MSELoss()
                self.bce = nn.BCEWithLogitsLoss()

            def calculate_loss(
                self,
                preds: torch.Tensor,
                time: torch.Tensor,
                event: torch.Tensor,
                **_: Any,
            ) -> torch.Tensor:
                normalized_time = time / torch.clamp(time.max(), min=1.0)
                return self.mse(preds.reshape(-1), normalized_time) + 0.1 * self.bce(
                    preds.reshape(-1),
                    event,
                )

    if not LOSSES.is_available("SmokeDiscreteSurvivalLoss"):

        @LOSSES.register("SmokeDiscreteSurvivalLoss")
        class SmokeDiscreteSurvivalLoss(SurvivalDiscreteLoss):
            """Discrete survival smoke loss based on time-bin cross entropy."""

            def __init__(self) -> None:
                super().__init__()
                self.ce = nn.CrossEntropyLoss()

            def calculate_loss(
                self,
                preds: torch.Tensor,
                time: torch.Tensor,
                event: torch.Tensor,
                **_: Any,
            ) -> torch.Tensor:
                _ = event
                return self.ce(preds, time)

    if not EXPLAINERS.is_available("smoke_heatmap"):

        @EXPLAINERS.register("smoke_heatmap")
        class SmokeHeatmapExplainer:
            """Normalize instance scores into an inference heatmap payload."""

            def initialize(self, config: dict[str, Any]) -> None:
                self.config = config

            def explain(self, payload: dict[str, torch.Tensor]) -> Any:
                scores = payload["instance_scores"].float()
                coords = payload["coords"].float()
                if "mask" in payload:
                    mask = payload["mask"].bool()
                    scores = scores[mask]
                    coords = coords[mask]
                score_range = torch.clamp(scores.max() - scores.min(), min=1e-12)
                normalized = (scores - scores.min()) / score_range

                @dataclass(frozen=True)
                class _Heatmap:
                    coords: torch.Tensor
                    scores: torch.Tensor

                return _Heatmap(coords=coords, scores=normalized)


def make_training_config(
    root_dir: Path,
    *,
    task: str,
    epochs: int,
    lr: float,
    dropout: float,
    batch_size: int = 1,
) -> Any:
    """Create a minimal writable PathBench config for trainer smoke tests."""
    from pathbench.config.config import Config

    annotations_csv = root_dir / "annotations.csv"
    annotations_csv.write_text("dataset,slide,patient,category\n", encoding="utf-8")
    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": f"smoke_{task}",
                "annotation_file": str(annotations_csv),
                "project_root": str((root_dir / "project").resolve()),
                "mode": "feature_extraction",
                "task": task,
                "num_workers": 0,
            },
            "mil": {
                "backend": "native",
                "epochs": epochs,
                "lr": lr,
                "dropout_p": dropout,
                "batch_size": batch_size,
                "patience": 1,
                "best_epoch_based_on": "val_loss",
            },
            "metrics": {
                "classification_backend": "native",
                "survival_continuous_backend": "native",
            },
        }
    )
    return cfg


def fit_smoke_model(
    root_dir: Path,
    *,
    dataset_train: Any,
    dataset_val: Any,
    input_dim: int,
    output_dim: int,
    task: str,
    loss_name: str,
    epochs: int,
    lr: float,
    dropout: float,
) -> tuple[Any, SmokeTrainingResult]:
    """Train one tiny MIL model through the production Lightning trainer."""
    import torch

    from pathbench.training.lightning import LightningTrainer
    from pathbench.utils.registries import LOSSES, MODELS

    register_smoke_components()
    cfg = make_training_config(
        root_dir,
        task=task,
        epochs=epochs,
        lr=lr,
        dropout=dropout,
        batch_size=1,
    )
    torch.manual_seed(7)
    model = MODELS.get("SmokePoolingMIL")(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=dropout,
    )
    loss_fn = LOSSES.get(loss_name)()
    trainer = LightningTrainer(cfg)
    best_model_path, best_score = trainer.fit(
        model, dataset_train, dataset_val, loss_fn
    )
    return model, SmokeTrainingResult(
        best_model_path=best_model_path,
        best_score=float(best_score),
        output_dim=output_dim,
        task_name=task,
    )
