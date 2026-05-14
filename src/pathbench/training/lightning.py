# src/pathbench/training/lightning.py
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback,
)
import torch
from torch.utils.data import DataLoader, Dataset

from pathbench.adapters.torchmil.collate import torchmil_or_pathbench_collate
from pathbench.adapters.torchmil.task_output import normalize_torchmil_output
from pathbench.core.datasets.bag_schema import assert_bag_schema
from pathbench.core.models.mil_base import MILModelBase
from pathbench.training.base import TrainerBase
from pathbench.training.metrics import (
    compute_task_metrics,
    save_task_evaluation_artifacts,
)
from pathbench.core.losses.base import BaseLoss
from pathbench.config.config import Config
from pathbench.utils.registries import TRAINERS


class LightningModuleAdapter(pl.LightningModule):
    """
    Adapter: Wraps a PathBench MILModelBase into a PL LightningModule.
    Handles optimization logic, logging, and scheduler configuration via Config.
    Pass loss_fn=None only for inference (predict_step does not use it).
    """

    def __init__(self, model: MILModelBase, loss_fn: BaseLoss | None, config: Config):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        experiment_cfg = getattr(config, "experiment", None)
        task_name = getattr(experiment_cfg, "task", None)
        self.task_name = str(task_name or "classification")
        self._val_predictions: list[torch.Tensor] = []
        self._val_targets: list[Any] = []

        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if self.loss_fn is None:
            raise RuntimeError("loss_fn is required for training but was not provided.")
        bag, target, model_kwargs = self._unpack_batch(batch)
        output = self.model.forward_bag(bag, label=target, **model_kwargs)
        if isinstance(output, dict):
            loss = output.get("loss")
            logits = self._normalize_predictions(output)
            if loss is not None:
                batch_size = bag.shape[0]
                self.log(
                    "train_loss",
                    loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch_size,
                )
                return loss
        else:
            logits = self._normalize_predictions(output)
        loss = self.loss_fn(logits, target)
        batch_size = bag.shape[0]
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if self.loss_fn is None:
            raise RuntimeError(
                "loss_fn is required for validation but was not provided."
            )
        bag, target, model_kwargs = self._unpack_batch(batch)
        output = self.model.forward_bag(bag, label=target, **model_kwargs)
        logits = self._normalize_predictions(output)
        loss = self.loss_fn(logits, target)
        batch_size = bag.shape[0]
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self._val_predictions.append(logits.detach().cpu())
        self._val_targets.append(self._detach_target(target))
        return loss

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        bag, _, model_kwargs = self._unpack_batch(batch)
        output = self.model.forward_bag(bag, **model_kwargs)
        return self._normalize_predictions(output)

    def _unpack_batch(
        self, batch: Any
    ) -> tuple[torch.Tensor, Any, dict[str, torch.Tensor]]:
        """
        Normalize legacy tuple batches and canonical dict batches.

        Args:
            batch: Either ``(bag, target)`` with ``bag`` shaped ``[B, N, D]`` or a
                canonical bag dictionary containing ``X`` shaped ``[B, N, D]``,
                ``Y``, optional ``mask`` shaped ``[B, N]``, optional ``coords``
                shaped ``[B, N, 2]``, and optional ``adj`` shaped ``[B, N, N]``.

        Returns:
            Tuple of feature tensor, task target, and model keyword tensors.
        """

        if isinstance(batch, dict):
            assert_bag_schema(batch, batched=True)
            kwargs = {
                key: batch[key]
                for key in ("mask", "coords", "adj")
                if key in batch and batch[key] is not None
            }
            return batch["X"], batch["Y"], kwargs
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            bag, target = batch
            if bag.ndim == 2:
                bag = bag.unsqueeze(0)
            return bag, target, {}
        raise TypeError(
            "Lightning MIL batches must be (bag, target) tuples or canonical bag dictionaries."
        )

    def on_validation_epoch_start(self) -> None:
        self._val_predictions = []
        self._val_targets = []

    def on_validation_epoch_end(self) -> None:
        if not self._val_predictions:
            return
        predictions = torch.cat(self._val_predictions, dim=0)
        target = self._stack_targets(self._val_targets)
        metrics = compute_task_metrics(
            predictions,
            target,
            task=self.task_name,
            selected_metrics=self.config.metrics.metrics_for_task(self.task_name),
        )
        for name, value in metrics.items():
            if value != value:
                continue
            metric_tensor = torch.tensor(float(value), device=self.device)
            self.log(
                name,
                metric_tensor,
                on_epoch=True,
                prog_bar=name in {"balanced_accuracy", "c_index"},
            )
            self.log(f"val_{name}", metric_tensor, on_epoch=True, prog_bar=False)

    def _normalize_predictions(self, output: Any) -> torch.Tensor:
        if (
            isinstance(output, dict)
            and "logits" in output
            and isinstance(output["logits"], torch.Tensor)
        ):
            tensor = output["logits"]
            if self.task_name in {"survival", "survival_discrete"}:
                return normalize_torchmil_output(output, task=self.task_name)
            return tensor.float()
        if self.task_name in {
            "classification",
            "survival",
            "survival_discrete",
            "regression",
        }:
            return normalize_torchmil_output(output, task=self.task_name)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "Model output must be a tensor or tensor-containing mapping."
            )
        return output.float()

    def _detach_target(self, target: Any) -> Any:
        if isinstance(target, dict):
            return {
                key: value.detach().cpu()
                for key, value in target.items()
                if isinstance(value, torch.Tensor)
            }
        if isinstance(target, torch.Tensor):
            return target.detach().cpu()
        return target

    def _stack_targets(self, targets: list[Any]) -> Any:
        if not targets:
            raise ValueError("No validation targets were collected.")
        first_target = targets[0]
        if isinstance(first_target, dict):
            keys = sorted(first_target)
            return {
                key: torch.cat([target[key].reshape(-1) for target in targets], dim=0)
                for key in keys
            }
        return torch.cat([target.reshape(-1) for target in targets], dim=0)

    def configure_optimizers(self):
        mil_cfg = self.config.mil
        optim_cls = getattr(torch.optim, mil_cfg.optimizer, None)
        if optim_cls is None:
            raise ValueError(
                f"Optimizer '{mil_cfg.optimizer}' not found in torch.optim."
            )
        optimizer = optim_cls(
            self.model.parameters(),
            lr=mil_cfg.lr,
            weight_decay=mil_cfg.weight_decay,
        )
        if mil_cfg.scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=mil_cfg.patience // 2,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": mil_cfg.scheduler_monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif mil_cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=mil_cfg.epochs, eta_min=0.0
            )
            return [optimizer], [scheduler]

        return optimizer


@TRAINERS.register("lightning")
class LightningTrainer(TrainerBase):
    """
    Trainer implementation using PyTorch Lightning.

    Config Usage:
    - mil.epochs -> max_epochs
    - mil.accumulate_grad_batches
    - mil.gradient_clip_val
    - experiment.num_workers
    """

    def __init__(self, config: Config, extra_callbacks: List[Callback] | None = None):
        self.config = config
        self.run_root = Path(config.experiment.project_root or ".").resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.run_root / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_artifacts_dir = self.run_root / "training_artifacts"
        self.metrics_artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.checkpoints_dir),
            monitor=config.mil.best_epoch_based_on,
            mode="min" if "loss" in config.mil.best_epoch_based_on else "max",
            save_top_k=1,
            filename="{epoch}-{val_loss:.2f}",
            verbose=True,
        )

        self.callbacks: List[Callback] = [
            self.checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(
                monitor=config.mil.best_epoch_based_on,
                patience=config.mil.patience,
                mode="min" if "loss" in config.mil.best_epoch_based_on else "max",
                verbose=True,
            ),
            *(extra_callbacks or []),
        ]

        self.trainer = pl.Trainer(
            max_epochs=config.mil.epochs,
            accumulate_grad_batches=config.mil.accumulate_grad_batches,
            gradient_clip_val=config.mil.gradient_clip_val,
            callbacks=self.callbacks,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            default_root_dir=str(self.run_root),
            logger=True,
            enable_checkpointing=True,
            log_every_n_steps=5,
        )

    def _collate_fn(self):
        if (
            self.config.mil.backend == "torchmil"
            and self.config.mil.use_torchmil_collate
        ):
            return lambda batch: torchmil_or_pathbench_collate(batch, use_torchmil=True)
        if self.config.mil.batch_size > 1:
            return lambda batch: torchmil_or_pathbench_collate(
                batch, use_torchmil=False
            )
        return None

    def fit(
        self,
        model: MILModelBase,
        dataset_train: Dataset,
        dataset_val: Dataset,
        loss_func: BaseLoss,
    ) -> Tuple[str, float]:
        """
        Train the model.
        Returns:
            (best_model_path, best_model_score)
        """
        train_loader = DataLoader(
            dataset_train,
            batch_size=self.config.mil.batch_size,
            shuffle=True,
            num_workers=self.config.experiment.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=self._collate_fn(),
        )
        val_loader = DataLoader(
            dataset_val,
            batch_size=self.config.mil.batch_size,
            shuffle=False,
            num_workers=self.config.experiment.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=self._collate_fn(),
        )

        pl_module = LightningModuleAdapter(model, loss_func, self.config)

        self.trainer.fit(
            pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        # Retrieve best path and score from the checkpoint callback
        best_path = self.checkpoint_callback.best_model_path
        best_score = self.checkpoint_callback.best_model_score
        best_module = pl_module
        if best_path:
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            best_module.load_state_dict(checkpoint["state_dict"])
        self._save_validation_artifacts(best_module, val_loader)

        # Handle case where score might be None (e.g., if training failed immediately)
        if best_score is None:
            best_score = (
                float("inf") if "loss" in self.config.mil.best_epoch_based_on else 0.0
            )

        # Convert tensor to float if necessary
        if isinstance(best_score, torch.Tensor):
            best_score = best_score.item()

        return best_path, best_score

    def _save_validation_artifacts(
        self,
        module: LightningModuleAdapter,
        val_loader: DataLoader,
    ) -> None:
        was_training = module.training
        module.eval()
        device = module.device
        predictions: list[torch.Tensor] = []
        targets: list[Any] = []
        with torch.no_grad():
            for batch in val_loader:
                bag, target, model_kwargs = module._unpack_batch(batch)
                bag = bag.to(device)
                moved_kwargs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in model_kwargs.items()
                }
                moved_target = self._move_target_to_device(target, device)
                output = module.model.forward_bag(
                    bag,
                    label=moved_target,
                    **moved_kwargs,
                )
                predictions.append(module._normalize_predictions(output).detach().cpu())
                targets.append(module._detach_target(moved_target))
        stacked_predictions = torch.cat(predictions, dim=0)
        stacked_target = module._stack_targets(targets)
        save_task_evaluation_artifacts(
            stacked_predictions,
            stacked_target,
            task=module.task_name,
            output_dir=self.metrics_artifacts_dir,
            prefix="val",
            selected_metrics=self.config.metrics.metrics_for_task(module.task_name),
        )
        if was_training:
            module.train()

    def _move_target_to_device(self, target: Any, device: torch.device) -> Any:
        if isinstance(target, dict):
            return {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            }
        if isinstance(target, torch.Tensor):
            return target.to(device)
        return target

    def predict(
        self,
        model: MILModelBase,
        dataset: Dataset,
    ) -> torch.Tensor:
        """Run inference. loss_fn is not used during prediction."""
        loader = DataLoader(
            dataset,
            batch_size=self.config.mil.batch_size,
            shuffle=False,
            num_workers=self.config.experiment.num_workers,
            collate_fn=self._collate_fn(),
        )
        pl_module = LightningModuleAdapter(model, loss_fn=None, config=self.config)
        predictions = self.trainer.predict(pl_module, dataloaders=loader)
        return torch.cat(predictions, dim=0)
