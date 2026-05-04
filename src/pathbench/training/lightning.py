# src/pathbench/training/lightning.py
from __future__ import annotations

from typing import Any, List, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor, 
    Callback
)
import torch
from torch.utils.data import DataLoader, Dataset

from pathbench.adapters.torchmil.collate import torchmil_or_pathbench_collate
from pathbench.core.datasets.bag_schema import assert_bag_schema
from pathbench.core.models.mil_base import MILModelBase
from pathbench.training.base import TrainerBase
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

        self.save_hyperparameters(ignore=['model', 'loss_fn'])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if self.loss_fn is None:
            raise RuntimeError("loss_fn is required for training but was not provided.")
        bag, target, model_kwargs = self._unpack_batch(batch)
        logits = self.model.forward_bag(bag, label=target, **model_kwargs)
        if isinstance(logits, dict):
            loss = logits.get("loss")
            logits = logits.get("logits")
            if loss is not None:
                batch_size = bag.shape[0]
                self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
                return loss
        loss = self.loss_fn(logits, target)
        batch_size = bag.shape[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if self.loss_fn is None:
            raise RuntimeError("loss_fn is required for validation but was not provided.")
        bag, target, model_kwargs = self._unpack_batch(batch)
        logits = self.model.forward_bag(bag, label=target, **model_kwargs)
        if isinstance(logits, dict):
            logits = logits.get("logits")
        loss = self.loss_fn(logits, target)
        batch_size = bag.shape[0]
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss
        
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        bag, _, model_kwargs = self._unpack_batch(batch)
        output = self.model.forward_bag(bag, **model_kwargs)
        if isinstance(output, dict):
            output = output.get("logits")
        return output

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, Any, dict[str, torch.Tensor]]:
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
            kwargs = {key: batch[key] for key in ("mask", "coords", "adj") if key in batch and batch[key] is not None}
            return batch["X"], batch["Y"], kwargs
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            bag, target = batch
            if bag.ndim == 2:
                bag = bag.unsqueeze(0)
            return bag, target, {}
        raise TypeError("Lightning MIL batches must be (bag, target) tuples or canonical bag dictionaries.")

    def configure_optimizers(self):
        mil_cfg = self.config.mil
        optim_cls = getattr(torch.optim, mil_cfg.optimizer, None)
        if optim_cls is None:
            raise ValueError(f"Optimizer '{mil_cfg.optimizer}' not found in torch.optim.")
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
                verbose=True
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
                optimizer,
                T_max=mil_cfg.epochs,
                eta_min=0.0
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

        self.checkpoint_callback = ModelCheckpoint(
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
            logger=True,
            enable_checkpointing=True,
            log_every_n_steps=5,
        )

    def _collate_fn(self):
        if self.config.mil.backend == "torchmil" and self.config.mil.use_torchmil_collate:
            return lambda batch: torchmil_or_pathbench_collate(batch, use_torchmil=True)
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

        self.trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Retrieve best path and score from the checkpoint callback
        best_path = self.checkpoint_callback.best_model_path
        best_score = self.checkpoint_callback.best_model_score
        
        # Handle case where score might be None (e.g., if training failed immediately)
        if best_score is None:
            best_score = float('inf') if "loss" in self.config.mil.best_epoch_based_on else 0.0

        # Convert tensor to float if necessary
        if isinstance(best_score, torch.Tensor):
            best_score = best_score.item()

        return best_path, best_score

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
