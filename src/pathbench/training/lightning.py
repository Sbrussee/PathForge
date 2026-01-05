# src/pathbench/training/lightning.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor, 
    Callback
)
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader

from pathbench.config.config import Config
from pathbench.core.datasets.base import BagDatasetBase
from pathbench.core.models.mil_base import MILModelBase
from pathbench.training.base import TrainerBase, TrainerOutput
from pathbench.training.utils import mil_collate, resolve_loss_and_logits, unpack_mil_batch
from pathbench.utils.registries import TRAINERS

class LightningModuleAdapter(pl.LightningModule):
    """
    Adapter: Wraps a PathBench MILModelBase into a PL LightningModule.
    Handles optimization logic, logging, and scheduler configuration via Config.
    """
    def __init__(self, model: MILModelBase, loss_fn: nn.Module, config: Config):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        
        # Save hyperparameters for checkpointing reproducibility
        self.save_hyperparameters(ignore=['model', 'loss_fn'])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        bag, target, mask = unpack_mil_batch(batch)
        output = self.model(bag, mask=mask)
        loss, _logits = resolve_loss_and_logits(output, target, self.loss_fn)

        # Log with batch_size=1 assumption for MIL, or actual batch size
        batch_size = bag.shape[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        bag, target, mask = unpack_mil_batch(batch)
        output = self.model(bag, mask=mask)
        loss, _logits = resolve_loss_and_logits(output, target, self.loss_fn)
        
        batch_size = bag.shape[0]
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        bag, _, mask = unpack_mil_batch(batch)
        output = self.model(bag, mask=mask)
        if isinstance(output, dict):
            return output.get("logits", output.get("preds", output.get("output")))
        return output

    def configure_optimizers(self):
        """
        Configures Optimizer and Scheduler based on `config.mil`.
        """
        mil_cfg = self.config.mil
        
        # 1. Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=mil_cfg.lr, 
            weight_decay=mil_cfg.weight_decay
        )
        
        # 2. Scheduler
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

    def __init__(self, config: Config, extra_callbacks: Optional[List[Callback]] = None):
        self.config = config
        self.extra_callbacks = extra_callbacks or []
        
        # We need to access the checkpoint callback later to get the score
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
        ]
        self.callbacks.extend(self.extra_callbacks)

        precision = 16 if config.experiment.mixed_precision else 32
        self.trainer = pl.Trainer(
            max_epochs=config.mil.epochs,
            accumulate_grad_batches=config.mil.accumulate_grad_batches,
            gradient_clip_val=config.mil.gradient_clip_val,
            callbacks=self.callbacks,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=True, # Defaults to TensorBoard; can be configured to WandB/CSV
            enable_checkpointing=True,
            log_every_n_steps=5, # Useful for small datasets.
            precision=precision,
        )

    def fit(
        self,
        model: MILModelBase,
        dataset_train: BagDatasetBase,
        dataset_val: Optional[BagDatasetBase],
        loss_fn: nn.Module,
    ) -> TrainerOutput:
        """
        Train the model.
        Returns:
            TrainerOutput(best_model_path, best_model_score)
        """
        train_loader = DataLoader(
            dataset_train, 
            batch_size=self.config.mil.batch_size, 
            shuffle=True, 
            num_workers=self.config.experiment.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=mil_collate
        )
        val_loader = None
        if dataset_val is not None:
            val_loader = DataLoader(
                dataset_val,
                batch_size=self.config.mil.batch_size,
                shuffle=False,
                num_workers=self.config.experiment.num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
                collate_fn=mil_collate,
            )

        # Wrap model using the Adapter (Factory logic could go here if complex)
        from pathbench.training.lightning import LightningModuleAdapter
        pl_module = LightningModuleAdapter(model, loss_fn, self.config)

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

    
        return TrainerOutput(best_model_path=best_path, best_score=best_score)


    def predict(
        self,
        model: MILModelBase,
        dataset: BagDatasetBase,
    ) -> torch.Tensor:
        """
        Run inference.
        """
        loader = DataLoader(
            dataset, 
            batch_size=self.config.mil.batch_size, 
            shuffle=False, 
            num_workers=self.config.experiment.num_workers,
            collate_fn=mil_collate,
        )

        loss_fn = self._default_loss()
        pl_module = LightningModuleAdapter(model, loss_fn, self.config)

        predictions = self.trainer.predict(pl_module, dataloaders=loader)

        # Concatenate all predictions
        return torch.cat(predictions, dim=0)

    def _default_loss(self) -> nn.Module:
        task = self.config.experiment.task or "classification"
        if task == "regression":
            return torch.nn.MSELoss()
        if task in {"survival", "survival_discrete"}:
            return torch.nn.BCEWithLogitsLoss()
        return torch.nn.CrossEntropyLoss()