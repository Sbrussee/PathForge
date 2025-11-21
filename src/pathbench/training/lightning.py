# src/pathbench/training/lightning.py
from __future__ import annotations

from typing import Any, Iterable, Dict, Optional, List, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor, 
    Callback
)
import torch
from torch.utils.data import DataLoader, Dataset

from pathbench.core.models.mil_base import MILModelBase
from pathbench.training.base import TrainerBase
from pathbench.core.losses.base import Loss
from pathbench.config.config import Config
from pathbench.utils.registries import TRAINERS

class LightningModuleAdapter(pl.LightningModule):
    """
    Adapter: Wraps a PathBench MILModelBase into a PL LightningModule.
    Handles optimization logic, logging, and scheduler configuration via Config.
    """
    def __init__(self, model: MILModelBase, loss_fn: Loss, config: Config):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        
        # Save hyperparameters for checkpointing reproducibility
        self.save_hyperparameters(ignore=['model', 'loss_fn'])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        bag, target = batch
        logits = self.model(bag)
        loss = self.loss_fn(logits, target)
        
        # Log with batch_size=1 assumption for MIL, or actual batch size
        batch_size = bag.shape[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        bag, target = batch
        logits = self.model(bag)
        loss = self.loss_fn(logits, target)
        
        batch_size = bag.shape[0]
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        # If classification, could add accuracy here (or use TorchMetrics)
        # preds = torch.argmax(logits, dim=1)
        # acc = (preds == target).float().mean()
        # self.log("val_acc", acc, on_epoch=True, batch_size=batch_size)
        
        return loss
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        bag, _ = batch
        return self.model(bag)

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
def __init__(self, config: Config, extra_callbacks: List[Callback] | None = None):
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

        self.callbacks = [
            self.checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(
                monitor=config.mil.best_epoch_based_on,
                patience=config.mil.patience,
                mode="min" if "loss" in config.mil.best_epoch_based_on else "max",
                verbose=True
            )
        ]
        self.callbacks.extend(self.extra_callbacks)

        self.trainer = pl.Trainer(
            max_epochs=config.mil.epochs,
            accumulate_grad_batches=config.mil.accumulate_grad_batches,
            gradient_clip_val=config.mil.gradient_clip_val,
            callbacks=self.callbacks,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=True,
            enable_checkpointing=True,
            log_every_n_steps=5
        )
        
        # Merge extra callbacks (e.g. LearningRateFinder if passed)
        self.callbacks.extend(self.extra_callbacks)

        self.trainer = pl.Trainer(
            max_epochs=config.mil.epochs,
            accumulate_grad_batches=config.mil.accumulate_grad_batches,
            gradient_clip_val=config.mil.gradient_clip_val,
            callbacks=self.callbacks,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=True, # Defaults to TensorBoard; can be configured to WandB/CSV
            enable_checkpointing=True,
            log_every_n_steps=5 # Useful for small datasets
        )

    def fit(
        self,
        model: MILModelBase,
        dataset_train: Dataset,
        dataset_val: Dataset,
        loss_func: Loss,
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
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            dataset_val, 
            batch_size=self.config.mil.batch_size, 
            shuffle=False, 
            num_workers=self.config.experiment.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Wrap model using the Adapter (Factory logic could go here if complex)
        from pathbench.training.lightning import LightningModuleAdapter
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
        """
        Run inference.
        """
        loader = DataLoader(
            dataset, 
            batch_size=self.config.mil.batch_size, 
            shuffle=False, 
            num_workers=self.config.experiment.num_workers
        )
        
        # We reuse the LightningModuleAdapter, assuming model is already loaded
        # (If model is raw MILModelBase, we wrap it lightly or use trainer.predict with raw model if supported?)
        # PL requires a LightningModule for trainer.predict
        # If 'model' here is the raw PyTorch model (trained), we wrap it again.
        
        # Hack: If we don't have the original config for prediction, we use self.config
        # Assuming loss_fn doesn't matter for prediction
        dummy_loss = self.config.benchmark_parameters.loss[0] # dummy
        from pathbench.utils.registries import LOSSES
        loss_fn = LOSSES.get("CrossEntropyLoss")() # generic
        
        pl_module = LightningModuleAdapter(model, loss_fn, self.config)

        predictions = self.trainer.predict(pl_module, dataloaders=loader)

        # Concatenate all predictions
        return torch.cat(predictions, dim=0)

