from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from pathbench.config.config import Config
from pathbench.core.datasets.base import BagDatasetBase
from pathbench.core.models.mil_base import MILModelBase
from pathbench.training.base import TrainerBase, TrainerOutput
from pathbench.training.utils import mil_collate, resolve_loss_and_logits, unpack_mil_batch
from pathbench.utils.registries import TRAINERS


@TRAINERS.register("simple")
@dataclass
class SimpleTrainer(TrainerBase):
    """
    Lightweight trainer that runs a basic PyTorch training loop.

    Intended for smoke/unit tests and CPU-only runs.
    """

    config: Config

    def fit(
        self,
        model: MILModelBase,
        dataset_train: BagDatasetBase,
        dataset_val: Optional[BagDatasetBase],
        loss_fn: nn.Module,
    ) -> TrainerOutput:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        loss_fn = loss_fn.to(device)

        train_loader = DataLoader(
            dataset_train,
            batch_size=self.config.mil.batch_size,
            shuffle=True,
            num_workers=self.config.experiment.num_workers,
            collate_fn=mil_collate,
        )

        val_loader = None
        if dataset_val is not None:
            val_loader = DataLoader(
                dataset_val,
                batch_size=self.config.mil.batch_size,
                shuffle=False,
                num_workers=self.config.experiment.num_workers,
                collate_fn=mil_collate,
            )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.mil.lr,
            weight_decay=self.config.mil.weight_decay,
        )

        best_score = float("inf")
        best_path = self._checkpoint_path()

        model.train()
        for _epoch in range(self.config.mil.epochs):
            for batch in train_loader:
                bag, target, mask = unpack_mil_batch(batch)
                bag = bag.to(device)
                target = self._move_target(target, device)
                mask = mask.to(device) if mask is not None else None

                optimizer.zero_grad()
                output = model(bag, mask=mask)
                loss, _logits = resolve_loss_and_logits(output, target, loss_fn)
                loss.backward()
                optimizer.step()

            val_loss = self._evaluate_loss(model, loss_fn, val_loader, device) if val_loader else loss.item()
            if val_loss < best_score:
                best_score = val_loss
                torch.save(model.state_dict(), best_path)

        return TrainerOutput(best_model_path=str(best_path), best_score=best_score)

    def predict(self, model: MILModelBase, dataset: BagDatasetBase) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        loader = DataLoader(
            dataset,
            batch_size=self.config.mil.batch_size,
            shuffle=False,
            num_workers=self.config.experiment.num_workers,
            collate_fn=mil_collate,
        )

        outputs = []
        with torch.no_grad():
            for batch in loader:
                bag, _target, mask = unpack_mil_batch(batch)
                bag = bag.to(device)
                mask = mask.to(device) if mask is not None else None
                output = model(bag, mask=mask)
                if isinstance(output, dict):
                    output = output.get("logits", output.get("preds", output.get("output")))
                outputs.append(output.detach().cpu())

        return torch.cat(outputs, dim=0) if outputs else torch.empty((0,))

    def _evaluate_loss(
        self,
        model: MILModelBase,
        loss_fn: nn.Module,
        loader: DataLoader | None,
        device: torch.device,
    ) -> float:
        if loader is None:
            return float("inf")
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                bag, target, mask = unpack_mil_batch(batch)
                bag = bag.to(device)
                target = self._move_target(target, device)
                mask = mask.to(device) if mask is not None else None
                output = model(bag, mask=mask)
                loss, _logits = resolve_loss_and_logits(output, target, loss_fn)
                total_loss += loss.item()
                count += 1
        model.train()
        return total_loss / max(count, 1)

    def _checkpoint_path(self) -> Path:
        root = Path(self.config.experiment.project_root or ".")
        root.mkdir(parents=True, exist_ok=True)
        return root / "simple_trainer_best.pt"

    @staticmethod
    def _move_target(target: object, device: torch.device) -> object:
        if isinstance(target, dict):
            return {key: value.to(device) for key, value in target.items()}
        return target.to(device)