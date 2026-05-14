from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple
import torch
import pandas as pd
from pathlib import Path

from pathbench.core.datasets.base import BagDatasetBase


@dataclass
class BagDataset(BagDatasetBase):
    """
    Concrete implementation of a MIL Bag Dataset.
    Assumes features are pre-extracted and stored in .pt (torch) files.
    """

    _name: str
    feature_path: str
    annotation_path: str
    target_column: str
    task: str = "classification"
    slide_column: str | None = None
    time_column: str | None = None
    event_column: str | None = None
    bag_size: int | None = None

    def __post_init__(self):
        self.annotations = pd.read_csv(self.annotation_path)
        self._resolved_slide_column = self._resolve_slide_column()
        # Validate paths
        if not Path(self.feature_path).exists():
            raise FileNotFoundError(f"Feature path {self.feature_path} does not exist.")

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_bags(self) -> int:
        return len(self.annotations)

    @property
    def feature_dim(self) -> int:
        """Return the per-instance feature dimension ``D`` inferred from one bag."""

        if self.num_bags == 0:
            raise ValueError("Cannot infer feature_dim from an empty bag dataset.")
        bag, _ = self[0]
        if bag.ndim != 2:
            raise ValueError(f"Bag tensors must have shape [N, D]. Got {bag.shape}.")
        return int(bag.shape[1])

    def output_dim(self) -> int:
        """Infer model output dimensionality from task annotations."""

        if self.task == "classification":
            return int(self.annotations[self.target_column].nunique())
        if self.task == "survival_discrete":
            time_column = self._resolve_time_column()
            return int(self.annotations[time_column].max()) + 1
        return 1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """
        Returns:
            bag: (N, D) Tensor
            label: scalar
        """
        row = self.annotations.iloc[index]
        slide_id = row[self._resolved_slide_column]

        # Construct path to feature file (assuming {slide_id}.pt pattern)
        # In production, consider a more robust path mapping strategy
        f_path = Path(self.feature_path) / f"{slide_id}.pt"

        if not f_path.exists():
            # Handle missing files gracefully or raise
            raise FileNotFoundError(
                f"Features for slide {slide_id} not found at {f_path}"
            )

        bag = torch.load(f_path)

        # Ensure bag is float32 and correct shape
        if isinstance(bag, torch.Tensor):
            bag = bag.float()
            bag = self._materialize_bag_size(bag)

        return bag, self._target_from_row(row)

    def _materialize_bag_size(self, bag: torch.Tensor) -> torch.Tensor:
        """Return a variable-size or fixed-size bag according to ``self.bag_size``.

        When ``bag_size`` is ``None``, the full bag is returned unchanged. When a
        fixed size is requested, bags larger than that size are deterministically
        downsampled across their instance axis, while smaller bags are expanded by
        repeating instances so the returned tensor always has shape
        ``[bag_size, feature_dim]``.
        """

        if self.bag_size is None:
            return bag
        if bag.ndim != 2:
            raise ValueError(
                f"Bag tensors must have shape [N, D] before bag-size materialization. Got {bag.shape}."
            )

        num_instances = int(bag.shape[0])
        target_size = int(self.bag_size)
        if num_instances == target_size:
            return bag
        if num_instances == 0:
            raise ValueError("Cannot materialize a fixed bag size from an empty bag.")
        if num_instances > target_size:
            indices = torch.linspace(
                0,
                num_instances - 1,
                steps=target_size,
                device=bag.device,
            ).round().long()
            return bag.index_select(0, indices)

        repeat_indices = torch.arange(target_size, device=bag.device) % num_instances
        return bag.index_select(0, repeat_indices)

    def _resolve_slide_column(self) -> str:
        if self.slide_column is not None:
            if self.slide_column not in self.annotations.columns:
                raise ValueError(
                    f"Configured slide column {self.slide_column!r} is not present in annotations."
                )
            return self.slide_column
        for candidate in ("slide_id", "slide"):
            if candidate in self.annotations.columns:
                return candidate
        raise ValueError(
            "Annotations must contain either 'slide_id' or 'slide' to resolve bag files."
        )

    def _resolve_time_column(self) -> str:
        if self.time_column is not None:
            return self.time_column
        for candidate in ("time_bin", "time", "os_months"):
            if candidate in self.annotations.columns:
                return candidate
        raise ValueError(
            "No survival time column was configured or inferable from annotations."
        )

    def _resolve_event_column(self) -> str:
        if self.event_column is not None:
            return self.event_column
        for candidate in ("event", "status"):
            if candidate in self.annotations.columns:
                return candidate
        raise ValueError(
            "No survival event column was configured or inferable from annotations."
        )

    def _target_from_row(self, row: pd.Series) -> Any:
        if self.task == "classification":
            return torch.tensor(int(row[self.target_column]), dtype=torch.long)
        if self.task == "regression":
            return torch.tensor(float(row[self.target_column]), dtype=torch.float32)
        if self.task in {"survival", "survival_discrete"}:
            time_column = self._resolve_time_column()
            event_column = self._resolve_event_column()
            is_discrete = self.task == "survival_discrete"
            return {
                "time": torch.tensor(
                    int(row[time_column]) if is_discrete else float(row[time_column]),
                    dtype=torch.long if is_discrete else torch.float32,
                ),
                "event": torch.tensor(float(row[event_column]), dtype=torch.float32),
            }
        return row[self.target_column]
