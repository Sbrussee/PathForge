from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from pathbench.config.config import BagDatasetConfig, Config, DatasetEntry
from pathbench.core.datasets.base import BagDatasetBase

#TODO: add support for multiple splitting strategies (no patient leakage, center-level splits, etc.)
@dataclass
class BagDataset(BagDatasetBase):
    """
    Concrete implementation of a MIL Bag Dataset.
    Assumes features are pre-extracted and stored in .pt (torch) files.
    """
    _name: str
    features_dir: Path
    annotation_path: Path
    config: BagDatasetConfig = field(default_factory=BagDatasetConfig)
    _annotations: pd.DataFrame = field(init=False, repr=False)
    _bags: list[dict[str, Any]] = field(init=False, repr=False)
    _labels: list[Any] = field(init=False, repr=False)
    _slide_ids: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.features_dir = Path(self.features_dir)
        self.annotation_path = Path(self.annotation_path)

        if not self.features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")

        annotations = pd.read_csv(self.annotation_path)
        annotations = self._filter_by_dataset(annotations)
        annotations = self._validate_and_clean(annotations)

        self._annotations = annotations.reset_index(drop=True)
        self._bags = self._build_bags(self._annotations)
        self._labels = [bag["label"] for bag in self._bags]
        self._slide_ids = [bag["group_id"] for bag in self._bags]

    @classmethod
    def from_config(cls, dataset: DatasetEntry, config: Config) -> "BagDataset":
        """Build a BagDataset from Config + DatasetEntry definitions."""
        if dataset.features_path is None:
            raise ValueError(
                f"Dataset '{dataset.name}' is missing features_path required for MIL bag datasets."
            )
        return cls(
            _name=dataset.name,
            features_dir=Path(dataset.features_path),
            annotation_path=Path(config.experiment.annotation_file),
            config=config.bag_dataset,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_bags(self) -> int:
        return len(self._bags)
    

    @property
    def labels(self) -> Sequence[Any]:
        """Labels aligned with dataset indices."""
        return self._labels

    @property
    def slide_ids(self) -> Sequence[str]:
        """Slide identifiers aligned with dataset indices."""
        return self._slide_ids

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any] | Tuple[torch.Tensor, Any, str]:
        bag_info = self._bags[index]
        slide_id = bag_info["group_id"]
        label = bag_info["label"]

        bag_tensors = []
        for row in bag_info["rows"]:
            bag_path = self._resolve_feature_path(row)
            if not bag_path.exists():
                if self.config.allow_missing_features:
                    continue
                raise FileNotFoundError(
                    f"Features for slide '{row[self.config.id_column]}' not found at {bag_path}"
                )
            bag_tensors.append(self._load_bag(bag_path))

        if bag_tensors:
            bag = torch.cat(bag_tensors, dim=0)
        else:
            bag = torch.empty((0, 0), dtype=torch.float32)

        bag = self._sample_instances(bag)
        if self.config.return_slide_id:
            return bag, label, slide_id
        return bag, label


    def infer_feature_dim(self) -> int:
        """
        Infer the feature dimension by inspecting the first available bag.

        Returns:
            Feature dimension (number of columns in the bag tensor).

        Raises:
            ValueError: If no valid feature files are found.
        """
        for _, row in self._annotations.iterrows():
            bag_path = self._resolve_feature_path(row)
            if not bag_path.exists():
                continue
            bag = self._load_bag(bag_path)
            if bag.ndim == 2 and bag.shape[1] > 0:
                return int(bag.shape[1])

        raise ValueError(
            f"No valid feature files found in '{self.features_dir}' to infer input_dim."
        )

    def _filter_by_dataset(self, annotations: pd.DataFrame) -> pd.DataFrame:
        dataset_column = self.config.dataset_column
        if dataset_column in annotations.columns:
            return annotations[annotations[dataset_column] == self._name]
        return annotations

    def _validate_and_clean(self, annotations: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in (self.config.id_column,) if col not in annotations.columns]

        if self._uses_survival_labels():
            missing.extend(
                col
                for col in (self.config.time_column, self.config.event_column)
                if col is not None and col not in annotations.columns
            )
        else:
            if self.config.label_column not in annotations.columns:
                missing.append(self.config.label_column)

        if missing:
            raise ValueError(f"Missing required columns in annotations: {missing}")

        if self.config.drop_missing_labels:
            subset = [self.config.label_column]
            if self._uses_survival_labels():
                subset = [
                    col
                    for col in (self.config.time_column, self.config.event_column)
                    if col is not None
                ]
            annotations = annotations.dropna(subset=subset)

        return annotations

    def _resolve_feature_path(self, row: pd.Series) -> Path:
        if self.config.feature_path_column and self.config.feature_path_column in row:
            raw_path = Path(row[self.config.feature_path_column])
            return raw_path if raw_path.is_absolute() else self.features_dir / raw_path

        slide_id = str(row[self.config.id_column])
        filename = f"{slide_id}{self.config.feature_extension}"
        return self.features_dir / filename

    def _load_bag(self, path: Path) -> torch.Tensor:
        bag = torch.load(path)
        if isinstance(bag, np.ndarray):
            bag = torch.from_numpy(bag)
        if not isinstance(bag, torch.Tensor):
            raise TypeError(f"Unsupported bag type at {path}: {type(bag)}")
        if bag.ndim == 1:
            bag = bag.unsqueeze(0)
        return bag.float()

    def _sample_instances(self, bag: torch.Tensor) -> torch.Tensor:
        if self.config.max_instances is None:
            return bag

        if bag.shape[0] <= self.config.max_instances:
            return bag

        if self.config.sampling_strategy == "first":
            return bag[: self.config.max_instances]

        generator = torch.Generator()
        generator.manual_seed(self.config.random_seed)
        indices = torch.randperm(bag.shape[0], generator=generator)[: self.config.max_instances]
        return bag[indices]

    def _coerce_label(self, value: Any) -> Any:
        if self.config.label_dtype == "int":
            return int(value)
        if self.config.label_dtype == "float":
            return float(value)
        return str(value)

    def _uses_survival_labels(self) -> bool:
        return self.config.time_column is not None and self.config.event_column is not None

    def _build_label(self, row: pd.Series) -> Any:
        if self._uses_survival_labels():
            time_val = row[self.config.time_column]
            event_val = row[self.config.event_column]
            return {"time": float(time_val), "event": float(event_val)}
        return self._coerce_label(row[self.config.label_column])
    
    def _build_bags(self, annotations: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Build bag definitions based on the configured grouping strategy.

        Returns:
            List of bag descriptors containing:
            - group_id: Group identifier (slide/patient/tissue).
            - label: Label for the bag (must be consistent within group).
            - rows: List of annotation rows belonging to the bag.
        """
        group_column = self._resolve_group_column()
        if group_column not in annotations.columns:
            raise ValueError(
                f"Grouping column '{group_column}' not found in annotations."
            )

        bags: list[dict[str, Any]] = []
        for group_id, group_df in annotations.groupby(group_column, dropna=False):
            rows = [row for _, row in group_df.iterrows()]
            label = self._build_label(rows[0])
            if not self._labels_consistent(rows, label):
                raise ValueError(
                    f"Inconsistent labels found for group '{group_id}' in column '{group_column}'."
                )
            bags.append({"group_id": str(group_id), "label": label, "rows": rows})
        return bags

    def _resolve_group_column(self) -> str:
        strategy = self.config.grouping_strategy
        if strategy == "slide":
            return self.config.id_column
        if strategy == "patient":
            return self.config.patient_column
        if strategy == "tissue":
            return self.config.tissue_column
        raise ValueError(f"Unsupported grouping strategy: {strategy}")

    def _labels_consistent(self, rows: list[pd.Series], expected: Any) -> bool:
        for row in rows[1:]:
            if not self._labels_equal(self._build_label(row), expected):
                return False
        return True

    @staticmethod
    def _labels_equal(left: Any, right: Any) -> bool:
        if isinstance(left, dict) and isinstance(right, dict):
            return left == right
        return left == right