from __future__ import annotations

"""Utilities for building MIL-ready bag datasets from pre-extracted features."""

from dataclasses import dataclass
from typing import Any, Iterable, Tuple
import torch
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from pathbench.core.datasets.base import BagDatasetBase


class _BagComponent:
    """Represents the contribution of a single slide to a bag.

    Attributes:
        slide_id: Identifier of the slide providing tiles.
        feature_path: Location of the ``.pt`` tensor with tile features.
        coord_path: Optional path to the ``.npz`` file containing tile coordinates.
        tile_indices: Indices of tiles belonging to the bag (used for tissue bags).
    """
    slide_id: str
    feature_path: Path
    coord_path: Optional[Path]
    tile_indices: Optional[np.ndarray] = None
    
@dataclass
class _BagDefinition:
    """Container describing one MIL bag and its metadata."""

    bag_id: str
    label: Any
    patient_id: Optional[str]
    tissue_id: Optional[int]
    components: List[_BagComponent]


class BagDataset(BagDatasetBase):
    """Multiple Instance Learning dataset backed by pre-computed features.

    The dataset expects one ``.pt`` file per slide containing a tensor with shape
    ``(num_tiles, num_features)``. When available, a companion ``.npz`` file per
    slide can provide coordinates in ``(num_tiles, 2 or 3)`` format where the third
    column stores an optional ``tissue_id`` used for tissue-level bagging.

    Parameters
    ----------
    name:
        Human readable dataset identifier (e.g. ``"train"``).
    annotations:
        Path to a CSV file or a pre-loaded ``pandas.DataFrame`` with at least the
        columns ``slide`` and ``label_column``. ``patient_column`` is optional but
        required for patient-level aggregation.
    feature_dir:
        Directory containing ``{slide_id}.pt`` files.
    label_column:
        Column in ``annotations`` containing target labels.
    coord_dir:
        Directory containing coordinate ``.npz`` files. If ``None`` coordinates are
        considered unavailable.
    bag_level:
        How to group tiles into bags: ``"slide"`` (default), ``"patient"`` or
        ``"tissue"``.
    patient_column:
        Column describing the patient identifier. Required when ``bag_level`` is
        ``"patient"``.
    tissue_column:
        Column name within the coordinate array to use as tissue id. By default the
        third column of the coordinate matrix is interpreted as tissue id when it
        exists; this argument is reserved for future explicit schema support.
    dataset_filter:
        Optional list of dataset names to retain (matching a ``dataset`` column in
        ``annotations``). When ``None`` all rows are used.
    """

    def __init__(
        self,
        name: str,
        annotations: str | Path | pd.DataFrame,
        feature_dir: str | Path,
        label_column: str,
        coord_dir: str | Path | None = None,
        bag_level: str = "slide",
        patient_column: str = "patient",
        tissue_column: str = "tissue_id",
        dataset_filter: Optional[Sequence[str]] = None,
    ) -> None:
        self._name = name
        self.feature_dir = Path(feature_dir)
        self.coord_dir = Path(coord_dir) if coord_dir is not None else None
        self.label_column = label_column
        self.patient_column = patient_column
        self.tissue_column = tissue_column
        self.bag_level = bag_level
        self.dataset_filter = set(dataset_filter) if dataset_filter else None

        self.annotations = self._load_annotations(annotations)
        self._bags: List[_BagDefinition] = self._build_bags()

    # ------------------------------------------------------------------
    # DatasetBase API
    # ------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def num_bags(self) -> int:
        return len(self._bags)
    
        return len(self._bags)

    def group_indices_by_patient(self) -> Dict[str, List[int]]:
        """Group dataset indices by patient identifier.

        Returns a mapping of ``patient_id`` to list of dataset indices, ensuring
        that downstream splitters can enforce patient-level separation.
        """
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx, bag in enumerate(self._bags):
            patient = str(bag.patient_id) if bag.patient_id is not None else f"bag_{bag.bag_id}"
            groups[patient].append(idx)
        return groups

    def sampler(self, shuffle: bool = True) -> Iterable[int]:
        """Return a PyTorch sampler suitable for DataLoader construction."""

        if shuffle:
            return RandomSampler(self)
        return SequentialSampler(self)

    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], Any]:
        bag_def = self._bags[index]

        features: List[torch.Tensor] = []
        coords: List[torch.Tensor] = []
        slide_ids: List[str] = []

        for comp in bag_def.components:
            feats = self._load_feature_tensor(comp.feature_path)

            if comp.tile_indices is not None:
                feats = feats[comp.tile_indices]
            features.append(feats)
            slide_ids.append(comp.slide_id)

            if comp.coord_path and comp.coord_path.exists():
                coord_arr = self._load_coords(comp.coord_path)
                if comp.tile_indices is not None:
                    coord_arr = coord_arr[comp.tile_indices]
                coords.append(torch.as_tensor(coord_arr, dtype=torch.float32))

        bag_tensor = torch.cat(features, dim=0) if features else torch.empty(0, 0)
        coord_tensor = torch.cat(coords, dim=0) if coords else None

        payload = {
            "features": bag_tensor,
            "coordinates": coord_tensor,
            "bag_id": bag_def.bag_id,
            "slides": slide_ids,
            "patient_id": bag_def.patient_id,
            "tissue_id": bag_def.tissue_id,
        }
        return payload, bag_def.label

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _load_annotations(self, annotations: str | Path | pd.DataFrame) -> pd.DataFrame:
        if isinstance(annotations, (str, Path)):
            df = pd.read_csv(annotations)
        else:
            df = annotations.copy()

        missing_cols = {"slide", self.label_column} - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Annotations are missing required columns: {sorted(missing_cols)}"
            )

        if self.dataset_filter is not None and "dataset" in df.columns:
            df = df[df["dataset"].isin(self.dataset_filter)].reset_index(drop=True)

        if df.empty:
            raise ValueError("No annotation rows remain after filtering.")

        return df

    def _build_bags(self) -> List[_BagDefinition]:
        if self.bag_level not in {"slide", "patient", "tissue"}:
            raise ValueError(
                "bag_level must be one of ['slide', 'patient', 'tissue']"
            )

        if self.bag_level == "slide":
            return [self._build_slide_bag(row) for _, row in self.annotations.iterrows()]

        if self.bag_level == "patient":
            if self.patient_column not in self.annotations.columns:
                raise ValueError(
                    f"patient_column '{self.patient_column}' is required for patient-level bags"
                )
            grouped = self.annotations.groupby(self.patient_column)
            return [self._build_patient_bag(pid, group) for pid, group in grouped]

        # tissue-level aggregation
        return self._build_tissue_bags()

    def _build_slide_bag(self, row: pd.Series) -> _BagDefinition:
        slide_id = row["slide"]
        label = row[self.label_column]
        comp = self._create_component(slide_id)
        return _BagDefinition(
            bag_id=str(slide_id),
            label=label,
            patient_id=row.get(self.patient_column, None),
            tissue_id=None,
            components=[comp],
        )

    def _build_patient_bag(self, patient_id: str, group: pd.DataFrame) -> _BagDefinition:
        labels = group[self.label_column].unique()
        if len(labels) != 1:
            raise ValueError(
                f"Patient {patient_id} has multiple labels: {labels}. Ensure label consistency."
            )

        comps = [self._create_component(row["slide"]) for _, row in group.iterrows()]
        return _BagDefinition(
            bag_id=str(patient_id),
            label=labels[0],
            patient_id=patient_id,
            tissue_id=None,
            components=comps,
        )

    def _build_tissue_bags(self) -> List[_BagDefinition]:
        if self.coord_dir is None:
            raise ValueError("Coordinate directory is required for tissue-level bags.")

        bags: List[_BagDefinition] = []
        for _, row in self.annotations.iterrows():
            slide_id = row["slide"]
            label = row[self.label_column]
            comp = self._create_component(slide_id, tissue_split=True)

            if comp.tile_indices is None:
                # No tissue information; fallback to single-bag representation
                bags.append(
                    _BagDefinition(
                        bag_id=str(slide_id),
                        label=label,
                        patient_id=row.get(self.patient_column, None),
                        tissue_id=None,
                        components=[comp],
                    )
                )
                continue

            coords = self._load_coords(comp.coord_path)  # type: ignore[arg-type]
            tissue_ids = (
                coords[:, 2].astype(int)
                if coords.shape[1] >= 3
                else np.zeros(len(coords), dtype=int)
            )

            for tissue_id in np.unique(tissue_ids):
                indices = np.where(tissue_ids == tissue_id)[0]
                bags.append(
                    _BagDefinition(
                        bag_id=f"{slide_id}_tissue{tissue_id}",
                        label=label,
                        patient_id=row.get(self.patient_column, None),
                        tissue_id=int(tissue_id),
                        components=[
                            _BagComponent(
                                slide_id=comp.slide_id,
                                feature_path=comp.feature_path,
                                coord_path=comp.coord_path,
                                tile_indices=indices,
                            )
                        ],
                    )
                )
        return bags

    def _create_component(self, slide_id: str, tissue_split: bool = False) -> _BagComponent:
        feature_path = self.feature_dir / f"{slide_id}.pt"
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        coord_path: Optional[Path] = None
        if self.coord_dir is not None:
            candidate = self.coord_dir / f"{slide_id}.npz"
            coord_path = candidate if candidate.exists() else None

        tile_indices = None
        if tissue_split and coord_path is None:
            # Without coordinate file, we cannot split by tissue
            return _BagComponent(slide_id=slide_id, feature_path=feature_path, coord_path=None)

        return _BagComponent(
            slide_id=slide_id,
            feature_path=feature_path,
            coord_path=coord_path,
            tile_indices=tile_indices,
        )
    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def _load_feature_tensor(self, path: Path) -> torch.Tensor:
        tensor = torch.load(path)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        return tensor.float()

    def _load_coords(self, path: Path) -> np.ndarray:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            # default key when saved via np.savez
            if "coords" in data.files:
                return np.array(data["coords"], dtype=float)
            return np.array(data[data.files[0]], dtype=float)
        return np.array(data, dtype=float)
    