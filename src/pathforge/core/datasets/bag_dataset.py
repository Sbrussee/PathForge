from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Literal
from typing import Optional

import pandas as pd
import torch

from pathforge.config.config import DatasetEntry
from pathforge.core.datasets.bag_schema import BagBatch
from pathforge.core.datasets.bag_schema import as_bag_batch
from pathforge.core.datasets.base import BagDatasetBase
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.core.experiments.combo_ids import build_feature_name
from pathforge.core.experiments.combo_ids import build_tiling_id
from pathforge.core.io.slide_artifacts import features as features_io
from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.utils.constants import AGGREGATION_LEVELS
from pathforge.utils.constants import CASE_ID_COL
from pathforge.utils.constants import CATEGORY_COL
from pathforge.utils.constants import DATASET_COL
from pathforge.utils.constants import PATIENT_ID_COL
from pathforge.utils.constants import SLIDE_ID_COL

AggregationLevel = Literal[tuple(AGGREGATION_LEVELS)]
FeatureLevel = Literal["patch", "slide", "unknown", "invalid"]


@dataclass(slots=True)
class BagSample:
    """One logical bag unit, including its source slides and artifact locations."""

    sample_id: str
    slide_ids: list[str]
    artifact_paths: list[Path]
    category: Any
    patient_id: Optional[str] = None
    case_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SlideRetrievalDatasetItem:
    """One materialized retrieval item produced from a bag dataset."""

    index: int
    sample: BagSample
    inputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = int(self.index)
        self.inputs = dict(self.inputs or {})


SlideRetrievalSampleLoader = Callable[..., dict[str, Any]]


class BagDataset(BagDatasetBase):
    """Canonical bag dataset supporting artifact-backed and prepared-bag modes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.ds_cfg: DatasetEntry | None = None
        self.annotations_df: pd.DataFrame | None = None
        self.combo_cfg: ComboConfig | None = None
        self.aggregation_level: AggregationLevel = "slide"
        self.target_column: str = CATEGORY_COL
        self.task: str = "classification"
        self.slide_column: str | None = None
        self.time_column: str | None = None
        self.event_column: str | None = None
        self.bag_size: int | None = None
        self._name = ""
        self._mode: Literal["artifact", "prepared"] = "prepared"
        self._feature_level: FeatureLevel = "unknown"
        self._feature_level_reason = "Feature level inference was not run."
        self.samples: list[BagSample] = []
        self.annotations = pd.DataFrame()
        self.artifacts_dir: Path | None = None
        self.feature_path: Path | None = None
        self.tiling_id: str | None = None
        self.extractor_name: str | None = None
        self._resolved_slide_column: str | None = None

        if self._looks_like_artifact_mode(args, kwargs):
            self._init_artifact_mode(*args, **kwargs)
            return
        self._init_prepared_mode(*args, **kwargs)

    def _looks_like_artifact_mode(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> bool:
        if "ds_cfg" in kwargs or "combo_cfg" in kwargs:
            return True
        return bool(args) and isinstance(args[0], DatasetEntry)

    def _init_artifact_mode(self, *args: Any, **kwargs: Any) -> None:
        if args:
            self.ds_cfg = args[0]
            self.annotations_df = args[1]
            self.combo_cfg = args[2]
            remaining = args[3:]
            if remaining:
                self.aggregation_level = remaining[0]
        else:
            self.ds_cfg = kwargs["ds_cfg"]
            self.annotations_df = kwargs["annotations_df"]
            self.combo_cfg = kwargs["combo_cfg"]
            self.aggregation_level = kwargs.get("aggregation_level", "slide")

        self.target_column = str(kwargs.get("target_column", self.target_column))
        self.task = str(kwargs.get("task", self.task))
        self.slide_column = kwargs.get("slide_column")
        self.time_column = kwargs.get("time_column")
        self.event_column = kwargs.get("event_column")
        bag_size = kwargs.get("bag_size")
        self.bag_size = None if bag_size is None else int(bag_size)
        self._mode = "artifact"

        if self.ds_cfg is None or self.annotations_df is None or self.combo_cfg is None:
            raise ValueError("Artifact-backed BagDataset requires ds_cfg, annotations_df, and combo_cfg.")

        self._name = self.ds_cfg.name
        self.annotations = self.annotations_df.copy()
        self.artifacts_dir = Path(self.ds_cfg.artifacts_dir).expanduser().resolve()
        self.tiling_id = build_tiling_id(self.combo_cfg)
        self.extractor_name = build_feature_name(self.combo_cfg)

        df = self.annotations[self.annotations[DATASET_COL] == self.ds_cfg.name].copy()
        if df.empty:
            self.samples = []
            return

        self.samples = self._build_samples(df)
        self._feature_level, self._feature_level_reason = self._infer_feature_level()

    def _init_prepared_mode(self, *args: Any, **kwargs: Any) -> None:
        if len(args) >= 4:
            name = args[0]
            feature_path = args[1]
            annotation_path = args[2]
            target_column = args[3]
        else:
            name = kwargs["name"]
            feature_path = kwargs["feature_path"]
            annotation_path = kwargs["annotation_path"]
            target_column = kwargs["target_column"]

        self._name = str(name)
        self.feature_path = Path(str(feature_path)).expanduser().resolve()
        self.annotation_path = Path(str(annotation_path)).expanduser().resolve()
        self.target_column = str(target_column)
        self.task = str(kwargs.get("task", self.task))
        dataset_name = kwargs.get("dataset_name")
        self.slide_column = kwargs.get("slide_column")
        self.time_column = kwargs.get("time_column")
        self.event_column = kwargs.get("event_column")
        bag_size = kwargs.get("bag_size")
        self.bag_size = None if bag_size is None else int(bag_size)
        self._mode = "prepared"

        if not self.feature_path.exists():
            raise FileNotFoundError(f"Feature path {self.feature_path} does not exist.")

        annotations_df = kwargs.get("annotations_df")
        if annotations_df is None:
            annotations_df = pd.read_csv(self.annotation_path)
        else:
            annotations_df = annotations_df.copy()
        if dataset_name is not None and DATASET_COL in annotations_df.columns:
            annotations_df = annotations_df[
                annotations_df[DATASET_COL].astype(str) == str(dataset_name)
            ].copy()
            if annotations_df.empty:
                raise FileNotFoundError(
                    f"No annotation rows found for dataset '{dataset_name}' in {self.annotation_path}."
                )
        self.annotations = annotations_df
        self._resolved_slide_column = self._resolve_slide_column()
        self._feature_level = "patch"
        self._feature_level_reason = "Prepared bag tensors are treated as patch-level bags."

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_bags(self) -> int:
        if self._mode == "artifact":
            return len(self.samples)
        return len(self.annotations)

    @property
    def feature_dim(self) -> int:
        if self.num_bags == 0:
            raise ValueError("Cannot infer feature_dim from an empty bag dataset.")
        bag = self[0]["X"]
        if bag.ndim != 2:
            raise ValueError(f"Bag tensors must have shape [N, D]. Got {bag.shape}.")
        return int(bag.shape[1])

    def output_dim(self) -> int:
        if self.task == "classification":
            return int(self._annotations_frame()[self.target_column].nunique())
        if self.task == "survival_discrete":
            time_column = self._resolve_time_column()
            return int(self._annotations_frame()[time_column].max()) + 1
        return 1

    def __getitem__(self, index: int) -> BagBatch:
        bag = self.load_bag(index)
        target = self._target_from_index(index)
        return as_bag_batch({"X": bag, "Y": target})

    def get_sample(self, index: int) -> BagSample:
        mode = getattr(self, "_mode", "artifact")
        if mode != "artifact":
            raise RuntimeError("get_sample() is only available for artifact-backed bag datasets.")
        return self.samples[index]

    def get_feature_level(self) -> FeatureLevel:
        return self._feature_level

    def get_feature_level_reason(self) -> str:
        return self._feature_level_reason

    def load_bag(self, index: int) -> torch.Tensor:
        if self._mode == "prepared":
            return self._load_prepared_bag(index)

        sample = self.samples[index]
        bags: list[torch.Tensor] = []
        for artifact_path in sample.artifact_paths:
            bags.append(self._load_slide_bag(artifact_path))
        if not bags:
            raise RuntimeError(f"No bags found for sample '{sample.sample_id}'.")
        bag = torch.cat(bags, dim=0).float()
        return self._materialize_bag_size(bag)

    def get_bag_sample(self, index: int) -> tuple[torch.Tensor, BagSample]:
        return self.load_bag(index), self.get_sample(index)

    def _target_from_index(self, index: int) -> Any:
        if self._mode == "prepared":
            row = self._annotations_frame().iloc[index]
            return self._target_from_row(row)

        sample = self.samples[index]
        if self.task == "classification":
            return torch.tensor(int(sample.category), dtype=torch.long)
        if self.task == "regression":
            return torch.tensor(float(sample.category), dtype=torch.float32)
        if self.task in {"survival", "survival_discrete"}:
            time_column = self._resolve_time_column()
            event_column = self._resolve_event_column()
            metadata = sample.metadata
            is_discrete = self.task == "survival_discrete"
            target = {
                "time": torch.tensor(
                    int(metadata[time_column]) if is_discrete else float(metadata[time_column]),
                    dtype=torch.long if is_discrete else torch.float32,
                ),
                "event": torch.tensor(float(metadata[event_column]), dtype=torch.float32),
            }
            continuous_time_column = self._resolve_continuous_time_column(time_column)
            if is_discrete and continuous_time_column in metadata:
                target["continuous_time"] = torch.tensor(
                    float(metadata[continuous_time_column]),
                    dtype=torch.float32,
                )
            return target
        return sample.category

    def _load_prepared_bag(self, index: int) -> torch.Tensor:
        row = self._annotations_frame().iloc[index]
        slide_id = str(row[self._resolve_slide_column()])
        assert self.feature_path is not None
        bag_path = self.feature_path / f"{slide_id}.pt"
        if not bag_path.exists():
            raise FileNotFoundError(
                f"Features for slide {slide_id} not found at {bag_path}"
            )
        bag = torch.load(bag_path)
        if not isinstance(bag, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor at {bag_path}, got {type(bag)!r}.")
        return self._materialize_bag_size(bag.float())

    def _materialize_bag_size(self, bag: torch.Tensor) -> torch.Tensor:
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

    def _annotations_frame(self) -> pd.DataFrame:
        return self.annotations if self._mode == "prepared" else self.annotations_df

    def _resolve_slide_column(self) -> str:
        if self._resolved_slide_column is not None:
            return self._resolved_slide_column
        annotations = self._annotations_frame()
        if self.slide_column is not None:
            if self.slide_column not in annotations.columns:
                raise ValueError(
                    f"Configured slide column {self.slide_column!r} is not present in annotations."
                )
            self._resolved_slide_column = self.slide_column
            return self._resolved_slide_column
        for candidate in (SLIDE_ID_COL, "slide", "slide_id"):
            if candidate in annotations.columns:
                self._resolved_slide_column = candidate
                return self._resolved_slide_column
        raise ValueError(
            "Annotations must contain either 'slide_id' or 'slide' to resolve bag files."
        )

    def _resolve_time_column(self) -> str:
        annotations = self._annotations_frame()
        if self.time_column is not None:
            return self.time_column
        for candidate in ("time_bin", "time", "os_months", "survival_time", "OS_MONTHS"):
            if candidate in annotations.columns:
                return candidate
        raise ValueError(
            "No survival time column was configured or inferable from annotations."
        )

    def _resolve_event_column(self) -> str:
        annotations = self._annotations_frame()
        if self.event_column is not None:
            return self.event_column
        for candidate in ("event", "status", "survival_event", "OS_STATUS"):
            if candidate in annotations.columns:
                return candidate
        raise ValueError(
            "No survival event column was configured or inferable from annotations."
        )

    def _resolve_continuous_time_column(self, discrete_time_column: str) -> str:
        annotations = self._annotations_frame()
        for candidate in ("continuous_time", "os_months", "survival_time", "OS_MONTHS", "time"):
            if candidate != discrete_time_column and candidate in annotations.columns:
                return candidate
        return discrete_time_column

    def _target_from_row(self, row: pd.Series) -> Any:
        if self.task == "classification":
            return torch.tensor(int(row[self.target_column]), dtype=torch.long)
        if self.task == "regression":
            return torch.tensor(float(row[self.target_column]), dtype=torch.float32)
        if self.task in {"survival", "survival_discrete"}:
            time_column = self._resolve_time_column()
            event_column = self._resolve_event_column()
            is_discrete = self.task == "survival_discrete"
            target = {
                "time": torch.tensor(
                    int(row[time_column]) if is_discrete else float(row[time_column]),
                    dtype=torch.long if is_discrete else torch.float32,
                ),
                "event": torch.tensor(float(row[event_column]), dtype=torch.float32),
            }
            continuous_time_column = self._resolve_continuous_time_column(time_column)
            if is_discrete and continuous_time_column in row.index:
                target["continuous_time"] = torch.tensor(
                    float(row[continuous_time_column]),
                    dtype=torch.float32,
                )
            return target
        return row[self.target_column]

    def _build_samples(self, df: pd.DataFrame) -> list[BagSample]:
        if self.aggregation_level == "slide":
            return self._build_slide_samples(df)
        if self.aggregation_level == "case":
            if CASE_ID_COL not in df.columns:
                raise ValueError(
                    f"aggregation_level='case' requires column '{CASE_ID_COL}' in annotations."
                )
            return self._build_grouped_samples(df, group_col=CASE_ID_COL)
        if self.aggregation_level == "patient":
            if PATIENT_ID_COL not in df.columns:
                raise ValueError(
                    f"aggregation_level='patient' requires column '{PATIENT_ID_COL}' in annotations."
                )
            return self._build_grouped_samples(df, group_col=PATIENT_ID_COL)
        raise ValueError(f"Unsupported aggregation_level: {self.aggregation_level!r}")

    def _build_slide_samples(self, df: pd.DataFrame) -> list[BagSample]:
        samples: list[BagSample] = []
        for _, row in df.iterrows():
            slide_id = str(row[SLIDE_ID_COL])
            row_df = pd.DataFrame([row])
            artifact_paths = [self._artifact_path(slide_id)]
            self._ensure_features_exist(
                sample_id=slide_id,
                slide_ids=[slide_id],
                artifact_paths=artifact_paths,
            )
            samples.append(
                BagSample(
                    sample_id=slide_id,
                    slide_ids=[slide_id],
                    artifact_paths=artifact_paths,
                    category=self._resolve_single_value(
                        row_df,
                        self.target_column,
                        missing_column_error=True,
                    ),
                    patient_id=self._resolve_single_value(row_df, PATIENT_ID_COL, cast=str),
                    case_id=self._resolve_single_value(row_df, CASE_ID_COL, cast=str),
                    metadata=self._build_metadata(row_df),
                )
            )
        return samples

    def _build_grouped_samples(self, df: pd.DataFrame, group_col: str) -> list[BagSample]:
        samples: list[BagSample] = []
        for group_value, group_df in df.groupby(group_col, sort=False):
            group_df = group_df.sort_values(SLIDE_ID_COL)
            slide_ids = [str(x) for x in group_df[SLIDE_ID_COL].tolist()]
            artifact_paths = [self._artifact_path(slide_id) for slide_id in slide_ids]
            self._ensure_features_exist(
                sample_id=str(group_value),
                slide_ids=slide_ids,
                artifact_paths=artifact_paths,
            )
            samples.append(
                BagSample(
                    sample_id=str(group_value),
                    slide_ids=slide_ids,
                    artifact_paths=artifact_paths,
                    category=self._resolve_single_value(
                        group_df,
                        self.target_column,
                        missing_column_error=True,
                    ),
                    patient_id=self._resolve_single_value(group_df, PATIENT_ID_COL, cast=str),
                    case_id=self._resolve_single_value(group_df, CASE_ID_COL, cast=str),
                    metadata=self._build_metadata(group_df),
                )
            )
        return samples

    def _resolve_single_value(
        self,
        group_df: pd.DataFrame,
        column: Optional[str],
        *,
        missing_column_error: bool = False,
        cast: Optional[type] = None,
    ) -> Any:
        if column is None:
            return None
        if column not in group_df.columns:
            if missing_column_error:
                raise ValueError(f"Target column '{column}' not found in annotations.")
            return None
        series = group_df[column].dropna()
        if cast is not None:
            series = series.astype(cast)
        values = series.unique().tolist()
        if len(values) == 0:
            return None
        if len(values) > 1:
            raise ValueError(
                f"Inconsistent values for grouped bag in column '{column}': {values}"
            )
        return values[0]

    def _build_metadata(self, group_df: pd.DataFrame) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            DATASET_COL: self.ds_cfg.name if self.ds_cfg is not None else self.name,
            "aggregation_level": self.aggregation_level,
            "num_slides": len(group_df),
        }
        tracked_columns = {
            CATEGORY_COL,
            PATIENT_ID_COL,
            CASE_ID_COL,
            self.target_column,
            self.time_column,
            self.event_column,
        }
        tracked_columns.discard(None)
        for column in group_df.columns:
            if column not in tracked_columns and column not in {DATASET_COL, SLIDE_ID_COL}:
                continue
            values = group_df[column].dropna().unique().tolist()
            if not values:
                continue
            metadata[column] = values[0] if len(values) == 1 else values
        return metadata

    def _ensure_features_exist(
        self,
        *,
        sample_id: str,
        slide_ids: list[str],
        artifact_paths: list[Path],
    ) -> None:
        if self.tiling_id is None or self.extractor_name is None:
            raise RuntimeError("Artifact-backed feature checks require tiling_id and extractor_name.")
        for slide_id, artifact_path in zip(slide_ids, artifact_paths):
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"Artifact file not found for sample '{sample_id}', slide '{slide_id}': "
                    f"{artifact_path}"
                )
            with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                if not features_io.features_exist(
                    slide_artifact,
                    bag_id=self.tiling_id,
                    extractor_name=self.extractor_name,
                ):
                    raise FileNotFoundError(
                        f"Missing features for sample '{sample_id}', slide '{slide_id}' in "
                        f"{artifact_path} for tiling_id='{self.tiling_id}', "
                        f"extractor='{self.extractor_name}'."
                    )

    def _infer_feature_level(self, max_slides_to_check: int = 10) -> tuple[FeatureLevel, str]:
        checked_paths: set[Path] = set()
        inferred_level: FeatureLevel | None = None
        inferred_from_path: Path | None = None
        checked_count = 0

        for sample in self.samples:
            for artifact_path in sample.artifact_paths:
                if artifact_path in checked_paths:
                    continue
                checked_paths.add(artifact_path)
                with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                    current_level = features_io.infer_feature_level(
                        slide_artifact,
                        bag_id=self.tiling_id,
                        extractor_name=self.extractor_name,
                    )
                    try:
                        feature_matrix = features_io.read_features(
                            slide_artifact,
                            bag_id=self.tiling_id,
                            extractor_name=self.extractor_name,
                        )
                        n_features = int(feature_matrix.shape[0])
                        n_patches = tiles_io.coords_num_rows(
                            slide_artifact,
                            bag_id=self.tiling_id,
                        )
                    except Exception as exc:
                        detail = f"failed to inspect feature rows/patch rows: {exc}"
                    else:
                        detail = f"feature_rows={n_features}, patch_rows={n_patches}"
                    checked_count += 1

                if current_level == "invalid":
                    return (
                        "invalid",
                        f"Dataset '{self.name}' artifact '{artifact_path.name}' has invalid feature structure ({detail}).",
                    )
                if current_level == "unknown":
                    if checked_count >= max_slides_to_check:
                        return (
                            inferred_level or "unknown",
                            f"Dataset '{self.name}' remained ambiguous after {checked_count} checked artifact(s); latest artifact '{artifact_path.name}' had {detail}.",
                        )
                    continue
                if inferred_level is None:
                    inferred_level = current_level
                    inferred_from_path = artifact_path
                elif inferred_level != current_level:
                    return (
                        "invalid",
                        f"Dataset '{self.name}' has inconsistent feature levels: first non-ambiguous artifact '{inferred_from_path.name if inferred_from_path else 'unknown'}' inferred '{inferred_level}', but '{artifact_path.name}' inferred '{current_level}' ({detail}).",
                    )
                if checked_count >= max_slides_to_check:
                    return (
                        inferred_level,
                        f"Dataset '{self.name}' inferred feature level '{inferred_level}' after {checked_count} checked artifact(s).",
                    )

        if inferred_level is not None:
            return (
                inferred_level,
                f"Dataset '{self.name}' inferred feature level '{inferred_level}' from checked artifacts.",
            )
        return (
            "unknown",
            f"Dataset '{self.name}' only had ambiguous artifacts where feature_rows == patch_rows == 1 across {checked_count} checked artifact(s).",
        )

    def _load_slide_bag(self, artifact_path: Path) -> torch.Tensor:
        if self.tiling_id is None or self.extractor_name is None:
            raise RuntimeError("Artifact-backed bag loading requires tiling_id and extractor_name.")
        with FileHandleH5(artifact_path, mode="r") as slide_artifact:
            feature_matrix = features_io.read_features(
                slide_artifact,
                bag_id=self.tiling_id,
                extractor_name=self.extractor_name,
            )
        return torch.from_numpy(feature_matrix)

    def _artifact_path(self, slide_id: str) -> Path:
        if self.artifacts_dir is None:
            raise RuntimeError("Artifact-backed bag loading requires artifacts_dir.")
        return self.artifacts_dir / f"{slide_id}.h5"


class MILBagDataset(BagDataset):
    """MIL dataset alias for the canonical bag schema."""


class SlideRetrievalBagDataset(BagDataset):
    """Bag dataset variant that binds retrieval-specific sample loaders."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.sample_loader: SlideRetrievalSampleLoader | None = None
        super().__init__(*args, **kwargs)

    def bind_sample_loader(
        self,
        sample_loader: SlideRetrievalSampleLoader,
    ) -> None:
        self.sample_loader = sample_loader

    def clear_sample_loader(self) -> None:
        self.sample_loader = None

    def __getitem__(self, index: int) -> SlideRetrievalDatasetItem:
        if self.sample_loader is None:
            raise RuntimeError(
                "SlideRetrievalBagDataset requires a bound sample_loader before __getitem__ can be used."
            )
        sample = self.get_sample(index)
        loaded_inputs = self.sample_loader(
            index=index,
            sample=sample,
            base_dataset=self,
        )
        return SlideRetrievalDatasetItem(
            index=index,
            sample=sample,
            inputs=loaded_inputs,
        )
