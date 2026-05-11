from __future__ import annotations

import copy
import importlib
import importlib.util
import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

REMOTE_SOURCES = {"gdc", "tcia", "patho-bench"}
ROLE_NAMES = ("training", "validation", "testing")
ANNOTATION_TABLE_KEYS = {
    "files_csv",
    "clinical_csv",
    "molecular_csv",
    "report_csv",
    "diagnosis_csv",
    "patient_studies_csv",
}


def resolve_external_dataset_sources(
    config_data: dict[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Resolve TCGA/TCIA dataset declarations into local PathBench datasets.

    Parameters
    ----------
    config_data:
        Mutable YAML payload parsed from a PathBench config file.
    config_path:
        Absolute path to the YAML file being loaded.

    Returns
    -------
    dict[str, Any]
        Config data with external dataset declarations replaced by concrete
        local dataset entries and a generated annotation file path.
    """
    resolved = copy.deepcopy(config_data)
    dataset_entries = list(resolved.get("datasets", []))
    if not dataset_entries:
        return resolved

    remote_entries = [
        entry for entry in dataset_entries if _is_remote_dataset_entry(entry)
    ]
    if not remote_entries:
        return resolved

    experiment_cfg = dict(resolved.get("experiment", {}))
    val_fraction = float(experiment_cfg.get("val_fraction", 0.1))
    datasets_root = _pathbench_repo_root() / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)

    generated_annotations: list[pd.DataFrame] = []
    concrete_datasets: list[dict[str, Any]] = []

    local_annotation_file = experiment_cfg.get("annotation_file")
    if local_annotation_file:
        local_annotation_path = Path(local_annotation_file)
        if local_annotation_path.is_file():
            generated_annotations.append(pd.read_csv(local_annotation_path))

    for entry in dataset_entries:
        if not _is_remote_dataset_entry(entry):
            concrete_datasets.append(entry)
            continue

        source = str(entry.get("source", "gdc")).lower()
        dataset_names = _normalize_dataset_names(entry)
        roles = _normalize_roles(entry.get("used_for", "training"))
        annotation_column = str(entry["annotation_column"])
        metadata_table = str(entry.get("metadata_table", "clinical_csv"))
        if metadata_table not in ANNOTATION_TABLE_KEYS:
            raise ValueError(
                f"Unsupported metadata_table '{metadata_table}'. "
                f"Expected one of {sorted(ANNOTATION_TABLE_KEYS)}."
            )

        requested_annotations = _normalize_string_list(entry.get("annotations"))
        required_annotations = _required_annotations_for_table(metadata_table)
        merged_annotations = sorted(
            set(requested_annotations).union(required_annotations)
        )

        dataset_root = (
            Path(entry.get("download_dir", datasets_root)).expanduser().resolve()
        )
        for dataset_name in dataset_names:
            artifacts = _materialize_dataset(
                dataset_name=dataset_name,
                source=source,
                dataset_root=dataset_root,
                metadata_table=metadata_table,
                requested_annotations=merged_annotations,
                datatype=entry.get("datatype"),
                task_name=entry.get("task_name"),
                download_raw_data=bool(entry.get("download_raw_data", False)),
                download_if_missing=bool(entry.get("download_if_missing", True)),
            )

            dataset_annotations = _build_annotation_frame(
                dataset_name=dataset_name,
                source=source,
                annotation_column=annotation_column,
                metadata_table=metadata_table,
                artifacts=artifacts,
            )
            generated_annotations.extend(
                _split_annotation_frame(
                    dataset_annotations,
                    dataset_name=dataset_name,
                    dataset_dir=Path(artifacts["dataset_dir"]),
                    roles=roles,
                    val_fraction=val_fraction,
                    concrete_datasets=concrete_datasets,
                )
            )

    merged_annotations = pd.concat(generated_annotations, ignore_index=True, sort=False)
    annotation_output = datasets_root / "pathbench_external_annotations.csv"
    merged_annotations.to_csv(annotation_output, index=False)

    logger.info(
        "Resolved %d external dataset declaration(s); generated annotations at %s",
        len(remote_entries),
        annotation_output,
    )
    experiment_cfg["annotation_file"] = str(annotation_output.resolve())
    resolved["experiment"] = experiment_cfg
    resolved["datasets"] = concrete_datasets
    return resolved


def _is_remote_dataset_entry(entry: dict[str, Any]) -> bool:
    source = entry.get("source")
    if source is not None and str(source).lower() in REMOTE_SOURCES:
        return True
    return "dataset_names" in entry or "annotation_column" in entry


def _normalize_dataset_names(entry: dict[str, Any]) -> list[str]:
    raw_names = entry.get(
        "dataset_names", entry.get("remote_dataset_names", entry.get("name"))
    )
    if raw_names is None:
        raise ValueError("Remote dataset entry requires 'dataset_names' or 'name'.")
    if isinstance(raw_names, str):
        dataset_names = [raw_names]
    else:
        dataset_names = [str(name) for name in raw_names]
    normalized = [name.strip() for name in dataset_names if str(name).strip()]
    if not normalized:
        raise ValueError("Remote dataset entry requires at least one dataset name.")
    return normalized


def _normalize_roles(raw_roles: Any) -> list[str]:
    if isinstance(raw_roles, str):
        roles = [raw_roles]
    elif isinstance(raw_roles, Iterable):
        roles = [str(role) for role in raw_roles]
    else:
        raise ValueError("used_for must be a string or list of strings.")

    normalized: list[str] = []
    for role in roles:
        role_name = role.strip().lower()
        if role_name == "all":
            normalized.extend(list(ROLE_NAMES))
            continue
        if role_name not in ROLE_NAMES:
            raise ValueError(
                f"Unsupported dataset role '{role}'. Expected one of {ROLE_NAMES} or 'all'."
            )
        normalized.append(role_name)

    deduplicated = list(dict.fromkeys(normalized))
    if not deduplicated:
        raise ValueError("At least one dataset role must be specified.")
    return deduplicated


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _required_annotations_for_table(metadata_table: str) -> set[str]:
    if metadata_table == "clinical_csv":
        return {"clinical"}
    if metadata_table == "molecular_csv":
        return {"molecular"}
    if metadata_table == "report_csv":
        return {"report"}
    if metadata_table == "diagnosis_csv":
        return {"diagnosis"}
    return set()


def _pathbench_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_tcga_tools_download():
    return _load_tcga_tools_module().Download


def _load_tcga_tools_list_datasets():
    return _load_tcga_tools_module().list_datasets


def _load_tcga_tools_module():
    if importlib.util.find_spec("tcga_tools") is None:
        tcga_repo = _pathbench_repo_root().parent / "TCGA_TOOLS_2.0"
        repo_str = str(tcga_repo)
        if not tcga_repo.is_dir():
            raise ImportError(
                "tcga-tools is required for external dataset integration. "
                "Install it or add ../TCGA_TOOLS_2.0 next to PathBench_2.0."
            )
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        if importlib.util.find_spec("tcga_tools") is None:
            raise ImportError(
                f"Could not resolve the tcga_tools package from {tcga_repo}."
            )
    return importlib.import_module("tcga_tools")


def _materialize_dataset(
    *,
    dataset_name: str,
    source: str,
    dataset_root: Path,
    metadata_table: str,
    requested_annotations: list[str],
    datatype: Any,
    task_name: str | None,
    download_raw_data: bool,
    download_if_missing: bool,
) -> dict[str, Any]:
    _validate_dataset_exists(dataset_name=dataset_name, source=source)
    download_fn = _load_tcga_tools_download()
    dataset_out_root = dataset_root.resolve()
    dataset_dir = dataset_out_root / dataset_name

    metadata_artifacts = download_fn(
        dataset_name=dataset_name,
        output_dir=str(dataset_out_root),
        annotations=requested_annotations or None,
        datatype=datatype,
        raw=True,
        source=source,
        task_name=task_name,
        download_raw_data=download_raw_data,
    )
    materialized = _coerce_artifact_paths(metadata_artifacts)
    materialized["dataset_dir"] = dataset_dir

    if download_if_missing and _download_required(materialized):
        logger.info(
            "Downloading dataset payload for %s into %s", dataset_name, dataset_dir
        )
        data_artifacts = download_fn(
            dataset_name=dataset_name,
            output_dir=str(dataset_out_root),
            annotations=requested_annotations or None,
            datatype=datatype,
            raw=False,
            source=source,
            task_name=task_name,
            download_raw_data=download_raw_data,
        )
        materialized.update(_coerce_artifact_paths(data_artifacts))
        materialized["dataset_dir"] = dataset_dir
    else:
        logger.info(
            "Dataset payload already present for %s at %s", dataset_name, dataset_dir
        )

    if metadata_table not in materialized:
        raise ValueError(
            f"TCGA-Tools did not return '{metadata_table}' for dataset '{dataset_name}'."
        )
    return materialized


def _validate_dataset_exists(*, dataset_name: str, source: str) -> None:
    list_datasets = _load_tcga_tools_list_datasets()
    available = list_datasets(source=source, as_dataframe=True)
    if not isinstance(available, pd.DataFrame) or available.empty:
        raise ValueError(
            f"tcga-tools could not enumerate datasets for source '{source}'."
        )

    candidate_columns = ("dataset_name", "project_id", "collection", "name")
    available_names: set[str] = set()
    for column_name in candidate_columns:
        if column_name in available.columns:
            available_names.update(
                str(value).strip().lower()
                for value in available[column_name].dropna().tolist()
            )

    if dataset_name.strip().lower() not in available_names:
        raise ValueError(
            f"Dataset '{dataset_name}' is not available in tcga-tools source '{source}'."
        )


def _coerce_artifact_paths(artifacts: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in artifacts.items():
        coerced[key] = (
            Path(value).resolve() if isinstance(value, (str, Path)) else value
        )
    return coerced


def _download_required(artifacts: dict[str, Any]) -> bool:
    data_dir = artifacts.get("data_dir")
    if not isinstance(data_dir, Path):
        return True
    if not data_dir.exists():
        return True
    return next(data_dir.iterdir(), None) is None


def _build_annotation_frame(
    *,
    dataset_name: str,
    source: str,
    annotation_column: str,
    metadata_table: str,
    artifacts: dict[str, Any],
) -> pd.DataFrame:
    files_df = _standardize_frame(
        pd.read_csv(Path(artifacts["files_csv"])),
        source=source,
        data_dir=Path(artifacts.get("data_dir", Path("."))),
    )

    metadata_path = Path(artifacts[metadata_table])
    metadata_df = (
        files_df.copy()
        if metadata_table == "files_csv"
        else _standardize_frame(
            pd.read_csv(metadata_path),
            source=source,
            data_dir=Path(artifacts.get("data_dir", Path("."))),
        )
    )

    if annotation_column not in metadata_df.columns:
        raise ValueError(
            f"Annotation column '{annotation_column}' was not found in {metadata_path}."
        )

    if metadata_table == "files_csv":
        merged = files_df.copy()
    else:
        join_key = _select_join_key(files_df, metadata_df)
        logger.info(
            "Using metadata table '%s' for dataset '%s'; selected task column '%s' joined on '%s'.",
            metadata_table,
            dataset_name,
            annotation_column,
            join_key,
        )
        merged = files_df.merge(
            metadata_df, how="left", on=join_key, suffixes=("", "_metadata")
        )

    merged["dataset"] = dataset_name
    merged["category"] = merged[annotation_column]
    merged["selected_annotation_column"] = annotation_column

    required_columns = ["dataset", "slide", "patient", "category", "wsi_path"]
    for column_name in required_columns:
        if column_name not in merged.columns:
            raise ValueError(
                f"Failed to derive required annotation column '{column_name}' for dataset '{dataset_name}'."
            )

    before_drop = len(merged)
    merged = merged.dropna(subset=["slide", "patient", "category", "wsi_path"]).copy()
    logger.info(
        "Built %d annotation rows for dataset '%s' from %d metadata rows.",
        len(merged),
        dataset_name,
        before_drop,
    )
    return merged


def _standardize_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    data_dir: Path,
) -> pd.DataFrame:
    standardized = frame.copy()
    standardized = standardized.rename(
        columns={
            "cases.submitter_id": "patient",
            "submitter_id": "patient",
            "PatientID": "patient",
            "case_submitter_id": "patient",
            "cases.case_id": "case_id",
            "case_id": "case_id",
            "file_name": "file_name",
            "SeriesInstanceUID": "series_instance_uid",
        }
    )

    if "slide" not in standardized.columns:
        if "file_name" in standardized.columns:
            standardized["slide"] = standardized["file_name"].map(
                lambda value: Path(str(value)).stem
            )
        elif "series_instance_uid" in standardized.columns:
            standardized["slide"] = standardized["series_instance_uid"].astype(str)

    if "patient" not in standardized.columns and "case_id" in standardized.columns:
        standardized["patient"] = standardized["case_id"]

    if "wsi_path" not in standardized.columns:
        if "local_path" in standardized.columns:
            standardized["wsi_path"] = standardized["local_path"]
        elif "file_name" in standardized.columns:
            standardized["wsi_path"] = standardized["file_name"].map(
                lambda name: str((data_dir / str(name)).resolve())
            )
        elif "series_instance_uid" in standardized.columns:
            standardized["wsi_path"] = standardized["series_instance_uid"].map(
                lambda uid: str((data_dir / f"{uid}.zip").resolve())
            )

    standardized["source"] = source
    return standardized


def _select_join_key(files_df: pd.DataFrame, metadata_df: pd.DataFrame) -> str:
    for key in ("case_id", "patient"):
        if key in files_df.columns and key in metadata_df.columns:
            return key
    raise ValueError(
        "Could not join downloaded metadata to file rows; expected a shared patient/case identifier."
    )


def _split_annotation_frame(
    annotations: pd.DataFrame,
    *,
    dataset_name: str,
    dataset_dir: Path,
    roles: list[str],
    val_fraction: float,
    concrete_datasets: list[dict[str, Any]],
) -> list[pd.DataFrame]:
    if len(roles) == 1:
        role = roles[0]
        concrete_datasets.append(
            _concrete_dataset_entry(
                dataset_name=dataset_name, dataset_dir=dataset_dir, role=role
            )
        )
        return [annotations]

    partitions = _partition_identifiers(
        identifiers=sorted(
            {str(value) for value in annotations["patient"].astype(str)}
        ),
        roles=roles,
        val_fraction=val_fraction,
    )

    split_frames: list[pd.DataFrame] = []
    for role, assigned_patients in partitions.items():
        concrete_name = f"{dataset_name}__{role}"
        role_frame = annotations[
            annotations["patient"].astype(str).isin(assigned_patients)
        ].copy()
        role_frame["dataset"] = concrete_name
        split_frames.append(role_frame)
        concrete_datasets.append(
            _concrete_dataset_entry(
                dataset_name=concrete_name, dataset_dir=dataset_dir, role=role
            )
        )
        logger.info(
            "Assigned %d patient(s) from dataset '%s' to role '%s'.",
            len(assigned_patients),
            dataset_name,
            role,
        )
    return split_frames


def _partition_identifiers(
    *,
    identifiers: list[str],
    roles: list[str],
    val_fraction: float,
) -> dict[str, set[str]]:
    if not identifiers:
        return {role: set() for role in roles}

    shuffled = identifiers[:]
    random.Random(0).shuffle(shuffled)
    ratios = _role_ratios(roles=roles, val_fraction=val_fraction)

    counts: list[int] = []
    remaining = len(shuffled)
    for index, role in enumerate(roles):
        if index == len(roles) - 1:
            count = remaining
        else:
            count = int(round(len(shuffled) * ratios[role]))
            max_remaining_for_others = len(roles) - index - 1
            count = max(0, min(count, remaining - max_remaining_for_others))
        counts.append(count)
        remaining -= count

    partitions: dict[str, set[str]] = {}
    cursor = 0
    for role, count in zip(roles, counts):
        partitions[role] = set(shuffled[cursor : cursor + count])
        cursor += count
    return partitions


def _role_ratios(*, roles: list[str], val_fraction: float) -> dict[str, float]:
    if len(roles) == 1:
        return {roles[0]: 1.0}
    if "validation" not in roles:
        shared = 1.0 / len(roles)
        return {role: shared for role in roles}

    validation_share = min(max(val_fraction, 0.0), 1.0)
    non_validation_roles = [role for role in roles if role != "validation"]
    if not non_validation_roles:
        return {"validation": 1.0}
    shared = (1.0 - validation_share) / len(non_validation_roles)
    ratios = {role: shared for role in non_validation_roles}
    ratios["validation"] = validation_share
    return ratios


def _concrete_dataset_entry(
    *,
    dataset_name: str,
    dataset_dir: Path,
    role: str,
) -> dict[str, Any]:
    return {
        "name": dataset_name,
        "slides_dir": str((dataset_dir / "data").resolve()),
        "artifacts_dir": str((dataset_dir / "pathbench_artifacts" / role).resolve()),
        "tissue_annotations_dir": None,
        "used_for": role,
    }
