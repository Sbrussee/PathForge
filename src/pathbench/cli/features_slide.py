# src/pathbench/cli/feature_extraction_slide.py

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import typer

from ..config.config import Config
from ..core.datasets.wsi_dataset import WSI, WSIDataset
from ..core.experiments.base import Experiment
from ..policy.feature_extraction import FeatureExtractionPolicy
from .common import LOG_LEVEL_CHOICES, configure_logging, resolve_config_path


def _parse_fallback_mpp(row: pd.Series) -> float | None:
    if "fallback_mpp" not in row.index or pd.isna(row["fallback_mpp"]):
        return None

    try:
        fallback_mpp = float(row["fallback_mpp"])
    except (TypeError, ValueError):
        return None

    if fallback_mpp <= 0:
        return None
    return fallback_mpp


def _build_single_slide_wsi(
    *,
    row: pd.Series,
    selected_dataset: WSIDataset,
    slide_id: str,
    input_slide_path: Path,
) -> WSI:
    dataset_wsi = next(
        (sample for sample in selected_dataset.samples if sample.slide == slide_id),
        None,
    )
    fallback_mpp = (
        dataset_wsi.fallback_mpp
        if dataset_wsi is not None
        else _parse_fallback_mpp(row)
    )

    return WSI(
        slide=slide_id,
        patient=str(row["patient"]),
        category=str(row["category"]),
        path=input_slide_path,
        artifact_path=selected_dataset.slide_artifact_path(slide_id),
        fallback_mpp=fallback_mpp,
    )


def _make_single_slide_project_name(base_name: str) -> str:
    """
    Create one reusable project name per job.

    This avoids:
    - one project per slide
    - collisions between parallel SLURM jobs

    Within the same SLURM job, repeated CLI calls reuse the same project folder
    and simply overwrite project_root/annotations.csv each time.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"{base_name}__single_slide__job_{job_id}"
    return f"{base_name}__single_slide"


def run_feature_extraction_single_slide(
    *,
    config: Path,
    dataset: str,
    input_path: Path,
    log_level: str = "INFO",
) -> int:
    config_path = resolve_config_path(config)
    input_slide_path = Path(input_path)
    configure_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting feature extraction CLI (single slide)")
    logger.info("Using config: %s", config_path)
    logger.info("Using dataset: %s", dataset)
    logger.info("Using input slide: %s", input_slide_path)

    if not input_slide_path.exists():
        raise FileNotFoundError(f"Input slide not found: {input_slide_path}")

    slide_id = input_slide_path.stem

    # ---- load config ----
    cfg = Config.from_yaml(config_path)

    # ---- keep only the selected dataset in cfg.datasets ----
    selected_ds_cfg = None
    for ds_cfg in cfg.datasets:
        if ds_cfg.name == dataset:
            selected_ds_cfg = ds_cfg
            break

    if selected_ds_cfg is None:
        available_names = [ds.name for ds in cfg.datasets]
        raise ValueError(
            f"Dataset '{dataset}' not found in config.datasets. Available: {available_names}"
        )

    cfg.datasets = [selected_ds_cfg]

    # ---- create one reusable project per job (not per slide) ----
    cfg.experiment.project_name = _make_single_slide_project_name(cfg.experiment.project_name)

    # ---- create experiment first (creates project folder / project.json / annotations.csv if needed) ----
    experiment = Experiment(cfg)
    logger.info("Project root: %s", experiment.project_root)

    # ---- build a single-row annotations.csv for just this slide ----
    source_annotations = pd.read_csv(cfg.experiment.annotation_file)

    row_df = source_annotations[
        (source_annotations["dataset"] == dataset)
        & (source_annotations["slide"].astype(str) == slide_id)
    ].copy()

    if row_df.empty:
        raise ValueError(
            f"No annotation row found for dataset='{dataset}' and slide='{slide_id}'."
        )

    if len(row_df) > 1:
        raise ValueError(
            f"Expected exactly 1 annotation row for dataset='{dataset}' and slide='{slide_id}', "
            f"but found {len(row_df)}."
        )

    project_annotations_path = Path(experiment.project_root) / "annotations.csv"
    row_df.to_csv(project_annotations_path, index=False)
    logger.info("Wrote single-slide annotations to: %s", project_annotations_path)

    # ---- now build policy (it reads the overwritten annotations.csv) ----
    policy = FeatureExtractionPolicy(experiment)
    selected_dataset = WSIDataset(selected_ds_cfg, experiment.load_annotations())

    row = row_df.iloc[0]
    wsi = _build_single_slide_wsi(
        row=row,
        selected_dataset=selected_dataset,
        slide_id=slide_id,
        input_slide_path=input_slide_path,
    )

    combos = policy.build_combos()
    logger.info("Number of combos to run: %d", len(combos))
    for i, combo_cfg in enumerate(combos, start=1):
        logger.info(
            "[CLI] === Running combo %d/%d: model=%s, tile_px=%s, tile_mpp=%s ===",
            i,
            len(combos),
            combo_cfg.feature_extraction,
            combo_cfg.tile_px,
            combo_cfg.tile_mpp,
        )
        policy.process_slide(
            dataset=selected_dataset,
            wsi=wsi,
            combo_cfg=combo_cfg,
        )

    logger.info("Single-slide feature extraction DONE.")
    return 0


def run_command(
    config: Path = typer.Option(..., "--config", help="Path to YAML config"),
    dataset: str = typer.Option(
        ...,
        "--dataset",
        help="Dataset name (must exist in config.datasets).",
    ),
    input_path: Path = typer.Option(
        ...,
        "--input",
        help="Path to a single WSI file.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level.",
        show_default=True,
    ),
) -> None:
    raise SystemExit(
        run_feature_extraction_single_slide(
            config=config,
            dataset=dataset,
            input_path=input_path,
            log_level=log_level,
        )
    )


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Feature extraction (single slide, all combos from config)"
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Dataset name (must exist in config.datasets)",
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to a single WSI file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)
    return run_feature_extraction_single_slide(
        config=args.config,
        dataset=args.dataset,
        input_path=args.input,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    raise SystemExit(main())
