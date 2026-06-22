# src/pathbench/cli/feature_extraction_slide.py

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from ..core.datasets.wsi_dataset import WSI
from ..policy.feature_extraction import FeatureExtractionPolicy
from .base import (
    add_config_argument,
    add_log_level_argument,
    configure_logging,
    load_config,
)
from ..core.experiments.base import Experiment


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


def main(argv: list[str] | None = None) -> int:
    """Run feature extraction for one slide described by a temporary annotation row."""
    parser = argparse.ArgumentParser(
        description="Feature extraction (single slide, all combos from config)"
    )
    add_config_argument(parser)
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Dataset name (must exist in config.datasets)",
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to a single WSI file")
    add_log_level_argument(parser)
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting feature extraction CLI (single slide)")
    logger.info("Using config: %s", args.config)
    logger.info("Using dataset: %s", args.dataset)
    logger.info("Using input slide: %s", args.input)

    input_slide_path = Path(args.input)

    if not input_slide_path.exists():
        raise FileNotFoundError(f"Input slide not found: {input_slide_path}")

    slide_id = input_slide_path.stem

    # ---- load config ----
    cfg = load_config(args.config)

    # ---- keep only the selected dataset in cfg.datasets ----
    selected_ds_cfg = None
    for ds_cfg in cfg.datasets:
        if ds_cfg.name == args.dataset:
            selected_ds_cfg = ds_cfg
            break

    if selected_ds_cfg is None:
        available_names = [ds.name for ds in cfg.datasets]
        raise ValueError(
            f"Dataset '{args.dataset}' not found in config.datasets. Available: {available_names}"
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
        (source_annotations["dataset"] == args.dataset)
        & (source_annotations["slide"].astype(str) == slide_id)
    ].copy()

    if row_df.empty:
        raise ValueError(
            f"No annotation row found for dataset='{args.dataset}' and slide='{slide_id}'."
        )

    if len(row_df) > 1:
        raise ValueError(
            f"Expected exactly 1 annotation row for dataset='{args.dataset}' and slide='{slide_id}', "
            f"but found {len(row_df)}."
        )

    project_annotations_path = Path(experiment.project_root) / "annotations.csv"
    row_df.to_csv(project_annotations_path, index=False)
    logger.info("Wrote single-slide annotations to: %s", project_annotations_path)

    # ---- now build policy (it reads the overwritten annotations.csv) ----
    policy = FeatureExtractionPolicy(experiment)

    if not policy.datasets:
        raise RuntimeError(
            f"No datasets were built after filtering to dataset '{args.dataset}'."
        )

    selected_dataset = policy.datasets[0]

    row = row_df.iloc[0]
    wsi = WSI(
        slide=slide_id,
        patient=str(row["patient"]),
        category=str(row["category"]),
        path=input_slide_path,
        artifact_path=selected_dataset.slide_artifact_path(slide_id),
    )

    logger.info("Number of combos to run: %d", len(policy.combos))
    for i, combo_cfg in enumerate(policy.combos, start=1):
        logger.info(
            "[CLI] === Running combo %d/%d: model=%s, tile_px=%s, tile_mpp=%s ===",
            i,
            len(policy.combos),
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


if __name__ == "__main__":
    raise SystemExit(main())
