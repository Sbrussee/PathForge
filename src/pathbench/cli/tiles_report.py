# src/pathbench/cli/tiles_report.py

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from pathbench.cli.base import (
    add_config_argument,
    add_log_level_argument,
    configure_logging,
    load_experiment,
)
from pathbench.config.config import Config
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.core.experiments.combo_ids import build_tiling_id
from pathbench.core.reports.tiles_report_pdf import create_tiles_report_pdf
from pathbench.utils.constants import DATASET_COL, SLIDE_ID_COL


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ArtifactOnlyWSI:
    slide: str
    artifact_path: Path


@dataclass(slots=True)
class _ArtifactOnlyDataset:
    name: str
    artifacts_dir: Path
    samples: list[_ArtifactOnlyWSI]

    def __len__(self) -> int:
        return len(self.samples)


def _build_artifact_only_datasets(cfg: Config, annotations_df) -> list[_ArtifactOnlyDataset]:
    datasets: list[_ArtifactOnlyDataset] = []

    for ds_cfg in cfg.datasets:
        if ds_cfg.used_for == "ignore":
            continue

        dataset_annotations = annotations_df[
            annotations_df[DATASET_COL] == ds_cfg.name
        ].copy()

        if dataset_annotations.empty:
            logger.warning(
                "[TilesReportCLI] No annotation rows found for dataset '%s'.",
                ds_cfg.name,
            )
            continue

        artifacts_dir = Path(ds_cfg.artifacts_dir).expanduser().resolve()
        samples = [
            _ArtifactOnlyWSI(
                slide=str(row[SLIDE_ID_COL]),
                artifact_path=artifacts_dir / f"{row[SLIDE_ID_COL]}.h5",
            )
            for _, row in dataset_annotations.iterrows()
        ]

        datasets.append(
            _ArtifactOnlyDataset(
                name=str(ds_cfg.name),
                artifacts_dir=artifacts_dir,
                samples=samples,
            )
        )

    return datasets


def _unique_tiling_ids_from_config(cfg: Config) -> list[str]:
    """
    Build unique tiling IDs from config benchmark tiling combinations.

    Keeps insertion order.
    """
    seen: set[str] = set()
    tiling_ids: list[str] = []

    for tile_px in cfg.benchmark_parameters.tile_px:
        for tile_mpp in cfg.benchmark_parameters.tile_mpp:
            combo_cfg = ComboConfig(
                tile_px=int(tile_px),
                tile_mpp=float(tile_mpp),
            )
            bid = build_tiling_id(combo_cfg)
            if bid not in seen:
                seen.add(bid)
                tiling_ids.append(bid)

    return tiling_ids


def main(argv: list[str] | None = None) -> int:
    """Build a PDF tiles-overview report from existing slide artifact files."""
    parser = argparse.ArgumentParser(
        description="Generate tile extraction PDF reports (from H5 tiles_overview) for all datasets/bags in config."
    )
    add_config_argument(parser)
    add_log_level_argument(parser)
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    logger.info("Starting tiles report CLI")
    logger.info("Using config: %s", args.config)

    experiment = load_experiment(args.config)
    annotations_df = experiment.load_annotations()
    datasets = _build_artifact_only_datasets(
        cfg=experiment.cfg,
        annotations_df=annotations_df,
    )
    tiling_ids = _unique_tiling_ids_from_config(experiment.cfg)

    logger.info("Found %d dataset(s) in config", len(datasets))
    logger.info("Found %d unique tiling_id(s): %s", len(tiling_ids), tiling_ids)

    if not datasets:
        logger.warning("No datasets found in config. Nothing to do.")
        return 0

    if not tiling_ids:
        logger.warning("No tile combinations found in config (tile_px/tile_mpp). Nothing to do.")
        return 0

    created_count = 0
    failed_count = 0
    skipped_count = 0

    for dataset in datasets:
        logger.info(
            "[TilesReportCLI] Dataset '%s' (%d slides) -> artifacts_dir=%s",
            dataset.name,
            len(dataset),
            dataset.artifacts_dir,
        )

        for tiling_id in tiling_ids:
            try:
                out_path = create_tiles_report_pdf(dataset=dataset, bag_id=tiling_id)
                created_count += 1
                logger.info("[TilesReportCLI] Created report: %s", out_path)

            except RuntimeError as e:
                # Expected case when no overviews exist yet for this dataset/tiling_id.
                skipped_count += 1
                logger.warning(
                    "[TilesReportCLI] Skipping dataset='%s', tiling_id='%s': %s",
                    dataset.name,
                    tiling_id,
                    e,
                )
            except Exception:
                failed_count += 1
                logger.exception(
                    "[TilesReportCLI] Failed report generation for dataset='%s', tiling_id='%s'",
                    dataset.name,
                    tiling_id,
                )

    logger.info(
        "[TilesReportCLI] Done. created=%d skipped=%d failed=%d",
        created_count,
        skipped_count,
        failed_count,
    )

    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
