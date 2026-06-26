# src/pathforge/cli/tiles_report.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import typer

from pathforge.config.config import Config
from pathforge.cli.common import LOG_LEVEL_CHOICES, configure_logging, resolve_config_path
from pathforge.core.experiments.base import Experiment
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.core.experiments.combo_ids import build_tiling_id
from pathforge.core.reports.tiles_report_pdf import create_tiles_report_pdf
from pathforge.utils.constants import DATASET_COL, SLIDE_ID_COL


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


def run_tiles_report(
    *,
    config: Path,
    log_level: str = "INFO",
) -> int:
    """Generate the tiles report for one YAML config and return an exit code."""
    configure_logging(log_level)
    config_path = resolve_config_path(config)

    logger.info("Starting tiles report CLI")
    logger.info("Using config: %s", config_path)

    cfg = Config.from_yaml(config_path)
    experiment = Experiment(cfg)
    annotations_df = experiment.load_annotations()
    datasets = _build_artifact_only_datasets(
        cfg=experiment.cfg,
        annotations_df=annotations_df,
    )
    tiling_ids = _unique_tiling_ids_from_config(cfg)

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


def run_command(
    config: Path = typer.Option(..., "--config", help="Path to YAML config"),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level.",
        show_default=True,
    ),
) -> None:
    """Typer command that builds the tiles report from the provided config option."""
    raise SystemExit(run_tiles_report(config=config, log_level=log_level))


def main(argv: list[str] | None = None) -> int:
    """Argparse entry point for the tiles-report CLI; returns a process exit code."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate tile extraction PDF reports (from H5 tiles_overview) for all datasets/bags in config."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)
    return run_tiles_report(config=args.config, log_level=args.log_level)


if __name__ == "__main__":
    raise SystemExit(main())
