# src/pathbench/cli/tiles_report.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pathbench.config.config import Config
from pathbench.core.datasets.factory import build_wsi_datasets
from pathbench.core.experiments.base import ComboConfig, Experiment
from pathbench.core.experiments.combo_ids import build_tiling_id
from pathbench.core.reports.tiles_report_pdf import create_tiles_report_pdf


logger = logging.getLogger(__name__)


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate tile extraction PDF reports (from H5 tiles_overview) for all datasets/bags in config."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)

    # ---- logging config ----
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info("Starting tiles report CLI")
    logger.info("Using config: %s", config_path)

    cfg = Config.from_yaml(config_path)
    experiment = Experiment(cfg)
    datasets = build_wsi_datasets(
        cfg=experiment.cfg,
        annotations_df=experiment.load_annotations(),
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


if __name__ == "__main__":
    raise SystemExit(main())
