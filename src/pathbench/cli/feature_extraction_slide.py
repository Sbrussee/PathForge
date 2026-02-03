# src/pathbench/cli/feature_extract.py

import argparse
from pathlib import Path
import logging

from ..config.config import Config
from ..core.experiments.base import Experiment
from ..policy.feature_extraction import FeatureExtractionPolicy
from ..core.datasets.wsi_dataset import WSI


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Feature extraction (single slide, all combos from config)")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name (must exist in config.datasets)")
    parser.add_argument("--input", required=True, type=Path, help="Path to a single WSI file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)

    # ---- logging config (once) ----
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting feature extraction CLI (single slide)")
    logger.info("Using config: %s", args.config)
    logger.info("Using dataset: %s", args.dataset)
    logger.info("Using input slide: %s", args.input)

    config_path = Path(args.config)
    input_slide_path = Path(args.input)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if not input_slide_path.exists():
        raise FileNotFoundError(f"Input slide not found: {input_slide_path}")

    cfg = Config.from_yaml(config_path)

    # Experiment() determines project_root and creates the project folder if needed
    experiment = Experiment(cfg)
    logger.info("Project root: %s", experiment.project_root)

    policy = FeatureExtractionPolicy(experiment)

    selected_dataset: object | None = None
    for dataset in policy.datasets:
        if dataset.name == args.dataset:
            selected_dataset = dataset
            break

    if selected_dataset is None:
        available_names = [ds.name for ds in policy.datasets]
        raise ValueError(
            f"Dataset '{args.dataset}' not found in config.datasets. Available: {available_names}"
        )

    slide_id = input_slide_path.stem
    wsi = WSI(
        slide=slide_id,
        patient="unknown",
        category="unknown",
        path=input_slide_path,
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
