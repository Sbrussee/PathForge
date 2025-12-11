from __future__ import annotations
from typing import Any
from tqdm import tqdm
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
import logging
import os 

from pathbench.policy.base import PolicyBase
from pathbench.config.config import Config
from pathbench.utils.registries import SLIDE_PROCESSORS
from pathbench.core.slide_processing.base import SlideProcessorBase
from pathbench.core.experiments.base import ComboConfig, Experiment
from pathbench.core.datasets.slides import SlideDataset
from pathbench.utils.constants import EXPERIMENTS_DIR

logger = logging.getLogger(__name__)

class FeatureExtractionPolicy(PolicyBase):
    """
    Policy for extracting features from WSI slides.
    Depends on Experiment for project context (paths, datasets, combos).
    """

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)
        # Pre-compute reusable context
        self.datasets: list[SlideDataset] = self.experiment.build_datasets()
        self.combos: list[ComboConfig] = self.experiment.build_combinations(
            ["feature_extraction", "tile_px", "tile_mpp"]
        )
        self.project_root = experiment.project_root
        self.features_dir = Path(self.project_root) / "features"

    def execute(self) -> dict[str, Any]:
        backend_name = self.config.slide_processing.backend
        ProcessorClass = SLIDE_PROCESSORS.get(backend_name)

        if not ProcessorClass:
            raise ValueError(
                f"Slide processing backend '{backend_name}' not found in registry."
            )

        processor: SlideProcessorBase = ProcessorClass()
        logger.info("[Policy] Using backend '%s' -> %s", backend_name, processor)

        seg_config = {
            "method": self.config.slide_processing.segmentation_method,
            "params": (
                self.config.slide_processing.qc_filters[0]
                if self.config.slide_processing.qc_filters
                else {}
            ),
        }


        logger.info("[Policy] Number of benchmark parameter combos: %d", len(self.combos))

        for i, combo_cfg in enumerate(self.combos):
            model_name = combo_cfg.feature_extraction
            tile_config = {
                "tile_px": combo_cfg.tile_px,
                "tile_mpp": combo_cfg.tile_mpp,
                "params": {},
            }
            feat_config = {
                "model": model_name,
                "params": {},
            }

            logger.info(
                "[Policy] === Running combo %d/%d: model=%s, tile_px=%s, tile_mpp=%s ===",
                i + 1,
                len(self.combos),
                model_name,
                combo_cfg.tile_px,
                combo_cfg.tile_mpp,
            )

            for ds in self.datasets:
                logger.info(
                    "[Policy] Processing dataset '%s' (%d slides, used_for=%s)",
                    ds.name,
                    len(ds),
                    ds.used_for,
                )

                for slide in tqdm(ds.samples, desc=f"Dataset: {ds.name}"):
                    try:
                        logger.debug(
                            "[Policy] Slide %s -> %s", slide.slide, slide.wsi_path
                        )
                        
                        exists = processor.check_for_existing_features(
                            tile_config=tile_config, feat_config=feat_config,
                            experiment_dir=self.project_root, features_dir=self.features_dir,
                            slide_id=slide.slide
                        )
                        
                        if not exists:
                            logger.info(
                                "[Policy] Extracting features for slide %s", slide.slide
                            )

                            wsi = processor.load_slide(str(slide.wsi_path))
                            wsi = processor.segment_tissue(wsi, config=seg_config)
                            wsi = processor.extract_patches(wsi, config=tile_config)
                            wsi = processor.extract_features(wsi, config=feat_config)

                            processor.save_features(wsi, tile_config=tile_config, feat_config=feat_config,
                                                    experiment_dir=self.project_root, features_dir=self.features_dir)
                        else:
                            logger.info(
                                "[Policy] Features already exist for slide %s, skipping.",
                                slide.slide,
                            )
                        
                    except Exception:
                        logger.exception(
                            "[Policy] Error processing slide %s (%s)",
                            slide.slide,
                            slide.wsi_path,
                        )

        logger.info("[Policy] Feature extraction DONE.")
        return {"status": "feature_extraction_done"}