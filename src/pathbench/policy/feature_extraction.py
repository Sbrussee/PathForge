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
from pathbench.core.experiments.base import ComboConfig
from pathbench.core.datasets.slides import SlideDataset
from pathbench.utils.constants import EXPERIMENTS_DIR

logger = logging.getLogger(__name__)

class FeatureExtractionPolicy(PolicyBase):
    """
    Policy for extracting features from WSI slides.
    DEPENDENCIES: Depends ONLY on core abstractions (SlideProcessorBase).
    """
    def __init__(self, config: Config, datasets: list[SlideDataset]):
        self.config = config
        self.datasets = datasets
    
    def execute(self, combo_cfg: ComboConfig) -> None:
        backend_name = self.config.slide_processing.backend
        ProcessorClass = SLIDE_PROCESSORS.get(backend_name)
        
        if not ProcessorClass:
            raise ValueError(f"Slide processing backend '{backend_name}' not found in registry.")
            
        processor: SlideProcessorBase = ProcessorClass()
        logger.info("[Policy] Using backend '%s' -> %s", backend_name, processor)

        seg_config = {
            "method": "otsu",
            "params": self.config.slide_processing.qc_filters[0]
                     if self.config.slide_processing.qc_filters else {}
        }

        model_name = combo_cfg.feature_extraction
        feat_config = {
            "model": model_name,
            "params": {},  # later: batch_size, workers, etc.
        }

        tile_config = {
            "tile_px": combo_cfg.tile_px,
            "tile_mpp": combo_cfg.tile_mpp,
            "params": {},
        }

        logger.info(
            "[Policy] Combo: model=%s, tile_px=%s, tile_mpp=%s",
            model_name, combo_cfg.tile_px, combo_cfg.tile_mpp,
        )
        
        for ds in self.datasets:
            logger.info(
                "[Policy] Processing dataset '%s' (%d slides, used_for=%s)",
                ds.name, len(ds), ds.used_for,
            )

            for slide in tqdm(ds.samples, desc=f"Dataset: {ds.name}"):
                try:
                    logger.debug("[Policy] Slide %s -> %s", slide.slide, slide.wsi_path)

                    wsi = processor.load_slide(str(slide.wsi_path))
                    wsi = processor.segment_tissue(wsi, config=seg_config)
                    wsi = processor.extract_patches(wsi, config=tile_config)
                    wsi = processor.extract_features(wsi, config=feat_config)

                    # later: verify/save features here

                except Exception:
                    logger.exception("[Policy] Error processing slide %s (%s)",
                                     slide.slide, slide.wsi_path)