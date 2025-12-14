from __future__ import annotations

from typing import Any
from tqdm import tqdm
from pathlib import Path
import logging

from pathbench.policy.base import PolicyBase
from pathbench.utils.registries import SLIDE_PROCESSORS
from pathbench.core.slide_processing.base import SlideProcessorBase
from pathbench.core.experiments.base import ComboConfig, Experiment
from pathbench.core.datasets.slides import SlideDataset

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
        logger.info("[Policy] Number of benchmark parameter combos: %d", len(self.combos))

        for i, combo_cfg in enumerate(self.combos, start=1):
            logger.info(
                "[Policy] === Running combo %d/%d: model=%s, tile_px=%s, tile_mpp=%s ===",
                i,
                len(self.combos),
                combo_cfg.feature_extraction,
                combo_cfg.tile_px,
                combo_cfg.tile_mpp,
            )
            self.run_combo(combo_cfg)

        logger.info("[Policy] Feature extraction DONE.")
        return {"status": "feature_extraction_done"}

    def run_combo(self, combo_cfg: ComboConfig) -> None:
        for ds in self.datasets:
            logger.info(
                "[Policy] Processing dataset '%s' (%d slides, used_for=%s)",
                ds.name,
                len(ds),
                ds.used_for,
            )
            self.run_dataset(ds=ds, combo_cfg=combo_cfg)

    def run_dataset(self, ds: SlideDataset, combo_cfg: ComboConfig) -> None:
        processor = self._build_processor()
        seg_config = self._build_seg_config()
        tile_config = self._build_tile_config(combo_cfg)
        feat_config = self._build_feat_config(combo_cfg)

        for slide in tqdm(ds.samples, desc=f"Dataset: {ds.name}"):
            slide_id = slide.slide
            wsi_path = slide.wsi_path

            try:
                logger.debug("[Policy] Slide %s -> %s", slide_id, wsi_path)

                exists = processor.check_for_existing_features(
                    tile_config=tile_config,
                    feat_config=feat_config,
                    experiment_dir=self.project_root,
                    features_dir=self.features_dir,
                    slide_id=slide_id,
                )

                if exists:
                    logger.info("[Policy] Features already exist for slide %s, skipping.", slide_id)
                    continue

                logger.info("[Policy] Extracting features for slide %s", slide_id)

                wsi = processor.load_slide(str(wsi_path))
                wsi = processor.segment_tissue(wsi, config=seg_config)
                wsi = processor.extract_patches(wsi, config=tile_config)
                wsi = processor.extract_features(wsi, config=feat_config)

                processor.save_features(
                    wsi,
                    tile_config=tile_config,
                    feat_config=feat_config,
                    experiment_dir=self.project_root,
                    features_dir=self.features_dir,
                )

            except Exception:
                logger.exception("[Policy] Error processing slide %s (%s)", slide_id, wsi_path)

    def _build_processor(self) -> SlideProcessorBase:
        backend_name = self.config.slide_processing.backend
        ProcessorClass = SLIDE_PROCESSORS.get(backend_name)
        if not ProcessorClass:
            raise ValueError(f"Slide processing backend '{backend_name}' not found in registry.")

        processor: SlideProcessorBase = ProcessorClass()
        logger.info("[Policy] Using backend '%s' -> %s", backend_name, processor)
        return processor

    def _build_seg_config(self) -> dict[str, Any]:
        return {
            "method": self.config.slide_processing.segmentation_method,
            "params": (
                self.config.slide_processing.qc_filters[0]
                if self.config.slide_processing.qc_filters
                else {}
            ),
        }

    def _build_tile_config(self, combo_cfg: ComboConfig) -> dict[str, Any]:
        return {
            "tile_px": combo_cfg.tile_px,
            "tile_mpp": combo_cfg.tile_mpp,
            "params": {},
        }

    def _build_feat_config(self, combo_cfg: ComboConfig) -> dict[str, Any]:
        return {
            "model": combo_cfg.feature_extraction,
            "params": {},
        }
