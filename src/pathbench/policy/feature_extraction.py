from __future__ import annotations

from typing import Any, Optional

from importlib import import_module

from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm

from pathbench.policy.base import PolicyBase
from pathbench.core.experiments.base import ComboConfig, Experiment
from pathbench.core.datasets.wsi_dataset import WSI, WSIDataset
from pathbench.core.slide_processing.base import SlideProcessorBase
from pathbench.utils.registries import SLIDE_PROCESSORS

from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5 import tiles as tiles_io
from pathbench.core.io.h5 import features as features_io
from pathbench.core.io.h5 import tissue as tissue_io

logger = logging.getLogger(__name__)


class FeatureExtractionPolicy(PolicyBase):
    """Extract tile features into per-slide H5 artifacts."""

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)
        self.datasets: list[WSIDataset] = self.experiment.build_datasets()
        self.combos: list[ComboConfig] = self.experiment.build_combinations(
            ["feature_extraction", "tile_px", "tile_mpp"]
        )
        self.config = experiment.cfg
        self.backend_name = self.config.slide_processing.backend

    def execute(self) -> dict[str, Any]:
        logger.info("[Policy] Number of parameter combos: %d", len(self.combos))

        for combo_index, combo_cfg in enumerate(self.combos, start=1):
            logger.info(
                "[Policy] === Combo %d/%d: extractor=%s, tile_px=%s, tile_mpp=%s ===",
                combo_index,
                len(self.combos),
                combo_cfg.feature_extraction,
                combo_cfg.tile_px,
                combo_cfg.tile_mpp,
            )
            self.execute_combo(combo_cfg)

        logger.info("[Policy] Feature extraction DONE.")
        return {"status": "feature_extraction_done"}

    def execute_combo(self, combo_cfg: ComboConfig) -> None:
        for dataset in self.datasets:
            logger.info(
                "[Policy] Dataset '%s' (%d slides, used_for=%s)",
                dataset.name,
                len(dataset),
                dataset.used_for,
            )
            self.execute_dataset(dataset=dataset, combo_cfg=combo_cfg)

    def execute_dataset(self, dataset: WSIDataset, combo_cfg: ComboConfig) -> None:
        slide_processor = self._build_processor()
        run_configs = self._build_run_configs(combo_cfg)

        for wsi in tqdm(dataset.samples, desc=f"Dataset: {dataset.name}"):
            self._execute_wsi(
                dataset=dataset,
                wsi=wsi,
                slide_processor=slide_processor,
                run_configs=run_configs,
            )

    def process_slide(self, dataset: WSIDataset, wsi: WSI, combo_cfg: ComboConfig) -> None:
        slide_processor = self._build_processor()
        run_configs = self._build_run_configs(combo_cfg)
        self._execute_wsi(
            dataset=dataset,
            wsi=wsi,
            slide_processor=slide_processor,
            run_configs=run_configs,
        )

    def _execute_wsi(
        self,
        *,
        dataset: WSIDataset,
        wsi: WSI,
        slide_processor: SlideProcessorBase,
        run_configs: dict[str, Any],
    ) -> None:
        slide_id = wsi.slide
        artifact_path = wsi.artifact_path

        segmentation_config = run_configs["seg_config"]
        tiling_config = run_configs["tile_config"]
        feature_config = run_configs["feat_config"]

        tile_px: int = int(tiling_config["tile_px"])
        tile_mpp: float = float(tiling_config["tile_mpp"])
        extractor_name: str = str(feature_config["model"])
        bag_id = self._bag_id(tile_px=tile_px, tile_mpp=tile_mpp)

        expected_tiling_spec = {
            "tile_px": tile_px,
            "tile_mpp": tile_mpp,
            "stride_px": tile_px, 
            "coord_space": "level0" 
        }

        try:
            with FileHandleH5(artifact_path, mode="a") as slide_artifact:
                # ---- Fast skip if features already valid (no slide open) ----
                coords_row_count = tiles_io.coords_num_rows(slide_artifact, bag_id)
                if features_io.features_exist(
                    slide_artifact,
                    bag_id=bag_id,
                    extractor_name=extractor_name,
                    expected_rows=coords_row_count,
                ):
                    logger.info("[Policy] Features exist for slide %s (%s/%s), skipping.", slide_id, bag_id, extractor_name)
                    return

                # ---- Coords reuse or recompute ----
                coords_are_valid = tiles_io.coords_exist(slide_artifact, bag_id) and tiles_io.tiling_spec_matches(
                    slide_artifact,
                    bag_id=bag_id,
                    expected_tiling_spec=expected_tiling_spec,
                )

                coords_array: Optional[np.ndarray] = None
                tiling_spec: Optional[dict[str, Any]] = None

                if coords_are_valid:
                    coords_array = tiles_io.read_coords(slide_artifact, bag_id)
                    tiling_spec = tiles_io.read_tiling_spec(slide_artifact, bag_id)
                else:
                    tissue_polygons = self._resolve_tissue_polygons(
                        dataset=dataset,
                        wsi=wsi,
                        slide_artifact=slide_artifact,
                        slide_processor=slide_processor,
                        segmentation_config=segmentation_config,
                    )

                    # coords + tiling_spec must be produced by backend for this config
                    slide_processor.load_wsi(wsi)
                    try:
                        coords_array, tiling_spec = slide_processor.extract_patches(
                            wsi,
                            tissue_polygons,
                            config=tiling_config,
                        )
                    finally:
                        slide_processor.close_wsi(wsi)

                    coords_array = self._ensure_coords_array(coords_array)
                    tiling_spec = self._ensure_tiling_spec_dict(tiling_spec, expected_tiling_spec=expected_tiling_spec)

                    tiles_io.write_coords(slide_artifact, bag_id, coords_array)
                    tiles_io.write_tiling_spec(slide_artifact, bag_id, tiling_spec)

                if coords_array is None or tiling_spec is None:
                    raise RuntimeError("[Policy] Internal error: coords_array/tiling_spec not resolved.")

                # ---- Features reuse or recompute ----
                if features_io.features_exist(
                    slide_artifact,
                    bag_id=bag_id,
                    extractor_name=extractor_name,
                    expected_rows=int(coords_array.shape[0]),
                ):
                    logger.info("[Policy] Features exist for slide %s (%s/%s), skipping.", slide_id, bag_id, extractor_name)
                    return

                slide_processor.load_wsi(wsi)
                try:
                    feature_matrix = slide_processor.extract_features(
                        wsi,
                        coords_array,
                        tiling_spec,
                        config={**feature_config, **tiling_config},
                    )
                finally:
                    slide_processor.close_wsi(wsi)

                feature_matrix = self._ensure_feature_matrix(feature_matrix, expected_rows=int(coords_array.shape[0]))
                features_io.write_features(slide_artifact, bag_id, extractor_name, feature_matrix)

        except Exception:
            logger.exception("[Policy] Error processing slide %s", slide_id)

    def _resolve_tissue_polygons(
        self,
        *,
        dataset: WSIDataset,
        wsi: WSI,
        slide_artifact: Any,
        slide_processor: SlideProcessorBase,
        segmentation_config: dict[str, Any],
    ) -> list[np.ndarray]:
        slide_id = wsi.slide

        # 1) Cache first 
        if tissue_io.tissue_exists(slide_artifact):
            polygons = tissue_io.read_tissue(slide_artifact)
            if polygons:
                return polygons

        # 2) If no cache yet: try external tissue (e.g. geojson), then cache it in H5
        external_roi_path = self._find_external_roi_file(dataset=dataset, slide_id=slide_id)
        if external_roi_path is not None:
            polygons = tissue_io.load_external_tissue_polygons(external_roi_path)
            if polygons:
                tissue_io.write_tissue(slide_artifact, polygons)
                return polygons
            logger.warning("[Policy] External ROI found but empty for slide %s: %s", slide_id, external_roi_path)

        # 3) Otherwise: compute with backend and cache in H5
        slide_processor.load_wsi(wsi)
        try:
            polygons = slide_processor.segment_tissue(wsi, config=segmentation_config)
        finally:
            slide_processor.close_wsi(wsi)

        tissue_io.write_tissue(slide_artifact, polygons)
        return polygons

    def _find_external_roi_file(self, *, dataset: WSIDataset, slide_id: str) -> Optional[Path]:
        roi_root = dataset.tissue_annotations_dir
        if roi_root is None or not roi_root.is_dir():
            return None

        suffixes = tissue_io.EXTERNAL_TISSUE_LOADERS.keys()
        candidates = []
        for suf in suffixes:
            candidates.extend(roi_root.glob(f"{slide_id}{suf}"))
        candidates = sorted(candidates)
        return candidates[0] if candidates else None

    def _build_run_configs(self, combo_cfg: ComboConfig) -> dict[str, Any]:
        return {
            "seg_config": self._build_seg_config(),
            "tile_config": self._build_tile_config(combo_cfg),
            "feat_config": self._build_feat_config(combo_cfg),
        }

    def _build_processor(self) -> SlideProcessorBase:
        # Ensure backend module is imported so decorators register it
        if not SLIDE_PROCESSORS.is_available(self.backend_name):
            import_module(f"pathbench.core.slide_processing.{self.backend_name}")

        ProcessorClass = SLIDE_PROCESSORS.get(self.backend_name)
        if not ProcessorClass:
            raise ValueError(f"Slide processing backend '{self.backend_name}' not found in registry.")

        slide_processor: SlideProcessorBase = ProcessorClass()
        logger.info("[Policy] Using backend '%s' -> %s", self.backend_name, slide_processor)
        return slide_processor

    def _build_seg_config(self) -> dict[str, Any]:
        return {
            "method": self.config.slide_processing.segmentation_method,
            "params": (self.config.slide_processing.qc_filters[0] if self.config.slide_processing.qc_filters else {}),
        }

    def _build_tile_config(self, combo_cfg: ComboConfig) -> dict[str, Any]:
        return {"tile_px": combo_cfg.tile_px, "tile_mpp": combo_cfg.tile_mpp, "params": {}}

    def _build_feat_config(self, combo_cfg: ComboConfig) -> dict[str, Any]:
        return {"model": combo_cfg.feature_extraction, "params": {}}

    def _bag_id(self, *, tile_px: int, tile_mpp: float) -> str:
        return f"{tile_px}px_{tile_mpp:g}mpp"

    def _ensure_coords_array(self, coords_array: Any) -> np.ndarray:
        coords_array = np.asarray(coords_array, dtype=np.int32)
        if coords_array.ndim != 2 or coords_array.shape[1] != 5:
            raise ValueError(f"[Policy] coords must have shape (N,5), got {coords_array.shape}.")
        return coords_array

    def _ensure_tiling_spec_dict(self, tiling_spec: Any, *, expected_tiling_spec: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(tiling_spec, dict):
            raise TypeError(f"[Policy] tiling_spec must be a dict, got {type(tiling_spec)}.")

        for key in ("tile_px", "tile_mpp", "stride_px", "coord_space"):
            if key not in tiling_spec:
                raise ValueError(f"[Policy] tiling_spec missing required key '{key}'.")

        if str(tiling_spec["coord_space"]) != "level0":
            raise ValueError(f"[Policy] tiling_spec.coord_space must be 'level0', got {tiling_spec['coord_space']!r}.")

        # enforce the expected tiling (source of truth is current run params)
        tiling_spec["tile_px"] = int(expected_tiling_spec["tile_px"])
        tiling_spec["tile_mpp"] = float(expected_tiling_spec["tile_mpp"])

        return tiling_spec

    def _ensure_feature_matrix(self, feature_matrix: Any, *, expected_rows: int) -> np.ndarray:
        feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
        if feature_matrix.ndim != 2:
            raise ValueError(f"[Policy] features must be 2D (N,D), got shape {feature_matrix.shape}.")
        if int(feature_matrix.shape[0]) != int(expected_rows):
            raise ValueError(
                f"[Policy] features rows must match coords rows: expected {expected_rows}, got {feature_matrix.shape[0]}."
            )
        return feature_matrix
