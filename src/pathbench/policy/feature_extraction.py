from __future__ import annotations

from typing import Any, Optional
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import anndata as ad

from pathbench.policy.base import PolicyBase
from pathbench.utils.registries import SLIDE_PROCESSORS
from pathbench.core.slide_processing.base import SlideProcessorBase
from pathbench.core.experiments.base import ComboConfig, Experiment
from pathbench.core.datasets.wsi_dataset import WSI, WSIDataset

from pathbench.core.io.features import load_features, save_features
from pathbench.core.io.tiles import load_tiles, save_tiles
from pathbench.core.io.tissues import load_tissues, save_tissues
from pathbench.core.io.utils import load_json, save_json

from pathbench.core.io.base import detect_artifact_path
from pathbench.core.io.tiles import SUPPORTED_SUFFIXES as TILES_SUFFIXES, DEFAULT_SUFFIX as TILES_DEFAULT
from pathbench.core.io.features import SUPPORTED_SUFFIXES as FEATS_SUFFIXES, DEFAULT_SUFFIX as FEATS_DEFAULT
from pathbench.core.io.tissues import SUPPORTED_SUFFIXES as TISSUES_SUFFIXES, DEFAULT_SUFFIX as TISSUES_DEFAULT

logger = logging.getLogger(__name__)


class FeatureExtractionPolicy(PolicyBase):
    """
    Policy for extracting features from WSI slides.
    Depends on Experiment for project context (paths, datasets, combos).
    """

    # ---- Initialization ----
    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

        # Pre-compute reusable context
        self.datasets: list[WSIDataset] = self.experiment.build_datasets()
        self.combos: list[ComboConfig] = self.experiment.build_combinations(
            ["feature_extraction", "tile_px", "tile_mpp"]
        )

        self.project_root = Path(experiment.project_root)
        self.features_dir = self.project_root / "features"
        self.config = experiment.cfg

        self.backend_name = self.config.slide_processing.backend

    # ---- Public Policy Entrypoints ----
    def execute(self) -> dict[str, Any]:
        """
        Run feature extraction for all benchmark combinations across all datasets.

        Returns:
            Status dictionary indicating completion.
        """
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
            self.execute_combo(combo_cfg)

        for dataset in self.datasets:
            dataset.reset_active()

        logger.info("[Policy] Feature extraction DONE.")
        return {"status": "feature_extraction_done"}

    def execute_combo(self, combo_cfg: ComboConfig) -> None:
        """
        Run one benchmark combination across all datasets.

        Args:
            combo_cfg: Combination containing model and tiling parameters.
        """
        for dataset in self.datasets:
            logger.info(
                "[Policy] Processing dataset '%s' (%d slides, used_for=%s)",
                dataset.name,
                len(dataset),
                dataset.used_for,
            )
            self.execute_dataset(dataset=dataset, combo_cfg=combo_cfg)

    def execute_dataset(self, dataset: WSIDataset, combo_cfg: ComboConfig) -> None:
        """
        Process all slides in a dataset for one benchmark combination.

        Args:
            dataset: Dataset to process.
            combo_cfg: Combination containing model and tiling parameters.
        """
        processor = self._build_processor()

        run_configs = self._build_run_configs(combo_cfg)
        combo_id, run_paths = self._prepare_paths_for_combo(dataset=dataset, run_configs=run_configs)

        for wsi in tqdm(dataset.samples, desc=f"Dataset: {dataset.name}"):
            self._execute_wsi(
                wsi=wsi,
                processor=processor,
                combo_id=combo_id,
                run_configs=run_configs,
                run_paths=run_paths,
            )

    # ---- Single-slide Wrapper ----
    def process_slide(
        self,
        dataset: WSIDataset,
        wsi: WSI,
        combo_cfg: ComboConfig,
    ) -> None:
        """
        Process a single slide for a single benchmark combination.

        Args:
            dataset: Dataset that determines output roots (tiles/features/roi).
            wsi: Slide to process.
            combo_cfg: Combination containing model and tiling parameters.
            processor: Slide processing backend instance (should be reused across slides).
        """
        processor = self._build_processor()

        run_configs = self._build_run_configs(combo_cfg)
        combo_id, run_paths = self._prepare_paths_for_combo(dataset=dataset, run_configs=run_configs)

        self._execute_wsi(
            wsi=wsi,
            processor=processor,
            combo_id=combo_id,
            run_configs=run_configs,
            run_paths=run_paths,
        )

    # ---- Core Slide Execution ----
    def _execute_wsi(
        self,
        *,
        wsi: WSI,
        processor: SlideProcessorBase,
        combo_id: str,
        run_configs: dict[str, Any],
        run_paths: dict[str, Optional[Path]],
    ) -> None:
        """
        Core implementation for processing exactly one slide for exactly one combo.
        """
        slide_id = wsi.slide

        tiles_directory = run_paths["tiles_dir"]
        features_directory = run_paths["features_dir"]
        roi_directory = run_paths["roi_dir"]

        if tiles_directory is None or features_directory is None:
            raise RuntimeError("[Policy] run_paths must contain 'tiles_dir' and 'features_dir'.")

        tiles_path = tiles_directory / slide_id
        features_path = features_directory / slide_id
        tissues_path = roi_directory / slide_id if roi_directory is not None else None

        segmentation_config = run_configs["seg_config"]
        tile_config = run_configs["tile_config"]
        feature_config = run_configs["feat_config"]

        try:
            # ---- Load cached artifacts (no WSI open required) ----
            tiles_dataframe, tile_specification = load_tiles(tiles_path)
            features_anndata: Optional[ad.AnnData] = load_features(features_path)
            tissues: Optional[list[np.ndarray]] = None

            # ---- Bind cached tiles if present ----
            if tiles_dataframe is not None:
                detected_tiles_path = detect_artifact_path(
                    tiles_path,
                    allowed_suffixes=TILES_SUFFIXES,
                    kind="tiles",
                    prefer_suffixes=(TILES_DEFAULT,),
                )
                if detected_tiles_path is not None:
                    wsi.bind_active_tiles(combo_id, detected_tiles_path)

            # ---- Bind cached features if present ----
            if features_anndata is not None:
                detected_features_path = detect_artifact_path(
                    features_path,
                    allowed_suffixes=FEATS_SUFFIXES,
                    kind="features",
                    prefer_suffixes=(FEATS_DEFAULT,),
                )
                if detected_features_path is not None:
                    wsi.bind_active_features(combo_id, detected_features_path)

                logger.info("[Policy] Features already exist for slide %s, skipping.", slide_id)
                return

            # ---- Open slide (always close in finally) ----
            processor.load_wsi(wsi)
            try:
                need_tiling = (tiles_dataframe is None) or (
                    not processor.validate_tile_spec(tile_specification, config=tile_config)
                )

                if need_tiling:
                    # ---- Load cached tissues if present ----
                    if tissues_path is not None:
                        detected_tissues_path = detect_artifact_path(
                            tissues_path,
                            allowed_suffixes=TISSUES_SUFFIXES,
                            kind="tissues",
                            prefer_suffixes=(TISSUES_DEFAULT,),
                        )
                        if detected_tissues_path is not None:
                            tissues = load_tissues(tissues_path)
                            if tissues is not None:
                                wsi.bind_active_tissues(combo_id, detected_tissues_path)

                    # ---- Segment tissues if needed ----
                    if tissues is None:
                        logger.info("[Policy] Segmenting tissue for slide %s", slide_id)
                        tissues = processor.segment_tissue(wsi, config=segmentation_config)

                        if tissues_path is not None:
                            saved_tissues_path = save_tissues(tissues, tissues_path)
                            if saved_tissues_path.exists():
                                wsi.bind_active_tissues(combo_id, saved_tissues_path)

                    # ---- Extract tiles ----
                    logger.info("[Policy] Tiling slide %s", slide_id)
                    tiles_dataframe, tile_specification = processor.extract_patches(
                        wsi, tissues, config=tile_config
                    )

                    saved_tiles_path = save_tiles(tiles_dataframe, tile_specification, tiles_path)
                    if saved_tiles_path.exists():
                        wsi.bind_active_tiles(combo_id, saved_tiles_path)

                # ---- Extract features ----
                logger.info("[Policy] Extracting features for slide %s", slide_id)
                features_anndata = processor.extract_features(
                    wsi,
                    tiles_dataframe,
                    tile_specification,
                    config={**feature_config, **tile_config},
                )

            finally:
                processor.close_wsi(wsi)

            # ---- Validate + save features ----
            features_anndata = self._ensure_features_schema(features_anndata, tiles_dataframe)
            saved_features_path = save_features(features_anndata, features_path)

            features_index_path = saved_features_path.with_suffix(".index.npz")
            if saved_features_path.exists() and features_index_path.exists():
                wsi.bind_active_features(combo_id, saved_features_path)

        except Exception:
            logger.exception("[Policy] Error processing slide %s", slide_id)

    # ---- Run Configuration ----
    def _build_run_configs(self, combo_cfg: ComboConfig) -> dict[str, Any]:
        """
        Build configs needed to run one combination.
        """
        segmentation_config = self._build_seg_config()
        tile_config = self._build_tile_config(combo_cfg)
        feature_config = self._build_feat_config(combo_cfg)

        return {
            "seg_config": segmentation_config,
            "tile_config": tile_config,
            "feat_config": feature_config,
        }

    # ---- Output Paths and Metadata ----
    def _prepare_paths_for_combo(
        self,
        *,
        dataset: WSIDataset,
        run_configs: dict[str, Any],
    ) -> tuple[str, dict[str, Optional[Path]]]:
        """
        Prepare combo-dependent output directories and meta.json files.

        Returns:
            (combo_id, run_paths)
            where run_paths contains: tiles_dir, features_dir, roi_dir
        """
        tile_config = run_configs["tile_config"]
        feature_config = run_configs["feat_config"]
        segmentation_config = run_configs["seg_config"]

        tile_size_pixels = tile_config["tile_px"]
        tile_mpp_value = tile_config["tile_mpp"]
        tile_mpp_string = f"{tile_mpp_value:g}"

        roi_name = self.config.slide_processing.segmentation_method
        tiles_directory_name = f"{tile_size_pixels}px_{tile_mpp_string}mpp"
        features_directory_name = f"{feature_config['model']}_{tile_size_pixels}px_{tile_mpp_string}mpp"

        combo_id = f"{roi_name}__{features_directory_name}"

        dataset.set_active_combo(combo_id, reset=True)

        tiles_root_directory = (
            Path(dataset.tiles_dir) if dataset.tiles_dir is not None else self.project_root / "tiles"
        )
        features_root_directory = (
            Path(dataset.features_dir) if dataset.features_dir is not None else self.project_root / "features"
        )

        tiles_directory = tiles_root_directory / tiles_directory_name
        features_directory = features_root_directory / features_directory_name

        roi_directory: Optional[Path] = None
        if dataset.rois_dir is not None:
            roi_directory = Path(dataset.rois_dir) / roi_name

        self._ensure_dir_metadata(
            tiles_directory,
            expected_meta={
                "kind": "tiles_combo",
                "tile_config": dict(tile_config),
                "seg_config": dict(segmentation_config),
            },
        )

        self._ensure_dir_metadata(
            features_directory,
            expected_meta={
                "kind": "features_combo",
                "tile_config": dict(tile_config),
                "feat_config": dict(feature_config),
            },
        )

        run_paths: dict[str, Optional[Path]] = {
            "tiles_dir": tiles_directory,
            "features_dir": features_directory,
            "roi_dir": roi_directory,
        }
        return combo_id, run_paths

    # ---- Backend and Config Builders ----
    def _build_processor(self) -> SlideProcessorBase:
        """
        Instantiate the configured slide processing backend.
        """
        ProcessorClass = SLIDE_PROCESSORS.get(self.backend_name)
        if not ProcessorClass:
            raise ValueError(f"Slide processing backend '{self.backend_name}' not found in registry.")

        processor: SlideProcessorBase = ProcessorClass()
        logger.info("[Policy] Using backend '%s' -> %s", self.backend_name, processor)
        return processor

    def _build_seg_config(self) -> dict[str, Any]:
        """
        Build segmentation config for the backend.
        """
        return {
            "method": self.config.slide_processing.segmentation_method,
            "params": (
                self.config.slide_processing.qc_filters[0]
                if self.config.slide_processing.qc_filters
                else {}
            ),
        }

    def _build_tile_config(self, combo_cfg: ComboConfig) -> dict[str, Any]:
        """
        Build tiling config for the backend.
        """
        return {
            "tile_px": combo_cfg.tile_px,
            "tile_mpp": combo_cfg.tile_mpp,
            "params": {},
        }

    def _build_feat_config(self, combo_cfg: ComboConfig) -> dict[str, Any]:
        """
        Build feature extraction config for the backend.
        """
        return {
            "model": combo_cfg.feature_extraction,
            "params": {},
        }

    # ---- Metadata and Schema ----
    def _ensure_dir_metadata(self, dir_path: Path, expected_meta: dict) -> None:
        """
        Ensure meta.json exists in dir_path and matches expected.
        """
        dir_path.mkdir(parents=True, exist_ok=True)
        meta_path = dir_path / "meta.json"

        if meta_path.exists():
            existing = load_json(meta_path)
            if existing != expected_meta:
                raise ValueError(
                    f"Metadata mismatch for {dir_path}.\n"
                    f"Expected: {expected_meta}\n"
                    f"Found:    {existing}"
                )
        else:
            save_json(expected_meta, meta_path)

    def _ensure_features_schema(self, feats: ad.AnnData, tiles_df: pd.DataFrame) -> ad.AnnData:
        """
        Ensure required feature fields exist and align with tiles.
        """
        if "tile_id" not in feats.obs.columns:
            raise ValueError("[Policy] Features AnnData must have obs['tile_id'].")

        if "spatial" not in feats.obsm:
            if not {"tile_id", "x", "y"}.issubset(set(tiles_df.columns)):
                raise ValueError("[Policy] Tiles must contain columns: tile_id, x, y to build spatial.")

            tiles_key = tiles_df.set_index("tile_id")[["x", "y"]]
            feature_tile_ids = feats.obs["tile_id"].to_numpy()

            try:
                xy = tiles_key.loc[feature_tile_ids].to_numpy(dtype=np.float32)
            except KeyError as e:
                raise ValueError("[Policy] Features tile_id does not match tiles tile_id.") from e

            feats.obsm["spatial"] = xy

        return feats
