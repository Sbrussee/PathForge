from __future__ import annotations

from typing import Any, Optional, Tuple
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import hashlib
import json
import anndata as ad

from pathbench.policy.base import PolicyBase
from pathbench.utils.registries import SLIDE_PROCESSORS
from pathbench.core.slide_processing.base import SlideProcessorBase
from pathbench.core.experiments.base import ComboConfig, Experiment
from pathbench.core.datasets.wsi_dataset import WSI, WSIDataset

from pathbench.policy.utils import (
    load_features_pt,
    save_features_pt,
    load_json, 
    save_json, 
    load_tiles_npz, 
    save_tiles_npz, 
    load_tissues_geojson, 
    save_tissues_geojson
    )

logger = logging.getLogger(__name__)

class FeatureExtractionPolicy(PolicyBase):
    """
    Policy for extracting features from WSI slides.
    Depends on Experiment for project context (paths, datasets, combos).
    """

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
        
        for ds in self.datasets:
            ds.reset_active()

        logger.info("[Policy] Feature extraction DONE.")
        return {"status": "feature_extraction_done"}

    def execute_combo(self, combo_cfg: ComboConfig) -> None:
        """
        Run one benchmark combination across all datasets.

        Args:
            combo_cfg: Combination containing model and tiling parameters.
        """

        for ds in self.datasets:
            logger.info(
                "[Policy] Processing dataset '%s' (%d slides, used_for=%s)",
                ds.name,
                len(ds),
                ds.used_for,
            )
            self.execute_dataset(ds=ds, combo_cfg=combo_cfg)

    def execute_dataset(self, ds: WSIDataset, combo_cfg: ComboConfig) -> None:
        """
        Process all slides in a dataset for one benchmark combination.

        Args:
            ds: Dataset to process.
            combo_cfg: Combination containing model and tiling parameters.
        """

        processor = self._build_processor()

        seg_config = self._build_seg_config()
        tile_config = self._build_tile_config(combo_cfg)
        feat_config = self._build_feat_config(combo_cfg)

        tile_px = tile_config["tile_px"]
        tile_mpp = tile_config["tile_mpp"]
        tile_mpp_str = f"{tile_mpp:g}"

        roi_combo_name = self.config.slide_processing.segmentation_method  # TODO: This should adhere to the tissue rois found using lazyslide, should be optional.
        tiles_combo_name = f"{tile_px}px_{tile_mpp_str}mpp"
        feats_combo_name = f"{feat_config['model']}_{tile_px}px_{tile_mpp_str}mpp"

        combo_id = f"{roi_combo_name}__{feats_combo_name}"
        ds.set_active_combo(combo_id, reset=True)

        roi_root = ds.rois_dir

        tiles_root = Path(ds.tiles_dir) if ds.tiles_dir is not None else self.project_root / "tiles"
        tiles_dir = tiles_root / tiles_combo_name

        features_root = Path(ds.features_dir) if ds.features_dir is not None else self.project_root / "features"
        feats_dir = features_root / feats_combo_name

        self._ensure_dir_metadata(
            tiles_dir,
            expected_meta={
                "kind": "tiles_combo",
                "tile_config": dict(tile_config),
                "seg_config": dict(seg_config),
            },
        )

        self._ensure_dir_metadata(
            feats_dir,
            expected_meta={
                "kind": "features_combo",
                "tile_config": dict(tile_config),
                "feat_config": dict(feat_config),
            },
        )

        for wsi in tqdm(ds.samples, desc=f"Dataset: {ds.name}"):
            slide_id = wsi.slide

            tiles_path = tiles_dir / slide_id
            feats_path = feats_dir / slide_id
            tissues_path = None
            if roi_root is not None:
                tissues_path = Path(roi_root) / roi_combo_name / slide_id  # base (no suffix)

            try:
                # ---- load cached artifacts (no WSI open required) ----
                tiles_df, tile_spec = self._load_tiles(tiles_path)
                feats: Optional[ad.AnnData] = self._load_features(feats_path)
                tissues: Optional[list[np.ndarray]] = None

                # If cached tiles exist -> bind tiles path now
                if tiles_df is not None:
                    # _load_tiles checks .npz exists, but bind the actual file path you use
                    wsi.bind_active_tiles(combo_id, tiles_path.with_suffix(".npz"))

                # If cached features exist -> bind features paths now (pt + index)
                if feats is not None:
                    wsi.bind_active_features(combo_id, feats_path.with_suffix(".pt"))
                    logger.info("[Policy] Features already exist for slide %s, skipping.", slide_id)
                    continue

                # ---- open slide (always close in finally) ----
                processor.load_wsi(wsi)
                try:
                    need_tiling = (tiles_df is None) or (
                        not processor.validate_tile_spec(tile_spec, config=tile_config)
                    )

                    if need_tiling:
                        # Try load tissues if roi_root configured and file exists
                        if tissues_path is not None:
                            tp = tissues_path.with_suffix(".geojson")
                            if tp.exists():
                                tissues = self._load_tissues(tissues_path)
                                if tissues is not None:
                                    wsi.bind_active_tissues(combo_id, tp)

                        if tissues is None:
                            logger.info("[Policy] Segmenting tissue for slide %s", slide_id)
                            tissues = processor.segment_tissue(wsi, config=seg_config)

                            # Save + bind tissues only if roi_root configured
                            if tissues_path is not None:
                                self._save_tissues(tissues, tissues_path)
                                tp = tissues_path.with_suffix(".geojson")
                                if tp.exists():
                                    wsi.bind_active_tissues(combo_id, tp)

                        logger.info("[Policy] Tiling slide %s", slide_id)
                        tiles_df, tile_spec = processor.extract_patches(wsi, tissues, config=tile_config)

                        self._save_tiles(tiles_df, tile_spec, tiles_path)

                        # After saving, bind tiles path (only if file exists)
                        npz = tiles_path.with_suffix(".npz")
                        if npz.exists():
                            wsi.bind_active_tiles(combo_id, npz)

                    logger.info("[Policy] Extracting features for slide %s", slide_id)
                    feats = processor.extract_features(
                        wsi, tiles_df, tile_spec, config={**feat_config, **tile_config}
                    )

                finally:
                    processor.close_wsi(wsi)

                feats = self._ensure_features_schema(feats, tiles_df)
                self._save_features(feats, feats_path)

                # After saving, bind features path(s) only if they exist
                pt = feats_path.with_suffix(".pt")
                idx = feats_path.with_suffix(".index.npz")
                if pt.exists() and idx.exists():
                    wsi.bind_active_features(combo_id, pt)

            except Exception:
                logger.exception("[Policy] Error processing slide %s", slide_id)

    def _load_tissues(self, path: Path) -> list[np.ndarray]:
        """
        Load tissue polygons from disk.

        Args:
            path: Path to the tissue file.

        Returns:
            List of polygons as arrays of shape (N, 2) in pixel coordinates.
        """
        p = path if path.suffix else path.with_suffix(".geojson")
        if not p.exists():
            return None
        return load_tissues_geojson(p)

    def _save_tissues(self, tissues: list[np.ndarray], path: Path) -> None:
        """
        Save tissue polygons to disk.

        Args:
            tissues: List of polygons as arrays of shape (N, 2).
            path: Output path.
        """
        save_tissues_geojson(tissues, path)

    def _load_tiles(self, path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Load tiles + tile_spec from disk.

        Args:
            path: Base path (directory or file path). If no suffix is provided,
                '.npz' is assumed.

        Returns:
            (tiles_df, tile_spec) where tiles_df has columns: 'tile_id','x','y'
            and tile_spec is a JSON string. If not found: (None, None).
        """
        p = path if path.suffix else path.with_suffix(".npz")
        if not p.exists():
            return None, None
        return load_tiles_npz(p)


    def _save_tiles(self, tiles_df: pd.DataFrame, tile_spec: str, path: Path) -> None:
        """
        Save tiles + tile_spec to disk.

        Args:
            tiles_df: DataFrame with required columns: 'tile_id', 'x', 'y'.
            tile_spec: JSON string (keys may vary).
            path: Output base path (directory or file path). If no suffix is provided,
                '.npz' is used.
        """
        p = path if path.suffix else path.with_suffix(".npz")
        save_tiles_npz(tiles_df, tile_spec, p)

    def _load_features(self, path: Path) -> ad.AnnData:
        """
        Load features from an .pt file.

        Args:
            path: Path to the .pt file.

        Returns:
            AnnData containing features with obs['tile_id'].
        """
        pt = path if path.suffix else path.with_suffix(".pt")
        idx = pt.with_suffix(".index.npz")
        if not pt.exists() or not idx.exists():
            return None
        return load_features_pt(path)

    def _save_features(self, feats: ad.AnnData, path: Path) -> None:
        """
        Save features as an .pt file.

        Args:
            feats: AnnData containing features (expects obs['tile_id']).
            path: Output path.
        """
        save_features_pt(feats, path)

    def _build_processor(self) -> SlideProcessorBase:
        """
        Instantiate the configured slide processing backend.

        Returns:
            Backend processor instance.

        Raises:
            ValueError: If the backend is not registered.
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

        Returns:
            Dict with segmentation method and parameters.
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

        Args:
            combo_cfg: Combination providing tile_px and tile_mpp.

        Returns:
            Dict containing tiling parameters.
        """
        return {
            "tile_px": combo_cfg.tile_px,
            "tile_mpp": combo_cfg.tile_mpp,
            "params": {},
        }

    def _build_feat_config(self, combo_cfg: ComboConfig) -> dict[str, Any]:
        """
        Build feature extraction config for the backend.

        Args:
            combo_cfg: Combination providing feature extraction model.

        Returns:
            Dict containing model and parameters.
        """
        return {
            "model": combo_cfg.feature_extraction,
            "params": {},
        }

    def _ensure_dir_metadata(self, dir_path: Path, expected_meta: dict) -> None:
        """
        Ensure meta.json exists in dir_path and matches expected.

        Args:
            dir_path: Output directory for a parameter combination.
            expected: Expected metadata content.

        Raises:
            ValueError: If existing metadata does not match expected.
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

        Requires obs['tile_id']; adds obsm['spatial'] from tiles_df if missing.

        Args:
            feats: Feature AnnData.
            tiles_df: Tiles table with 'tile_id', 'x', 'y'.

        Returns:
            Updated AnnData with guaranteed obsm['spatial'].
        """
        # Require tile_id
        if "tile_id" not in feats.obs.columns:
            raise ValueError("[Policy] Features AnnData must have obs['tile_id'].")

        # Make spatial mandatory (top-left x,y), derived from tiles_df
        if "spatial" not in feats.obsm:
            if not {"tile_id", "x", "y"}.issubset(set(tiles_df.columns)):
                raise ValueError("[Policy] Tiles must contain columns: tile_id, x, y to build spatial.")

            tiles_key = tiles_df.set_index("tile_id")[["x", "y"]]
            feat_tile_ids = feats.obs["tile_id"].to_numpy()

            try:
                xy = tiles_key.loc[feat_tile_ids].to_numpy(dtype=np.float32)
            except KeyError as e:
                raise ValueError("[Policy] Features tile_id does not match tiles tile_id.") from e

            feats.obsm["spatial"] = xy

        return feats
