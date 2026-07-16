from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from importlib import import_module

from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm

from pathforge.policy.base import PolicyBase
from pathforge.core.datasets.factory import build_wsi_datasets
from pathforge.core.experiments.base import Experiment
from pathforge.core.experiments.combinations import ComboConfig, build_combinations
from pathforge.core.experiments.combo_ids import (
    build_feature_name,
    build_tiling_id,
)
from pathforge.core.datasets.wsi_dataset import WSI, WSIDataset
from pathforge.core.slide_processing.base import SlideProcessorBase
from pathforge.utils.registries import SLIDE_PROCESSORS

from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.io.slide_artifacts.atomic import (
    atomic_slide_artifact_write,
    ensure_artifact_readable_or_quarantine,
)
from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts import features as features_io
from pathforge.core.io.slide_artifacts import thumbnail as thumbnail_io
from pathforge.core.io.slide_artifacts import tissue as tissue_io

from pathforge.core.visualization.thumbnail import render_thumbnail_image
from pathforge.core.visualization.tiles_overview import render_tiles_overview
from pathforge.core.reports.tiles_report_pdf import collect_tiles_overview_entries, create_tiles_report_pdf

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingArtifactWrites:
    tissue_polygons: Optional[tissue_io.TissuePolygons] = None
    thumbnail_image_bytes: Optional[bytes] = None
    thumbnail_spec: Optional[dict[str, Any]] = None
    runtime_thumbnail_image: Optional[Any] = None
    runtime_thumbnail_downscale_x: Optional[float] = None
    runtime_thumbnail_downscale_y: Optional[float] = None
    coords_array: Optional[np.ndarray] = None
    tiling_spec: Optional[dict[str, Any]] = None
    tiles_overview_bytes: Optional[bytes] = None
    feature_matrix: Optional[np.ndarray] = None

    def has_updates(self) -> bool:
        return any(
            value is not None
            for value in (
                self.tissue_polygons,
                self.thumbnail_image_bytes,
                self.thumbnail_spec,
                self.coords_array,
                self.tiling_spec,
                self.tiles_overview_bytes,
                self.feature_matrix,
            )
        )


class FeatureExtractionPolicy(PolicyBase):
    """Extract tile features into per-slide H5 artifacts."""

    combo_keys = [
        "feature_extraction",
        "tile_px",
        "tile_mpp",
        "color_norm",
    ]

    def __init__(
        self,
        experiment: Experiment,
    ):
        super().__init__(experiment)
        self.config = experiment.cfg
        self.backend_name = self.config.slide_processing.backend

    def build_combos(self) -> list[ComboConfig]:
        return build_combinations(
            cfg=self.experiment.cfg,
            keys=self.combo_keys,
        )

    def execute(
        self,
        *,
        datasets: list[WSIDataset] | None = None,
        combos: list[ComboConfig] | None = None,
    ) -> dict[str, Any]:
        if datasets is None:
            datasets = build_wsi_datasets(
                cfg=self.experiment.cfg,
                annotations_df=self.experiment.load_annotations(),
            )
        if combos is None:
            combos = self.build_combos()

        logger.info("[Policy] Number of parameter combos: %d", len(combos))

        for combo_index, combo_cfg in enumerate(combos, start=1):
            logger.info(
                "[Policy] === Combo %d/%d: extractor=%s, tile_px=%s, tile_mpp=%s ===",
                combo_index,
                len(combos),
                combo_cfg.feature_extraction,
                combo_cfg.tile_px,
                combo_cfg.tile_mpp,
            )
            self.execute_combo(combo_cfg=combo_cfg, datasets=datasets)

        # ---- Generate tile reports once after all combos (dedupe on bag_id) ----
        if bool(self.config.experiment.report):
            self._generate_tiles_reports_after_extraction(
                datasets=datasets,
                combos=combos,
            )

        logger.info("[Policy] Feature extraction DONE.")
        return {"status": "feature_extraction_done"}

    def execute_combo(
        self,
        *,
        combo_cfg: ComboConfig,
        datasets: list[WSIDataset],
    ) -> None:
        for dataset in datasets:
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
                combo_cfg=combo_cfg,
                slide_processor=slide_processor,
                run_configs=run_configs,
            )

    def process_slide(self, dataset: WSIDataset, wsi: WSI, combo_cfg: ComboConfig) -> None:
        slide_processor = self._build_processor()
        run_configs = self._build_run_configs(combo_cfg)
        self._execute_wsi(
            dataset=dataset,
            wsi=wsi,
            combo_cfg=combo_cfg,
            slide_processor=slide_processor,
            run_configs=run_configs,
        )

    def _execute_wsi(
        self,
        *,
        dataset: WSIDataset,
        wsi: WSI,
        combo_cfg: ComboConfig,
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
        feature_name = build_feature_name(combo_cfg)
        tiling_combo_cfg = ComboConfig(tile_px=tile_px, tile_mpp=tile_mpp)
        tiling_id = build_tiling_id(tiling_combo_cfg)

        report_enabled = bool(self.config.experiment.report)
        thumbnail_enabled = bool(getattr(self.config.experiment, "thumbnail", False))

        try:
            slide_processor.load_wsi(wsi)
            _ = slide_processor.get_base_mpp(wsi)
        except Exception:
            logger.warning(
                "[Policy] Skipping slide %s because no valid base MPP is available.",
                slide_id,
            )
            try:
                slide_processor.close_wsi(wsi)
            except Exception:
                pass
            return

        expected_tiling_spec = {
            "tile_px": tile_px,
            "tile_mpp": tile_mpp,
            "stride_px": tile_px,
            "coord_space": "level0"
        }

        try:
            ensure_artifact_readable_or_quarantine(artifact_path)

            pending_writes = _PendingArtifactWrites()
            coords_array, tiling_spec = self._resolve_features(
                dataset=dataset,
                wsi=wsi,
                slide_id=slide_id,
                artifact_path=artifact_path,
                tiling_id=tiling_id,
                extractor_name=feature_name,
                expected_tiling_spec=expected_tiling_spec,
                slide_processor=slide_processor,
                segmentation_config=segmentation_config,
                tiling_config=tiling_config,
                feature_config=feature_config,
                report_enabled=report_enabled,
                thumbnail_enabled=thumbnail_enabled,
                pending_writes=pending_writes,
            )

            if coords_array is None or tiling_spec is None:
                return

            if thumbnail_enabled:
                self._resolve_thumbnail(
                    artifact_path=artifact_path,
                    wsi=wsi,
                    slide_processor=slide_processor,
                    pending_writes=pending_writes,
                )

            # ---- tiles overview -------------------------------------------------
            if report_enabled:
                self._resolve_tiles_overview(
                    dataset=dataset,
                    wsi=wsi,
                    artifact_path=artifact_path,
                    slide_id=slide_id,
                    tiling_id=tiling_id,
                    expected_tiling_spec=expected_tiling_spec,
                    slide_processor=slide_processor,
                    segmentation_config=segmentation_config,
                    tiling_config=tiling_config,
                    coords_array=coords_array,
                    tiling_spec=tiling_spec,
                    pending_writes=pending_writes,
                )

            # ---- atomic write ---------------------------------------------------
            if pending_writes.has_updates():
                with atomic_slide_artifact_write(artifact_path) as slide_artifact:
                    self._write_pending_artifact_updates(
                        slide_artifact=slide_artifact,
                        tiling_id=tiling_id,
                        extractor_name=feature_name,
                        pending_writes=pending_writes,
                    )
        except Exception:
            logger.exception("[Policy] Error processing slide %s", slide_id)
        finally:
            slide_processor.close_wsi(wsi)

    def _resolve_features(
        self,
        *,
        dataset: WSIDataset,
        wsi: WSI,
        slide_id: str,
        artifact_path: Path,
        tiling_id: str,
        extractor_name: str,
        expected_tiling_spec: dict[str, Any],
        slide_processor: SlideProcessorBase,
        segmentation_config: dict[str, Any],
        tiling_config: dict[str, Any],
        feature_config: dict[str, Any],
        report_enabled: bool,
        thumbnail_enabled: bool,
        pending_writes: _PendingArtifactWrites,
    ) -> tuple[Optional[np.ndarray], Optional[dict[str, Any]]]:
        # ---- features ---------------------------------------------------------
        if artifact_path.is_file():
            try:
                with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                    coords_row_count = tiles_io.coords_num_rows(slide_artifact, tiling_id)
                    features_ready = features_io.features_exist(
                        slide_artifact,
                        bag_id=tiling_id,
                        extractor_name=extractor_name,
                        expected_rows=coords_row_count,
                    )
                    thumbnail_ready = (not thumbnail_enabled) or (
                        thumbnail_io.thumbnail_image_exists(slide_artifact)
                        and thumbnail_io.thumbnail_spec_exists(slide_artifact)
                    )
                    overview_ready = (not report_enabled) or tiles_io.tiles_overview_exists(
                        slide_artifact,
                        tiling_id,
                    )

                    if features_ready and overview_ready and thumbnail_ready:
                        logger.info("[Policy] Features exist for slide %s (%s/%s), skipping.", slide_id, tiling_id, extractor_name)
                        return None, None

                    if features_ready and tiles_io.coords_exist(
                        slide_artifact,
                        tiling_id,
                    ) and tiles_io.tiling_spec_matches(
                        slide_artifact,
                        bag_id=tiling_id,
                        expected_tiling_spec=expected_tiling_spec,
                    ):
                        return (
                            tiles_io.read_coords(slide_artifact, tiling_id),
                            tiles_io.read_tiling_spec(slide_artifact, tiling_id),
                        )
            except Exception:
                logger.warning(
                    "[Policy] Live artifact read check failed for %s; rebuilding via atomic write path.",
                    artifact_path,
                )

        # ---- tiles ------------------------------------------------------------
        coords_array, tiling_spec = self._resolve_tiles(
            dataset=dataset,
            wsi=wsi,
            artifact_path=artifact_path,
            tiling_id=tiling_id,
            expected_tiling_spec=expected_tiling_spec,
            slide_processor=slide_processor,
            segmentation_config=segmentation_config,
            tiling_config=tiling_config,
            pending_writes=pending_writes,
        )

        if coords_array is None or tiling_spec is None:
            raise RuntimeError("[Policy] Internal error: coords_array/tiling_spec not resolved.")

        feature_matrix = slide_processor.extract_features(
            wsi,
            coords_array,
            tiling_spec,
            config={**feature_config, **tiling_config},
        )

        pending_writes.feature_matrix = self._ensure_feature_matrix(
            feature_matrix,
            expected_rows=int(coords_array.shape[0]),
        )
        return coords_array, tiling_spec

    def _resolve_tiles(
        self,
        *,
        dataset: WSIDataset,
        wsi: WSI,
        artifact_path: Path,
        tiling_id: str,
        expected_tiling_spec: dict[str, Any],
        slide_processor: SlideProcessorBase,
        segmentation_config: dict[str, Any],
        tiling_config: dict[str, Any],
        pending_writes: _PendingArtifactWrites,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # ---- tiles ------------------------------------------------------------
        if artifact_path.is_file():
            try:
                with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                    if tiles_io.coords_exist(slide_artifact, tiling_id) and tiles_io.tiling_spec_matches(
                        slide_artifact,
                        bag_id=tiling_id,
                        expected_tiling_spec=expected_tiling_spec,
                    ):
                        return (
                            tiles_io.read_coords(slide_artifact, tiling_id),
                            tiles_io.read_tiling_spec(slide_artifact, tiling_id),
                        )
            except Exception:
                logger.warning(
                    "[Policy] Live artifact read check failed for %s; rebuilding via atomic write path.",
                    artifact_path,
                )

        # ---- tissue -----------------------------------------------------------
        tissue_polygons = self._resolve_tissue(
            dataset=dataset,
            wsi=wsi,
            artifact_path=artifact_path,
            slide_processor=slide_processor,
            segmentation_config=segmentation_config,
            pending_writes=pending_writes,
        )

        coords_array, tiling_spec = slide_processor.extract_patches(
            wsi,
            tissue_polygons,
            config=tiling_config,
        )

        coords_array = self._ensure_coords_array(coords_array)
        tiling_spec = self._ensure_tiling_spec_dict(tiling_spec, expected_tiling_spec=expected_tiling_spec)
        pending_writes.coords_array = coords_array
        pending_writes.tiling_spec = tiling_spec
        return coords_array, tiling_spec

    def _resolve_tissue(
        self,
        *,
        dataset: WSIDataset,
        wsi: WSI,
        artifact_path: Path,
        slide_processor: SlideProcessorBase,
        segmentation_config: dict[str, Any],
        pending_writes: _PendingArtifactWrites,
    ) -> tissue_io.TissuePolygons:
        # ---- tissue -----------------------------------------------------------
        if artifact_path.is_file():
            try:
                with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                    if tissue_io.tissue_exists(slide_artifact):
                        polygons = tissue_io.read_tissue(slide_artifact)
                        if polygons:
                            return polygons
            except Exception:
                logger.warning(
                    "[Policy] Live artifact read check failed for %s; rebuilding via atomic write path.",
                    artifact_path,
                )

        external_roi_path = self._find_external_roi_file(dataset=dataset, slide_id=wsi.slide)
        if external_roi_path is not None:
            polygons = tissue_io.load_external_tissue_polygons(external_roi_path)
            if polygons:
                pending_writes.tissue_polygons = polygons
                return polygons
            logger.warning("[Policy] External ROI found but empty for slide %s: %s", wsi.slide, external_roi_path)

        polygons = slide_processor.segment_tissue(wsi, config=segmentation_config)

        pending_writes.tissue_polygons = polygons
        return polygons

    def _resolve_thumbnail(
        self,
        *,
        artifact_path: Path,
        wsi: WSI,
        slide_processor: SlideProcessorBase,
        pending_writes: _PendingArtifactWrites,
    ) -> None:
        if artifact_path.is_file():
            try:
                with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                    if thumbnail_io.thumbnail_image_exists(
                        slide_artifact
                    ) and thumbnail_io.thumbnail_spec_exists(slide_artifact):
                        return
            except Exception:
                logger.warning(
                    "[Policy] Live artifact read check failed for %s; rebuilding via atomic write path.",
                    artifact_path,
                )

        thumbnail_image, downscale_x, downscale_y = slide_processor.get_thumbnail(
            wsi,
            level=-1,
        )
        pending_writes.thumbnail_image_bytes = render_thumbnail_image(
            thumbnail_image=thumbnail_image,
        )
        pending_writes.runtime_thumbnail_image = thumbnail_image
        pending_writes.runtime_thumbnail_downscale_x = float(downscale_x)
        pending_writes.runtime_thumbnail_downscale_y = float(downscale_y)
        pending_writes.thumbnail_spec = {
            "image_format": "jpeg",
            "coord_space": "level0",
            "thumbnail_level": -1,
            "downscale_x": float(downscale_x),
            "downscale_y": float(downscale_y),
        }

    def _resolve_tiles_overview(
        self,
        *,
        dataset: WSIDataset,
        wsi: WSI,
        artifact_path: Path,
        slide_id: str,
        tiling_id: str,
        expected_tiling_spec: dict[str, Any],
        slide_processor: SlideProcessorBase,
        segmentation_config: dict[str, Any],
        tiling_config: dict[str, Any],
        coords_array: Optional[np.ndarray],
        tiling_spec: Optional[dict[str, Any]],
        pending_writes: _PendingArtifactWrites,
    ) -> None:
        if artifact_path.is_file():
            try:
                with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                    if tiles_io.tiles_overview_exists(slide_artifact, tiling_id):
                        return
            except Exception:
                logger.warning(
                    "[Policy] Live artifact read check failed for %s; rebuilding via atomic write path.",
                    artifact_path,
                )

        if coords_array is None or tiling_spec is None:
            coords_array, tiling_spec = self._resolve_tiles(
                dataset=dataset,
                wsi=wsi,
                artifact_path=artifact_path,
                tiling_id=tiling_id,
                expected_tiling_spec=expected_tiling_spec,
                slide_processor=slide_processor,
                segmentation_config=segmentation_config,
                tiling_config=tiling_config,
                pending_writes=pending_writes,
            )

        if (
            pending_writes.runtime_thumbnail_image is not None
            and pending_writes.runtime_thumbnail_downscale_x is not None
            and pending_writes.runtime_thumbnail_downscale_y is not None
        ):
            thumbnail_image = pending_writes.runtime_thumbnail_image
            downscale_x = pending_writes.runtime_thumbnail_downscale_x
            downscale_y = pending_writes.runtime_thumbnail_downscale_y
        else:
            thumbnail_image, downscale_x, downscale_y = slide_processor.get_thumbnail(
                wsi,
                level=-1,
            )
        base_mpp = slide_processor.get_base_mpp(wsi)

        overview_result = render_tiles_overview(
            thumbnail_image=thumbnail_image,
            coords_array=coords_array,
            downscale_x=downscale_x,
            downscale_y=downscale_y,
            slide_id=slide_id,
            tiling_spec=tiling_spec,
            base_mpp=base_mpp,
        )
        pending_writes.tiles_overview_bytes = overview_result.image_bytes
        if pending_writes.tiling_spec is not None:
            pending_writes.tiling_spec["tiles_overview_downscale_x"] = overview_result.downscale_x
            pending_writes.tiling_spec["tiles_overview_downscale_y"] = overview_result.downscale_y

    def _write_pending_artifact_updates(
        self,
        *,
        slide_artifact: FileHandleH5,
        tiling_id: str,
        extractor_name: str,
        pending_writes: _PendingArtifactWrites,
    ) -> None:
        if pending_writes.tissue_polygons is not None and not tissue_io.tissue_exists(slide_artifact):
            tissue_io.write_tissue(slide_artifact, pending_writes.tissue_polygons)

        if pending_writes.thumbnail_image_bytes is not None and not thumbnail_io.thumbnail_image_exists(slide_artifact):
            thumbnail_io.write_thumbnail_image(
                slide_artifact,
                pending_writes.thumbnail_image_bytes,
            )
        if pending_writes.thumbnail_spec is not None and not thumbnail_io.thumbnail_spec_exists(slide_artifact):
            thumbnail_io.write_thumbnail_spec(
                slide_artifact,
                pending_writes.thumbnail_spec,
            )

        if pending_writes.coords_array is not None:
            tiles_io.write_coords(slide_artifact, tiling_id, pending_writes.coords_array)
        if pending_writes.tiling_spec is not None:
            tiles_io.write_tiling_spec(slide_artifact, tiling_id, pending_writes.tiling_spec)
        if pending_writes.tiles_overview_bytes is not None and not tiles_io.tiles_overview_exists(slide_artifact, tiling_id):
            tiles_io.write_tiles_overview(slide_artifact, tiling_id, pending_writes.tiles_overview_bytes)
        if pending_writes.feature_matrix is not None and not features_io.features_exist(
            slide_artifact,
            bag_id=tiling_id,
            extractor_name=extractor_name,
            expected_rows=int(pending_writes.feature_matrix.shape[0]),
        ):
            features_io.write_features(slide_artifact, tiling_id, extractor_name, pending_writes.feature_matrix)

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
            import_module(f"pathforge.core.slide_processing.{self.backend_name}")

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
        runtime = self.config.slide_processing.feature_extraction
        model_params = combo_cfg.get_hyperparams("feature_extraction")
        return {
            "model": combo_cfg.feature_extraction,
            "color_norm": combo_cfg.get("color_norm"),
            "params": {
                "batch_size": runtime.batch_size,
                "num_workers": runtime.num_workers,
                "amp": runtime.amp,
                **model_params,
            },
        }

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

    # ------------------------------------------------------------------
    # Report generation (after all combos)
    # ------------------------------------------------------------------

    def _generate_tiles_reports_after_extraction(
        self,
        *,
        datasets: list[WSIDataset],
        combos: list[ComboConfig],
    ) -> None:
        unique_bag_ids = self._collect_unique_report_bag_ids(combos=combos)

        if not unique_bag_ids:
            logger.info("[Policy] report=True but no tiling combos found; skipping tile reports.")
            return

        logger.info(
            "[Policy] Generating tile overview PDF reports for %d unique bag_ids across %d datasets.",
            len(unique_bag_ids),
            len(datasets),
        )

        for dataset in datasets:
            for bag_id in unique_bag_ids:
                try:
                    self._generate_tiles_report_for_dataset_bag(dataset=dataset, bag_id=bag_id)
                except Exception:
                    logger.exception(
                        "[Policy] Failed to generate tile report for dataset='%s', bag_id='%s'",
                        dataset.name,
                        bag_id,
                    )

    def _collect_unique_report_bag_ids(
        self,
        *,
        combos: list[ComboConfig],
    ) -> list[str]:
        """
        Deduplicate bag_ids across combos while preserving order.

        Important: tiling_id depends only on tile_px + tile_mpp, so multiple feature
        extractors should map to the same report target.
        """
        seen: set[str] = set()
        bag_ids: list[str] = []

        for combo_cfg in combos:
            tiling_id = build_tiling_id(combo_cfg)
            if tiling_id in seen:
                continue
            seen.add(tiling_id)
            bag_ids.append(tiling_id)

        return bag_ids

    def _generate_tiles_report_for_dataset_bag(self, *, dataset: WSIDataset, bag_id: str) -> None:
        """
        Wrapper around the PDF report generator.

        Keeps the rest of the policy independent from the exact reporting function
        signature used in pathforge.core.reporting.tiles_report_pdf.
        """

        logger.info(
            "[Policy] Generating tile report for dataset='%s' bag_id='%s' (artifacts_dir=%s)",
            dataset.name,
            bag_id,
            dataset.artifacts_dir,
        )

        collection = collect_tiles_overview_entries(dataset=dataset, bag_id=bag_id)
        if not collection.entries:
            logger.info(
                "[Policy] Skipping tile report for dataset='%s' bag_id='%s': no tiles_overview entries found.",
                dataset.name,
                bag_id,
            )
            return

        output_pdf = create_tiles_report_pdf(
            dataset=dataset,
            bag_id=bag_id,
            output_path=None,   # timestamped default in dataset.artifacts_dir
            timestamp=None,     # use current time
        )

        logger.info(
            "[Policy] Tile report generated for dataset='%s' bag_id='%s': %s",
            dataset.name,
            bag_id,
            output_pdf,
        )
