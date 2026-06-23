from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pathbench.config.config import DatasetEntry
from pathbench.core.datasets.wsi_dataset import WSI
from pathbench.core.experiments.base import Experiment
from pathbench.core.io.slide_artifacts.base import FileHandleH5
from pathbench.core.io.slide_artifacts import tissue as tissue_io
from pathbench.core.io.slide_artifacts import thumbnail as thumbnail_io
from pathbench.core.io.slide_artifacts import tiles as tiles_io
from pathbench.core.slide_processing.base import SlideProcessorBase
from pathbench.core.visualization.thumbnail import (
    crop_thumbnail_to_tissue_bounds,
    decode_thumbnail_image,
)
from pathbench.slide_retrieval.io import (
    load_slide_retrieval_representation,
    read_slide_retrieval_results_csv,
    resolve_slide_retrieval_results_path,
)
from pathbench.slide_retrieval.representation_strategies.storage import (
    build_retrieval_representation_artifact_path,
    build_retrieval_representation_entry_id,
)
from pathbench.slide_retrieval.search_strategies.types import SearchResult
from pathbench.slide_retrieval.visualization.renderers import (
    render_retrieval_representation_image,
    render_retrieval_results_image,
)
from pathbench.utils.constants import (
    CATEGORY_COL,
    DATASET_COL,
    PATIENT_ID_COL,
    SLIDE_FILE_FORMATS,
    SLIDE_ID_COL,
)
from pathbench.utils.registries import SLIDE_PROCESSORS


logger = logging.getLogger(__name__)

_RESULTS_VIS_NAME = "retrieval_results"
_REPRESENTATION_VIS_NAME = "retrieval_representation"
_SUPPORTED_VISUALIZATIONS = {
    _RESULTS_VIS_NAME,
    _REPRESENTATION_VIS_NAME,
}


@dataclass(frozen=True, slots=True)
class SlideVisualizationAsset:
    """
    Slide-level artifact context used by retrieval visualization.

    Inputs:
    - `slide_id`: slide/sample identifier in slide-level retrieval mode.
    - `dataset_name`: configured dataset that owns the slide.
    - `artifact_path`: per-slide H5 artifact path.
    - `slide_path`: optional original slide path used for thumbnail/patch fallback.
    - `patient_id`: patient identifier from annotations.
    - `category`: category/label-like metadata from annotations.
    - `fallback_mpp`: optional fallback MPP from annotations.
    - `metadata`: row-like metadata lookup keyed by annotation column name.

    Example:
    ```python
    asset = SlideVisualizationAsset(
        slide_id="slide_001",
        dataset_name="cohort_a",
        artifact_path=Path("/tmp/artifacts/slide_001.h5"),
        slide_path=None,
        patient_id="patient_1",
        category="tumor",
        fallback_mpp=None,
        metadata={"slide": "slide_001", "category": "tumor"},
    )
    ```
    """

    slide_id: str
    dataset_name: str
    artifact_path: Path
    slide_path: Path | None
    patient_id: str
    category: str
    fallback_mpp: float | None
    metadata: dict[str, Any]


class SlideRetrievalVisualizationService:
    """
    Render run-level slide retrieval visualizations.

    Inputs:
    - `experiment`: configured experiment used to load annotations and backends.
    - `run_dir`: one concrete retrieval run directory containing `manifest.json`
      and ranked results (`query_results.xlsx` or legacy `query_results.csv`).
    - `manifest`: parsed retrieval manifest for the run.

    Returns:
    - Service object that can render configured retrieval visualizations into
      `run_dir / f"vis_{visualization_name}"`.

    Example:
    ```python
    service = SlideRetrievalVisualizationService(
        experiment=experiment,
        run_dir=Path("/tmp/run_abc"),
        manifest={"tiling_id": "256px_0.5mpp", "representation_id": "repr_id"},
    )
    files = service.render_requested_visualizations(
        requested_visualizations=["retrieval_results"],
        subset_ids={"slide_1"},
    )
    ```
    """

    def __init__(
        self,
        *,
        experiment: Experiment,
        run_dir: Path,
        manifest: dict[str, Any],
        visualization_root: Path | None = None,
    ) -> None:
        self.experiment = experiment
        self.cfg = experiment.cfg
        self.run_dir = Path(run_dir).resolve()
        self.visualization_root = (
            Path(visualization_root).resolve()
            if visualization_root is not None
            else None
        )
        self.manifest = dict(manifest)
        self.annotations_df = self.experiment.load_annotations().copy()
        self.slide_retrieval_cfg = self.cfg.slide_retrieval
        self.dataset_cfg_by_name = {
            str(ds_cfg.name): ds_cfg for ds_cfg in self.cfg.datasets
        }
        self._slide_processor: SlideProcessorBase | None = None

    def render_requested_visualizations(
        self,
        *,
        requested_visualizations: list[str],
        subset_ids: set[str] | None,
    ) -> list[Path]:
        """
        Render the configured retrieval visualizations for one run.

        Inputs:
        - `requested_visualizations`: visualization names from config.
        - `subset_ids`: optional set of selected slide IDs from the subset CSV.

        Returns:
        - `list[Path]`: absolute PNG paths written for this run.
        """

        unknown_visualizations = sorted(
            {
                str(name).strip()
                for name in requested_visualizations
                if str(name).strip() not in _SUPPORTED_VISUALIZATIONS
            }
        )
        if unknown_visualizations:
            raise ValueError(
                "Unsupported slide-retrieval visualizations requested: "
                f"{unknown_visualizations}. Supported values: "
                f"{sorted(_SUPPORTED_VISUALIZATIONS)}"
            )

        created_files: list[Path] = []

        if _RESULTS_VIS_NAME in requested_visualizations:
            results_path = resolve_slide_retrieval_results_path(
                self.run_dir / "query_results.xlsx"
            )
            if not results_path.is_file():
                logger.warning(
                    "Skipping retrieval-results visualization because no ranked-results file was found at '%s'.",
                    results_path,
                )
            else:
                results = read_slide_retrieval_results_csv(results_path)
                created_files.extend(
                    self._render_retrieval_results_visualization(
                        results=results,
                        subset_ids=subset_ids,
                    )
                )

        if _REPRESENTATION_VIS_NAME in requested_visualizations:
            results_path = resolve_slide_retrieval_results_path(
                self.run_dir / "query_results.xlsx"
            )
            created_files.extend(
                self._render_retrieval_representation_visualization(
                    results=None if not results_path.is_file() else read_slide_retrieval_results_csv(results_path),
                    subset_ids=subset_ids,
                )
            )

        return created_files

    def _render_retrieval_results_visualization(
        self,
        *,
        results: list[SearchResult],
        subset_ids: set[str] | None,
    ) -> list[Path]:
        output_dir = self._ensure_output_dir(_RESULTS_VIS_NAME)
        top_k = min(10, int(self.slide_retrieval_cfg.visualization_top_k))
        selected_results = self._select_query_results(results, subset_ids=subset_ids)

        created_files: list[Path] = []
        for result in selected_results:
            query_asset = self._resolve_slide_asset(result.query_sample_id)
            if query_asset is None:
                logger.warning(
                    "Skipping retrieval-results visualization for query '%s' because it could not be resolved.",
                    result.query_sample_id,
                )
                continue

            query_thumbnail = self._load_thumbnail_or_placeholder(query_asset)
            query_lines = self._build_metadata_lines(query_asset)
            hit_panels: list[tuple[Image.Image, list[str]]] = []
            for hit in result.hits[:top_k]:
                hit_asset = self._resolve_slide_asset(hit.sample_id)
                if hit_asset is None:
                    hit_panels.append(
                        (
                            self._build_placeholder_image(
                                title=str(hit.sample_id),
                                message="slide unavailable",
                            ),
                            [f"slide: {hit.sample_id}", "status: unavailable"],
                        )
                    )
                    continue

                hit_panels.append(
                    (
                        self._load_thumbnail_or_placeholder(hit_asset),
                        self._build_metadata_lines(hit_asset),
                    )
                )

            rendered = render_retrieval_results_image(
                query_thumbnail=query_thumbnail,
                query_lines=query_lines,
                hit_panels=hit_panels,
            )
            output_path = output_dir / f"{self._safe_filename(query_asset.slide_id)}.png"
            rendered.save(output_path, format="PNG")
            created_files.append(output_path.resolve())

        return created_files

    def _render_retrieval_representation_visualization(
        self,
        *,
        results: list[SearchResult] | None,
        subset_ids: set[str] | None,
    ) -> list[Path]:
        output_dir = self._ensure_output_dir(_REPRESENTATION_VIS_NAME)
        slide_ids = self._select_representation_slide_ids(
            results=results,
            subset_ids=subset_ids,
        )

        created_files: list[Path] = []
        for slide_id in slide_ids:
            asset = self._resolve_slide_asset(slide_id)
            if asset is None:
                logger.warning(
                    "Skipping retrieval-representation visualization for slide '%s' because it could not be resolved.",
                    slide_id,
                )
                continue

            representation_payload = self._load_representation_payload(asset)
            if representation_payload is None:
                logger.warning(
                    "Skipping retrieval-representation visualization for slide '%s' because no representation payload was found.",
                    slide_id,
                )
                continue

            thumbnail_and_spec = self._load_thumbnail_and_spec(asset)
            if thumbnail_and_spec is None:
                logger.warning(
                    "Skipping retrieval-representation visualization for slide '%s' because no thumbnail was available.",
                    slide_id,
                )
                continue

            coords_and_spec = self._load_slide_coords(asset)
            if coords_and_spec is None:
                logger.warning(
                    "Skipping retrieval-representation visualization for slide '%s' because no coords were available for tiling_id '%s'.",
                    slide_id,
                    self.manifest.get("tiling_id"),
                )
                continue

            thumbnail_image, thumbnail_spec = thumbnail_and_spec
            coords_array, tiling_spec = coords_and_spec
            base_mpp = self._load_base_mpp(asset)
            tissue_polygons = self._load_tissue_polygons(asset)
            group_ids = self._coerce_optional_array(
                representation_payload.additional_data.get("group_ids")
            )
            selected_coords = self._coerce_optional_array(
                representation_payload.additional_data.get("selected_coords")
            )
            patch_strip_images = self._load_patch_strip_images(
                asset=asset,
                selected_coords=selected_coords,
                coords_array=coords_array,
            )
            patch_group_ids = self._resolve_selected_patch_group_ids(
                selected_coords=selected_coords,
                coords_array=coords_array,
                group_ids=group_ids,
            )

            rendered = render_retrieval_representation_image(
                thumbnail_image=thumbnail_image,
                downscale_x=float(thumbnail_spec["downscale_x"]),
                downscale_y=float(thumbnail_spec["downscale_y"]),
                coords_array=coords_array,
                tiling_spec=tiling_spec,
                base_mpp=base_mpp,
                group_ids=group_ids,
                selected_coords=selected_coords,
                tissue_polygons=tissue_polygons,
                patch_strip_images=patch_strip_images,
                patch_group_ids=patch_group_ids,
            )
            output_path = output_dir / f"{self._safe_filename(asset.slide_id)}.png"
            rendered.save(output_path, format="PNG")
            created_files.append(output_path.resolve())

        return created_files

    def _select_query_results(
        self,
        results: list[SearchResult],
        *,
        subset_ids: set[str] | None,
    ) -> list[SearchResult]:
        if subset_ids is None:
            return list(results)
        return [
            result
            for result in results
            if str(result.query_sample_id) in subset_ids
        ]

    def _select_representation_slide_ids(
        self,
        *,
        results: list[SearchResult] | None,
        subset_ids: set[str] | None,
    ) -> list[str]:
        if subset_ids is not None:
            return sorted(str(slide_id) for slide_id in subset_ids)

        if results is not None:
            # Without an explicit subset, default to visualizing the query slides only.
            return sorted({str(result.query_sample_id) for result in results})

        return sorted(
            {
                str(slide_id).strip()
                for slide_id in self.annotations_df[SLIDE_ID_COL].tolist()
                if str(slide_id).strip()
            }
        )

    def _resolve_slide_asset(self, slide_id: str) -> SlideVisualizationAsset | None:
        normalized_slide_id = str(slide_id).strip()
        matching_rows = self.annotations_df[
            self.annotations_df[SLIDE_ID_COL].astype(str) == normalized_slide_id
        ]
        if matching_rows.empty:
            return None

        row = matching_rows.iloc[0]
        dataset_name = str(row.get(DATASET_COL, "")).strip()
        dataset_cfg = self.dataset_cfg_by_name.get(dataset_name)
        if dataset_cfg is None:
            return None

        slide_path = self._find_slide_path(dataset_cfg, normalized_slide_id)
        artifact_path = (
            Path(dataset_cfg.artifacts_dir).expanduser().resolve()
            / f"{normalized_slide_id}.h5"
        )
        fallback_mpp = self._parse_optional_float(row.get("fallback_mpp"))
        metadata = {
            str(column_name): row.get(column_name)
            for column_name in matching_rows.columns
        }
        return SlideVisualizationAsset(
            slide_id=normalized_slide_id,
            dataset_name=dataset_name,
            artifact_path=artifact_path,
            slide_path=slide_path,
            patient_id=str(row.get(PATIENT_ID_COL, "")),
            category=str(row.get(CATEGORY_COL, "")),
            fallback_mpp=fallback_mpp,
            metadata=metadata,
        )

    def _build_metadata_lines(self, asset: SlideVisualizationAsset) -> list[str]:
        configured_columns = list(self.slide_retrieval_cfg.visualization_metadata_columns)
        lines = [f"slide: {asset.slide_id}"]
        for column_name in configured_columns:
            normalized_name = str(column_name).strip()
            if not normalized_name or normalized_name == SLIDE_ID_COL:
                continue
            value = asset.metadata.get(normalized_name, "-")
            if pd.isna(value):
                value = "-"
            lines.append(f"{normalized_name}: {value}")
        return lines

    def _load_thumbnail_or_placeholder(self, asset: SlideVisualizationAsset) -> Image.Image:
        thumbnail_and_spec = self._load_thumbnail_and_spec(asset)
        if thumbnail_and_spec is None:
            return self._build_placeholder_image(
                title=asset.slide_id,
                message="thumbnail unavailable",
            )
        thumbnail_image, thumbnail_spec = thumbnail_and_spec
        return crop_thumbnail_to_tissue_bounds(
            thumbnail_image,
            tissue_polygons=self._load_tissue_polygons(asset),
            downscale_x=float(thumbnail_spec["downscale_x"]),
            downscale_y=float(thumbnail_spec["downscale_y"]),
        )

    def _load_thumbnail_and_spec(
        self,
        asset: SlideVisualizationAsset,
    ) -> tuple[Image.Image, dict[str, Any]] | None:
        if asset.artifact_path.is_file():
            with FileHandleH5(asset.artifact_path, mode="r") as slide_artifact:
                if thumbnail_io.thumbnail_image_exists(
                    slide_artifact
                ) and thumbnail_io.thumbnail_spec_exists(slide_artifact):
                    image_bytes = thumbnail_io.read_thumbnail_image(slide_artifact)
                    spec = thumbnail_io.read_thumbnail_spec(slide_artifact)
                    return decode_thumbnail_image(image_bytes), dict(spec)

        if asset.slide_path is None or not asset.slide_path.exists():
            return None

        slide_processor = self._build_processor()
        wsi = WSI(
            slide=asset.slide_id,
            patient=asset.patient_id,
            category=asset.category,
            path=asset.slide_path,
            artifact_path=asset.artifact_path,
            fallback_mpp=asset.fallback_mpp,
        )
        slide_processor.load_wsi(wsi)
        try:
            thumbnail_image, downscale_x, downscale_y = slide_processor.get_thumbnail(
                wsi,
                level=-1,
            )
        finally:
            slide_processor.close_wsi(wsi)

        return (
            Image.fromarray(np.asarray(thumbnail_image)).convert("RGB")
            if not isinstance(thumbnail_image, Image.Image)
            else thumbnail_image.convert("RGB"),
            {
                "image_format": "memory",
                "coord_space": "level0",
                "thumbnail_level": -1,
                "downscale_x": float(downscale_x),
                "downscale_y": float(downscale_y),
            },
        )

    def _load_slide_coords(
        self,
        asset: SlideVisualizationAsset,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        if not asset.artifact_path.is_file():
            return None

        tiling_id = str(self.manifest["tiling_id"])
        with FileHandleH5(asset.artifact_path, mode="r") as slide_artifact:
            if not (
                tiles_io.coords_exist(slide_artifact, tiling_id)
                and tiles_io.tiling_spec_exists(slide_artifact, tiling_id)
            ):
                return None

            coords_array = tiles_io.read_coords(slide_artifact, tiling_id)
            tiling_spec = tiles_io.read_tiling_spec(slide_artifact, tiling_id)
        return coords_array, tiling_spec

    def _load_tissue_polygons(
        self,
        asset: SlideVisualizationAsset,
    ) -> list[Any] | None:
        if not asset.artifact_path.is_file():
            return None

        with FileHandleH5(asset.artifact_path, mode="r") as slide_artifact:
            if not tissue_io.tissue_exists(slide_artifact):
                return None
            return tissue_io.read_tissue(slide_artifact)

    def _load_base_mpp(
        self,
        asset: SlideVisualizationAsset,
    ) -> float | None:
        if asset.slide_path is None or not asset.slide_path.exists():
            return None

        slide_processor = self._build_processor()
        wsi = WSI(
            slide=asset.slide_id,
            patient=asset.patient_id,
            category=asset.category,
            path=asset.slide_path,
            artifact_path=asset.artifact_path,
            fallback_mpp=asset.fallback_mpp,
        )
        slide_processor.load_wsi(wsi)
        try:
            return float(slide_processor.get_base_mpp(wsi))
        except Exception:
            logger.warning(
                "Failed to resolve base_mpp for slide '%s' while rendering retrieval representation.",
                asset.slide_id,
            )
            return None
        finally:
            slide_processor.close_wsi(wsi)

    def _load_representation_payload(self, asset: SlideVisualizationAsset) -> Any | None:
        retrieval_artifact_path = build_retrieval_representation_artifact_path(
            artifacts_dir=asset.artifact_path.parent,
            aggregation_level=str(self.manifest["aggregation_level"]),
            sample_id=asset.slide_id,
        )
        if not retrieval_artifact_path.is_file():
            return None

        entry_id = build_retrieval_representation_entry_id(
            [asset.slide_id],
            aggregation_level=str(self.manifest["aggregation_level"]),
        )
        with FileHandleH5(retrieval_artifact_path, mode="r") as retrieval_artifact:
            return load_slide_retrieval_representation(
                retrieval_artifact=retrieval_artifact,
                tile_id=str(self.manifest["tiling_id"]),
                representation_id=str(self.manifest["representation_id"]),
                entry_id=entry_id,
            )

    def _load_patch_strip_images(
        self,
        *,
        asset: SlideVisualizationAsset,
        selected_coords: np.ndarray | None,
        coords_array: np.ndarray,
    ) -> list[Image.Image]:
        if (
            selected_coords is None
            or selected_coords.size == 0
            or asset.slide_path is None
            or not asset.slide_path.exists()
        ):
            return []

        coords_lookup = {
            (int(row[0]), int(row[1])): row for row in np.asarray(coords_array)
        }
        default_width = int(np.median(coords_array[:, 2])) if coords_array.size else 256
        default_height = int(np.median(coords_array[:, 3])) if coords_array.size else 256
        default_level = int(np.median(coords_array[:, 4])) if coords_array.size else 0

        slide_processor = self._build_processor()
        wsi = WSI(
            slide=asset.slide_id,
            patient=asset.patient_id,
            category=asset.category,
            path=asset.slide_path,
            artifact_path=asset.artifact_path,
            fallback_mpp=asset.fallback_mpp,
        )
        slide_processor.load_wsi(wsi)
        patch_images: list[Image.Image] = []
        try:
            for coord in np.asarray(selected_coords):
                coord_key = (int(coord[0]), int(coord[1]))
                coord_row = coords_lookup.get(coord_key)
                if coord_row is None:
                    width = default_width
                    height = default_height
                    level = default_level
                else:
                    width = int(coord_row[2])
                    height = int(coord_row[3])
                    level = int(coord_row[4])

                try:
                    patch_array = slide_processor.read_patch_region(
                        wsi,
                        int(coord[0]),
                        int(coord[1]),
                        width,
                        height,
                        level,
                    )
                    patch_images.append(
                        Image.fromarray(
                            np.asarray(patch_array, dtype=np.uint8)
                        ).convert("RGB")
                    )
                except Exception as exc:
                    logger.warning(
                        "Using unreadable-patch placeholder for slide '%s' at "
                        "x=%s, y=%s, width=%s, height=%s, level=%s: %s",
                        asset.slide_id,
                        int(coord[0]),
                        int(coord[1]),
                        width,
                        height,
                        level,
                        exc,
                    )
                    patch_images.append(
                        self._build_unreadable_patch_placeholder(
                            width=width,
                            height=height,
                        )
                    )
                    slide_processor.close_wsi(wsi)
                    try:
                        slide_processor.load_wsi(wsi)
                    except Exception:
                        logger.warning(
                            "Could not reopen slide '%s' after an unreadable "
                            "patch; remaining patch-strip entries will use "
                            "placeholders.",
                            asset.slide_id,
                        )
                        remaining_count = len(np.asarray(selected_coords)) - len(
                            patch_images
                        )
                        for _ in range(max(0, remaining_count)):
                            patch_images.append(
                                self._build_unreadable_patch_placeholder(
                                    width=width,
                                    height=height,
                                )
                            )
                        break
        finally:
            slide_processor.close_wsi(wsi)

        return patch_images

    def _build_unreadable_patch_placeholder(
        self,
        *,
        width: int,
        height: int,
    ) -> Image.Image:
        placeholder_width = max(1, int(width))
        placeholder_height = max(1, int(height))
        image = Image.new(
            "RGB",
            (placeholder_width, placeholder_height),
            (196, 57, 45),
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [(0, 0), (placeholder_width - 1, placeholder_height - 1)],
            outline=(122, 31, 26),
            width=max(1, min(placeholder_width, placeholder_height) // 32),
        )
        draw.line(
            [(0, 0), (placeholder_width - 1, placeholder_height - 1)],
            fill=(255, 178, 142),
            width=max(1, min(placeholder_width, placeholder_height) // 24),
        )
        draw.line(
            [(0, placeholder_height - 1), (placeholder_width - 1, 0)],
            fill=(255, 178, 142),
            width=max(1, min(placeholder_width, placeholder_height) // 24),
        )
        label = "unreadable"
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = max(2, (placeholder_width - text_width) // 2)
        text_y = max(2, (placeholder_height - text_height) // 2)
        draw.rectangle(
            [
                (text_x - 4, text_y - 3),
                (text_x + text_width + 4, text_y + text_height + 3),
            ],
            fill=(122, 31, 26),
        )
        draw.text((text_x, text_y), label, fill=(255, 244, 237), font=font)
        return image

    def _resolve_selected_patch_group_ids(
        self,
        *,
        selected_coords: np.ndarray | None,
        coords_array: np.ndarray,
        group_ids: np.ndarray | None,
    ) -> list[int | None]:
        if (
            selected_coords is None
            or selected_coords.size == 0
            or group_ids is None
            or coords_array.shape[0] != int(group_ids.shape[0])
        ):
            return []

        coords_lookup = {
            (int(row[0]), int(row[1])): int(group_ids[idx])
            for idx, row in enumerate(np.asarray(coords_array))
        }
        return [
            coords_lookup.get((int(coord[0]), int(coord[1])))
            for coord in np.asarray(selected_coords)
        ]

    def _build_processor(self) -> SlideProcessorBase:
        if self._slide_processor is not None:
            return self._slide_processor

        backend_name = str(self.cfg.slide_processing.backend)
        if not SLIDE_PROCESSORS.is_available(backend_name):
            import_module(f"pathbench.core.slide_processing.{backend_name}")

        processor_cls = SLIDE_PROCESSORS.get(backend_name)
        if processor_cls is None:
            raise ValueError(
                f"Slide processing backend '{backend_name}' is not registered."
            )

        self._slide_processor = processor_cls()
        return self._slide_processor

    def _ensure_output_dir(self, visualization_name: str) -> Path:
        visualization_dir_name = f"vis_{visualization_name}"
        if self.visualization_root is not None:
            output_dir = self.visualization_root / visualization_dir_name
        else:
            output_dir = self.run_dir / visualization_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _find_slide_path(
        self,
        dataset_cfg: DatasetEntry,
        slide_id: str,
    ) -> Path | None:
        slides_dir = Path(dataset_cfg.slides_dir).expanduser().resolve()
        if not slides_dir.is_dir():
            return None

        direct_matches = sorted(
            path
            for path in slides_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SLIDE_FILE_FORMATS
            and path.stem == slide_id
        )
        if len(direct_matches) == 1:
            return direct_matches[0]
        if len(direct_matches) > 1:
            logger.warning(
                "Multiple direct slide matches found for slide '%s' in dataset '%s'.",
                slide_id,
                dataset_cfg.name,
            )
            return None

        dicom_dir = slides_dir / slide_id
        if dicom_dir.is_dir():
            dicom_matches = sorted(
                path
                for path in dicom_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".dcm"
            )
            if dicom_matches:
                return dicom_matches[0]

        return None

    def _build_placeholder_image(self, *, title: str, message: str) -> Image.Image:
        width, height = 512, 512
        canvas = Image.new("RGB", (width, height), (248, 248, 248))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        draw.rectangle([(0, 0), (width - 1, height - 1)], outline=(0, 0, 0), width=1)
        draw.text((20, 20), str(title), fill=(0, 0, 0), font=font)
        draw.text((20, 52), str(message), fill=(80, 80, 80), font=font)
        return canvas

    def _parse_optional_float(self, value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_optional_array(self, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value)
        if array.size == 0:
            return array
        return array

    def _safe_filename(self, value: str) -> str:
        filename = str(value).strip().replace("/", "-").replace("\\", "-")
        filename = filename.replace(" ", "_")
        return filename or "visualization"
