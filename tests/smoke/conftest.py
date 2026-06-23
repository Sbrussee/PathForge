"""Session-scoped fixtures for realistic PathBench smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ._smoke_dataset import (
    GTEX_ARTERY_SLIDE_IDS,
    ExtractedWsiWorkspace,
    PreparedBagWorkspace,
    aggregate_slide_feature_matrix,
    attach_smoke_outputs,
    build_gtex_smoke_annotations,
    configured_smoke_report_dir,
    capture_smoke_metrics,
    default_smoke_cache_dir,
    download_smoke_assets,
    link_or_copy,
    merge_survival_metadata,
    read_h5_feature_matrix,
    DownloadedSmokeAssets,
    write_smoke_report,
)
from ._smoke_training import register_smoke_components


@dataclass(frozen=True)
class RetrievalDatasets:
    """Real SlideRetrievalBagDataset instances for slide retrieval smoke tests."""

    reference: Any
    query: Any
    single_ref: Any
    all_slide_ids: list[str]


@pytest.fixture(scope="session", autouse=True)
def _finalize_smoke_report() -> None:
    """Write one aggregate smoke report when report output is configured."""
    yield
    report_dir = configured_smoke_report_dir()
    if report_dir is not None:
        write_smoke_report(report_dir)


@pytest.fixture(scope="session")
def smoke_assets() -> DownloadedSmokeAssets:
    """Download and cache the Hugging Face assets used by the smoke suite."""
    pytest.importorskip("huggingface_hub")
    return download_smoke_assets(default_smoke_cache_dir())


@pytest.fixture(scope="session")
def extracted_wsi_workspace(
    smoke_assets: DownloadedSmokeAssets,
    tmp_path_factory: pytest.TempPathFactory,
) -> ExtractedWsiWorkspace:
    """Run session-scoped PathBench tile extraction on a few small WSIs."""
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("lazyslide")

    register_smoke_components()
    from pathbench.config.config import Config
    from pathbench.core.experiments.base import Experiment
    from pathbench.policy.feature_extraction import FeatureExtractionPolicy

    root_dir = tmp_path_factory.mktemp("hf_smoke_wsi")
    slides_dir = root_dir / "slides"
    artifacts_dir = root_dir / "artifacts"
    metrics_dir = root_dir / "metrics"
    project_root = root_dir / "project_root"
    slides_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    slide_rows = build_gtex_smoke_annotations(
        smoke_assets.gtex_artery_csv,
        slide_ids=list(GTEX_ARTERY_SLIDE_IDS),
    )
    slide_rows["dataset"] = "hf_smoke"
    for slide_id in GTEX_ARTERY_SLIDE_IDS:
        filename = f"{slide_id}.svs"
        link_or_copy(smoke_assets.slides[filename], slides_dir / filename)

    annotations_csv = root_dir / "annotations.csv"
    slide_rows.to_csv(annotations_csv, index=False)

    extractor_name = "resnet18"
    tile_px = 224
    tile_mpp = 1.0
    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "hf_smoke_feature_extraction",
                "annotation_file": str(annotations_csv),
                "project_root": str(project_root.resolve()),
                "mode": "feature_extraction",
                "report": True,
                "thumbnail": True,
            },
            "slide_processing": {
                "backend": "lazyslide",
                "segmentation_method": "otsu",
            },
            "datasets": [
                {
                    "name": "hf_smoke",
                    "slides_dir": str(slides_dir.resolve()),
                    "artifacts_dir": str(artifacts_dir.resolve()),
                    "used_for": "all",
                }
            ],
            "benchmark_parameters": {
                "tile_px": [tile_px],
                "tile_mpp": [tile_mpp],
                "feature_extraction": [extractor_name],
                "mil": [],
            },
        }
    )

    experiment = Experiment(cfg)
    policy = FeatureExtractionPolicy(experiment)
    policy._build_seg_config = lambda: {"method": "otsu", "params": {}}  # type: ignore[method-assign]
    policy._build_feat_config = (  # type: ignore[method-assign]
        lambda combo_cfg: {
            "model": combo_cfg.feature_extraction,
            "params": {"pretrained": False},
        }
    )

    with capture_smoke_metrics(
        metrics_dir,
        step_name="hf_wsi_feature_extraction",
        metadata={
            "num_slides": len(slide_rows),
            "extractor_name": extractor_name,
            "tile_px": tile_px,
            "tile_mpp": tile_mpp,
        },
    ) as metadata:
        result = policy.execute()
        assert result["status"] == "feature_extraction_done"
        attach_smoke_outputs(
            metadata,
            step_name="hf_wsi_feature_extraction",
            intermediate={
                "slides_dir": slides_dir,
                "annotations_csv": annotations_csv,
                "gtex_artery_csv": smoke_assets.gtex_artery_csv,
            },
            final={
                "artifacts_dir": artifacts_dir,
                "project_root": project_root,
            },
        )

    bag_id = f"{tile_px}px_{tile_mpp:g}mpp"
    artifact_paths = {
        slide_id: artifacts_dir / f"{slide_id}.h5"
        for slide_id in slide_rows["slide"].astype(str).tolist()
    }
    return ExtractedWsiWorkspace(
        root_dir=root_dir,
        slides_dir=slides_dir,
        artifacts_dir=artifacts_dir,
        annotations_csv=annotations_csv,
        artifact_paths=artifact_paths,
        bag_id=bag_id,
        extractor_name=extractor_name,
        metrics_path=metrics_dir / "hf_wsi_feature_extraction.metrics.json",
    )


@pytest.fixture(scope="session")
def extracted_bag_workspace(
    extracted_wsi_workspace: ExtractedWsiWorkspace,
    tmp_path_factory: pytest.TempPathFactory,
) -> PreparedBagWorkspace:
    """Convert extracted H5 tile features into reusable MIL bag tensors."""
    torch = pytest.importorskip("torch")
    root_dir = tmp_path_factory.mktemp("hf_smoke_bags")
    feature_dir = root_dir / "bags"
    feature_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = root_dir / "metrics"

    annotations_df = pd.read_csv(extracted_wsi_workspace.annotations_csv)
    category_by_slide = {
        str(row["slide"]): str(row["category"])
        for _, row in annotations_df.iterrows()
    }
    age_bracket_by_slide = {
        str(row["slide"]): str(row["age_bracket"])
        for _, row in annotations_df.iterrows()
    }
    unique_age_brackets = sorted(set(age_bracket_by_slide.values()))
    age_bracket_map = {bracket: idx for idx, bracket in enumerate(unique_age_brackets)}
    label_rows: list[dict[str, int | str]] = []
    bag_lengths: dict[str, int] = {}
    input_dim = 0

    with capture_smoke_metrics(
        metrics_dir,
        step_name="hf_prepare_extracted_bags",
        metadata={"num_slides": len(extracted_wsi_workspace.artifact_paths)},
    ) as metadata:
        for slide_id, artifact_path in sorted(
            extracted_wsi_workspace.artifact_paths.items()
        ):
            feature_matrix = read_h5_feature_matrix(
                artifact_path,
                bag_id=extracted_wsi_workspace.bag_id,
                extractor_name=extracted_wsi_workspace.extractor_name,
            )
            input_dim = int(feature_matrix.shape[1])
            bag_lengths[slide_id] = int(feature_matrix.shape[0])
            torch.save(
                torch.from_numpy(feature_matrix.astype(np.float32, copy=False)),
                feature_dir / f"{slide_id}.pt",
            )
            category_name = category_by_slide[slide_id]
            label_rows.append(
                {
                    "slide_id": slide_id,
                    "category": category_name,
                    "binary_label": int(category_name == "calcification"),
                    "multiclass_label": age_bracket_map[age_bracket_by_slide[slide_id]],
                }
            )
        attach_smoke_outputs(
            metadata,
            step_name="hf_prepare_extracted_bags",
            intermediate={"source_artifacts_dir": extracted_wsi_workspace.artifacts_dir},
            final={"feature_dir": feature_dir},
        )

    metadata_csv = root_dir / "bag_metadata.csv"
    pd.DataFrame(label_rows).to_csv(metadata_csv, index=False)
    return PreparedBagWorkspace(
        root_dir=root_dir,
        feature_dir=feature_dir,
        metadata_csv=metadata_csv,
        slide_ids=[row["slide_id"] for row in label_rows],
        input_dim=input_dim,
        bag_lengths=bag_lengths,
        metrics_path=metrics_dir / "hf_prepare_extracted_bags.metrics.json",
    )


@pytest.fixture(scope="session")
def survival_bag_workspace(
    smoke_assets: DownloadedSmokeAssets,
    tmp_path_factory: pytest.TempPathFactory,
) -> PreparedBagWorkspace:
    """Create tiny one-instance MIL bags from TCGA READ slide-level features."""
    ad = pytest.importorskip("anndata")
    torch = pytest.importorskip("torch")

    root_dir = tmp_path_factory.mktemp("hf_smoke_survival")
    feature_dir = root_dir / "bags"
    feature_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = root_dir / "metrics"

    with capture_smoke_metrics(
        metrics_dir,
        step_name="hf_prepare_survival_bags",
        metadata={"source_h5ad": str(smoke_assets.survival_h5ad)},
    ) as metadata:
        adata = ad.read_h5ad(smoke_assets.survival_h5ad)
        merged = merge_survival_metadata(adata.obs, smoke_assets.survival_csv)
        merged["status"] = (
            merged["OS_STATUS"].map({"0:LIVING": 0, "1:DECEASED": 1}).astype(int)
        )
        merged["os_months"] = pd.to_numeric(merged["OS_MONTHS"], errors="coerce")
        merged = merged.dropna(subset=["slide_id", "status", "os_months"])
        merged = merged.loc[merged["os_months"] > 0.0].reset_index(drop=True)
        merged["time_bin"] = pd.qcut(
            merged["os_months"], q=3, labels=False, duplicates="drop"
        ).astype(int)

        feature_matrix = np.asarray(adata.X, dtype=np.float32)[
            merged["feature_row_index"].to_numpy(dtype=np.int64)
        ]
        input_dim = int(feature_matrix.shape[1])
        bag_lengths: dict[str, int] = {}
        for row_index, row in merged.iterrows():
            slide_id = str(row["slide_id"])
            bag_tensor = (
                torch.from_numpy(feature_matrix[row_index])
                .reshape(1, input_dim)
                .float()
            )
            torch.save(bag_tensor, feature_dir / f"{slide_id}.pt")
            bag_lengths[slide_id] = 1
        attach_smoke_outputs(
            metadata,
            step_name="hf_prepare_survival_bags",
            intermediate={
                "source_h5ad": smoke_assets.survival_h5ad,
                "source_survival_csv": smoke_assets.survival_csv,
            },
            final={"feature_dir": feature_dir},
        )

    metadata_csv = root_dir / "survival_metadata.csv"
    merged.loc[:, ["slide_id", "status", "os_months", "time_bin"]].to_csv(
        metadata_csv, index=False
    )
    return PreparedBagWorkspace(
        root_dir=root_dir,
        feature_dir=feature_dir,
        metadata_csv=metadata_csv,
        slide_ids=merged["slide_id"].astype(str).tolist(),
        input_dim=input_dim,
        bag_lengths=bag_lengths,
        metrics_path=metrics_dir / "hf_prepare_survival_bags.metrics.json",
    )


@pytest.fixture(scope="session")
def slide_level_feature_matrix(
    extracted_wsi_workspace: ExtractedWsiWorkspace,
) -> tuple[list[str], np.ndarray]:
    """Expose deterministic slide-level features pooled from tile artifacts."""
    return aggregate_slide_feature_matrix(
        extracted_wsi_workspace.artifact_paths,
        bag_id=extracted_wsi_workspace.bag_id,
        extractor_name=extracted_wsi_workspace.extractor_name,
    )


@pytest.fixture(scope="session")
def gtex_survival_workspace(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path_factory: pytest.TempPathFactory,
) -> PreparedBagWorkspace:
    """Derive survival labels from GTEx bag metadata, reusing existing .pt features.

    Reuses the tile-level feature bags already produced by
    ``extracted_bag_workspace``, so no additional feature extraction is needed.
    Synthetic survival labels are derived deterministically from GTEx metadata:

    - ``status`` = ``binary_label`` (calcification = event, clean_specimens = censored)
    - ``os_months`` = ``(multiclass_label + 1) * 24`` (age-bracket proxy for
      observation time: 20-29 → 24 m, 30-39 → 48 m, …, 60-69 → 120 m)
    - ``time_bin`` = quantile-binned ``os_months`` into three discrete bins
    """
    root_dir = tmp_path_factory.mktemp("gtex_survival")
    metrics_dir = root_dir / "metrics"

    metadata_df = pd.read_csv(extracted_bag_workspace.metadata_csv).copy()
    metadata_df["status"] = metadata_df["binary_label"].astype(int)
    metadata_df["os_months"] = (metadata_df["multiclass_label"] + 1) * 24.0
    metadata_df["time_bin"] = pd.qcut(
        metadata_df["os_months"], q=3, labels=False, duplicates="drop"
    ).astype(int)

    with capture_smoke_metrics(
        metrics_dir,
        step_name="hf_prepare_gtex_survival_bags",
        metadata={"num_slides": len(metadata_df)},
    ) as sm:
        attach_smoke_outputs(
            sm,
            step_name="hf_prepare_gtex_survival_bags",
            intermediate={"source_bag_metadata": extracted_bag_workspace.metadata_csv},
        )

    survival_csv = root_dir / "gtex_survival_metadata.csv"
    metadata_df[["slide_id", "status", "os_months", "time_bin"]].to_csv(
        survival_csv, index=False
    )
    return PreparedBagWorkspace(
        root_dir=root_dir,
        feature_dir=extracted_bag_workspace.feature_dir,
        metadata_csv=survival_csv,
        slide_ids=metadata_df["slide_id"].tolist(),
        input_dim=extracted_bag_workspace.input_dim,
        bag_lengths=extracted_bag_workspace.bag_lengths,
        metrics_path=metrics_dir / "hf_prepare_gtex_survival_bags.metrics.json",
    )


@pytest.fixture(scope="session")
def retrieval_wsi_datasets(extracted_wsi_workspace: ExtractedWsiWorkspace) -> RetrievalDatasets:
    """Pre-built real SlideRetrievalBagDataset instances from GTEx H5 artifacts."""
    from pathbench.config.config import DatasetEntry
    from pathbench.core.experiments.combinations import ComboConfig
    from pathbench.core.datasets.bag_dataset import SlideRetrievalBagDataset

    annotations_df = pd.read_csv(extracted_wsi_workspace.annotations_csv)
    all_slide_ids = annotations_df["slide"].astype(str).tolist()

    combo_cfg = ComboConfig(
        tile_px=224,
        tile_mpp=1.0,
        feature_extraction="resnet18",
    )

    def _make(name: str, slide_ids: list[str]) -> SlideRetrievalBagDataset:
        filtered = annotations_df[annotations_df["slide"].isin(slide_ids)].copy()
        filtered = filtered.copy()
        filtered["dataset"] = name
        ds_cfg = DatasetEntry(
            name=name,
            slides_dir=str(extracted_wsi_workspace.slides_dir),
            artifacts_dir=str(extracted_wsi_workspace.artifacts_dir),
            used_for="all",
        )
        return SlideRetrievalBagDataset(ds_cfg, filtered, combo_cfg)

    ref_ids = all_slide_ids[:10]
    qry_ids = all_slide_ids[10:20]
    single_id = all_slide_ids[:1]

    return RetrievalDatasets(
        reference=_make("gtex_ref", ref_ids),
        query=_make("gtex_qry", qry_ids),
        single_ref=_make("gtex_single", single_id),
        all_slide_ids=all_slide_ids,
    )
