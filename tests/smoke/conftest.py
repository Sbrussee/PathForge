"""Session-scoped fixtures for realistic PathBench smoke tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ._smoke_dataset import (
    ExtractedWsiWorkspace,
    PreparedBagWorkspace,
    aggregate_slide_feature_matrix,
    capture_smoke_metrics,
    default_smoke_cache_dir,
    download_smoke_assets,
    link_or_copy,
    read_h5_feature_matrix,
    DownloadedSmokeAssets,
)
from ._smoke_training import register_smoke_components


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

    slide_rows = [
        ("hf_smoke", "sample", "P_SAMPLE", "sample"),
        ("hf_smoke", "GTEX-1117F-0526", "P_GTEX", "artery"),
        ("hf_smoke", "lung_carcinoma", "P_LUNG", "tumor"),
    ]
    filename_by_slide = {
        "sample": "sample.svs",
        "GTEX-1117F-0526": "GTEX-1117F-0526.svs",
        "lung_carcinoma": "lung_carcinoma.ndpi",
    }
    for slide_id, filename in filename_by_slide.items():
        link_or_copy(smoke_assets.slides[filename], slides_dir / filename)

    annotations_csv = root_dir / "annotations.csv"
    pd.DataFrame(
        slide_rows,
        columns=["dataset", "slide", "patient", "category"],
    ).to_csv(annotations_csv, index=False)

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
    ):
        result = policy.execute()
        assert result["status"] == "feature_extraction_done"

    bag_id = f"{tile_px}px_{tile_mpp:g}mpp"
    artifact_paths = {
        slide_id: artifacts_dir / f"{slide_id}.h5" for _, slide_id, _, _ in slide_rows
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

    label_rows: list[dict[str, int | str]] = []
    bag_lengths: dict[str, int] = {}
    input_dim = 0

    binary_map = {
        "sample": 0,
        "GTEX-1117F-0526": 0,
        "lung_carcinoma": 1,
    }
    multiclass_map = {
        "sample": 0,
        "GTEX-1117F-0526": 1,
        "lung_carcinoma": 2,
    }

    with capture_smoke_metrics(
        metrics_dir,
        step_name="hf_prepare_extracted_bags",
        metadata={"num_slides": len(extracted_wsi_workspace.artifact_paths)},
    ):
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
            label_rows.append(
                {
                    "slide_id": slide_id,
                    "binary_label": binary_map[slide_id],
                    "multiclass_label": multiclass_map[slide_id],
                }
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
    ):
        adata = ad.read_h5ad(smoke_assets.survival_h5ad)
        obs = adata.obs.copy()
        if "status" not in obs.columns:
            obs["status"] = (
                obs["OS_STATUS"].map({"0:LIVING": 0, "1:DECEASED": 1}).astype(int)
            )
        obs["os_months"] = pd.to_numeric(obs["OS_MONTHS"], errors="coerce")
        obs["slide_id"] = (
            obs["FILE_NAME"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
        )
        obs = obs.dropna(subset=["slide_id", "status", "os_months"]).reset_index(
            drop=True
        )
        obs["time_bin"] = pd.qcut(
            obs["os_months"], q=3, labels=False, duplicates="drop"
        ).astype(int)

        feature_matrix = np.asarray(adata.X, dtype=np.float32)
        input_dim = int(feature_matrix.shape[1])
        bag_lengths: dict[str, int] = {}
        for row_index, row in obs.iterrows():
            slide_id = str(row["slide_id"])
            bag_tensor = (
                torch.from_numpy(feature_matrix[row_index])
                .reshape(1, input_dim)
                .float()
            )
            torch.save(bag_tensor, feature_dir / f"{slide_id}.pt")
            bag_lengths[slide_id] = 1

    metadata_csv = root_dir / "survival_metadata.csv"
    obs.loc[:, ["slide_id", "status", "os_months", "time_bin"]].to_csv(
        metadata_csv, index=False
    )
    return PreparedBagWorkspace(
        root_dir=root_dir,
        feature_dir=feature_dir,
        metadata_csv=metadata_csv,
        slide_ids=obs["slide_id"].astype(str).tolist(),
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
