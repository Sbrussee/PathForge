from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from tests.smoke._smoke_dataset import (
    build_gtex_smoke_annotations,
    default_smoke_cache_dir,
    merge_survival_metadata,
)


def test_default_smoke_cache_dir_falls_back_to_tmp_for_unwritable_home_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PATHFORGE_SMOKE_CACHE", raising=False)
    monkeypatch.setenv("TMPDIR", "/tmp/pathforge-tests")
    monkeypatch.setattr(
        os,
        "access",
        lambda path, mode: False if str(path).startswith("/readonly-home") else True,
    )
    monkeypatch.setenv("HOME", "/readonly-home")

    assert default_smoke_cache_dir() == Path("/tmp/pathforge-tests/pathforge_smoke")


def test_build_gtex_smoke_annotations_uses_requested_slide_ids(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "GTEx_artery_dataset.csv.gz"
    pd.DataFrame(
        {
            "FILE_NAME": [
                "GTEX-1117F-0526.svs",
                "GTEX-2222A-0001.svs",
            ],
            "PATIENT_ID": ["P1", "P2"],
            "tissue_type": ["artery_aorta", "artery_coronary"],
        }
    ).to_csv(metadata_csv, index=False, compression="gzip")

    annotations = build_gtex_smoke_annotations(
        metadata_csv,
        slide_ids=["GTEX-1117F-0526"],
    )

    assert annotations.to_dict(orient="records") == [
        {
            "dataset": "hf_gtex_artery",
            "slide": "GTEX-1117F-0526",
            "patient": "P1",
            "category": "artery_aorta",
            "age_bracket": "unknown",
        }
    ]


def test_build_gtex_smoke_annotations_raises_for_missing_slide(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "GTEx_artery_dataset.csv.gz"
    pd.DataFrame({"FILE_NAME": ["GTEX-1117F-0526.svs"]}).to_csv(
        metadata_csv,
        index=False,
        compression="gzip",
    )

    with pytest.raises(ValueError, match="missing requested slides"):
        build_gtex_smoke_annotations(
            metadata_csv,
            slide_ids=["GTEX-NOT-PRESENT"],
        )


def test_build_gtex_smoke_annotations_can_fallback_to_dataset_level(
    tmp_path: Path,
) -> None:
    metadata_csv = tmp_path / "GTEx_artery_dataset.csv.gz"
    pd.DataFrame(
        {
            "Tissue Sample Id": ["GTEX-15RJE-0526"],
            "Pathology Categories": ["artery"],
        }
    ).to_csv(metadata_csv, index=False, compression="gzip")

    annotations = build_gtex_smoke_annotations(
        metadata_csv,
        slide_ids=["GTEX-1117F-0526"],
        strict=False,
    )

    assert annotations.to_dict(orient="records") == [
        {
            "dataset": "hf_gtex_artery",
            "slide": "GTEX-1117F-0526",
            "patient": "GTEX-1117F-0526",
            "category": "artery",
            "age_bracket": "unknown",
        }
    ]


def test_merge_survival_metadata_joins_on_slide_filename(monkeypatch) -> None:
    feature_obs = pd.DataFrame(
        {
            "FILE_NAME": [
                "TCGA-01-0001-01Z-00-DX1.aaa.svs",
                "TCGA-01-0002-01Z-00-DX1.bbb.svs",
            ]
        }
    )
    survival_csv = Path("unused.csv")
    survival_df = pd.DataFrame(
        {
            "FILE_NAME": [
                "TCGA-01-0002-01Z-00-DX1.bbb.svs",
                "TCGA-01-0001-01Z-00-DX1.aaa.svs",
            ],
            "OS_STATUS": ["0:LIVING", "1:DECEASED"],
            "OS_MONTHS": [12.0, 24.0],
        }
    )

    monkeypatch.setattr(pd, "read_csv", lambda _: survival_df.copy())
    merged = merge_survival_metadata(feature_obs, survival_csv)

    assert merged["slide_id"].tolist() == [
        "TCGA-01-0001-01Z-00-DX1",
        "TCGA-01-0002-01Z-00-DX1",
    ]
    assert merged["feature_row_index"].tolist() == [0, 1]
    assert merged["OS_STATUS"].tolist() == ["1:DECEASED", "0:LIVING"]


def test_merge_survival_metadata_raises_when_no_rows_match(monkeypatch) -> None:
    feature_obs = pd.DataFrame({"FILE_NAME": ["TCGA-01-9999-01Z-00-DX1.zzz.svs"]})
    survival_df = pd.DataFrame(
        {
            "FILE_NAME": ["TCGA-01-0001-01Z-00-DX1.aaa.svs"],
            "OS_STATUS": ["1:DECEASED"],
            "OS_MONTHS": [24.0],
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda _: survival_df.copy())

    with pytest.raises(ValueError, match="No feature rows matched"):
        merge_survival_metadata(feature_obs, Path("unused.csv"))
