from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd

from pathbench.adapters import tcga_tools as tcga_adapter
from pathbench.config.config import Config
from tests.conftest import DUMMY_FE


def _write_fake_tcga_dataset(
    *,
    root: Path,
    dataset_name: str,
    raw: bool,
    patients: list[tuple[str, str, str]],
) -> dict[str, Path]:
    dataset_dir = root / dataset_name
    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    files_df = pd.DataFrame(
        [
            {
                "file_name": f"{slide}.svs",
                "cases.submitter_id": patient,
                "cases.case_id": case_id,
            }
            for patient, case_id, slide in patients
        ]
    )
    clinical_df = pd.DataFrame(
        [
            {
                "submitter_id": patient,
                "case_id": case_id,
                "label": f"label-{patient}",
            }
            for patient, case_id, _slide in patients
        ]
    )

    files_csv = dataset_dir / "files_metadata.csv"
    clinical_csv = dataset_dir / "clinical.csv"
    files_df.to_csv(files_csv, index=False)
    clinical_df.to_csv(clinical_csv, index=False)

    if not raw:
        for _patient, _case_id, slide in patients:
            (data_dir / f"{slide}.svs").write_bytes(b"slide")

    return {
        "files_csv": files_csv,
        "clinical_csv": clinical_csv,
        "data_dir": data_dir,
        "dataset_dir": dataset_dir,
    }


def test_from_yaml_resolves_tcga_dataset_into_local_annotations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "PathBench_2.0"
    repo_root.mkdir()

    patients = [("TCGA-01", "CASE-01", "slide_01")]

    def fake_download(**kwargs):
        return _write_fake_tcga_dataset(
            root=Path(kwargs["output_dir"]),
            dataset_name=str(kwargs["dataset_name"]),
            raw=bool(kwargs["raw"]),
            patients=patients,
        )

    def fake_list_datasets(*, source: str, as_dataframe: bool = True, **_kwargs):
        assert as_dataframe is True
        column_name = "project_id" if source == "gdc" else "dataset_name"
        return pd.DataFrame([{column_name: "TCGA-LUSC"}])

    monkeypatch.setattr(tcga_adapter, "_pathbench_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        tcga_adapter, "_load_tcga_tools_download", lambda: fake_download
    )
    monkeypatch.setattr(
        tcga_adapter, "_load_tcga_tools_list_datasets", lambda: fake_list_datasets
    )

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        dedent(
            f"""
            experiment:
              project_name: tcga_remote
              annotation_file: ignored.csv
              project_root: "{(tmp_path / "projects").resolve()}"
              mode: feature_extraction

            slide_processing:
              backend: lazyslide

            datasets:
              - source: gdc
                dataset_names: ["TCGA-LUSC"]
                annotation_column: label
                metadata_table: clinical_csv
                annotations: ["clinical"]
                used_for: training

            benchmark_parameters:
              tile_px: [256]
              tile_mpp: [0.5]
              feature_extraction: ["{DUMMY_FE}"]
              mil: []
            """
        ).strip(),
        encoding="utf-8",
    )

    cfg = Config.from_yaml(cfg_path)

    assert len(cfg.datasets) == 1
    dataset_cfg = cfg.datasets[0]
    assert dataset_cfg.name == "TCGA-LUSC"
    assert (
        Path(dataset_cfg.slides_dir)
        == (repo_root / "datasets" / "TCGA-LUSC" / "data").resolve()
    )

    annotation_file = Path(cfg.experiment.annotation_file)
    assert (
        annotation_file
        == (repo_root / "datasets" / "pathbench_external_annotations.csv").resolve()
    )
    annotations = pd.read_csv(annotation_file)
    assert annotations.loc[0, "dataset"] == "TCGA-LUSC"
    assert annotations.loc[0, "category"] == "label-TCGA-01"
    assert annotations.loc[0, "selected_annotation_column"] == "label"
    assert (
        Path(annotations.loc[0, "wsi_path"])
        == (repo_root / "datasets" / "TCGA-LUSC" / "data" / "slide_01.svs").resolve()
    )


def test_from_yaml_splits_remote_dataset_across_multiple_roles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "PathBench_2.0"
    repo_root.mkdir()

    patients = [
        ("TCGA-01", "CASE-01", "slide_01"),
        ("TCGA-02", "CASE-02", "slide_02"),
        ("TCGA-03", "CASE-03", "slide_03"),
        ("TCGA-04", "CASE-04", "slide_04"),
    ]

    def fake_download(**kwargs):
        return _write_fake_tcga_dataset(
            root=Path(kwargs["output_dir"]),
            dataset_name=str(kwargs["dataset_name"]),
            raw=bool(kwargs["raw"]),
            patients=patients,
        )

    def fake_list_datasets(*, source: str, as_dataframe: bool = True, **_kwargs):
        assert as_dataframe is True
        column_name = "project_id" if source == "gdc" else "dataset_name"
        return pd.DataFrame([{column_name: "TCGA-LUAD"}])

    monkeypatch.setattr(tcga_adapter, "_pathbench_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        tcga_adapter, "_load_tcga_tools_download", lambda: fake_download
    )
    monkeypatch.setattr(
        tcga_adapter, "_load_tcga_tools_list_datasets", lambda: fake_list_datasets
    )

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        dedent(
            f"""
            experiment:
              project_name: tcga_split
              annotation_file: ignored.csv
              project_root: "{(tmp_path / "projects").resolve()}"
              mode: feature_extraction
              val_fraction: 0.25

            slide_processing:
              backend: lazyslide

            datasets:
              - source: gdc
                dataset_names: ["TCGA-LUAD"]
                annotation_column: label
                metadata_table: clinical_csv
                annotations: ["clinical"]
                used_for: ["training", "testing"]

            benchmark_parameters:
              tile_px: [256]
              tile_mpp: [0.5]
              feature_extraction: ["{DUMMY_FE}"]
              mil: []
            """
        ).strip(),
        encoding="utf-8",
    )

    cfg = Config.from_yaml(cfg_path)

    assert {dataset.name for dataset in cfg.datasets} == {
        "TCGA-LUAD__training",
        "TCGA-LUAD__testing",
    }
    annotations = pd.read_csv(cfg.experiment.annotation_file)
    training_patients = set(
        annotations.loc[annotations["dataset"] == "TCGA-LUAD__training", "patient"]
    )
    testing_patients = set(
        annotations.loc[annotations["dataset"] == "TCGA-LUAD__testing", "patient"]
    )
    assert training_patients
    assert testing_patients
    assert training_patients.isdisjoint(testing_patients)
