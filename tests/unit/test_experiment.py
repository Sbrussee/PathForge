# tests/unit/test_experiment.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pathbench.config.config import BenchmarkParamEntry, Config
from pathbench.core.datasets.factory import build_wsi_datasets
from pathbench.core.experiments.base import Experiment
from pathbench.core.experiments.combinations import ComboConfig, build_combinations
from pathbench.utils.registries import FEATURE_EXTRACTORS


# -----------------------------------------------------------------------------
# Registry setup (feature extractor names must be registered for Config validation)
# -----------------------------------------------------------------------------

EXTRACTOR_1 = "exp_ut_extractor_1"
EXTRACTOR_2 = "exp_ut_extractor_2"


def _ensure_feature_extractor_registered(name: str) -> None:
    # FEATURE_EXTRACTORS is a global registry; avoid duplicate registration errors
    if hasattr(FEATURE_EXTRACTORS, "is_available") and FEATURE_EXTRACTORS.is_available(name):
        return

    @FEATURE_EXTRACTORS.register(name)
    def _dummy_extractor():  # pragma: no cover
        return "ok"


_ensure_feature_extractor_registered(EXTRACTOR_1)
_ensure_feature_extractor_registered(EXTRACTOR_2)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _write_annotations_csv(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _make_cfg(
    *,
    tmp_path: Path,
    annotation_file: Path,
    project_name: str = "proj",
    project_root_base: Path | None = None,
    mode: str = "feature_extraction",
    datasets: list[dict] | None = None,
    benchmark_parameters: dict | None = None,
) -> Config:
    if project_root_base is None:
        project_root_base = tmp_path / "projects_base"
    project_root_base.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = []

    if benchmark_parameters is None:
        benchmark_parameters = {
            "tile_px": [256],
            "tile_mpp": [0.5],
            "feature_extraction": [EXTRACTOR_1],
        }

    cfg_data = {
        "experiment": {
            "project_name": project_name,
            "annotation_file": str(annotation_file),
            "project_root": str(project_root_base.resolve()),
            "mode": mode,
            # task omitted on purpose (allowed in feature_extraction mode)
        },
        "slide_processing": {"backend": "lazyslide"},
        "datasets": datasets,
        "benchmark_parameters": benchmark_parameters,
    }
    return Config.model_validate(cfg_data)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_project_root_uses_absolute_base(tmp_path: Path) -> None:
    ann_src = tmp_path / "ann.csv"
    _write_annotations_csv(
        ann_src,
        [{"dataset": "ds", "slide": "S1", "patient": "P1", "category": "C1"}],
    )

    base = tmp_path / "abs_base"
    cfg = _make_cfg(tmp_path=tmp_path, annotation_file=ann_src, project_root_base=base, project_name="myproj")

    exp = Experiment(cfg)
    assert Path(exp.project_root).is_absolute()
    assert Path(exp.project_root) == (base / "myproj").resolve()


def test_project_root_requires_absolute_path(tmp_path: Path) -> None:
    ann_src = tmp_path / "ann.csv"
    _write_annotations_csv(
        ann_src,
        [{"dataset": "ds", "slide": "S1", "patient": "P1", "category": "C1"}],
    )

    cfg_data = {
        "experiment": {
            "project_name": "proj",
            "annotation_file": str(ann_src),
            "project_root": "relative/path",  # must be absolute
            "mode": "feature_extraction",
        },
        "slide_processing": {"backend": "lazyslide"},
        "datasets": [],
        "benchmark_parameters": {
            "tile_px": [256],
            "tile_mpp": [0.5],
            "feature_extraction": [EXTRACTOR_1],
        },
    }

    cfg = Config.model_validate(cfg_data)

    with pytest.raises(ValueError) as excinfo:
        _ = Experiment(cfg)

    assert "must be an absolute path" in str(excinfo.value)


def test_prepare_project_creates_project_json_and_copies_annotations(tmp_path: Path) -> None:
    ann_src = tmp_path / "ann_src.csv"
    src_rows = [{"dataset": "ds", "slide": "S1", "patient": "P1", "category": "C1"}]
    _write_annotations_csv(ann_src, src_rows)

    base = tmp_path / "base"
    cfg = _make_cfg(tmp_path=tmp_path, annotation_file=ann_src, project_root_base=base, project_name="projA")

    exp = Experiment(cfg)
    root = Path(exp.project_root)

    assert root.is_dir()
    assert (root / "project.json").is_file()
    assert (root / "annotations.csv").is_file()

    # project.json sanity
    pj = json.loads((root / "project.json").read_text(encoding="utf-8"))
    assert pj["project_name"] == "projA"
    assert "created_at" in pj
    assert "annotation_source" in pj

    # annotations.csv copied content
    copied = pd.read_csv(root / "annotations.csv")
    assert copied.to_dict(orient="records") == src_rows


def test_existing_project_json_mismatch_raises(tmp_path: Path) -> None:
    ann_src = tmp_path / "ann.csv"
    _write_annotations_csv(
        ann_src,
        [{"dataset": "ds", "slide": "S1", "patient": "P1", "category": "C1"}],
    )

    base = tmp_path / "base"
    cfg = _make_cfg(tmp_path=tmp_path, annotation_file=ann_src, project_root_base=base, project_name="projA")

    exp1 = Experiment(cfg)
    root = Path(exp1.project_root)
    pj_path = root / "project.json"

    # Tamper with project.json to force mismatch
    pj = json.loads(pj_path.read_text(encoding="utf-8"))
    pj["project_name"] = "different"
    pj_path.write_text(json.dumps(pj, indent=2), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        _ = Experiment(cfg)

    assert "does not match" in str(excinfo.value)


def test_load_annotations_reads_copied_file(tmp_path: Path) -> None:
    ann_src = tmp_path / "ann.csv"
    rows = [
        {"dataset": "ds", "slide": "S1", "patient": "P1", "category": "C1"},
        {"dataset": "ds", "slide": "S2", "patient": "P2", "category": "C2"},
    ]
    _write_annotations_csv(ann_src, rows)

    cfg = _make_cfg(tmp_path=tmp_path, annotation_file=ann_src)
    exp = Experiment(cfg)

    df = exp.load_annotations()
    assert list(df.columns) == ["dataset", "slide", "patient", "category"]
    assert df.to_dict(orient="records") == rows


def test_build_combinations_builds_expected_grid(tmp_path: Path) -> None:
    ann_src = tmp_path / "ann.csv"
    _write_annotations_csv(
        ann_src,
        [{"dataset": "ds", "slide": "S1", "patient": "P1", "category": "C1"}],
    )

    cfg = _make_cfg(
        tmp_path=tmp_path,
        annotation_file=ann_src,
        benchmark_parameters={
            "tile_px": [256, 512],
            "tile_mpp": [0.5, 1.0],
            "feature_extraction": [EXTRACTOR_1, EXTRACTOR_2],
        },
    )
    exp = Experiment(cfg)

    combos = build_combinations(
        cfg=exp.cfg,
        keys=["feature_extraction", "tile_px", "tile_mpp"],
    )
    assert len(combos) == 2 * 2 * 2  # 8

    # spot-check attributes exist
    c0 = combos[0]
    assert hasattr(c0, "feature_extraction")
    assert hasattr(c0, "tile_px")
    assert hasattr(c0, "tile_mpp")
    assert c0.get("tile_px") in {256, 512}
    assert c0.get_hyperparams("tile_px") == {}


def test_build_combinations_missing_key_raises(tmp_path: Path) -> None:
    ann_src = tmp_path / "ann.csv"
    _write_annotations_csv(
        ann_src,
        [{"dataset": "ds", "slide": "S1", "patient": "P1", "category": "C1"}],
    )

    cfg = _make_cfg(tmp_path=tmp_path, annotation_file=ann_src)
    exp = Experiment(cfg)

    with pytest.raises(AttributeError):
        build_combinations(cfg=exp.cfg, keys=["does_not_exist"])


def test_build_combinations_empty_values_raises(tmp_path: Path) -> None:
    ann_src = tmp_path / "ann.csv"
    _write_annotations_csv(
        ann_src,
        [{"dataset": "ds", "slide": "S1", "patient": "P1", "category": "C1"}],
    )

    cfg = _make_cfg(
        tmp_path=tmp_path,
        annotation_file=ann_src,
        benchmark_parameters={
            "tile_px": [],  # empty -> should fail
            "tile_mpp": [0.5],
            "feature_extraction": [EXTRACTOR_1],
        },
    )
    exp = Experiment(cfg)

    with pytest.raises(ValueError) as excinfo:
        build_combinations(cfg=exp.cfg, keys=["tile_px"])
    assert "is empty" in str(excinfo.value)


def test_build_combinations_preserves_hyperparams_on_combo_cfg(tmp_path: Path) -> None:
    ann_src = tmp_path / "ann.csv"
    _write_annotations_csv(
        ann_src,
        [{"dataset": "ds", "slide": "S1", "patient": "P1", "category": "C1"}],
    )

    cfg = _make_cfg(
        tmp_path=tmp_path,
        annotation_file=ann_src,
        mode="feature_extraction",
        benchmark_parameters={
            "tile_px": [256],
            "tile_mpp": [0.5],
            "feature_extraction": [EXTRACTOR_1],
            "search_strategy": [
                {"yottixel": {"k": 3}},
                {"yottixel": {"k": 5}},
            ],
            "retrieval_representation": [
                "sdm_features",
                {"hshr_features": {"n_patches": 10}},
            ],
        },
    )
    exp = Experiment(cfg)

    combos = build_combinations(
        cfg=exp.cfg,
        keys=["search_strategy", "retrieval_representation"],
    )

    assert len(combos) == 4
    assert {combo.get_hyperparams("search_strategy")["k"] for combo in combos} == {3, 5}
    assert {
        tuple(sorted(combo.get_hyperparams("retrieval_representation").items()))
        for combo in combos
    } == {(), (("n_patches", 10),)}


def test_combo_cfg_get_and_get_hyperparams_round_trip() -> None:
    combo_cfg = ComboConfig.from_keys_values(
        ["feature_extraction"],
        [BenchmarkParamEntry(value="uni", hyperparams={"family": "foundation"})],
    )

    assert combo_cfg.get("feature_extraction") == "uni"
    assert combo_cfg.get("missing", "fallback") == "fallback"
    assert combo_cfg.get_hyperparams("feature_extraction") == {"family": "foundation"}
    assert combo_cfg.get_hyperparams("missing") == {}


def test_build_wsi_datasets_skips_ignore(tmp_path: Path) -> None:
    ann_src = tmp_path / "ann.csv"
    _write_annotations_csv(
        ann_src,
        [
            {"dataset": "keep", "slide": "S1", "patient": "P1", "category": "C1"},
            {"dataset": "ignoreme", "slide": "S2", "patient": "P2", "category": "C2"},
        ],
    )

    slides_keep = tmp_path / "slides_keep"
    slides_keep.mkdir()
    artifacts_keep = tmp_path / "art_keep"

    slides_ignore = tmp_path / "slides_ignore"
    slides_ignore.mkdir()
    artifacts_ignore = tmp_path / "art_ignore"

    cfg = _make_cfg(
        tmp_path=tmp_path,
        annotation_file=ann_src,
        datasets=[
            {
                "name": "keep",
                "slides_dir": str(slides_keep),
                "artifacts_dir": str(artifacts_keep),
                "used_for": "all",
            },
            {
                "name": "ignoreme",
                "slides_dir": str(slides_ignore),
                "artifacts_dir": str(artifacts_ignore),
                "used_for": "ignore",
            },
        ],
    )

    exp = Experiment(cfg)
    datasets = build_wsi_datasets(cfg=cfg, annotations_df=exp.load_annotations())

    assert len(datasets) == 1
    assert datasets[0].name == "keep"
    assert datasets[0].artifacts_dir.is_dir()
