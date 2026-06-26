from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pathbench.core.experiments.combinations import ComboConfig
from pathbench.slide_retrieval.representation_strategies.panther import (
    PantherAssignment,
)
from pathbench.slide_retrieval.representation_strategies.prototype_bundles import (
    SalvPrototypeBundleResolver,
)
from pathbench.slide_retrieval.representation_strategies.registry import (
    build_representation_strategy,
    import_representation_strategy_modules,
    is_representation_strategy_available,
)
from pathbench.slide_retrieval.representation_strategies.strategies.pbms import (
    PBMSFeatures,
)


def test_salv_prototype_bundle_resolver_matches_combo(tmp_path: Path) -> None:
    run_dir = _write_salv_bundle(
        tmp_path,
        run_name="run-a",
        tile_px=256,
        tile_mpp=0.5,
        feature_extraction="uni2",
        color_norm="reinhard",
    )

    bundle = SalvPrototypeBundleResolver(tmp_path).resolve(
        tile_px=256,
        tile_mpp=0.5,
        feature_extraction="uni2",
        color_norm="reinhard",
    )

    assert bundle.run_dir == run_dir
    assert bundle.prototype_matrix.shape == (3, 2)
    assert bundle.labels == ("include", "exclude", "include")
    assert bundle.feature_dim == 2


def test_salv_prototype_bundle_resolver_uses_explicit_feature_run_path(
    tmp_path: Path,
) -> None:
    _write_salv_bundle(
        tmp_path,
        run_name="run-a",
        tile_px=256,
        tile_mpp=0.5,
        feature_extraction="uni2",
        color_norm="reinhard",
    )
    run_b = _write_salv_bundle(
        tmp_path,
        run_name="run-b",
        tile_px=256,
        tile_mpp=0.5,
        feature_extraction="uni2",
        color_norm="reinhard",
    )

    bundle = SalvPrototypeBundleResolver(tmp_path).resolve(
        tile_px=256,
        tile_mpp=0.5,
        feature_extraction="uni2",
        color_norm="reinhard",
        prototype_run_name="kmeans/run-b",
    )

    assert bundle.run_dir == run_b


def test_pbms_features_excludes_labeled_prototypes_and_saves_panther_data(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_salv_bundle(
        tmp_path,
        run_name="run-a",
        tile_px=256,
        tile_mpp=0.5,
        feature_extraction="uni2",
        color_norm=None,
    )

    class _FakeAssigner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def assign(self, features: np.ndarray) -> PantherAssignment:
            return PantherAssignment(
                resp=np.asarray(
                    [
                        [0.9, 0.1, 0.0],
                        [0.1, 0.8, 0.1],
                        [0.1, 0.1, 0.8],
                        [0.2, 0.7, 0.1],
                    ],
                    dtype=np.float32,
                ),
                top1=np.asarray([0, 1, 2, 1], dtype=np.int32),
                top1_prob=np.asarray([0.9, 0.8, 0.8, 0.7], dtype=np.float32),
                slide_embed=np.asarray([1.0, 2.0], dtype=np.float32),
                proto_mean=np.ones((3, 2), dtype=np.float32),
                proto_cov=np.full((3, 2), 0.1, dtype=np.float32),
                proto_prob=np.asarray([0.2, 0.3, 0.5], dtype=np.float32),
            )

    monkeypatch.setattr(
        "pathbench.slide_retrieval.representation_strategies.strategies.pbms.PantherPrototypeAssigner",
        _FakeAssigner,
    )

    cfg = SimpleNamespace(
        experiment=SimpleNamespace(random_state=7),
        slide_retrieval=SimpleNamespace(prototypes_root=str(tmp_path)),
    )
    combo_cfg = ComboConfig(
        tile_px=256,
        tile_mpp=0.5,
        feature_extraction="uni2",
        color_norm=None,
    )
    strategy = PBMSFeatures(
        params={"perc_selected": 100.0, "save_resp": True},
        config=cfg,
    )
    strategy.prepare_for_combo(
        combo_cfg=combo_cfg,
        feature_name="uni2",
        tiling_id="256px_0.5mpp",
    )

    features = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
        dtype=np.float32,
    )
    coords = np.asarray([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.int32)
    representation = strategy.run(
        bag=features,
        sample=SimpleNamespace(sample_id="slide-1"),
        coords=coords,
        combo_cfg=combo_cfg,
        tiling_id="256px_0.5mpp",
    )

    np.testing.assert_allclose(representation.data, features[[0, 2]])
    np.testing.assert_array_equal(
        representation.additional_data["selected_indices"],
        np.asarray([0, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        representation.additional_data["group_ids"],
        np.asarray([1, 0, 3, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        representation.additional_data["prototype_ids"],
        np.asarray([0, 1, 2, 1], dtype=np.int32),
    )
    assert representation.additional_data["panther_resp"].dtype == np.float16
    assert representation.additional_data["panther_proto_mean"].shape == (3, 2)
    assert "prototype_bundle_hash" in strategy.hyperparam_values()


def test_pbms_features_registers_strategy() -> None:
    import_representation_strategy_modules()

    assert is_representation_strategy_available("pbms-features")
    assert isinstance(build_representation_strategy("pbms-features"), PBMSFeatures)


def _write_salv_bundle(
    root: Path,
    *,
    run_name: str,
    tile_px: int,
    tile_mpp: float,
    feature_extraction: str,
    color_norm: str | None,
) -> Path:
    feature_name = feature_extraction if color_norm is None else f"{feature_extraction}_{color_norm}"
    run_dir = root / f"{int(tile_px)}px_{float(tile_mpp):g}mpp" / feature_name / "kmeans" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "prototype_discovery.json").write_text(
        json.dumps(
            {
                "algorithm": "kmeans",
                "feature_dim": 2,
                "params": {"normalize": False},
                "prototypes": [
                    {"prototype_id": 0, "label": "no_label", "vector": [1.0, 0.0]},
                    {"prototype_id": 1, "label": "no_label", "vector": [0.0, 1.0]},
                    {"prototype_id": 2, "label": "no_label", "vector": [2.0, 0.0]},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "proto_labels.json").write_text(
        json.dumps(
            {
                "prototypes": [
                    {"prototype_id": 0, "label": "include"},
                    {"prototype_id": 1, "label": "exclude"},
                    {"prototype_id": 2, "label": "include"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "prototype_discovery_summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "experiment": {
                        "feature_source": {
                            "tile_px": tile_px,
                            "mpp": tile_mpp,
                            "feat_extractor": feature_extraction,
                            "color_norm": color_norm,
                        }
                    }
                },
                "discovery": {
                    "feature_dim": 2,
                    "n_prototypes": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir.resolve()
