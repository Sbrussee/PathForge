from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.search_strategies.registry import (
    get_search_strategy,
    import_search_strategy_modules,
)
from pathbench.slide_retrieval.search_strategies.strategies.sish.sish_search import (
    SISHSearch,
)
from pathbench.slide_retrieval.types import RetrievalItemMetadata


def _make_representation(
    *,
    sample_id: str,
    patient_id: str,
    category: str,
    features: list[list[float]],
    patch_indices: list[int],
    coords: list[list[int]] | None = None,
) -> RetrievalRepresentation:
    """
    Build one minimal SISH retrieval representation for tests.

    Inputs:
        sample_id:
            Retrieval item identifier.
        patient_id:
            Patient identifier.
        category:
            Slide label.
        features:
            Patch feature matrix with shape ``(N, D)``.
        patch_indices:
            Precomputed SISH indices with shape ``(N,)``.
        coords:
            Optional patch coordinates with shape ``(N, 2)``.

    Output:
        RetrievalRepresentation:
            Patch-vector retrieval representation with SISH auxiliary arrays in
            ``additional_data``.
    """
    feature_array = np.asarray(features, dtype=np.float32)
    if coords is None:
        coords = np.zeros((feature_array.shape[0], 2), dtype=np.int32).tolist()

    return RetrievalRepresentation(
        sample_id=sample_id,
        representation_type="patch_vector",
        data=feature_array,
        metadata=RetrievalItemMetadata(
            category=category,
            patient_id=patient_id,
        ),
        additional_data={
            "sish_patch_indices": np.asarray(patch_indices, dtype=np.int64),
            "selected_coords": np.asarray(coords, dtype=np.int32),
        },
    )


def test_import_search_strategy_modules_registers_sish() -> None:
    import_search_strategy_modules()

    assert get_search_strategy("sish") is SISHSearch


def test_sish_search_ranks_hits_and_excludes_same_patient(
    tmp_path: Path,
) -> None:
    reference_representations = [
        _make_representation(
            sample_id="same-patient",
            patient_id="patient-query",
            category="tumor",
            features=[[1.0, 1.0, 1.0, 1.0]],
            patch_indices=[19],
            coords=[[0, 0]],
        ),
        _make_representation(
            sample_id="slide-a",
            patient_id="patient-a",
            category="tumor",
            features=[[1.0, 1.0, 1.0, 1.0]],
            patch_indices=[19],
            coords=[[10, 10]],
        ),
        _make_representation(
            sample_id="slide-b",
            patient_id="patient-b",
            category="tumor",
            features=[[1.0, 1.0, 1.0, -1.0]],
            patch_indices=[21],
            coords=[[20, 20]],
        ),
    ]
    query_representation = _make_representation(
        sample_id="query-slide",
        patient_id="patient-query",
        category="tumor",
        features=[[1.0, 1.0, 1.0, 1.0]],
        patch_indices=[20],
        coords=[[5, 5]],
    )

    strategy = SISHSearch(
        params={
            "k": 2,
            "seed_fanout_t": 1,
            "pre_step": 1,
            "succ_step": 1,
            "hamming_thr": 8,
        },
        config=SimpleNamespace(
            experiment=SimpleNamespace(
                num_workers=0,
                sish=SimpleNamespace(
                    shard_dir=str(tmp_path / "shards"),
                    index_veb_path=str(tmp_path / "veb.pkl"),
                    meta_database_path=str(tmp_path / "meta.pkl"),
                ),
            )
        ),
    )
    strategy.build_database(reference_representations)

    result = strategy.search(query_representation=query_representation)

    assert result.query_id == "query-slide"
    assert result.metadata["predicted_category"] == "tumor"
    assert result.metadata["top_k_labels"] == ["tumor", "tumor"]
    assert [hit.item_id for hit in result.hits] == ["slide-a", "slide-b"]
    assert [hit.rank for hit in result.hits] == [1, 2]
    assert [hit.score for hit in result.hits] == [0.0, 1.0]


def test_sish_build_database_rejects_missing_patch_indices_and_mosaic_path() -> None:
    representation = RetrievalRepresentation(
        sample_id="slide-a",
        representation_type="patch_vector",
        data=np.asarray([[1.0, -1.0, 1.0]], dtype=np.float32),
        metadata=RetrievalItemMetadata(category="tumor", patient_id="patient-a"),
        additional_data={},
    )
    strategy = SISHSearch(params={"seed_fanout_t": 1})

    with pytest.raises(ValueError, match="sish_patch_indices|mosaic_pkl_path"):
        strategy.build_database([representation])
