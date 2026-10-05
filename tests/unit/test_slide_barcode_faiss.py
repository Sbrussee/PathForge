from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.types import RetrievalItemMetadata


def _slide_representation(
    sample_id: str,
    values: list[float],
    *,
    exclusion_key: str | None = None,
) -> RetrievalRepresentation:
    return RetrievalRepresentation(
        sample_id=sample_id,
        data=np.asarray([values], dtype=np.float32),
        representation_type="slide_vector",
        exclusion_key=exclusion_key,
        metadata=RetrievalItemMetadata(category="test"),
    )


def test_shared_minmax_encoder_matches_sish_and_salv_known_bytes() -> None:
    from pathforge.slide_retrieval.representations.minmax import encode_minmax
    from pathforge.slide_retrieval.search_strategies.strategies.sish.sish_bits import (
        pack_adjacent_feature_bits,
    )

    features = np.asarray([[4.0, 2.0, 2.0, 3.0, 1.0]], dtype=np.float32)

    encoded = encode_minmax(features)

    assert encoded.feature_dimension == 5
    assert encoded.faiss_bit_dimension == 8
    assert encoded.packed.tolist() == [[0b00110000]]
    np.testing.assert_array_equal(encoded.packed, pack_adjacent_feature_bits(features))


@pytest.mark.parametrize(
    "features",
    [
        np.asarray([], dtype=np.float32),
        np.asarray([[np.nan]], dtype=np.float32),
        np.asarray([[np.inf]], dtype=np.float32),
    ],
)
def test_shared_minmax_encoder_rejects_invalid_features(features: np.ndarray) -> None:
    from pathforge.slide_retrieval.representations.minmax import encode_minmax

    with pytest.raises(ValueError):
        encode_minmax(features)


def test_slide_features_preserves_the_single_raw_feature_vector() -> None:
    from pathforge.slide_retrieval.representation_strategies.strategies.slide_features import (
        SlideFeatures,
    )

    feature_matrix = np.asarray([[3.0, 2.0, 1.0]], dtype=np.float32)
    representation = SlideFeatures().run(
        bag=feature_matrix,
        sample=SimpleNamespace(sample_id="slide-1"),
    )

    assert representation.sample_id == "slide-1"
    assert representation.representation_type == "slide_vector"
    np.testing.assert_array_equal(representation.data, feature_matrix)


def test_slide_features_round_trip_uses_the_standard_h5_representation_layout(
    tmp_path: Path,
) -> None:
    from pathforge.core.io.slide_artifacts.base import FileHandleH5
    from pathforge.slide_retrieval.io import (
        load_slide_retrieval_representation,
        save_slide_retrieval_representation,
    )
    from pathforge.slide_retrieval.representation_strategies.strategies.slide_features import (
        SlideFeatures,
    )

    feature_matrix = np.asarray([[3.0, 2.0, 1.0]], dtype=np.float32)
    representation = SlideFeatures().run(
        bag=feature_matrix,
        sample=SimpleNamespace(sample_id="slide-1"),
    )
    with FileHandleH5(tmp_path / "slide-retrieval.h5", mode="a") as artifact:
        save_slide_retrieval_representation(
            retrieval_artifact=artifact,
            tile_id="256px_0.5mpp",
            representation_id="uni__slide_features__test",
            entry_id=None,
            representation=representation,
        )
    with FileHandleH5(tmp_path / "slide-retrieval.h5", mode="r") as artifact:
        loaded = load_slide_retrieval_representation(
            retrieval_artifact=artifact,
            tile_id="256px_0.5mpp",
            representation_id="uni__slide_features__test",
            entry_id=None,
        )

    assert loaded is not None
    assert loaded.feature_level == "slide"
    assert loaded.representation_type == "slide_vector"
    np.testing.assert_array_equal(loaded.data, feature_matrix)


def test_slide_only_pair_declares_its_compatibility_contract() -> None:
    from pathforge.slide_retrieval.representation_strategies.registry import (
        get_representation_strategy_output_kind,
        get_representation_strategy_supported_feature_levels,
    )
    from pathforge.slide_retrieval.search_strategies.registry import (
        get_search_strategy_supported_representation_kinds,
    )

    assert get_representation_strategy_supported_feature_levels("slide_features") == {
        "slide"
    }
    assert get_representation_strategy_output_kind("slide_features") == "slide_vector"
    assert get_search_strategy_supported_representation_kinds(
        "slide-barcode-faiss"
    ) == {"slide_vector"}


@pytest.mark.parametrize(
    "feature_matrix",
    [
        np.asarray([[1.0], [2.0]], dtype=np.float32),
        np.asarray([[np.nan]], dtype=np.float32),
        np.asarray([[True]], dtype=bool),
    ],
)
def test_slide_features_rejects_non_single_or_non_finite_vectors(
    feature_matrix: np.ndarray,
) -> None:
    from pathforge.slide_retrieval.representation_strategies.strategies.slide_features import (
        SlideFeatures,
    )

    with pytest.raises(ValueError):
        SlideFeatures().run(bag=feature_matrix)


def test_slide_barcode_faiss_returns_exact_hamming_neighbours() -> None:
    from pathforge.slide_retrieval.search_strategies.strategies.slide_barcode_faiss import (
        SlideBarcodeFaissSearch,
    )

    strategy = SlideBarcodeFaissSearch(params={"k": 3})
    strategy.build_database(
        [
            _slide_representation("exact", [3.0, 2.0, 1.0]),
            _slide_representation("furthest", [1.0, 2.0, 3.0]),
            _slide_representation("middle", [2.0, 3.0, 1.0]),
        ]
    )

    result = strategy.search(_slide_representation("query", [3.0, 2.0, 1.0]))

    assert [hit.sample_id for hit in result.hits] == ["exact", "middle", "furthest"]
    assert [hit.score for hit in result.hits] == [0.0, 1.0, 2.0]


def test_slide_barcode_faiss_overfetches_after_exclusion() -> None:
    from pathforge.slide_retrieval.search_strategies.strategies.slide_barcode_faiss import (
        SlideBarcodeFaissSearch,
    )

    strategy = SlideBarcodeFaissSearch(params={"k": 2})
    strategy.build_database(
        [
            _slide_representation("excluded", [3.0, 2.0, 1.0], exclusion_key="p1"),
            _slide_representation("first", [2.0, 3.0, 1.0], exclusion_key="p2"),
            _slide_representation("second", [1.0, 2.0, 3.0], exclusion_key="p3"),
        ]
    )

    result = strategy.search(
        _slide_representation("query", [3.0, 2.0, 1.0], exclusion_key="p1")
    )

    assert [hit.sample_id for hit in result.hits] == ["first", "second"]


def test_slide_barcode_faiss_rejects_non_slide_vector_representations() -> None:
    from pathforge.slide_retrieval.search_strategies.strategies.slide_barcode_faiss import (
        SlideBarcodeFaissSearch,
    )

    strategy = SlideBarcodeFaissSearch()
    representation = RetrievalRepresentation(
        sample_id="patch",
        data=np.asarray([[1.0, 2.0]], dtype=np.float32),
        representation_type="patch_vector",
    )

    with pytest.raises(ValueError, match="slide_vector"):
        strategy.build_database([representation])
