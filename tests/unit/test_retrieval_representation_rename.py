from __future__ import annotations

import hashlib
import json

import pytest

from pathbench.slide_retrieval.representation_strategies.base import (
    BaseRetrievalRepresentationStrategy,
)
from pathbench.slide_retrieval.representation_strategies.registry import (
    _RETRIEVAL_REPRESENTATION_REGISTRY,
    build_representation_strategy,
    get_representation_strategy,
    is_representation_strategy_available,
    list_representation_strategies,
    register_representation_strategy,
)
from pathbench.slide_retrieval.representation_strategies.storage import (
    build_retrieval_representation_id,
)
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.types import RetrievalItemMetadata


class _TestRetrievalRepresentationStrategy(BaseRetrievalRepresentationStrategy):
    name = "unit_test_strategy"


def test_retrieval_representation_id_uses_new_name() -> None:
    params = {"alpha": 0.25, "labels": ["a", "b"]}
    params_payload = json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    expected_hash = hashlib.sha1(params_payload.encode("utf-8")).hexdigest()[:16]

    representation_id = build_retrieval_representation_id(
        feature_extraction="uni",
        retrieval_representation="mean_pool",
        params=params,
    )

    assert representation_id == f"uni__mean_pool__{expected_hash}"


def test_retrieval_representation_id_rejects_empty_representation_name() -> None:
    with pytest.raises(
        ValueError,
        match="retrieval_representation must be a non-empty string",
    ):
        build_retrieval_representation_id(
            feature_extraction="uni",
            retrieval_representation="  ",
        )


def test_retrieval_representation_strategy_registry_round_trip() -> None:
    register_representation_strategy("unit_test_strategy")(
        _TestRetrievalRepresentationStrategy
    )

    try:
        assert (
            get_representation_strategy("unit_test_strategy")
            is _TestRetrievalRepresentationStrategy
        )
        assert is_representation_strategy_available("unit_test_strategy")
        assert "unit_test_strategy" in list_representation_strategies()
        assert isinstance(
            build_representation_strategy("unit_test_strategy"),
            _TestRetrievalRepresentationStrategy,
        )
    finally:
        _RETRIEVAL_REPRESENTATION_REGISTRY.pop("unit_test_strategy", None)


def test_retrieval_representation_dataclass_uses_new_type_name() -> None:
    representation = RetrievalRepresentation(
        sample_id="sample-1",
        representation_type="mean_pool",
        data=[1.0, 2.0],
        metadata={"patient_id": "patient-1"},
    )

    assert representation.sample_id == "sample-1"
    assert representation.representation_type == "mean_pool"
    assert representation.metadata["patient_id"] == "patient-1"
    assert isinstance(representation.metadata, RetrievalItemMetadata)
