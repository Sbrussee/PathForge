from __future__ import annotations

import hashlib
import json

import pytest

from pathforge.slide_retrieval.representation_strategies.base import (
    BaseRetrievalRepresentationStrategy,
)
from pathforge.slide_retrieval.representation_strategies.registry import (
    _RETRIEVAL_REPRESENTATION_REGISTRY,
    build_representation_strategy,
    get_representation_strategy,
    is_representation_strategy_available,
    list_representation_strategies,
    register_representation_strategy,
)
from pathforge.slide_retrieval.representation_strategies.storage import (
    build_retrieval_representation_entry_id,
    build_retrieval_representation_id,
)
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.types import RetrievalItemMetadata
from pathforge.slide_retrieval.types import RetrievalItemIdentity


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


def test_retrieval_representation_id_lowercases_non_hash_parts() -> None:
    params = {"alpha": 0.25}
    params_payload = json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    expected_hash = hashlib.sha1(params_payload.encode("utf-8")).hexdigest()[:16]

    representation_id = build_retrieval_representation_id(
        feature_extraction="UNI2",
        retrieval_representation="Yottixel_Features",
        params=params,
    )

    assert representation_id == f"uni2__yottixel_features__{expected_hash}"


def test_retrieval_representation_entry_id_is_none_for_slide_aggregation() -> None:
    entry_id = build_retrieval_representation_entry_id(
        ["slide-1"],
        aggregation_level="slide",
    )
    assert entry_id is None


def test_retrieval_representation_entry_id_uses_member_hash_for_patient_aggregation() -> None:
    entry_id = build_retrieval_representation_entry_id(
        ["slide-b", "slide-a"],
        aggregation_level="patient",
    )
    assert isinstance(entry_id, str)
    assert entry_id.startswith("members_")
    assert len(entry_id) == len("members_") + 16


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


def test_retrieval_item_identity_to_dict_omits_exclusion_key() -> None:
    identity = RetrievalItemIdentity(
        sample_id="sample-1",
        exclusion_key="patient-1",
    )

    assert identity.to_dict() == {"sample_id": "sample-1"}
