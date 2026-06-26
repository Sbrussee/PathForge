from __future__ import annotations

import numpy as np
import pytest

from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.search_strategies.strategies.pbss import (
    PrototypeSimilaritySearch,
)


def _make_pbms_representation(
    *,
    sample_id: str,
    proto_mean: list[list[float]],
    proto_cov: list[list[float]],
    labels: list[str] | None = None,
    exclusion_key: str | None = None,
) -> RetrievalRepresentation:
    additional_data = {
        "panther_proto_mean": np.asarray(proto_mean, dtype=np.float32),
        "panther_proto_cov": np.asarray(proto_cov, dtype=np.float32),
    }
    if labels is not None:
        additional_data["prototype_labels"] = np.asarray(labels, dtype=str)

    return RetrievalRepresentation(
        sample_id=sample_id,
        data=np.empty((0, 2), dtype=np.float32),
        additional_data=additional_data,
        exclusion_key=exclusion_key,
    )


def test_pbss_ranks_by_included_prototype_distance() -> None:
    query = _make_pbms_representation(
        sample_id="query",
        proto_mean=[[0.0, 0.0], [10.0, 10.0]],
        proto_cov=[[1.0, 1.0], [1.0, 1.0]],
        labels=["include", "exclude"],
    )
    close_reference = _make_pbms_representation(
        sample_id="close",
        proto_mean=[[1.0, 0.0], [99.0, 99.0]],
        proto_cov=[[1.0, 1.0], [1.0, 1.0]],
        labels=["include", "exclude"],
    )
    far_reference = _make_pbms_representation(
        sample_id="far",
        proto_mean=[[5.0, 0.0], [10.0, 10.0]],
        proto_cov=[[1.0, 1.0], [1.0, 1.0]],
        labels=["include", "exclude"],
    )

    strategy = PrototypeSimilaritySearch(params={"k": 2})
    strategy.build_database([far_reference, close_reference])
    result = strategy.search(query)

    assert [hit.item_id for hit in result.hits] == ["close", "far"]
    assert result.hits[0].score < result.hits[1].score


def test_pbss_filters_same_exclusion_key() -> None:
    query = _make_pbms_representation(
        sample_id="query",
        proto_mean=[[0.0, 0.0]],
        proto_cov=[[1.0, 1.0]],
        exclusion_key="patient-1",
    )
    reference = _make_pbms_representation(
        sample_id="reference",
        proto_mean=[[0.0, 0.0]],
        proto_cov=[[1.0, 1.0]],
        exclusion_key="patient-1",
    )

    strategy = PrototypeSimilaritySearch(params={"k": 1})
    strategy.build_database([reference])
    result = strategy.search(query)

    assert result.hits == []


def test_pbss_requires_pbms_panther_statistics() -> None:
    representation = RetrievalRepresentation(
        sample_id="not-pbms",
        data=np.empty((0, 2), dtype=np.float32),
    )

    strategy = PrototypeSimilaritySearch(params={"k": 1})

    with pytest.raises(ValueError, match="Use retrieval_representation: pbms-features"):
        strategy.build_database([representation])
