from __future__ import annotations

import pytest

from pathforge.slide_retrieval.hyperparams import HyperParam
from pathforge.slide_retrieval.representation_strategies.base import (
    BaseRetrievalRepresentationStrategy,
)
from pathforge.slide_retrieval.search_strategies.base import BaseSearchStrategy


class _RepresentationBase(BaseRetrievalRepresentationStrategy):
    k = HyperParam(int, default=3, min=1, help="Base retrieval depth.")


class _RepresentationChild(_RepresentationBase):
    temperature = HyperParam(
        float,
        default=0.5,
        min=0.0,
        max=1.0,
        help="Similarity temperature.",
    )


class _SearchBase(BaseSearchStrategy):
    k = HyperParam(int, default=5, min=1, help="Top-k retrieval depth.")


class _SearchChild(_SearchBase):
    threshold = HyperParam(
        float,
        default=0.7,
        min=0.0,
        max=1.0,
        help="Minimum similarity threshold.",
    )

    def rank(self, query_item, database_items, **kwargs):  # pragma: no cover
        _ = query_item, database_items, kwargs
        return []


def test_representation_strategy_collects_inherited_hyperparams() -> None:
    strategy = _RepresentationChild(params={"k": "4"})

    assert strategy.k == 4
    assert strategy.temperature == 0.5
    assert _RepresentationChild.hyperparam_spec() == {
        "k": {
            "type": int,
            "default": 3,
            "min": 1,
            "help": "Base retrieval depth.",
        },
        "temperature": {
            "type": float,
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "help": "Similarity temperature.",
        },
    }
    assert strategy.hyperparam_values() == {"k": 4, "temperature": 0.5}


def test_search_strategy_binds_declared_hyperparams_automatically() -> None:
    strategy = _SearchChild(params={"k": "2", "threshold": 5.0})

    assert strategy.k == 2
    assert strategy.threshold == 1.0
    assert strategy.hyperparam_values() == {"k": 2, "threshold": 1.0}


def test_declared_hyperparams_reject_invalid_choice_values() -> None:
    class _ChoiceStrategy(BaseSearchStrategy):
        mode = HyperParam(
            str,
            default="cosine",
            choices=("cosine", "l2"),
            help="Distance metric.",
        )

        def rank(self, query_item, database_items, **kwargs):  # pragma: no cover
            _ = query_item, database_items, kwargs
            return []

    with pytest.raises(ValueError, match="must be one of"):
        _ChoiceStrategy(params={"mode": "invalid"})
