from __future__ import annotations

import pytest

from pathbench.core.evaluation.registry import (
    evaluation_metric,
    import_evaluation_metric_modules,
    resolve_metric_request,
)


def test_resolve_metric_request_parses_registered_metric_family() -> None:
    import_evaluation_metric_modules()

    spec, request = resolve_metric_request(
        task_name="slide_retrieval",
        raw_name="hit_at_10",
    )

    assert spec.canonical_name == "hit_at_k"
    assert request.raw_name == "hit_at_10"
    assert request.canonical_name == "hit_at_k"
    assert request.params == {"k": 10}


def test_register_evaluation_metric_supports_exact_name_resolution() -> None:
    metric_name = "unit_eval_exact_metric"

    @evaluation_metric(metric_name, tasks=("slide_retrieval",))
    def _compute_metric(*args, **kwargs):  # pragma: no cover
        return {"ok": True}

    spec, request = resolve_metric_request(
        task_name="slide_retrieval",
        raw_name=metric_name,
    )

    assert spec.canonical_name == metric_name
    assert request.canonical_name == metric_name
    assert request.params == {}


def test_register_parameterized_evaluation_metric_rejects_duplicate_names() -> None:
    metric_name = "unit_eval_duplicate_family"

    @evaluation_metric(
        metric_name,
        tasks=("slide_retrieval",),
        pattern=r"^unit_eval_duplicate_family_(?P<k>[1-9]\d*)$",
        param_builder=lambda match: {"k": int(match.group("k"))},
    )
    def _compute_metric(*args, **kwargs):  # pragma: no cover
        return {"ok": True}

    with pytest.raises(ValueError, match="already registered"):

        @evaluation_metric(
            metric_name,
            tasks=("slide_retrieval",),
            pattern=r"^unit_eval_duplicate_family_(?P<k>[1-9]\d*)$",
            param_builder=lambda match: {"k": int(match.group("k"))},
        )
        def _compute_metric_duplicate(*args, **kwargs):  # pragma: no cover
            return {"ok": False}


def test_resolve_metric_request_rejects_unknown_metric() -> None:
    import_evaluation_metric_modules()

    with pytest.raises(ValueError, match="not registered"):
        resolve_metric_request(
            task_name="slide_retrieval",
            raw_name="does_not_exist",
        )
