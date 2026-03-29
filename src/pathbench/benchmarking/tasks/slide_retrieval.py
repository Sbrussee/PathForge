from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pathbench.benchmarking.registry import register_task
from pathbench.benchmarking.tasks.base import TaskBase
from pathbench.core.datasets.bag_dataset import BagDataset, BagSample
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.slide_retrieval.search_strategies.registry import (
    build_search_strategy,
)
from pathbench.slide_retrieval.search_strategies.types import SearchResult
from pathbench.slide_retrieval.representation_strategies.registry import (
    build_representation_strategy,
)
from pathbench.slide_retrieval.representation_strategies.types import RetrievalRepresentation

from pathbench.slide_retrieval.types import (
    RetrievalItemMetadata,
    SlideRetrievalRunSpec,
)
from pathbench.slide_retrieval.representation_strategies.storage import (
    build_retrieval_representation_artifact_path,
    build_retrieval_representation_entry_id,
    build_retrieval_representation_id,
)

from pathbench.slide_retrieval.io import (
    load_slide_retrieval_representation,
    save_slide_retrieval_representation,
    write_slide_retrieval_eval_outputs,
)
from pathbench.slide_retrieval.validation.registry import (
    get_validation_metric,
    import_validation_metric_modules,
    parse_validation_metric_name,
)
from pathbench.slide_retrieval.validation.types import (
    NormalizedSearchHit,
    NormalizedSearchResult,
)

logger = logging.getLogger(__name__)

_VALID_RETRIEVAL_USES = {"reference", "query", "query_reference"}


@register_task("slide_retrieval")
class SlideRetrievalTask(TaskBase):
    """
    Run slide retrieval from bag-level features.

    Flow:
    - ensure retrieval representations exist for all relevant samples
    - split representations into reference/query sets
    - build the search database once
    - run retrieval for each query
    - evaluate and persist outputs
    """

    grid_keys = [
        "feature_extraction",
        "tile_px",
        "tile_mpp",
        "retrieval_representation",
        "search_strategy",
    ]

    def execute(
        self,
        combo_cfg: ComboConfig,
        datasets_by_use: dict[str, list[BagDataset]],
    ) -> dict[str, Any]:
        self.bag_id, self.aggregation_level = self._infer_dataset_context(datasets_by_use=datasets_by_use)
        exclude_same_patient = bool(combo_cfg.get("exclude_same_patient", True))

        self.retrieval_representation_strategy = build_representation_strategy(
            str(combo_cfg.get("retrieval_representation")),
            params=combo_cfg.get_hyperparams("retrieval_representation"),
            bag_id=self.bag_id,
            config=getattr(self.experiment, "cfg", None),
        )
        slide_representation_params = dict(
            self.retrieval_representation_strategy.hyperparam_values()
        )

        self.search_strategy = build_search_strategy(
            str(combo_cfg.get("search_strategy")),
            params=combo_cfg.get_hyperparams("search_strategy"),
            config=getattr(self.experiment, "cfg", None),
        )
        search_params = dict(self.search_strategy.hyperparam_values())

        is_valid, reason = self._validate_combo_compatibility(datasets_by_use=datasets_by_use)
        if not is_valid:
            logger.warning("[SlideRetrieval] Skipping combo: %s", reason)
            return {
                "status": "skipped_incompatible_combo",
                "reason": reason,
            }

        self.representation_id = build_retrieval_representation_id(
            feature_extraction=str(combo_cfg.get("feature_extraction")),
            retrieval_representation=str(combo_cfg.get("retrieval_representation")),
            params=self.retrieval_representation_strategy.hyperparam_values(),
        )
        self.run_spec = SlideRetrievalRunSpec(
            project_root=Path(self.experiment.project_root),
            bag_id=self.bag_id,
            aggregation_level=self.aggregation_level,
            feature_extraction=str(combo_cfg.get("feature_extraction")),
            slide_representation=str(combo_cfg.get("retrieval_representation")),
            slide_representation_params=slide_representation_params,
            search_method=str(combo_cfg.get("search_strategy")),
            search_params=search_params,
            representation_id=self.representation_id,
            exclude_same_patient=exclude_same_patient,
        )

        representations_by_use = self._ensure_representations(
            datasets_by_use=datasets_by_use,
            combo_cfg=combo_cfg,
        )

        reference_representations, query_representations = (
            self._split_representations_by_use(
                representations_by_use=representations_by_use
            )
        )

        if not reference_representations:
            raise ValueError("No reference representations found for slide retrieval.")
        if not query_representations:
            raise ValueError("No query representations found for slide retrieval.")

        self.search_strategy.build_database(reference_representations)

        results: list[SearchResult] = []
        for query_representation in query_representations:
            result = self.search_strategy.search(
                query_representation=query_representation,
                filter_same_patient=self.run_spec.exclude_same_patient,
            )
            results.append(result)

        metrics = self._evaluate_results(
            results=results,
            query_representations=query_representations,
        )

        output_dir = self._write_outputs(
            results=results,
            metrics=metrics,
        )

        return {
            "output_dir": str(output_dir),
            "num_queries": len(query_representations),
            "num_reference_items": len(reference_representations),
            "metrics": metrics,
        }

    allowed_dataset_uses = frozenset(_VALID_RETRIEVAL_USES)

    # ------------------------------------------------------------------
    # Representation creation / loading
    # ------------------------------------------------------------------

    def _infer_dataset_context(
        self,
        *,
        datasets_by_use: dict[str, list[BagDataset]],
    ) -> tuple[str, str]:
        all_bag_datasets = [
            bag_dataset
            for bag_datasets in datasets_by_use.values()
            for bag_dataset in bag_datasets
        ]
        if not all_bag_datasets:
            raise ValueError("No bag datasets were provided for slide retrieval.")

        bag_ids = {str(bag_dataset.tiling_id) for bag_dataset in all_bag_datasets}
        if len(bag_ids) != 1:
            raise ValueError(
                "Slide retrieval requires all datasets in one run to share "
                f"the same bag_id. Got {sorted(bag_ids)}."
            )

        aggregation_levels = {
            str(bag_dataset.aggregation_level)
            for bag_dataset in all_bag_datasets
        }
        if len(aggregation_levels) != 1:
            raise ValueError(
                "Slide retrieval requires all datasets in one run to share "
                "the same aggregation_level. "
                f"Got {sorted(aggregation_levels)}."
            )

        return next(iter(bag_ids)), next(iter(aggregation_levels))

    def _ensure_representations(
        self,
        datasets_by_use: dict[str, list[BagDataset]],
        combo_cfg: ComboConfig,
    ) -> dict[str, list[RetrievalRepresentation]]:
        representations_by_use: dict[str, list[RetrievalRepresentation]] = {}

        for use, bag_datasets in datasets_by_use.items():
            if use not in _VALID_RETRIEVAL_USES:
                raise ValueError(
                    f"Unsupported retrieval dataset use '{use}'. "
                    f"Expected one of: {sorted(_VALID_RETRIEVAL_USES)}"
                )

            representations_by_use.setdefault(use, [])

            for bag_dataset in bag_datasets:
                for index in range(bag_dataset.num_bags):
                    bag, sample = bag_dataset.get_bag_sample(index)

                    representation = self._load_or_compute_representation(
                        bag_dataset=bag_dataset,
                        bag=bag,
                        sample=sample,
                        combo_cfg=combo_cfg,
                    )

                    representations_by_use[use].append(representation)

        return representations_by_use

    def _load_or_compute_representation(
        self,
        bag_dataset: BagDataset,
        bag: Any,
        sample: BagSample,
        combo_cfg: ComboConfig,
    ) -> RetrievalRepresentation:
        artifact_path = build_retrieval_representation_artifact_path(
            artifacts_dir=bag_dataset.artifacts_dir,
            aggregation_level=bag_dataset.aggregation_level,
            sample_id=sample.sample_id,
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        entry_id = build_retrieval_representation_entry_id(sorted(sample.slide_ids))

        with FileHandleH5(artifact_path, mode="a") as slide_artifact:
            representation = load_slide_retrieval_representation(
                slide_artifact=slide_artifact,
                tile_id=bag_dataset.tiling_id,
                representation_id=self.representation_id,
                entry_id=entry_id,
            )
            if representation is not None:
                return representation

            representation = self.retrieval_representation_strategy.run(
                bag=bag,
                sample=sample,
                bag_dataset=bag_dataset,
                combo_cfg=combo_cfg,
            )
            
            representation.metadata = self._build_representation_metadata(
                sample=sample,
                strategy_metadata=representation.metadata,
            )

            save_slide_retrieval_representation(
                slide_artifact=slide_artifact,
                tile_id=bag_dataset.tiling_id,
                representation_id=self.representation_id,
                entry_id=entry_id,
                representation=representation,
                params=self.retrieval_representation_strategy.hyperparam_values(),
                slide_ids=sample.slide_ids,
            )

            return representation

    def _build_representation_metadata(
        self,
        sample: BagSample,
        strategy_metadata: RetrievalItemMetadata | dict[str, Any] | None = None,
    ) -> RetrievalItemMetadata:
        normalized_strategy_metadata = RetrievalItemMetadata.from_any(strategy_metadata)
        return RetrievalItemMetadata.from_dict(
            {
                **dict(sample.metadata),
                **dict(normalized_strategy_metadata),
                "category": sample.category,
                "patient_id": sample.patient_id,
                "case_id": sample.case_id,
                "member_ids": list(sample.slide_ids),
            }
        )

    # ------------------------------------------------------------------
    # Use splitting
    # ------------------------------------------------------------------

    def _split_representations_by_use(
        self,
        representations_by_use: dict[str, list[RetrievalRepresentation]],
    ) -> tuple[list[RetrievalRepresentation], list[RetrievalRepresentation]]:
        reference_representations: list[RetrievalRepresentation] = []
        query_representations: list[RetrievalRepresentation] = []

        reference_representations.extend(representations_by_use.get("reference", []))
        reference_representations.extend(representations_by_use.get("query_reference", []))

        query_representations.extend(representations_by_use.get("query", []))
        query_representations.extend(representations_by_use.get("query_reference", []))

        return reference_representations, query_representations

    # ------------------------------------------------------------------
    # Evaluation / persistence
    # ------------------------------------------------------------------

    def _evaluate_results(
        self,
        results: list[SearchResult],
        query_representations: list[RetrievalRepresentation],
    ) -> dict[str, Any]:
        evaluation_metric_names = list(self.cfg.experiment.evaluation or [])
        if not evaluation_metric_names:
            return {}

        import_validation_metric_modules()
        normalized_results = self._normalize_results_for_validation(
            results=results,
            query_representations=query_representations,
        )

        metrics: dict[str, Any] = {}
        for metric_name in evaluation_metric_names:
            request = parse_validation_metric_name(metric_name)
            metric_fn = get_validation_metric(request.registry_key)
            metrics.update(metric_fn(normalized_results, k=request.k))

        return metrics

    def _normalize_results_for_validation(
        self,
        *,
        results: list[SearchResult],
        query_representations: list[RetrievalRepresentation],
    ) -> list[NormalizedSearchResult]:
        """
        Normalize retrieval outputs into scalar evaluation records.

        Inputs:
        - `results`: `list[SearchResult]` produced by the active search
          strategy. Each result contains ranked hits for one query.
        - `query_representations`: `list[RetrievalRepresentation]` aligned with
          `results` by query identifier.

        Returns:
        - `list[NormalizedSearchResult]` used by the validation metrics.

        Example:
            ```python
            normalized = self._normalize_results_for_validation(
                results=results,
                query_representations=query_representations,
            )
            ```
        """

        query_metadata_by_id = {
            representation.sample_id: representation.metadata
            for representation in query_representations
        }
        normalized_results: list[NormalizedSearchResult] = []

        for result in results:
            query_metadata = query_metadata_by_id.get(result.query_id, result.metadata)
            hits = [
                NormalizedSearchHit(
                    item_id=hit.item_id,
                    label=hit.metadata.category,
                    patient_id=hit.metadata.patient_id,
                    score=hit.score,
                    rank=hit.rank,
                )
                for hit in result.hits
            ]
            normalized_results.append(
                NormalizedSearchResult(
                    query_id=result.query_id,
                    query_label=query_metadata.category,
                    query_patient_id=query_metadata.patient_id,
                    hits=hits,
                    available_k=len(hits),
                )
            )

        return normalized_results

    def _write_outputs(
        self,
        results: list[SearchResult],
        metrics: dict[str, float],
    ) -> Path:
        _ = metrics

        if not hasattr(self, "run_spec") or self.run_spec is None:
            raise ValueError("Slide retrieval context was not initialized.")

        return write_slide_retrieval_eval_outputs(
            run_spec=self.run_spec,
            results=results,
            reference_items=self.search_strategy.search_database,
            metrics=metrics,
        )
    
    def _validate_combo_compatibility(
        self,
        *,
        datasets_by_use: dict[str, list[BagDataset]],
    ) -> tuple[bool, str]:
        feature_levels = {
            bag_dataset.get_feature_level()
            for bag_datasets in datasets_by_use.values()
            for bag_dataset in bag_datasets
        }

        if "invalid" in feature_levels:
            return False, "Invalid feature structure detected in one or more bag datasets."

        if "unknown" in feature_levels:
            return False, "Could not determine feature level for one or more bag datasets."

        if len(feature_levels) != 1:
            return False, f"Inconsistent feature levels across datasets: {sorted(feature_levels)}"

        feature_level = next(iter(feature_levels))

        if feature_level not in self.retrieval_representation_strategy.supported_feature_levels:
            return (
                False,
                f"Representation strategy '{self.retrieval_representation_strategy.name}' "
                f"does not support feature level '{feature_level}'.",
            )

        representation_kind = self.retrieval_representation_strategy.output_representation_kind
        if representation_kind not in self.search_strategy.supported_representation_kinds:
            return (
                False,
                f"Search strategy '{self.search_strategy.name}' does not support "
                f"representation kind '{representation_kind}'.",
            )

        return True, ""
