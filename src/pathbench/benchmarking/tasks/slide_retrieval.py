from __future__ import annotations

from pathlib import Path
import logging
from typing import Any

from torch.utils.data import DataLoader, Subset

from pathbench.benchmarking.registry import register_task
from pathbench.benchmarking.tasks.base import TaskBase
from pathbench.core.datasets.bag_dataset import (
    BagDataset,
    BagSample,
    SlideRetrievalBagDataset,
    SlideRetrievalDatasetItem,
)
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.core.experiments.combo_ids import build_feature_name, build_tiling_id
from pathbench.core.io.slide_artifacts.base import FileHandleH5
from pathbench.slide_retrieval.io import (
    build_slide_retrieval_output_root,
    load_slide_retrieval_representation,
    save_slide_retrieval_representation,
    write_slide_retrieval_manifest,
    write_slide_retrieval_results_csv,
)
from pathbench.slide_retrieval.representation_strategies.registry import (
    build_representation_strategy,
)
from pathbench.slide_retrieval.representation_strategies.storage import (
    build_retrieval_representation_artifact_path,
    build_retrieval_representation_entry_id,
    build_retrieval_representation_id,
)
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.search_strategies.registry import (
    build_search_strategy,
)
from pathbench.slide_retrieval.search_strategies.types import SearchResult
from pathbench.slide_retrieval.types import ExclusionLevel, SlideRetrievalManifest

logger = logging.getLogger(__name__)

_VALID_RETRIEVAL_USES = {"reference", "query", "query_reference"}


def _single_item_collate(
    batch: list[SlideRetrievalDatasetItem],
) -> SlideRetrievalDatasetItem:
    """Return the only element from a batch-size-1 retrieval loader batch."""
    if len(batch) != 1:
        raise ValueError(
            "Slide retrieval currently expects DataLoader batches of size 1. "
            f"Got batch size {len(batch)}."
        )
    return batch[0]


@register_task("slide_retrieval")
class SlideRetrievalTask(TaskBase):
    """Run slide retrieval from bag-level features."""

    grid_keys = [
        "tile_px",
        "tile_mpp",
        "feature_extraction",
        "color_norm",
        "retrieval_representation",
        "search_strategy",
    ]

    allowed_dataset_uses = frozenset(_VALID_RETRIEVAL_USES)

    def execute(
        self,
        combo_cfg: ComboConfig,
        datasets_by_use: dict[str, list[BagDataset]],
    ) -> dict[str, Any]:
        # ------------------------------------------------------------------
        # Resolve run identity
        # ------------------------------------------------------------------
        tiling_id = build_tiling_id(combo_cfg)
        aggregation_level = str(self.cfg.experiment.aggregation_level)
        self._validate_dataset_context(
            datasets_by_use=datasets_by_use,
            tiling_id=tiling_id,
            aggregation_level=aggregation_level,
        )
        feature_name = build_feature_name(combo_cfg)
        exclusion_level = self._resolve_exclusion_level()
        
        # ------------------------------------------------------------------
        # Build and run the representation stage
        # ------------------------------------------------------------------
        representation_strategy = build_representation_strategy(
            str(combo_cfg.get("retrieval_representation")),
            params=combo_cfg.get_hyperparams("retrieval_representation"),
            bag_id=tiling_id,
            config=getattr(self.experiment, "cfg", None),
        )
        representation_id = build_retrieval_representation_id(
            feature_extraction=feature_name,
            retrieval_representation=str(combo_cfg.get("retrieval_representation")),
            params=representation_strategy.hyperparam_values(),
        )

        is_valid, reason = self._validate_representation_stage(
            datasets_by_use=datasets_by_use,
            representation_strategy=representation_strategy,
            aggregation_level=aggregation_level,
            exclusion_level=exclusion_level,
        )
        if not is_valid:
            logger.warning("[SlideRetrieval] Skipping combo: %s", reason)
            return {
                "status": "skipped_incompatible_combo",
                "reason": reason,
            }

        representations_by_use = self._load_or_compute_representations(
            datasets_by_use=datasets_by_use,
            combo_cfg=combo_cfg,
            representation_strategy=representation_strategy,
            representation_id=representation_id,
            aggregation_level=aggregation_level,
            exclusion_level=exclusion_level,
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

        # ------------------------------------------------------------------
        # Build and run the search stage
        # ------------------------------------------------------------------
        search_strategy = build_search_strategy(
            str(combo_cfg.get("search_strategy")),
            params=combo_cfg.get_hyperparams("search_strategy"),
            config=getattr(self.experiment, "cfg", None),
        )
        is_valid, reason = self._validate_search_stage(
            representation_strategy=representation_strategy,
            search_strategy=search_strategy,
        )
        if not is_valid:
            logger.warning("[SlideRetrieval] Skipping combo: %s", reason)
            return {
                "status": "skipped_incompatible_combo",
                "reason": reason,
            }

        search_strategy.build_database(reference_representations)
        results = self._run_queries(
            search_strategy=search_strategy,
            query_representations=query_representations,
        )

        # ------------------------------------------------------------------
        # Build and write outputs
        # ------------------------------------------------------------------
        manifest = self._build_manifest(
            tiling_id=tiling_id,
            aggregation_level=aggregation_level,
            feature_name=feature_name,
            combo_cfg=combo_cfg,
            representation_strategy=representation_strategy,
            search_strategy=search_strategy,
            representation_id=representation_id,
            exclusion_level=exclusion_level,
            results=results,
        )
        output_root = self._build_output_root(
            project_root=str(self.experiment.project_root),
            tiling_id=tiling_id,
            feature_name=feature_name,
            slide_representation=str(combo_cfg.get("retrieval_representation")),
            search_method=str(combo_cfg.get("search_strategy")),
        )
        run_dir = output_root / f"run_{self._build_run_id(manifest)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_slide_retrieval_manifest(run_dir / "manifest.json", manifest)
        write_slide_retrieval_results_csv(run_dir / "query_results.csv", results)

        return {
            "output_dir": str(run_dir),
            "num_queries": len(query_representations),
            "num_reference_items": len(reference_representations),
        }

    def _resolve_exclusion_level(self) -> ExclusionLevel:
        slide_retrieval_cfg = getattr(self.cfg, "slide_retrieval", None)
        return getattr(slide_retrieval_cfg, "exclusion_level", "patient")

    def _validate_dataset_context(
        self,
        *,
        datasets_by_use: dict[str, list[BagDataset]],
        tiling_id: str,
        aggregation_level: str,
    ) -> None:
        all_bag_datasets = [
            bag_dataset
            for bag_datasets in datasets_by_use.values()
            for bag_dataset in bag_datasets
        ]
        if not all_bag_datasets:
            raise ValueError("No bag datasets were provided for slide retrieval.")

        dataset_tiling_ids = {
            str(bag_dataset.tiling_id) for bag_dataset in all_bag_datasets
        }
        if dataset_tiling_ids != {str(tiling_id)}:
            raise ValueError(
                "Slide retrieval datasets do not match the expected tiling_id. "
                f"Expected {tiling_id!r}, got {sorted(dataset_tiling_ids)}."
            )

        aggregation_levels = {
            str(bag_dataset.aggregation_level)
            for bag_dataset in all_bag_datasets
        }
        if aggregation_levels != {str(aggregation_level)}:
            raise ValueError(
                "Slide retrieval datasets do not match the expected "
                "aggregation_level. "
                f"Expected {aggregation_level!r}, got {sorted(aggregation_levels)}."
            )

    def _load_or_compute_representations(
        self,
        *,
        datasets_by_use: dict[str, list[BagDataset]],
        combo_cfg: ComboConfig,
        representation_strategy: Any,
        representation_id: str,
        aggregation_level: str,
        exclusion_level: ExclusionLevel,
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
                if not isinstance(bag_dataset, SlideRetrievalBagDataset):
                    raise TypeError(
                        "slide_retrieval requires SlideRetrievalBagDataset instances. "
                        f"Got {type(bag_dataset).__name__}."
                    )

                cached_representations, missing_indices = (
                    self._load_cached_representations_for_dataset(
                        bag_dataset=bag_dataset,
                        representation_id=representation_id,
                        aggregation_level=aggregation_level,
                        exclusion_level=exclusion_level,
                    )
                )
                representations_by_use[use].extend(cached_representations)

                if not missing_indices:
                    continue

                bag_dataset.bind_sample_loader(representation_strategy.load_sample)
                try:
                    loader = self._build_retrieval_dataloader(
                        bag_dataset=bag_dataset,
                        indices=missing_indices,
                    )
                    for item in loader:
                        representation = self._compute_and_save_representation(
                            bag_dataset=bag_dataset,
                            sample=item.sample,
                            loaded_inputs=item.inputs,
                            combo_cfg=combo_cfg,
                            representation_strategy=representation_strategy,
                            representation_id=representation_id,
                            aggregation_level=aggregation_level,
                            exclusion_level=exclusion_level,
                        )
                        representations_by_use[use].append(representation)
                finally:
                    bag_dataset.clear_sample_loader()

        return representations_by_use

    def _build_retrieval_dataloader(
        self,
        bag_dataset: SlideRetrievalBagDataset,
        indices: list[int],
    ) -> DataLoader[SlideRetrievalDatasetItem]:
        """Build a deterministic DataLoader for retrieval sample materialization."""
        num_workers = int(getattr(self.cfg.experiment, "num_workers", 0) or 0)
        return DataLoader(
            Subset(bag_dataset, indices),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_single_item_collate,
        )

    def _load_cached_representations_for_dataset(
        self,
        *,
        bag_dataset: BagDataset,
        representation_id: str,
        aggregation_level: str,
        exclusion_level: ExclusionLevel,
    ) -> tuple[list[RetrievalRepresentation], list[int]]:
        cached_representations: list[RetrievalRepresentation] = []
        missing_indices: list[int] = []

        for index in range(bag_dataset.num_bags):
            sample = bag_dataset.get_sample(index)
            cached_representation = self._load_cached_representation(
                bag_dataset=bag_dataset,
                sample=sample,
                representation_id=representation_id,
                aggregation_level=aggregation_level,
                exclusion_level=exclusion_level,
            )
            if cached_representation is None:
                missing_indices.append(index)
                continue
            cached_representations.append(cached_representation)

        return cached_representations, missing_indices

    def _load_cached_representation(
        self,
        *,
        bag_dataset: BagDataset,
        sample: BagSample,
        representation_id: str,
        aggregation_level: str,
        exclusion_level: ExclusionLevel,
    ) -> RetrievalRepresentation | None:
        artifact_path = build_retrieval_representation_artifact_path(
            artifacts_dir=bag_dataset.artifacts_dir,
            aggregation_level=bag_dataset.aggregation_level,
            sample_id=sample.sample_id,
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        entry_id = build_retrieval_representation_entry_id(sorted(sample.slide_ids))

        with FileHandleH5(artifact_path, mode="a") as retrieval_artifact:
            representation = load_slide_retrieval_representation(
                retrieval_artifact=retrieval_artifact,
                tile_id=bag_dataset.tiling_id,
                representation_id=representation_id,
                entry_id=entry_id,
            )
            if representation is not None:
                representation.exclusion_key = self._build_exclusion_key(
                    sample=sample,
                    aggregation_level=aggregation_level,
                    exclusion_level=exclusion_level,
                )
                representation.additional_data = {
                    **representation.additional_data,
                    "dataset_name": bag_dataset.name,
                    "source_slide_ids": list(sample.slide_ids),
                }
                return representation

        return None

    def _compute_and_save_representation(
        self,
        *,
        bag_dataset: BagDataset,
        sample: BagSample,
        loaded_inputs: dict[str, Any],
        combo_cfg: ComboConfig,
        representation_strategy: Any,
        representation_id: str,
        aggregation_level: str,
        exclusion_level: ExclusionLevel,
    ) -> RetrievalRepresentation:
        artifact_path = build_retrieval_representation_artifact_path(
            artifacts_dir=bag_dataset.artifacts_dir,
            aggregation_level=bag_dataset.aggregation_level,
            sample_id=sample.sample_id,
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        entry_id = build_retrieval_representation_entry_id(sorted(sample.slide_ids))

        with FileHandleH5(artifact_path, mode="a") as retrieval_artifact:

            representation = representation_strategy.run(
                sample=sample,
                bag_dataset=bag_dataset,
                combo_cfg=combo_cfg,
                **loaded_inputs,
            )
            representation.exclusion_key = self._build_exclusion_key(
                sample=sample,
                aggregation_level=aggregation_level,
                exclusion_level=exclusion_level,
            )
            representation.additional_data = {
                **representation.additional_data,
                "dataset_name": bag_dataset.name,
                "source_slide_ids": list(sample.slide_ids),
            }

            save_slide_retrieval_representation(
                retrieval_artifact=retrieval_artifact,
                tile_id=bag_dataset.tiling_id,
                representation_id=representation_id,
                entry_id=entry_id,
                representation=representation,
                params=representation_strategy.hyperparam_values(),
            )
            return representation

    def _build_exclusion_key(
        self,
        *,
        sample: BagSample,
        aggregation_level: str,
        exclusion_level: ExclusionLevel,
    ) -> str | None:
        if exclusion_level == "none":
            return None
        if exclusion_level == "slide":
            if aggregation_level != "slide":
                raise ValueError(
                    "slide_retrieval.exclusion_level='slide' requires "
                    "experiment.aggregation_level='slide'."
                )
            return str(sample.sample_id)
        if exclusion_level == "case":
            return None if sample.case_id is None else str(sample.case_id)
        if exclusion_level == "patient":
            return None if sample.patient_id is None else str(sample.patient_id)
        raise ValueError(f"Unsupported slide retrieval exclusion level: {exclusion_level!r}")

    def _split_representations_by_use(
        self,
        *,
        representations_by_use: dict[str, list[RetrievalRepresentation]],
    ) -> tuple[list[RetrievalRepresentation], list[RetrievalRepresentation]]:
        reference_representations: list[RetrievalRepresentation] = []
        query_representations: list[RetrievalRepresentation] = []

        reference_representations.extend(representations_by_use.get("reference", []))
        reference_representations.extend(representations_by_use.get("query_reference", []))

        query_representations.extend(representations_by_use.get("query", []))
        query_representations.extend(representations_by_use.get("query_reference", []))

        return reference_representations, query_representations

    def _run_queries(
        self,
        *,
        search_strategy: Any,
        query_representations: list[RetrievalRepresentation],
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        for query_representation in query_representations:
            results.append(
                search_strategy.search(
                    query_representation=query_representation,
                )
            )
        return results

    def _build_output_root(
        self,
        *,
        project_root: str,
        tiling_id: str,
        feature_name: str,
        slide_representation: str,
        search_method: str,
    ) -> Path:
        return build_slide_retrieval_output_root(
            project_root=project_root,
            tiling_id=tiling_id,
            feature_name=feature_name,
            slide_representation=slide_representation,
            search_method=search_method,
        )

    def _build_manifest(
        self,
        *,
        tiling_id: str,
        aggregation_level: str,
        feature_name: str,
        combo_cfg: ComboConfig,
        representation_strategy: Any,
        search_strategy: Any,
        representation_id: str,
        exclusion_level: ExclusionLevel,
        results: list[SearchResult],
    ) -> SlideRetrievalManifest:
        return SlideRetrievalManifest(
            tiling_id=tiling_id,
            aggregation_level=aggregation_level,
            feature_extraction=feature_name,
            slide_representation=str(combo_cfg.get("retrieval_representation")),
            slide_representation_params=dict(
                representation_strategy.hyperparam_values()
            ),
            search_method=str(combo_cfg.get("search_strategy")),
            search_params=dict(search_strategy.hyperparam_values()),
            representation_id=representation_id,
            exclusion_level=exclusion_level,
            num_queries=len(results),
            num_reference_items=len(search_strategy.search_database),
            top_k_saved=max((len(result.hits) for result in results), default=0),
        )

    def _build_run_id(
        self,
        manifest: SlideRetrievalManifest,
    ) -> str:
        return manifest.short_hash()

    def _validate_representation_stage(
        self,
        *,
        datasets_by_use: dict[str, list[BagDataset]],
        representation_strategy: Any,
        aggregation_level: str,
        exclusion_level: ExclusionLevel,
    ) -> tuple[bool, str]:
        all_bag_datasets = [
            bag_dataset
            for bag_datasets in datasets_by_use.values()
            for bag_dataset in bag_datasets
        ]
        feature_levels = {
            bag_dataset.get_feature_level()
            for bag_dataset in all_bag_datasets
        }

        if "invalid" in feature_levels:
            details = " | ".join(
                bag_dataset.get_feature_level_reason()
                for bag_dataset in all_bag_datasets
                if bag_dataset.get_feature_level() == "invalid"
            )
            return False, (
                "Invalid feature structure detected in one or more bag datasets."
                + (f" Details: {details}" if details else "")
            )

        if "unknown" in feature_levels:
            details = " | ".join(
                bag_dataset.get_feature_level_reason()
                for bag_dataset in all_bag_datasets
                if bag_dataset.get_feature_level() == "unknown"
            )
            return False, (
                "Could not determine feature level for one or more bag datasets."
                + (f" Details: {details}" if details else "")
            )

        if len(feature_levels) != 1:
            return False, f"Inconsistent feature levels across datasets: {sorted(feature_levels)}"

        feature_level = next(iter(feature_levels))

        if feature_level not in representation_strategy.supported_feature_levels:
            return (
                False,
                f"Representation strategy '{representation_strategy.name}' "
                f"does not support feature level '{feature_level}'.",
            )

        if exclusion_level == "slide" and aggregation_level != "slide":
            return (
                False,
                "slide_retrieval.exclusion_level='slide' is only supported when "
                "experiment.aggregation_level='slide'.",
            )

        return True, ""

    def _validate_search_stage(
        self,
        *,
        representation_strategy: Any,
        search_strategy: Any,
    ) -> tuple[bool, str]:
        representation_kind = representation_strategy.output_representation_kind
        if representation_kind not in search_strategy.supported_representation_kinds:
            return (
                False,
                f"Search strategy '{search_strategy.name}' does not support "
                f"representation kind '{representation_kind}'.",
            )

        return True, ""
