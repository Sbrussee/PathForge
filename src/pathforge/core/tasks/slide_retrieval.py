from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from pathforge.core.datasets.bag_dataset import (
    BagDataset,
    BagSample,
    SlideRetrievalBagDataset,
    SlideRetrievalDatasetItem,
)
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.core.experiments.combo_ids import build_feature_name, build_tiling_id
from pathforge.core.io.slide_artifacts.atomic import atomic_slide_artifact_write
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.tasks.base import TaskBase
from pathforge.core.tasks.registry import register_task
from pathforge.slide_retrieval.aggregation import aggregate_slide_representations
from pathforge.slide_retrieval.io import (
    build_slide_retrieval_inference_output_root,
    build_slide_retrieval_output_root,
    load_slide_retrieval_representation,
    save_slide_retrieval_representation,
    write_slide_retrieval_manifest,
    write_slide_retrieval_results_xlsx,
)
from pathforge.slide_retrieval.representation_strategies.registry import (
    build_representation_strategy,
    get_representation_strategy_output_kind,
    get_representation_strategy_supported_feature_levels,
)
from pathforge.slide_retrieval.representation_strategies.storage import (
    build_retrieval_representation_artifact_path,
    build_retrieval_representation_entry_id,
    build_retrieval_representation_id,
)
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.search_strategies.registry import (
    build_search_strategy,
    get_search_strategy_supported_representation_kinds,
)
from pathforge.slide_retrieval.search_strategies.types import SearchResult
from pathforge.slide_retrieval.types import ExclusionLevel, SlideRetrievalManifest

logger = logging.getLogger(__name__)

_VALID_RETRIEVAL_USES = {"reference", "query", "query_reference"}
_PROGRESS_MIN_INTERVAL_SECONDS = 30.0
_PROGRESS_NCOLS = 100


def _retrieval_batch_collate(
    batch: list[SlideRetrievalDatasetItem],
) -> list[SlideRetrievalDatasetItem]:
    """Return retrieval loader batches as-is for batched materialization."""
    return list(batch)


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
    inference_dataset_uses = frozenset({"reference", "query_reference"})
    inference_input_use = "query"

    def execute(
        self,
        combo_cfg: ComboConfig,
        datasets_by_use: dict[str, list[BagDataset]],
    ) -> dict[str, Any]:
        return self._run_retrieval(
            combo_cfg=combo_cfg,
            datasets_by_use=datasets_by_use,
            output_mode="benchmark",
            inference_run_root=None,
            include_query_reference_as_queries=True,
        )

    def inference(
        self,
        combo_cfg: ComboConfig,
        datasets_by_use: dict[str, list[BagDataset]],
        inference_run_root: Path,
    ) -> dict[str, Any]:
        return self._run_retrieval(
            combo_cfg=combo_cfg,
            datasets_by_use=datasets_by_use,
            output_mode="inference",
            inference_run_root=inference_run_root,
            include_query_reference_as_queries=False,
        )

    def _run_retrieval(
        self,
        combo_cfg: ComboConfig,
        datasets_by_use: dict[str, list[BagDataset]],
        output_mode: str,
        inference_run_root: Path | None,
        include_query_reference_as_queries: bool,
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
        representation_name = str(combo_cfg.get("retrieval_representation"))
        search_strategy_name = str(combo_cfg.get("search_strategy"))
        num_workers = int(getattr(self.cfg.experiment, "num_workers", 0) or 0)
        logger.info(
            "[SlideRetrieval] Starting execute | tiling_id=%s, aggregation=%s, "
            "representation=%s, search=%s",
            tiling_id,
            aggregation_level,
            representation_name,
            search_strategy_name,
        )

        # Fail fast when the selected strategies and dataset feature levels cannot match.
        combination_is_valid, reason = self._validate_combination_compatibility(
            datasets_by_use=datasets_by_use,
            representation_name=representation_name,
            search_strategy_name=search_strategy_name,
            aggregation_level=aggregation_level,
            exclusion_level=exclusion_level,
        )
        if not combination_is_valid:
            logger.warning("[SlideRetrieval] Skipping combo: %s", reason)
            return {
                "status": "skipped_incompatible_combo",
                "reason": reason,
            }

        # ------------------------------------------------------------------
        # Build and run the representation stage
        # ------------------------------------------------------------------
        logger.info(
            "[SlideRetrieval] Stage 1/3: preparing retrieval representation strategy"
        )
        representation_strategy = build_representation_strategy(
            representation_name,
            params=combo_cfg.get_hyperparams("retrieval_representation"),
            bag_id=tiling_id,
            config=getattr(self.experiment, "cfg", None),
        )
        prepare_for_combo = getattr(representation_strategy, "prepare_for_combo", None)
        if callable(prepare_for_combo):
            prepare_for_combo(
                combo_cfg=combo_cfg,
                feature_name=feature_name,
                tiling_id=tiling_id,
            )
        representation_cache_params = self._representation_cache_params(
            representation_strategy
        )
        representation_id = build_retrieval_representation_id(
            feature_extraction=feature_name,
            retrieval_representation=representation_name,
            params=representation_cache_params,
        )
        logger.info(
            "[SlideRetrieval] Representation strategy ready | representation_id=%s",
            representation_id,
        )

        logger.info(
            "[SlideRetrieval] Stage 2/3: materializing physical-slide representations"
        )
        representations_by_use = self._materialize_and_aggregate_representations(
            datasets_by_use=datasets_by_use,
            representation_strategy=representation_strategy,
            representation_id=representation_id,
            combo_cfg=combo_cfg,
            aggregation_level=aggregation_level,
            exclusion_level=exclusion_level,
            representation_cache_params=representation_cache_params,
        )

        # Convert per-use buckets into final search roles.
        logger.info("[SlideRetrieval] Stage 3/3: running search strategy")
        reference_representations, query_representations = (
            self._split_representations_by_use(
                representations_by_use=representations_by_use,
                include_query_reference_as_queries=include_query_reference_as_queries,
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
            search_strategy_name,
            params=combo_cfg.get_hyperparams("search_strategy"),
            config=getattr(self.experiment, "cfg", None),
        )

        search_strategy.build_database(reference_representations)
        search_workers = self._resolve_search_workers(
            num_queries=len(query_representations)
        )
        logger.info(
            "[SlideRetrieval] Search stage ready | queries=%d, references=%d, workers=%d",
            len(query_representations),
            len(reference_representations),
            search_workers,
        )
        prepare_queries = getattr(search_strategy, "prepare_queries", None)
        search_prepared = getattr(search_strategy, "search_prepared", None)
        if callable(prepare_queries) and callable(search_prepared):
            logger.info(
                "[SlideRetrieval] Preparing %d search query item(s) before ranking.",
                len(query_representations),
            )
            prepared_query_items = prepare_queries(query_representations)
            results = self._run_search_items(
                search_items=prepared_query_items,
                search_fn=search_prepared,
                search_workers=search_workers,
            )
        else:
            results = self._run_search_items(
                search_items=query_representations,
                search_fn=search_strategy.search,
                search_workers=search_workers,
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
            query_representations=query_representations,
            reference_representations=reference_representations,
        )
        run_dir = self._build_run_dir(
            output_mode=output_mode,
            inference_run_root=inference_run_root,
            tiling_id=tiling_id,
            feature_name=feature_name,
            representation_name=representation_name,
            search_strategy_name=search_strategy_name,
            run_hash=manifest.build_run_hash(),
        )
        # Persist both run configuration and ranking outputs under one run directory.
        run_dir.mkdir(parents=True, exist_ok=True)
        write_slide_retrieval_manifest(run_dir / "manifest.json", manifest)
        write_slide_retrieval_results_xlsx(run_dir / "query_results.xlsx", results)
        logger.info(
            "[SlideRetrieval] Completed execute | output_dir=%s, queries=%d, references=%d",
            run_dir,
            len(query_representations),
            len(reference_representations),
        )

        return {
            "output_dir": str(run_dir),
            "num_queries": len(query_representations),
            "num_reference_items": len(reference_representations),
        }

    def _resolve_search_workers(self, *, num_queries: int) -> int:
        """Resolve query-level search parallelism from slide_retrieval config."""
        slide_retrieval_cfg = getattr(self.cfg, "slide_retrieval", None)
        configured_workers = getattr(slide_retrieval_cfg, "search_workers", 1)
        try:
            search_workers = int(configured_workers or 1)
        except (TypeError, ValueError):
            search_workers = 1
        return min(max(1, search_workers), max(1, int(num_queries)))

    def _representation_cache_params(
        self, representation_strategy: Any
    ) -> dict[str, Any]:
        """Return the cache identity for a slide-local representation."""
        params = dict(representation_strategy.hyperparam_values())
        params["_schema_version"] = 2
        if str(getattr(representation_strategy, "name", "")).startswith("yottixel"):
            params["_kmeans_n_init"] = 10
            params["_random_state"] = getattr(self.cfg.experiment, "random_state", None)
        return params

    def _resolve_representation_loader_workers(
        self,
        *,
        representation_strategy: Any,
        default_workers: int,
    ) -> int:
        """Resolve DataLoader workers for retrieval representation creation."""
        slide_retrieval_cfg = getattr(self.cfg, "slide_retrieval", None)
        configured_workers = getattr(
            slide_retrieval_cfg,
            "representation_loader_workers",
            None,
        )
        if configured_workers is None:
            configured_workers = getattr(
                representation_strategy,
                "preferred_loader_workers",
                default_workers,
            )
        try:
            loader_workers = int(configured_workers or 0)
        except (TypeError, ValueError):
            loader_workers = int(default_workers or 0)
        return max(0, loader_workers)

    def _resolve_representation_workers(
        self,
        *,
        representation_strategy: Any,
        default_workers: int,
    ) -> int:
        """Resolve thread workers for retrieval representation materialization."""
        slide_retrieval_cfg = getattr(self.cfg, "slide_retrieval", None)
        configured_workers = getattr(
            slide_retrieval_cfg,
            "representation_workers",
            None,
        )
        if configured_workers is None:
            configured_workers = getattr(
                representation_strategy,
                "preferred_materialization_workers",
                default_workers,
            )
        try:
            representation_workers = int(configured_workers or 1)
        except (TypeError, ValueError):
            representation_workers = int(default_workers or 1)
        return max(1, representation_workers)

    def _run_search_items(
        self,
        *,
        search_items: list[Any],
        search_fn: Any,
        search_workers: int,
    ) -> list[SearchResult]:
        """Run independent search items, preserving input order."""
        if search_workers <= 1:
            return [
                search_fn(search_item)
                for search_item in tqdm(
                    search_items,
                    desc="[SlideRetrieval] Search queries",
                    unit="query",
                    mininterval=_PROGRESS_MIN_INTERVAL_SECONDS,
                    ncols=_PROGRESS_NCOLS,
                )
            ]

        results_by_index: list[SearchResult | None] = [None] * len(search_items)
        with ThreadPoolExecutor(max_workers=search_workers) as executor:
            future_to_index = {
                executor.submit(search_fn, search_item): index
                for index, search_item in enumerate(search_items)
            }
            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc="[SlideRetrieval] Search queries",
                unit="query",
                mininterval=_PROGRESS_MIN_INTERVAL_SECONDS,
                ncols=_PROGRESS_NCOLS,
            ):
                results_by_index[future_to_index[future]] = future.result()

        missing_indices = [
            index for index, result in enumerate(results_by_index) if result is None
        ]
        if missing_indices:
            raise RuntimeError(
                f"Search completed without results for query indices: {missing_indices}"
            )
        return [result for result in results_by_index if result is not None]

    def _resolve_exclusion_level(self) -> ExclusionLevel:
        # Read task-specific exclusion policy, defaulting to patient-level filtering.
        slide_retrieval_cfg = getattr(self.cfg, "slide_retrieval", None)
        return getattr(slide_retrieval_cfg, "exclusion_level", "patient")

    def _validate_dataset_context(
        self,
        *,
        datasets_by_use: dict[str, list[BagDataset]],
        tiling_id: str,
        aggregation_level: str,
    ) -> None:
        # Flatten all provided datasets so shared context checks are global.
        all_bag_datasets = [
            bag_dataset
            for bag_datasets in datasets_by_use.values()
            for bag_dataset in bag_datasets
        ]
        if not all_bag_datasets:
            raise ValueError("No bag datasets were provided for slide retrieval.")

        # Ensure every dataset was built for the same tiling configuration as the combo.
        dataset_tiling_ids = {
            str(bag_dataset.tiling_id) for bag_dataset in all_bag_datasets
        }
        if dataset_tiling_ids != {str(tiling_id)}:
            raise ValueError(
                "Slide retrieval datasets do not match the expected tiling_id. "
                f"Expected {tiling_id!r}, got {sorted(dataset_tiling_ids)}."
            )

        aggregation_levels = {
            str(bag_dataset.aggregation_level) for bag_dataset in all_bag_datasets
        }
        if aggregation_levels != {str(aggregation_level)}:
            raise ValueError(
                "Slide retrieval datasets do not match the expected "
                "aggregation_level. "
                f"Expected {aggregation_level!r}, got {sorted(aggregation_levels)}."
            )

    def _materialize_and_aggregate_representations(
        self,
        *,
        datasets_by_use: dict[str, list[BagDataset]],
        representation_strategy: Any,
        representation_id: str,
        combo_cfg: ComboConfig,
        aggregation_level: str,
        exclusion_level: ExclusionLevel,
        representation_cache_params: dict[str, Any],
    ) -> dict[str, list[RetrievalRepresentation]]:
        """Build each physical slide once, then aggregate logical retrieval items.

        The H5 cache belongs to the physical slide artifact.  Grouped case and
        patient items are assembled only after those slide-local selections
        have been loaded, so selectors can never see a concatenated raw bag.
        """
        cache: dict[tuple[str, str], RetrievalRepresentation] = {}
        by_use: dict[str, list[RetrievalRepresentation]] = {}
        for use, datasets in datasets_by_use.items():
            if use not in _VALID_RETRIEVAL_USES:
                raise ValueError(f"Unsupported retrieval dataset use {use!r}.")
            output = by_use.setdefault(use, [])
            for dataset in datasets:
                if not isinstance(dataset, SlideRetrievalBagDataset):
                    raise TypeError(
                        "slide_retrieval requires SlideRetrievalBagDataset instances. "
                        f"Got {type(dataset).__name__}."
                    )

                # The standalone representation CLI stores the final logical
                # retrieval item under ``artifacts_dir/slide_retrieval``.
                # Consult that shared cache before opening physical-slide
                # artifacts, so precomputation is reused by benchmark runs.
                cached_representations, missing_subset = (
                    self._collect_existing_representations(
                        bag_dataset=dataset,
                        representation_id=representation_id,
                        aggregation_level=aggregation_level,
                        exclusion_level=exclusion_level,
                    )
                )
                if missing_subset is None:
                    output.extend(cached_representations)
                    continue

                cached_by_sample_id = {
                    representation.sample_id: representation
                    for representation in cached_representations
                }
                for index in range(dataset.num_bags):
                    group = dataset.get_sample(index)
                    cached_representation = cached_by_sample_id.get(group.sample_id)
                    if cached_representation is not None:
                        output.append(cached_representation)
                        continue

                    slides: list[RetrievalRepresentation] = []
                    for slide_id, artifact_path in zip(
                        group.slide_ids, group.artifact_paths
                    ):
                        key = (str(Path(artifact_path).resolve()), str(slide_id))
                        if key not in cache:
                            slide_sample = BagSample(
                                sample_id=str(slide_id),
                                slide_ids=[str(slide_id)],
                                artifact_paths=[Path(artifact_path)],
                                category=group.category,
                                patient_id=group.patient_id,
                                case_id=group.case_id,
                                metadata=dict(group.metadata),
                            )
                            try:
                                cache[key] = self._load_or_create_slide_representation(
                                    dataset=dataset,
                                    sample=slide_sample,
                                    representation_strategy=representation_strategy,
                                    representation_id=representation_id,
                                    combo_cfg=combo_cfg,
                                    representation_cache_params=representation_cache_params,
                                )
                            except Exception as exc:
                                raise RuntimeError(
                                    "Slide retrieval representation creation failed "
                                    f"for sample {slide_sample.sample_id!r}: {exc}"
                                ) from exc
                        slides.append(cache[key])
                    if aggregation_level == "slide":
                        slide_representation = slides[0]
                        # Never mutate the cached object: a slide can belong
                        # to more than one use and runtime identities differ.
                        representation = RetrievalRepresentation(
                            sample_id=group.sample_id,
                            data=slide_representation.data,
                            representation_type=slide_representation.representation_type,
                            additional_data=dict(slide_representation.additional_data),
                        )
                        representation.metadata.category = group.category
                        representation.metadata.patient_id = group.patient_id
                        representation.metadata.case_id = group.case_id
                    else:
                        representation = aggregate_slide_representations(
                            sample_id=group.sample_id,
                            member_slide_ids=list(group.slide_ids),
                            slide_representations=slides,
                            category=group.category,
                            patient_id=group.patient_id,
                            case_id=group.case_id,
                        )
                    representation.exclusion_key = self._build_exclusion_key(
                        sample=group,
                        aggregation_level=aggregation_level,
                        exclusion_level=exclusion_level,
                    )
                    representation.additional_data = {
                        **representation.additional_data,
                        "dataset_name": dataset.name,
                        "source_slide_ids": list(group.slide_ids),
                    }
                    output.append(representation)
        return by_use

    def _load_or_create_slide_representation(
        self,
        *,
        dataset: SlideRetrievalBagDataset,
        sample: BagSample,
        representation_strategy: Any,
        representation_id: str,
        combo_cfg: ComboConfig,
        representation_cache_params: dict[str, Any],
    ) -> RetrievalRepresentation:
        """Load or create one slide-local representation in its source artifact."""
        artifact_path = sample.artifact_paths[0]
        with FileHandleH5(artifact_path, mode="r") as artifact:
            cached = load_slide_retrieval_representation(
                retrieval_artifact=artifact,
                tile_id=dataset.tiling_id,
                representation_id=representation_id,
                entry_id=None,
            )
        if cached is not None:
            return cached

        # The adapter deliberately exposes a bag consisting of this one slide.
        # Custom strategy loaders receive the same single-slide sample.
        class _SingleSlideDataset:
            tiling_id = dataset.tiling_id
            extractor_name = dataset.extractor_name

            def load_bag(self, _: int) -> Any:
                return dataset._load_slide_bag(artifact_path)

        inputs = representation_strategy.load_sample(
            index=0, sample=sample, base_dataset=_SingleSlideDataset()
        )
        representation = representation_strategy.run(
            sample=sample, bag_dataset=dataset, combo_cfg=combo_cfg, **inputs
        )
        representation.sample_id = sample.sample_id
        with atomic_slide_artifact_write(artifact_path) as artifact:
            save_slide_retrieval_representation(
                retrieval_artifact=artifact,
                tile_id=dataset.tiling_id,
                representation_id=representation_id,
                entry_id=None,
                representation=representation,
                params=representation_cache_params,
            )
        return representation

    def _collect_existing_representations(
        self,
        *,
        bag_dataset: SlideRetrievalBagDataset,
        representation_id: str,
        aggregation_level: str,
        exclusion_level: ExclusionLevel,
    ) -> tuple[
        list[RetrievalRepresentation],
        Subset[SlideRetrievalBagDataset] | None,
    ]:
        """
        Load cached representations and return a subset of missing samples.

        Inputs:
        - `bag_dataset`: retrieval dataset for one configured source dataset.
        - `representation_id`: retrieval representation artifact key.
        - `aggregation_level`: active experiment aggregation level.
        - `exclusion_level`: configured exclusion key level.

        Returns:
        - tuple:
          - `existing_representations`: cached representations already present.
          - `missing_subset`: subset with uncached sample indices, or `None` when
            all samples were available.
        """
        existing_representations: list[RetrievalRepresentation] = []
        missing_indices: list[int] = []

        # Try loading each sample from cache and track only missing indices.
        for index in range(bag_dataset.num_bags):
            sample = bag_dataset.get_sample(index)
            # Build the per-sample cache address in the retrieval artifact store.
            artifact_path = build_retrieval_representation_artifact_path(
                artifacts_dir=bag_dataset.artifacts_dir,
                aggregation_level=bag_dataset.aggregation_level,
                sample_id=sample.sample_id,
            )
            entry_id = build_retrieval_representation_entry_id(
                sorted(sample.slide_ids),
                aggregation_level=aggregation_level,
            )
            if artifact_path.exists():
                with FileHandleH5(artifact_path, mode="r") as retrieval_artifact:
                    cached_representation = load_slide_retrieval_representation(
                        retrieval_artifact=retrieval_artifact,
                        tile_id=bag_dataset.tiling_id,
                        representation_id=representation_id,
                        entry_id=entry_id,
                    )
            else:
                cached_representation = None
            if cached_representation is None:
                missing_indices.append(index)
                continue

            # Re-attach runtime metadata that is not persisted in the same shape.
            cached_representation.metadata.category = sample.category
            cached_representation.metadata.patient_id = sample.patient_id
            cached_representation.metadata.case_id = sample.case_id
            cached_representation.exclusion_key = self._build_exclusion_key(
                sample=sample,
                aggregation_level=aggregation_level,
                exclusion_level=exclusion_level,
            )
            cached_representation.additional_data = {
                **cached_representation.additional_data,
                "dataset_name": bag_dataset.name,
                "source_slide_ids": list(sample.slide_ids),
            }
            existing_representations.append(cached_representation)

        # Return both cache hits and a subset view for materialization misses.
        missing_subset = (
            Subset(bag_dataset, missing_indices) if missing_indices else None
        )
        return existing_representations, missing_subset

    def compute_retrieval_representations(
        self,
        *,
        bag_dataset: SlideRetrievalBagDataset,
        retrieval_loader: DataLoader[list[SlideRetrievalDatasetItem]],
        batch_thread_workers: int,
        combo_cfg: ComboConfig,
        representation_strategy: Any,
        representation_id: str,
        aggregation_level: str,
        exclusion_level: ExclusionLevel,
    ) -> tuple[list[RetrievalRepresentation], dict[str, str]]:
        """
        Create and persist retrieval representations for one retrieval dataset.

        Args:
            bag_dataset: Dataset that owns the target samples.
            retrieval_loader: Loader yielding retrieval batches to materialize.
            batch_thread_workers: Threads used per retrieval batch.
            combo_cfg: Active representation configuration.
            representation_strategy: Instantiated representation strategy.
            representation_id: Stable representation artifact key.
            aggregation_level: Active experiment aggregation level.
            exclusion_level: Configured exclusion key level.

        Returns:
            Created representations and failure details keyed by sample ID.
        """
        created_retrieval_representations: list[RetrievalRepresentation] = []
        creation_errors_by_sample: dict[str, str] = {}

        def _materialize_one(
            dataset_item: SlideRetrievalDatasetItem,
        ) -> RetrievalRepresentation:
            # Build the target artifact location for the current sample.
            artifact_path = build_retrieval_representation_artifact_path(
                artifacts_dir=bag_dataset.artifacts_dir,
                aggregation_level=bag_dataset.aggregation_level,
                sample_id=dataset_item.sample.sample_id,
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            entry_id = build_retrieval_representation_entry_id(
                sorted(dataset_item.sample.slide_ids),
                aggregation_level=aggregation_level,
            )

            # Compute the retrieval representation for the current sample
            representation = representation_strategy.run(
                sample=dataset_item.sample,
                bag_dataset=bag_dataset,
                combo_cfg=combo_cfg,
                **dataset_item.inputs,
            )

            # Attach retrieval-time filtering and provenance fields.
            representation.metadata.category = dataset_item.sample.category
            representation.metadata.patient_id = dataset_item.sample.patient_id
            representation.metadata.case_id = dataset_item.sample.case_id
            representation.exclusion_key = self._build_exclusion_key(
                sample=dataset_item.sample,
                aggregation_level=aggregation_level,
                exclusion_level=exclusion_level,
            )
            representation.additional_data = {
                **representation.additional_data,
                "dataset_name": bag_dataset.name,
                "source_slide_ids": list(dataset_item.sample.slide_ids),
            }

            # Persist under one atomic write transaction so both new-file and
            # existing-file updates are replace-only on successful completion.
            with atomic_slide_artifact_write(artifact_path) as retrieval_artifact:
                save_slide_retrieval_representation(
                    retrieval_artifact=retrieval_artifact,
                    tile_id=bag_dataset.tiling_id,
                    representation_id=representation_id,
                    entry_id=entry_id,
                    representation=representation,
                    params=representation_strategy.hyperparam_values(),
                )
            return representation

        # Materialize each retrieval batch concurrently with one thread per batch item.
        with ThreadPoolExecutor(max_workers=max(1, batch_thread_workers)) as executor:
            for retrieval_batch in retrieval_loader:
                results_by_index: list[RetrievalRepresentation | None] = [None] * len(
                    retrieval_batch
                )
                future_to_item = {
                    executor.submit(_materialize_one, dataset_item): (
                        index,
                        str(dataset_item.sample.sample_id),
                    )
                    for index, dataset_item in enumerate(retrieval_batch)
                }
                for future in tqdm(
                    as_completed(future_to_item),
                    total=len(future_to_item),
                    desc=f"[SlideRetrieval] Representations {bag_dataset.name}",
                    unit="sample",
                    mininterval=_PROGRESS_MIN_INTERVAL_SECONDS,
                    ncols=_PROGRESS_NCOLS,
                ):
                    index, sample_id = future_to_item[future]
                    try:
                        results_by_index[index] = future.result()
                    except Exception as exc:
                        # Continue processing remaining items while collecting failures.
                        tb_text = "".join(
                            traceback.format_exception(
                                type(exc),
                                exc,
                                exc.__traceback__,
                            )
                        ).strip()
                        creation_errors_by_sample[sample_id] = tb_text
                        continue
                created_retrieval_representations.extend(
                    result for result in results_by_index if result is not None
                )

        return created_retrieval_representations, creation_errors_by_sample

    def _build_exclusion_key(
        self,
        *,
        sample: BagSample,
        aggregation_level: str,
        exclusion_level: ExclusionLevel,
    ) -> str | None:
        # Map configured exclusion policy to the corresponding sample identity key.
        if exclusion_level == "none":
            return None
        if exclusion_level == "slide":
            # Slide-level exclusion requires one item per slide.
            if aggregation_level != "slide":
                raise ValueError(
                    "slide_retrieval.exclusion_level='slide' requires "
                    "experiment.aggregation_level='slide'."
                )
            return str(sample.sample_id)
        # Case- and patient-level keys may be unavailable in some datasets.
        if exclusion_level == "case":
            return None if sample.case_id is None else str(sample.case_id)
        if exclusion_level == "patient":
            return None if sample.patient_id is None else str(sample.patient_id)
        raise ValueError(
            f"Unsupported slide retrieval exclusion level: {exclusion_level!r}"
        )

    def _split_representations_by_use(
        self,
        *,
        representations_by_use: dict[str, list[RetrievalRepresentation]],
        include_query_reference_as_queries: bool = True,
    ) -> tuple[list[RetrievalRepresentation], list[RetrievalRepresentation]]:
        # Build search DB from reference + shared items.
        reference_representations: list[RetrievalRepresentation] = []
        # Build query set from query + shared items.
        query_representations: list[RetrievalRepresentation] = []

        # Shared items are intentionally included in both sets.
        reference_representations.extend(representations_by_use.get("reference", []))
        reference_representations.extend(
            representations_by_use.get("query_reference", [])
        )

        query_representations.extend(representations_by_use.get("query", []))
        if include_query_reference_as_queries:
            query_representations.extend(
                representations_by_use.get("query_reference", [])
            )

        return reference_representations, query_representations

    def _build_run_dir(
        self,
        *,
        output_mode: str,
        inference_run_root: Path | None,
        tiling_id: str,
        feature_name: str,
        representation_name: str,
        search_strategy_name: str,
        run_hash: str,
    ) -> Path:
        if output_mode == "benchmark":
            output_root = build_slide_retrieval_output_root(
                project_root=str(self.experiment.project_root),
                tiling_id=tiling_id,
                feature_name=feature_name,
                slide_representation=representation_name,
                search_method=search_strategy_name,
            )
            return output_root / f"run_{run_hash}"

        if output_mode == "inference":
            if inference_run_root is None:
                raise ValueError("inference_run_root is required for inference output.")
            return build_slide_retrieval_inference_output_root(
                inference_run_root=inference_run_root,
                tiling_id=tiling_id,
                feature_name=feature_name,
                slide_representation=representation_name,
                search_method=search_strategy_name,
                run_hash=run_hash,
            )

        raise ValueError(f"Unsupported slide retrieval output mode: {output_mode!r}")

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
        query_representations: list[RetrievalRepresentation],
        reference_representations: list[RetrievalRepresentation],
    ) -> SlideRetrievalManifest:
        # Capture the run configuration and key output counters for reproducibility.
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
            combo_cfg=combo_cfg.to_dict(),
            query_sample_ids=[
                str(representation.sample_id).strip()
                for representation in query_representations
            ],
            reference_sample_ids=[
                str(representation.sample_id).strip()
                for representation in reference_representations
            ],
        )

    def _validate_combination_compatibility(
        self,
        *,
        datasets_by_use: dict[str, list[BagDataset]],
        representation_name: str,
        search_strategy_name: str,
        aggregation_level: str,
        exclusion_level: ExclusionLevel,
    ) -> tuple[bool, str]:
        # Determine one shared feature level across all dataset uses.
        all_bag_datasets = [
            bag_dataset
            for bag_datasets in datasets_by_use.values()
            for bag_dataset in bag_datasets
        ]
        feature_levels = {
            bag_dataset.get_feature_level() for bag_dataset in all_bag_datasets
        }

        # Stop early when dataset feature structure cannot be interpreted reliably.
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
            return (
                False,
                f"Inconsistent feature levels across datasets: {sorted(feature_levels)}",
            )

        # The single shared level is used for all downstream compatibility checks.
        feature_level = next(iter(feature_levels))

        # Resolve class-level compatibility metadata from registries by strategy name.
        try:
            supported_feature_levels = (
                get_representation_strategy_supported_feature_levels(
                    representation_name
                )
            )
            representation_kind = get_representation_strategy_output_kind(
                representation_name
            )
            supported_representation_kinds = (
                get_search_strategy_supported_representation_kinds(search_strategy_name)
            )
        except ValueError as exc:
            return False, str(exc)

        if feature_level not in supported_feature_levels:
            return (
                False,
                f"Representation strategy '{representation_name}' "
                f"does not support feature level '{feature_level}'.",
            )

        if exclusion_level == "slide" and aggregation_level != "slide":
            return (
                False,
                "slide_retrieval.exclusion_level='slide' is only supported when "
                "experiment.aggregation_level='slide'.",
            )

        # Ensure representation output kind can be consumed by the search strategy.
        if representation_kind not in supported_representation_kinds:
            return (
                False,
                f"Search strategy '{search_strategy_name}' does not support "
                f"representation kind '{representation_kind}'.",
            )

        return True, ""
