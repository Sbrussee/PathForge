"""Plan, load, create, and assemble physical-slide retrieval representations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathforge.core.datasets.bag_dataset import (
    BagDataset,
    BagSample,
    SlideRetrievalBagDataset,
)
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.core.io.slide_artifacts.atomic import atomic_slide_artifact_write
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.io.slide_retrieval.retrieval_representations import (
    retrieval_representation_entry_exists,
)
from pathforge.slide_retrieval.aggregation import aggregate_slide_representations
from pathforge.slide_retrieval.io import (
    load_slide_retrieval_representation,
    save_slide_retrieval_representation,
)
from pathforge.slide_retrieval.representation_strategies.storage import (
    build_retrieval_representation_artifact_path,
)
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.types import ExclusionLevel

# A slide cache is scoped by its destination, tiling, and representation identity.
RepresentationKey = tuple[Path, str, str]


@dataclass(frozen=True)
class SlideRepresentationRequest:
    """Single-slide inputs and cache address; no feature arrays are loaded."""

    key: RepresentationKey
    dataset: SlideRetrievalBagDataset
    sample: BagSample


@dataclass(frozen=True)
class RetrievalGroup:
    """Ordered member keys and runtime identity for one logical retrieval item."""

    dataset_name: str
    sample: BagSample
    member_keys: tuple[RepresentationKey, ...]


@dataclass(frozen=True)
class RepresentationPlan:
    """Disjoint slide requests plus groups in their original dataset order."""

    existing: dict[RepresentationKey, SlideRepresentationRequest]
    missing: dict[RepresentationKey, SlideRepresentationRequest]
    groups_by_use: dict[str, list[RetrievalGroup]]


def plan_representations(
    *,
    datasets_by_use: dict[str, list[BagDataset]],
    representation_id: str,
) -> RepresentationPlan:
    """Inspect H5 entry presence without reading arrays or writing artifacts.

    Each physical cache address is checked once across all groups and uses.
    The returned plan can drive creation alone or full retrieval preparation.

    Example: ``plan_representations(datasets_by_use=datasets, representation_id=key)``.
    """
    existing: dict[RepresentationKey, SlideRepresentationRequest] = {}
    missing: dict[RepresentationKey, SlideRepresentationRequest] = {}
    groups_by_use: dict[str, list[RetrievalGroup]] = {}

    # Walk each retrieval role and its datasets to preserve output ordering.
    for use, datasets in datasets_by_use.items():
        if use not in {"reference", "query", "query_reference"}:
            raise ValueError(f"Unsupported retrieval dataset use {use!r}.")
        groups = groups_by_use.setdefault(use, [])
        for dataset in datasets:
            if not isinstance(dataset, SlideRetrievalBagDataset):
                raise TypeError(
                    "slide_retrieval requires SlideRetrievalBagDataset instances. "
                    f"Got {type(dataset).__name__}."
                )
            # Record each logical slide/case/patient and inspect its physical members.
            for index in range(dataset.num_bags):
                group = dataset.get_sample(index)
                if not group.slide_ids or len(group.slide_ids) != len(
                    group.artifact_paths
                ):
                    raise ValueError(
                        f"Invalid physical-slide members for sample {group.sample_id!r}."
                    )
                member_keys: list[RepresentationKey] = []
                # Schedule each physical member once, even across multiple roles.
                for slide_id, source_path in zip(group.slide_ids, group.artifact_paths):
                    artifact_path = build_retrieval_representation_artifact_path(
                        artifacts_dir=dataset.artifacts_dir, slide_id=str(slide_id)
                    ).resolve()
                    key = (artifact_path, str(dataset.tiling_id), representation_id)
                    member_keys.append(key)
                    if key in existing or key in missing:
                        continue
                    sample = BagSample(
                        sample_id=str(slide_id),
                        slide_ids=[str(slide_id)],
                        artifact_paths=[Path(source_path)],
                        category=group.category,
                        patient_id=group.patient_id,
                        case_id=group.case_id,
                        metadata=dict(group.metadata),
                        annotations_df=getattr(group, "annotations_df", None),
                    )
                    request = SlideRepresentationRequest(key, dataset, sample)
                    present = False
                    if artifact_path.is_file():
                        with FileHandleH5(artifact_path, mode="r") as artifact:
                            present = retrieval_representation_entry_exists(
                                artifact, key[1], representation_id, None
                            )
                    if present:
                        existing[key] = request
                    else:
                        missing[key] = request
                groups.append(RetrievalGroup(dataset.name, group, tuple(member_keys)))
    return RepresentationPlan(existing, missing, groups_by_use)


def load_representations(
    requests: dict[RepresentationKey, SlideRepresentationRequest],
) -> dict[RepresentationKey, RetrievalRepresentation]:
    """Load only planned existing slides, returning representations by cache key.

    Example: ``loaded = load_representations(plan.existing)``.
    A cache disappearing after inspection raises rather than triggering creation.
    """
    loaded: dict[RepresentationKey, RetrievalRepresentation] = {}
    # Read each unique cache once; groups reuse these objects during assembly.
    for key, request in requests.items():
        with FileHandleH5(key[0], mode="r") as artifact:
            representation = load_slide_retrieval_representation(
                retrieval_artifact=artifact,
                tile_id=key[1],
                representation_id=key[2],
                entry_id=None,
            )
        if representation is None:
            raise RuntimeError(
                "Planned representation disappeared "
                f"for slide {request.sample.sample_id!r}."
            )
        loaded[key] = representation
    return loaded


def create_representations(
    requests: dict[RepresentationKey, SlideRepresentationRequest],
    *,
    representation_strategy: Any,
    combo_cfg: ComboConfig,
    representation_cache_params: dict[str, Any],
    feature_level: str,
) -> dict[RepresentationKey, RetrievalRepresentation]:
    """Compute and atomically persist only supplied missing physical slides.

    Both this function and ``load_representations`` return the same keyed type.
    Example: ``created = create_representations(plan.missing, ...)``.
    """
    created: dict[RepresentationKey, RetrievalRepresentation] = {}
    # Run the selector on one physical slide at a time, never a grouped raw bag.
    for key, request in requests.items():
        dataset, sample = request.dataset, request.sample

        class _SingleSlideDataset:
            tiling_id = dataset.tiling_id
            extractor_name = dataset.extractor_name

            def load_bag(self, _: int) -> Any:
                """Read the source feature bag for this physical slide."""
                return dataset._load_slide_bag(sample.artifact_paths[0])

        try:
            inputs = representation_strategy.load_sample(
                index=0, sample=sample, base_dataset=_SingleSlideDataset()
            )
            representation = representation_strategy.run(
                sample=sample, bag_dataset=dataset, combo_cfg=combo_cfg, **inputs
            )
            representation.sample_id = sample.sample_id
            representation.feature_level = feature_level
            with atomic_slide_artifact_write(key[0]) as artifact:
                save_slide_retrieval_representation(
                    retrieval_artifact=artifact,
                    tile_id=key[1],
                    representation_id=key[2],
                    entry_id=None,
                    representation=representation,
                    params=representation_cache_params,
                )
        except Exception as exc:
            raise RuntimeError(
                "Slide retrieval representation creation failed "
                f"for sample {sample.sample_id!r}: {exc}"
            ) from exc
        created[key] = representation
    return created


def assemble_representations(
    plan: RepresentationPlan,
    *,
    loaded: dict[RepresentationKey, RetrievalRepresentation],
    created: dict[RepresentationKey, RetrievalRepresentation],
    aggregation_level: str,
    exclusion_level: ExclusionLevel,
    build_exclusion_key: Callable[..., str | None],
) -> dict[str, list[RetrievalRepresentation]]:
    """Combine both sources and assemble runtime items without artifact I/O.

    Member order comes from the plan, independent of cache hit/miss ordering.
    Cached objects remain slide-local; metadata is attached to fresh group objects.
    Example: ``assemble_representations(plan, loaded=loaded, created=created, ...)``.
    """
    if loaded.keys() & created.keys():
        raise ValueError("Loaded and created representation keys must be disjoint.")
    slides_by_key = {**loaded, **created}
    expected_keys = plan.existing.keys() | plan.missing.keys()
    if slides_by_key.keys() != expected_keys:
        raise ValueError("Representations do not match the planned physical slides.")
    by_use: dict[str, list[RetrievalRepresentation]] = {}
    # Assemble each role's groups from members in their original slide order.
    for use, groups in plan.groups_by_use.items():
        output = by_use.setdefault(use, [])
        for group in groups:
            sample = group.sample
            slides = [slides_by_key[key] for key in group.member_keys]
            if aggregation_level == "slide":
                if len(slides) != 1:
                    raise ValueError(
                        "Slide aggregation requires exactly one physical member."
                    )
                slide = slides[0]
                representation = RetrievalRepresentation(
                    sample_id=sample.sample_id,
                    data=slide.data,
                    representation_type=slide.representation_type,
                    feature_level=slide.feature_level,
                    additional_data=dict(slide.additional_data),
                )
                representation.metadata.category = sample.category
                representation.metadata.patient_id = sample.patient_id
                representation.metadata.case_id = sample.case_id
            else:
                representation = aggregate_slide_representations(
                    sample_id=sample.sample_id,
                    member_slide_ids=list(sample.slide_ids),
                    slide_representations=slides,
                    category=sample.category,
                    patient_id=sample.patient_id,
                    case_id=sample.case_id,
                )
                representation.feature_level = slides[0].feature_level
            representation.exclusion_key = build_exclusion_key(
                sample=sample,
                aggregation_level=aggregation_level,
                exclusion_level=exclusion_level,
            )
            representation.additional_data.update(
                dataset_name=group.dataset_name, source_slide_ids=list(sample.slide_ids)
            )
            output.append(representation)
    return by_use
