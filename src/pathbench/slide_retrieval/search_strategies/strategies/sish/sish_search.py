from __future__ import annotations

import gc
import json
import logging
import os
import pickle
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency for precomputed mode
    torch = None
try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency
    psutil = None

from pathbench.slide_retrieval.hyperparams import HyperParam
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.search_strategies.base import BaseSearchStrategy
from pathbench.slide_retrieval.search_strategies.registry import (
    register_search_strategy,
)
from pathbench.slide_retrieval.search_strategies.strategies.sish.sish_eval import (
    Clean,
    Filtered_BY_Prediction,
    Uncertainty_Cal,
)
from pathbench.slide_retrieval.search_strategies.strategies.sish.sish_veb import VEB
from pathbench.slide_retrieval.search_strategies.types import (
    SearchDatabaseItem,
    SearchHit,
    SearchResult,
)


logger = logging.getLogger(__name__)


def scale_to_minus1_to_1(x: Any) -> Any:
    """
    Transform a tensor from ``[0, 1]`` into ``[-1, 1]``.

    Inputs:
        x:
            Float tensor with arbitrary shape ``(...,)`` and values expected in
            ``[0, 1]``.

    Output:
        torch.Tensor:
            Tensor with the same shape as ``x`` mapped to ``[-1, 1]``.
    """
    return (2.0 * x) - 1.0


def hamming_bytes(left: bytes, right: bytes) -> int:
    """
    Compute the Hamming distance between two equal-length packed bit strings.

    Inputs:
        left:
            Packed binary feature bytes with shape ``(B,)``.
        right:
            Packed binary feature bytes with shape ``(B,)``.

    Output:
        int:
            Number of differing bits.
    """
    if len(left) != len(right):
        raise ValueError(
            "Packed byte strings must have equal length for Hamming distance. "
            f"Got {len(left)} and {len(right)}."
        )
    if len(left) == 0:
        return 0

    xor = np.frombuffer(left, dtype=np.uint8) ^ np.frombuffer(right, dtype=np.uint8)
    return int(sum(int(value).bit_count() for value in xor))


def log_mem(tag: str) -> None:
    """
    Log lightweight host/CUDA memory diagnostics for long SISH indexing runs.

    Inputs:
        tag:
            Human-readable checkpoint name.

    Output:
        None.
    """
    if psutil is None:
        logger.debug("[%s] psutil not installed; skipping memory log.", tag)
        return

    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / 1e9
    child_rss_gb = sum(
        (
            child.memory_info().rss
            for child in process.children(recursive=True)
            if child.is_running()
        ),
        0,
    ) / 1e9
    if _torch_cuda_available():
        cuda_alloc_mb = torch.cuda.memory_allocated() / 1e6
        cuda_reserved_mb = torch.cuda.memory_reserved() / 1e6
    else:
        cuda_alloc_mb = 0.0
        cuda_reserved_mb = 0.0

    logger.info(
        "[%s] RSS=%.2f GB kids=%.2f GB CUDA alloc/res=%.0f/%.0f MB",
        tag,
        rss_gb,
        child_rss_gb,
        cuda_alloc_mb,
        cuda_reserved_mb,
    )


def _torch_cuda_available() -> bool:
    """Return whether CUDA is available when torch is installed."""
    return bool(torch is not None and torch.cuda.is_available())


@register_search_strategy("sish")
class SISHSearch(BaseSearchStrategy):
    """
    Selection of Informative Samples in Histopathology (SISH) retrieval.

    Semantic goal:
        Preserve the original SISH bidirectional Van Emde Boas traversal,
        Hamming-based patch matching, uncertainty-based patch-bag cleaning, and
        slide-level aggregation while adapting the inputs and outputs to
        PathBench 2.0 search interfaces.

    Inputs:
        params:
            Hyperparameter mapping controlling traversal and ranking.
        config:
            Optional experiment config carrying SISH asset/output paths. This is
            required for VQ-VAE-backed index/query preparation unless the
            representations already provide precomputed ``sish_patch_indices``.

    Outputs:
        Use ``build_database(...)`` with multi-vector patch representations, then
        ``search(...)`` to obtain a ``SearchResult`` with ranked slide-level
        ``SearchHit`` entries.
    """

    name = "sish"
    supports = {"patch_vector"}
    supported_representation_kinds = frozenset({"patch_vector"})

    k = HyperParam(int, default=10, min=1, help="top-k retrieval depth")
    seed_interval_c = HyperParam(int, default=50, min=1, help="seed index stride")
    seed_fanout_t = HyperParam(int, default=10, min=1, help="seed fanout per side")
    pre_step = HyperParam(int, default=375, min=0, help="max predecessor steps")
    succ_step = HyperParam(int, default=375, min=0, help="max successor steps")
    hamming_thr = HyperParam(int, default=512, min=0, help="Hamming accept threshold")
    resume_shards = HyperParam(bool, default=True, help="resume shard building")
    shard_size = HyperParam(int, default=25, min=1, help="slides per shard")
    return_patch_matches = HyperParam(
        bool,
        default=False,
        help="return patch-level matches instead of slide-level hits",
    )

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(params=params, **kwargs)
        self.config = self.extra.get("config")
        self.project_root = self.extra.get("project_root")
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch is not None
            else "cpu"
        )

        self.codebook_semantic: dict[int, int] | None = None
        self.vqvae: Any | None = None
        self.transform_vqvqe: Any | None = None

        self.pool_layers = (
            [
                torch.nn.AvgPool2d((2, 2)),
                torch.nn.AvgPool2d((2, 2)),
                torch.nn.AvgPool2d((2, 2)),
            ]
            if torch is not None
            else []
        )

        self.meta: dict[int, list[dict[str, Any]]] = {}
        self.keys: list[int] = []
        self.vebtree: VEB | None = None
        self.pool = None
        self._database_item_by_id: dict[str, SearchDatabaseItem] = {}
        self._precompute: Any | None = None
        default_sish_dir = (
            Path(self.project_root) / "sish"
            if self.project_root is not None
            else Path.cwd() / "artifacts" / "sish"
        )

        if self.return_patch_matches:
            raise NotImplementedError(
                "return_patch_matches=True is not supported in PathBench 2.0."
            )

        self.shard_dir = self._resolve_path(
            [
                ("experiment", "sish", "shard_dir"),
                ("experiment", "SISH_metrics", "shard_dir"),
                ("sish", "shard_dir"),
            ],
            default=default_sish_dir / "tmp_shards",
        )
        self.index_veb_path = self._resolve_path(
            [
                ("experiment", "sish", "index_veb_path"),
                ("experiment", "SISH_metrics", "index_veb_path"),
                ("sish", "index_veb_path"),
            ],
            default=default_sish_dir / "veb.pkl",
        )
        self.meta_database_path = self._resolve_path(
            [
                ("experiment", "sish", "meta_database_path"),
                ("experiment", "SISH_metrics", "meta_database_path"),
                ("sish", "meta_database_path"),
            ],
            default=default_sish_dir / "meta.pkl",
        )
        self._codebook_path = self._resolve_path(
            [
                ("experiment", "sish", "codebook_semantic"),
                ("experiment", "SISH_metrics", "codebook_semantic"),
                ("sish", "codebook_semantic"),
            ],
            default=None,
        )
        self._checkpoint_path = self._resolve_path(
            [
                ("experiment", "sish", "vqvae_checkpoint"),
                ("experiment", "SISH_metrics", "vqvae_checkpoint"),
                ("sish", "vqvae_checkpoint"),
            ],
            default=None,
        )

    def build_database_item(
        self,
        representation: RetrievalRepresentation,
    ) -> SearchDatabaseItem:
        """
        Convert one retrieval representation into a searchable SISH item.

        Inputs:
            representation:
                Multi-vector patch representation with feature matrix shape
                ``(N, D)`` and optional SISH extras:
                - ``additional_data["sish_patch_indices"]`` with shape ``(N,)``
                - ``additional_data["sish_packed_bits"]`` with length ``N``
                - ``additional_data["mosaic_pkl_path"]`` as ``str``

        Output:
            SearchDatabaseItem:
                Standard container whose ``data`` field stores the prepared SISH
                payload dictionary.
        """
        self._validate_representations([representation])
        representation = self._maybe_enrich_representation(representation)

        return SearchDatabaseItem(
            sample_id=representation.sample_id,
            exclusion_key=representation.exclusion_key,
            data=self._build_item_payload(
                item_id=representation.sample_id,
                features=representation.data,
                additional_data=representation.additional_data,
            ),
            additional_data=representation.additional_data,
        )

    def prepare_query(
        self,
        query_representation: RetrievalRepresentation,
    ) -> SearchDatabaseItem:
        """
        Convert one query representation into the prepared SISH query payload.

        Inputs:
            query_representation:
                Query multi-vector representation with shape ``(N_q, D)``.

        Output:
            SearchDatabaseItem:
                Prepared query item with SISH payload in ``data``.
        """
        self._validate_representations([query_representation])
        query_representation = self._maybe_enrich_representation(query_representation)

        return SearchDatabaseItem(
            sample_id=query_representation.sample_id,
            exclusion_key=query_representation.exclusion_key,
            data=self._build_item_payload(
                item_id=query_representation.sample_id,
                features=query_representation.data,
                additional_data=query_representation.additional_data,
            ),
            additional_data=query_representation.additional_data,
        )

    def build_index(self) -> None:
        """
        Build the searchable VEB index from prepared SISH database items.

        Inputs:
            None. Uses ``self.search_database`` populated by ``build_database``.

        Outputs:
            None. Updates in-memory VEB/meta state and persists final artifacts.
        """
        self._database_item_by_id = {
            item.item_id: item for item in self.search_database
        }

        if any(self._payload_has_precomputed_indices(item.data) for item in self.search_database):
            self._build_index_in_memory()
            self._persist_final_index()
            return

        self.build_index_shards()

    def search(
        self,
        query_representation: RetrievalRepresentation,
        **kwargs: Any,
    ) -> SearchResult:
        """
        Run SISH slide retrieval for one query representation.

        Inputs:
            query_representation:
                Query patch-feature representation with shape ``(N_q, D)``.
            filter_same_patient:
                If ``True``, exclude reference slides sharing the query patient.

        Output:
            SearchResult:
                Ranked slide-level retrieval result. Query metadata includes
                ``predicted_category`` and ``top_k_labels``.
        """
        _ = kwargs
        query_item = self.prepare_query(query_representation)
        hits = self.rank(
            query_item=query_item,
            database_items=self.filter_database_by_exclusion_key(query_item=query_item),
        )

        return SearchResult(
            query_sample_id=query_item.item_id,
            hits=hits,
        )

    def rank(
        self,
        query_item: SearchDatabaseItem,
        database_items: list[SearchDatabaseItem],
        **kwargs: Any,
    ) -> list[SearchHit]:
        """
        Rank database slides for one query using the original SISH procedure.

        Inputs:
            query_item:
                Prepared query item containing patch features and optional
                precomputed SISH patch indices.
            database_items:
                Candidate database items after optional patient filtering.

        Output:
            list[SearchHit]:
                Slide-level top-``k`` hits ordered by ascending cleaned SISH
                distance.
        """
        _ = kwargs
        if not database_items:
            return []
        if self.vebtree is None:
            self.build_index()

        candidate_ids = {item.item_id for item in database_items}
        query_payload = self._ensure_query_payload_ready(query_item)
        patient_id = query_item.exclusion_key
        weights = self.compute_database_weights(
            patient_id=patient_id,
            allowed_slide_ids=candidate_ids,
        )

        slide_outputs = [
            self.query(
                index=int(index),
                dense_feat=bits,
                patient_id=patient_id,
                allowed_slide_ids=candidate_ids,
            )
            for index, bits in zip(
                query_payload["patch_indices"],
                query_payload["packed_bits"],
                strict=False,
            )
        ]

        cleaned = self.clean_single_result(
            slide_id=query_item.item_id,
            data={
                "results": slide_outputs,
                "label_query": None,
            },
            weights=weights,
        )

        hits: list[SearchHit] = []
        for rank, entry in enumerate(cleaned["top_k"][: self.k], start=1):
            item_id = str(entry["slide_id"])
            database_item = self._database_item_by_id.get(item_id)
            if database_item is None:
                continue
            hits.append(
                SearchHit(
                    sample_id=item_id,
                    score=float(entry["distance"]),
                    rank=rank,
                )
            )

        return hits

    def build_index_shards(
        self,
        resume: bool | None = None,
        shard_size: int | None = None,
    ) -> None:
        """
        Build the SISH index with resumable on-disk shards.

        Inputs:
            resume:
                Whether to resume from an existing shard manifest.
            shard_size:
                Number of slides to flush per shard.

        Output:
            None. Persists shard files plus final VEB/meta artifacts.
        """
        resume = self.resume_shards if resume is None else bool(resume)
        shard_size = self.shard_size if shard_size is None else int(shard_size)

        if self.shard_dir is None:
            raise ValueError("SISH shard_dir could not be resolved.")
        if self.index_veb_path is None or self.meta_database_path is None:
            raise ValueError("SISH final artifact paths could not be resolved.")

        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.index_veb_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_database_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = self.shard_dir / "manifest.json"

        if resume and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            processed_slides = set(manifest.get("processed_slides", []))
            shard_idx = int(manifest.get("next_shard_idx", 0))
            max_key_seen = int(manifest.get("max_key_seen", -1))
            logger.info(
                "[SISH] Resuming shard build with %d processed slides.",
                len(processed_slides),
            )
        else:
            processed_slides = set()
            shard_idx = 0
            max_key_seen = -1

        meta_batch: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        keys_batch: list[int] = []
        processed_count = len(processed_slides)

        def flush_shard() -> None:
            nonlocal shard_idx, meta_batch, keys_batch
            if not keys_batch and not meta_batch:
                return

            meta_path = self.shard_dir / f"meta_shard_{shard_idx:04d}.pkl"
            keys_path = self.shard_dir / f"keys_shard_{shard_idx:04d}.npy"
            with meta_path.open("wb") as handle:
                pickle.dump(dict(meta_batch), handle, protocol=pickle.HIGHEST_PROTOCOL)
            np.save(keys_path, np.asarray(keys_batch, dtype=np.int64))

            manifest_payload = {
                "processed_slides": sorted(processed_slides),
                "next_shard_idx": shard_idx + 1,
                "max_key_seen": max_key_seen,
            }
            manifest_path.write_text(
                json.dumps(manifest_payload, indent=2),
                encoding="utf-8",
            )

            meta_batch = defaultdict(list)
            keys_batch = []
            shard_idx += 1
            gc.collect()

        for item in self.search_database:
            if item.item_id in processed_slides:
                continue

            payload = self._ensure_database_payload_ready(item)
            for entry in self._iter_payload_entries(item, payload):
                key = int(entry["index"])
                meta_batch[key].append(entry["meta"])
                keys_batch.append(key)
                max_key_seen = max(max_key_seen, key)

            processed_slides.add(item.item_id)
            processed_count += 1
            gc.collect()
            if _torch_cuda_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            if processed_count % shard_size == 0:
                flush_shard()

        flush_shard()

        self.meta = defaultdict(list)
        self.keys = []
        for index in range(shard_idx):
            meta_path = self.shard_dir / f"meta_shard_{index:04d}.pkl"
            keys_path = self.shard_dir / f"keys_shard_{index:04d}.npy"
            if not meta_path.exists():
                continue
            with meta_path.open("rb") as handle:
                partial_meta = pickle.load(handle)
            for key, entries in partial_meta.items():
                self.meta[int(key)].extend(entries)
            self.keys.extend(np.load(keys_path).astype(np.int64).tolist())
            del partial_meta
            gc.collect()

        self.meta = dict(self.meta)
        self._build_veb_from_keys(max_key_seen=max_key_seen)
        self._persist_final_index()

    def leave_one_patient(self, patient_id: str) -> dict[int, list[dict[str, Any]]]:
        """
        Build a patient-filtered metadata view for legacy SISH workflows.

        Inputs:
            patient_id:
                Patient identifier to exclude.

        Output:
            dict[int, list[dict[str, Any]]]:
                Metadata keyed by VEB index without entries from ``patient_id``.
        """
        filtered: dict[int, list[dict[str, Any]]] = {}
        for key, entries in self.meta.items():
            filtered[key] = [
                entry for entry in entries if entry["patient_id"] != patient_id
            ]
        return filtered

    def search_single(
        self,
        query_index: int,
        dense_feat: bytes,
        patient_id: str | None,
        allowed_slide_ids: set[str] | None = None,
    ) -> list[tuple[Any, ...]]:
        """
        Run the bidirectional VEB-guided SISH patch search for one query patch.

        Inputs:
            query_index:
                Integer patch index.
            dense_feat:
                Packed binary feature bytes for the query patch.
            patient_id:
                Query patient identifier used for LOPO filtering.
            allowed_slide_ids:
                Optional whitelist of reference slide IDs.

        Output:
            list[tuple]:
                Raw literature-style SISH tuples before final postprocessing.
        """
        if self.vebtree is None:
            raise ValueError("SISH index has not been built.")

        seed_indices: list[int] = []
        seed_indices.extend(
            int(query_index - (offset * self.seed_interval_c * 1e11))
            for offset in range(self.seed_fanout_t)
        )
        seed_indices.extend(
            int(query_index + (offset * self.seed_interval_c * 1e11))
            for offset in range(self.seed_fanout_t)
        )

        results: list[tuple[Any, ...]] = []
        visited: dict[int, bool] = {}

        for seed_index in seed_indices:
            predecessor_prev = seed_index
            predecessor_count = 0
            while predecessor_count < self.pre_step:
                predecessor = self.vebtree.predecessor(predecessor_prev)
                if predecessor is None or predecessor in visited:
                    break

                candidates = self._filter_candidates(
                    candidates=self.meta.get(predecessor, []),
                    patient_id=patient_id,
                    allowed_slide_ids=allowed_slide_ids,
                )
                if not candidates:
                    predecessor_prev = predecessor
                    continue

                match_index, hamming_dist = self._best_hamming_match(
                    candidates=candidates,
                    dense_feat=dense_feat,
                )
                if hamming_dist <= self.hamming_thr:
                    entry = candidates[match_index]
                    visited[predecessor] = True
                    results.append(
                        (
                            query_index,
                            predecessor,
                            abs(predecessor - query_index),
                            hamming_dist,
                            entry["slide_name"],
                            entry["category"],
                            entry["patient_id"],
                            entry["x"],
                            entry["y"],
                        )
                    )

                predecessor_count += 1
                predecessor_prev = predecessor

            successor_prev = seed_index
            successor_count = 0
            while successor_count < self.succ_step:
                successor = self.vebtree.successor(successor_prev)
                if successor is None or successor in visited:
                    break

                candidates = self._filter_candidates(
                    candidates=self.meta.get(successor, []),
                    patient_id=patient_id,
                    allowed_slide_ids=allowed_slide_ids,
                )
                if not candidates:
                    successor_prev = successor
                    continue

                match_index, hamming_dist = self._best_hamming_match(
                    candidates=candidates,
                    dense_feat=dense_feat,
                )
                if hamming_dist <= self.hamming_thr:
                    entry = candidates[match_index]
                    visited[successor] = True
                    results.append(
                        (
                            query_index,
                            successor,
                            abs(successor - query_index),
                            hamming_dist,
                            entry["slide_name"],
                            entry["category"],
                            entry["patient_id"],
                            entry["x"],
                            entry["y"],
                        )
                    )

                successor_count += 1
                successor_prev = successor

        return results

    def postprocessing(self, raw_results: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
        """
        Sort raw SISH patch matches and convert them into dictionaries.

        Inputs:
            raw_results:
                Raw SISH tuples with fields
                ``(query, index, global_dist, hamming_dist, slide_name, category,
                patient_id, x, y)``.

        Output:
            list[dict[str, Any]]:
                Sorted patch-match dictionaries ordered by increasing
                ``hamming_dist``.
        """
        field_names = [
            "query",
            "index",
            "global_dist",
            "hamming_dist",
            "slide_name",
            "category",
            "patient_id",
            "x",
            "y",
        ]
        sorted_results = sorted(raw_results, key=lambda value: value[3])
        return [dict(zip(field_names, entry, strict=True)) for entry in sorted_results]

    def query(
        self,
        index: int,
        dense_feat: bytes,
        patient_id: str | None,
        allowed_slide_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run one SISH patch query and format its patch-level matches.

        Inputs:
            index:
                Query patch index.
            dense_feat:
                Packed binary query feature bytes.
            patient_id:
                Query patient identifier for LOPO filtering.
            allowed_slide_ids:
                Optional whitelist of reference slide IDs.

        Output:
            list[dict[str, Any]]:
                Sorted patch-level matches.
        """
        return self.postprocessing(
            self.search_single(
                query_index=index,
                dense_feat=dense_feat,
                patient_id=patient_id,
                allowed_slide_ids=allowed_slide_ids,
            )
        )

    def compute_database_weights(
        self,
        patient_id: str | None,
        allowed_slide_ids: set[str] | None = None,
    ) -> dict[str, float]:
        """
        Compute inverse-frequency SISH class weights over searchable entries.

        Inputs:
            patient_id:
                Query patient identifier to exclude.
            allowed_slide_ids:
                Optional whitelist of searchable slide IDs.

        Output:
            dict[str, float]:
                Label-to-weight mapping normalized to the SISH retrieval depth.
        """
        total_per_label: dict[str, int] = {}
        for patch_bag in self.meta.values():
            for entry in patch_bag:
                if patient_id is not None and entry["patient_id"] == patient_id:
                    continue
                if (
                    allowed_slide_ids is not None
                    and entry["slide_name"] not in allowed_slide_ids
                ):
                    continue
                label = str(entry["category"])
                total_per_label[label] = total_per_label.get(label, 0) + 1

        total_per_label = {
            label: count for label, count in total_per_label.items() if count > 0
        }
        inverse_sum = sum(1.0 / count for count in total_per_label.values())
        if inverse_sum == 0.0:
            return {label: 1.0 for label in total_per_label}

        normalization = self.k / inverse_sum
        return {
            label: normalization * (1.0 / count)
            for label, count in total_per_label.items()
        }

    def clean_single_result(
        self,
        slide_id: str,
        data: dict[str, Any],
        weights: dict[str, float],
    ) -> dict[str, Any]:
        """
        Aggregate one query slide's patch-level SISH outputs to slide level.

        Inputs:
            slide_id:
                Query slide identifier.
            data:
                Dictionary with:
                - ``results``: list of patch-query result bags
                - ``label_query``: ground-truth query category
            weights:
                Label weights used by ``Uncertainty_Cal``.

        Output:
            dict[str, Any]:
                Cleaned slide-level prediction containing ``top_k`` entries with
                keys ``slide_id``, ``label``, and ``distance``.
        """
        bags = list(data["results"])
        total_bags = sum(len(bag) for bag in bags)
        if total_bags == 0:
            return {
                "query_slide_id": slide_id,
                "query_label": data["label_query"],
                "predicted_label": None,
                "top_k": [],
            }

        bag_summary: list[tuple[int, float, list[float], int]] = []
        label_count_summary: dict[int, dict[str, float]] = {}

        for bag_index, bag in enumerate(bags):
            hamming_distances = sorted(entry["hamming_dist"] for entry in bag)
            entropy, label_count, _ = Uncertainty_Cal(bag, weights)
            if entropy is None or not hamming_distances:
                continue
            label_count_summary[bag_index] = label_count
            bag_summary.append(
                (
                    bag_index,
                    float(entropy),
                    hamming_distances,
                    len(hamming_distances),
                )
            )

        if not bag_summary:
            return {
                "query_slide_id": slide_id,
                "query_label": data["label_query"],
                "predicted_label": None,
                "top_k": [],
            }

        bag_summary, hamming_thr = Clean(
            [summary[3] for summary in bag_summary],
            bag_summary,
        )
        if not bag_summary:
            return {
                "query_slide_id": slide_id,
                "query_label": data["label_query"],
                "predicted_label": None,
                "top_k": [],
            }
        removed = Filtered_BY_Prediction(bag_summary, label_count_summary)

        retrieval_final: list[tuple[str, float, str | None, float, int]] = []
        visited_slides: set[str] = set()
        for bag_index, uncertainty, _, _ in bag_summary:
            for entry in bags[bag_index]:
                slide_name = str(entry["slide_name"])
                hamming_dist = float(entry["hamming_dist"])
                label = entry.get("diagnosis", entry.get("category"))
                if uncertainty == 0.0 or (
                    hamming_dist <= hamming_thr and slide_name not in visited_slides
                ):
                    retrieval_final.append(
                        (
                            slide_name,
                            hamming_dist,
                            label,
                            float(uncertainty),
                            bag_index,
                        )
                    )
                    visited_slides.add(slide_name)

        retrieval_final = [
            entry
            for entry in sorted(retrieval_final, key=lambda value: (value[3], value[1]))
            if entry[-1] not in removed
        ]
        top_k_info = [
            {
                "slide_id": slide_name,
                "label": label,
                "distance": float(hamming_dist),
            }
            for slide_name, hamming_dist, label, _uncertainty, _bag_index in retrieval_final[
                : self.k
            ]
        ]
        predicted_label = (
            Counter(entry["label"] for entry in top_k_info).most_common(1)[0][0]
            if top_k_info
            else None
        )

        return {
            "query_slide_id": slide_id,
            "query_label": data.get("label_query"),
            "predicted_label": predicted_label,
            "top_k": top_k_info,
        }

    def _build_item_payload(
        self,
        *,
        item_id: str,
        features: Any,
        additional_data: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the normalized per-item payload used throughout SISH."""
        feature_matrix = self._as_feature_matrix(features=features, item_id=item_id)
        extras = dict(additional_data or {})

        packed_bits = extras.get("sish_packed_bits")
        if packed_bits is None:
            packed_bits = [self._pack_feature_bits(row) for row in feature_matrix]
        else:
            packed_bits = [self._normalize_packed_bits(bits) for bits in packed_bits]

        patch_indices = extras.get("sish_patch_indices")
        if patch_indices is not None:
            patch_indices = np.asarray(patch_indices, dtype=np.int64)
            if patch_indices.ndim != 1:
                raise ValueError(
                    "SISH patch indices must have shape (N,). "
                    f"Got {patch_indices.shape} for '{item_id}'."
                )
            if patch_indices.shape[0] != feature_matrix.shape[0]:
                raise ValueError(
                    "SISH patch indices must align with feature rows. "
                    f"Got indices={patch_indices.shape[0]} and features={feature_matrix.shape[0]} "
                    f"for '{item_id}'."
                )

        coords = extras.get("selected_coords")
        if coords is None:
            coords = extras.get("coords")
        if coords is None:
            coords_array = np.zeros((feature_matrix.shape[0], 2), dtype=np.int64)
        else:
            coords_array = np.asarray(coords, dtype=np.int64)
            if coords_array.ndim != 2 or coords_array.shape[1] != 2:
                raise ValueError(
                    "SISH coordinates must have shape (N, 2). "
                    f"Got {coords_array.shape} for '{item_id}'."
                )
            if coords_array.shape[0] != feature_matrix.shape[0]:
                raise ValueError(
                    "SISH coordinates must align with feature rows. "
                    f"Got coords={coords_array.shape[0]} and features={feature_matrix.shape[0]} "
                    f"for '{item_id}'."
                )

        return {
            "features": feature_matrix,
            "packed_bits": packed_bits,
            "patch_indices": patch_indices,
            "coords": coords_array,
            "mosaic_pkl_path": self._extract_mosaic_path(extras=extras),
        }

    def _build_index_in_memory(self) -> None:
        """Build the SISH meta/key structures directly from precomputed payloads."""
        self.meta = defaultdict(list)
        self.keys = []
        max_key_seen = -1

        for item in self.search_database:
            payload = self._ensure_database_payload_ready(item)
            for entry in self._iter_payload_entries(item, payload):
                key = int(entry["index"])
                self.meta[key].append(entry["meta"])
                self.keys.append(key)
                max_key_seen = max(max_key_seen, key)

        self.meta = dict(self.meta)
        self._build_veb_from_keys(max_key_seen=max_key_seen)

    def _build_veb_from_keys(self, *, max_key_seen: int) -> None:
        """Instantiate the VEB tree from the collected keys."""
        if max_key_seen < 0 or not self.keys:
            raise ValueError("Cannot build SISH VEB tree without at least one key.")

        self.vebtree = VEB(max_key_seen)
        for key in self.keys:
            self.vebtree.insert(int(key))

    def _persist_final_index(self) -> None:
        """Persist the final VEB and metadata objects to disk."""
        if self.index_veb_path is None or self.meta_database_path is None:
            raise ValueError("SISH final artifact paths could not be resolved.")
        self.index_veb_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_veb_path.open("wb") as handle:
            pickle.dump(self.vebtree, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with self.meta_database_path.open("wb") as handle:
            pickle.dump(dict(self.meta), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _ensure_database_payload_ready(self, item: SearchDatabaseItem) -> dict[str, Any]:
        """Populate missing database payload fields lazily before indexing."""
        payload = dict(item.data)
        if payload.get("patch_indices") is None:
            mosaic_pkl_path = payload.get("mosaic_pkl_path")
            if not mosaic_pkl_path:
                raise ValueError(
                    "SISH requires either precomputed 'sish_patch_indices' or a "
                    f"'mosaic_pkl_path' for database item '{item.item_id}'."
                )
            payload["patch_indices"] = self._compute_patch_indices_from_mosaic(
                mosaic_pkl_path
            )
        item.data = payload
        return payload

    def _ensure_query_payload_ready(self, item: SearchDatabaseItem) -> dict[str, Any]:
        """Populate missing query payload fields lazily before ranking."""
        payload = dict(item.data)
        if payload.get("patch_indices") is None:
            mosaic_pkl_path = payload.get("mosaic_pkl_path")
            if not mosaic_pkl_path:
                raise ValueError(
                    "SISH query preparation requires precomputed 'sish_patch_indices' "
                    f"or a 'mosaic_pkl_path' for query item '{item.item_id}'."
                )
            payload["patch_indices"] = self._compute_patch_indices_from_mosaic(
                mosaic_pkl_path
            )
        item.data = payload
        return payload

    def _iter_payload_entries(
        self,
        item: SearchDatabaseItem,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Convert one prepared SISH payload into meta/key entries."""
        patch_indices = np.asarray(payload["patch_indices"], dtype=np.int64)
        coords = np.asarray(payload["coords"], dtype=np.int64)
        packed_bits = list(payload["packed_bits"])

        entries: list[dict[str, Any]] = []
        for row_index, key in enumerate(patch_indices):
            x_coord, y_coord = coords[row_index].tolist()
            entries.append(
                {
                    "index": int(key),
                    "meta": {
                        "slide_name": item.item_id,
                        "bits": packed_bits[row_index],
                        "patient_id": item.exclusion_key,
                        "category": None,
                        "x": int(x_coord),
                        "y": int(y_coord),
                    },
                }
            )
        return entries

    def _compute_patch_indices_from_mosaic(
        self,
        mosaic_pkl_path: str | os.PathLike[str],
    ) -> np.ndarray:
        """Compute SISH patch indices from a mosaic pickle using the VQ-VAE encoder."""
        self._ensure_vqvae_ready()
        from pathbench.slide_retrieval.search_strategies.strategies.sish.sish_index import (
            compute_latent_features,
            slide_to_index,
        )

        with torch.no_grad():
            latents = compute_latent_features(
                str(mosaic_pkl_path),
                transform=self.transform_vqvqe,
                vqvae=self.vqvae,
                device=self.device,
                batch_size=8,
                num_workers=int(
                    self._get_config_value(
                        [
                            ("experiment", "num_workers"),
                            ("num_workers",),
                        ],
                        default=0,
                    )
                ),
            )

        patch_indices = slide_to_index(
            latents,
            self.codebook_semantic,
            pool_layers=self.pool_layers,
            pool=self.pool,
        )
        del latents
        gc.collect()
        if _torch_cuda_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        log_mem(f"prepared {mosaic_pkl_path}")
        return np.asarray(patch_indices, dtype=np.int64)

    def _ensure_vqvae_ready(self) -> None:
        """Lazy-load the SISH VQ-VAE encoder and semantic codebook."""
        if torch is None:
            raise ImportError(
                "SISH VQ-VAE-backed indexing requires PyTorch to be installed."
            )
        if (
            self.vqvae is not None
            and self.codebook_semantic is not None
            and self.transform_vqvqe is not None
        ):
            return
        if self._codebook_path is None or self._checkpoint_path is None:
            raise ValueError(
                "SISH VQ-VAE assets are missing from config. Expected "
                "'codebook_semantic' and 'vqvae_checkpoint'."
            )

        from torchvision import transforms

        from pathbench.slide_retrieval.search_strategies.strategies.sish.sish_vqvae import (
            LargeVectorQuantizedVAE_Encode,
        )

        self.codebook_semantic = torch.load(self._codebook_path, map_location="cpu")
        self.vqvae = LargeVectorQuantizedVAE_Encode(code_dim=256, code_size=128)
        checkpoint = torch.load(self._checkpoint_path, map_location="cpu")["model"]
        encoder_weights = OrderedDict(
            (
                key[len("module."):],
                value,
            )
            for key, value in checkpoint.items()
            if key.startswith("module.encoder.") or key.startswith("module.codebook.")
        )
        self.vqvae.load_state_dict(encoder_weights, strict=False)
        self.vqvae.to(self.device).eval()
        self.transform_vqvqe = transforms.Lambda(scale_to_minus1_to_1)

    def _payload_has_precomputed_indices(self, payload: Any) -> bool:
        """Return whether a payload already carries precomputed SISH indices."""
        return isinstance(payload, Mapping) and payload.get("patch_indices") is not None

    def _best_hamming_match(
        self,
        *,
        candidates: list[dict[str, Any]],
        dense_feat: bytes,
    ) -> tuple[int, int]:
        """Return the best candidate index and its Hamming distance."""
        if len(candidates) == 1:
            return 0, hamming_bytes(candidates[0]["bits"], dense_feat)

        distances = [hamming_bytes(entry["bits"], dense_feat) for entry in candidates]
        best_index = int(np.argmin(distances))
        return best_index, int(distances[best_index])

    def _filter_candidates(
        self,
        *,
        candidates: list[dict[str, Any]],
        patient_id: str | None,
        allowed_slide_ids: set[str] | None,
    ) -> list[dict[str, Any]]:
        """Apply SISH same-patient and candidate-set filtering."""
        filtered = []
        for entry in candidates:
            if patient_id is not None and entry["patient_id"] == patient_id:
                continue
            if allowed_slide_ids is not None and entry["slide_name"] not in allowed_slide_ids:
                continue
            filtered.append(entry)
        return filtered

    def _as_feature_matrix(self, *, features: Any, item_id: str) -> np.ndarray:
        """Normalize one SISH representation to a 2D float feature matrix."""
        feature_matrix = np.asarray(features, dtype=np.float32)
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix[None, :]
        if feature_matrix.ndim != 2:
            raise ValueError(
                "SISH expects retrieval items with shape (N, D). "
                f"Got {feature_matrix.shape} for item '{item_id}'."
            )
        return feature_matrix

    def _pack_feature_bits(self, feature_row: np.ndarray) -> bytes:
        """Pack one dense feature row into the SISH binary byte representation."""
        bits = (np.asarray(feature_row, dtype=np.float32) > 0).astype(np.uint8, copy=False)
        return np.packbits(bits).tobytes()

    def _normalize_packed_bits(self, bits: Any) -> bytes:
        """Normalize precomputed packed bits to raw bytes."""
        if isinstance(bits, bytes):
            return bits
        if isinstance(bits, bytearray):
            return bytes(bits)
        if isinstance(bits, np.ndarray):
            return np.asarray(bits, dtype=np.uint8).tobytes()
        raise TypeError(
            "SISH packed bits must be bytes-like or uint8 arrays. "
            f"Got {type(bits).__name__}."
        )

    def _extract_mosaic_path(
        self,
        *,
        extras: Mapping[str, Any],
    ) -> str | None:
        """Resolve an optional mosaic pickle path from representation payload data."""
        for key in ("mosaic_pkl_path", "mosaic_path", "slide_representation_path"):
            value = extras.get(key)
            if value:
                return str(value)
        return None

    def _get_config_value(
        self,
        candidate_paths: list[tuple[str, ...]],
        *,
        default: Any,
    ) -> Any:
        """Resolve the first non-``None`` value from a list of config key paths."""
        for path in candidate_paths:
            current = self.config
            for key in path:
                if current is None:
                    break
                if isinstance(current, Mapping):
                    current = current.get(key)
                else:
                    current = getattr(current, key, None)
            if current is not None:
                return current
        return default

    def _resolve_path(
        self,
        candidate_paths: list[tuple[str, ...]],
        *,
        default: str | os.PathLike[str] | None,
    ) -> Path | None:
        """Resolve an optional filesystem path from config."""
        value = self._get_config_value(candidate_paths, default=default)
        if value is None:
            return None
        return Path(value)

    def _maybe_enrich_representation(
        self,
        representation: RetrievalRepresentation,
    ) -> RetrievalRepresentation:
        """Compute missing SISH per-patch indices from the representation payload."""
        if "sish_patch_indices" in representation.additional_data:
            if "sish_packed_bits" not in representation.additional_data:
                features = self._as_feature_matrix(
                    features=representation.data,
                    item_id=representation.sample_id,
                )
                representation.additional_data["sish_packed_bits"] = np.packbits(
                    (features > 0).astype(np.uint8, copy=False),
                    axis=1,
                )
            return representation

        bag_id = representation.additional_data.get("bag_id")
        if bag_id is None:
            raise ValueError(
                "SISH requires 'sish_patch_indices' or a search-reconstructable "
                "'bag_id' in representation.additional_data."
            )

        if self._precompute is None:
            from pathbench.slide_retrieval.sish_precompute import SISHPrecompute

            self._precompute = SISHPrecompute(config=self.config)

        return self._precompute.enrich_representation(
            representation=representation,
            sample=self._build_sample_like(representation),
            bag_id=str(bag_id),
        )

    def _build_sample_like(
        self,
        representation: RetrievalRepresentation,
    ) -> Any:
        """Build a lightweight sample-like object for SISH patch reconstruction."""
        extras = dict(representation.additional_data)
        slide_ids = [str(slide_id) for slide_id in extras.get("source_slide_ids", [])]
        if not slide_ids:
            raise ValueError(
                "SISH requires representation.additional_data['source_slide_ids'] "
                "to reconstruct patches."
            )

        dataset_name = extras.get("dataset_name")
        if not dataset_name:
            raise ValueError(
                "SISH requires representation.additional_data['dataset_name'] "
                "to resolve slide/artifact paths."
            )

        dataset_cfg = self._find_dataset_config(dataset_name=str(dataset_name))
        artifacts_dir = Path(self._get_value(dataset_cfg, "artifacts_dir"))
        artifact_paths = [artifacts_dir / f"{slide_id}.h5" for slide_id in slide_ids]

        class _SampleLike:
            def __init__(self, sample_id: str, slide_ids: list[str], artifact_paths: list[Path], metadata_dict: dict[str, Any]) -> None:
                self.sample_id = sample_id
                self.slide_ids = slide_ids
                self.artifact_paths = artifact_paths
                self.metadata = metadata_dict

        return _SampleLike(
            sample_id=str(representation.sample_id),
            slide_ids=slide_ids,
            artifact_paths=artifact_paths,
            metadata={"dataset": str(dataset_name)},
        )

    def _find_dataset_config(self, *, dataset_name: str) -> Any:
        """Resolve one dataset config entry by name."""
        datasets = self._get_value(self.config, "datasets", default=[])
        for dataset_cfg in list(datasets):
            if str(self._get_value(dataset_cfg, "name")) == dataset_name:
                return dataset_cfg
        raise ValueError(f"SISH could not resolve dataset '{dataset_name}' from config.")

    def _get_value(self, source: Any, key: str, default: Any = ...) -> Any:
        """Read one config value from either mapping-like or attribute-like objects."""
        if isinstance(source, Mapping):
            if key in source:
                return source[key]
        elif hasattr(source, key):
            return getattr(source, key)

        if default is ...:
            raise ValueError(f"Required config value '{key}' is missing.")
        return default
