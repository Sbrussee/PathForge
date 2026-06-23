# PathBench-MIL Retrieval Representation H5 Structure

This document defines the current H5 structure used for slide-retrieval representation artifacts.

It is derived from:
- `src/pathbench/core/io/slide_retrieval/layout.py`
- `src/pathbench/core/io/slide_retrieval/retrieval_representations.py`
- `src/pathbench/core/io/slide_retrieval/descriptors.py`

---

## 1. Artifact File Location

One retrieval artifact file lives at:

`artifacts_dir/slide_retrieval/{aggregation_level}/{sample_id}.h5`

The canonical builder is:
- `build_retrieval_representation_artifact_path(artifacts_dir, aggregation_level, sample_id)`
  in `src/pathbench/slide_retrieval/representation_strategies/storage.py`

---

## 2. Top-Level Layout

Data is partitioned by tiling identifier under `bags/{tile_id}`:

```text
bags/{tile_id}/
  descriptors/{descriptor_name}

  retrieval_representations/
    {representation_id}/
      (for case/patient aggregation)
        {entry_id}/
          representation_type
          metadata
          params
          embedding
          additional_data/
            {name}
      (for slide aggregation)
        representation_type
        metadata
        params
        embedding
        additional_data/
          {name}
```

Notes:
- `tile_id` is the retrieval-layout field name and should receive the canonical `tiling_id` value.
- `descriptors/` and `retrieval_representations/` are independent subtrees under the same tile group.
- `entry_id` is only present for multi-slide aggregations (`case`/`patient`).
  For `slide` aggregation, the stored entry root is the `{representation_id}` group itself.

---

## 3. Identifier Definitions

### 3.1 `tile_id` (input to retrieval I/O/layout)
- Semantic meaning: canonical tiling key for the bag group.
- Typical value: `256px_0.5mpp`.
- Source: pass the value produced by `build_tiling_id(combo_cfg)`.

### 3.2 `representation_id`
- Built with:
  `build_retrieval_representation_id(feature_extraction, retrieval_representation, params)`
- Format:
  `{feature_extraction}_{retrieval_representation}_{params_hash16}`

### 3.3 `entry_id`
- Built with:
  `build_retrieval_representation_entry_id(slide_ids, aggregation_level=...)`
- Format:
  `members_{sha1_16}` for `case`/`patient`
- For `slide` aggregation this function returns `None`, so no `{entry_id}` group is used.
- The hash is computed from sorted member slide IDs.
- `entry_id` identifies the member set, but does not store the member slide IDs in reversible form.

### 3.4 Aggregation membership source
- The member slides used to build an aggregated retrieval entry come from `sample.slide_ids`.
- For `slide` aggregation this is exactly one slide ID.
- For `case` and `patient` aggregation this is the full grouped slide list for that sample.
- Grouped samples are built from the bag dataset annotations and the grouped slide list is
  sorted by `SLIDE_ID_COL` before artifact addressing.

---

## 4. Descriptor Cache Section

Path:
- `bags/{tile_id}/descriptors/{descriptor_name}`

Type/shape:
- 2D numeric matrix: `(N, D)`
- Stored/read as `float32`

Validation behavior:
- Must be 2D.
- Optional existence checks can enforce expected rows and/or dimensionality.

Intended semantics:
- Retrieval-side per-patch descriptor cache (for example `mean_rgb`).
- `N` should match the number of patches for the same `tile_id`.

---

## 5. Retrieval Representation Entry Section

Entry root:
- `bags/{tile_id}/retrieval_representations/{representation_id}/{entry_id}` for `case`/`patient`
- `bags/{tile_id}/retrieval_representations/{representation_id}` for `slide`

### 5.1 Required-for-existence fields
`retrieval_representation_entry_exists(...)` currently requires:
- `embedding` to exist
- `metadata` to exist

### 5.2 `embedding`
Path:
- `{entry_root}/embedding`

Type/shape:
- Stored as an array dataset (strategy-dependent shape/dtype).
- Commonly `float32`, but not hard-enforced by this layer.

### 5.3 `metadata`
Path:
- `{entry_root}/metadata`

Type:
- Scalar UTF-8 JSON string, decoded as `dict[str, Any]`.

Typical content:
- Identity/provenance fields written via `RetrievalItemIdentity(...).to_dict()`
  in `save_slide_retrieval_representation(...)`.

Current behavior:
- `metadata` currently stores only the minimal persisted retrieval identity
  (for example `sample_id`).
- The explicit member slide list is not stored in `metadata`.

### 5.4 `params`
Path:
- `{entry_root}/params`

Type:
- Scalar UTF-8 JSON string, decoded as `dict[str, Any]`.

Notes:
- Always written by `write_retrieval_representation_entry(...)`.
- If absent in legacy files, reader falls back to `{}`.

### 5.5 `representation_type`
Path:
- `{entry_root}/representation_type`

Type:
- Scalar UTF-8 string.

Notes:
- Supported by low-level I/O helpers.
- Not required by current entry-existence checks.

### 5.6 `additional_data/{name}`
Path:
- `{entry_root}/additional_data/{name}`

Type/shape:
- Arbitrary array datasets, strategy-specific.

Notes:
- Optional.
- On full entry rewrite, existing `additional_data/` is removed and re-created from provided values.

Current retrieval-task usage:
- `additional_data/source_slide_ids` stores the explicit ordered list of slide IDs that
  were used to build the retrieval representation for this sample.
- `additional_data/dataset_name` stores the source dataset name.
- Because `entry_id` is hash-based, `additional_data/source_slide_ids` is the canonical
  stored place to recover which slide IDs contributed to an aggregated entry.

---

## 6. Read/Write Contract (Current)

### 6.1 Write contract
`write_retrieval_representation_entry(...)` writes:
- `metadata`
- `params`
- `embedding`
- optional `additional_data/*`

### 6.2 Read contract
`read_retrieval_representation_entry(...)` returns:
- `metadata: dict[str, Any]`
- `params: dict[str, Any]` (or `{}` if missing)
- `embedding: np.ndarray`
- `additional_data: dict[str, np.ndarray]`

### 6.3 Deletion helpers
Available granular deletion:
- one entry
- all entries for one `representation_id` under a `tile_id`
- all retrieval representations under a `tile_id`

---

## 7. Compatibility Notes

- `tile_id` in retrieval I/O corresponds to the same tiling concept as `tiling_id` elsewhere.
- `bag_id` in feature/benchmark grouping may represent a broader combo identity
  (`tiling_id__feature_name`), which is distinct from retrieval `tile_id`.
- New readers should treat `representation_type` as optional unless/until promoted to required.
- For aggregated (`case`/`patient`) entries, consumers should not attempt to infer member
  slide IDs from `entry_id`; use `additional_data/source_slide_ids` when available.
