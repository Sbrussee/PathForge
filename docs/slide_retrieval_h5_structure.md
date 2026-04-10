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
      {entry_id}/
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
  `{feature_extraction}__{retrieval_representation}__{params_hash16}`

### 3.3 `entry_id`
- Built with:
  `build_retrieval_representation_entry_id(slide_ids)`
- Format:
  `members_{sha1_16}`
- The hash is computed from sorted member slide IDs.

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
- `bags/{tile_id}/retrieval_representations/{representation_id}/{entry_id}`

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
