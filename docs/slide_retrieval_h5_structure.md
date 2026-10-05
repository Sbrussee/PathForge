# PathForge Retrieval Representation H5 Structure

This document defines the current H5 structure used for slide-retrieval representation artifacts.

It is derived from:
- `src/pathforge/core/io/slide_retrieval/layout.py`
- `src/pathforge/core/io/slide_retrieval/retrieval_representations.py`
- `src/pathforge/core/io/slide_retrieval/descriptors.py`

---

## 1. Artifact File Location

One retrieval artifact file lives per physical slide at:

`artifacts_dir/slide_retrieval/{slide_id}.h5`

The canonical builder is:
- `build_retrieval_representation_artifact_path(artifacts_dir, slide_id)`
  in `src/pathforge/slide_retrieval/representation_strategies/storage.py`

Case and patient retrieval items are assembled in memory from their member
slide files; they never produce a case- or patient-level retrieval H5 file.

---

## 2. Top-Level Layout

Data is partitioned by tiling identifier under `bags/{tile_id}`:

```text
bags/{tile_id}/
  descriptors/{descriptor_name}

  retrieval_representations/
    {representation_id}/
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
- Slide retrieval caches store data directly below `{representation_id}`; no
  `entry_id` is written.

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

### 3.3 Aggregation membership source
- Each cache file is addressed by exactly one physical `slide_id`.
- Case and patient items use `sample.slide_ids` only while combining already
  cached slide representations in memory.

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
- SISH VQ-VAE descriptors are row-aligned to the *source* `tile_id`, but each
  row is produced from a canonical 1024px at 0.5mpp crop centred on that source
  tile. The descriptor dataset records this crop contract as HDF5 attributes.

---

## 5. Retrieval Representation Entry Section

Entry root:
- `bags/{tile_id}/retrieval_representations/{representation_id}`

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
- `additional_data/source_slide_ids` contains the one physical slide ID used to
  build the cached representation.
- `additional_data/dataset_name` stores the source dataset name.

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
- Case and patient retrieval entries are not persisted; consumers combine the
  relevant slide-level cache entries at runtime.
