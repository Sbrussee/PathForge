# PathForge-MIL HDF5 Structure — Canonical Overview

**File:**  

{slide_stem}.h5


This document defines the **authoritative structure** of PathForge-MIL slide-level HDF5 files, including explicit rules for **where** each artifact is stored and **how** it is formatted.

---

## Global Rules (Apply Everywhere)

- All spatial coordinates are in **level-0 pixel space**, unless explicitly stated.
- Geometry is **vector-first**; raster representations are optional caches.
- Patch-level data is **row-aligned** with `coords/`.
- Instance- and object-level data is **object-aligned**, never row-aligned with tiles.
- Graphs are **explicit objects**; no implicit adjacency is assumed.
- Every spatial dataset must declare its coordinate system via attributes.

---

## 0. Global Metadata (Required)

**Path:**  

meta/


**Required attributes**
- `schema_name = "pathforge_mil_extended"`
- `schema_version = "1.0"`
- `coord_space = "level0_pixel"`
- `num_levels : int32`
- `level_dimensions : (num_levels, 2) int32`
- `level_downsamples : (num_levels,) float32`

**Optional attributes**
- `mpp_x`, `mpp_y : float32`
- `scanner : string`
- `dataset : string`
- `annotation_source : string`

---

## 1. Tile Coordinates (Patch Sampling)

**Purpose:** spatial definition of MIL bags (tile sampling grid), including the tiling metadata required to interpret the coordinates and an optional overview visualization of the tiled slide.

**Bag identity:**  
`bag_id = "{tile_px}px_{tile_mpp:g}mpp"`  
(e.g. `256px_0.5mpp`)

### 1.1 Coordinates

**Path (per bag):**  
`bags/{bag_id}/coords`

**Shape:** `(N, 5)`  
**Dtype:** `int32`

**Columns (fixed order):**
1. `x` – level-0 pixel x (top-left)
2. `y` – level-0 pixel y (top-left)
3. `read_w` – patch width in pixels **at `read_level`**
4. `read_h` – patch height in pixels **at `read_level`**
5. `read_level` – pyramid level to read from

**Rules**
- Row order is **authoritative**
- All tile-aligned matrices must match this row order
- For a given `bag_id`, `read_w`, `read_h`, and `read_level` are expected to be **constant** across rows (single tiling setup per bag)
- `x,y` are always in **level-0 coordinate space** (`coord_space = "level0"`)

### 1.2 Tiling specification

**Purpose:** backend-agnostic tiling intent and guardrails for cache reuse / reconstruction.

**Path (per bag):**  
`bags/{bag_id}/tiling_spec`

**Encoding:** scalar UTF-8 JSON string

**Required keys**
- `tile_px` (int) – **output** tile size (pixels)
- `tile_mpp` (float) – target microns-per-pixel
- `stride_px` (int) – **output** stride in pixels (no-overlap: `stride_px == tile_px`)
- `coord_space` (str) – must be `"level0"`

**Optional keys**
- `backend` (str) – backend that generated the coords (used for warnings/debug)

**Rules**
- `tiling_spec` is **backend-agnostic** (no lazyslide/wsidata internals stored here)
- `coords` provide the **read window** (`read_w/read_h/read_level`); `tiling_spec` provides the **output intent** (`tile_px/tile_mpp/stride_px`)
- Cache reuse checks compare a **subset** of keys (typically `tile_px`, `tile_mpp`, `stride_px`, `coord_space`)

### 1.3 Tiles overview (optional visualization)

**Purpose:** compact visualization of the tiling result for reporting/inspection (thumbnail with tile grid overlay).

**Path (per bag):**  
`bags/{bag_id}/tiles_overview`

**Encoding:** compressed image bytes stored as a 1D `uint8` array (currently JPEG bytes)

**Shape:** `(M,)`  
**Dtype:** `uint8`

**Content**
- RGB thumbnail of the slide, encoded as JPEG
- Tile grid overlay derived from `coords`, `tiling_spec`, and the slide base MPP
- No embedded title text; PDF/report renderers add any surrounding text consistently

**Rules**
- Optional: only written when `experiment.report = true`
- Stored under the same `bag_id` as `coords` and `tiling_spec`
- Written when missing; existing `tiles_overview` datasets are reused
- `tiles_overview` is tied to the bag (`bag_id`) and therefore corresponds to the same tiling setup as `coords` and `tiling_spec`

---

## 2. Thumbnail (Optional Slide-Level Visualization Cache)

**Purpose:** reusable full-slide thumbnail for downstream visualization without reopening the original WSI.

**Path:**  

`thumbnail/`

### 2.1 Thumbnail Image

**Path:**  
`thumbnail/image`

**Encoding:** compressed image bytes stored as a 1D `uint8` array (currently JPEG bytes)

**Shape:** `(M,)`  
**Dtype:** `uint8`

**Rules**
- Optional: only written when `experiment.thumbnail = true`
- Represents the **full slide**, not a tissue crop
- Intended as a slide-level cache reusable across tasks and visualizations
- Current writer stores JPEG bytes with a maximum long side of 1200 pixels by default

### 2.2 Thumbnail Spec

**Path:**  
`thumbnail/spec`

**Encoding:** scalar UTF-8 JSON string

**Required keys**
- `image_format` (str) – currently `"jpeg"`
- `coord_space` (str) – must be `"level0"`
- `thumbnail_level` (int) – pyramid level requested when creating the thumbnail
- `downscale_x` (float) – level-0 x pixels per thumbnail x pixel
- `downscale_y` (float) – level-0 y pixels per thumbnail y pixel

**Rules**
- The thumbnail is always interpreted in full-slide level-0 coordinate space
- Patch coordinates can be projected onto the thumbnail using `downscale_x` and `downscale_y`
- Width and height are intentionally not duplicated in the spec; consumers should decode the image when needed

---

## 3. Feature Matrices (Tile-Level)

**Purpose:** MIL inputs (tile embeddings)

**Path (per bag, per extractor):**  
`bags/{bag_id}/features/{extractor_name}`

**Shape:** `(N, D)`  
**Dtype:** `float32`

**Rules**
- One dataset per extractor
- Must be **row-aligned with** `bags/{bag_id}/coords`
- No spatial information stored here

---

## 4. Tissue Masks (Vector-Based)

**Purpose:** define valid tissue regions (used for tiling / filtering)

**Path (per slide):**  
`annotations/tissue`

**Encoding:** scalar UTF-8 JSON string

**Coordinate space:** level-0 pixels (`coord_space = "level0"` by convention)

**Rules**
- Polygons may contain holes
- Polygons may overlap (union semantics implied)
- Raster tissue masks are **derived artifacts only**

---

## 5. Semantic Annotations / Segmentations (Vector-Based)

**Purpose:** region-level semantic labeling

**Path:**  

annotations/semantic/


**Datasets**
- `points : (M, 2) float32`
- `ring_offsets : (R+1,) int64`
- `poly_offsets : (P+1,) int64`
- `poly_label_id : (P,) int32`  
  → index into `labels/semantic/classes`

**Optional datasets**
- `poly_confidence : (P,) float32`
- `poly_attrs_json : (P,) string`

**Attributes**
- `geometry_type = "polygon_rings"`
- `coord_space = "level0_pixel"`
- `fill_rule = "even-odd"`
- `overlap_rule = "allow" | "priority"`

**Rules**
- Vector annotations are authoritative
- Raster semantic masks must declare `derived_from`

---

## 6. Instance Annotations / Segmentations  
*(cells, nuclei, points, small objects)*

**Purpose:** object-level entities

**Path:**  

instances/{instance_type}/


Example:

instances/nuclei/


### 6.1 Instance Table (Required)

**Datasets**
- `id : (I,) uint64`
- `centroid_xy : (I, 2) float32`
- `bbox_xyxy : (I, 4) int32`
- `class_id : (I,) int32` (optional)

**Rules**
- One row = one instance
- IDs are unique per slide
- Geometry may be implicit (point) or explicit (below)

---

### 6.2 Optional Instance Geometry

Choose **one** representation:

#### Polygon Geometry
- `poly_points`
- `poly_ring_offsets`
- `poly_offsets`

#### RLE Geometry
- `rle_counts`
- `rle_offsets`

**Attributes**
- `geometry = "point" | "polygon" | "rle"`
- `coord_space = "level0_pixel"`

---

## 7. Feature Matrices for Instances / Semantic Objects

### 7.1 Instance-Level Features (Cells, Nuclei)

**Path:**  

instance_features/{instance_type}/{extractor_name}


**Shape:** `(I, D)`  
**Dtype:** `float32`

**Rules**
- Row-aligned with `instances/{instance_type}/id`
- No spatial data stored here

---

### 7.2 Semantic Object Features (Regions)

**Path:**  

semantic_features/{extractor_name}


**Shape:** `(P, D)`  
**Dtype:** `float32`

**Rules**
- Row-aligned with semantic polygons
- One row per semantic object

---

## 7. Adjacency Matrices / Graphs

Graphs are explicit, first-class objects.

---

### 7.1 Tile Graphs (Patch Adjacency)

**Path:**  

graphs/tiles/{graph_name}/


**Datasets**
- `edge_index : (2, E) int64`
- `edge_weight : (E,) float32` (optional)

**Rules**
- Node indices refer to rows in `coords/`
- Graph construction method must be stored as attributes

---

### 7.2 Instance Graphs (Cell / Object Graphs)

**Path:**  

graphs/instances/{instance_type}/{graph_name}/


**Datasets**
- `edge_index : (2, E) int64`
- `edge_weight : (E,) float32` (optional)

**Rules**
- Node indices refer to `instances/{instance_type}/id`
- Graph semantics declared via attributes

---

### 7.3 Semantic Graphs (Region Adjacency)

**Path:**  

graphs/semantic/{graph_name}/


**Datasets**
- `edge_index : (2, E) int64`
- `edge_weight : (E,) float32`

**Rules**
- Nodes correspond to semantic polygons
- Adjacency often defined by touching or overlap

---

## 8. Label Tables (Shared)

**Paths**

labels/tissue/classes
labels/semantic/classes
labels/instances/{instance_type}/classes


**Rules**
- Integer IDs always index into label tables
- Raw strings must never be embedded in geometry tables

---

## 9. Explicit Do / Don’t Rules

### DO
- Use **vector geometry** as the source of truth
- Keep tile-aligned, instance-aligned, and object-aligned data separate
- Store graphs explicitly
- Declare coordinate systems via attributes

### DON’T
- Store full-resolution raster masks as primary data
- Mix tile-row and object-row matrices
- Encode geometry implicitly in feature matrices
- Assume adjacency without storing a graph

---

## 10. Mental Model (One Sentence per Layer)

- `coords/` → where tiles are  
- `features/` → what tiles look like  
- `annotations/` → where regions are  
- `instances/` → where objects are  
- `instance_features/` → what objects look like  
- `graphs/` → how things connect  
