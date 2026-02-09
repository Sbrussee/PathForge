# PathBench-MIL HDF5 Structure — Canonical Overview

**File:**  

{slide_stem}.h5


This document defines the **authoritative structure** of PathBench-MIL slide-level HDF5 files, including explicit rules for **where** each artifact is stored and **how** it is formatted.

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
- `schema_name = "pathbench_mil_extended"`
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

**Purpose:** spatial definition of MIL bags

**Path:**  

coords/


**Shape:** `(N, 5)`  
**Dtype:** `int32`

**Columns (fixed order):**
1. `x` – level-0 pixel x
2. `y` – level-0 pixel y
3. `read_w` – patch width at read level
4. `read_h` – patch height at read level
5. `level` – pyramid level read from

**Rules**
- Row order is **authoritative**
- All tile-aligned matrices must match this row order

---

## 2. Feature Matrices (Tile-Level)

**Purpose:** MIL inputs

**Path:**  

features/{extractor_name}


**Shape:** `(N, D)`  
**Dtype:** `float32`

**Rules**
- One dataset per extractor
- Must be **row-aligned with `coords/`**
- No spatial information stored here

---

## 3. Tissue Masks (Vector-Based)

**Purpose:** define valid tissue regions

**Path:**  

annotations/tissue/


**Vector encoding:** polygon rings

**Datasets**
- `points : (M, 2) float32`
- `ring_offsets : (R+1,) int64`
- `poly_offsets : (P+1,) int64`
- `poly_label_id : (P,) int32`  
  → index into `labels/tissue/classes`

**Attributes**
- `geometry_type = "polygon_rings"`
- `coord_space = "level0_pixel"`
- `fill_rule = "even-odd"`

**Rules**
- Polygons may contain holes
- Polygons may overlap (union semantics implied)
- Raster tissue masks are **derived artifacts only**

---

## 4. Semantic Annotations / Segmentations (Vector-Based)

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

## 5. Instance Annotations / Segmentations  
*(cells, nuclei, points, small objects)*

**Purpose:** object-level entities

**Path:**  

instances/{instance_type}/


Example:

instances/nuclei/


### 5.1 Instance Table (Required)

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

### 5.2 Optional Instance Geometry

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

## 6. Feature Matrices for Instances / Semantic Objects

### 6.1 Instance-Level Features (Cells, Nuclei)

**Path:**  

instance_features/{instance_type}/{extractor_name}


**Shape:** `(I, D)`  
**Dtype:** `float32`

**Rules**
- Row-aligned with `instances/{instance_type}/id`
- No spatial data stored here

---

### 6.2 Semantic Object Features (Regions)

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
