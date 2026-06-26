# PathForge-MIL — Design Document

## 1. Overview

**PathForge-MIL** is a modular, extensible benchmarking framework for **Multiple Instance Learning (MIL)** in **computational pathology**.  
It supports interchangeable slide-processing backends for feature extraction and modular **PyTorch Lightning–based** MIL training pipelines.  
Pipeline configuration and optimization are driven via **Optuna**.

The framework is designed for:
- Fair and reproducible benchmarking of MIL methods
- Clean separation of concerns
- Framework-agnostic extensibility via registries and interfaces
- Long-term maintainability through strict architectural boundaries

---

## 2. Architectural Principles

PathForge-MIL strictly adheres to **Clean Architecture** principles.

### 2.1 Dependency Rule

All source code dependencies must point **inward**:

Interfaces → Adapters → Application / Use Cases → Domain


Outer layers may depend on inner layers, but **never the reverse**.

---

### 2.2 Domain Layer Constraints

The **Domain layer**:
- Contains **business rules and core logic only**
- Must **not** depend on concrete implementations
- Minimizes framework-specific imports
- May depend on *well-standardized scientific frameworks* (e.g. `torch`, `numpy`)
- Must be completely agnostic to:
  - CLI
  - File system layout
  - Training frameworks
  - Slide backends
  - Experiment orchestration

---

### 2.3 Application / Use Case Layer

The **Use Case / Application layer**:
- Coordinates domain objects to achieve specific goals
- Depends only on **interfaces**, never concrete implementations
- Encodes *what* should happen, not *how*

Examples:
- Running a benchmarking workflow
- Executing a feature extraction pipeline
- Performing hyperparameter optimization

---

### 2.4 Interface-Based Implementations

All concrete implementations (e.g. slide backends, models, trainers) must:
- Hide behind **abstract base classes or protocols**
- Be resolved dynamically via **registries**
- Never be imported directly by domain or use-case code

This ensures:
- Framework-agnostic design
- Plug-and-play extensibility
- Safe refactoring and experimentation

---

## 3. High-Level System Flow

Config
→ Experiment
→ Policy
→ Registry-resolved implementations
→ Artifacts (ROIs / tiles / features)


1. **Configuration** defines runtime behavior and parameters
2. **Experiment** establishes project context and lifecycle
3. **Policies** define runtime strategies
4. **Registries** resolve concrete implementations
5. **Artifacts** are produced and persisted

---

## 4. Codebase Structure

### 4.1 CLI Layer  
**Path:** `src/pathforge/cli`

**Responsibilities:**
- Command-line argument parsing
- Configuration loading and validation (via **Pydantic**)
- Entry-point for invoking experiments
- Selecting and launching policies

**Key Characteristics:**
- No domain logic
- No direct dependency on implementations
- Thin orchestration layer only

---

### 4.2 Policy Layer

**Responsibilities:**
- Orchestrates workflows over an experiment context
- Defines *runtime strategies* for executing experiments

**Examples:**
- Benchmarking policies
- Feature extraction policies
- Optimization (Optuna) policies

**Inputs:**
- An `ExperimentLike` protocol

**Characteristics:**
- Coordinates, but does not implement logic
- Depends on interfaces and registries
- Drives execution order and control flow

---

### 4.3 Core Layer

The **Core layer** contains stable abstractions and domain objects.

**Includes:**
- Abstract base classes
- Domain models
- Protocols and interfaces
- Core I/O definitions
- Dataset abstractions
- Slide processing interfaces
- Model, loss, and task definitions

This is the **contract layer** of PathForge-MIL.

---

## 5. Experiments

### 5.1 Experiment Definition

An **Experiment**:
- Represents a single project lifecycle
- Encapsulates filesystem layout and run context
- Manages:
  - Annotation loading
  - Dataset instantiation
  - Experiment-level metadata
- Builds the **search space** consumed by policies

### 5.2 Responsibilities

An experiment is responsible for:
- Defining where artifacts live
- Defining how data is loaded
- Providing context for policies
- Remaining implementation-agnostic

---

## 6. Policies

Policies define **how an experiment is executed**.

### 6.1 Characteristics

- Stateless or minimally stateful
- Operate over an experiment context
- Invoke registry-resolved implementations
- Do not own data or filesystem structure

### 6.2 Examples

- Feature extraction policy
- MIL benchmarking policy
- Hyperparameter optimization policy

---

## 7. Registries (Plugin System)

Registries act as the **plugin backbone** of PathForge-MIL.

### 7.1 Registry Responsibilities

Registries:
- Hold mappings from symbolic names → implementations
- Resolve implementations at runtime
- Enable external and downloadable extensions

### 7.2 Registry Targets

All extensible components must be registry-based, including:
- Slide processing backends
- Feature extractors
- MIL models
- Training frameworks
- Tasks and objectives
- Explainers
- Datasets

---

## 8. Whole Slide Image (WSI) Abstraction

### 8.1 WSI Dataclass

Whole Slide Images must be interfaced via a **WSI dataclass**.

**Supported Input Formats (backend-dependent):**
- `.tiff`
- `.svs`
- `.ndpi`
- Other OpenSlide-compatible formats

The WSI abstraction:
- Shields domain logic from backend-specific details
- Exposes standardized access to slide properties

---

## 9. Slide-Level Artifact Format (`.h5`)

For each WSI, PathForge-MIL produces a:

{slide_stem}.h5


This format is **row-aligned**, **extensible**, and **backend-agnostic**.

---

### 9.1 Coordinates

**Path:** `coords/`

**Shape:** `(N, 5)`  
**Dtype:** `int32`

**Columns:**
1. `x` – level-0 pixel coordinate
2. `y` – level-0 pixel coordinate
3. `read_w` – width of the patch as read
4. `read_h` – height of the patch as read
5. `level` – pyramid level from which the patch was read

Coordinates and features **must share row order**.

---

### 9.2 Features

**Path:** `features/{extractor_name}`

**Shape:** `(N, D)`  
**Dtype:** `float32`

- One feature matrix per extractor
- Row-aligned with `coords`

---

### 9.3 Metadata and Attributes

#### Patch-Level Metadata
- `patch_size_level0`  
  Records the patch size in **level-0 pixel space**

#### Slide-Level Metadata
- `num_levels`
- `level0_width`
- `level0_height`

#### Optional Attributes (if available)
- `mpp` — microns per pixel
- `scanner` — scanner identifier
- `dataset` — dataset name

#### Additional Metadata
- ROIs
- Masks
- Arbitrary attributes or matrices

---

### 9.4 Extensibility

The `.h5` format is explicitly designed to:
- Add new feature groups
- Add new coordinate-aligned matrices
- Add arbitrary metadata attributes
- Maintain backward compatibility

## 9.5 Vector-Based Annotations (ASAP / QuPath Compatible)

PathForge-MIL supports **vector-based annotations** for tissue and semantic regions, designed for interoperability with **ASAP XML** and **QuPath GeoJSON**.

Vector annotations are:
- Stored in **level-0 pixel coordinates**
- Polygon-based (with optional holes)
- Independent of patch sampling
- Backend-agnostic
- Round-trip convertible to ASAP and QuPath formats

This section defines a **minimal, canonical vector storage schema** inside the slide-level `.h5` file.

---

### 9.5.1 Coordinate Convention

All vector annotations:
- Use **level-0 pixel space**
- Follow image coordinate conventions:
  - `x` increases to the right
  - `y` increases downward

This matches:
- `coords/` patch coordinates
- ASAP base image coordinates
- QuPath image coordinates

---

### 9.5.2 Label Definitions

#### Tissue Labels
**Path:** `labels/tissue/classes`  
**Shape:** `(2,)`  
**Type:** UTF-8 strings

Recommended:

["background", "tissue"]


#### Semantic Labels
**Path:** `labels/semantic/classes`  
**Shape:** `(K,)`  
**Type:** UTF-8 strings

Example:

["background", "tumor", "stroma", "necrosis"]


Label indices are used by vector annotations via `poly_label_id`.

---

### 9.5.3 Vector Geometry Encoding (Polygon Rings)

All vector annotations are stored as **polygons with rings**:

- A polygon consists of one **exterior ring**
- Followed by zero or more **hole rings**
- Rings are simple vertex sequences `(x, y)`

To remain HDF5-native and efficient, geometry is stored using **packed arrays with offsets**.

---

### 9.5.4 Tissue Annotations (Vector)

**Path:** `annotations/tissue/`

#### Datasets
- `points` : `(M, 2) float32`  
  All polygon vertices, concatenated.
- `ring_offsets` : `(R+1,) int64`  
  Ring `r` uses vertices:

points[ring_offsets[r] : ring_offsets[r+1]]

- `poly_offsets` : `(P+1,) int64`  
Polygon `p` uses rings:

rings[poly_offsets[p] : poly_offsets[p+1]]

- `poly_label_id` : `(P,) int32`  
Label index into `/labels/tissue/classes`
(typically all `1` for tissue).

#### Attributes
- `geometry_type = "polygon_rings"`
- `coord_space = "level0_pixel"`
- `classes_path = "/labels/tissue/classes"`
- `ring_convention = "first_exterior_rest_holes"`
- `fill_rule = "even-odd"`

---

### 9.5.5 Semantic Annotations (Vector)

**Path:** `annotations/semantic/`

#### Datasets
- `points` : `(M, 2) float32`
- `ring_offsets` : `(R+1,) int64`
- `poly_offsets` : `(P+1,) int64`
- `poly_label_id` : `(P,) int32`  
Class index into `/labels/semantic/classes`

#### Optional Datasets
- `poly_name` : `(P,)` UTF-8 string  
Original annotation class name
- `poly_confidence` : `(P,) float32`  
Model-derived confidence (if applicable)
- `poly_attrs_json` : `(P,)` UTF-8 string  
Tool-specific metadata

#### Attributes
- `geometry_type = "polygon_rings"`
- `coord_space = "level0_pixel"`
- `classes_path = "/labels/semantic/classes"`
- `ring_convention = "first_exterior_rest_holes"`
- `fill_rule = "even-odd"`
- `overlap_rule = "allow" | "priority"`

If `overlap_rule = "priority"`, class priorities may be stored at:

labels/semantic/priority : (K,) int32


---

### 9.5.6 ASAP XML Interoperability

#### ASAP → HDF5
- Each ASAP polygon becomes one HDF5 polygon
- Holes are stored as additional rings
- ASAP group / class → `poly_label_id`
- Coordinates are converted to level-0 pixels if needed

Metadata:

meta/annotation_source = "asap_xml"


#### HDF5 → ASAP
- Each polygon is exported as an ASAP annotation
- Ring structure preserved
- Label IDs mapped back to ASAP groups/classes

---

### 9.5.7 QuPath GeoJSON Interoperability

#### QuPath → HDF5
- `Polygon` → single polygon
- `MultiPolygon` → multiple polygons
- `geometry.coordinates` → rings
- `properties.classification.name` → semantic label

Metadata:

meta/annotation_source = "qupath_geojson"


#### HDF5 → QuPath
- Polygons exported as GeoJSON `Polygon`
- Class names restored from label table
- Optional metadata written into `properties`

---

### 9.5.8 Required Conversion Utilities

Support for vector annotations requires the following functions:

#### Parsing
- `asap_xml_to_h5(xml_path, h5_path)`
- `qupath_geojson_to_h5(geojson_path, h5_path)`

#### Exporting
- `h5_to_asap_xml(h5_path, xml_path)`
- `h5_to_qupath_geojson(h5_path, geojson_path)`

All converters must:
- Preserve geometry and holes
- Preserve class labels
- Maintain level-0 coordinate consistency
- 
---
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
## 10. Summary

PathForge-MIL is designed as a **clean, extensible, and reproducible** framework for MIL benchmarking in computational pathology.  
By enforcing strict architectural boundaries, interface-based design, and extensible artifact formats, it enables rapid experimentation without sacrificing correctness, maintainability, or scientific rigor.

---
