# PathBench-MIL — Design Document

## 1. Overview

**PathBench-MIL** is a modular, extensible benchmarking framework for **Multiple Instance Learning (MIL)** in **computational pathology**.  
It supports interchangeable slide-processing backends for feature extraction and modular **PyTorch Lightning–based** MIL training pipelines.  
Pipeline configuration and optimization are driven via **Optuna**.

The framework is designed for:
- Fair and reproducible benchmarking of MIL methods
- Clean separation of concerns
- Framework-agnostic extensibility via registries and interfaces
- Long-term maintainability through strict architectural boundaries

---

## 2. Architectural Principles

PathBench-MIL strictly adheres to **Clean Architecture** principles.

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
**Path:** `src/pathbench/cli`

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

This is the **contract layer** of PathBench-MIL.

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

Registries act as the **plugin backbone** of PathBench-MIL.

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

For each WSI, PathBench-MIL produces a:

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

---

## 10. Summary

PathBench-MIL is designed as a **clean, extensible, and reproducible** framework for MIL benchmarking in computational pathology.  
By enforcing strict architectural boundaries, interface-based design, and extensible artifact formats, it enables rapid experimentation without sacrificing correctness, maintainability, or scientific rigor.

---