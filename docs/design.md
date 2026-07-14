# PathForge — Design Document

## 1. Overview

**PathForge** is a modular, extensible benchmarking framework for **Multiple Instance Learning (MIL)** in **computational pathology**.  
It supports interchangeable slide-processing backends for feature extraction and modular **PyTorch Lightning–based** MIL training pipelines.  
Pipeline configuration and optimization are driven via **Optuna**.

The framework is designed for:
- Fair and reproducible benchmarking of MIL methods
- Clean separation of concerns
- Framework-agnostic extensibility via registries and interfaces
- Long-term maintainability through documented package boundaries


## 2. Architectural Principles

PathForge uses interface-oriented modules and documented package boundaries,
with explicit cross-package dependencies for orchestration and integration.

### 2.1 Dependency Guidelines

Stable contracts should avoid unnecessary dependencies on CLI entry points and
concrete integrations. Application modules may compose core, training,
retrieval, configuration, and adapter functionality when required by a
workflow. New cross-package dependencies should be explicit and covered by
interface tests.

---

### 2.2 Domain Layer Constraints

The **Domain layer**:
- Contains **business rules and core logic only**
- Must **not** depend on concrete implementations
- Minimizes framework-specific imports
- May depend on *well-standardized scientific frameworks* (e.g. `torch`, `numpy`)
- Stable domain contracts should remain agnostic to:
  - CLI
  - File system layout
  - Training frameworks
  - Slide backends
  - Experiment orchestration (except the application-facing ``core.tasks``
    modules, which intentionally compose policy helpers)

---

### 2.3 Application / Use Case Layer

The **Use Case / Application layer**:
- Coordinates domain objects to achieve specific goals
- Depends only on **interfaces**, never concrete implementations
- Encodes *what* should happen, not *how*

Examples:
- Running a benchmarking workflow
- Executing a feature extraction pipeline
- Performing pipeline optimization

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

This is the **contract layer** of PathForge.

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
- Pipeline optimization policy

---

## 7. Registries (Plugin System)

Registries act as the **plugin backbone** of PathForge.

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

## 9. Artifact Contracts

Artifact layout is specified in dedicated canonical references rather than
duplicated in this design document:

- [Slide-level HDF5 structure](HDF5_structure.md)
- [Slide-retrieval HDF5 structure](slide_retrieval_h5_structure.md)

## 10. Summary

PathForge is designed as an **extensible and reproducible** framework for MIL benchmarking in computational pathology.
Documented interfaces, tested package boundaries, and extensible artifact formats support rapid experimentation while preserving correctness and maintainability.
