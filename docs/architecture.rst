Architecture
============

PathForge is a configuration-driven workflow framework for computational
pathology. It connects command-line workflows to reusable policies, task
implementations, scientific components, and persistent artifacts. Registries
allow a configuration value such as a model or search-strategy name to select
the corresponding implementation at runtime.

The architecture is modular, but it is not a strict sequence of isolated
layers. In particular, ``pathforge.core.tasks`` contains application-level
orchestration, and configuration is shared by most runtime subsystems. The
diagrams and dependency notes below describe the implemented code rather than
an idealized layering model.

System at a Glance
------------------

The normal direction of control is from a user command to an orchestrated
workflow and then to stored results:

.. code-block:: text

   YAML configuration
          │
          ▼
   ┌──────────────────────┐
   │ CLI and execution    │  Parse commands or schedule work units
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Policies and tasks   │  Coordinate complete use cases
   └──────────┬───────────┘
              ▼
   ┌──────────────────────────────────────────────────────┐
   │ Datasets │ models │ trainers │ retrieval strategies │
   │ slide processing │ evaluation │ visualization        │
   └──────────┬───────────────────────────────────────────┘
              ▼
   ┌──────────────────────────────────────────────────────┐
   │ HDF5 slide artifacts │ model packages │ CSV/XLSX/PDF │
   └──────────────────────────────────────────────────────┘

Component selection happens through shared registries:

.. code-block:: text

   configuration name
          │
          ▼
   ┌───────────────────┐       imports/registers       ┌──────────────────┐
   │ Registry          │◄──────────────────────────────│ Implementation   │
   │ models, tasks,    │                               │ native or adapter│
   │ losses, trainers, │────── constructs/returns ───►│                  │
   │ extractors, ...   │                               └──────────────────┘
   └───────────────────┘

Implemented Packages
--------------------

``pathforge.cli``
~~~~~~~~~~~~~~~~~

The Typer command tree is the main user-facing entry point. It exposes feature
extraction, slide retrieval, benchmarking, evaluation, visualization,
reporting, inference, pipeline optimization, and distributed-execution
commands. CLI modules translate command-line arguments into validated
configuration and delegate work to policies or dedicated services; they do
not contain the scientific algorithms themselves.

``pathforge.config``
~~~~~~~~~~~~~~~~~~~~

Pydantic models define and validate the YAML configuration for experiments,
datasets, feature extraction, MIL, retrieval, evaluation, optimization, and
execution. The resulting :class:`~pathforge.config.config.Config` object is the
shared construction input for policies, tasks, trainers, and execution plans.
Configuration also delegates external TCGA source resolution to the TCGA
adapter.

``pathforge.policy``
~~~~~~~~~~~~~~~~~~~~

Policies coordinate application workflows:

* :class:`~pathforge.policy.feature_extraction.FeatureExtractionPolicy`
  processes configured WSIs and writes reusable slide artifacts.
* :class:`~pathforge.policy.benchmarking.BenchmarkingPolicy` expands benchmark
  combinations, ensures their features exist, builds datasets, and delegates
  each run to its registered task.
* :class:`~pathforge.policy.optimization.OptimizationPolicy` runs Optuna-based
  searches over the configured pipeline.
* :class:`~pathforge.policy.inference.InferencePolicy` prepares datasets and
  delegates task-specific inference.

Policies depend on core data structures and use registries to resolve trainers,
losses, models, and other replaceable components.

``pathforge.core``
~~~~~~~~~~~~~~~~~~

``core`` contains both shared contracts and most reusable scientific services:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Area
     - Responsibility
   * - ``annotations``
     - Annotation contracts, CSV-backed annotations, and target binning.
   * - ``datasets``
     - WSI and feature-bag data models, schemas, factories, samplers, and
       task-specific dataset wrappers.
   * - ``experiments``
     - Project metadata, benchmark combinations, and canonical identifiers for
       tilings, features, and bags.
   * - ``models`` and ``losses``
     - Model/loss contracts plus native MIL, slide-vector, and scikit-learn
       implementations.
   * - ``tasks``
     - The task contract and registry, MIL task execution, and the complete
       slide-retrieval use case.
   * - ``io.h5``
     - Low-level reads and writes for HDF5 coordinates, features, tissue,
       descriptors, tiles, and prediction heatmaps.
   * - ``io.slide_artifacts``
     - Higher-level slide-artifact layouts and atomic update operations.
   * - ``io.slide_retrieval``
     - Retrieval-specific layouts, descriptors, and representation storage.
   * - ``slide_processing``
     - The slide-processor contract and LazySlide-based implementation.
   * - ``evaluation``
     - Evaluation orchestration, task adapters, and slide-retrieval metrics.
   * - ``visualization``
     - Visualization orchestration, thumbnails, tile overviews, and task
       visualization adapters.
   * - ``reports`` and ``explainer_base``
     - Report generation and the shared explainer contract.

Although tasks live under ``core``, they are use-case orchestrators rather than
inner domain objects. MIL tasks use policy construction helpers and trainer/loss
registries. ``SlideRetrievalTask`` coordinates the separate
``pathforge.slide_retrieval`` subsystem. The architecture tests deliberately
exclude ``core.tasks`` from the rule that prevents core from importing outer
application packages.

``pathforge.training``
~~~~~~~~~~~~~~~~~~~~~~

Training defines :class:`~pathforge.training.base.TrainerBase` and provides
Lightning and scikit-learn trainers. Trainers consume PathForge models and bag
datasets, calculate task metrics, and return checkpoints and objective scores.
The Lightning implementation also uses adapter-provided collation/output
normalization and can create an inference model package.

``pathforge.adapters``
~~~~~~~~~~~~~~~~~~~~~~

Adapters translate third-party behavior into PathForge conventions. Current
integrations cover TorchMIL, MIL-Lab, TorchMetrics, TorchSurv, PyTorch losses,
and TCGA Tools. Optional backend modules are loaded when their packages are
available; registrations make their implementations selectable in the same way
as native implementations.

``pathforge.slide_retrieval``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Slide retrieval is organized as two independently selectable stages:

.. code-block:: text

   bag features
       │
       ▼
   representation strategy ──► stored retrieval representation
       │
       ▼
   search strategy ──────────► ranked matches
       │
       ├──► retrieval metrics
       └──► manifests, spreadsheets, and visualizations

Representation strategies include mean-RGB, feature aggregation, prototype,
SPLICE, and Yottixel implementations. Search strategies include PBSS, RetCCL,
SISH, and Yottixel implementations. Each family has a base class, typed values,
and a registry. ``SlideRetrievalTask`` validates that the selected
representation, search strategy, feature level, and dataset roles are
compatible before running them.

``pathforge.inference``
~~~~~~~~~~~~~~~~~~~~~~~

Inference supports portable MIL model packages and attention heatmaps. A model
package records the configuration, registry model name, dimensions, metadata,
and learned state required to restore a trained model. Prediction helpers load
feature bags from supported array or HDF5 inputs and normalize backend-specific
outputs. Heatmap services write per-instance prediction scores back to slide
artifacts.

``pathforge.execution``
~~~~~~~~~~~~~~~~~~~~~~~

Distributed execution turns an experiment into immutable, scheduler-neutral
work records. Feature records contain deterministic slide shards; benchmark
records contain parameter combinations and depend on feature-record IDs.
Workers can execute locally, while generated SLURM array scripts use the same
manifests. Atomic JSON status files make work resumable and aggregatable.

``pathforge.utils``
~~~~~~~~~~~~~~~~~~~

Utilities provide registries, optional-dependency detection, constants,
serialization, logging, general I/O, and reusable test samples. This is a
cross-cutting package rather than an architectural layer. The global registry
module connects core registry contracts to built-in and optional adapter
registration.

End-to-End Workflows
--------------------

Feature Extraction
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   CLI ─► Config + Experiment ─► FeatureExtractionPolicy
                                      │
                                      ▼
                                 WSI datasets
                                      │
                                      ▼
                              registered slide processor
                                      │
                                      ▼
                     per-slide HDF5 artifact
                     ├── tissue information
                     ├── level-0 tile coordinates
                     ├── thumbnails/overviews
                     └── row-aligned feature matrices

Coordinates and feature rows share an index. A compatible existing tiling can
therefore be reused when another workflow needs the same bag features.

Benchmarking and Training
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   BenchmarkingPolicy
       ├── expand configured combinations
       ├── group combinations that share one bag
       ├── ensure the bag features exist
       ├── build datasets by their configured use
       └── registered task
             ├── MIL task ─► model + loss + trainer ─► checkpoint/score
             └── retrieval task ─► representation + search ─► ranked results

Grouping combinations by bag identity avoids repeating feature extraction for
model and loss combinations that consume the same features.

Pipeline Optimization
~~~~~~~~~~~~~~~~~~~~~

The optimization policy creates an Optuna study, samples configured pipeline
choices, and evaluates them through the same task, dataset, model, trainer, and
metric machinery used by benchmarking. This keeps optimized and benchmarked
pipelines comparable.

Inference
~~~~~~~~~

.. code-block:: text

   model package + input features
              │
              ▼
      restore registered model
              │
              ▼
       task-aware prediction
              │
              ├──► prediction output
              └──► optional attention heatmap in slide HDF5

Task-level inference is coordinated by ``InferencePolicy``. Model-package
helpers provide the lower-level restoration and bag-prediction path.

Distributed Execution
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   experiment configuration
             │
             ▼
        execution plan
        ├── feature manifest ─────┐
        └── benchmark manifest ◄──┘ dependency IDs
             │
             ├── local workers
             └── SLURM arrays
                    │
                    ▼
              atomic statuses ─► aggregate results

Dependency Boundaries
---------------------

The implemented import direction is summarized below. Arrows mean "imports
from"; they describe source dependencies, not the runtime flow shown earlier.

.. code-block:: text

   cli ─────────────► config, core, execution, inference, policy, retrieval
   policy ──────────► config, core, training, utils
   execution ───────► cli, config, core, policy
   training ────────► adapters, config, core, inference, utils
   inference ───────► adapters, config, core, policy, utils
   slide_retrieval ─► config, core, utils
   config ──────────► adapters, core, slide_retrieval, utils
   core ────────────► config, utils
   core.tasks ──────► policy and slide_retrieval as explicit exceptions
   adapters ────────► core, utils
   utils ───────────► adapters, core

The maintained boundary rules are narrower than strict Clean Architecture:

* core modules outside ``core.tasks`` must not import CLI, policy, training, or
  inference packages;
* policies must not import CLI modules;
* training must not import CLI modules;
* concrete trainers and retrieval/evaluation/visualization strategies must
  extend their declared base classes.

These rules are checked in ``tests/interface/test_dependency_boundaries.py``
and ``tests/interface/test_adapter_structure.py``. See :doc:`contributing` for
the placement rules contributors should follow when extending the system.

Artifact Contract
-----------------

Feature extraction and inference share one per-slide HDF5 artifact. Its central
layout is:

.. code-block:: text

   slide.h5
   └── bags/
       └── {tiling_id}/
           ├── coords              int32 (N, 5)
           ├── tiling_spec         JSON metadata
           ├── features/
           │   └── {feature_name}  float32 (N, D)
           ├── tiles_overview      encoded image bytes
           └── predictions/
               └── heatmaps/
                   └── {name}/
                       ├── coords
                       ├── scores
                       └── metadata

``coords`` stores ``[x0, y0, read_w, read_h, level]`` in level-0 coordinate
space. Feature matrices are row-aligned with those coordinates. Detailed
storage contracts are documented in :doc:`HDF5_structure` and
:doc:`slide_retrieval_h5_structure`.
