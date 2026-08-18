Contributing
============

This guide explains where new PathForge code belongs and how to preserve the
implemented architecture. Read :doc:`architecture` first for the package map
and workflow diagrams.

Choose a Location by Responsibility
-----------------------------------

Start with the behavior the new code owns:

.. code-block:: text

   Is it command-line parsing or command dispatch?
      └── yes ─► pathforge/cli

   Does it coordinate a complete user workflow?
      └── yes ─► pathforge/policy

   Is it a benchmark or inference task selected by experiment.task?
      └── yes ─► pathforge/core/tasks

   Is it a reusable scientific/data/storage abstraction or native implementation?
      └── yes ─► the matching pathforge/core subpackage

   Does it translate a third-party package into PathForge conventions?
      └── yes ─► pathforge/adapters

   Is it a trainer implementation?
      └── yes ─► pathforge/training

   Is it a retrieval representation, search method, or retrieval renderer?
      └── yes ─► pathforge/slide_retrieval

   Is it model restoration, package inference, or prediction heatmap logic?
      └── yes ─► pathforge/inference

   Is it scheduler/work-manifest logic?
      └── yes ─► pathforge/execution

   Is it small, domain-neutral, and reused by several packages?
      └── yes ─► pathforge/utils

Placement Reference
-------------------

.. list-table::
   :widths: 26 37 37
   :header-rows: 1

   * - Location
     - Put here
     - Do not put here
   * - ``cli``
     - Typer options, validation of command inputs, and delegation.
     - Training loops, feature algorithms, storage formats, or retrieval logic.
   * - ``config``
     - Pydantic fields, defaults, cross-field validation, and configuration
       serialization.
     - Workflow execution or expensive backend initialization.
   * - ``policy``
     - Multi-step use cases shared by CLI and execution workers.
     - Third-party compatibility shims or low-level HDF5 operations.
   * - ``core``
     - Contracts, typed domain values, datasets, native models, metrics,
       reusable storage, evaluation, and visualization services.
     - CLI dependencies. Avoid new dependencies on policies, training, or
       inference outside the existing ``core.tasks`` exception.
   * - ``core.tasks``
     - Task-specific benchmark and inference orchestration selected through the
       task registry.
     - Generic algorithms that can live in their dedicated core or retrieval
       subpackage.
   * - ``adapters``
     - Wrappers, conversions, and registrations for third-party libraries.
     - Backend-independent contracts or native algorithms.
   * - ``training``
     - Implementations of ``TrainerBase``, callbacks, and training metrics.
     - CLI parsing or task selection.
   * - ``slide_retrieval``
     - Representation/search strategies, compatibility types, retrieval
       storage helpers, and retrieval renderers.
     - General MIL training or general-purpose slide artifact I/O.
   * - ``inference``
     - Model-package restoration, direct bag prediction, and heatmap services.
     - Training orchestration or CLI presentation.
   * - ``execution``
     - Work records, manifests, worker lifecycle, scheduler rendering, and
       result aggregation.
     - Scientific computations already owned by policies or tasks.
   * - ``utils``
     - Small cross-cutting helpers with no better domain owner.
     - Workflow-specific logic; prefer the package that owns the concept.

Extending Registered Components
-------------------------------

Use an existing registry when users need to select an implementation by name.
Do not add a separate conditional dispatch chain for a component family that
already has a registry.

Before implementing a new component:

#. Search the repository for an existing base class, registry, and equivalent
   implementation.
#. Implement the relevant contract in the package that owns the behavior.
#. Register the implementation with the canonical registry or family-specific
   builder.
#. Add configuration validation only when the selectable name or parameters
   must be represented in YAML.
#. Test the contract, registry lookup, normal behavior, and edge cases.

Common extension points are:

.. list-table::
   :widths: 20 30 25 25
   :header-rows: 1

   * - Component
     - Must inherit from
     - Put the implementation in
     - Register with
   * - General dataset
     - ``DatasetBase``
     - ``core/datasets``
     - Dataset factory wiring; see the note below
   * - MIL bag dataset
     - ``BagDatasetBase`` (which already inherits ``DatasetBase``)
     - ``core/datasets``
     - Dataset factory wiring; see the note below
   * - Tile dataset
     - ``TileDatasetBase``
     - ``core/datasets``
     - Dataset factory wiring; see the note below
   * - Native PyTorch model
     - ``TorchModelBase``; use the more specific row below when possible
     - ``core/models``
     - ``MODELS`` when user-selectable
   * - MIL model
     - ``MILModelBase``
     - ``core/models`` for native models; ``adapters/<backend>`` for wrappers
     - ``MODELS``
   * - Pre-aggregated slide-vector model
     - ``SlideLevelModel`` for PyTorch, or ``ScikitBase`` for scikit-learn
     - ``core/models``
     - the existing slide-model catalog/registry path
   * - Loss
     - For native losses, the narrowest applicable class:
       ``ClassificationLoss``,
       ``RegressionLoss``, ``SurvivalContinuousLoss``, or
       ``SurvivalDiscreteLoss``. Third-party adapters may inherit ``BaseLoss``
       directly when they normalize the complete task contract themselves.
     - ``core/losses`` for native behavior; ``adapters/losses.py`` for
       third-party wrappers
     - ``LOSSES`` or ``SURVIVAL_LOSSES``
   * - Trainer
     - ``TrainerBase``
     - ``training``
     - ``TRAINERS``
   * - Experiment task
     - ``TaskBase``
     - ``core/tasks``
     - ``@register_task(<task-name>)``
   * - Slide-processing backend
     - ``SlideProcessorBase``
     - ``core/slide_processing`` for native behavior; ``adapters`` when it is
       principally a third-party translation
     - ``SLIDE_PROCESSORS``
   * - Explainer backend
     - ``ExplainerBase``
     - ``core`` for native behavior; ``adapters/<backend>`` for third-party
       behavior
     - ``EXPLAINERS``
   * - Retrieval representation strategy
     - ``BaseRetrievalRepresentationStrategy``
     - ``slide_retrieval/representation_strategies/strategies``
     - ``@register_representation_strategy(<name>)``
   * - Retrieval search strategy
     - ``BaseSearchStrategy``
     - ``slide_retrieval/search_strategies/strategies``
     - ``@register_search_strategy(<name>)``
   * - Task evaluation adapter
     - ``TaskEvaluationAdapterBase``
     - ``core/evaluation/tasks`` or the task-specific evaluation package
     - ``@evaluation_task_adapter(<task-name>)``
   * - Task visualization adapter
     - ``TaskVisualizationAdapterBase``
     - ``core/visualization/tasks``
     - ``@task_visualization_adapter(<task-name>)``

Inheritance Is Part of the Contract
-----------------------------------

When the table says “must inherit,” direct or indirect subclassing is required;
matching method names without subclassing is not sufficient. The base class
defines the callable surface used by policies, builders, trainers, or task
orchestrators. Implement every abstract method and preserve the documented
input/output types and tensor or array shapes.

Choose the narrowest base class that describes the implementation. For
example, a native classification loss should inherit ``ClassificationLoss``
rather than only ``BaseLoss`` because the intermediate class normalizes the
task's input contract. A third-party loss adapter may inherit ``BaseLoss`` and
perform that normalization itself, as the Torch loss adapters do. A MIL model
should inherit ``MILModelBase`` rather than only ``TorchModelBase`` because
trainers call ``forward_bag`` and may request instance scores.

The main required method contracts are:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Base class
     - What the subclass must provide
   * - ``DatasetBase``
     - ``name``, ``num_samples``, and indexed access. ``BagDatasetBase`` adds
       ``num_bags``; ``TileDatasetBase`` adds ``num_tiles``.
   * - ``MILModelBase``
     - ``bag_size`` and ``forward_bag`` for a bag shaped ``[B, N, D]``.
       Implement compatible attention output if the model supports heatmaps.
   * - ``SlideLevelModel``
     - ``forward_slide`` for pre-aggregated input shaped ``[B, D]``.
   * - ``ScikitBase``
     - ``fit`` and ``predict_as_tensor``; the inherited implementation supplies
       pickle-based ``save`` and ``load``.
   * - Loss task classes
     - ``calculate_loss`` with the task-specific target structure established
       by the intermediate base class.
   * - ``TrainerBase``
     - ``fit`` and ``predict`` using PathForge model, dataset, and loss
       contracts.
   * - ``TaskBase``
     - ``execute`` and the task's grid/dataset-use declarations. Also implement
       ``inference`` when the task supports config-driven inference.
   * - ``SlideProcessorBase``
     - WSI lifecycle, MPP lookup, thumbnail/tissue/tile/feature operations,
       tile-spec validation, region reads, cell extraction, and inspection.
   * - ``ExplainerBase``
     - ``initialize`` and ``explain``.
   * - ``BaseRetrievalRepresentationStrategy``
     - Compatibility declarations (supported feature levels and output kind)
       plus the strategy's representation-building behavior.
   * - ``BaseSearchStrategy``
     - Supported representation kinds and the search behavior; override index
       construction when the method needs a prepared index.
   * - ``TaskEvaluationAdapterBase``
     - Discovery keys, run discovery, and conversion to metric-ready data.
   * - ``TaskVisualizationAdapterBase``
     - Run discovery and rendering for the requested visualization names.

Some extension points intentionally use functions or factories instead of base
classes. Evaluation metrics registered with ``@evaluation_metric`` must follow
the registry's compute-function signature. Feature-extractor factories and
other factory-based registry entries must return the type consumed by their
caller. Do not invent a base-class requirement where the existing registry is
explicitly callable-based; copy a current registered implementation and add a
contract test for its callable signature.

The global ``DATASETS`` registry exists, but the current WSI and bag workflow
factories construct their concrete dataset classes directly. Registering a new
dataset class in ``DATASETS`` alone therefore does not make policies use it.
Add the appropriate branch or injectable construction path in
``core/datasets/factory.py`` and test the complete factory-to-policy path. Do
not document a dataset as configuration-selectable until that wiring exists.

Registration does not replace inheritance. A class can be present in a
registry and still violate the architecture if it does not extend the required
base. Interface tests enforce this for trainers, retrieval strategies, and
task evaluation/visualization adapters; new component families should receive
the same kind of test.

Canonical imports for the principal extension contracts are:

.. code-block:: python

   from pathforge.core.datasets.base import (
       DatasetBase,
       BagDatasetBase,
       TileDatasetBase,
   )
   from pathforge.core.models.base import TorchModelBase, ScikitBase
   from pathforge.core.models.mil_base import MILModelBase
   from pathforge.core.models.slide_base import SlideLevelModel
   from pathforge.core.losses.base import (
       BaseLoss,
       ClassificationLoss,
       RegressionLoss,
       SurvivalContinuousLoss,
       SurvivalDiscreteLoss,
   )
   from pathforge.training.base import TrainerBase
   from pathforge.core.tasks.base import TaskBase
   from pathforge.core.slide_processing.base import SlideProcessorBase
   from pathforge.core.explainer_base import ExplainerBase
   from pathforge.core.evaluation.base import TaskEvaluationAdapterBase
   from pathforge.core.visualization.base import TaskVisualizationAdapterBase
   from pathforge.slide_retrieval.representation_strategies.base import (
       BaseRetrievalRepresentationStrategy,
   )
   from pathforge.slide_retrieval.search_strategies.base import BaseSearchStrategy

Import the decorator or registry from its canonical registry module rather than
from an implementation module. This avoids making one implementation the
accidental public entry point for the entire component family.

Third-Party Integrations
------------------------

Code that imports and translates a third-party backend belongs in
``pathforge.adapters``. Keep the PathForge-facing contract and types in core,
then make the adapter implement that contract. If the dependency is optional:

* add availability detection under ``pathforge.utils.optional``;
* avoid importing the package merely by importing ``pathforge``;
* populate its registrations explicitly through the registry-loading path;
* provide a clear error when a configured component requires an unavailable
  package;
* add an import-safety or architecture test for the boundary.

Retrieval algorithms are the exception to the adapter-directory rule when they
implement PathForge's retrieval strategy contracts directly. Those belong
beside the other representation or search strategies under
``pathforge.slide_retrieval``.

Preserve the Dependency Boundaries
----------------------------------

Prefer dependencies that point from orchestration toward reusable contracts:

.. code-block:: text

   CLI / execution
          │
          ▼
      policies / tasks
          │
          ▼
   core contracts and services ◄──── adapters

In practical terms:

* never import ``pathforge.cli`` from policy, training, core, or adapters;
* do not import policy, training, or inference from core modules outside
  ``core.tasks``;
* do not extend the ``core.tasks`` exception unless the code genuinely
  coordinates a task-level use case;
* keep backend-specific conversions in adapters;
* share behavior through a focused contract or helper rather than importing an
  outer workflow to reuse an implementation detail;
* use local imports when an optional backend should only be loaded on demand.

If a feature appears to require a new dependency cycle, first extract the
shared contract or pure helper into the package that owns the underlying
concept. Any intentional exception should be documented and covered by an
architecture test.

Artifacts and Identifiers
-------------------------

Do not construct artifact identifiers with ad-hoc string formatting. Use the
canonical builders:

* ``build_tiling_id(combo_cfg)`` for a tiling-only ID;
* ``build_feature_name(combo_cfg)`` for feature storage;
* ``build_bag_id(combo_cfg)`` for the full bag identity;
* ``build_retrieval_representation_id(...)`` for a retrieval representation;
* ``build_retrieval_representation_entry_id(slide_ids)`` for a representation
  entry.

Use the ``tiling_id`` value for the HDF5 ``tile_id`` argument. New slide data
should use the existing HDF5 or slide-artifact layout helpers. Preserve the
row-alignment invariant between coordinates and feature matrices, and use
atomic slide-artifact writes when updating an existing file.

Tests Required for a Contribution
---------------------------------

Place tests according to their scope:

* ``tests/unit`` for each function's expected behavior and edge cases;
* ``tests/interface`` for base-class, registry, import-boundary, and public
  contract guarantees;
* ``tests/integration`` for interactions between real subsystems;
* ``tests/smoke`` for a complete policy or pipeline using the shared sample
  data in ``pathforge.utils.test_samples``.

Pipeline smoke tests should exercise the full workflow and record useful time
and memory metrics. Reuse fixtures and already-computed results instead of
repeating expensive calculations inside the test suite.

For an architectural change, run at least:

.. code-block:: bash

   uv run ruff check src tests
   uv run pytest tests/interface
   uv run pytest tests/unit/test_docs_build.py

Then run the focused unit, integration, or smoke tests for the changed
subsystem. Documentation changes should build with the ``docs`` extra and must
keep documented commands and code examples synchronized with the source.

Documentation and Code Style
----------------------------

Shared documentation belongs in ``docs`` and must be linked from an existing
toctree. Public functions and classes need typed docstrings that explain their
semantic purpose, expected inputs and outputs (including array/tensor shapes),
and a short usage example. Use inline comments to clarify non-obvious
implementation decisions.

Format and lint Python with Ruff. Before adding a helper or builder, search for
an existing implementation so that identifiers, storage layouts, registry
behavior, and configuration rules remain centralized.

Contribution Checklist
----------------------

Before submitting a change, confirm that:

* the code is in the package that owns its responsibility;
* an existing abstraction or helper was reused where possible;
* selectable implementations use the appropriate registry;
* optional dependencies remain lazy and isolated;
* canonical artifact identifiers and layout helpers are used;
* normal, edge, interface, and workflow behavior has proportional test
  coverage;
* architecture-boundary and documentation tests pass;
* user-facing behavior is documented and linked from the documentation tree.
