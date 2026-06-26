Architecture
============

PathForge follows the `Clean Architecture <https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html>`_
pattern. Concrete implementations depend on abstractions; abstractions never
depend on implementations.

Layer Overview
--------------

.. code-block:: text

   ┌─────────────────────────────────────────────────┐
   │  CLI  (thin shell, argument parsing only)        │
   └──────────────────────┬──────────────────────────┘
                          │
   ┌──────────────────────▼──────────────────────────┐
   │  Policy  (use-case orchestration)                │
   │  FeatureExtractionPolicy                        │
   │  BenchmarkingPolicy                             │
   │  OptimizationPolicy                             │
   └──────────────────────┬──────────────────────────┘
                          │
   ┌──────────────────────▼──────────────────────────┐
   │  Core / Domain  (stable, framework-agnostic)     │
   │  Models, Losses, Datasets, Experiments, IO       │
   └──────────────┬──────────────────────────────────┘
                  │ ◄── depends on abstractions only
   ┌──────────────▼──────────────────────────────────┐
   │  Infrastructure / Adapters                       │
   │  LightningTrainer, TorchMILBackendModel,         │
   │  TorchMetrics, TorchSurv, LazySlide             │
   └─────────────────────────────────────────────────┘
                  │
   ┌──────────────▼──────────────────────────────────┐
   │  Config  (Pydantic v2 validation)                │
   │  Drives all layer construction                   │
   └─────────────────────────────────────────────────┘

Dependency Rule
---------------

Inner layers never import from outer layers:

- ``core/`` does not import from ``policy/``, ``cli/``, or any adapter.
- ``policy/`` imports from ``core/`` and resolves implementations via registries.
- ``cli/`` imports from ``policy/`` and ``config/`` only.
- ``adapters/`` and ``infrastructure/`` implement ``core/`` interfaces and
  register themselves.

This means the domain logic is always testable without any framework or
optional package.

Core Layer (``pathforge.core``)
--------------------------------

The stable domain layer. Contains abstractions that never change when
frameworks are swapped.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Sub-package
     - Contents
   * - ``core.models``
     - :class:`~pathforge.core.models.base.ModelBase`,
       :class:`~pathforge.core.models.base.TorchModelBase`,
       :class:`~pathforge.core.models.mil_base.MILModelBase`,
       :class:`~pathforge.core.models.base.ScikitBase`
   * - ``core.losses``
     - :class:`~pathforge.core.losses.base.BaseLoss`,
       :class:`~pathforge.core.losses.base.ClassificationLoss`,
       :class:`~pathforge.core.losses.base.SurvivalContinuousLoss`,
       :class:`~pathforge.core.losses.base.SurvivalDiscreteLoss`
   * - ``core.datasets``
     - :class:`~pathforge.core.datasets.base.DatasetBase`,
       :class:`~pathforge.core.datasets.base.BagDatasetBase`,
       :class:`~pathforge.core.datasets.wsi_dataset.WSIDataset`,
       :class:`~pathforge.core.datasets.wsi_dataset.WSI`
   * - ``core.experiments``
     - :class:`~pathforge.core.experiments.base.Experiment`,
       :class:`~pathforge.core.experiments.base.ComboConfig`
   * - ``core.tasks``
     - :class:`~pathforge.core.tasks.base.TaskBase`, task registry
       (:func:`~pathforge.core.tasks.registry.register_task`,
       :func:`~pathforge.core.tasks.registry.build_task`,
       :func:`~pathforge.core.tasks.registry.import_task_modules`),
       MIL tasks (``ClassificationMilTask``, ``RegressionMilTask``,
       ``SurvivalMilTask``, ``SurvivalDiscreteMilTask``),
       :class:`~pathforge.core.tasks.slide_retrieval.SlideRetrievalTask`
   * - ``core.io.h5``
     - :class:`~pathforge.core.io.h5.base.FileHandleH5`, coordinate/feature/tissue I/O helpers
   * - ``core.slide_processing``
     - :class:`~pathforge.core.slide_processing.base.SlideProcessorBase`
   * - ``core.explainer_base``
     - :class:`~pathforge.core.explainer_base.ExplainerBase`

Policy Layer (``pathforge.policy``)
-------------------------------------

Orchestrates domain objects. Never imports concrete framework packages directly.

- :class:`~pathforge.policy.feature_extraction.FeatureExtractionPolicy` — Runs tiling and feature extraction for all configured combinations.
- :class:`~pathforge.policy.benchmarking.BenchmarkingPolicy` — Grid-searches model/loss/feature configurations.
- :class:`~pathforge.policy.optimization.OptimizationPolicy` — Runs Optuna studies.

CLI Layer (``pathforge.cli``)
------------------------------

Thin shells that parse arguments and delegate to policies.

- :func:`~pathforge.cli.feature_extraction.main` → ``pathforge-features``
- :func:`~pathforge.cli.benchmark.main` → ``pathforge-benchmark``
- :func:`~pathforge.cli.optimize.main` → ``pathforge-optimize``
- :func:`~pathforge.cli.inference.main` → ``pathforge-infer``
- :func:`~pathforge.cli.slide_retrieval_representations.main` → ``pathforge-slide-retrieval-representations``

Infrastructure / Adapters (``pathforge.adapters``, ``pathforge.training``)
---------------------------------------------------------------------------

Concrete implementations registered through the registry system.

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Module
     - What it provides
   * - ``training.lightning``
     - PyTorch Lightning trainer registered as ``"lightning"`` in :data:`~pathforge.utils.registries.TRAINERS`.
   * - ``adapters.torchmil.backend``
     - ``TorchMILBackendModel`` — generic PathForge wrapper for any TorchMIL model.
   * - ``adapters.torchmil.collate``
     - Collate functions for canonical PathForge bag dictionaries and padded TorchMIL-compatible batches.
   * - ``adapters.torchmil.heatmap_explainer``
     - TorchMIL heatmap explainer registered in :data:`~pathforge.utils.registries.EXPLAINERS`.
   * - ``adapters.metrics.classification``
     - TorchMetrics classification backend.
   * - ``adapters.metrics.survival``
     - TorchSurv survival metrics backend.
   * - ``adapters.losses``
     - ``torch.nn`` and TorchSurv-backed loss adapters registered in :data:`~pathforge.utils.registries.LOSSES`.
   * - ``core.slide_processing.lazyslide``
     - Lazyslide/WSIData/timm slide processor registered in :data:`~pathforge.utils.registries.SLIDE_PROCESSORS`.

Registry System
---------------

All extensible components use a registry pattern:

.. code-block:: python

   from pathforge.utils.registries import MODELS, LOSSES, TRAINERS
   from pathforge.utils.registries import FEATURE_EXTRACTORS, SLIDE_PROCESSORS
   from pathforge.utils.registries import CLASSIFICATION_METRICS, SURVIVAL_METRICS
   from pathforge.utils.registries import EXPLAINERS

Registration is done via a decorator:

.. code-block:: python

   @MODELS.register("MyMIL")
   class MyMIL(MILModelBase):
       ...

Lookup:

.. code-block:: python

   cls = MODELS.get("MyMIL")
   print(MODELS.is_available("MyMIL"))

Dynamic population of optional backends:

.. code-block:: python

   from pathforge.utils.registries import populate_dynamic_registries
   populate_dynamic_registries()  # conditionally registers TorchMIL, TorchMetrics, TorchSurv

H5 Artifact Contract
--------------------

PathForge writes one H5 file per slide. The layout is backend-agnostic:

.. code-block:: text

   slide.h5
   └── bags/
       └── {tile_px}px_{tile_mpp:g}mpp/
           ├── coords         — int32  (N, 5): [x0, y0, read_w, read_h, level]
           ├── tiling_spec    — JSON:  tile_px, tile_mpp, stride_px, coord_space
           ├── features/
           │   └── {extractor}  — float32 (N, D), row-aligned with coords
           ├── tiles_overview — uint8  (M,): compressed JPEG/PNG bytes
           └── predictions/
               └── heatmaps/
                   └── {name}/
                       ├── coords    — float32 (K, 2)
                       ├── scores    — float32 (K,) in [0, 1]
                       └── metadata  — JSON

Invariants:

- ``coords`` rows and ``features`` rows share the same index.
- ``tiling_spec`` always contains ``coord_space: "level0"``.
- Feature extraction can reuse existing rows when the tiling spec matches.

Optional Package Isolation
--------------------------

Optional dependencies (``torchmil``, ``torchmetrics``, ``torchsurv``) are
confined to two locations:

- ``pathforge.adapters.*`` — concrete implementations
- ``pathforge.utils.optional.*`` — availability guards

All other layers (core, policy, training, config, CLI) are import-safe without
these packages. Architecture tests in ``tests/unit/test_torchmil_architecture.py``
enforce this boundary automatically.
