Architecture
============

PathBench follows the `Clean Architecture <https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html>`_
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

Core Layer (``pathbench.core``)
--------------------------------

The stable domain layer. Contains abstractions that never change when
frameworks are swapped.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Sub-package
     - Contents
   * - ``core.models``
     - :class:`~pathbench.core.models.base.ModelBase`,
       :class:`~pathbench.core.models.base.TorchModelBase`,
       :class:`~pathbench.core.models.mil_base.MILModelBase`,
       :class:`~pathbench.core.models.base.ScikitBase`
   * - ``core.losses``
     - :class:`~pathbench.core.losses.base.BaseLoss`,
       :class:`~pathbench.core.losses.base.ClassificationLoss`,
       :class:`~pathbench.core.losses.base.SurvivalContinuousLoss`,
       :class:`~pathbench.core.losses.base.SurvivalDiscreteLoss`
   * - ``core.datasets``
     - :class:`~pathbench.core.datasets.base.DatasetBase`,
       :class:`~pathbench.core.datasets.base.BagDatasetBase`,
       :class:`~pathbench.core.datasets.wsi_dataset.WSIDataset`,
       :class:`~pathbench.core.datasets.wsi_dataset.WSI`
   * - ``core.experiments``
     - :class:`~pathbench.core.experiments.base.Experiment`,
       :class:`~pathbench.core.experiments.base.ComboConfig`
   * - ``core.io.h5``
     - :class:`~pathbench.core.io.h5.base.FileHandleH5`, coordinate/feature/tissue I/O helpers
   * - ``core.slide_processing``
     - :class:`~pathbench.core.slide_processing.base.SlideProcessorBase`
   * - ``core.explainer_base``
     - :class:`~pathbench.core.explainer_base.ExplainerBase`

Policy Layer (``pathbench.policy``)
-------------------------------------

Orchestrates domain objects. Never imports concrete framework packages directly.

- :class:`~pathbench.policy.feature_extraction.FeatureExtractionPolicy` — Runs tiling and feature extraction for all configured combinations.
- :class:`~pathbench.policy.benchmarking.BenchmarkingPolicy` — Grid-searches model/loss/feature configurations.
- :class:`~pathbench.policy.optimization.OptimizationPolicy` — Runs Optuna studies.

CLI Layer (``pathbench.cli``)
------------------------------

Thin shells that parse arguments and delegate to policies.

- :func:`~pathbench.cli.feature_extraction.main` → ``pathbench-features``
- :func:`~pathbench.cli.benchmark.main` → ``pathbench-benchmark``
- :func:`~pathbench.cli.optimize.main` → ``pathbench-optimize``
- :func:`~pathbench.cli.inference.main` → ``pathbench-infer``

Infrastructure / Adapters (``pathbench.adapters``, ``pathbench.training``)
---------------------------------------------------------------------------

Concrete implementations registered through the registry system.

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Module
     - What it provides
   * - ``training.lightning``
     - PyTorch Lightning trainer registered as ``"lightning"`` in :data:`~pathbench.utils.registries.TRAINERS`.
   * - ``adapters.torchmil.backend``
     - ``TorchMILBackendModel`` — generic PathBench wrapper for any TorchMIL model.
   * - ``adapters.torchmil.collate``
     - Collate functions for canonical PathBench bag dictionaries and padded TorchMIL-compatible batches.
   * - ``adapters.torchmil.heatmap_explainer``
     - TorchMIL heatmap explainer registered in :data:`~pathbench.utils.registries.EXPLAINERS`.
   * - ``adapters.metrics.classification``
     - TorchMetrics classification backend.
   * - ``adapters.metrics.survival``
     - TorchSurv survival metrics backend.
   * - ``adapters.losses``
     - ``torch.nn`` and TorchSurv-backed loss adapters registered in :data:`~pathbench.utils.registries.LOSSES`.
   * - ``core.slide_processing.lazyslide``
     - Lazyslide/WSIData/timm slide processor registered in :data:`~pathbench.utils.registries.SLIDE_PROCESSORS`.

Registry System
---------------

All extensible components use a registry pattern:

.. code-block:: python

   from pathbench.utils.registries import MODELS, LOSSES, TRAINERS
   from pathbench.utils.registries import FEATURE_EXTRACTORS, SLIDE_PROCESSORS
   from pathbench.utils.registries import CLASSIFICATION_METRICS, SURVIVAL_METRICS
   from pathbench.utils.registries import EXPLAINERS

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

   from pathbench.utils.registries import populate_dynamic_registries
   populate_dynamic_registries()  # conditionally registers TorchMIL, TorchMetrics, TorchSurv

H5 Artifact Contract
--------------------

PathBench writes one H5 file per slide. The layout is backend-agnostic:

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

- ``pathbench.adapters.*`` — concrete implementations
- ``pathbench.utils.optional.*`` — availability guards

All other layers (core, policy, training, config, CLI) are import-safe without
these packages. Architecture tests in ``tests/unit/test_torchmil_architecture.py``
enforce this boundary automatically.
