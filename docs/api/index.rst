API Reference
=============

PathForge is organized into several packages following Clean Architecture
principles. All public interfaces are documented here.

.. toctree::
   :maxdepth: 2

   config
   cli
   core
   policy
   training
   adapters
   utils
   inference
   optimization

Package Map
-----------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Package
     - Role
   * - :doc:`config`
     - Pydantic v2 config classes loaded from YAML. Drives all construction.
   * - :doc:`cli`
     - Thin CLI shells wrapping policies. Entry points for all commands.
   * - :doc:`core`
     - Domain layer: model/loss/dataset abstractions, H5 I/O, experiments.
   * - :doc:`policy`
     - Use-case layer: feature extraction, benchmarking, optimization.
   * - :doc:`training`
     - Trainer abstractions and the Lightning implementation.
   * - :doc:`adapters`
     - Concrete integrations: TorchMIL, TorchMetrics, TorchSurv, LazySlide.
   * - :doc:`utils`
     - Registries, optional guards, constants, logging utilities.
   * - :doc:`inference`
     - Heatmap generation and H5-backed inference visualization helpers.
   * - :doc:`optimization`
     - Optuna helpers removed; see :doc:`policy` (``OptimizationPolicy``) and ``pathforge.policy.utils``.
