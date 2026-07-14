API Reference
=============

PathForge is organized into packages for shared contracts, workflow policies,
integrations, training, and command-line entry points. This section documents
the stable interfaces most useful to users and extension authors; internal
implementation modules are intentionally omitted.

.. toctree::
   :maxdepth: 2

   config
   cli
   core
   artifacts
   evaluation
   slide_retrieval
   visualization
   policy
   training
   adapters
   utils
   inference

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
   * - :doc:`artifacts`
     - Slide artifact and retrieval artifact storage contracts.
   * - :doc:`evaluation`
     - Evaluation orchestration and slide-retrieval metrics.
   * - :doc:`slide_retrieval`
     - Representation, search, and result-rendering interfaces.
   * - :doc:`visualization`
     - Visualization registries, orchestration, and render result types.
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
