PathForge
=========

**PathForge** is a modular benchmarking framework for multiple instance
learning (MIL) in computational pathology. It supports whole-slide image (WSI)
feature extraction, H5 artifact generation, tile overview reports, MIL
benchmarking, pipeline optimization, optional TorchMIL and MIL-Lab backends,
metric adapters, and explainability hooks.

Policies and trainers resolve implementations through PathForge interfaces and
registries, while concrete third-party integrations are concentrated in
adapter modules where practical.

----

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   introduction
   installation
   data_preparation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   configuration
   mil_options
   testing
   backends
   architecture
   HDF5_structure
   slide_retrieval_h5_structure
   slide-retrieval-results-and-metrics
   task_outputs
   supported_slide_files
   troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index

----

Key Capabilities
----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Feature
     - Description
   * - Feature extraction
     - Tile WSIs, segment tissue, extract tile features, persist row-aligned H5 artifacts.
   * - Benchmarking
     - Grid-search over model, loss, feature extractor, activation, and optimizer combinations.
   * - Optimization
     - Optuna-driven pipeline search with configurable samplers and pruners.
   * - Inference
     - Checkpoint-based prediction and per-instance attention heatmap generation.
   * - Backends
     - Native PathForge models, TorchMIL, or MIL-Lab via backend adapters.
   * - Metrics/losses
     - Optional TorchMetrics (classification) and TorchSurv (survival) integrations.
   * - Explainability
     - Per-instance MIL attention heatmaps stored alongside slide H5 artifacts.
   * - TCGA integration
     - Direct dataset download via ``tcga-tools``, auto-generated annotation CSVs.
