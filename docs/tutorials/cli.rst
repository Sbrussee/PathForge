Tutorial: Command-Line Interface
================================

All PathBench workflows are driven by CLI commands. Every command accepts a
``--config`` flag pointing to a YAML file and an optional ``--log-level`` flag.

Entry Points
------------

When the package is installed, four commands are available globally:

.. code-block:: bash

   pathbench-features   --config features.yaml
   pathbench-benchmark  --config benchmark.yaml
   pathbench-optimize   --config optimize.yaml
   pathbench-infer      --model_path ckpt --input h5 --output json

All commands can also be invoked as Python modules (useful during development):

.. code-block:: bash

   python -m pathbench.cli.feature_extraction  --config features.yaml
   python -m pathbench.cli.benchmark           --config benchmark.yaml
   python -m pathbench.cli.optimize            --config optimize.yaml
   python -m pathbench.cli.inference           --model_path ckpt --input h5 --output json
   python -m pathbench.cli.feature_extraction_slide \
     --config features.yaml --dataset TrainingSet --input /path/to/slide.svs
   python -m pathbench.cli.tiles_report        --config features.yaml

``pathbench-features``
-----------------------

Extract tile features from all configured datasets and combinations.

.. code-block:: text

   usage: pathbench-features [-h] --config CONFIG [--log-level {DEBUG,INFO,WARNING,ERROR}]

   options:
     --config CONFIG        Path to YAML config file.
     --log-level LEVEL      Logging verbosity (default: INFO).

Example:

.. code-block:: bash

   pathbench-features --config features.yaml --log-level DEBUG

``pathbench-benchmark``
------------------------

Run grid-search benchmarking over model/loss/extractor combinations.

.. code-block:: text

   usage: pathbench-benchmark [-h] --config CONFIG [--log-level {DEBUG,INFO,WARNING,ERROR}]

Example:

.. code-block:: bash

   pathbench-benchmark --config benchmark.yaml

``pathbench-optimize``
-----------------------

Run an Optuna hyperparameter optimization study.

.. code-block:: text

   usage: pathbench-optimize [-h] --config CONFIG [--log-level {DEBUG,INFO,WARNING,ERROR}]

Example:

.. code-block:: bash

   pathbench-optimize --config optimize.yaml

``pathbench-infer``
--------------------

Run inference with a trained checkpoint and optionally generate heatmaps.

.. code-block:: text

   usage: pathbench-infer [-h]
                          --model_path MODEL_PATH
                          --input INPUT
                          --output OUTPUT
                          [--heatmap-backend {native,torchmil}]
                          [--bag-id BAG_ID]
                          [--scores SCORES]
                          [--coords COORDS]
                          [--mask MASK]
                          [--heatmap-name HEATMAP_NAME]
                          [--heatmap-output HEATMAP_OUTPUT]

   options:
     --model_path           Path to .ckpt checkpoint file.
     --input                Path to slide H5 artifact.
     --output               Output JSON for predictions.
     --heatmap-backend      Heatmap explainer backend (default: native).
     --bag-id               Bag identifier, e.g. "256px_0.5mpp".
     --scores               Per-instance scores (.npy / .npz / .json).
     --coords               Per-instance coordinates (.npy shaped [N,2]).
     --mask                 Binary mask (.npy shaped [N]). False = exclude.
     --heatmap-name         H5 namespace for storing this heatmap.
     --heatmap-output       Optional JSON sidecar for the heatmap.

Example (prediction only):

.. code-block:: bash

   pathbench-infer \
     --model_path checkpoints/best.ckpt \
     --input artifacts/SLIDE_001.h5 \
     --output predictions/SLIDE_001.json

Example (with heatmap):

.. code-block:: bash

   pathbench-infer \
     --model_path checkpoints/best.ckpt \
     --input artifacts/SLIDE_001.h5 \
     --output predictions/SLIDE_001.json \
     --heatmap-backend torchmil \
     --bag-id 256px_0.5mpp \
     --scores scores/SLIDE_001.npy \
     --heatmap-name abmil_attention \
     --heatmap-output predictions/SLIDE_001_heatmap.json

``feature_extraction_slide`` (module only)
------------------------------------------

Extract features for a single slide. Designed for SLURM array jobs.

.. code-block:: text

   usage: python -m pathbench.cli.feature_extraction_slide
          --config CONFIG
          --dataset DATASET
          --input INPUT
          [--log-level LEVEL]

   options:
     --config    Path to YAML config file.
     --dataset   Dataset name matching datasets[].name in config.
     --input     Path to the single WSI file to process.

Example:

.. code-block:: bash

   python -m pathbench.cli.feature_extraction_slide \
     --config features.yaml \
     --dataset TrainingSet \
     --input /data/slides/train/TCGA-A1-A0SB-01Z.svs \
     --log-level INFO

``tiles_report`` (module only)
--------------------------------

Generate PDF tile overview reports from stored H5 tile overview images.

.. code-block:: text

   usage: python -m pathbench.cli.tiles_report
          --config CONFIG
          [--log-level LEVEL]

Example:

.. code-block:: bash

   python -m pathbench.cli.tiles_report --config features.yaml

Logging
-------

All commands support ``--log-level``:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Level
     - Output
   * - ``DEBUG``
     - Per-slide and per-tile progress, all registry lookups, timing.
   * - ``INFO``
     - **(Default)** Per-combination and per-dataset summary.
   * - ``WARNING``
     - Only warnings and errors.
   * - ``ERROR``
     - Only errors.

Tips
----

- Use ``--log-level DEBUG`` when diagnosing slide loading or segmentation issues.
- Use the module form (``python -m pathbench.cli.benchmark``) during development to
  see which file is actually executing.
- Config validation runs before any work starts — use a dummy run to check a
  new config without processing slides:

  .. code-block:: bash

     pathbench-features --config features.yaml --log-level DEBUG 2>&1 | head -30
