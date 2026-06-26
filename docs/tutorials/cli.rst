Tutorial: Command-Line Interface
================================

All PathForge workflows are driven by CLI commands. Every command accepts a
``--config`` flag pointing to a YAML file and an optional ``--log-level`` flag.

Entry Points
------------

PathForge installs a unified ``pathforge`` command plus flat console scripts.
Every workflow command takes ``--config`` and an optional ``--log-level``.

Unified ``pathforge`` command (run ``pathforge --help`` for the full list):

.. code-block:: bash

   pathforge features run    --config features.yaml
   pathforge features slide  --config features.yaml --dataset TrainingSet --input /path/to/slide.svs
   pathforge benchmark run   --config benchmark.yaml
   pathforge evaluate run    --config benchmark.yaml
   pathforge visualize run   --config benchmark.yaml
   pathforge optimize run    --config optimize.yaml
   pathforge report tiles    --config features.yaml
   pathforge retrieval representations --config retrieval.yaml
   pathforge retrieval mean-rgb        --config retrieval.yaml --dataset ReferenceSet --slide-id SLIDE_001
   pathforge retrieval sish-vqvae      --config retrieval.yaml
   pathforge infer run       --config inference.yaml --input-csv slides.csv

Flat console scripts (shortcuts for the common workflows):

.. code-block:: bash

   pathforge-features    --config features.yaml
   pathforge-benchmark   --config benchmark.yaml
   pathforge-evaluate    --config benchmark.yaml
   pathforge-optimize    --config optimize.yaml
   pathforge-visualize   --config benchmark.yaml
   pathforge-mean-rgb    --config retrieval.yaml --dataset ReferenceSet --slide-id SLIDE_001
   pathforge-slide-retrieval-representations --config retrieval.yaml
   pathforge-infer       --config inference.yaml --input-csv slides.csv
   pathforge-infer-model --model_path best.ckpt --input artifact.h5 --output predictions.json

``pathforge-features``
-----------------------

Extract tile features from all configured datasets and combinations.

.. code-block:: text

   usage: pathforge-features [-h] --config CONFIG [--log-level {DEBUG,INFO,WARNING,ERROR}]

   options:
     --config CONFIG        Path to YAML config file.
     --log-level LEVEL      Logging verbosity (default: INFO).

Example:

.. code-block:: bash

   pathforge-features --config features.yaml --log-level DEBUG

``pathforge-benchmark``
------------------------

Run grid-search benchmarking over model/loss/extractor combinations.

.. code-block:: text

   usage: pathforge-benchmark [-h] --config CONFIG [--log-level {DEBUG,INFO,WARNING,ERROR}]

Example:

.. code-block:: bash

   pathforge-benchmark --config benchmark.yaml

``pathforge-optimize``
-----------------------

Run an Optuna hyperparameter optimization study.

.. code-block:: text

   usage: pathforge-optimize [-h] --config CONFIG [--log-level {DEBUG,INFO,WARNING,ERROR}]

Example:

.. code-block:: bash

   pathforge-optimize --config optimize.yaml

``pathforge-evaluate``
----------------------

Run post-hoc metrics and visualization workflows from the ``evaluation`` block.

.. code-block:: text

   usage: pathforge-evaluate [-h] --config CONFIG [--log-level {DEBUG,INFO,WARNING,ERROR}]

Example:

.. code-block:: bash

   pathforge-evaluate --config benchmark.yaml

``pathforge-mean-rgb``
----------------------

Precompute patch mean RGB descriptors used by RGB-based slide retrieval representations.

.. code-block:: text

   usage: pathforge-mean-rgb [-h] --config CONFIG --dataset DATASET --slide-id SLIDE_ID
                             [--input INPUT] [--bag-id BAG_ID] [--artifact-path ARTIFACT_PATH]
                             [--log-level {DEBUG,INFO,WARNING,ERROR}]

Example:

.. code-block:: bash

   pathforge-mean-rgb \
     --config retrieval.yaml \
     --dataset ReferenceSet \
     --slide-id SLIDE_001 \
     --bag-id 256px_0.5mpp

``pathforge-infer``
--------------------

Run config-driven inference: load a project in ``experiment.mode='inference'``
and score the slides selected by an input CSV.

.. code-block:: text

   usage: pathforge-infer [-h] --config CONFIG --input-csv INPUT_CSV
                          [--log-level {DEBUG,INFO,WARNING,ERROR}]

   options:
     --config       Path to YAML config (experiment.mode must be 'inference').
     --input-csv    CSV selecting the slides to run inference for.

Example:

.. code-block:: bash

   pathforge-infer --config inference.yaml --input-csv slides.csv

``pathforge-infer-model``
--------------------------

Run a packaged checkpoint on a single feature artifact and optionally attach a
heatmap. Use this for ad-hoc prediction/explainability outside the config flow.

.. code-block:: text

   usage: pathforge-infer-model [-h]
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
     --model_path           Path to packaged model checkpoint.
     --input                Feature artifact (.h5 / .pt / .npy / .npz).
     --output               Output JSON for predictions.
     --heatmap-backend      Heatmap explainer backend (e.g. torchmil).
     --bag-id               Bag identifier, e.g. "256px_0.5mpp".
     --scores               Per-instance scores (.npy / .npz / .json).
     --coords               Per-instance coordinates (.npy shaped [N,2]).
     --mask                 Binary mask (.npy shaped [N]). False = exclude.
     --heatmap-name         H5 namespace for storing this heatmap.
     --heatmap-output       Optional JSON sidecar for the heatmap.

Example (prediction only):

.. code-block:: bash

   pathforge-infer-model \
     --model_path checkpoints/best.ckpt \
     --input artifacts/SLIDE_001.h5 \
     --output predictions/SLIDE_001.json

Example (with heatmap):

.. code-block:: bash

   pathforge-infer-model \
     --model_path checkpoints/best.ckpt \
     --input artifacts/SLIDE_001.h5 \
     --output predictions/SLIDE_001.json \
     --heatmap-backend torchmil \
     --bag-id 256px_0.5mpp \
     --scores scores/SLIDE_001.npy \
     --heatmap-name abmil_attention \
     --heatmap-output predictions/SLIDE_001_heatmap.json

``pathforge-visualize``
-----------------------

Run visualization workflows configured in the ``evaluation`` block.

.. code-block:: text

   usage: pathforge-visualize [-h] --config CONFIG [--log-level {DEBUG,INFO,WARNING,ERROR}]

Example:

.. code-block:: bash

   pathforge-visualize --config benchmark.yaml

``pathforge-slide-retrieval-representations``
---------------------------------------------

Materialize slide retrieval representations without running the full benchmark stage.

.. code-block:: text

   usage: pathforge-slide-retrieval-representations [-h] --config CONFIG
                                                    [--log-level {DEBUG,INFO,WARNING,ERROR}]

Example:

.. code-block:: bash

   pathforge-slide-retrieval-representations --config retrieval.yaml

``pathforge features slide``
----------------------------

Extract features for a single slide. Designed for SLURM array jobs.

.. code-block:: text

   usage: pathforge features slide
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

   pathforge features slide \
     --config features.yaml \
     --dataset TrainingSet \
     --input /data/slides/train/TCGA-A1-A0SB-01Z.svs \
     --log-level INFO

``pathforge report tiles``
--------------------------

Generate PDF tile overview reports from stored H5 tile overview images.

.. code-block:: text

   usage: pathforge report tiles
          --config CONFIG
          [--log-level LEVEL]

Example:

.. code-block:: bash

   pathforge report tiles --config features.yaml

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
- Run ``pathforge --help`` or ``pathforge <group> --help`` to discover every
  command and its flags.
- Config validation runs before any work starts — use a dummy run to check a
  new config without processing slides:

  .. code-block:: bash

     pathforge-features --config features.yaml --log-level DEBUG 2>&1 | head -30
