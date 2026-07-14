End-to-End Classification Tutorial
==================================

This tutorial takes a small classification cohort from whole-slide images to
tile features, a trained MIL model, saved metrics, and a prediction. Commands
are run from the PathForge repository root. Replace paths below with absolute
paths on your machine; PathForge intentionally does not guess data locations.

Prerequisites
-------------

Install PathForge with its slide-processing and MIL backends:

.. code-block:: bash

   uv sync --extra lazyslide --extra mil-backends

Use at least four slides from two classes. Two slides are enough to exercise
the commands, but are not enough for a meaningful train/validation result.
This tutorial assumes this layout:

.. code-block:: text

   /data/pathforge_tutorial/
   ├── annotations.csv
   ├── slides/
   │   ├── CASE_001.svs
   │   ├── CASE_002.svs
   │   ├── CONTROL_001.svs
   │   └── CONTROL_002.svs
   └── artifacts/

Create ``annotations.csv`` with one row per slide. The ``slide`` value is the
filename stem, and every ``dataset`` value must match the config below.

.. code-block:: text

   dataset,slide,patient,category
   Tutorial,CASE_001,PATIENT_001,case
   Tutorial,CASE_002,PATIENT_002,case
   Tutorial,CONTROL_001,PATIENT_003,control
   Tutorial,CONTROL_002,PATIENT_004,control

See :doc:`/data_preparation` for supported formats, explicit ``wsi_path``
values, DICOM inputs, and MPP fallback metadata.

1. Extract features
-------------------

Save the following as ``tutorial-features.yaml``. ``project_root`` controls
where run metadata is written; slide artifacts are written to
``datasets[].artifacts_dir``.

.. code-block:: yaml

   experiment:
     project_name: pathforge_tutorial_features
     project_root: /data/pathforge_tutorial/runs
     annotation_file: /data/pathforge_tutorial/annotations.csv
     mode: feature_extraction
     task: classification
     num_workers: 0

   slide_processing:
     backend: lazyslide
     segmentation_method: otsu
     save_tiles: false

   datasets:
     - name: Tutorial
       slides_dir: /data/pathforge_tutorial/slides
       artifacts_dir: /data/pathforge_tutorial/artifacts
       tissue_annotations_dir: null
       used_for: training

   benchmark_parameters:
     tile_px: [256]
     tile_mpp: [0.5]
     feature_extraction: [resnet18]
     mil: []

Validate the command surface, then extract:

.. code-block:: bash

   uv run pathforge features run --help
   uv run pathforge features run --config tutorial-features.yaml

After completion, each slide has an H5 file in
``/data/pathforge_tutorial/artifacts``. The feature bag identifier is
``256px_0.5mpp__resnet18``; see :doc:`/HDF5_structure` for the stored arrays.

2. Train and evaluate MIL
-------------------------

Copy the feature config to ``tutorial-benchmark.yaml`` and make these changes:

.. code-block:: yaml

   experiment:
     project_name: pathforge_tutorial_benchmark
     project_root: /data/pathforge_tutorial/runs
     annotation_file: /data/pathforge_tutorial/annotations.csv
     mode: benchmark
     task: classification
     aggregation_level: slide
     num_workers: 0

   classification:
     split_technique: fixed
     val_fraction: 0.25
     balancing: none
     class_weighting: false
     epochs: 2
     batch_size: 1
     bag_size: 1000
     best_epoch_based_on: val_loss

   evaluation:
     label_column: category
     metrics: []
     visualization: []

   datasets:
     - name: Tutorial
       slides_dir: /data/pathforge_tutorial/slides
       artifacts_dir: /data/pathforge_tutorial/artifacts
       tissue_annotations_dir: null
       used_for: training

   benchmark_parameters:
     tile_px: [256]
     tile_mpp: [0.5]
     feature_extraction: [resnet18]
     mil: [AttentionMIL]
     loss: [CrossEntropyLoss]
     activation_function: [ReLU]
     optimizer: [Adam]

The two epochs keep this walkthrough short; increase them for real training.
The empty metric list keeps the minimal native-backend example independent of
optional metric adapters; after installing ``mil-backends``, add registered
classification metrics such as ``balanced_accuracy`` and ``f1``.
Run the benchmark and inspect its summary:

.. code-block:: bash

   uv run pathforge benchmark run --config tutorial-benchmark.yaml
   test -s /data/pathforge_tutorial/runs/pathforge_tutorial_benchmark/benchmark_results.csv

``benchmark_results.csv`` records the status, metrics, and checkpoint path for
each combination. Per-run checkpoints, packaged models, and training artifacts
are stored below the same experiment directory. A failed combination is
recorded in the summary; do not treat the presence of the CSV alone as a
successful run.

3. Re-evaluate saved predictions
--------------------------------

Evaluation and visualization can be repeated without retraining:

.. code-block:: bash

   uv run pathforge evaluate run --config tutorial-benchmark.yaml
   uv run pathforge visualize run --config tutorial-benchmark.yaml

With ``visualization: []`` the second command is intentionally a no-op. Add a
task-compatible renderer from :doc:`benchmarking` when plots are required.

4. Run inference
----------------

Read the successful row in ``benchmark_results.csv`` and locate its
``checkpoint_path``. Training writes the package next to that checkpoint as
``<checkpoint_stem>_package.pt``. A packaged model contains the construction
metadata needed by the inference command; a bare Lightning checkpoint may need
additional arguments.

.. code-block:: bash

   uv run pathforge-infer-model \
     --model_path /absolute/checkpoint/directory/<checkpoint_stem>_package.pt \
     --input /data/pathforge_tutorial/artifacts/CASE_001.h5 \
     --output /data/pathforge_tutorial/predictions/CASE_001.json

Verify the output and inspect it:

.. code-block:: bash

   test -s /data/pathforge_tutorial/predictions/CASE_001.json
   python -m json.tool /data/pathforge_tutorial/predictions/CASE_001.json

You have now exercised the complete data path: WSI and annotation input,
segmentation and tiling, feature persistence, MIL training and evaluation,
model packaging, and inference. For attention heatmaps, continue with
:doc:`inference`; for larger cohorts and split design, continue with
:doc:`benchmarking`.

Troubleshooting checkpoints
---------------------------

* If feature extraction produces no patches, inspect tissue segmentation and
  slide MPP metadata before changing the model.
* If training cannot find a bag, confirm the exact identifier
  ``256px_0.5mpp__resnet18`` exists in every slide artifact.
* If a class is absent from validation, add slides or use a split strategy
  appropriate for the cohort. Never use the tiny tutorial split for reporting
  model performance.
* Use :doc:`/troubleshooting` for optional-backend and GPU issues.
