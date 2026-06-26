Quick Start
===========

This page shows the minimum working examples for each major workflow.

Annotation CSV
--------------

All workflows need an annotation CSV. At minimum:

.. code-block:: text

   dataset,slide,patient,category
   TrainingSet,SLIDE_001,PATIENT_001,case
   TrainingSet,SLIDE_002,PATIENT_002,control
   TestSet,SLIDE_003,PATIENT_003,case

- ``dataset`` must match a name in ``datasets[].name``.
- ``slide`` is matched against files in ``slides_dir`` using ``{slide}.*``.
- Supported suffixes: ``.svs``, ``.ndpi``, ``.tiff``, ``.tif``, ``.mrxs``.

See :doc:`data_preparation` for the complete column reference, per-task
annotation examples, slide naming rules, and a validation checklist.

Feature Extraction
------------------

**Config** (``features.yaml``):

.. code-block:: yaml

   experiment:
     project_name: my_features
     annotation_file: /data/annotations.csv
     mode: feature_extraction

   slide_processing:
     backend: lazyslide
     segmentation_method: otsu

   datasets:
     - name: TrainingSet
       slides_dir: /data/slides/train
       artifacts_dir: /data/artifacts/train
       used_for: training

   benchmark_parameters:
     tile_px: [256]
     tile_mpp: [0.5]
     feature_extraction: [resnet18]
     mil: []

**Run:**

.. code-block:: bash

   pathbench-features --config features.yaml

Benchmarking
------------

**Config** (``benchmark.yaml``):

.. code-block:: yaml

   experiment:
     project_name: my_benchmark
     annotation_file: /data/annotations.csv
     mode: benchmark
     task: classification

   mil:
     backend: native
     lr: 0.0001
     batch_size: 1
     epochs: 20

   metrics:
     classification_backend: torchmetrics

   datasets:
     - name: TrainingSet
       slides_dir: /data/slides/train
       artifacts_dir: /data/artifacts/train
       used_for: training
     - name: TestSet
       slides_dir: /data/slides/test
       artifacts_dir: /data/artifacts/test
       used_for: testing

   benchmark_parameters:
     tile_px: [256]
     tile_mpp: [0.5]
     feature_extraction: [resnet18]
     mil: [AttentionMIL]
     loss: [CrossEntropyLoss]

**Run:**

.. code-block:: bash

   pathbench-benchmark --config benchmark.yaml

Hyperparameter Optimization
----------------------------

**Config** (``optimize.yaml``):

.. code-block:: yaml

   experiment:
     project_name: my_optimization
     annotation_file: /data/annotations.csv
     mode: optimization
     task: classification

   mil:
     backend: native
     epochs: 20

   metrics:
     classification_backend: torchmetrics

   datasets:
     - name: TrainingSet
       slides_dir: /data/slides/train
       artifacts_dir: /data/artifacts/train
       used_for: training

   optimization:
     study_name: my_study
     objective_metric: val_loss
     objective_mode: min
     sampler: TPESampler
     pruner: HyperbandPruner
     trials: 30

   benchmark_parameters:
     tile_px: [256]
     tile_mpp: [0.5]
     feature_extraction: [resnet18]
     mil: [AttentionMIL]
     loss: [CrossEntropyLoss]

**Run:**

.. code-block:: bash

   pathbench-optimize --config optimize.yaml

Inference
---------

Run inference with a saved checkpoint:

.. code-block:: bash

   pathbench-infer \
     --model_path /experiments/my_benchmark/checkpoints/best.ckpt \
     --input /data/artifacts/train/SLIDE_001.h5 \
     --output predictions.json

Generate attention heatmap alongside inference:

.. code-block:: bash

   pathbench-infer \
     --model_path /experiments/my_benchmark/checkpoints/best.ckpt \
     --input /data/artifacts/train/SLIDE_001.h5 \
     --output predictions.json \
     --heatmap-backend torchmil \
     --bag-id 256px_0.5mpp \
     --scores attention_scores.npy \
     --heatmap-name abmil_attention \
     --heatmap-output heatmap.json
