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
- ``slide`` must exactly match a supported file stem in ``slides_dir`` (for
  example, ``SLIDE_001`` resolves to ``SLIDE_001.svs``). A DICOM slide may
  instead use an exact-name directory containing ``.dcm`` files.
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

   pathforge-features --config features.yaml

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

   pathforge-benchmark --config benchmark.yaml

Pipeline Optimization
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

   pathforge-optimize --config optimize.yaml

Inference
---------

Run a packaged checkpoint on one feature artifact:

.. code-block:: bash

   pathforge-infer-model \
     --model_path /experiments/my_benchmark/checkpoints/best.ckpt \
     --input /data/artifacts/train/SLIDE_001.h5 \
     --output predictions.json

Generate an attention heatmap alongside the prediction:

.. code-block:: bash

   pathforge-infer-model \
     --model_path /experiments/my_benchmark/checkpoints/best.ckpt \
     --input /data/artifacts/train/SLIDE_001.h5 \
     --output predictions.json \
     --heatmap-backend torchmil \
     --bag-id 256px_0.5mpp \
     --scores attention_scores.npy \
     --heatmap-name abmil_attention \
     --heatmap-output heatmap.json

For config-driven inference over many slides (``experiment.mode='inference'``),
use ``pathforge-infer --config inference.yaml --input-csv slides.csv``.

End-to-End Example
------------------

A typical classification project chains four commands. Reusing the configs above:

.. code-block:: bash

   # 1. Extract tile features into per-slide H5 artifacts.
   pathforge-features --config features.yaml

   # 2. Train + evaluate the MIL grid (writes a results CSV + visualizations).
   pathforge-benchmark --config benchmark.yaml

   # 3. (Optional) Re-run metrics/visualizations from saved predictions.
   pathforge-evaluate --config benchmark.yaml

   # 4. Predict on one slide with the best checkpoint.
   pathforge-infer-model \
     --model_path /experiments/my_benchmark/checkpoints/best.ckpt \
     --input /data/artifacts/train/SLIDE_001.h5 \
     --output predictions.json

The same steps work through the unified command, e.g.
``pathforge features run --config features.yaml`` and
``pathforge benchmark run --config benchmark.yaml``.

Slide Retrieval
---------------

Rank reference slides against query slides using bag-level representations and a
search strategy.

**Config** (``retrieval.yaml``):

.. code-block:: yaml

   experiment:
     project_name: my_retrieval
     annotation_file: /data/annotations.csv
     mode: benchmark
     task: slide_retrieval
     aggregation_level: slide

   slide_retrieval:
     exclusion_level: patient

   datasets:
     - name: ReferenceSet
       slides_dir: /data/slides/reference
       artifacts_dir: /data/artifacts/reference
       used_for: reference
     - name: QuerySet
       slides_dir: /data/slides/query
       artifacts_dir: /data/artifacts/query
       used_for: query

   benchmark_parameters:
     tile_px: [256]
     tile_mpp: [0.5]
     feature_extraction: [resnet18]
     retrieval_representation: [yottixel-features]
     search_strategy: [yottixel]

**Run** (precompute representations, then the retrieval benchmark):

.. code-block:: bash

   pathforge-slide-retrieval-representations --config retrieval.yaml
   pathforge-benchmark --config retrieval.yaml

Each combination writes a ranked ``query_results.xlsx`` and ``manifest.json``
under the project root; see :doc:`/slide-retrieval-results-and-metrics`.
