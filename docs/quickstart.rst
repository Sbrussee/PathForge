Quick Start
===========

This page shows the minimum configuration shape for each major workflow.
Replace every ``/data/...`` and ``/experiments/...`` example with an absolute
path that exists on the machine running PathForge.

How the Examples Fit Together
-----------------------------

Feature extraction produces the HDF5 slide artifacts consumed by
benchmarking, optimization, retrieval, and config-driven inference. The
examples use the same ``tile_px``, ``tile_mpp``, and feature extractor so that
downstream commands can reuse those artifacts.

.. code-block:: text

   annotations.csv + slides
            │
            ▼  feature extraction
      artifacts/*.h5
        ├──► benchmark ─► metrics + checkpoint + model package
        ├──► optimize  ─► trials + ranked pipeline results
        ├──► retrieval ─► ranked similar slides
        └──► inference ─► task-specific predictions

The snippets omit settings for which defaults are sufficient. Use
:doc:`configuration` for the full schema and :ref:`core-terms` for definitions
of tile, MPP, bag, artifact, task, combination, and trial.

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

This step reads each WSI belonging to ``TrainingSet``, identifies tissue,
creates tile coordinates, runs ``resnet18`` over the tiles, and writes one HDF5
file per slide beneath ``/data/artifacts/train``. It does not train a MIL model.

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

After completion, a file such as
``/data/artifacts/train/SLIDE_001.h5`` should exist. Re-running a compatible
configuration normally reuses already-complete feature data.

Benchmarking
------------

Benchmarking forms combinations from the active lists under
``benchmark_parameters``. This example contains one value on every active
axis, so it trains one ``PerceiverMIL`` model. Dataset roles come from
``used_for``; names such as ``TrainingSet`` do not assign a role by themselves.

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
     mil: [PerceiverMIL]
     loss: [CrossEntropyLoss]

**Run:**

.. code-block:: bash

   pathforge-benchmark --config benchmark.yaml

The policy can create a missing required feature matrix before training.
Successful Lightning runs write a best checkpoint, a self-contained
``*_package.pt`` inference package, validation metrics, and figures. See
:doc:`task_outputs` for the output layout.

Pipeline Optimization
----------------------------

Optimization uses Optuna. A **trial** is one sampled pipeline configuration;
it can complete, fail, or be stopped early by the pruner. ``trials: 30`` is the
maximum requested number of trial evaluations, not the number of epochs.

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
     mil: [PerceiverMIL]
     loss: [CrossEntropyLoss]

**Run:**

.. code-block:: bash

   pathforge-optimize --config optimize.yaml

This configuration fixes every pipeline component to one value but still uses
the schema's default numeric search space for learning rate, epochs, latent
dimension, dropout, and weight decay. Declare ``optimization.search_space``
explicitly to make those ranges visible and reproducible in the YAML, or to
change them; see :doc:`configuration`.

Inference
---------

Run a packaged model on one feature artifact. Benchmarking and optimization
write a ``*_package.pt`` model beside each successful Lightning checkpoint;
the raw ``.ckpt`` file is not a self-contained inference input.

The input HDF5 must contain the tiling and feature extractor recorded in the
package. The command writes the prediction to the requested JSON path and does
not modify the source WSI.

.. code-block:: bash

   pathforge-infer-model \
     --model_path /experiments/my_benchmark/checkpoints/best_package.pt \
     --input /data/artifacts/train/SLIDE_001.h5 \
     --output predictions.json

Generate an attention heatmap alongside the prediction:

.. code-block:: bash

   pathforge-infer-model \
     --model_path /experiments/my_benchmark/checkpoints/best_package.pt \
     --input /data/artifacts/train/SLIDE_001.h5 \
     --output predictions.json \
     --heatmap-backend torchmil \
     --bag-id 256px_0.5mpp \
     --scores attention_scores.npy \
     --heatmap-name abmil_attention \
     --heatmap-output heatmap.json

For config-driven inference over many slides (``experiment.mode='inference'``),
use ``pathforge-infer --config inference.yaml --input-csv slides.csv``.
That path creates a timestamped inference directory and delegates to the
configured task. See :doc:`tutorials/inference` for the input CSV contract.

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

   # 4. Predict on one slide with its packaged model.
   pathforge-infer-model \
     --model_path /experiments/my_benchmark/checkpoints/best_package.pt \
     --input /data/artifacts/train/SLIDE_001.h5 \
     --output predictions.json

The same steps work through the unified command, e.g.
``pathforge features run --config features.yaml`` and
``pathforge benchmark run --config benchmark.yaml``.

Slide Retrieval
---------------

Rank reference slides against query slides using bag-level representations and a
search strategy.

``reference`` slides form the searchable collection; ``query`` slides are
looked up against it. ``exclusion_level: patient`` removes candidates from the
same patient before ranking, preventing related slides from becoming trivial
matches.

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
The first command materializes reusable retrieval representations from feature
bags. The second applies the search strategy and records ranked matches.
