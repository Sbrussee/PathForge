Tutorial: Benchmarking
======================

Benchmarking evaluates every combination from ``benchmark_parameters`` and
reports per-combination metrics. It requires pre-computed H5 features from the
:doc:`feature_extraction` step.

What You Need
-------------

- H5 artifacts from feature extraction.
- The ``mil-backends`` extra for TorchMetrics/TorchSurv.

Step 1 — Write the Config
--------------------------

**Native backend (classification)**

Save as ``benchmark.yaml``:

.. code-block:: yaml

   experiment:
     project_name: luad_benchmark
     annotation_file: /data/annotations.csv
     project_root: /data/pathbench_projects
     mode: benchmark
     task: classification
     num_workers: 4
     mixed_precision: true

   mil:
     backend: native
     lr: 0.0001
     weight_decay: 0.00001
     batch_size: 1
     epochs: 30
     patience: 10
     optimizer: Adam
     bag_size: 512
     z_dim: 256
     dropout_p: 0.1

   metrics:
     classification_backend: torchmetrics
     classification_metrics: [accuracy, balanced_accuracy, auroc, f1]

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
     tile_px: [224, 256]
     tile_mpp: [0.5, 1.0]
     feature_extraction: [resnet50, uni]
     mil: [AttentionMIL, TransMIL]
     loss: [CrossEntropyLoss, FocalLoss]
     activation_function: [ReLU]
     optimizer: [Adam, AdamW]
     epochs: [20, 40]
     batch_size: [1, 4]
     lr: [1e-4, 5e-4]
     weight_decay: [1e-5, 1e-4]
     dropout_p: [0.1, 0.3]
     z_dim: [128, 256]
     bag_size: [256, 512]

Step 2 — Run Benchmarking
--------------------------

.. code-block:: bash

   pathbench-benchmark --config benchmark.yaml

For each combination PathBench:

1. Builds the bag dataset from H5 features.
2. Resolves the MIL model and loss via registries.
3. Instantiates a ``LightningTrainer``.
4. Applies sampled training and architecture settings such as ``epochs``, ``lr``, ``dropout_p``, and ``z_dim``.
5. Trains for up to ``epochs`` with early stopping.
6. Evaluates on the test set and logs metrics.

Outputs are written to ``project_root/luad_benchmark/``.

Benchmarking with TorchMIL
---------------------------

To benchmark TorchMIL models:

.. code-block:: yaml

   experiment:
     mode: benchmark
     task: classification

   mil:
     backend: torchmil
     torchmil_model: ABMIL
     torchmil_model_kwargs:
       in_shape: [2048]
       out_shape: 2
     use_torchmil_collate: true
     batch_size: 4
     epochs: 30

   metrics:
     classification_backend: torchmetrics

   benchmark_parameters:
     feature_extraction: [resnet50]
     mil: [torchmil]
     loss: [CrossEntropyLoss]

The registry key in ``benchmark_parameters.mil`` is always ``torchmil`` —
the actual TorchMIL class is specified in ``mil.torchmil_model``.

To compare multiple TorchMIL models, use separate config files (model
constructor kwargs differ per class).

Survival Benchmarking
----------------------

For continuous survival analysis:

.. code-block:: yaml

   experiment:
     task: survival
     label_column: vital_status
     survival_time_column: days_to_death
     survival_event_column: vital_status_binary

   mil:
     backend: torchmil
     torchmil_model: SurvivalABMIL
     torchmil_model_kwargs:
       in_shape: [1024]
       out_shape: 1

   metrics:
     survival_continuous_backend: torchsurv
     survival_metrics: [c_index, td_auc]

   benchmark_parameters:
     mil: [torchmil]
     loss: [CoxPHLoss]

For discrete survival:

.. code-block:: yaml

   experiment:
     task: survival_discrete

   benchmark_parameters:
     loss: [NLLSurvLoss]

The annotation CSV must contain valid survival time and event columns. The
event column must be binary (0 = censored, 1 = observed event).

Multi-Extractor Benchmark
--------------------------

Compare multiple feature extractors in a single run:

.. code-block:: yaml

   benchmark_parameters:
     tile_px: [224, 256]
     tile_mpp: [0.5]
     feature_extraction: [resnet50, uni, conch]
     mil: [AttentionMIL]
     loss: [CrossEntropyLoss]

This generates ``2 × 1 × 2 × 1 × 1 = 4`` pipeline combinations before any
training-hyperparameter grids are multiplied in. H5 artifacts must
already exist for all combinations (run feature extraction first with the
same tile_px/tile_mpp/feature_extraction values).

Results
-------

PathBench writes an experiment-wide ranked summary CSV plus benchmark-level
visualizations under the experiment root:

.. code-block:: text

   project_root/luad_benchmark/
   ├── benchmark_results.csv
   ├── benchmark_visualizations/
   │   ├── benchmark_performance_ranked.html
   │   └── benchmark_rank_scatter.html
   └── checkpoints/
       └── ...

``benchmark_results.csv`` contains one row per configuration run with:

- active pipeline choices such as ``tile_px``, ``tile_mpp``,
  ``feature_extraction``, ``model``, and ``loss``
- sampled training settings such as ``batch_size``, ``epochs``, ``lr``,
  ``weight_decay``, ``dropout_p``, and ``z_dim``
- run outputs such as ``status``, ``objective_metric``, ``objective_value``,
  ``checkpoint_path``, and ``rank``

The rows are sorted by performance. Metrics such as accuracy, AUROC, and
``c_index`` are ranked highest-first, while loss/error metrics are ranked
lowest-first. The HTML visualizations are generated from this same CSV so the
experiment-wide ranking remains auditable from a single artifact.

Slide Retrieval
---------------

Slide retrieval is a separate task type. See :doc:`slide_retrieval` for the
dedicated tutorial. In brief, use ``task: slide_retrieval`` with
``benchmark_parameters.retrieval_representation`` and
``benchmark_parameters.search_strategy`` instead of ``mil`` and ``loss``.
