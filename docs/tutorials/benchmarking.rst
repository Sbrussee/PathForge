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
     project_root: /data/pathforge_projects
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
     mil: [PerceiverMIL, VarMIL]
     loss: [CrossEntropyLoss, NLLLoss]

Step 2 — Run Benchmarking
--------------------------

.. code-block:: bash

   pathforge-benchmark --config benchmark.yaml

For each combination PathForge:

1. Builds the bag dataset from H5 features.
2. Resolves the MIL model and loss via registries.
3. Instantiates a ``LightningTrainer``.
4. Uses the fixed training and architecture settings from the ``mil`` block.
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
     mil: [ABMIL, CLAM]
     loss: [CrossEntropyLoss]

``benchmark_parameters.mil`` accepts concrete available names from native
PathForge, TorchMIL, and MIL-Lab catalogs. The selected name determines the
backend automatically. Models can share a grid when their constructor kwargs
are compatible; otherwise use separate config files.

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
     torchmil_model_kwargs:
       in_shape: [1024]
       out_shape: 1

   metrics:
     survival_continuous_backend: torchsurv
     survival_metrics: [c_index, td_auc]

   benchmark_parameters:
     mil: [PerceiverMIL]
     loss: [CoxPHLoss]

For discrete survival:

.. code-block:: yaml

   experiment:
     task: survival_discrete

   benchmark_parameters:
     loss: [DiscreteTimeNLLLoss]

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
     mil: [PerceiverMIL]
     loss: [CrossEntropyLoss]

This generates ``2 × 1 × 3 × 1 × 1 = 6`` pipeline combinations. H5 artifacts must
already exist for all combinations (run feature extraction first with the
same tile_px/tile_mpp/feature_extraction values).

Results
-------

PathForge writes an experiment-wide ranked summary CSV plus benchmark-level
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
- fixed training settings from the ``mil`` block, such as ``batch_size``,
  ``epochs``, ``lr``, ``weight_decay``, ``dropout_p``, and ``z_dim``
- run outputs such as ``status``, ``objective_metric``, ``objective_value``,
  ``checkpoint_path``, and ``rank``

The rows are sorted by performance. Metrics such as accuracy, AUROC, and
``c_index`` are ranked highest-first, while loss/error metrics are ranked
lowest-first. The HTML visualizations are generated from this same CSV so the
experiment-wide ranking remains auditable from a single artifact.

Recreate global summary charts later, without training again:

.. code-block:: bash

   pathforge visualize summary \
     --input /data/pathforge_projects/luad_benchmark/benchmark_results.csv

Slide Retrieval
---------------

Slide retrieval is a separate task type. See :doc:`slide_retrieval` for the
dedicated tutorial. In brief, use ``task: slide_retrieval`` with
``benchmark_parameters.retrieval_representation`` and
``benchmark_parameters.search_strategy`` instead of ``mil`` and ``loss``.
