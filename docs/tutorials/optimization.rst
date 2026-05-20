Tutorial: Hyperparameter Optimization
======================================

Optimization mode runs an `Optuna <https://optuna.readthedocs.io>`_ study to
search across both pipeline-component choices and training hyperparameters
while using the same registry boundaries as benchmarking. It requires
pre-computed H5 features.

What You Need
-------------

- H5 artifacts from feature extraction.
- ``mil-backends`` extra for TorchMetrics/TorchSurv (recommended).

Step 1 — Write the Config
--------------------------

Save as ``optimize.yaml``:

.. code-block:: yaml

   experiment:
     project_name: luad_optimization
     annotation_file: /data/annotations.csv
     project_root: /data/pathbench_projects
     mode: optimization
     task: classification
     num_workers: 4

   mil:
     backend: native
     epochs: 30
     patience: 10

   metrics:
     classification_backend: torchmetrics
     classification_metrics: [auroc, balanced_accuracy]

   datasets:
     - name: TrainingSet
       slides_dir: /data/slides/train
       artifacts_dir: /data/artifacts/train
       used_for: training
     - name: TestSet
       slides_dir: /data/slides/test
       artifacts_dir: /data/artifacts/test
       used_for: testing

   optimization:
     study_name: luad_abmil_search
     objective_metric: val_auroc
     objective_mode: max
     sampler: TPESampler
     pruner: HyperbandPruner
     trials: 50
     search_space:
       lr: {type: float, low: 1e-5, high: 1e-3, log: true}
       weight_decay: {type: float, low: 1e-6, high: 1e-3, log: true}
       dropout_p: {type: float, low: 0.0, high: 0.5}
       epochs: {type: int, low: 10, high: 50, step: 5}
       z_dim: {type: categorical, choices: [128, 256, 512]}
       bag_size: {type: categorical, choices: [256, 512, 1024]}

   benchmark_parameters:
     tile_px: [256, 512]
     tile_mpp: [0.5, 1.0]
     feature_extraction: [resnet50, uni]
     mil: [AttentionMIL, TransMIL]
     loss: [CrossEntropyLoss, FocalLoss]
     optimizer: [Adam, AdamW]

Step 2 — Run Optimization
--------------------------

.. code-block:: bash

   pathbench-optimize --config optimize.yaml

Each Optuna trial:

1. Samples numeric and explicit categorical hyperparameters from ``optimization.search_space``.
2. Samples pipeline-component choices from multi-valued ``benchmark_parameters`` lists.
3. Applies them to the active config via ``apply_search_params()``.
4. Trains the model using the Lightning trainer.
5. Reports the ``objective_metric`` to Optuna.
6. Prunes poor trials early via the configured pruner.

Optuna Study Settings
---------------------

``optimization.sampler``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Sampler
     - When to use
   * - ``TPESampler``
     - **Default.** Tree-structured Parzen Estimator. Best for most cases.
   * - ``RandomSampler``
     - Baseline random search. Use to verify TPE improvement.
   * - ``CmaEsSampler``
     - CMA-ES for continuous parameter spaces. Use with many float params.
   * - ``GridSampler``
     - Exhaustive grid search over a fixed space.

``optimization.pruner``
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Pruner
     - When to use
   * - ``HyperbandPruner``
     - **Default.** Hyperband successive halving. Most aggressive pruning.
   * - ``MedianPruner``
     - Prune trials below the median. Conservative.
   * - ``NopPruner``
     - No pruning. Use when comparing full training curves.

Search Space Types
------------------

Each entry in ``optimization.search_space`` maps a config parameter name to a
type spec. The documented key is ``type``; PathBench also accepts ``kind`` for
the same field.

.. code-block:: yaml

   search_space:
     lr:
       type: float
       low: 1e-5
       high: 1e-2
       log: true          # log-uniform sampling
     dropout_p:
       type: float
       low: 0.0
       high: 0.5
     z_dim:
       type: categorical
       choices: [128, 256, 512]
     epochs:
       type: int
       low: 10
       high: 50

Supported types: ``float``, ``int``, ``categorical``.

Pipeline Component Search
-------------------------

Optimization automatically treats every multi-valued list in
``benchmark_parameters`` as a categorical search dimension, except ``seeds``.
That includes pipeline components such as:

- ``tile_px``
- ``tile_mpp``
- ``feature_extraction``
- ``mil``
- ``loss``
- ``optimizer``

Benchmarking uses the same lists as a full grid, while optimization samples
from them trial-by-trial and combines them with ranged hyperparameters such as
``lr``, ``weight_decay``, ``dropout_p``, ``epochs``, ``z_dim``, and
``bag_size``.

TorchMIL Optimization
----------------------

Optimize TorchMIL model hyperparameters:

.. code-block:: yaml

   mil:
     backend: torchmil
     torchmil_model: ABMIL
     torchmil_model_kwargs:
       in_shape: [2048]
       out_shape: 2
     use_torchmil_collate: true
     batch_size: 4

   optimization:
     study_name: torchmil_abmil_search
     objective_metric: val_loss
     objective_mode: min
     sampler: TPESampler
     pruner: HyperbandPruner
     trials: 30
     search_space:
       lr: {type: float, low: 1e-5, high: 1e-3, log: true}
       dropout_p: {type: float, low: 0.0, high: 0.5}

   benchmark_parameters:
     mil: [torchmil]
     loss: [CrossEntropyLoss]

Resuming a Study
----------------

Optuna persists study state in a SQLite database under the experiment
directory. Rerun the same command to resume:

.. code-block:: bash

   pathbench-optimize --config optimize.yaml

Optuna will detect the existing study by ``study_name`` and continue from
where it left off.

Inspecting Results
------------------

PathBench writes both raw and ranked optimization summaries under the
experiment root:

.. code-block:: text

   project_root/luad_optimization/
   ├── luad_abmil_search_results.csv
   ├── optimization_results.csv
   └── optimization_visualizations/
       ├── plot_optimization_history.html
       ├── plot_param_importances.html
       ├── plot_rank.html
       ├── plot_timeline.html
       └── plot_hypervolume_history.html   # multi-objective studies only

``luad_abmil_search_results.csv`` is the raw ``study.trials_dataframe()``
export from Optuna. ``optimization_results.csv`` is the experiment-wide summary
used for ranking completed trials by the configured objective metric.

Optuna study results are also accessible via the Optuna API:

.. code-block:: python

   import optuna

   study = optuna.load_study(
       study_name="luad_abmil_search",
       storage="sqlite:////data/pathbench_projects/luad_optimization/study.db",
   )

   print(study.best_trial.params)
   print(study.best_value)

   # All trials as a DataFrame
   df = study.trials_dataframe()

Visualization notes
-------------------

PathBench exports the Plotly-backed Optuna visualizations documented in the
current Optuna API:

- ``plot_optimization_history``
- ``plot_param_importances``
- ``plot_rank``
- ``plot_timeline``
- ``plot_hypervolume_history`` for multi-objective studies

The current Optuna documentation states that
``plot_param_importances`` requires ``scikit-learn`` and that
``plot_hypervolume_history`` requires a study with at least two objectives plus
an explicit reference point. PathBench therefore skips only the figures whose
backend requirements are not met while still writing the summary CSVs and the
remaining visualization files.
