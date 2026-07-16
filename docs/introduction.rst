Introduction
============

What PathForge Is
-----------------

PathForge is a benchmarking and pipeline-optimization framework for multiple
instance learning (MIL) in computational pathology. It is designed to answer a
practical question: *which complete MIL pipeline configuration works best for
my pathology task and data?*

Rather than treating the model architecture as the only experimental choice,
PathForge represents the workflow from whole-slide images to evaluated results
as a configurable pipeline. A study can compare choices such as tile size and
resolution, stain normalization, feature extractor, MIL architecture, loss,
optimizer, and training hyperparameters. PathForge records the resulting
combinations, metrics, artifacts, checkpoints, and visualizations in a
consistent experiment structure.

What Sets It Apart
------------------

PathForge focuses on reproducible comparison across the *entire* MIL pipeline:

- **Benchmarking** evaluates an explicit grid of pipeline configurations under
  the same data, split, training, and reporting contracts.
- **Pipeline optimization** uses Optuna to search component choices and
  training parameters instead of limiting optimization to a model's learning
  rate or architecture parameters.
- **Backend catalogs** expose native PathForge, TorchMIL, and MIL-Lab models
  through one configuration interface. Optional metric and survival adapters
  integrate TorchMetrics and TorchSurv.
- **Reusable artifacts** separate WSI processing and feature extraction from
  downstream experiments, so compatible feature bags can be benchmarked,
  optimized, evaluated, retrieved, and used for inference without repeating
  slide processing.
- **Comparable outputs** provide task-aware metrics, visualizations, summaries,
  and packaged models through shared workflow conventions.

The purpose is not to claim that one MIL method is universally best. PathForge
helps researchers and practitioners efficiently identify strong, reproducible
pipeline configurations for a particular computational pathology cohort and
prediction objective.

Supported Tasks and Use Cases
-----------------------------

Classification
~~~~~~~~~~~~~~

Classification predicts a categorical label, such as disease subtype,
mutation status, or treatment-response group. PathForge supports binary and
multiclass MIL experiments and reports classification metrics such as
accuracy, balanced accuracy, F1, AUROC, and precision-recall AUC.

Regression
~~~~~~~~~~

Regression predicts a continuous target, such as a biomarker value or
quantitative tissue property. The current configurable regression metric
registry exposes MAE and MSE.

Survival Analysis
~~~~~~~~~~~~~~~~~

PathForge distinguishes two survival formulations:

- **Continuous survival** predicts a continuous risk or log-hazard score from
  an observed time and event indicator. Optional TorchSurv integration provides
  continuous-survival losses and metrics.
- **Discrete survival** divides follow-up time into intervals and predicts a
  conditional survival or hazard distribution over those intervals.

Both formulations account for censored observations, but they require
different model outputs and losses. The survival time and event columns are
declared explicitly in the experiment configuration.

Slide Retrieval
~~~~~~~~~~~~~~~

Slide retrieval finds reference slides that are most similar to a query slide.
It reuses stored feature bags and supports configurable representation and
search strategies, retrieval metrics, ranked-result tables, and retrieval
visualizations. This is useful for cohort exploration, case-based search, and
content-based pathology retrieval without fitting a predictive MIL model.
Slide retrieval is supported through benchmarking and inference; the current
Optuna optimization policy trains MIL prediction models and does not optimize
retrieval strategies.

Where to Go Next
----------------

- Follow :doc:`installation` and :doc:`data_preparation` to prepare the
  environment, slides, and annotation table.
- Use the :doc:`tutorials/end_to_end` tutorial for a complete classification
  workflow from WSIs to packaged-model inference.
- See :doc:`mil_options` for the pipeline components that can be benchmarked or
  optimized and :doc:`configuration` for the full YAML schema.
- See :doc:`task_outputs` for metrics, visualizations, and artifacts by task.
- Read :doc:`backends` for native, TorchMIL, MIL-Lab, metric, survival, and WSI
  backend details.
- Use :doc:`tutorials/slide_retrieval` for the retrieval workflow and
  :doc:`api/index` when integrating PathForge from Python.
