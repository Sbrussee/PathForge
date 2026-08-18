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

Why PathForge Exists
--------------------

A computational-pathology result depends on much more than the final neural
network. The slide resolution, tissue segmentation, tile size, stain handling,
feature extractor, sampling scheme, model, loss, optimizer, and evaluation
choices all influence the result. When those choices are implemented in
separate one-off scripts, comparisons become difficult to trust: two models
may silently receive different inputs, preprocessing may be repeated with
slightly different settings, and the connection between a result and the
configuration that produced it can be lost.

PathForge exists to make the *complete pipeline* the unit of comparison. A
pipeline is the ordered set of choices that turns raw whole-slide images into
an evaluated prediction or retrieval result. PathForge gives those choices a
validated configuration, applies the same data and reporting contracts to each
combination, and records the artifacts needed to reproduce or inspect the run.

The central idea is to separate expensive, reusable work from experiment-
specific work:

.. code-block:: text

   expensive and reusable                    experiment-specific

   WSI ─► tissue ─► tiles ─► feature bag ───► model/search ─► metrics
                              │
                              └── stored once in a per-slide HDF5 artifact

Tiling and feature extraction may take substantial time. Once a compatible
feature bag has been stored, many models, losses, retrieval methods, and
evaluation settings can consume it without reopening and reprocessing the WSI.
This both reduces computation and ensures that downstream methods are compared
on the same inputs.

Why Policies Exist
------------------

A **policy** is a use-case coordinator: it describes the ordered decisions and
steps required to complete a user workflow without embedding those decisions
in the command-line interface or in a model implementation. “Policy” here does
not mean an access-control rule or a learned decision policy.

For example, the benchmarking policy must:

#. expand the relevant configured combination axes;
#. group combinations that reuse the same feature bag;
#. create missing features;
#. construct datasets with the correct training and validation roles;
#. select the registered task, model, loss, and trainer;
#. execute each run; and
#. preserve successful, failed, and skipped outcomes in a comparable summary.

Those steps involve several subsystems, but none belongs inside a MIL model,
dataset, or CLI parser. Keeping them in a policy provides one workflow that can
be called from the CLI, tests, or distributed workers. It also keeps the
replaceable scientific components focused on their own contracts.

.. code-block:: text

   CLI / distributed worker
              │ asks for a use case
              ▼
           policy
      ┌───────┼─────────┐
      ▼       ▼         ▼
   datasets  task    artifact checks
              │
              ▼ resolves registered components
       model / loss / trainer / strategy

The four implemented policies have distinct responsibilities:

* **Feature extraction policy:** turns configured WSIs into reusable HDF5
  slide artifacts.
* **Benchmarking policy:** evaluates explicit pipeline combinations under one
  task and reporting contract.
* **Optimization policy:** lets Optuna sample supported choices and parameters,
  then evaluates trials through the same underlying machinery.
* **Inference policy:** prepares configured inputs and delegates prediction to
  the selected task.

Tasks complement policies. A policy owns the workflow that is common across
tasks; a **task** owns behavior specific to the objective, such as training a
classification MIL model or executing representation-and-search stages for
slide retrieval.

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

.. _core-terms:

Core Terms and Variables
------------------------

These definitions apply throughout the documentation and configuration
examples.

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Term or symbol
     - Meaning in PathForge
   * - **WSI / slide**
     - A whole-slide image: one supported image file, or one directory of DICOM
       files representing a slide. ``slide`` is also the default annotation
       column containing its identifier.
   * - **Tile / instance**
     - A rectangular image crop sampled from tissue. In multiple instance
       learning, one tile is one instance.
   * - **MPP**
     - Microns per pixel, the physical resolution of an image. Lower MPP means
       each pixel covers less tissue and therefore shows greater detail.
   * - ``tile_px``
     - Tile width and height in model-input pixels. ``256`` describes a
       ``256 × 256`` pixel tile.
   * - ``tile_mpp``
     - Physical resolution at which the tile is read, expressed in microns per
       pixel. It is distinct from the slide's native scanner resolution.
   * - **Feature vector**
     - A numeric embedding of one tile. ``D`` denotes its number of values or
       feature dimension.
   * - **Bag**
     - The collection of feature vectors representing one slide or aggregated
       entity to a MIL model. A feature matrix shaped ``(N, D)`` contains ``N``
       instances with feature dimension ``D``.
   * - ``N``, ``D``, ``B``, ``K``
     - Common shape/count symbols: number of instances, feature dimension,
       batch size, and retrieval cutoff/top-results count, respectively.
   * - **Slide artifact**
     - A per-slide HDF5 file containing coordinates, tiling metadata, feature
       matrices, and optional thumbnails, overviews, descriptors, or heatmaps.
       It is designed to be reused across experiment runs.
   * - **Dataset entry**
     - A YAML mapping from an annotation ``dataset`` value to slide/artifact
       directories and a role such as ``training`` or ``query``. It does not
       mean that the source images are copied.
   * - **Combination**
     - One concrete selection from the active benchmark lists—for example, one
       tile size, feature extractor, MIL model, and loss.
   * - **Trial**
     - One pipeline configuration sampled and evaluated by Optuna during
       optimization. A trial may complete, fail, or be pruned early.
   * - **Task**
     - The objective-specific behavior selected by ``experiment.task``:
       classification, regression, continuous survival, discrete survival, or
       slide retrieval.
   * - **Backend**
     - A native or third-party implementation behind a PathForge contract,
       such as a model family, metric engine, trainer, or slide processor.
   * - **Registry**
     - A name-to-implementation catalog. It lets configuration select a class
       or factory without hard-coding a large conditional statement.
   * - **Project root**
     - The named experiment directory containing summaries, checkpoints,
       visualizations, and run metadata. Reusable slide HDF5 files live in each
       dataset's separate ``artifacts_dir``.
   * - **Aggregation level**
     - The entity at which results are combined or evaluated: ``slide``,
       ``case``, or ``patient``.
   * - **Objective metric/value**
     - The named scalar used to compare benchmark runs or optimization trials,
       and its measured numeric value. Whether lower or higher is better
       depends on the metric.
