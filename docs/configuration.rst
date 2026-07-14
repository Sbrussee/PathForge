Configuration Reference
=======================

PathForge is driven entirely by YAML configuration files. Every CLI command
accepts ``--config <path>`` and builds a :class:`~pathforge.config.config.Config`
object via Pydantic v2 validation before any work begins.

Top-Level Structure
-------------------

.. code-block:: yaml

   experiment: ...
   classification: ...        # for task: classification
   slide_retrieval: ...       # for task: slide_retrieval
   mil: ...
   slide_processing: ...
   evaluation: ...
   datasets: [...]
   benchmark_parameters: ...
   metrics: ...
   explainability: ...
   optimization: ...
   weights_dir: ./pretrained_weights
   hf_key: null

``experiment``
--------------

Universal project lifecycle settings.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``project_name``
     - *required*
     - Name of the experiment. Creates a subdirectory under ``project_root``.
   * - ``annotation_file``
     - *required*
     - Absolute path to the annotation CSV.
   * - ``project_root``
     - ``./experiments``
     - Absolute path to write experiment outputs. Defaults to ``experiments/`` in the repo root.
   * - ``mode``
     - ``benchmark``
     - One of ``feature_extraction``, ``benchmark``, ``optimization``.
   * - ``task``
     - ``null``
     - One of ``classification``, ``regression``, ``survival``, ``survival_discrete``, ``slide_retrieval``. Required unless ``mode`` is ``feature_extraction``.
   * - ``prediction_level``
     - ``mil``
     - ``mil`` for bag-level prediction or ``slide`` for slide-level aggregation.
   * - ``aggregation_level``
     - ``slide``
     - ``slide``, ``case``, or ``patient`` — group level for metrics computation.
   * - ``label_column``
     - ``category``
     - Column in the annotation CSV holding the target label.
   * - ``slide_column``
     - ``slide``
     - Column in the annotation CSV holding the slide ID.
   * - ``survival_time_column``
     - ``null``
     - Column with survival time values (required for survival tasks).
   * - ``survival_event_column``
     - ``null``
     - Column with event indicator (0 = censored, 1 = event; required for survival tasks).
   * - ``num_workers``
     - ``0``
     - DataLoader worker processes.
   * - ``report``
     - ``false``
     - When true, writes tile overview images to H5 and enables PDF report generation.
   * - ``thumbnail``
     - ``false``
     - When true, stores slide thumbnails in the per-slide H5 artifact.
   * - ``mixed_precision``
     - ``false``
     - Enable 16-bit mixed precision training.
   * - ``visualization`` / ``evaluation`` / ``custom_metrics``
     - ``[]``
     - Compatibility lists preserved on the experiment block for legacy configs.

``classification``
------------------

Classification-only settings used when ``experiment.task: classification``.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``split_technique``
     - ``k-fold``
     - One of ``k-fold``, ``k-fold-stratified``, or ``fixed``.
   * - ``val_fraction``
     - ``0.1``
     - Validation fraction used by fixed-split workflows.

``mil``
-------

MIL model and training loop settings.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``backend``
     - ``native``
     - ``native``, ``torchmil``, or ``mil-lab``.
   * - ``torchmil_model``
     - ``null``
     - TorchMIL model class name (e.g. ``ABMIL``, ``DSMIL``). Used when ``backend: torchmil``.
   * - ``torchmil_model_kwargs``
     - ``{}``
     - Constructor kwargs forwarded to the TorchMIL model.
   * - ``use_torchmil_collate``
     - ``true``
     - Enable padded dict batches compatible with TorchMIL semantics.
   * - ``epochs``
     - ``20``
     - Maximum training epochs.
   * - ``batch_size``
     - ``1``
     - Batch size. Use ``1`` for native per-bag training, larger for TorchMIL.
   * - ``best_epoch_based_on``
     - ``val_loss``
     - Metric to monitor for early stopping and model selection.
   * - ``patience``
     - ``10``
     - Early stopping patience in epochs.
   * - ``accumulate_grad_batches``
     - ``1``
     - Gradient accumulation steps.
   * - ``gradient_clip_val``
     - ``0.0``
     - Maximum gradient norm (0 disables clipping).
   * - ``optimizer``
     - ``Adam``
     - Any ``torch.optim`` optimizer name (e.g. ``Adam``, ``AdamW``, ``SGD``).
   * - ``lr``
     - ``1e-4``
     - Learning rate.
   * - ``weight_decay``
     - ``1e-5``
     - Optimizer weight decay.
   * - ``scheduler``
     - ``none``
     - ``none``, ``reduce_on_plateau``, or ``cosine``.
   * - ``balancing``
     - ``null``
     - Sampler strategy for imbalanced datasets.
   * - ``class_weighting``
     - ``false``
     - Apply inverse-frequency class weights to the loss.
   * - ``bag_size``
     - ``512``
     - Maximum number of instances sampled per bag during training.
   * - ``z_dim``
     - ``256``
     - Latent dimension for attention/encoder modules.
   * - ``dropout_p``
     - ``0.1``
     - Dropout probability.
   * - ``encoder_layers``
     - ``1``
     - Number of encoder layers in attention networks.
   * - ``skip_extracted``
     - ``true``
     - Skip slides whose H5 features already exist.
   * - ``skip_feature_extraction``
     - ``true``
     - Skip feature extraction step if artifacts exist.

``slide_processing``
--------------------

WSI loading and tissue segmentation settings.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``backend``
     - ``lazyslide``
     - WSI loading backend: ``lazyslide``, ``openslide``, or ``cucim``.
   * - ``segmentation_method``
     - ``null``
     - Optional tissue segmentation algorithm name such as ``otsu``.
   * - ``save_tiles``
     - ``false``
     - Write extracted tile images to disk alongside H5 artifacts.
   * - ``qc_filters``
     - ``[]``
     - List of quality-control filter names applied per tile before feature extraction.

``datasets``
------------

A list of dataset entries. Each entry:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``name``
     - *required*
     - Dataset name. Must match the ``dataset`` column in the annotation CSV.
   * - ``slides_dir``
     - *required*
     - Directory containing WSI files.
   * - ``artifacts_dir``
     - *required*
     - Directory to write (or read) H5 feature files. Created if absent.
   * - ``features_dir``
     - ``null``
     - Optional external feature directory for workflows that read precomputed features.
   * - ``used_for``
     - *required*
     - Role: ``training``, ``validation``, or ``testing`` for MIL tasks. For
       ``slide_retrieval``, use ``reference``, ``query``, or
       ``query_reference`` (shared by both reference pool and query set).
   * - ``tissue_annotations_dir``
     - ``null``
     - Optional directory with pre-computed tissue polygon annotations.
   * - ``source``
     - ``null``
     - Set to ``gdc`` to use TCGA/TCIA datasets via ``tcga-tools``.

TCGA Dataset Entry
~~~~~~~~~~~~~~~~~~

Install the optional TCGA integration before using remote dataset entries:

.. code-block:: bash

   uv sync --extra tcga

When ``source: gdc``, use these fields instead of ``slides_dir``/``name``:

.. code-block:: yaml

   datasets:
     - source: gdc
       dataset_names: ["TCGA-LUSC", "TCGA-LUAD"]
       annotation_column: diagnoses.0.vital_status
       metadata_table: clinical_csv
       annotations: ["clinical"]
       datatype: ["wsi"]
       used_for: ["training", "testing"]

``benchmark_parameters``
-------------------------

See :doc:`mil_options` for the complete option catalogue, backend-aware model
lists, validation constraints, and benchmark/optimization examples.

Grid-search axes. All lists are combined exhaustively.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``tile_px``
     - *required*
     - List of tile sizes in pixels (e.g. ``[256, 512]``).
   * - ``tile_mpp``
     - *required*
     - List of microns-per-pixel target resolutions (e.g. ``[0.5, 1.0]``).
   * - ``feature_extraction``
     - *required*
     - List of feature extractor names. See :doc:`backends` for available options.
   * - ``mil``
     - ``[]``
     - List of MIL model registry keys (e.g. ``[AttentionMIL, torchmil]``).
   * - ``loss``
     - ``[]``
     - List of loss function registry keys (e.g. ``[CrossEntropyLoss]``).
   * - ``activation_function``
     - ``[]``
     - List of ``torch.nn`` activation class names.
   * - ``optimizer``
     - ``[]``
     - List of ``torch.optim`` optimizer class names.
   * - ``retrieval_representation``
     - ``null``
     - List of retrieval representation strategy names (e.g. ``[yottixel-features]``). Required when ``task: slide_retrieval``.
   * - ``search_strategy``
     - ``null``
     - List of search strategy names (e.g. ``[yottixel]``). Required when ``task: slide_retrieval``.

``slide_retrieval``
-------------------

Slide retrieval task settings (only used when ``task: slide_retrieval``).

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``exclusion_level``
     - ``patient``
     - Self-retrieval exclusion granularity. One of ``none`` (no exclusion),
       ``slide`` (exclude exact slide), ``case`` (exclude same case),
       ``patient`` (exclude same patient). ``slide`` requires
       ``experiment.aggregation_level: slide``.

``metrics``
-----------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``classification_backend``
     - ``torchmetrics``
     - ``native`` or ``torchmetrics``.
   * - ``survival_continuous_backend``
     - ``torchsurv``
     - ``torchsurv`` or another registered survival backend.
   * - ``classification_metrics``
     - ``[accuracy, balanced_accuracy, f1, auroc, pr_auc]``
     - Metrics to compute. Allowed: ``accuracy``, ``balanced_accuracy``, ``f1``, ``auroc``, ``pr_auc``, ``brier_score``.
   * - ``survival_metrics``
     - ``[c_index, td_auc, brier_score, num_eval_times]``
     - Survival metrics to compute.
   * - ``regression_metrics``
     - ``[mae, mse]``
     - Regression metrics.

``explainability``
------------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``heatmap_backend``
     - ``native``
     - ``native`` or ``torchmil``.
   * - ``heatmap_colormap``
     - ``inferno``
     - Matplotlib colormap for heatmap rendering.
   * - ``heatmap_tile_alpha``
     - ``0.65``
     - Tile overlay opacity (0–1).
   * - ``heatmap_smoothed_alpha``
     - ``0.8``
     - Gaussian-smoothed overlay opacity (0–1).
   * - ``heatmap_smoothing_sigma_scale``
     - ``0.75``
     - Gaussian sigma scale factor for smoothing.
   * - ``heatmap_top_k_tiles``
     - ``10``
     - Number of highest-attention tiles to highlight.

``optimization``
----------------

Optuna study settings (only used when ``mode: optimization``). Search spaces
must be declared in the YAML config under ``optimization.search_space``;
PathForge does not infer numeric ranges from model constructors.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``study_name``
     - ``study``
     - Optuna study name.
   * - ``objective_metric``
     - ``balanced_accuracy``
     - Metric to optimize.
   * - ``objective_mode``
     - ``max``
     - ``min`` or ``max``.
   * - ``sampler``
     - ``TPESampler``
     - Optuna sampler class name (e.g. ``TPESampler``, ``RandomSampler``, ``CmaEsSampler``).
   * - ``pruner``
     - ``HyperbandPruner``
     - Optuna pruner class name (e.g. ``HyperbandPruner``, ``MedianPruner``, ``NopPruner``).
   * - ``trials``
     - ``100``
     - Number of Optuna trials to run.
   * - ``search_space``
     - See below
     - Mapping from parameter names to typed search-space specifications.

Define every explicit range in the config. ``float`` and ``int`` parameters
require ``low`` and ``high``; both may set ``step`` and ``log``.
``categorical`` parameters require ``choices``:

.. code-block:: yaml

   optimization:
     study_name: abmil_search
     objective_metric: balanced_accuracy
     objective_mode: max
     sampler: TPESampler
     pruner: HyperbandPruner
     trials: 100
     search_space:
       lr:
         kind: float
         low: 1.0e-5
         high: 1.0e-3
         log: true
       epochs:
         kind: int
         low: 10
         high: 50
         step: 5
       z_dim:
         kind: categorical
         choices: [128, 256, 512]

The current optimization policy applies these training keys:
``optimizer``, ``scheduler``, ``batch_size``, ``epochs``, ``lr``,
``weight_decay``, ``dropout_p``, ``bag_size``, ``z_dim``,
``encoder_layers``, and ``k``. It also applies active ``mil``, ``loss``, and
``feature_extraction`` component choices. Multi-value lists in
``benchmark_parameters`` are automatically added as categorical dimensions
unless the same name is explicitly defined in ``optimization.search_space``.

Backend constructor dictionaries such as ``mil.torchmil_model_kwargs`` are
fixed config in the current implementation; arbitrary dotted keys or kwargs in
``optimization.search_space`` are sampled but are not applied to the model.
Use separate configs to optimize different backend constructor layouts.

Top-Level Fields
----------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``weights_dir``
     - ``./pretrained_weights``
     - Directory for pre-trained model weights.
   * - ``hf_key``
     - ``null``
     - Hugging Face API token for gated model downloads.
