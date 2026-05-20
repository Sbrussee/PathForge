Configuration Reference
=======================

PathBench is driven entirely by YAML configuration files. Every CLI command
accepts ``--config <path>`` and builds a :class:`~pathbench.config.config.Config`
object via Pydantic v2 validation before any work begins.

Top-Level Structure
-------------------

.. code-block:: yaml

   experiment: ...
   mil: ...
   slide_processing: ...
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
     - One of ``classification``, ``regression``, ``survival``, ``survival_discrete``. Required unless ``mode`` is ``feature_extraction``.
   * - ``prediction_level``
     - ``mil``
     - ``mil`` for bag-level prediction or ``slide`` for slide-level aggregation.
   * - ``aggregation_level``
     - ``slide``
     - ``slide`` or ``patient`` — group level for metrics computation.
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
   * - ``split_technique``
     - ``k-fold``
     - ``k-fold``, ``k-fold-stratified``, or ``fixed``.
   * - ``val_fraction``
     - ``0.1``
     - Fraction of training data held out for validation.
   * - ``num_workers``
     - ``0``
     - DataLoader worker processes.
   * - ``report``
     - ``false``
     - When true, writes tile overview images to H5 and enables PDF report generation.
   * - ``mixed_precision``
     - ``false``
     - Enable 16-bit mixed precision training.

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
     - ``otsu``
     - Tissue segmentation algorithm. ``otsu`` is the default; backends may support additional methods.
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
   * - ``used_for``
     - *required*
     - Role: ``training``, ``validation``, or ``testing``.
   * - ``tissue_annotations_dir``
     - ``null``
     - Optional directory with pre-computed tissue polygon annotations.
   * - ``source``
     - ``null``
     - Set to ``gdc`` to use TCGA/TCIA datasets via ``tcga-tools``.

TCGA Dataset Entry
~~~~~~~~~~~~~~~~~~

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
     - ``[ReLU]``
     - List of ``torch.nn`` activation class names.
   * - ``optimizer``
     - ``[Adam]``
     - List of ``torch.optim`` optimizer class names.

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
     - ``[c_index, td_auc, num_eval_times]``
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

Optuna study settings (only used when ``mode: optimization``).

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``study_name``
     - *required*
     - Optuna study name.
   * - ``objective_metric``
     - ``val_loss``
     - Metric to optimize.
   * - ``objective_mode``
     - ``min``
     - ``min`` or ``max``.
   * - ``sampler``
     - ``TPESampler``
     - Optuna sampler class name (e.g. ``TPESampler``, ``RandomSampler``, ``CmaEsSampler``).
   * - ``pruner``
     - ``HyperbandPruner``
     - Optuna pruner class name (e.g. ``HyperbandPruner``, ``MedianPruner``, ``NopPruner``).
   * - ``trials``
     - ``50``
     - Number of Optuna trials to run.
   * - ``search_space``
     - ``{}``
     - Dict defining the trial parameter search space.

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
