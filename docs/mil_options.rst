MIL Benchmark and Optimization Options
======================================

This page summarizes the choices accepted by the current PathForge code. The
source of truth is
:class:`pathforge.config.config.BenchmarkParameters`, its Pydantic validators,
and the catalogs in :mod:`pathforge.utils.registries`. Lists under
``benchmark_parameters`` form a Cartesian benchmark grid. Optuna ranges are
configured separately under ``optimization.search_space``.

Preprocessing grid
------------------

.. list-table::
   :widths: 22 23 55
   :header-rows: 1

   * - Field
     - Accepted values
     - Notes
   * - ``tile_px``
     - Even integers; default ``[256]``
     - Tile edge length in pixels. Examples include ``[224, 256, 512]``.
       The current PathForge validator checks evenness rather than enforcing a
       fixed catalogue or explicitly checking positivity; use positive values.
   * - ``tile_mpp``
     - Positive floats; default ``[0.5]``
     - Physical resolution in microns per pixel. Prefer MPP to scanner
       magnification: nominal magnification is scanner-dependent. Common
       starting points are 0.25 MPP (approximately 40x), 0.5 MPP
       (approximately 20x), and 1.0 MPP (approximately 10x).
   * - ``color_norm``
     - ``null``, ``reinhard``, ``macenko``
     - ``null`` disables stain normalization. These are the only names allowed
       by the current validator.
   * - ``feature_extraction``
     - Installed PathForge, timm, or LazySlide extractor name
     - The catalogue is discovered at runtime; see `Feature extractors`_.

Feature extractors
------------------

PathForge accepts the union returned by
:func:`pathforge.utils.registries.all_feature_extractor_names`: registered
PathForge extractors, ``timm.list_models()``, and the installed LazySlide model
catalogue. The exact list therefore depends on installed package versions and
access to gated weights. The `LazySlide model zoo
<https://lazyslide.readthedocs.io/en/stable/avail_models.html>`_ documents
pathology-specific models, input assumptions, access requirements, and
licenses. LazySlide also supports timm vision models.

Inspect the actual options in the active environment:

.. code-block:: python

   from pathforge.utils.registries import list_feature_extractors

   for item in list_feature_extractors():
       print(item.name, item.backend, item.available)

MIL architectures
-----------------

The current catalog is exposed by
:func:`pathforge.utils.registries.list_mil_models`.

.. list-table::
   :widths: 20 45 35
   :header-rows: 1

   * - Backend
     - Models declared by PathForge
     - Configuration
   * - Native
     - ``PerceiverMIL``, ``PrototypeMIL``, ``VarMIL``, ``SlideVectorMLP``;
       ``MambaMIL`` when the ``mamba`` dependency is installed
     - Select names in ``benchmark_parameters.mil``.
   * - TorchMIL
     - Explicitly catalogued: ``ABMIL``, ``CLAM``, ``DSMIL``. The generic
       adapter also accepts another class exposed by the installed
       ``torchmil.models`` package.
     - Select concrete names in ``benchmark_parameters.mil``. Pass shared
       constructor arguments through ``mil.torchmil_model_kwargs``.
   * - MIL-Lab
     - ``abmil``, ``clam``, ``dftd``, ``dsmil``, ``ilra``, ``rrt``,
       ``transformer``, ``transmil``, ``wikg``
     - Select concrete names in ``benchmark_parameters.mil``. Pass shared
       constructor arguments through ``mil.mil_lab_model_kwargs``.

TorchMIL evolves independently, so consult its `model API
<https://torchmil.readthedocs.io/en/stable/api/models/>`_ for constructor
parameters, input shapes, and outputs. PathForge forwards
``mil.torchmil_model_kwargs`` directly to the chosen constructor.

The concrete model name determines the backend for each combination. This
allows a grid such as ``[PerceiverMIL, ABMIL, clam]`` when PathForge, TorchMIL,
and MIL-Lab are installed. Generic ``torchmil`` and ``mil-lab`` sentinel keys
remain supported for legacy configs but should not be used in new grids.

.. code-block:: python

   from pathforge.utils.registries import list_mil_models

   for item in list_mil_models():
       print(item.name, item.backend, item.available)

Slide-level alternatives
------------------------

For ``experiment.prediction_level: slide``, ``slide_level_models`` accepts:

- Classification: ``SklearnLogisticRegressionClassifier``,
  ``SklearnRandomForestClassifier``, ``SklearnGradientBoostingClassifier``,
  ``SklearnSVMClassifier``.
- Regression: ``SklearnLinearRegressor``, ``SklearnRidgeRegressor``,
  ``SklearnElasticNetRegressor``, ``SklearnRandomForestRegressor``,
  ``SklearnGradientBoostingRegressor``, ``SklearnSVMRegressor``.
- Survival (requires scikit-survival): ``SklearnCoxPH``, ``SklearnCoxnet``,
  ``SklearnIPCRidge``, ``SklearnHingeLossSurvivalSVM``,
  ``SklearnNaiveSurvivalSVM``, ``SklearnSurvivalTree``,
  ``SklearnRandomSurvivalForest``.
- Neural slide vector: ``SlideVectorMLP``.

``slide_aggregation`` accepts ``mean``, ``max``, and ``mean_max``.

Losses
------

``benchmark_parameters.loss`` resolves names through
:data:`pathforge.utils.registries.LOSSES`.

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Task
     - Registered names
   * - Classification
     - ``BCELoss``, ``BCEWithLogitsLoss``, ``CrossEntropyLoss``, ``NLLLoss``
   * - Regression
     - ``HuberLoss``, ``L1Loss``, ``MSELoss``, ``SmoothL1Loss``
   * - Survival
     - With TorchSurv: ``CoxPHLoss``, ``neg_partial_log_likelihood``,
       ``DiscreteTimeNLLLoss``, ``neg_log_likelihood``,
       ``neg_log_likelihood_weibull``

Training grid
-------------

.. list-table::
   :widths: 24 31 45
   :header-rows: 1

   * - Field
     - Accepted values
     - Meaning
   * - ``activation_function``
     - Classes exposed by ``torch.nn`` activation modules
     - Examples: ``ReLU``, ``GELU``, ``LeakyReLU``.
   * - ``optimizer``
     - ``torch.optim.Optimizer`` subclasses
     - Examples: ``Adam``, ``AdamW``, ``SGD``.
   * - ``scheduler``
     - ``none``, ``reduce_on_plateau``, ``cosine``
     - Learning-rate scheduling policy.
   * - ``batch_size``
     - Integers greater than zero; default ``[16]``
     - Bags per optimization step before gradient accumulation.
   * - ``epochs``
     - Integers greater than zero
     - Maximum epochs per grid combination.
   * - ``lr``
     - Floats greater than zero
     - Optimizer learning rate.
   * - ``weight_decay``
     - Floats greater than or equal to zero
     - Optimizer weight decay.
   * - ``dropout_p``
     - Floats in ``[0, 1]``
     - General model dropout probability.
   * - ``bag_size``
     - Integers greater than zero
     - Maximum sampled instances per bag.
   * - ``z_dim``
     - Integers greater than zero
     - General latent dimension; backend models may use another argument.
   * - ``encoder_layers``
     - Integers greater than zero
     - General encoder depth.
   * - ``k``
     - Integers greater than zero
     - Model-specific neighborhood or top-instance parameter.
   * - ``seeds``
     - Integers; default ``[1, 2, 3]``
     - Repeated runs used to quantify seed variability.

Backend constructor options belong in an entry's ``hyperparams`` mapping or
the backend-specific ``mil.*_model_kwargs`` mapping. Check the model API before
assuming that general fields such as ``z_dim`` or ``k`` apply.

Grid-search example
-------------------

.. code-block:: yaml

   mil:
     backend: native

   benchmark_parameters:
     tile_px: [256, 512]
     tile_mpp: [0.5, 1.0]
     color_norm: [null, macenko]
     feature_extraction: [uni2]
     mil: [PerceiverMIL, VarMIL]
     loss: [CrossEntropyLoss]
     activation_function: [ReLU, GELU]
     optimizer: [Adam, AdamW]
     scheduler: [none, cosine]
     batch_size: [8, 16]
     epochs: [20, 40]
     lr: [0.0001, 0.00001]
     weight_decay: [0.0, 0.00001]
     dropout_p: [0.1, 0.3]
     bag_size: [256, 512]
     z_dim: [128, 256]
     encoder_layers: [1, 2]
     k: [1, 2]
     seeds: [1, 2, 3]

Optuna search spaces
--------------------

``optimization.search_space`` accepts parameter names consumed by the
optimization policy. Each value is a
:class:`pathforge.config.config.SearchSpaceParameter`:

- ``kind: float`` or ``kind: int`` requires ``low`` and ``high`` and may set
  ``step`` and ``log``.
- ``kind: categorical`` requires a non-empty ``choices`` list.

The code defaults to ``lr``, ``epochs``, ``z_dim``, ``dropout_p``, and
``weight_decay``. It also turns non-empty multi-value benchmark lists into
categorical search dimensions. The optimization policy currently maps
``optimizer``, ``scheduler``, ``batch_size``, ``epochs``, ``lr``,
``weight_decay``, ``dropout_p``, ``bag_size``, ``z_dim``, ``encoder_layers``,
and ``k`` onto ``mil`` settings, and tracks active ``mil``, ``loss``, and
``feature_extraction`` choices. Adding an unrelated custom key to
``search_space`` does not by itself make a model consume it.

.. code-block:: yaml

   optimization:
     objective_metric: balanced_accuracy
     objective_mode: max
     sampler: TPESampler
     trials: 100
     pruner: HyperbandPruner
     search_space:
       lr: {kind: float, low: 1.0e-5, high: 1.0e-3, log: true}
       epochs: {kind: int, low: 10, high: 50, step: 5}
       z_dim: {kind: categorical, choices: [128, 256, 512]}
       dropout_p: {kind: float, low: 0.1, high: 0.5}
       weight_decay: {kind: float, low: 1.0e-6, high: 1.0e-3, log: true}
