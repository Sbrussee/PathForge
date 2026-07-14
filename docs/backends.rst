Backends
========

PathForge exposes several swappable backends through its registry system. Each
backend is selected by name in the YAML config and resolved at runtime — no
code changes required to switch implementations.

WSI Processing Backends
------------------------

Configured via ``slide_processing.backend``.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Key
     - Description
   * - ``lazyslide``
     - **Default and currently implemented processor.** Uses the
       `Lazyslide <https://lazyslide.readthedocs.io>`_ / WSIData stack.
       Lazyslide itself can route I/O through tiffslide and related readers.
       Installed by default and integrates with ``timm`` feature
       extractors.
   * - ``openslide``
     - Reserved configuration key for a future dedicated OpenSlide processor.
       It is documented here for API direction, but PathForge currently ships
       the Lazyslide processor only.
   * - ``cucim``
     - Reserved configuration key for a future dedicated cuCIM processor.
       GPU-aware WSI acceleration currently comes through the Lazyslide stack
       rather than a separate registered PathForge processor.

All backends implement :class:`~pathforge.core.slide_processing.base.SlideProcessorBase`
and are registered in :data:`~pathforge.utils.registries.SLIDE_PROCESSORS`.

Feature Extraction Backends
----------------------------

Configured via ``benchmark_parameters.feature_extraction``.

Feature extractors are identified by name and resolved through
:data:`~pathforge.utils.registries.FEATURE_EXTRACTORS` and the Lazyslide /
timm model registries.

Common extractors
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - ``resnet18``
     - ResNet-18 pre-trained on ImageNet (timm). Fast baseline.
   * - ``resnet50``
     - ResNet-50 pre-trained on ImageNet (timm). Stronger baseline.
   * - ``uni``
     - UNI pathology foundation model (requires HF token).
   * - ``conch``
     - CONCH pathology foundation model (requires HF token).
   * - ``gigapath``
     - GigaPath slide-level encoder (requires HF token).
   * - ``phikon``
     - Phikon pathology ViT (requires HF token).

Any model available through ``timm.list_models()`` or registered via the
Lazyslide model registry can be used as a feature extractor name.

Check available extractors at runtime:

.. code-block:: python

   from pathforge.utils.registries import all_feature_extractor_names
   print(all_feature_extractor_names())

MIL Backends
-------------

Configured via ``mil.backend``.

native
~~~~~~

The ``native`` backend uses PathForge model classes registered directly in
:data:`~pathforge.utils.registries.MODELS`. No optional dependencies required.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Registry Key
     - Model
   * - ``PerceiverMIL``
     - Perceiver-based MIL.
   * - ``VarMIL``
     - Variational MIL.
   * - ``PrototypeMIL``
     - Prototype-based MIL.
   * - ``MambaMIL``
     - Optional Mamba-based MIL; available when ``mamba`` is installed.
   * - ``SlideVectorMLP``
     - Neural slide-vector model exposed by the native catalog.

Example config:

.. code-block:: yaml

   mil:
     backend: native

   benchmark_parameters:
     mil: [PerceiverMIL]
     loss: [CrossEntropyLoss]

torchmil
~~~~~~~~~

The ``torchmil`` backend wraps any TorchMIL model class through a single
generic adapter :class:`~pathforge.adapters.torchmil.backend.TorchMILBackendModel`.
Requires the ``mil-backends`` extra.

Available TorchMIL models — see the `TorchMIL model API
<https://torchmil.readthedocs.io/en/latest/api/models/>`_ for the full list.
Common models include ``ABMIL``, ``DSMIL``, ``TransMIL``,
``CLAM_SB``, ``CLAM_MB``.

Example config:

.. code-block:: yaml

   mil:
     torchmil_model_kwargs:
       in_shape: [1024]
       out_shape: 2
     use_torchmil_collate: true

   benchmark_parameters:
     mil: [ABMIL, CLAM]
     loss: [CrossEntropyLoss]

Concrete available TorchMIL class names are registered as model keys. The
generic ``torchmil`` key remains supported only for compatibility with older
configs that set ``mil.torchmil_model`` separately.

mil-lab
~~~~~~~

The ``mil-lab`` backend is an optional third backend registered conditionally
when the ``mil-lab`` package is installed. Available models are listed in the
`MIL-Lab repository <https://github.com/mahmoodlab/MIL-Lab>`_.

Use concrete available MIL-Lab names in the same grid:

.. code-block:: yaml

   mil:
     mil_lab_from_pretrained: false

   benchmark_parameters:
     mil: [abmil, clam, transmil]

Metrics Backends
-----------------

Classification
~~~~~~~~~~~~~~

Configured via ``metrics.classification_backend``.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Key
     - Description
   * - ``torchmetrics``
     - **Default.** Uses `TorchMetrics <https://torchmetrics.readthedocs.io>`_.
       Requires the ``mil-backends`` extra.
   * - ``native``
     - Sklearn-based metrics fallback. Does not require optional packages.

Survival
~~~~~~~~

Configured via ``metrics.survival_continuous_backend``.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Key
     - Description
   * - ``torchsurv``
     - **Default.** Uses `TorchSurv <https://torchsurv.readthedocs.io>`_ C-index
       and time-dependent AUC. Requires the ``mil-backends`` extra.

Explainability Backends
-----------------------

Configured via ``explainability.heatmap_backend``.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Key
     - Description
   * - ``native``
     - Placeholder; no heatmap is rendered.
   * - ``torchmil``
     - Resolves the TorchMIL heatmap explainer through
       :data:`~pathforge.utils.registries.EXPLAINERS`.
       Produces per-instance attention maps normalized to ``[0, 1]``.

Registering a Custom Backend
-----------------------------

All registries follow the same decorator pattern:

.. code-block:: python

   from pathforge.core.models.mil_base import MILModelBase
   from pathforge.utils.registries import MODELS

   @MODELS.register("MyMIL")
   class MyMIL(MILModelBase):
       def __init__(self, in_dim: int, n_classes: int) -> None:
           super().__init__()
           ...

       def forward_bag(self, bag: torch.Tensor) -> torch.Tensor:
           ...

Import the module before calling :func:`~pathforge.config.config.Config.from_yaml`
to ensure the registration runs. For optional backends, wrap registration in an
availability check:

.. code-block:: python

   from pathforge.utils.optional.torchmil import is_torchmil_available
   from pathforge.utils.registries import MODELS

   if is_torchmil_available():
       @MODELS.register("my_optional_model")
       class MyOptionalModel(MILModelBase):
           ...
