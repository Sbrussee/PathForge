pathforge.utils
===============

Registries, optional package guards, constants, and helper utilities.

Registries
----------

.. automodule:: pathforge.utils.registries
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.utils.registry
   :members:
   :undoc-members:
   :show-inheritance:

Available registries:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Registry
     - Purpose
   * - ``MODELS``
     - MIL and slide-level model classes.
   * - ``LOSSES``
     - Loss functions for all task types.
   * - ``TRAINERS``
     - Trainer implementations (e.g. ``"lightning"``).
   * - ``FEATURE_EXTRACTORS``
     - Feature extraction backends.
   * - ``SLIDE_PROCESSORS``
     - WSI loading backends.
   * - ``CLASSIFICATION_METRICS``
     - Classification metric backends.
   * - ``SURVIVAL_METRICS``
     - Survival metric backends.
   * - ``SURVIVAL_LOSSES``
     - Survival loss backends.
   * - ``EXPLAINERS``
     - Heatmap/explainability methods.

Optional Package Guards
-----------------------

.. automodule:: pathforge.utils.optional.torchmil
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.utils.optional.mil_lab
   :members:
   :undoc-members:
   :show-inheritance:

Constants
---------

.. automodule:: pathforge.utils.constants
   :members:
   :undoc-members:
   :show-inheritance:

Logging
-------

.. automodule:: pathforge.utils.logging
   :members:
   :undoc-members:
   :show-inheritance:

I/O Utilities
-------------

.. automodule:: pathforge.utils.io
   :members:
   :undoc-members:
   :show-inheritance:

Serialization
-------------

.. automodule:: pathforge.utils.serialization
   :members:
   :undoc-members:
   :show-inheritance:
