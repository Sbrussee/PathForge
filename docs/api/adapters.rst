pathforge.adapters
==================

Concrete integrations with optional third-party packages. All adapter imports
are isolated here and in :mod:`pathforge.utils.optional`. The rest of
PathForge is import-safe without these packages.

TorchMIL
---------

Backend model adapter
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pathforge.adapters.torchmil.backend
   :members:
   :undoc-members:
   :show-inheritance:

Collate function
~~~~~~~~~~~~~~~~

.. automodule:: pathforge.adapters.torchmil.collate
   :members:
   :undoc-members:
   :show-inheritance:

Heatmap explainer
~~~~~~~~~~~~~~~~~

.. automodule:: pathforge.adapters.torchmil.heatmap_explainer
   :members:
   :undoc-members:
   :show-inheritance:

Task output
~~~~~~~~~~~

.. automodule:: pathforge.adapters.torchmil.task_output
   :members:
   :undoc-members:
   :show-inheritance:

Losses
------

Torch / TorchSurv loss adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pathforge.adapters.losses
   :members:
   :undoc-members:
   :show-inheritance:

Metrics
-------

Classification metrics (TorchMetrics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pathforge.adapters.metrics.classification
   :members:
   :undoc-members:
   :show-inheritance:

Survival metrics (TorchSurv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pathforge.adapters.metrics.survival
   :members:
   :undoc-members:
   :show-inheritance:

MIL-Lab
-------

.. automodule:: pathforge.adapters.mil_lab.backend
   :members:
   :undoc-members:
   :show-inheritance:

TCGA Tools
----------

.. automodule:: pathforge.adapters.tcga_tools
   :members:
   :undoc-members:
   :show-inheritance:
