pathbench.adapters
==================

Concrete integrations with optional third-party packages. All adapter imports
are isolated here and in :mod:`pathbench.utils.optional`. The rest of
PathBench is import-safe without these packages.

TorchMIL
---------

Backend model adapter
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pathbench.adapters.torchmil.backend
   :members:
   :undoc-members:
   :show-inheritance:

Collate function
~~~~~~~~~~~~~~~~

.. automodule:: pathbench.adapters.torchmil.collate
   :members:
   :undoc-members:
   :show-inheritance:

Heatmap explainer
~~~~~~~~~~~~~~~~~

.. automodule:: pathbench.adapters.torchmil.heatmap_explainer
   :members:
   :undoc-members:
   :show-inheritance:

Task output
~~~~~~~~~~~

.. automodule:: pathbench.adapters.torchmil.task_output
   :members:
   :undoc-members:
   :show-inheritance:

Losses
------

Torch / TorchSurv loss adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pathbench.adapters.losses
   :members:
   :undoc-members:
   :show-inheritance:

Metrics
-------

Classification metrics (TorchMetrics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pathbench.adapters.metrics.classification
   :members:
   :undoc-members:
   :show-inheritance:

Survival metrics (TorchSurv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pathbench.adapters.metrics.survival
   :members:
   :undoc-members:
   :show-inheritance:

MIL-Lab
-------

.. automodule:: pathbench.adapters.mil_lab.backend
   :members:
   :undoc-members:
   :show-inheritance:

TCGA Tools
----------

.. automodule:: pathbench.adapters.tcga_tools
   :members:
   :undoc-members:
   :show-inheritance:
