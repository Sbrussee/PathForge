pathforge.core
==============

The stable domain layer. Framework-agnostic abstractions that all other layers
depend on. Never imports from policies, CLI, or adapters.

Models
------

.. autoclass:: pathforge.core.models.base.ModelBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.models.base.TorchModelBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.models.base.ScikitBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.models.mil_base.MILModelBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.models.slide_base.SlideModelBase
   :members:
   :undoc-members:
   :show-inheritance:

Native MIL Models
~~~~~~~~~~~~~~~~~

.. automodule:: pathforge.core.models.layers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.models.perceiver_mil
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.models.prototype_mil
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.models.var_mil
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.models.mil_ens
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.models.mil_graph
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.models.mil_mm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.models.slide_mlp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.models.sklearn_slide
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.models.utils
   :members:
   :undoc-members:
   :show-inheritance:

Losses
------

.. autoclass:: pathforge.core.losses.base.BaseLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.losses.base.ClassificationLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.losses.base.RegressionLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.losses.base.SurvivalContinuousLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.losses.base.SurvivalDiscreteLoss
   :members:
   :undoc-members:
   :show-inheritance:

Annotations
-----------

.. automodule:: pathforge.core.annotations.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.annotations.binning
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.annotations.csv
   :members:
   :undoc-members:
   :show-inheritance:

Datasets
--------

.. autoclass:: pathforge.core.datasets.base.DatasetBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.datasets.base.BagDatasetBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.datasets.base.TileDatasetBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.datasets.wsi_dataset.WSI
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.datasets.wsi_dataset.WSIDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.datasets.bag_dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.datasets.bag_schema
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.datasets.samplers
   :members:
   :undoc-members:
   :show-inheritance:

Experiments
-----------

.. autoclass:: pathforge.core.experiments.base.Experiment
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.experiments.base.ComboConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.experiments.utils
   :members:
   :undoc-members:
   :show-inheritance:

H5 I/O
------

.. autoclass:: pathforge.core.io.h5.base.FileHandleH5
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.io.h5.tiles
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.io.h5.features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.io.h5.heatmaps
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.io.h5.tissue
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.io.h5.layout
   :members:
   :undoc-members:
   :show-inheritance:

Slide Processing
----------------

.. autoclass:: pathforge.core.slide_processing.base.SlideProcessorBase
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.slide_processing.lazyslide
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.slide_processing.utils
   :members:
   :undoc-members:
   :show-inheritance:

Explainability
--------------

.. autoclass:: pathforge.core.explainer_base.ExplainerBase
   :members:
   :undoc-members:
   :show-inheritance:

Reports And Visualization
-------------------------

.. automodule:: pathforge.core.reports.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.reports.tiles_report_pdf
   :members:
   :undoc-members:
   :show-inheritance:


.. automodule:: pathforge.core.visualization.tiles_overview
   :members:
   :undoc-members:
   :show-inheritance:

Annotations
-----------

.. automodule:: pathforge.core.annotations.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.annotations.csv
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathforge.core.annotations.binning
   :members:
   :undoc-members:
   :show-inheritance:

Reports
-------

.. automodule:: pathforge.core.reports.tiles_report_pdf
   :members:
   :undoc-members:
   :show-inheritance:

Registry
--------

.. automodule:: pathforge.core.registry
   :members:
   :undoc-members:
   :show-inheritance:

Tasks
-----

Task registry and base class for all benchmarking and retrieval tasks.

.. automodule:: pathforge.core.tasks.registry
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.core.tasks.base.TaskBase
   :members:
   :undoc-members:
   :show-inheritance:

MIL Tasks
~~~~~~~~~

.. automodule:: pathforge.core.tasks.mil
   :members:
   :undoc-members:
   :show-inheritance:

Slide Retrieval Task
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pathforge.core.tasks.slide_retrieval.SlideRetrievalTask
   :members:
   :undoc-members:
   :show-inheritance:
