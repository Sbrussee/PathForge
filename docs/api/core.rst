pathbench.core
==============

The stable domain layer. Framework-agnostic abstractions that all other layers
depend on. Never imports from policies, CLI, or adapters.

Models
------

.. autoclass:: pathbench.core.models.base.ModelBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.core.models.base.TorchModelBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.core.models.base.ScikitBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.core.models.mil_base.MILModelBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.core.models.slide_base.SlideModelBase
   :members:
   :undoc-members:
   :show-inheritance:

Native MIL Models
~~~~~~~~~~~~~~~~~

.. automodule:: pathbench.core.models.layers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.models.perceiver_mil
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.models.prototype_mil
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.models.var_mil
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.models.mil_ens
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.models.mil_graph
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.models.mil_mm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.models.slide_mlp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.models.sklearn_slide
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.models.utils
   :members:
   :undoc-members:
   :show-inheritance:

Losses
------

.. autoclass:: pathbench.core.losses.base.BaseLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.core.losses.base.ClassificationLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.core.losses.base.RegressionLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.core.losses.base.SurvivalContinuousLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.core.losses.base.SurvivalDiscreteLoss
   :members:
   :undoc-members:
   :show-inheritance:

Annotations
-----------

.. automodule:: pathbench.core.annotations.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.annotations.binning
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.annotations.csv
   :members:
   :undoc-members:
   :show-inheritance:

Datasets
--------

.. autoclass:: pathbench.core.datasets.base.DatasetBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.core.datasets.base.BagDatasetBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.core.datasets.base.TileDatasetBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.core.datasets.wsi_dataset.WSI
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.core.datasets.wsi_dataset.WSIDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.datasets.bag_dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.datasets.bag_schema
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.datasets.samplers
   :members:
   :undoc-members:
   :show-inheritance:

Experiments
-----------

.. autoclass:: pathbench.core.experiments.base.Experiment
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.core.experiments.base.ComboConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.experiments.utils
   :members:
   :undoc-members:
   :show-inheritance:

H5 I/O
------

.. autoclass:: pathbench.core.io.h5.base.FileHandleH5
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.io.h5.tiles
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.io.h5.features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.io.h5.heatmaps
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.io.h5.tissue
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.io.h5.layout
   :members:
   :undoc-members:
   :show-inheritance:

Slide Processing
----------------

.. autoclass:: pathbench.core.slide_processing.base.SlideProcessorBase
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.slide_processing.lazyslide
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.slide_processing.utils
   :members:
   :undoc-members:
   :show-inheritance:

Explainability
--------------

.. autoclass:: pathbench.core.explainer_base.ExplainerBase
   :members:
   :undoc-members:
   :show-inheritance:

Reports And Visualization
-------------------------

.. automodule:: pathbench.core.reports.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.reports.tiles_report_pdf
   :members:
   :undoc-members:
   :show-inheritance:


.. automodule:: pathbench.core.visualization.tiles_overview
   :members:
   :undoc-members:
   :show-inheritance:

Annotations
-----------

.. automodule:: pathbench.core.annotations.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.annotations.csv
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pathbench.core.annotations.binning
   :members:
   :undoc-members:
   :show-inheritance:

Reports
-------

.. automodule:: pathbench.core.reports.tiles_report_pdf
   :members:
   :undoc-members:
   :show-inheritance:

Registry
--------

.. automodule:: pathbench.core.registry
   :members:
   :undoc-members:
   :show-inheritance:
