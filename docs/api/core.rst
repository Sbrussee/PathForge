pathforge.core
==============

Stable domain abstractions and task orchestration shared by the other layers.

Common contracts
----------------

.. automodule:: pathforge.core.base
   :members:

Models
------

.. autoclass:: pathforge.core.models.base.ModelBase
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.models.base.TorchModelBase
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.models.base.ScikitBase
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.models.mil_base.MILModelBase
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.models.slide_base.SlideLevelModel
   :members:
   :show-inheritance:

Native MIL Models
~~~~~~~~~~~~~~~~~

.. automodule:: pathforge.core.models.layers
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.models.perceiver_mil
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.models.prototype_mil
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.models.var_mil
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.models.mil_ens
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.models.mil_graph
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.models.mil_mm
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.models.slide_mlp
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.models.sklearn_slide
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.models.utils
   :members:
   :show-inheritance:

Losses
------

.. autoclass:: pathforge.core.losses.base.BaseLoss
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.losses.base.ClassificationLoss
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.losses.base.RegressionLoss
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.losses.base.SurvivalContinuousLoss
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.losses.base.SurvivalDiscreteLoss
   :members:
   :show-inheritance:

Annotations
-----------

.. automodule:: pathforge.core.annotations.base
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.annotations.binning
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.annotations.csv
   :members:
   :show-inheritance:

Datasets
--------

.. autoclass:: pathforge.core.datasets.base.DatasetBase
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.datasets.base.BagDatasetBase
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.datasets.base.TileDatasetBase
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.datasets.wsi_dataset.WSI
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.datasets.wsi_dataset.WSIDataset
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.datasets.bag_dataset
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.datasets.bag_schema
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.datasets.samplers
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.datasets.factory
   :members:

.. automodule:: pathforge.core.datasets.utils
   :members:

Experiments
-----------

.. autoclass:: pathforge.core.experiments.base.Experiment
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.experiments.base.ComboConfig
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.experiments.combinations
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.experiments.combo_ids
   :members:

Feature helpers
---------------

.. automodule:: pathforge.core.features.utils
   :members:

H5 I/O
------

.. autoclass:: pathforge.core.io.h5.base.FileHandleH5
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.io.h5.tiles
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.io.h5.features
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.io.h5.heatmaps
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.io.h5.tissue
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.io.h5.layout
   :members:
   :show-inheritance:

Slide Processing
----------------

.. autoclass:: pathforge.core.slide_processing.base.SlideProcessorBase
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.slide_processing.lazyslide
   :members:
   :show-inheritance:

Explainability
--------------

.. autoclass:: pathforge.core.explainer_base.ExplainerBase
   :members:
   :show-inheritance:

Reports And Visualization
-------------------------

.. automodule:: pathforge.core.reports.base
   :members:
   :show-inheritance:

.. automodule:: pathforge.core.reports.tiles_report_pdf
   :members:
   :show-inheritance:


.. automodule:: pathforge.core.visualization.tiles_overview
   :members:
   :show-inheritance:

Registry
--------

.. automodule:: pathforge.core.registry
   :members:
   :show-inheritance:

Tasks
-----

Task registry and base class for all benchmarking and retrieval tasks.

.. automodule:: pathforge.core.tasks.registry
   :members:
   :show-inheritance:

.. autoclass:: pathforge.core.tasks.base.TaskBase
   :members:
   :show-inheritance:

MIL Tasks
~~~~~~~~~

.. automodule:: pathforge.core.tasks.mil
   :members:
   :show-inheritance:

Slide Retrieval Task
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pathforge.core.tasks.slide_retrieval.SlideRetrievalTask
   :members:
   :show-inheritance:
