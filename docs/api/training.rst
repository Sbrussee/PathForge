pathbench.training
==================

Trainer abstractions and the PyTorch Lightning implementation. Trainers are
registered in :data:`~pathbench.utils.registries.TRAINERS` and resolved by
name at runtime.

Base Abstractions
-----------------

.. autoclass:: pathbench.training.base.TrainerBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathbench.training.base.MILTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Lightning Trainer
-----------------

The Lightning trainer is registered as ``"lightning"`` in
:data:`~pathbench.utils.registries.TRAINERS`.

.. automodule:: pathbench.training.lightning
   :members:
   :undoc-members:
   :show-inheritance:

Callbacks
---------

.. automodule:: pathbench.training.callbacks
   :members:
   :undoc-members:
   :show-inheritance:

Metrics
-------

.. automodule:: pathbench.training.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Scikit-Learn Trainer
--------------------

.. automodule:: pathbench.training.sklearn_trainer
   :members:
   :undoc-members:
   :show-inheritance:
