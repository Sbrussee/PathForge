pathforge.training
==================

Trainer abstractions and the PyTorch Lightning implementation. Trainers are
registered in :data:`~pathforge.utils.registries.TRAINERS` and resolved by
name at runtime.

Base Abstractions
-----------------

.. autoclass:: pathforge.training.base.TrainerBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pathforge.training.base.MILTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Lightning Trainer
-----------------

The Lightning trainer is registered as ``"lightning"`` in
:data:`~pathforge.utils.registries.TRAINERS`.

.. automodule:: pathforge.training.lightning
   :members:
   :undoc-members:
   :show-inheritance:

Callbacks
---------

.. automodule:: pathforge.training.callbacks
   :members:
   :undoc-members:
   :show-inheritance:

Metrics
-------

.. automodule:: pathforge.training.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Scikit-Learn Trainer
--------------------

.. automodule:: pathforge.training.sklearn_trainer
   :members:
   :undoc-members:
   :show-inheritance:
