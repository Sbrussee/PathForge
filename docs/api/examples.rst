API Usage Examples
==================

These examples use stable entry points from the API reference. Complete
end-to-end workflows, including input preparation and generated artifacts, are
covered in :doc:`../tutorials/index`.

Load and validate configuration
-------------------------------

:meth:`pathforge.config.config.Config.from_yaml` accepts a string or
:class:`pathlib.Path` and returns a fully validated configuration object.

.. code-block:: python

   from pathlib import Path

   from pathforge.config.config import Config

   config = Config.from_yaml(Path("config.yaml"))
   print(config.experiment.project_root)

Use canonical artifact identifiers
----------------------------------

Use the identifier helpers instead of formatting storage keys manually. The
``combo_cfg`` parameter is a
:class:`~pathforge.core.experiments.combinations.ComboConfig` containing one
materialized benchmark combination.

.. code-block:: python

   from pathforge.core.experiments.combinations import ComboConfig
   from pathforge.core.experiments.combo_ids import (
       build_bag_id,
       build_feature_name,
       build_tiling_id,
   )

   combo_cfg = ComboConfig(
       tile_px=256,
       tile_mpp=0.5,
       feature_extraction="uni2",
       color_norm=None,
   )
   assert build_tiling_id(combo_cfg) == "256px_0.5mpp"
   assert build_feature_name(combo_cfg) == "uni2"
   assert build_bag_id(combo_cfg) == "256px_0.5mpp__uni2"

Inspect registered extensions
-----------------------------

Registries expose availability checks and deterministic plugin listings. Call
:func:`~pathforge.utils.registries.populate_dynamic_registries` first when
optional installed backends should be discovered.

.. code-block:: python

   from pathforge.utils.registries import MODELS, populate_dynamic_registries

   populate_dynamic_registries()
   print(MODELS.list_plugins())
   if MODELS.is_available("ABMIL"):
       model_factory = MODELS.get("ABMIL")

Inspect retrieval strategies
----------------------------

The retrieval registries lazily import built-in strategies when a named
strategy is requested. Listing functions return the currently registered
names.

.. code-block:: python

   from pathforge.slide_retrieval.representation_strategies.registry import (
       import_representation_strategy_modules,
       list_representation_strategies,
   )
   from pathforge.slide_retrieval.search_strategies.registry import (
       import_search_strategy_modules,
       list_search_strategies,
   )

   import_representation_strategy_modules()
   import_search_strategy_modules()
   print(list_representation_strategies())
   print(list_search_strategies())

Run a configured workflow
-------------------------

For application workflows, the CLI is the supported composition boundary. It
loads configuration, resolves optional backends, and invokes the appropriate
policy.

.. code-block:: console

   pathforge features run --config config.yaml
   pathforge benchmark run --config config.yaml
   pathforge evaluate run --config config.yaml

Use ``--help`` at any command level to see every parameter, accepted value,
and default:

.. code-block:: console

   pathforge benchmark run --help

Reading parameter documentation
-------------------------------

Every function and method entry displays its complete Python signature. Type
annotations are shown both in the signature and in the parameter list;
defaults are preserved as written in the implementation. ``*args`` and
``**kwargs`` indicate parameters forwarded to the selected backend constructor.
