Troubleshooting
===============

This page covers common errors and how to resolve them.

Installation Errors
-------------------

**``MIL backend 'torchmil' selected, but 'torchmil' is not installed.``**

Install the optional backend extra:

.. code-block:: bash

   uv sync --extra mil-backends

Or switch to the native backend:

.. code-block:: yaml

   mil:
     backend: native

**``Classification metrics backend requires 'torchmetrics'.``**

Install torchmetrics:

.. code-block:: bash

   uv sync --extra mil-backends

Or use the native backend:

.. code-block:: yaml

   metrics:
     classification_backend: native

**``Continuous survival backend requires 'torchsurv'.``**

Install torchsurv:

.. code-block:: bash

   uv sync --extra mil-backends

Configuration Errors
--------------------

**``experiment.task is required unless mode == 'feature_extraction'.``**

Add the ``task`` field for benchmark or optimization modes:

.. code-block:: yaml

   experiment:
     mode: benchmark
     task: classification   # add this

**``cfg.experiment.project_root must be an absolute path.``**

Use an absolute path:

.. code-block:: yaml

   experiment:
     project_root: /data/pathforge_projects   # must be absolute

**``Feature extractor '<name>' is not registered.``**

Ensure the extractor package is installed. For Lazyslide/timm extractors:

.. code-block:: bash

   uv sync

Check available extractors:

.. code-block:: python

   from pathforge.utils.registries import all_feature_extractor_names, populate_dynamic_registries
   populate_dynamic_registries()
   print(all_feature_extractor_names())

Dataset Errors
--------------

**No slides found for a dataset.**

Check all three conditions:

1. The ``dataset`` column in your annotation CSV matches ``datasets[].name`` exactly.
2. ``slides_dir`` exists and is readable.
3. Slide files use a supported suffix: ``.svs``, ``.ndpi``, ``.tiff``, ``.tif``, ``.mrxs``.

**``The source annotation CSV must contain exactly one row matching the dataset and slide stem.``**

When running ``feature_extraction_slide``, the annotation CSV must have exactly
one row for the requested ``--dataset`` / ``--input`` slide combination.

MPP Errors
----------

**Slides fail with an MPP validation error.**

Some WSI files do not embed valid microns-per-pixel metadata. Add a
``fallback_mpp`` column to your annotation CSV:

.. code-block:: text

   dataset,slide,patient,category,fallback_mpp
   TrainingSet,SLIDE_001,P001,case,0.5

Training Errors
---------------

**OOM (out of memory) during training.**

Reduce ``mil.bag_size`` (fewer tiles sampled per bag) or ``mil.batch_size``.
For large bags with TorchMIL, enable mixed precision:

.. code-block:: yaml

   experiment:
     mixed_precision: true

**Loss is NaN after the first step.**

Common causes:

- Learning rate too high. Try ``lr: 1e-5``.
- Gradient explosion. Enable clipping:

  .. code-block:: yaml

     mil:
       gradient_clip_val: 1.0

H5 Artifact Errors
------------------

**H5 files have mismatched coordinates and features.**

PathForge enforces row alignment. If you see shape mismatches, the artifact
is corrupt. Delete the affected H5 file and re-run feature extraction:

.. code-block:: bash

   rm /data/artifacts/train/SLIDE_001.h5
   pathforge-features --config features.yaml

**Tile overview reports show no images.**

Set ``experiment.report: true`` in your feature extraction config *before*
running extraction. Reports cannot be generated retroactively for slides that
were extracted without the flag.

Cluster (SLURM) Jobs
--------------------

**Array jobs write to the same project directory.**

For the single-slide feature-extraction command, PathForge suffixes the project
name with the job ID when ``SLURM_JOB_ID`` is set:

.. code-block:: text

   my_project_12345678/

This prevents race conditions between array tasks.
Other PathForge commands do not currently apply this suffix automatically. See
:doc:`scaling` for the supported single-slide pattern and the proposed
distributed execution model.

**Single-slide extraction on a cluster.**

Use ``feature_extraction_slide`` for per-task execution:

.. code-block:: bash

   pathforge features slide \
     --config features.yaml \
     --dataset TrainingSet \
     --input /data/slides/train/SLIDE_${SLURM_ARRAY_TASK_ID}.svs

Debugging
---------

Increase log verbosity with ``--log-level DEBUG``:

.. code-block:: bash

   pathforge-features --config features.yaml --log-level DEBUG

Run just the failing unit tests:

.. code-block:: bash

   uv run pytest -q tests/unit/test_config_validation.py -v
