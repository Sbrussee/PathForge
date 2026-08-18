Installation
============

Requirements
------------

- Python ≥ 3.11
- PyTorch (CPU or CUDA)
- `uv <https://docs.astral.sh/uv/>`_ (recommended) or pip

The commands below assume that PathForge has been cloned and the shell is in
the repository root—the directory containing ``pyproject.toml``. ``uv sync``
creates or updates the local ``.venv`` and installs that checkout.

Recommended Install
-------------------

LazySlide feature extraction is installed by default. Add the TorchMIL model
backend and optional TorchMetrics/TorchSurv integrations with:

.. code-block:: bash

   uv sync --extra mil-backends

Use the default/CPU PyTorch installation when CUDA 12.8 is unavailable or GPU
acceleration is unnecessary. PathForge does not install a system CUDA driver;
the device support comes from the selected PyTorch build.

GPU (CUDA 12.8) build:

.. code-block:: bash

   uv sync --extra mil-backends --extra cu128

Development install (adds pytest and coverage):

.. code-block:: bash

   uv sync --extra mil-backends --extra dev

Extras are additive. A contributor who also builds the docs can run
``uv sync --extra mil-backends --extra dev --extra docs``.

Optional Extras
---------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Extra
     - Installs
   * - ``mil-backends``
     - ``torchmil``, ``torchmetrics``, ``torchsurv``, ``pycox``
   * - ``cu128``
     - CUDA 12.8 PyTorch builds (via the ``pytorch-cu128`` index)
   * - ``gnn``
     - ``torch-geometric``
   * - ``distributed``
     - Dask distributed, dask-jobqueue, and the PostgreSQL driver used by
       shared Optuna studies
   * - ``hf``
     - ``huggingface_hub``, ``typer``
   * - ``dev``
     - ``pytest``, ``pytest-cov``
   * - ``docs``
     - Sphinx + Furo theme for building this documentation

The ``mil-backends`` extra installs:

- **torchmil** — generic TorchMIL adapter for ABMIL, DSMIL, and other models
- **torchmetrics** — classification metrics (accuracy, AUROC, F1, etc.)
- **torchsurv** — survival metrics and losses (C-index, time-dependent AUC)

These are optional. Native PathForge workflows are import-safe and runnable
without them.

MIL-Lab is a separate optional MIL backend and is not part of
``mil-backends``. Install it from its `upstream repository
<https://github.com/mahmoodlab/MIL-Lab>`_ following the upstream instructions.
PathForge discovers its ``src.builder`` or ``builder`` module at runtime.

Installing for Documentation
-----------------------------

.. code-block:: bash

   uv sync --extra docs
   cd docs
   make html

Then open ``docs/_build/html/index.html`` in a browser.

Verifying the Install
---------------------

.. code-block:: bash

   python -c "import pathforge; print('ok')"

Confirm that the command-line entry point is visible:

.. code-block:: bash

   pathforge --help

If importing succeeds but the command is not found, use
``uv run pathforge --help`` or activate the repository's ``.venv``.

Check optional backends:

.. code-block:: python

   from pathforge.utils.optional.torchmil import (
       is_torchmil_available,
       is_torchmetrics_available,
       is_torchsurv_available,
   )
   from pathforge.utils.optional.mil_lab import is_mil_lab_available

   print(is_torchmil_available())
   print(is_mil_lab_available())
   print(is_torchmetrics_available())
   print(is_torchsurv_available())

``False`` is expected for an optional backend that was not installed. It is
only a problem when the active configuration selects that backend.
