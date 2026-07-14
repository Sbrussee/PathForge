Installation
============

Requirements
------------

- Python ≥ 3.11
- PyTorch (CPU or CUDA)
- `uv <https://docs.astral.sh/uv/>`_ (recommended) or pip

Recommended Install
-------------------

LazySlide feature extraction is installed by default. Add the MIL backends
(TorchMIL, TorchMetrics, and TorchSurv) with:

.. code-block:: bash

   uv sync --extra mil-backends

GPU (CUDA 12.8) build:

.. code-block:: bash

   uv sync --extra mil-backends --extra cu128

Development install (adds pytest and coverage):

.. code-block:: bash

   uv sync --extra mil-backends --extra dev

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

Check optional backends:

.. code-block:: python

   from pathforge.utils.optional.torchmil import (
       is_torchmil_available,
       is_torchmetrics_available,
       is_torchsurv_available,
   )
   print(is_torchmil_available())
   print(is_torchmetrics_available())
   print(is_torchsurv_available())
