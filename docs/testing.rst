Testing
=======

This page documents the supported PathForge test suites, how to run them, and
what outputs to expect from each one.

All commands below assume you are running from the repository root:

.. code-block:: bash

   cd /exports/path-cutane-lymfomen-hpc/siemen/PathForge

Install the standard test environment first:

.. code-block:: bash

   uv sync --extra mil-backends --extra hf --extra dev

If you also need the documentation build dependencies:

.. code-block:: bash

   uv sync --extra mil-backends --extra hf --extra dev --extra docs

Unit Tests
----------

Unit tests cover individual modules, value-shape invariants, H5 I/O helpers,
config validation, and doc-backed API examples.

Run all unit tests:

.. code-block:: bash

   uv run pytest -q tests/unit

Run one unit test module:

.. code-block:: bash

   uv run pytest -q tests/unit/test_config_validation.py

Interface Tests
---------------

Interface tests enforce the architecture contract: base-class conformance,
dependency boundaries, and duplicate-implementation checks.

Run all interface tests:

.. code-block:: bash

   uv run pytest -q tests/interface

Run one interface-focused test module:

.. code-block:: bash

   uv run pytest -q tests/interface/test_dependency_boundaries.py

Integration Tests
-----------------

Integration tests exercise multi-module workflows that are broader than a unit
test but narrower than full smoke coverage.

Run all integration tests:

.. code-block:: bash

   uv run pytest -q tests/integration

Run one integration test module:

.. code-block:: bash

   uv run pytest -q tests/integration/test_sklearn_slide_trainer.py

Smoke Tests
-----------

Smoke tests run end-to-end workflows, including CLI entry points, feature
extraction, benchmarking, optimization, inference, and Hugging Face-backed
sample-data flows.

Run the full smoke suite:

.. code-block:: bash

   export PATHFORGE_SMOKE_CACHE=/path/to/shared/cache/pathforge_smoke
   uv run pytest -q -m smoke tests/smoke

Run the lighter CLI smoke checks:

.. code-block:: bash

   uv run pytest -q \
     tests/smoke/test_feature_extract_cli.py \
     tests/smoke/test_feature_extraction_smoke.py \
     tests/smoke/test_benchmark_cli.py

Run the Hugging Face-backed smoke workflows:

.. code-block:: bash

   export PATHFORGE_SMOKE_CACHE=/path/to/shared/cache/pathforge_smoke
   uv run pytest -q \
     tests/smoke/test_hf_feature_workflows.py \
     tests/smoke/test_hf_mil_benchmarking.py \
     tests/smoke/test_hf_survival_optuna_inference.py

Run Everything
--------------

Run the entire repository suite in one command:

.. code-block:: bash

   uv run pytest -q

If you want the suite split explicitly by layer:

.. code-block:: bash

   uv run pytest -q tests/unit tests/interface tests/integration tests/smoke

Quality Checks
--------------

Before merging documentation or code changes, run the standard repository
checks:

.. code-block:: bash

   uv run ruff check . --fix
   uv run ruff format .
   uv run ruff check .
   uv run pytest -q

Documentation Validation
------------------------

The documentation is covered by unit tests:

- ``tests/unit/test_docs_build.py`` checks toctree targets, autodoc imports,
  and runs a warning-as-error Sphinx build when the ``docs`` extra is
  installed.
- ``tests/unit/test_docs_code_examples.py`` checks that documented code paths
  and API claims still match the live implementation.

You can run only the docs-focused tests with:

.. code-block:: bash

   uv run pytest -q \
     tests/unit/test_docs_build.py \
     tests/unit/test_docs_code_examples.py

What The Tests Output
---------------------

All suites use ``pytest``, so the terminal output follows the same pattern:

- ``collected N items`` tells you how many tests were discovered.
- ``.`` marks a passing test, ``s`` marks a skipped test, and ``F`` marks a
  failing test.
- The final summary reports counts such as ``passed``, ``skipped``, ``failed``,
  and any warnings emitted during the run.
- A failing test prints the assertion, traceback, and the path to the failing
  test file so you can jump directly to the issue.

Some suites also produce workflow artifacts while they run:

- Unit and integration tests mainly validate return values, shapes, generated
  reports, and temporary H5 outputs inside pytest-managed temp directories.
- Smoke tests may create temporary experiment folders, slide artifacts, PDF or
  image reports, cached sample data, and JSON summaries before pytest cleans up
  per-test temporary directories.
- When ``PATHFORGE_SMOKE_CACHE`` is set, downloaded sample data and reused
  smoke intermediates persist in that cache location across runs.
