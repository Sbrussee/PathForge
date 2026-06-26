Tutorial: Slide Retrieval
=========================

Slide retrieval ranks a database of reference slides against one or more query
slides using bag-level features. It reuses the same H5 artifacts produced by
feature extraction, so no additional training is required.

What You Need
-------------

- H5 artifacts from the :doc:`feature_extraction` step.
- At least one dataset marked ``used_for: reference`` and one marked
  ``used_for: query`` (or ``query_reference`` for a shared pool).

Dataset Roles
-------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - ``used_for``
     - Role
   * - ``reference``
     - Slides added to the retrieval database only.
   * - ``query``
     - Slides used as queries only.
   * - ``query_reference``
     - Slides in both the database and the query set (leave-one-out style).

Step 1 — Write the Config
--------------------------

Save as ``retrieval.yaml``:

.. code-block:: yaml

   experiment:
     project_name: tcga_retrieval
     annotation_file: /data/annotations.csv
     project_root: /data/pathforge_projects
     mode: benchmark
     task: slide_retrieval
     aggregation_level: slide
     num_workers: 4

   datasets:
     - name: ReferenceSet
       slides_dir: /data/slides/reference
       artifacts_dir: /data/artifacts/reference
       used_for: reference
     - name: QuerySet
       slides_dir: /data/slides/query
       artifacts_dir: /data/artifacts/query
       used_for: query

   benchmark_parameters:
     tile_px: [256]
     tile_mpp: [0.5]
     feature_extraction: [uni]
     retrieval_representation: [yottixel-features]
     search_strategy: [yottixel]

   slide_retrieval:
     exclusion_level: patient

For a leave-one-out setup where every slide is both a reference and a query:

.. code-block:: yaml

   datasets:
     - name: AllSlides
       slides_dir: /data/slides
       artifacts_dir: /data/artifacts
       used_for: query_reference

   slide_retrieval:
     exclusion_level: patient   # exclude slides from the same patient when querying

Step 2 — Run Retrieval
-----------------------

.. code-block:: bash

   pathforge-benchmark --config retrieval.yaml

For each combination PathForge:

1. Resolves the representation strategy and builds a ``representation_id``.
2. Loads cached per-slide representations from H5 artifacts; computes and
   caches any missing ones.
3. Builds a search database from reference representations.
4. Queries the database with each query representation.
5. Writes ranked results to the run directory.

Pre-computing Representations
------------------------------

For large datasets, compute representations independently before running
retrieval (useful for SLURM array jobs):

.. code-block:: bash

   pathforge-slide-retrieval-representations --config retrieval.yaml

This saturates I/O-bound representation work ahead of the retrieval step so
the benchmark run only performs the fast in-memory search.

Self-Retrieval Exclusion
------------------------

The ``slide_retrieval.exclusion_level`` field controls which slides are
excluded from the result set when a query slide appears in the reference pool:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Value
     - Exclusion applied
   * - ``none``
     - No exclusion — the query slide itself may appear in results.
   * - ``slide``
     - Excludes the exact query slide. Requires
       ``experiment.aggregation_level: slide``.
   * - ``case``
     - Excludes all slides sharing the same case ID.
   * - ``patient``
     - Excludes all slides sharing the same patient ID (default).

Multi-Strategy Grid Search
---------------------------

Search multiple representation and search strategy combinations in a single
run:

.. code-block:: yaml

   benchmark_parameters:
     tile_px: [224, 256]
     tile_mpp: [0.5]
     feature_extraction: [resnet50, uni]
     retrieval_representation: [yottixel-features, hshr-features]
     search_strategy: [yottixel, retccl]

This generates ``2 × 1 × 2 × 2 × 2 = 32`` combinations. Representations are
cached per ``(feature_extraction, tile_px, tile_mpp, retrieval_representation)``
key, so repeated runs only recompute what is missing.

Outputs
-------

PathForge writes one run directory per combination under the experiment root:

.. code-block:: text

   project_root/tcga_retrieval/
   └── slide_retrieval/
       └── {tiling_id}/{feature_name}/{representation}/{search_method}/
           └── run_{hash}/
               ├── manifest.json      — run configuration and summary counts
               └── query_results.csv  — ranked hits per query slide

``query_results.csv`` contains one row per (query, hit) pair with:

- ``query_id``: query slide identifier
- ``hit_id``: retrieved reference slide identifier
- ``rank``: rank in the result list (1 = closest)
- ``score``: similarity score from the search strategy

``manifest.json`` records the full run configuration including
``representation_id``, ``exclusion_level``, ``num_queries``,
``num_reference_items``, and ``top_k_saved``.
