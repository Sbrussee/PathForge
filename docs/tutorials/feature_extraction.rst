Tutorial: Feature Extraction
=============================

Feature extraction is the first step in every PathBench workflow. It tiles
whole-slide images (WSIs), segments tissue, extracts per-tile embeddings with a
chosen encoder, and writes everything to row-aligned H5 artifacts.

What You Need
-------------

- A directory of WSI files (``.svs``, ``.ndpi``, ``.tiff``, ``.tif``, ``.mrxs``).
- An annotation CSV with at least ``dataset``, ``slide``, ``patient``, and
  ``category`` columns.
- The ``lazyslide`` extra: ``uv sync --extra lazyslide``.

Step 1 — Prepare the Annotation CSV
-------------------------------------

.. code-block:: text

   dataset,slide,patient,category
   TrainingSet,TCGA-A1-A0SB-01Z,PATIENT-001,case
   TrainingSet,TCGA-A1-A0SD-01Z,PATIENT-002,control
   TestSet,TCGA-A2-A0D0-01Z,PATIENT-003,case

Rules:

- ``dataset`` must match a name in ``datasets[].name``.
- ``slide`` is matched against files using ``{slide}.*`` — no extension needed.
- Add a ``fallback_mpp`` column for slides missing valid MPP metadata.

See :doc:`/data_preparation` for the complete annotation format reference,
slide naming rules, and per-task examples.

Step 2 — Write the Config
--------------------------

Save as ``features.yaml``:

.. code-block:: yaml

   experiment:
     project_name: luad_features
     annotation_file: /data/annotations.csv
     project_root: /data/pathbench_projects
     mode: feature_extraction
     report: true           # generates tile overview PDFs
     num_workers: 8
     mixed_precision: true

   slide_processing:
     backend: lazyslide
     segmentation_method: otsu
     save_tiles: false
     qc_filters: []

   datasets:
     - name: TrainingSet
       slides_dir: /data/slides/train
       artifacts_dir: /data/artifacts/train
       used_for: training
     - name: TestSet
       slides_dir: /data/slides/test
       artifacts_dir: /data/artifacts/test
       used_for: testing

   benchmark_parameters:
     tile_px: [256]
     tile_mpp: [0.5]
     feature_extraction: [resnet50]
     mil: []                # empty — no MIL training in extraction mode

   weights_dir: /data/pretrained_weights

Step 3 — Run Feature Extraction
---------------------------------

.. code-block:: bash

   pathbench-features --config features.yaml --log-level INFO

PathBench will:

1. Load the annotation CSV and resolve slides for each dataset.
2. Create the experiment directory under ``project_root/luad_features/``.
3. For each (feature_extractor × tile_px × tile_mpp) combination:

   a. Validate base MPP for each slide.
   b. Reuse existing valid H5 coordinates when possible.
   c. Segment tissue using Otsu thresholding.
   d. Extract tile coordinates.
   e. Write ``coords`` and ``tiling_spec`` to H5.
   f. Write ``tiles_overview`` (compressed image) when ``report: true``.
   g. Extract tile embeddings with ResNet-50.
   h. Write feature matrix (row-aligned with ``coords``) to H5.

The resulting H5 layout for one slide:

.. code-block:: text

   TCGA-A1-A0SB-01Z.h5
   └── bags/
       └── 256px_0.5mpp/
           ├── coords              — int32 (N, 5)
           ├── tiling_spec         — JSON
           ├── features/
           │   └── resnet50        — float32 (N, 2048)
           └── tiles_overview      — uint8 (M,)

Step 4 — Generate Tile Reports (Optional)
------------------------------------------

If ``report: true`` was set, generate PDF tile overview reports:

.. code-block:: bash

   python -m pathbench.cli.tiles_report --config features.yaml --log-level INFO

PDFs are written to the experiment directory.

Running on a SLURM Cluster
---------------------------

For large cohorts, run one slide per array task:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --array=0-99
   #SBATCH --job-name=pathbench_features

   SLIDES=(/data/slides/train/*.svs)
   SLIDE=${SLIDES[$SLURM_ARRAY_TASK_ID]}

   python -m pathbench.cli.feature_extraction_slide \
     --config features.yaml \
     --dataset TrainingSet \
     --input "$SLIDE" \
     --log-level INFO

PathBench automatically appends ``_${SLURM_JOB_ID}`` to the project name to
avoid directory collisions between tasks.

Using Multiple Feature Extractors
----------------------------------

Add multiple names to ``benchmark_parameters.feature_extraction``:

.. code-block:: yaml

   benchmark_parameters:
     tile_px: [224, 256]
     tile_mpp: [0.5]
     feature_extraction: [resnet50, uni]

This generates four combinations (2 sizes × 1 MPP × 2 extractors) and writes
a separate feature matrix for each combination into the same H5 file:

.. code-block:: text

   256px_0.5mpp/features/resnet50   — float32 (N, 2048)
   256px_0.5mpp/features/uni        — float32 (N, 1024)
   224px_0.5mpp/features/resnet50   — float32 (N, 2048)
   224px_0.5mpp/features/uni        — float32 (N, 1024)

Foundation models (``uni``, ``conch``, ``gigapath``, ``phikon``) require a
Hugging Face token:

.. code-block:: yaml

   hf_key: hf_your_token_here

Skipping Existing Artifacts
-----------------------------

By default PathBench skips slides whose H5 features already exist
(``mil.skip_extracted: true``). To force re-extraction, delete the H5 file
or set:

.. code-block:: yaml

   mil:
     skip_extracted: false
     skip_feature_extraction: false
