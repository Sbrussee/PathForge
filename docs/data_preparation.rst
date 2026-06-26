Data Preparation
================

PathBench requires two inputs before any workflow can run:

1. An **annotation CSV** that describes every slide, its patient grouping, its
   dataset split, and its task-specific label.
2. A **slides directory** whose files follow a strict naming convention.

This page documents both requirements in full.

----

Annotation CSV
--------------

The annotation CSV is the single source of truth for all PathBench workflows.
Every CLI command reads it at startup via ``experiment.annotation_file``.

Minimum Required Columns
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - ``dataset``
     - Dataset split name. Must match a ``name`` entry in ``datasets[].name``
       in the config. Used to route each slide to the correct
       ``slides_dir`` / ``artifacts_dir`` pair.
   * - ``slide``
     - Slide identifier. Must equal the stem of the WSI file on disk
       (filename without extension). See `Slide Naming`_ below.
   * - ``category``
     - Target label column. For ``task: classification``: 0-based integer class
       index. For ``task: regression``: any finite float. For survival tasks
       the column is read but ignored during training (survival labels come
       from the time/event columns instead). For slide retrieval it is used
       only for evaluation metrics. The column name defaults to ``category``
       and can be overridden with ``experiment.label_column``.

.. note::

   Column names ``slide`` and ``category`` are defaults. Override them with
   ``experiment.slide_column`` and ``experiment.label_column`` in the config.

Optional Columns
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - ``patient``
     - Patient identifier used for patient-level metric aggregation and
       cross-validation splitting. When absent, PathBench falls back to the
       slide ID (i.e. one patient per slide). Strongly recommended for any
       cohort where one patient has multiple slides.
   * - ``wsi_path``
     - Absolute path to the WSI file. When present, PathBench uses this path
       directly instead of resolving the slide through ``slides_dir``. Useful
       when slides are stored in heterogeneous subdirectory trees or remote
       mounts.
   * - ``fallback_mpp``
     - Fallback microns-per-pixel (MPP) value as a float. Used for slides
       whose metadata does not contain valid MPP information (e.g. scanned
       without calibration). Provide this value so PathBench can still tile
       at the correct resolution.

Survival Task Columns
~~~~~~~~~~~~~~~~~~~~~

Survival tasks (``task: survival`` and ``task: survival_discrete``) require
two additional numeric columns. Their names are configured in the experiment
block.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Config key
     - Column description
   * - ``experiment.survival_time_column``
     - Observation time in any consistent unit (e.g. months). Must be > 0.
       Rows with missing or non-positive times are dropped automatically.
   * - ``experiment.survival_event_column``
     - Binary event indicator. ``1`` = event occurred (e.g. death),
       ``0`` = censored observation.

Example config:

.. code-block:: yaml

   experiment:
     survival_time_column: os_months
     survival_event_column: vital_status

Example annotation row:

.. code-block:: text

   dataset,slide,patient,category,os_months,vital_status
   TrainingSet,TCGA-A1-A0SB-01Z,PATIENT-001,case,24.5,1
   TrainingSet,TCGA-A1-A0SD-01Z,PATIENT-002,control,60.0,0

Per-Task Annotation Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Classification**

.. code-block:: text

   dataset,slide,patient,category
   TrainingSet,TCGA-A1-A0SB-01Z,PATIENT-001,1
   TrainingSet,TCGA-A1-A0SD-01Z,PATIENT-001,1
   ValSet,TCGA-A2-A0CU-01Z,PATIENT-002,0
   TestSet,TCGA-A2-A0D0-01Z,PATIENT-003,0

Rules:

- ``category`` must be a 0-based integer class index.
- Class 0 is the reference/negative class; highest index is the positive class
  for binary tasks.
- Multiple slides per patient are fully supported; use the same ``patient``
  value to link them for patient-level aggregation.

Config (``task: classification``):

.. code-block:: yaml

   experiment:
     task: classification
     label_column: category     # default
   metrics:
     classification_backend: torchmetrics
     classification_metrics: [accuracy, balanced_accuracy, auroc, f1]

**Regression**

.. code-block:: text

   dataset,slide,patient,category
   TrainingSet,SLIDE_001,PAT-001,0.83
   TrainingSet,SLIDE_002,PAT-002,1.47
   ValSet,SLIDE_003,PAT-003,0.21
   TestSet,SLIDE_004,PAT-004,2.05

Rules:

- ``category`` must be a numeric float (any finite value).
- No class mapping is applied; PathBench uses the raw value as the regression
  target.
- The column name can be changed via ``experiment.label_column``; the column
  referenced must contain floats.

Config (``task: regression``):

.. code-block:: yaml

   experiment:
     task: regression
     label_column: category     # or any float column name
   metrics:
     regression_metrics: [mae, mse]

**Survival**

.. code-block:: text

   dataset,slide,patient,category,os_months,vital_status
   TrainingSet,TCGA-A1-A0SB-01Z,PATIENT-001,case,18.3,1
   TrainingSet,TCGA-A1-A0SD-01Z,PATIENT-002,case,45.0,0
   TestSet,TCGA-A2-A0D0-01Z,PATIENT-003,control,30.1,1

Rules:

- ``category`` is ignored during survival training but must still be present
  unless ``experiment.label_column`` points to a different column.
- ``os_months`` and ``vital_status`` are example names; configure the actual
  column names via ``experiment.survival_time_column`` and
  ``experiment.survival_event_column``.
- Rows where ``vital_status`` is missing or ``os_months`` ≤ 0 are dropped.

**Slide Retrieval**

.. code-block:: text

   dataset,slide,patient,category
   ReferenceSet,TCGA-A1-A0SB-01Z,PATIENT-001,case
   ReferenceSet,TCGA-A1-A0SD-01Z,PATIENT-002,control
   QuerySet,TCGA-A2-A0D0-01Z,PATIENT-003,case

Rules:

- Dataset ``used_for`` values for retrieval are ``reference``, ``query``,
  or ``query_reference`` (pool used for both roles simultaneously).
- ``category`` labels are used for retrieval evaluation metrics (precision
  @ k) but are not required for the search itself.
- Self-retrieval exclusion is controlled by ``slide_retrieval.exclusion_level``
  (``none`` | ``slide`` | ``case`` | ``patient``).

----

Slide Naming
------------

Slide IDs in the annotation CSV must match files in ``slides_dir`` exactly.
PathBench supports two layouts.

Layout 1 — Single-file slide (most common)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The file stem must equal ``slide_id`` exactly:

.. code-block:: text

   slides_dir/
   ├── TCGA-A1-A0SB-01Z.svs
   ├── TCGA-A1-A0SD-01Z.ndpi
   └── TCGA-A2-A0D0-01Z.tiff

Supported extensions: ``.svs``, ``.ndpi``, ``.tiff``, ``.tif``, ``.mrxs``.

Do **not** add suffixes, version numbers, or dates to the slide ID:

.. code-block:: text

   # INVALID — stem does not match slide_id
   TCGA-A1-A0SB-01Z_scan1.svs
   TCGA-A1-A0SB-01Z.v2.svs

Layout 2 — Multi-file DICOM slide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Store all DICOM files for one slide in a folder named exactly like
``slide_id``:

.. code-block:: text

   slides_dir/
   └── T12-00126/
       ├── 000001.dcm
       ├── 000002.dcm
       └── 000003.dcm

Rules:

- Folder name must equal ``slide_id`` exactly.
- ``.dcm`` files must be directly inside the folder (no nested subdirectories).
- When both a direct file and a DICOM folder exist for the same ID, the
  direct file takes precedence.

Layout 3 — Explicit path override
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a ``wsi_path`` column to the annotation CSV with absolute paths:

.. code-block:: text

   dataset,slide,patient,category,wsi_path
   TrainingSet,slide-001,PAT-001,1,/mnt/nas/project_a/raw/slide-001.svs

PathBench uses ``wsi_path`` directly and skips ``slides_dir`` resolution for
that row. Useful for heterogeneous storage.

----

Directory Structure
-------------------

A typical project layout for two dataset splits:

.. code-block:: text

   /data/
   ├── annotations.csv
   ├── slides/
   │   ├── train/
   │   │   ├── TCGA-A1-A0SB-01Z.svs
   │   │   └── TCGA-A1-A0SD-01Z.svs
   │   └── test/
   │       └── TCGA-A2-A0D0-01Z.svs
   └── artifacts/
       ├── train/
       │   ├── TCGA-A1-A0SB-01Z.h5    ← written by pathbench-features
       │   └── TCGA-A1-A0SD-01Z.h5
       └── test/
           └── TCGA-A2-A0D0-01Z.h5

Corresponding config:

.. code-block:: yaml

   experiment:
     annotation_file: /data/annotations.csv

   datasets:
     - name: TrainingSet
       slides_dir: /data/slides/train
       artifacts_dir: /data/artifacts/train
       used_for: training
     - name: TestSet
       slides_dir: /data/slides/test
       artifacts_dir: /data/artifacts/test
       used_for: testing

.. note::

   ``artifacts_dir`` is created automatically if it does not exist.
   ``slides_dir`` must already exist and contain all referenced WSI files.

----

Column-Name Overrides
---------------------

All default column names can be changed in the config:

.. code-block:: yaml

   experiment:
     slide_column: slide_id          # default: "slide"
     label_column: label             # default: "category"
     survival_time_column: months    # default: null (required for survival)
     survival_event_column: event    # default: null (required for survival)
     aggregation_level: patient      # slide | case | patient

The ``dataset`` and ``patient`` column names are fixed and cannot be
overridden via config.

----

Validation Checklist
---------------------

Before running PathBench, verify:

.. code-block:: text

   ✓ annotation CSV has "dataset" and "slide" columns
   ✓ every "dataset" value matches a datasets[].name entry in the config
   ✓ every "slide" value has a matching file in the corresponding slides_dir
   ✓ file stem == slide ID (no extra suffixes or extensions in the ID)
   ✓ "patient" column present when multiple slides share a patient
   ✓ "category" column present (or label_column override set)
   ✓ survival_time_column / survival_event_column set for survival tasks
   ✓ no duplicate slide IDs within a dataset split
   ✓ no NaN values in required columns

A quick sanity check using pandas:

.. code-block:: python

   import pandas as pd
   from pathlib import Path

   df = pd.read_csv("/data/annotations.csv")
   slides_dir = Path("/data/slides/train")

   missing = [
       row.slide for _, row in df[df.dataset == "TrainingSet"].iterrows()
       if not any(slides_dir.glob(f"{row.slide}.*"))
   ]
   print("Missing slides:", missing)
