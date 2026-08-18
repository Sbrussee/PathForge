Scaling and Cluster Execution
=============================

Current Execution Model
-----------------------

PathForge provides CPU data-loading workers and single-device GPU acceleration.
The standard benchmark and optimization commands remain sequential; the
distributed commands split their work into scheduler-neutral records.

Terminology
~~~~~~~~~~~

* A **plan** is the immutable description of one distributed invocation.
* A **manifest** is a JSON-lines file containing indexed work records.
* A **work record** is the smallest resumable unit claimed by one worker.
* A **feature shard** is a work record containing one or more slides; that
  worker processes the slides sequentially.
* A **worker** is a local process, Dask task, or SLURM array task executing one
  record.
* **Aggregation/finalization** collects worker statuses and results. It does
  not rerun failed scientific work.

This distributes independent work at artifact boundaries. It does not make one
model forward pass multi-GPU, and the standard benchmark command does not
launch parallel jobs by itself. See :ref:`core-terms` for artifact, bag,
combination, trial, and other shared terms.

Feature-extraction runtime is configured globally:

.. code-block:: yaml

   slide_processing:
     feature_extraction:
       batch_size: 32
       num_workers: 4
       amp: true

Distributed Pipeline
--------------------

Distribution follows durable artifact boundaries:

.. code-block:: text

   feature extraction arrays
             │
             ▼
   benchmark combinations or Optuna workers
             │
             ▼
   result aggregation and visualization

How Work Is Sharded
~~~~~~~~~~~~~~~~~~~

PathForge first creates a scheduler-independent execution plan. The plan
contains a frozen configuration and manifests whose line numbers are work-unit
indices. SLURM and Dask execute the same conceptual units; they differ in who
assigns those units to compute resources.

.. list-table::
   :widths: 18 27 30 25
   :header-rows: 1

   * - Pipeline stage
     - One work unit
     - Work performed inside the unit
     - Parallelism control
   * - Feature extraction
     - One shard containing ``execution.slides_per_shard`` slides, or more when
       necessary to satisfy ``execution.max_shards``.
     - Slides are processed sequentially. For each slide, the worker produces
       every configured tiling/feature bag needed by the configuration.
     - Number of slides per record and maximum concurrent workers/tasks.
   * - Benchmarking
     - One concrete combination returned by the selected task's grid keys.
     - Loads the already-written feature bags, constructs task datasets, and
       executes that single model/loss or representation/search combination in
       an isolated experiment directory.
     - Number of combinations and maximum concurrent workers/tasks.
   * - Optimization
     - One optimization worker, assigned up to
       ``optimization.trials_per_worker`` trials.
     - Claims trials dynamically from the shared Optuna study and evaluates
       them sequentially. Trial parameters are sampled at runtime, so they are
       not prewritten as benchmark-manifest rows.
     - ``trials``, ``trials_per_worker``, shared storage, and maximum concurrent
       SLURM tasks.
   * - Aggregation/finalization
     - One lightweight dependent job.
     - Collects status/result files, or reads the shared Optuna study and writes
       final reports.
     - Runs after the compute array leaves the scheduler queue.

The selected task determines what constitutes a benchmark combination. MIL
prediction tasks vary the task's active tiling, extractor, model, and loss
axes. Slide retrieval instead varies tiling, feature extraction, retrieval
representation, and search strategy. In both cases, the planner asks the task
for its grid keys; unused configuration lists do not create work units.

Feature extraction is sharded by slide rather than by model or loss because
models and losses consume features without changing them. A feature record has
a stable hash-based ID. Every benchmark record conservatively records all
feature-record IDs as dependencies, so benchmark execution begins only after
the complete feature stage succeeds. Within a slide shard, processing is
sequential to prevent two workers from writing the same HDF5 file.

.. code-block:: text

   all annotated slides
      ├── shard 0: slides 0..S-1 ──► one feature worker
      ├── shard 1: slides S..2S-1 ─► one feature worker
      └── shard n: remaining slides ─► one feature worker
                    │ all feature dependencies successful
                    ▼
   task combinations
      ├── combination 0 ─► isolated benchmark worker
      ├── combination 1 ─► isolated benchmark worker
      └── combination m ─► isolated benchmark worker

Here ``S`` is the effective shard size. PathForge starts with
``execution.slides_per_shard``. When ``execution.max_shards`` is set, it raises
``S`` when necessary so that all slides fit in no more than that many shards:

.. code-block:: text

   effective S = max(slides_per_shard, ceil(number_of_slides / max_shards))
   number of feature shards = ceil(number_of_slides / effective S)

A larger ``S`` creates fewer, longer feature jobs; a smaller ``S`` creates
more, shorter jobs and finer retry granularity. Neither setting drops slides or
changes the scientific output.

Choosing the Number of Feature Shards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to bound the feature array:

.. code-block:: yaml

   execution:
     slides_per_shard: 4  # process four slides sequentially in each task
     max_shards: 50       # never create more than 50 feature tasks
     slurm:
       max_concurrent: 10 # run at most 10 array tasks at the same time

``slides_per_shard`` is a lower bound on how many slides are grouped together
(except for the final remainder). ``max_shards`` is an optional upper bound on
the total number of feature work records created. If both are set, PathForge
uses whichever implies the larger shard size.

For example, 103 slides with ``slides_per_shard: 4`` would normally produce 26
shards. Adding ``max_shards: 10`` raises the effective shard size to 11 and
produces 10 shards: nine with 11 slides and one with 4. With
``max_concurrent: 3``, SLURM creates the 10-element array but allows only three
elements to run simultaneously.

These settings answer different questions:

* use ``slides_per_shard`` to control sequential work and retry granularity;
* use ``max_shards`` to cap the total number of scheduler records;
* use ``execution.slurm.max_concurrent`` to cap simultaneous SLURM jobs;
* use ``execution.max_workers`` to cap simultaneous local worker processes.

``max_shards`` applies when the plan is created, so it affects local, Dask, and
SLURM feature manifests equally. It does not cap benchmark combinations or
Optuna workers; those are controlled by their task grid and optimization
settings.

How SLURM Is Used
~~~~~~~~~~~~~~~~~

For a benchmark plan, PathForge generates a feature-extraction job array, a
dependent benchmark job array, and a final aggregation job. The SLURM array
index selects the corresponding JSONL manifest row. Stage-specific resource
blocks set CPUs, GPUs, memory, and wall time, while
``execution.slurm.max_concurrent`` limits the number of simultaneously running
array tasks.

For an optimization plan, the second array contains Optuna workers instead of
precomputed benchmark combinations. The workers join the same study through
``optimization.storage``. The generated dependency chain is:

.. code-block:: text

   feature array
        │ afterok: every feature task must succeed
        ▼
   benchmark array                 Optuna worker array
        │ afterany                      │ afterany
        ▼                               ▼
   result aggregation             optimization finalization

Only one of the two middle paths is used: benchmark mode uses the left path;
optimization mode uses the right. ``afterany`` lets the final reporting job run
even if a compute task fails, so failures remain visible in the final status or
study output.

How Dask Is Used
~~~~~~~~~~~~~~~~

The Dask backend reads the same feature or benchmark manifest as local
execution and maps one work-record index to one Dask future. If a stage's
resource configuration requests a GPU, PathForge requests one Dask resource
named ``GPU`` for each future; Dask workers must advertise that resource.
PathForge gathers all futures before the stage returns.

Dask therefore replaces the process/array scheduler for manifest stages; it
does not change how PathForge shards slides or task combinations. The current
Dask stage runner covers feature and benchmark manifests. Distributed Optuna
worker-array generation and finalization are currently provided by the SLURM
path; running parallel Optuna by another scheduler requires launching multiple
``pathforge optimize worker`` processes against the same supported shared
storage.

SLURM and Dask are alternatives for a given manifest stage. A normal deployment
uses either SLURM arrays or Dask futures to avoid executing the same record
twice. Resume checks make successful work records idempotent, but they are not
a substitute for deliberately choosing one scheduler owner.

Create a plan and submit its generated SLURM workflow:

.. code-block:: bash

   pathforge execution plan --config benchmark.yaml --output /project/work/plan
   bash /project/work/plan/slurm/submit.sh

The first command only creates files; the second submits them. Inspect the
generated manifests, paths, and resource directives before running
``submit.sh``. Submission requires ``sbatch`` and valid cluster-specific
account, partition, and resource settings.

Before launching a large study, run the repository's distributed smoke job:

.. code-block:: bash

   sbatch --partition=PATHgpu --time=01:00:00 \
     scripts/run_distributed_smoke.sbatch

It checks planning, resumable workers, concurrent Optuna storage, and the
optimization CLI. See :doc:`testing` for its outputs and monitoring commands.

The planner writes an immutable config snapshot, JSONL manifests, stable work
IDs, SLURM scripts, and dependency-aware submission commands. Each feature
worker owns one shard and processes its slides sequentially, including all
configured bag combinations. The default shard size is one slide, preventing
concurrent writes to the same H5 file. Benchmark workers use isolated experiment
directories and write atomic status JSON files. Successful records are skipped
when resumed.

Inspect, retry, or aggregate records explicitly:

.. code-block:: bash

   pathforge execution worker --plan /project/work/plan/plan.json \
     --stage benchmark --index 12
   pathforge execution status --plan /project/work/plan/plan.json
   pathforge execution aggregate --plan /project/work/plan/plan.json

Local and Dask Execution
------------------------

The same manifests work without SLURM:

.. code-block:: bash

   pathforge execution run --plan /project/work/plan/plan.json \
     --stage features --backend local
   pathforge execution run --plan /project/work/plan/plan.json \
     --stage benchmark --backend dask \
     --scheduler-address tcp://scheduler.example.org:8786

Use ``local`` to validate sharding and resume behavior on one machine; its
workers still share that machine's CPU, RAM, and GPUs. ``dask`` additionally
requires a compatible Dask cluster. Both use the same manifests and dependency
rules as SLURM.

Install the ``distributed`` extra for Dask, dask-jobqueue, and the PostgreSQL
driver. GPU stages request one Dask ``GPU`` resource, which cluster workers must
advertise.

Parallel Optuna
---------------

Optuna workers claim trials dynamically from one shared study. PostgreSQL is
recommended on clusters; do not use SQLite on a shared HPC filesystem.

.. code-block:: yaml

   experiment:
     mode: optimization

   optimization:
     study_name: cohort_search
     storage: postgresql+psycopg://user:password@db-host/pathforge
     trials: 200
     trials_per_worker: 10
     heartbeat_interval: 60
     stale_trial_timeout: 900

For optimization configs, the planner creates a feature array, an Optuna worker
array, and a dependent finalization job. Each trial writes to an isolated
``trials/trial_<number>`` directory. More than one worker requires
``optimization.storage``.

Execution Configuration
-----------------------

.. code-block:: yaml

   execution:
     backend: slurm
     work_dir: /project/work
     resume: true
     max_workers: 4
     slides_per_shard: 4
     resources:
       feature_extraction: {cpus: 8, gpus: 1, memory_gb: 32, time: "04:00:00"}
       benchmarking: {cpus: 4, gpus: 1, memory_gb: 24, time: "08:00:00"}
       optimization: {cpus: 4, gpus: 1, memory_gb: 24, time: "08:00:00"}
       aggregation: {cpus: 2, gpus: 0, memory_gb: 8, time: "01:00:00"}
     slurm:
       partition: gpu
       account: my-account
       max_concurrent: 20
       extra_directives: []

Generated SLURM arrays use ``afterok`` between feature extraction and training,
then ``afterany`` before aggregation/finalization so failed work remains visible
in the final report. Increasing ``slides_per_shard`` reduces the number of array
tasks; each task runs those slides sequentially and therefore takes longer.
