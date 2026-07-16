Scaling and Cluster Execution
=============================

Current Execution Model
-----------------------

PathForge provides CPU data-loading workers and single-device GPU acceleration.
The standard benchmark and optimization commands remain sequential; the
distributed commands split their work into scheduler-neutral records.

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

Create a plan and submit its generated SLURM workflow:

.. code-block:: bash

   pathforge execution plan --config benchmark.yaml --output /project/work/plan
   bash /project/work/plan/slurm/submit.sh

Before launching a large study, run the repository's distributed smoke job:

.. code-block:: bash

   sbatch --partition=PATHgpu --time=01:00:00 \
     scripts/run_distributed_smoke.sbatch

It checks planning, resumable workers, concurrent Optuna storage, and the
optimization CLI. See :doc:`testing` for its outputs and monitoring commands.

The planner writes an immutable config snapshot, JSONL manifests, stable work
IDs, SLURM scripts, and dependency-aware submission commands. Each feature
worker owns one slide and processes all configured bag combinations, preventing
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
in the final report.
