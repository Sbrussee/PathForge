Tutorials
=========

These tutorials walk through the complete PathForge workflow from raw slides to
trained models and inference heatmaps. Start with the end-to-end example to see
how the stages fit together, then use the focused tutorials for each subtask.

.. toctree::
   :maxdepth: 2

   end_to_end
   feature_extraction
   benchmarking
   optimization
   inference
   model_packaging
   slide_retrieval
   cli

Workflow Overview
-----------------

.. code-block:: text

   Slides + Annotation CSV
          │
          ▼
   1. Feature Extraction   ──►  H5 artifacts  (coords + features)
          │
          ▼
   2. Benchmarking         ──►  Result CSVs + checkpoints
    ─ or ─
   2. Slide Retrieval      ──►  Ranked retrieval CSVs + manifests
    ─ or ─
   2. Optimization         ──►  Optuna study + best checkpoint
          │
          ▼
   3. Inference            ──►  prediction JSON + heatmap H5
          │
          ▼
   4. Model Packaging      ──►  portable checkpoint bundle

Each step is config-driven. The same YAML file can cover feature extraction
and benchmarking together, or you can use separate files per stage.
