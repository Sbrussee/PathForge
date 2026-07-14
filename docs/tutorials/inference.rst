Tutorial: Inference and Heatmaps
================================

After benchmarking or optimization you have a packaged model artifact
(``*_package.pt``). The inference CLI loads that self-contained package, reads
the selected feature bag, runs the model, and optionally generates
per-instance attention heatmaps.

What You Need
-------------

- A trained packaged model artifact from benchmarking or optimization.
- H5 feature artifacts (from feature extraction).
- For heatmaps: the ``mil-backends`` extra and per-instance attention scores.

Basic Inference
---------------

Run prediction on a single slide:

.. code-block:: bash

   pathforge-infer-model \
     --model_path /experiments/luad_benchmark/checkpoints/best_package.pt \
     --input /data/artifacts/train/TCGA-A1-A0SB-01Z.h5 \
     --output /data/predictions/TCGA-A1-A0SB-01Z.json

The JSON output contains:

.. code-block:: json

   {
     "status": "ok",
     "task": "classification",
     "model_name": "VarMIL",
     "bag_id": "224px_1.0mpp",
     "feature_extractor": "resnet18",
     "predictions": [0.31, -0.14],
     "probs": [0.12, 0.88]
   }

Batch Inference
---------------

Loop over all slides in a cohort:

.. code-block:: bash

   for H5 in /data/artifacts/test/*.h5; do
     SLIDE=$(basename "$H5" .h5)
     pathforge-infer-model \
       --model_path /experiments/luad_benchmark/checkpoints/best_package.pt \
       --input "$H5" \
       --output "/data/predictions/${SLIDE}.json"
   done

Generating Attention Heatmaps
------------------------------

Heatmaps require per-instance attention scores. These are typically saved
alongside the checkpoint during training or extracted by running a forward
pass with :meth:`~pathforge.core.models.mil_base.MILModelBase.instance_scores`
from the model.

Step 1 — Extract attention scores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read features from an H5 artifact using the
:class:`~pathforge.core.io.h5.base.FileHandleH5` context manager, then run a
forward pass to collect per-instance scores:

.. code-block:: python

   import torch
   import numpy as np
   from pathforge.core.io.h5.base import FileHandleH5
   from pathforge.core.io.h5.features import read_features

   # Load bag features from H5 artifact using the FileHandleH5 context manager
   with FileHandleH5("/data/artifacts/train/TCGA-A1-A0SB-01Z.h5", mode="r") as fh:
       features = read_features(fh, bag_id="256px_0.5mpp", extractor_name="resnet50")

   bag = torch.from_numpy(features).unsqueeze(0)  # [1, N, D]

   # model must be a MILModelBase instance that exposes attention
   # instance_scores returns [B, N] — squeeze for a single bag
   with torch.no_grad():
       scores = model.instance_scores(bag)  # [B, N] = [1, N]
       scores_1d = scores.squeeze(0)         # [N]

   np.save("/data/predictions/TCGA-A1-A0SB-01Z_attention.npy", scores_1d.numpy())

.. note::

   :meth:`~pathforge.core.models.mil_base.MILModelBase.instance_scores` returns
   a tensor shaped ``[B, N]`` (batch × instances). Call ``.squeeze(0)`` to get
   ``[N]`` for a single-bag inference pass.

   Only models that return an ``"attention"`` key from ``forward_bag``
   support this method. It raises ``AttributeError`` otherwise.

Step 2 — Run inference with heatmap generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pathforge-infer-model \
     --model_path /experiments/luad_benchmark/checkpoints/best.ckpt \
     --input /data/artifacts/train/TCGA-A1-A0SB-01Z.h5 \
     --output /data/predictions/TCGA-A1-A0SB-01Z.json \
     --heatmap-backend torchmil \
     --bag-id 256px_0.5mpp \
     --scores /data/predictions/TCGA-A1-A0SB-01Z_attention.npy \
     --heatmap-name abmil_attention \
     --heatmap-output /data/predictions/TCGA-A1-A0SB-01Z_heatmap.json

CLI Arguments
~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Argument
     - Description
   * - ``--model_path``
     - Path to the self-contained ``*_package.pt`` model artifact.
   * - ``--input``
     - Feature artifact path. ``.h5``, ``.pt``, ``.npy``, and ``.npz`` are supported.
   * - ``--output``
     - Output JSON for slide-level predictions.
   * - ``--feature-extractor``
     - Optional H5 extractor key. Defaults to the value stored in the model package.
   * - ``--bag-id``
     - Bag identifier (e.g. ``256px_0.5mpp``). Defaults to the value stored in the model package.
   * - ``--scores``
     - Per-instance scores in ``.npy``, ``.npz``, or ``.json`` format, shaped ``(N,)``.
   * - ``--coords``
     - Optional per-instance coordinates ``.npy`` shaped ``(N, 2)``. Overrides H5 coords.
   * - ``--mask``
     - Optional binary mask ``.npy`` shaped ``(N,)``; false entries are dropped.
   * - ``--heatmap-backend``
     - Use ``torchmil`` to resolve the TorchMIL heatmap explainer.
   * - ``--heatmap-name``
     - H5 namespace for storing the heatmap.
   * - ``--heatmap-output``
     - Optional JSON sidecar for downstream tools that don't read H5.

H5 Heatmap Storage
-------------------

Heatmaps are persisted in the slide H5 artifact under:

.. code-block:: text

   bags/{bag_id}/predictions/heatmaps/{heatmap_name}/
   ├── coords    — float32 (K, 2)  — x/y coordinates in level-0 space
   ├── scores    — float32 (K,)    — normalized to [0, 1]
   └── metadata  — JSON            — backend, model path, score range, coord space

Reading back the heatmap using
:func:`~pathforge.core.io.h5.heatmaps.read_prediction_heatmap`:

.. code-block:: python

   from pathforge.core.io.h5.base import FileHandleH5
   from pathforge.core.io.h5.heatmaps import read_prediction_heatmap

   with FileHandleH5("/data/artifacts/train/TCGA-A1-A0SB-01Z.h5", mode="r") as fh:
       result = read_prediction_heatmap(fh, bag_id="256px_0.5mpp",
                                        heatmap_name="abmil_attention")

   coords = result["coords"]    # float32 (K, 2)
   scores = result["scores"]    # float32 (K,) in [0, 1]
   metadata = result["metadata"]  # dict with backend, coord_space, score_range, etc.

.. note::

   The function is :func:`~pathforge.core.io.h5.heatmaps.read_prediction_heatmap`
   (not ``read_heatmap``). It takes a
   :class:`~pathforge.core.io.h5.base.FileHandleH5` context-manager object,
   not a file path, and returns a ``dict`` with ``"coords"``, ``"scores"``,
   and ``"metadata"`` keys.

Rendering Heatmaps
------------------

Use the stored coordinates and scores to overlay on the WSI thumbnail:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pathforge.core.io.h5.base import FileHandleH5
   from pathforge.core.io.h5.heatmaps import read_prediction_heatmap

   with FileHandleH5(h5_path, mode="r") as fh:
       result = read_prediction_heatmap(fh, bag_id="256px_0.5mpp",
                                        heatmap_name="abmil_attention")
   coords = result["coords"]
   scores = result["scores"]

   plt.figure(figsize=(12, 8))
   plt.scatter(coords[:, 0], coords[:, 1], c=scores, cmap="inferno",
               s=5, alpha=0.7, vmin=0, vmax=1)
   plt.colorbar(label="Attention score")
   plt.gca().invert_yaxis()
   plt.title("ABMIL attention heatmap")
   plt.savefig("heatmap.png", dpi=150)
